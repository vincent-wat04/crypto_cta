"""
FVG 特征工程：为每个 FVG 事件构建高频特征向量。

特征类别：
1. FVG 自身属性（gap 大小、方向、K 线形态）
2. 前置微观结构（FVG 发生前的市场状态）
3. 后置微观结构（FVG 发生后的早期市场反应）
4. 跨时间框架特征（多频率一致性）
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

from .detector import FVG, FVGType, FVGStatus


def _compute_trade_features_window(
    trades: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    prefix: str = "",
) -> Dict[str, float]:
    """计算指定时间窗口内的逐笔成交特征。"""
    window = trades[(trades["timestamp"] >= start) & (trades["timestamp"] < end)]
    
    if window.empty:
        return {f"{prefix}n_trades": 0}
    
    n = len(window)
    total_vol = window["amount"].sum()
    buy_vol = window.loc[window["side"] == "buy", "amount"].sum()
    sell_vol = window.loc[window["side"] == "sell", "amount"].sum()
    
    prices = window["price"]
    price_range = prices.max() - prices.min()
    price_std = prices.std() if n > 1 else 0
    price_ret = (prices.iloc[-1] - prices.iloc[0]) / (prices.iloc[0] + 1e-10) * 100
    
    # 订单流
    imbalance = (buy_vol - sell_vol) / (total_vol + 1e-10)
    
    # 成交加速度
    mid_idx = n // 2
    vol_first_half = window["amount"].iloc[:mid_idx].sum()
    vol_second_half = window["amount"].iloc[mid_idx:].sum()
    vol_acceleration = (vol_second_half - vol_first_half) / (vol_first_half + 1e-10)
    
    # Price impact
    impact = abs(price_ret) / (total_vol + 1e-10) if total_vol > 0 else 0
    
    # 大单比例（>2x 平均）
    avg_trade = window["amount"].mean()
    large_trade_pct = (window["amount"] > 2 * avg_trade).mean()
    
    # 价格连续性（同方向连续 tick 数）
    price_diff = prices.diff().dropna()
    if len(price_diff) > 0:
        signs = np.sign(price_diff)
        max_run = _max_consecutive_run(signs)
    else:
        max_run = 0
    
    return {
        f"{prefix}n_trades": n,
        f"{prefix}total_volume": total_vol,
        f"{prefix}buy_volume": buy_vol,
        f"{prefix}sell_volume": sell_vol,
        f"{prefix}imbalance": imbalance,
        f"{prefix}price_return": price_ret,
        f"{prefix}price_range": price_range,
        f"{prefix}price_std": price_std,
        f"{prefix}volume_acceleration": vol_acceleration,
        f"{prefix}impact": impact,
        f"{prefix}large_trade_pct": large_trade_pct,
        f"{prefix}max_consecutive_run": max_run,
    }


def _max_consecutive_run(signs: pd.Series) -> int:
    """计算同方向最长连续 tick 数。"""
    if len(signs) == 0:
        return 0
    max_run = 1
    current = 1
    for i in range(1, len(signs)):
        if signs.iloc[i] == signs.iloc[i-1] and signs.iloc[i] != 0:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 1
    return max_run


def _compute_spread_features(
    trades: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    prefix: str = "",
) -> Dict[str, float]:
    """从逐笔成交估计 spread 特征。"""
    window = trades[(trades["timestamp"] >= start) & (trades["timestamp"] < end)]
    
    if len(window) < 5:
        return {f"{prefix}spread_mean": 0, f"{prefix}spread_std": 0, f"{prefix}spread_max": 0}
    
    # 使用相邻买卖成交价差估计 spread
    diffs = window["price"].diff().abs()
    
    return {
        f"{prefix}spread_mean": diffs.mean(),
        f"{prefix}spread_std": diffs.std(),
        f"{prefix}spread_max": diffs.max(),
    }


def build_fvg_feature_matrix(
    fvgs: List[FVG],
    trades: pd.DataFrame,
    ohlcv: pd.DataFrame,
    pre_window_ms: int = 60000,
    post_window_ms: int = 30000,
) -> pd.DataFrame:
    """
    为每个 FVG 构建完整的特征矩阵。
    
    Args:
        fvgs: FVG 事件列表
        trades: 逐笔成交数据
        ohlcv: OHLCV 数据
        pre_window_ms: FVG 前的观察窗口（毫秒）
        post_window_ms: FVG 后的观察窗口（毫秒）
    
    Returns:
        DataFrame，每行一个 FVG，列为特征 + 标签
    """
    if not fvgs:
        return pd.DataFrame()
    
    records = []
    
    for fvg in fvgs:
        ts = fvg.timestamp
        pre_start = ts - pd.Timedelta(milliseconds=pre_window_ms)
        post_end = ts + pd.Timedelta(milliseconds=post_window_ms)
        
        feat: Dict[str, Any] = {}
        
        # ===== 1. FVG 自身属性 =====
        feat["fvg_type"] = 1 if fvg.fvg_type == FVGType.BULLISH else -1
        feat["gap_size_pct"] = fvg.gap_size_pct
        feat["gap_size_abs"] = fvg.gap_size
        feat["mid_price"] = fvg.mid_price
        feat["candle2_volume"] = fvg.candle2_volume
        feat["candle2_range"] = fvg.candle2_range
        
        # 成交量相对均值
        fvg_idx = ohlcv.index.get_loc(ts) if ts in ohlcv.index else None
        if fvg_idx is not None and fvg_idx >= 20:
            avg_vol = ohlcv["volume"].iloc[fvg_idx-20:fvg_idx].mean()
            feat["volume_ratio"] = fvg.candle2_volume / (avg_vol + 1e-10)
        else:
            feat["volume_ratio"] = 1.0
        
        # ===== 2. 前置微观结构（FVG 发生前） =====
        pre_feats = _compute_trade_features_window(trades, pre_start, ts, prefix="pre_")
        feat.update(pre_feats)
        
        pre_spread = _compute_spread_features(trades, pre_start, ts, prefix="pre_")
        feat.update(pre_spread)
        
        # 前置趋势（更长窗口）
        long_pre_start = ts - pd.Timedelta(milliseconds=pre_window_ms * 3)
        long_feats = _compute_trade_features_window(trades, long_pre_start, ts, prefix="longpre_")
        feat.update(long_feats)
        
        # ===== 3. 后置微观结构（FVG 发生后） =====
        post_feats = _compute_trade_features_window(trades, ts, post_end, prefix="post_")
        feat.update(post_feats)
        
        post_spread = _compute_spread_features(trades, ts, post_end, prefix="post_")
        feat.update(post_spread)
        
        # ===== 4. 跨指标特征 =====
        # 前后订单流对比
        feat["imbalance_shift"] = feat.get("post_imbalance", 0) - feat.get("pre_imbalance", 0)
        feat["volume_shift"] = (
            feat.get("post_total_volume", 0) / (feat.get("pre_total_volume", 1e-10))
        )
        feat["spread_shift"] = (
            feat.get("post_spread_mean", 0) - feat.get("pre_spread_mean", 0)
        )
        
        # ===== 5. 标签 =====
        feat["label_filled"] = 1 if fvg.status == FVGStatus.FILLED else 0
        feat["label_fill_bars"] = fvg.fill_bars
        feat["label_max_extension"] = fvg.max_extension
        
        # 分类标签：是否在 N 根 K 线内填充
        feat["label_fill_fast"] = 1 if fvg.status == FVGStatus.FILLED and fvg.fill_bars <= 10 else 0
        feat["label_fill_slow"] = 1 if fvg.status == FVGStatus.FILLED and fvg.fill_bars > 10 else 0
        feat["label_not_fill"] = 1 if fvg.status in (FVGStatus.OPEN, FVGStatus.EXPIRED) else 0
        
        # ===== 6. 价格方向标签（核心标签）=====
        # 检查 FVG 之后 N 根 K 线的实际价格变动
        if fvg_idx is not None:
            for horizon in [5, 10, 20]:
                exit_idx = min(fvg_idx + horizon, len(ohlcv) - 1)
                entry_price = ohlcv.iloc[fvg_idx]["close"]
                exit_price = ohlcv.iloc[exit_idx]["close"]
                future_ret = (exit_price - entry_price) / entry_price * 100
                feat[f"future_ret_{horizon}"] = future_ret
                # 方向标签：1=涨, 0=跌
                feat[f"label_direction_{horizon}"] = 1 if future_ret > 0 else 0
            
            # 最大有利/不利偏移（用于止损研究）
            lookahead = ohlcv.iloc[fvg_idx:min(fvg_idx + 20, len(ohlcv))]
            if len(lookahead) > 1:
                future_highs = lookahead["high"]
                future_lows = lookahead["low"]
                feat["max_favorable_up"] = (future_highs.max() - entry_price) / entry_price * 100
                feat["max_adverse_down"] = (entry_price - future_lows.min()) / entry_price * 100
            else:
                feat["max_favorable_up"] = 0
                feat["max_adverse_down"] = 0
        else:
            for horizon in [5, 10, 20]:
                feat[f"future_ret_{horizon}"] = 0
                feat[f"label_direction_{horizon}"] = 0
            feat["max_favorable_up"] = 0
            feat["max_adverse_down"] = 0
        
        feat["timestamp"] = ts
        feat["fvg_status"] = fvg.status.value
        
        records.append(feat)
    
    df = pd.DataFrame(records)
    return df


def get_feature_columns() -> List[str]:
    """返回用于模型训练的特征列名。"""
    return [
        # FVG 属性
        "fvg_type", "gap_size_pct", "candle2_volume", "candle2_range", "volume_ratio",
        # 前置 trade 特征
        "pre_n_trades", "pre_total_volume", "pre_imbalance",
        "pre_price_return", "pre_price_range", "pre_price_std",
        "pre_volume_acceleration", "pre_impact", "pre_large_trade_pct",
        "pre_max_consecutive_run",
        # 前置 spread 特征
        "pre_spread_mean", "pre_spread_std", "pre_spread_max",
        # 长窗口前置特征
        "longpre_imbalance", "longpre_price_return", "longpre_volume_acceleration",
        # 后置 trade 特征
        "post_n_trades", "post_total_volume", "post_imbalance",
        "post_price_return", "post_volume_acceleration", "post_impact",
        # 后置 spread 特征
        "post_spread_mean",
        # 跨指标对比
        "imbalance_shift", "volume_shift", "spread_shift",
    ]
