"""
FVG V3 特征工程：基于 indicators/ 库的全面特征构建。

改进：
1. 直接调用 indicators/ 中的基础指标（不再自己重复计算）
2. 每个特征可配置独立的 time window
3. 包含 taker order 级别的指标
4. Triple Barrier Method 作为标签
5. 支持 orderbook lifecycle 指标（如有 orderbook 数据）
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

from .detector import FVG, FVGType, FVGStatus


# ─────────────────────────────────────────────────────────
# 默认特征窗口配置（可 grid search）
# ─────────────────────────────────────────────────────────

DEFAULT_FEATURE_WINDOWS = {
    # 指标名: (pre_window_ms, post_window_ms)
    "price_impact":     (10000, 5000),
    "spread":           (10000, 5000),
    "imbalance":        (30000, 10000),
    "taker_flow":       (60000, 30000),   # taker 指标需要更长窗口
    "volume":           (30000, 10000),
    "regime":           (300000, None),    # regime 只看 pre，5min 回溯
}


def _tz_normalize(ts: pd.Timestamp, ref_index: pd.Index) -> pd.Timestamp:
    """确保 ts 和 ref_index 的时区一致。"""
    idx_tz = getattr(ref_index, "tz", None)
    ts_tz = getattr(ts, "tz", None) or getattr(ts, "tzinfo", None)
    if idx_tz is None and ts_tz is not None:
        # indicator 是 tz-naive，event 是 tz-aware → 去掉 tz
        return ts.tz_localize(None)
    if idx_tz is not None and ts_tz is None:
        # indicator 是 tz-aware，event 是 tz-naive → 加 tz
        return ts.tz_localize(idx_tz)
    return ts


def _resample_indicator_at_event(
    indicator: pd.Series,
    event_time: pd.Timestamp,
    pre_ms: int,
    post_ms: Optional[int],
    prefix: str,
) -> Dict[str, float]:
    """在事件时间点前后提取指标统计量。"""
    feats = {}
    event_time = _tz_normalize(event_time, indicator.index)

    # Pre-event window
    pre_start = event_time - pd.Timedelta(milliseconds=pre_ms)
    pre_data = indicator.loc[
        (indicator.index >= pre_start) & (indicator.index < event_time)
    ]
    if len(pre_data) > 0:
        feats[f"{prefix}_pre_mean"] = pre_data.mean()
        feats[f"{prefix}_pre_std"] = pre_data.std() if len(pre_data) > 1 else 0
        feats[f"{prefix}_pre_last"] = pre_data.iloc[-1]
        feats[f"{prefix}_pre_max"] = pre_data.max()
        feats[f"{prefix}_pre_min"] = pre_data.min()
        feats[f"{prefix}_pre_trend"] = pre_data.iloc[-1] - pre_data.iloc[0] if len(pre_data) > 1 else 0
    else:
        for s in ["mean", "std", "last", "max", "min", "trend"]:
            feats[f"{prefix}_pre_{s}"] = 0

    # Post-event window
    if post_ms is not None:
        post_end = event_time + pd.Timedelta(milliseconds=post_ms)
        post_data = indicator.loc[
            (indicator.index > event_time) & (indicator.index <= post_end)
        ]
        if len(post_data) > 0:
            feats[f"{prefix}_post_mean"] = post_data.mean()
            feats[f"{prefix}_post_first"] = post_data.iloc[0]
        else:
            feats[f"{prefix}_post_mean"] = 0
            feats[f"{prefix}_post_first"] = 0

        # Shift: post vs pre
        feats[f"{prefix}_shift"] = feats[f"{prefix}_post_mean"] - feats[f"{prefix}_pre_mean"]

    return feats


def build_fvg_features_v3(
    fvgs: List[FVG],
    trades: pd.DataFrame,
    ohlcv: pd.DataFrame,
    taker_orders: Optional[pd.DataFrame] = None,
    book: Optional[pd.DataFrame] = None,
    feature_windows: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    为每个 FVG 构建 V3 特征矩阵（基于 indicators/ 库）。

    Args:
        fvgs: FVG 事件列表
        trades: aggTrades DataFrame
        ohlcv: OHLCV 数据
        taker_orders: 预计算的 taker orders（由 group_taker_orders 生成）
        book: orderbook DataFrame（可选）
        feature_windows: 各指标的时间窗口配置

    Returns:
        DataFrame, 每行一个 FVG, 列 = 特征 + 标签
    """
    if not fvgs:
        return pd.DataFrame()

    windows = feature_windows or DEFAULT_FEATURE_WINDOWS

    # ── Pre-compute all indicators as time series ──
    print("[FVG V3] Pre-computing indicators...")
    indicators = _precompute_indicators(trades, taker_orders, book)

    # ── Build per-FVG features ──
    print(f"[FVG V3] Building features for {len(fvgs)} FVGs...")
    records = []

    for i, fvg in enumerate(fvgs):
        ts = fvg.timestamp
        feat: Dict[str, Any] = {}

        # 1. FVG 自身属性
        feat["fvg_type"] = 1 if fvg.fvg_type == FVGType.BULLISH else -1
        feat["gap_size_pct"] = fvg.gap_size_pct
        feat["gap_size_abs"] = fvg.gap_size
        feat["candle2_volume"] = fvg.candle2_volume
        feat["candle2_range"] = fvg.candle2_range

        fvg_idx = ohlcv.index.get_loc(ts) if ts in ohlcv.index else None
        if fvg_idx is not None and fvg_idx >= 20:
            avg_vol = ohlcv["volume"].iloc[fvg_idx - 20:fvg_idx].mean()
            feat["volume_ratio"] = fvg.candle2_volume / (avg_vol + 1e-10)
        else:
            feat["volume_ratio"] = 1.0

        # 2. 从 indicators 时序提取事件周围特征
        for ind_name, ind_series in indicators.items():
            category = _get_indicator_category(ind_name)
            w = windows.get(category, (30000, 10000))
            pre_ms, post_ms = w

            ind_feats = _resample_indicator_at_event(
                ind_series, ts, pre_ms, post_ms, prefix=ind_name,
            )
            feat.update(ind_feats)

        # 3. Taker order 特征（如果有 taker_orders）
        if taker_orders is not None and not taker_orders.empty:
            taker_w = windows.get("taker_flow", (60000, 30000))
            taker_feats = _extract_taker_features_at_event(
                taker_orders, ts, taker_w[0], taker_w[1],
            )
            feat.update(taker_feats)

        # 4. 标签（保留旧标签 + 新增 Triple Barrier 占位）
        feat["label_filled"] = 1 if fvg.status == FVGStatus.FILLED else 0
        feat["label_fill_bars"] = fvg.fill_bars
        feat["fvg_status"] = fvg.status.value

        if fvg_idx is not None:
            entry_price = ohlcv.iloc[fvg_idx]["close"]
            for horizon in [5, 10, 20]:
                exit_idx = min(fvg_idx + horizon, len(ohlcv) - 1)
                exit_price = ohlcv.iloc[exit_idx]["close"]
                feat[f"future_ret_{horizon}"] = (exit_price - entry_price) / entry_price * 100
                feat[f"label_direction_{horizon}"] = 1 if feat[f"future_ret_{horizon}"] > 0 else 0

        feat["timestamp"] = ts
        records.append(feat)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(fvgs)}]")

    return pd.DataFrame(records)


def _precompute_indicators(
    trades: pd.DataFrame,
    taker_orders: Optional[pd.DataFrame],
    book: Optional[pd.DataFrame],
) -> Dict[str, pd.Series]:
    """一次性计算所有基础指标时序。"""
    from indicators.base.price_impact import tick_price_impact, volume_weighted_impact
    from indicators.base.spread import trade_diff_spread, roll_spread
    from indicators.base.order_flow import trade_imbalance, flow_toxicity
    from indicators.base.volume_profile import (
        taker_volume_corr, taker_volume_skewness, buy_sell_volume_ratio,
    )

    inds = {}

    # Tick-level
    inds["tick_impact"] = tick_price_impact(trades, window_ms=5000)
    inds["vw_impact"] = volume_weighted_impact(trades, window_ms=10000)
    inds["spread"] = trade_diff_spread(trades, window_trades=100)
    inds["roll_spread"] = roll_spread(trades, window_trades=200)
    inds["imbalance"] = trade_imbalance(trades, window_trades=100)
    inds["toxicity"] = flow_toxicity(trades, window_trades=200)

    # Taker-level (from aggTrades)
    if "agg_trade_id" in trades.columns:
        inds["taker_corr"] = taker_volume_corr(trades, resample_freq="5s", window=60)
        inds["taker_skew"] = taker_volume_skewness(trades, resample_freq="5s", window=120)
        inds["bs_ratio"] = buy_sell_volume_ratio(trades, resample_freq="5s", window=60)

    # Taker order level
    if taker_orders is not None and not taker_orders.empty:
        from indicators.base.taker_flow import (
            avg_levels_swept, impact_efficiency, taker_arrival_rate,
            sweep_depth_autocorr, taker_imbalance as taker_imb_fn,
        )
        inds["levels_swept"] = avg_levels_swept(taker_orders, resample_freq="5s", window=60)
        inds["impact_eff"] = impact_efficiency(taker_orders, resample_freq="10s", window=30)
        inds["taker_arrival"] = taker_arrival_rate(taker_orders, resample_freq="5s", window=60)
        inds["sweep_autocorr"] = sweep_depth_autocorr(taker_orders, resample_freq="5s", window=60)
        inds["taker_imb"] = taker_imb_fn(taker_orders, resample_freq="5s", window=60)

    # Orderbook (if available)
    if book is not None and not book.empty:
        from indicators.base.orderbook_pressure import depth_imbalance, vwap_pressure
        from indicators.base.orderbook_lifecycle import depth_change_rate, cancel_rate

        inds["ob_depth_imb"] = depth_imbalance(book, levels=5)
        vwap_df = vwap_pressure(book, levels=10)
        if not vwap_df.empty:
            inds["ob_ask_pressure"] = vwap_df["ask_pressure"]
            inds["ob_bid_pressure"] = vwap_df["bid_pressure"]
        dcr = depth_change_rate(book, levels=5, window=10)
        if not dcr.empty:
            inds["ob_bid_change_rate"] = dcr["bid_depth_change_rate"]
            inds["ob_ask_change_rate"] = dcr["ask_depth_change_rate"]
        cr = cancel_rate(book, levels=5, window=20)
        if not cr.empty:
            inds["ob_bid_cancel"] = cr["bid_cancel_rate"]
            inds["ob_ask_cancel"] = cr["ask_cancel_rate"]

    return inds


def _get_indicator_category(name: str) -> str:
    """Map indicator name to window category."""
    if "impact" in name or "eff" in name:
        return "price_impact"
    if "spread" in name or "roll" in name:
        return "spread"
    if "imbalance" in name or "imb" in name or "toxicity" in name:
        return "imbalance"
    if "taker" in name or "levels" in name or "sweep" in name or "arrival" in name:
        return "taker_flow"
    if "ob_" in name:
        return "volume"
    return "volume"


def _extract_taker_features_at_event(
    taker_orders: pd.DataFrame,
    event_time: pd.Timestamp,
    pre_ms: int,
    post_ms: int,
) -> Dict[str, float]:
    """从 taker orders 中提取事件前后的特征。"""
    # 统一时区
    to_ts = taker_orders["timestamp"]
    if to_ts.dt.tz is None and getattr(event_time, "tzinfo", None) is not None:
        event_time = event_time.tz_localize(None)
    elif to_ts.dt.tz is not None and getattr(event_time, "tzinfo", None) is None:
        event_time = event_time.tz_localize(to_ts.dt.tz)

    pre_start = event_time - pd.Timedelta(milliseconds=pre_ms)
    post_end = event_time + pd.Timedelta(milliseconds=post_ms)

    pre = taker_orders[
        (to_ts >= pre_start) & (to_ts < event_time)
    ]
    post = taker_orders[
        (to_ts >= event_time) & (to_ts < post_end)
    ]

    feats = {}

    # Pre-event taker stats
    if len(pre) > 0:
        feats["taker_pre_n_orders"] = len(pre)
        feats["taker_pre_avg_levels"] = pre["levels_swept"].mean()
        feats["taker_pre_max_levels"] = pre["levels_swept"].max()
        feats["taker_pre_total_vol"] = pre["total_volume"].sum()
        feats["taker_pre_avg_impact_pct"] = pre["price_impact_pct"].mean()
        feats["taker_pre_large_pct"] = (pre["levels_swept"] >= 3).mean()

        buy_vol = pre.loc[pre["side"] == "buy", "total_volume"].sum()
        sell_vol = pre.loc[pre["side"] == "sell", "total_volume"].sum()
        feats["taker_pre_imbalance"] = (buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-10)
    else:
        for s in ["n_orders", "avg_levels", "max_levels", "total_vol",
                   "avg_impact_pct", "large_pct", "imbalance"]:
            feats[f"taker_pre_{s}"] = 0

    # Post-event
    if len(post) > 0:
        feats["taker_post_n_orders"] = len(post)
        feats["taker_post_avg_levels"] = post["levels_swept"].mean()
        feats["taker_post_total_vol"] = post["total_volume"].sum()
    else:
        feats["taker_post_n_orders"] = 0
        feats["taker_post_avg_levels"] = 0
        feats["taker_post_total_vol"] = 0

    # Shift
    feats["taker_vol_shift"] = (
        feats["taker_post_total_vol"] / (feats["taker_pre_total_vol"] + 1e-10)
    )
    feats["taker_levels_shift"] = (
        feats["taker_post_avg_levels"] - feats["taker_pre_avg_levels"]
    )

    return feats


def add_triple_barrier_labels(
    feature_df: pd.DataFrame,
    ohlcv: pd.DataFrame,
    horizons: Optional[List[int]] = None,
    upper_pct: float = 0.5,
    lower_pct: float = 0.3,
) -> pd.DataFrame:
    """
    为 FVG 特征矩阵添加 Triple Barrier 标签。

    Args:
        feature_df: build_fvg_features_v3 的输出
        ohlcv: OHLCV 数据
        horizons: 时间窗口列表 (bars)
        upper_pct / lower_pct: 上下轨百分比
    """
    from backtest.labeling import triple_barrier_labels

    if horizons is None:
        horizons = [10, 30, 60]

    prices = ohlcv["close"]
    result = feature_df.copy()

    for h in horizons:
        tb = triple_barrier_labels(prices, upper_pct=upper_pct, lower_pct=lower_pct, max_bars=h)

        col_label = f"tb_label_{h}"
        col_barrier = f"tb_barrier_{h}"
        col_bars = f"tb_bars_{h}"
        col_ret = f"tb_ret_{h}"

        labels = []
        barriers = []
        bars = []
        rets = []

        for _, row in feature_df.iterrows():
            ts = row["timestamp"]
            # 统一时区
            if tb.index.tz is None and getattr(ts, "tzinfo", None) is not None:
                ts = ts.tz_localize(None)
            elif tb.index.tz is not None and getattr(ts, "tzinfo", None) is None:
                ts = ts.tz_localize(tb.index.tz)
            if ts in tb.index:
                labels.append(int(tb.loc[ts, "label"]))
                barriers.append(tb.loc[ts, "barrier_hit"])
                bars.append(int(tb.loc[ts, "bars_to_hit"]))
                rets.append(float(tb.loc[ts, "return_at_hit"]))
            else:
                labels.append(0)
                barriers.append("unknown")
                bars.append(0)
                rets.append(0)

        result[col_label] = labels
        result[col_barrier] = barriers
        result[col_bars] = bars
        result[col_ret] = rets

    return result


def get_v3_feature_columns(feature_df: pd.DataFrame) -> List[str]:
    """
    从 V3 特征矩阵中自动提取可用于模型训练的特征列。
    排除标签列、timestamp、fvg_status 等。
    """
    exclude_prefixes = ("label_", "tb_label", "tb_barrier", "tb_bars", "tb_ret",
                        "future_ret", "timestamp", "fvg_status",
                        "target_", "pred", "proba")
    return [
        c for c in feature_df.columns
        if not any(c.startswith(p) for p in exclude_prefixes)
        and feature_df[c].dtype in [np.float64, np.int64, float, int]
    ]
