"""
Price Impact 计算：单位成交量对价格的推动力度。

理论基础：
- Kyle's Lambda: 衡量信息冲击对价格的影响
- Price Impact = |ΔP| / V（价格变动 / 成交量）
- 大 impact 表示流动性差或存在信息交易
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class ImpactEvent:
    """单个价格冲击事件。"""
    timestamp: pd.Timestamp
    direction: int  # 1 = 上冲击, -1 = 下冲击
    price_change_pct: float
    volume: float
    impact_ratio: float  # |price_change| / volume
    duration_ms: float
    num_trades: int


def compute_tick_price_impact(
    trades: pd.DataFrame,
    window_ms: int = 1000,
) -> pd.DataFrame:
    """
    计算每个时间窗口内的 tick-level price impact。
    
    Args:
        trades: DataFrame with columns [timestamp, price, amount, side]
        window_ms: 时间窗口（毫秒）
        
    Returns:
        DataFrame with columns [timestamp, price_change, volume, impact, direction]
    """
    if trades.empty:
        return pd.DataFrame()
    
    df = trades.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # 按时间窗口分组
    df["window"] = (df["timestamp"].astype(np.int64) // 10**6 // window_ms).astype(int)
    
    results = []
    for window_id, group in df.groupby("window"):
        if len(group) < 2:
            continue
            
        first_price = group["price"].iloc[0]
        last_price = group["price"].iloc[-1]
        price_change = last_price - first_price
        price_change_pct = price_change / first_price * 100
        
        total_volume = group["amount"].sum()
        buy_volume = group.loc[group["side"] == "buy", "amount"].sum()
        sell_volume = group.loc[group["side"] == "sell", "amount"].sum()
        
        # 方向：净买入为正，净卖出为负
        net_flow = buy_volume - sell_volume
        direction = 1 if net_flow > 0 else (-1 if net_flow < 0 else 0)
        
        # Impact ratio: |price_change_pct| / volume
        impact = abs(price_change_pct) / (total_volume + 1e-10)
        
        results.append({
            "timestamp": group["timestamp"].iloc[0],
            "price_change_pct": price_change_pct,
            "volume": total_volume,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "impact": impact,
            "direction": direction,
            "num_trades": len(group),
        })
    
    return pd.DataFrame(results)


def compute_volume_weighted_impact(
    trades: pd.DataFrame,
    lookback_trades: int = 100,
) -> pd.Series:
    """
    计算成交量加权的 price impact（滚动）。
    
    每笔成交的 impact = |price_change| / amount
    返回滚动均值。
    """
    if trades.empty or len(trades) < 2:
        return pd.Series(dtype=float)
    
    df = trades.copy()
    df["price_change"] = df["price"].diff().abs()
    df["tick_impact"] = df["price_change"] / (df["amount"] + 1e-10)
    
    # 成交量加权的滚动平均
    df["weighted_impact"] = (
        (df["tick_impact"] * df["amount"])
        .rolling(lookback_trades, min_periods=1)
        .sum()
        / df["amount"].rolling(lookback_trades, min_periods=1).sum()
    )
    
    return df.set_index("timestamp")["weighted_impact"]


def compute_kyle_lambda(
    trades: pd.DataFrame,
    window_trades: int = 50,
) -> pd.Series:
    """
    计算 Kyle's Lambda（信息冲击系数）。
    
    Lambda = Cov(ΔP, OF) / Var(OF)
    其中 OF = signed order flow (买量 - 卖量)
    
    高 Lambda 表示市场对订单流敏感（流动性差或信息不对称）。
    """
    if trades.empty or len(trades) < window_trades:
        return pd.Series(dtype=float)
    
    df = trades.copy()
    df["price_change"] = df["price"].diff()
    df["signed_flow"] = np.where(df["side"] == "buy", df["amount"], -df["amount"])
    
    # 滚动计算协方差和方差
    def rolling_lambda(window):
        if len(window) < 2:
            return np.nan
        cov = np.cov(window["price_change"].dropna(), window["signed_flow"].dropna())
        if cov.shape != (2, 2):
            return np.nan
        var_flow = cov[1, 1]
        if var_flow < 1e-10:
            return np.nan
        return cov[0, 1] / var_flow
    
    lambdas = []
    for i in range(len(df)):
        start = max(0, i - window_trades + 1)
        window = df.iloc[start:i+1]
        lambdas.append(rolling_lambda(window))
    
    df["kyle_lambda"] = lambdas
    return df.set_index("timestamp")["kyle_lambda"]


def detect_large_impact_events(
    trades: pd.DataFrame,
    impact_threshold_pct: float = 0.3,
    volume_threshold_ratio: float = 2.0,
    window_ms: int = 5000,
    lookback_windows: int = 20,
) -> List[ImpactEvent]:
    """
    检测大规模价格冲击事件。
    
    条件：
    1. 价格变动 > impact_threshold_pct
    2. 成交量 > 历史均值 * volume_threshold_ratio
    
    Args:
        trades: 逐笔成交数据
        impact_threshold_pct: 价格变动阈值（%）
        volume_threshold_ratio: 成交量相对历史均值的倍数
        window_ms: 统计窗口（毫秒）
        lookback_windows: 用于计算历史均值的窗口数
    
    Returns:
        List of ImpactEvent
    """
    impact_df = compute_tick_price_impact(trades, window_ms=window_ms)
    if impact_df.empty:
        return []
    
    # 计算历史成交量均值
    impact_df["vol_avg"] = impact_df["volume"].rolling(
        lookback_windows, min_periods=1
    ).mean()
    impact_df["vol_ratio"] = impact_df["volume"] / (impact_df["vol_avg"] + 1e-10)
    
    events = []
    for _, row in impact_df.iterrows():
        if (
            abs(row["price_change_pct"]) >= impact_threshold_pct
            and row["vol_ratio"] >= volume_threshold_ratio
        ):
            events.append(ImpactEvent(
                timestamp=row["timestamp"],
                direction=1 if row["price_change_pct"] > 0 else -1,
                price_change_pct=row["price_change_pct"],
                volume=row["volume"],
                impact_ratio=row["impact"],
                duration_ms=window_ms,
                num_trades=row["num_trades"],
            ))
    
    return events
