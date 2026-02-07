"""
Spread 估计：从逐笔成交数据估计买卖价差。

方法：
1. Roll Spread Estimator: 基于价格序列自相关
2. Trade-based Spread: 基于买卖成交价差
3. High-Low Spread: 基于最高/最低价
"""
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import pandas as pd


def estimate_spread_from_trades(
    trades: pd.DataFrame,
    method: str = "trade_diff",
    window_trades: int = 100,
) -> pd.Series:
    """
    从逐笔成交估计 spread。
    
    Methods:
    - "trade_diff": 买卖成交价差的中位数
    - "roll": Roll spread estimator (基于价格自相关)
    - "tick_range": 短窗口内最高-最低价
    
    Args:
        trades: DataFrame with [timestamp, price, amount, side]
        method: 估计方法
        window_trades: 滚动窗口大小
        
    Returns:
        Series with estimated spread (indexed by timestamp)
    """
    if trades.empty:
        return pd.Series(dtype=float)
    
    df = trades.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    if method == "trade_diff":
        return _spread_trade_diff(df, window_trades)
    elif method == "roll":
        return _spread_roll_estimator(df, window_trades)
    elif method == "tick_range":
        return _spread_tick_range(df, window_trades)
    else:
        raise ValueError(f"Unknown method: {method}")


def _spread_trade_diff(df: pd.DataFrame, window: int) -> pd.Series:
    """
    基于买卖成交价差估计 spread。
    
    Spread ≈ 2 * |买成交价 - 卖成交价| 的中位数
    （假设买单成交价接近 ask，卖单成交价接近 bid）
    """
    buy_prices = df.loc[df["side"] == "buy", "price"]
    sell_prices = df.loc[df["side"] == "sell", "price"]
    
    spreads = []
    for i in range(len(df)):
        start = max(0, i - window + 1)
        window_df = df.iloc[start:i+1]
        
        buy_p = window_df.loc[window_df["side"] == "buy", "price"]
        sell_p = window_df.loc[window_df["side"] == "sell", "price"]
        
        if len(buy_p) > 0 and len(sell_p) > 0:
            # 估计 spread: 买卖价的平均差
            spread = abs(buy_p.mean() - sell_p.mean())
        else:
            spread = np.nan
        spreads.append(spread)
    
    df["spread_est"] = spreads
    return df.set_index("timestamp")["spread_est"]


def _spread_roll_estimator(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Roll (1984) Spread Estimator.
    
    基于价格变动的自相关：
    Spread² ≈ -4 * Cov(ΔP_t, ΔP_{t-1})
    
    当自相关为负（反转）时，spread 估计才有效。
    """
    df["price_change"] = df["price"].diff()
    
    def roll_spread(window_data):
        changes = window_data["price_change"].dropna()
        if len(changes) < 3:
            return np.nan
        # 计算 Cov(ΔP_t, ΔP_{t-1})
        cov = np.cov(changes.iloc[1:], changes.iloc[:-1])[0, 1]
        if cov >= 0:
            return np.nan  # 自相关非负时估计无效
        return 2 * np.sqrt(-cov)
    
    spreads = []
    for i in range(len(df)):
        start = max(0, i - window + 1)
        spreads.append(roll_spread(df.iloc[start:i+1]))
    
    df["roll_spread"] = spreads
    return df.set_index("timestamp")["roll_spread"]


def _spread_tick_range(df: pd.DataFrame, window: int) -> pd.Series:
    """
    基于短窗口价格范围估计 spread。
    
    Spread ≈ High - Low （在足够短的窗口内）
    """
    spreads = []
    for i in range(len(df)):
        start = max(0, i - window + 1)
        window_prices = df.iloc[start:i+1]["price"]
        spreads.append(window_prices.max() - window_prices.min())
    
    df["range_spread"] = spreads
    return df.set_index("timestamp")["range_spread"]


def compute_spread_percentile(
    spread: pd.Series,
    lookback: int = 1000,
) -> pd.Series:
    """
    计算 spread 的历史分位水平。
    
    用于判断当前 spread 是否异常扩大。
    """
    if spread.empty:
        return pd.Series(dtype=float)
    
    def percentile_rank(window):
        if len(window) < 2:
            return np.nan
        current = window.iloc[-1]
        return (window < current).sum() / len(window) * 100
    
    return spread.rolling(lookback, min_periods=10).apply(percentile_rank, raw=False)


def detect_spread_expansion(
    trades: pd.DataFrame,
    percentile_threshold: float = 90.0,
    lookback: int = 1000,
    window_trades: int = 50,
) -> pd.DataFrame:
    """
    检测 spread 扩大事件。
    
    Returns:
        DataFrame with columns [timestamp, spread, percentile, is_expansion]
    """
    spread = estimate_spread_from_trades(trades, method="trade_diff", window_trades=window_trades)
    percentile = compute_spread_percentile(spread, lookback=lookback)
    
    result = pd.DataFrame({
        "spread": spread,
        "percentile": percentile,
    })
    result["is_expansion"] = result["percentile"] >= percentile_threshold
    
    return result
