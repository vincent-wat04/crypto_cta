"""
Spread 估计指标。

原理：无直接 orderbook 时从逐笔成交推断 spread。
  - trade_diff_spread: 相邻成交价差的中位数
  - roll_spread: Roll (1984) 估计量 = 2 * sqrt(-Cov(ΔP_t, ΔP_{t-1}))
  - effective_spread: 基于买卖成交价差
  - spread_percentile: spread 的历史分位数

机制：spread 扩大 → 流动性降低 → 做市商退出 → 反转前兆。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def trade_diff_spread(
    trades: pd.DataFrame,
    window_trades: int = 100,
) -> pd.Series:
    """
    相邻成交价差的滚动中位数。
    """
    ts = trades.set_index("timestamp").sort_index()
    ts = ts[~ts.index.duplicated(keep="last")]
    diffs = ts["price"].diff().abs()
    result = diffs.rolling(window_trades, min_periods=10).median()
    result.name = "trade_diff_spread"
    return result


def roll_spread(
    trades: pd.DataFrame,
    window_trades: int = 200,
) -> pd.Series:
    """
    Roll (1984) Spread Estimator: 2 * sqrt(max(0, -Cov(ΔP_t, ΔP_{t-1})))
    """
    ts = trades.set_index("timestamp").sort_index()
    ts = ts[~ts.index.duplicated(keep="last")]
    dp = ts["price"].diff()
    dp_lag = dp.shift(1)
    cov = dp.rolling(window_trades, min_periods=30).cov(dp_lag)
    result = 2 * np.sqrt(np.maximum(0, -cov))
    result.name = "roll_spread"
    return result


def effective_spread(
    trades: pd.DataFrame,
    window_ms: int = 5000,
) -> pd.Series:
    """
    有效 spread 估计：同一窗口内买入最低价与卖出最高价的差。
    """
    ts = trades.set_index("timestamp").sort_index()
    window = pd.Timedelta(milliseconds=window_ms)

    buy_prices = ts.loc[ts["side"] == "buy", "price"]
    sell_prices = ts.loc[ts["side"] == "sell", "price"]

    buy_min = buy_prices.rolling(window, min_periods=1).min()
    sell_max = sell_prices.rolling(window, min_periods=1).max()

    # 合并到统一索引
    combined = pd.DataFrame(index=ts.index)
    combined["buy_min"] = buy_min.reindex(ts.index, method="ffill")
    combined["sell_max"] = sell_max.reindex(ts.index, method="ffill")
    result = (combined["buy_min"] - combined["sell_max"]).abs()
    result.name = "effective_spread"
    return result


def spread_percentile(
    spread_series: pd.Series,
    lookback: int = 500,
) -> pd.Series:
    """
    Spread 的滚动历史分位数 (0~100)。
    """
    result = spread_series.rolling(lookback, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )
    result.name = "spread_percentile"
    return result
