"""
Orderbook 压力指标（需要 orderbook 数据）。

原理：
  - vwap_pressure: (ask_vwap_N - mid) / mid 与 (mid - bid_vwap_N) / mid
  - depth_imbalance: bid 侧深度 vs ask 侧深度
  - weighted_depth_slope: 按价格距离加权的深度分布斜率

机制：
  ask_vwap 远离 mid → 卖方流动性稀薄 → 买方更容易推动价格上涨
  depth_imbalance 极端 → 一侧被扫空 → 反转前兆
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def vwap_pressure(
    book: pd.DataFrame,
    levels: int = 10,
) -> pd.DataFrame:
    """
    Orderbook VWAP 偏离 mid 的程度：
      ask_pressure = (ask_vwap_N - mid) / mid
      bid_pressure = (mid - bid_vwap_N) / mid

    Args:
        book: DataFrame with columns bid_0_price, bid_0_size, ..., ask_9_size
        levels: 使用的档位数

    Returns:
        DataFrame with: mid, ask_vwap, bid_vwap, ask_pressure, bid_pressure, pressure_imbalance
    """
    # Check available levels
    avail_bid = [i for i in range(levels) if f"bid_{i}_price" in book.columns]
    avail_ask = [i for i in range(levels) if f"ask_{i}_price" in book.columns]

    if not avail_bid or not avail_ask:
        return pd.DataFrame(index=book.index)

    mid = (book["bid_0_price"].astype(float) + book["ask_0_price"].astype(float)) / 2

    # Ask VWAP
    ask_pv = sum(book[f"ask_{i}_price"].astype(float) * book[f"ask_{i}_size"].astype(float)
                 for i in avail_ask)
    ask_v = sum(book[f"ask_{i}_size"].astype(float) for i in avail_ask) + 1e-10
    ask_vwap = ask_pv / ask_v

    # Bid VWAP
    bid_pv = sum(book[f"bid_{i}_price"].astype(float) * book[f"bid_{i}_size"].astype(float)
                 for i in avail_bid)
    bid_v = sum(book[f"bid_{i}_size"].astype(float) for i in avail_bid) + 1e-10
    bid_vwap = bid_pv / bid_v

    ask_pressure = (ask_vwap - mid) / (mid + 1e-10)
    bid_pressure = (mid - bid_vwap) / (mid + 1e-10)
    pressure_imbalance = ask_pressure - bid_pressure

    return pd.DataFrame({
        "mid": mid,
        "ask_vwap": ask_vwap,
        "bid_vwap": bid_vwap,
        "ask_pressure": ask_pressure,
        "bid_pressure": bid_pressure,
        "pressure_imbalance": pressure_imbalance,
    }, index=book.index)


def depth_imbalance(
    book: pd.DataFrame,
    levels: int = 10,
) -> pd.Series:
    """
    Bid/Ask 深度不平衡：(bid_depth - ask_depth) / (bid_depth + ask_depth)。
    范围 [-1, 1]。正值 → 买方更厚 → 支撑。
    """
    avail_bid = [i for i in range(levels) if f"bid_{i}_size" in book.columns]
    avail_ask = [i for i in range(levels) if f"ask_{i}_size" in book.columns]

    bid_depth = sum(book[f"bid_{i}_size"].astype(float) for i in avail_bid)
    ask_depth = sum(book[f"ask_{i}_size"].astype(float) for i in avail_ask)

    total = bid_depth + ask_depth + 1e-10
    result = (bid_depth - ask_depth) / total
    result.name = "depth_imbalance"
    return result


def weighted_depth_slope(
    book: pd.DataFrame,
    levels: int = 10,
) -> pd.DataFrame:
    """
    按价格距离加权的深度分布斜率。

    斜率越陡 → 流动性集中在近档 → 容易被冲击。
    """
    avail_bid = [i for i in range(levels) if f"bid_{i}_price" in book.columns]
    avail_ask = [i for i in range(levels) if f"ask_{i}_price" in book.columns]

    if len(avail_bid) < 2 or len(avail_ask) < 2:
        return pd.DataFrame(index=book.index)

    mid = (book["bid_0_price"].astype(float) + book["ask_0_price"].astype(float)) / 2

    # Ask side slope: size vs distance
    def _slope(side, levels_list):
        n = len(levels_list)
        if n < 2:
            return pd.Series(0, index=book.index)
        distances = []
        sizes = []
        for i in levels_list:
            p = book[f"{side}_{i}_price"].astype(float)
            s = book[f"{side}_{i}_size"].astype(float)
            d = (p - mid).abs() / (mid + 1e-10)
            distances.append(d)
            sizes.append(s)

        # Simple linear regression slope for each row
        d_stack = np.column_stack([d.values for d in distances])
        s_stack = np.column_stack([s.values for s in sizes])

        d_mean = d_stack.mean(axis=1, keepdims=True)
        s_mean = s_stack.mean(axis=1, keepdims=True)
        cov = ((d_stack - d_mean) * (s_stack - s_mean)).sum(axis=1)
        var = ((d_stack - d_mean) ** 2).sum(axis=1) + 1e-10
        return pd.Series(cov / var, index=book.index)

    return pd.DataFrame({
        "ask_depth_slope": _slope("ask", avail_ask),
        "bid_depth_slope": _slope("bid", avail_bid),
    }, index=book.index)
