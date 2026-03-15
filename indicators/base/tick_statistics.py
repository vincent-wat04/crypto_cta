"""
Tick-Level 秒内微观统计特征（从原始 aggTrades 提取）。

每个 bar（如 1s）内部的 tick-level 统计：
  - tick_n_prices:       成交了多少个不同价位 (spread/深度碎片化)
  - tick_max_trade:      最大单笔成交量 (鲸鱼/机构检测)
  - tick_avg_trade:      平均单笔成交量
  - tick_size_cv:        成交量变异系数 (交易异质性)
  - tick_vwap_dev:       VWAP vs close 偏差 bps (执行压力方向)
  - tick_buy_avg_size:   买方平均成交 size
  - tick_sell_avg_size:  卖方平均成交 size
  - tick_size_imbalance: 买卖方 size 不对称度
  - tick_large_pct:      大单成交量占比 (>2x 均值)
  - tick_range_per_trade: 价格范围 / 交易笔数
  - tick_n_agg:          aggTrade 笔数

这些特征捕捉了 OHLCV bar 无法表达的秒内微观结构。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_tick_stats(
    trades: pd.DataFrame,
    freq: str = "1s",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    从原始 aggTrades 提取每个 bar 内的 tick-level 微观统计。

    Args:
        trades: aggTrades DataFrame (columns: timestamp, price, amount, side, ...)
        freq: bar 频率 (默认 "1s")
        windows: rolling 窗口 (bar 数)。默认 [3, 5, 10, 20]

    Returns:
        DataFrame indexed by bar timestamp
    """
    if windows is None:
        windows = [3, 5, 10, 20]

    t = trades.copy()
    t["timestamp"] = pd.to_datetime(t["timestamp"])
    t["bar"] = t["timestamp"].dt.floor(freq)

    g = t.groupby("bar")
    tf = pd.DataFrame(index=sorted(g.groups.keys()))
    tf.index.name = "timestamp"

    # ── Distinct price levels ──
    tf["tick_n_prices"] = g["price"].nunique()

    # ── Trade size stats ──
    tf["tick_max_trade"] = g["amount"].max()
    tf["tick_avg_trade"] = g["amount"].mean()
    qty_std = g["amount"].std().fillna(0)
    qty_mean = g["amount"].mean()
    tf["tick_size_cv"] = qty_std / (qty_mean + 1e-10)

    # ── VWAP deviation (bps) ──
    t["pv"] = t["price"] * t["amount"]
    pv_sum = t.groupby("bar")["pv"].sum()
    v_sum = g["amount"].sum()
    vwap = pv_sum / (v_sum + 1e-10)
    last_price = g["price"].last()
    tf["tick_vwap_dev"] = (vwap - last_price) / (last_price + 1e-10) * 10000

    # ── Buy vs Sell avg size ──
    is_buy = t["side"] == "buy"
    buy_sum = t[is_buy].groupby("bar")["amount"].sum()
    buy_cnt = t[is_buy].groupby("bar")["amount"].count()
    sell_sum = t[~is_buy].groupby("bar")["amount"].sum()
    sell_cnt = t[~is_buy].groupby("bar")["amount"].count()

    buy_avg = (buy_sum / (buy_cnt + 1e-10)).reindex(tf.index).fillna(0)
    sell_avg = (sell_sum / (sell_cnt + 1e-10)).reindex(tf.index).fillna(0)
    tf["tick_buy_avg_size"] = buy_avg
    tf["tick_sell_avg_size"] = sell_avg
    tf["tick_size_imbalance"] = (buy_avg - sell_avg) / (buy_avg + sell_avg + 1e-10)

    # ── Large trade ratio (volume from trades > 2× mean) ──
    overall_mean = t.groupby("bar")["amount"].transform("mean")
    t["_is_large"] = t["amount"] > 2 * overall_mean
    large_vol = t[t["_is_large"]].groupby("bar")["amount"].sum()
    total_vol = g["amount"].sum()
    tf["tick_large_pct"] = (large_vol / (total_vol + 1e-10)).reindex(tf.index).fillna(0)

    # ── Range per trade (bps) ──
    p_range = g["price"].max() - g["price"].min()
    p_count = g["price"].count()
    p_mean = g["price"].mean()
    tf["tick_range_per_trade"] = p_range / (p_count + 1e-10) / (p_mean + 1e-10) * 10000

    # ── Count (number of agg trades at same price level) ──
    tf["tick_n_agg"] = g["price"].count()

    # ── Rolling aggregations ──
    roll_cols = [
        "tick_n_prices", "tick_max_trade", "tick_size_cv",
        "tick_vwap_dev", "tick_size_imbalance", "tick_large_pct",
        "tick_range_per_trade", "tick_n_agg",
    ]
    for w in windows:
        for col in roll_cols:
            if col in tf.columns:
                tf[f"{col}_ma_{w}"] = tf[col].rolling(w, min_periods=1).mean()

    tf = tf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return tf
