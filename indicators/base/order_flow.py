"""
订单流指标。

原理：通过买卖量的不对称性衡量市场方向性压力。
  - trade_imbalance: (BuyVol - SellVol) / TotalVol
  - vpin: Volume-Synchronized Probability of Informed Trading
  - aggressive_flow_events: 激进订单流事件检测
  - flow_toxicity: 订单流毒性（大单 taker 对价格的影响）

机制：极端不平衡 → 单侧流动性被消耗 → 反转信号。
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def trade_imbalance(
    trades: pd.DataFrame,
    window_trades: int = 100,
) -> pd.Series:
    """
    滚动窗口内的订单流不平衡：(BuyVol - SellVol) / TotalVol。
    范围 [-1, 1]。
    """
    ts = trades.copy().sort_values("timestamp")
    sign = ts["side"].map({"buy": 1, "sell": -1}).fillna(0)
    signed_vol = sign * ts["amount"]

    ts_idx = ts.set_index("timestamp")
    ts_idx = ts_idx[~ts_idx.index.duplicated(keep="last")]

    buy_vol = ts_idx.loc[ts_idx["side"] == "buy", "amount"].rolling(window_trades, min_periods=5).sum()
    sell_vol = ts_idx.loc[ts_idx["side"] == "sell", "amount"].rolling(window_trades, min_periods=5).sum()

    # Reindex to full
    bv = buy_vol.reindex(ts_idx.index, method="ffill").fillna(0)
    sv = sell_vol.reindex(ts_idx.index, method="ffill").fillna(0)
    total = bv + sv + 1e-10

    result = (bv - sv) / total
    result.name = "trade_imbalance"
    return result


def vpin(
    trades: pd.DataFrame,
    bucket_volume: float = 1000,
    n_buckets: int = 50,
    sigma_window: int = 200,
) -> pd.Series:
    """
    VPIN: Volume-Synchronized Probability of Informed Trading.

    标准 Bulk Volume Classification (BVC) 方法 (Easley, Lopez de Prado, O'Hara):

        V_τ^B = Σ V_i · Z( (P_i - P_{i-1}) / σ_{ΔP} )
        V_τ^S = Σ V_i · [1 - Z( (P_i - P_{i-1}) / σ_{ΔP} )] = V_τ - V_τ^B

    其中 Z 为标准正态 CDF。通过价格变动的标准化概率来分配每笔成交量
    到买方/卖方，而非依赖 trade side 字段。

    VPIN = rolling_mean( |V_τ^B - V_τ^S| / V_τ )

    Args:
        trades: 含 timestamp, price, amount 列的 DataFrame
        bucket_volume: 每桶的目标成交量
        n_buckets: VPIN 滚动窗口中的桶数
        sigma_window: 计算价格变动标准差的滚动窗口（逐笔）
    """
    from scipy.stats import norm

    ts = trades.copy().sort_values("timestamp").reset_index(drop=True)

    # ── Step 1: BVC — 用标准正态 CDF 概率分配买卖量 ──
    dp = ts["price"].diff().fillna(0)
    sigma_dp = dp.rolling(sigma_window, min_periods=20).std().fillna(dp.expanding().std())
    sigma_dp = sigma_dp.replace(0, np.nan).ffill().fillna(1e-10)

    z_score = dp / sigma_dp
    buy_prob = norm.cdf(z_score)  # Z((P_i - P_{i-1}) / σ_{ΔP})

    ts["buy_vol"] = ts["amount"] * buy_prob       # V_i · Z(·)
    ts["sell_vol"] = ts["amount"] * (1 - buy_prob)  # V_i · [1 - Z(·)]

    # ── Step 2: 等量分桶 ──
    cum_vol = ts["amount"].cumsum()
    ts["bucket"] = (cum_vol / bucket_volume).astype(int)

    bucket_stats = ts.groupby("bucket").agg(
        buy_vol=("buy_vol", "sum"),
        sell_vol=("sell_vol", "sum"),
        total_vol=("amount", "sum"),
        last_ts=("timestamp", "last"),
    )

    # ── Step 3: 计算 VPIN ──
    bucket_stats["imbalance"] = (
        (bucket_stats["buy_vol"] - bucket_stats["sell_vol"]).abs()
        / (bucket_stats["total_vol"] + 1e-10)
    )

    vpin_series = bucket_stats["imbalance"].rolling(n_buckets, min_periods=5).mean()

    # Map back to timestamp
    result = pd.Series(index=pd.to_datetime(bucket_stats["last_ts"]), data=vpin_series.values)
    result = result[~result.index.duplicated(keep="last")]
    result.name = "vpin"
    return result


def aggressive_flow_events(
    trades: pd.DataFrame,
    window_ms: int = 5000,
    imbalance_threshold: float = 0.7,
    volume_ratio_threshold: float = 2.0,
) -> pd.DataFrame:
    """
    检测激进订单流事件：短时间内大量单侧成交。

    Returns:
        DataFrame with event timestamps and metadata
    """
    ts = trades.set_index("timestamp").sort_index()
    ts = ts[~ts.index.duplicated(keep="last")]
    window = pd.Timedelta(milliseconds=window_ms)

    bv = ts.loc[ts["side"] == "buy", "amount"].rolling(window).sum().reindex(ts.index, method="ffill").fillna(0)
    sv = ts.loc[ts["side"] == "sell", "amount"].rolling(window).sum().reindex(ts.index, method="ffill").fillna(0)
    tv = bv + sv + 1e-10

    imbalance = (bv - sv) / tv
    avg_vol = tv.rolling(window * 5).mean()
    vol_ratio = tv / (avg_vol + 1e-10)

    events_mask = (imbalance.abs() >= imbalance_threshold) & (vol_ratio >= volume_ratio_threshold)
    events = ts[events_mask].copy()
    events["imbalance"] = imbalance[events_mask]
    events["vol_ratio"] = vol_ratio[events_mask]

    return events.reset_index()


def flow_toxicity(
    trades: pd.DataFrame,
    window_trades: int = 200,
) -> pd.Series:
    """
    订单流毒性：衡量 taker 对价格的破坏力。
    toxicity = abs(ΔPrice_window) / mean(spread_proxy)
    """
    ts = trades.set_index("timestamp").sort_index()
    ts = ts[~ts.index.duplicated(keep="last")]

    dp = ts["price"].diff(window_trades).abs()
    spread_proxy = ts["price"].diff().abs().rolling(window_trades).mean() + 1e-10

    result = dp / spread_proxy
    result.name = "flow_toxicity"
    return result
