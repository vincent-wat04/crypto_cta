"""
Order Flow 分析：订单流不平衡与激进交易检测。

核心指标：
1. Trade Imbalance: (买量 - 卖量) / 总量
2. VPIN: Volume-synchronized Probability of Informed Trading
3. Aggressive Flow: 大单/快速连续成交检测
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd


@dataclass
class AggressiveFlowEvent:
    """激进订单流事件。"""
    timestamp: pd.Timestamp
    direction: int  # 1 = 买方激进, -1 = 卖方激进
    volume: float
    num_trades: int
    duration_ms: float
    avg_trade_size: float


def compute_trade_imbalance(
    trades: pd.DataFrame,
    window_trades: int = 100,
) -> pd.Series:
    """
    计算订单流不平衡。
    
    Imbalance = (Buy Volume - Sell Volume) / Total Volume
    范围 [-1, 1]，正值表示买方主导，负值表示卖方主导。
    """
    if trades.empty:
        return pd.Series(dtype=float)
    
    df = trades.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    df["buy_vol"] = np.where(df["side"] == "buy", df["amount"], 0)
    df["sell_vol"] = np.where(df["side"] == "sell", df["amount"], 0)
    
    buy_rolling = df["buy_vol"].rolling(window_trades, min_periods=1).sum()
    sell_rolling = df["sell_vol"].rolling(window_trades, min_periods=1).sum()
    total_rolling = buy_rolling + sell_rolling
    
    df["imbalance"] = (buy_rolling - sell_rolling) / (total_rolling + 1e-10)
    
    return df.set_index("timestamp")["imbalance"]


def compute_vpin(
    trades: pd.DataFrame,
    bucket_size: float = 1000.0,
    n_buckets: int = 50,
) -> pd.Series:
    """
    计算 VPIN (Volume-synchronized Probability of Informed Trading)。
    
    VPIN = Σ|V_buy - V_sell| / (n * bucket_size)
    
    高 VPIN 表示存在信息交易（订单流方向性强）。
    
    Args:
        trades: 逐笔成交
        bucket_size: 每个成交量桶的大小
        n_buckets: 用于计算 VPIN 的桶数
    """
    if trades.empty:
        return pd.Series(dtype=float)
    
    df = trades.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # 计算累计成交量
    df["cum_volume"] = df["amount"].cumsum()
    df["bucket_id"] = (df["cum_volume"] / bucket_size).astype(int)
    
    # 按桶聚合
    buckets = df.groupby("bucket_id").agg({
        "timestamp": "first",
        "amount": "sum",
        "side": lambda x: (x == "buy").sum(),  # 买单数量
    }).rename(columns={"side": "n_buys", "amount": "volume"})
    
    # 估计每桶的买/卖量
    # 简化：假设 trade 数量比例 ≈ 成交量比例
    total_trades_per_bucket = df.groupby("bucket_id").size()
    buckets["buy_ratio"] = buckets["n_buys"] / total_trades_per_bucket
    buckets["buy_vol"] = buckets["volume"] * buckets["buy_ratio"]
    buckets["sell_vol"] = buckets["volume"] * (1 - buckets["buy_ratio"])
    
    # 计算 |buy - sell|
    buckets["abs_imbalance"] = abs(buckets["buy_vol"] - buckets["sell_vol"])
    
    # 滚动 VPIN
    buckets["vpin"] = (
        buckets["abs_imbalance"].rolling(n_buckets, min_periods=1).sum()
        / (n_buckets * bucket_size)
    )
    
    # 映射回原始时间戳
    df["vpin"] = df["bucket_id"].map(buckets["vpin"])
    
    return df.set_index("timestamp")["vpin"]


def detect_aggressive_flow(
    trades: pd.DataFrame,
    window_ms: int = 1000,
    imbalance_threshold: float = 0.7,
    min_volume: float = 100.0,
    volume_ratio_threshold: float = 2.0,
) -> List[AggressiveFlowEvent]:
    """
    检测激进订单流事件。
    
    条件：
    1. 订单流不平衡 > imbalance_threshold（或 < -threshold）
    2. 成交量 > min_volume
    3. 成交量 > 历史均值 * volume_ratio_threshold
    
    Returns:
        List of AggressiveFlowEvent
    """
    if trades.empty:
        return []
    
    df = trades.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # 按时间窗口分组
    df["window"] = (df["timestamp"].astype(np.int64) // 10**6 // window_ms).astype(int)
    
    # 计算每窗口的统计
    window_stats = df.groupby("window").agg({
        "timestamp": "first",
        "amount": "sum",
        "side": lambda x: ((x == "buy").sum(), (x == "sell").sum()),
    })
    
    window_stats["buy_vol"] = df.groupby("window").apply(
        lambda x: x.loc[x["side"] == "buy", "amount"].sum()
    )
    window_stats["sell_vol"] = df.groupby("window").apply(
        lambda x: x.loc[x["side"] == "sell", "amount"].sum()
    )
    window_stats["total_vol"] = window_stats["buy_vol"] + window_stats["sell_vol"]
    window_stats["imbalance"] = (
        (window_stats["buy_vol"] - window_stats["sell_vol"]) 
        / (window_stats["total_vol"] + 1e-10)
    )
    window_stats["n_trades"] = df.groupby("window").size()
    
    # 历史成交量均值
    window_stats["vol_avg"] = window_stats["total_vol"].rolling(20, min_periods=1).mean()
    window_stats["vol_ratio"] = window_stats["total_vol"] / (window_stats["vol_avg"] + 1e-10)
    
    events = []
    for idx, row in window_stats.iterrows():
        if (
            abs(row["imbalance"]) >= imbalance_threshold
            and row["total_vol"] >= min_volume
            and row["vol_ratio"] >= volume_ratio_threshold
        ):
            events.append(AggressiveFlowEvent(
                timestamp=row["timestamp"],
                direction=1 if row["imbalance"] > 0 else -1,
                volume=row["total_vol"],
                num_trades=row["n_trades"],
                duration_ms=window_ms,
                avg_trade_size=row["total_vol"] / (row["n_trades"] + 1e-10),
            ))
    
    return events


def compute_flow_toxicity(
    trades: pd.DataFrame,
    window_trades: int = 100,
) -> pd.Series:
    """
    计算订单流毒性（flow toxicity）。
    
    毒性 = |累计不平衡| / 成交笔数
    高毒性表示订单流方向一致（可能是知情交易）。
    """
    if trades.empty:
        return pd.Series(dtype=float)
    
    df = trades.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["signed_vol"] = np.where(df["side"] == "buy", df["amount"], -df["amount"])
    
    # 累计有符号成交量
    df["cum_signed"] = df["signed_vol"].cumsum()
    
    # 滚动毒性
    def toxicity(window):
        if len(window) < 2:
            return np.nan
        abs_cum = abs(window["signed_vol"].sum())
        return abs_cum / len(window)
    
    toxicities = []
    for i in range(len(df)):
        start = max(0, i - window_trades + 1)
        toxicities.append(toxicity(df.iloc[start:i+1]))
    
    df["toxicity"] = toxicities
    return df.set_index("timestamp")["toxicity"]
