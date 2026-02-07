"""
Taker Order 预处理 + 指标（基于 aggTrades 的 f/l 字段）。

核心：将连续的 aggTrades 分组为 taker orders，还原完整的
主动单消耗 orderbook 的路径。

分组规则：
  连续 aggTrade 满足 f[i] == l[i-1] + 1 且 m[i] == m[i-1]
  时属于同一 taker order。

单条 aggTrade = 同一 taker + 同一价格档位的全部 fills
taker order  = 同一 taker 的全部 aggTrades（可能跨多个价格档位）

衍生指标：
  Per-order:  levels_swept, total_volume, price_impact, impact_per_volume, ...
  Time series: rolling avg_levels, sweep_autocorr, large_taker_ratio, ...
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────
# Taker Order 分组
# ─────────────────────────────────────────────────────────

def group_taker_orders(trades: pd.DataFrame) -> pd.DataFrame:
    """
    将 aggTrades 分组为 taker orders。

    要求 trades 包含列：
        agg_trade_id, first_trade_id, last_trade_id, side, price, amount, timestamp

    分组规则：
        连续的 aggTrade 满足以下条件时属于同一 taker order：
        1. first_trade_id[i] == last_trade_id[i-1] + 1
        2. side[i] == side[i-1]  (m 字段一致)

    Returns:
        DataFrame, 每行一个 taker order:
            taker_id, timestamp, side,
            levels_swept, total_volume, total_fills,
            first_price, last_price, vwap,
            price_impact, price_impact_pct, impact_per_volume,
            duration_ms, volume_taper,
            n_fills_per_level (list → mean/std)
    """
    if trades.empty or "first_trade_id" not in trades.columns:
        return pd.DataFrame()

    df = trades.sort_values("agg_trade_id").reset_index(drop=True)

    # ── 1. 标记 taker order 边界 ──
    # 新 taker order 开始条件：
    #   first_trade_id[i] != last_trade_id[i-1] + 1
    #   OR side[i] != side[i-1]
    prev_last = df["last_trade_id"].shift(1)
    prev_side = df["side"].shift(1)

    is_new_taker = (
        (df["first_trade_id"] != prev_last + 1)
        | (df["side"] != prev_side)
        | (df.index == 0)
    )
    df["taker_id"] = is_new_taker.cumsum()

    # ── 2. 向量化聚合（避免 groupby.apply 对大数据集的性能瓶颈）──
    tid = df["taker_id"].values
    prices = df["price"].values
    amounts = df["amount"].values
    timestamps = df["timestamp"].values
    sides = df["side"].values

    # cost
    if "cost" in df.columns:
        costs = df["cost"].values
    else:
        costs = prices * amounts

    # n_trades_in_agg
    if "n_trades_in_agg" in df.columns:
        n_fills_arr = df["n_trades_in_agg"].values
    else:
        n_fills_arr = np.ones(len(df), dtype=np.float64)

    # 标记每个 taker group 的首行和尾行
    first_mask = np.empty(len(df), dtype=bool)
    first_mask[0] = True
    first_mask[1:] = tid[1:] != tid[:-1]

    last_mask = np.empty(len(df), dtype=bool)
    last_mask[-1] = True
    last_mask[:-1] = tid[:-1] != tid[1:]

    # 用 groupby agg（高效原生聚合，不用 apply）
    gb = df.groupby("taker_id", sort=False)
    agg_result = gb.agg(
        levels_swept=("amount", "size"),
        total_volume=("amount", "sum"),
    )

    # cost groupby sum
    df["_cost"] = costs
    df["_nfills"] = n_fills_arr
    cost_fills = df.groupby("taker_id", sort=False).agg(
        total_cost=("_cost", "sum"),
        total_fills=("_nfills", "sum"),
        fills_mean=("_nfills", "mean"),
        fills_std=("_nfills", "std"),
    )
    agg_result = agg_result.join(cost_fills)
    agg_result["fills_std"] = agg_result["fills_std"].fillna(0)

    # first/last per group (from boolean masks)
    first_idx = np.where(first_mask)[0]
    last_idx = np.where(last_mask)[0]

    agg_result["timestamp"] = timestamps[first_idx]
    agg_result["side"] = sides[first_idx]
    agg_result["first_price"] = prices[first_idx]
    agg_result["last_price"] = prices[last_idx]

    first_amount = amounts[first_idx]
    last_amount = amounts[last_idx]

    ts_first = timestamps[first_idx]
    ts_last = timestamps[last_idx]

    # 向量化计算派生字段
    fp = agg_result["first_price"].values
    lp = agg_result["last_price"].values
    tv = agg_result["total_volume"].values
    tc = agg_result["total_cost"].values
    ls = agg_result["levels_swept"].values.astype(float)

    agg_result["vwap"] = tc / (tv + 1e-15)
    price_impact = np.abs(lp - fp)
    agg_result["price_impact"] = price_impact
    mid = (fp + lp) / 2 + 1e-15
    agg_result["price_impact_pct"] = price_impact / mid * 100
    agg_result["impact_per_volume"] = price_impact / (tv + 1e-15)
    agg_result["impact_per_level"] = np.where(ls > 0, price_impact / ls, 0)

    # duration
    ts_f = pd.to_datetime(ts_first)
    ts_l = pd.to_datetime(ts_last)
    agg_result["duration_ms"] = (ts_l - ts_f).total_seconds() * 1000

    # volume taper
    taper = np.where(ls > 1, last_amount / (first_amount + 1e-15), 1.0)
    agg_result["volume_taper"] = taper

    # rename fills columns
    agg_result = agg_result.rename(columns={
        "fills_mean": "fills_per_level_mean",
        "fills_std": "fills_per_level_std",
    })

    # Clean up temp columns
    df.drop(columns=["_cost", "_nfills"], inplace=True, errors="ignore")

    taker_orders = agg_result.reset_index()
    taker_orders["timestamp"] = pd.to_datetime(taker_orders["timestamp"])
    return taker_orders.sort_values("timestamp").reset_index(drop=True)


# ─────────────────────────────────────────────────────────
# Taker Order 时序指标
# ─────────────────────────────────────────────────────────

def avg_levels_swept(
    taker_orders: pd.DataFrame,
    resample_freq: str = "5s",
    window: int = 60,
) -> pd.Series:
    """
    滚动平均 levels_swept（吃穿档数）。

    突升 → 流动性正在被扫空 → 可能的趋势加速或反转前兆。
    """
    ts = taker_orders.set_index("timestamp")["levels_swept"]
    ts = ts[~ts.index.duplicated(keep="last")]
    resampled = ts.resample(resample_freq).mean()
    result = resampled.rolling(window, min_periods=5).mean()
    result.name = "avg_levels_swept"
    return result


def large_taker_ratio(
    taker_orders: pd.DataFrame,
    resample_freq: str = "5s",
    window: int = 60,
    levels_threshold: int = 3,
) -> pd.Series:
    """
    激进 taker 单占比：levels_swept >= threshold 的 taker volume / total volume。

    高 → 信息交易者活跃 → 方向性风险高。
    """
    to = taker_orders.copy()
    to["is_large"] = (to["levels_swept"] >= levels_threshold).astype(float)
    to = to.set_index("timestamp")

    large_vol = (to["total_volume"] * to["is_large"]).resample(resample_freq).sum()
    total_vol = to["total_volume"].resample(resample_freq).sum() + 1e-10

    result = (large_vol / total_vol).rolling(window, min_periods=5).mean()
    result.name = "large_taker_ratio"
    return result


def taker_imbalance(
    taker_orders: pd.DataFrame,
    resample_freq: str = "5s",
    window: int = 60,
) -> pd.Series:
    """
    Taker order 层面的买卖不平衡。
    与 trade imbalance 不同，这里每个 taker order 只算一次。
    """
    to = taker_orders.copy()
    to["signed_vol"] = to["total_volume"] * to["side"].map({"buy": 1, "sell": -1})
    to = to.set_index("timestamp")

    buy = to.loc[to["side"] == "buy", "total_volume"].resample(resample_freq).sum().fillna(0)
    sell = to.loc[to["side"] == "sell", "total_volume"].resample(resample_freq).sum().fillna(0)
    total = buy + sell + 1e-10

    result = ((buy - sell) / total).rolling(window, min_periods=5).mean()
    result.name = "taker_imbalance"
    return result


def sweep_depth_autocorr(
    taker_orders: pd.DataFrame,
    resample_freq: str = "5s",
    window: int = 60,
    lag: int = 1,
) -> pd.Series:
    """
    levels_swept 的自相关。

    高 → 连续多笔 taker 都在扫盘 → 趋势加速
    突降 → 扫盘停止 → 趋势衰竭
    """
    ts = taker_orders.set_index("timestamp")["levels_swept"]
    ts = ts[~ts.index.duplicated(keep="last")]
    resampled = ts.resample(resample_freq).mean().fillna(1)

    lagged = resampled.shift(lag)
    result = resampled.rolling(window, min_periods=10).corr(lagged)
    result.name = "sweep_autocorr"
    return result


def impact_efficiency(
    taker_orders: pd.DataFrame,
    resample_freq: str = "10s",
    window: int = 30,
) -> pd.Series:
    """
    全局 impact 效率：Σ price_impact / Σ total_volume（滚动窗口）。
    即 Kyle's Lambda 的 taker-order 版本。

    高 → 小量就能推动价格 → 流动性薄 → 反转策略有利。
    """
    to = taker_orders.set_index("timestamp")
    impact_sum = to["price_impact"].resample(resample_freq).sum()
    vol_sum = to["total_volume"].resample(resample_freq).sum() + 1e-10

    result = (impact_sum / vol_sum).rolling(window, min_periods=5).mean()
    result.name = "impact_efficiency"
    return result


def taker_arrival_rate(
    taker_orders: pd.DataFrame,
    resample_freq: str = "5s",
    window: int = 60,
) -> pd.Series:
    """
    Taker order 到达率（每时间桶内的 taker order 数）。

    突增 → 市场活跃度升高
    """
    ts = taker_orders.set_index("timestamp")["levels_swept"]
    ts = ts[~ts.index.duplicated(keep="last")]
    count = ts.resample(resample_freq).count()
    result = count.rolling(window, min_periods=5).mean()
    result.name = "taker_arrival_rate"
    return result


def taker_size_skewness(
    taker_orders: pd.DataFrame,
    resample_freq: str = "10s",
    window: int = 60,
) -> pd.Series:
    """
    Taker order size 的滚动偏度。

    高偏度 → 少量超大 taker 单 → 可能的信息交易者。
    """
    ts = taker_orders.set_index("timestamp")["total_volume"]
    ts = ts[~ts.index.duplicated(keep="last")]
    resampled = ts.resample(resample_freq).sum().fillna(0)
    result = resampled.rolling(window, min_periods=20).skew()
    result.name = "taker_size_skewness"
    return result


def iceberg_score(
    taker_orders: pd.DataFrame,
    resample_freq: str = "30s",
    window: int = 20,
    fills_threshold: float = 5.0,
) -> pd.Series:
    """
    冰山单检测分数：同一价格档位反复出现高 fill count 的频率。

    冰山单 = 在同一价格挂了很多单，被逐批消耗。
    高分数 → 有大型隐藏挂单 → 该价格附近有强支撑/阻力。
    """
    to = taker_orders.copy()
    to["is_iceberg"] = (to["fills_per_level_mean"] >= fills_threshold).astype(float)
    to = to.set_index("timestamp")
    iceberg_ratio = to["is_iceberg"].resample(resample_freq).mean().fillna(0)
    result = iceberg_ratio.rolling(window, min_periods=5).mean()
    result.name = "iceberg_score"
    return result
