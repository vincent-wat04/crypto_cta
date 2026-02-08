"""
Trades 微观结构特征（纯粹从 aggTrades 推导，不需要 orderbook）。

利用 aggTrades 的 (f, l, q, p, m) 字段构造：
1. 档位平均成交量：qty/(l-f+1) → 每笔 fill 的平均 size → 按 price bucket 聚合
2. trade side autocorrelation：buy/sell 序列的自相关
3. 成交密度（trades per bar）
4. VWAP 偏离度
5. 价格档位集中度

所有函数接受 1-min (或其他频率) resampled 的 trades DataFrame，
返回与 OHLCV bar 对齐的特征 Series/DataFrame。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────
# 1. 档位平均 order size：qty/(l-f+1)
# ─────────────────────────────────────────────────────────

def avg_fill_size(trades: pd.DataFrame, freq: str = "1min") -> pd.Series:
    """
    每根 bar 内 aggTrades 的平均 fill size = qty / (last_trade_id - first_trade_id + 1)。
    衡量单笔撮合的平均大小：大 → 流动性好，小 → 被拆成多笔小成交。
    """
    df = trades.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["fill_size"] = df["amount"] / df["n_trades_in_agg"].clip(lower=1)
    ts = df.set_index("timestamp").sort_index()
    result = ts["fill_size"].resample(freq).mean()
    result.name = "avg_fill_size"
    return result


def level_volume_profile(
    trades: pd.DataFrame,
    freq: str = "1min",
    n_levels: int = 5,
) -> pd.DataFrame:
    """
    按价格档位聚合成交量。
    
    在每根 bar 内，将价格分为 n_levels 个等距 bucket，
    计算每个 bucket 的总成交量、平均 fill size。
    
    返回: 每个 bar 的 top-N level 特征。
    """
    df = trades.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["fill_size"] = df["amount"] / df["n_trades_in_agg"].clip(lower=1)
    ts = df.set_index("timestamp").sort_index()

    results = []
    for bar_start, bar_df in ts.resample(freq):
        if bar_df.empty:
            results.append({"timestamp": bar_start})
            continue

        row = {"timestamp": bar_start}
        prices = bar_df["price"].values
        p_min, p_max = prices.min(), prices.max()
        p_range = p_max - p_min

        if p_range < 1e-10:
            # 所有成交在同一价格
            row["level_vol_concentration"] = 1.0
            row["level_top1_vol_share"] = 1.0
            row["level_avg_fill_size_top1"] = bar_df["fill_size"].mean()
            row["level_n_active"] = 1
        else:
            # 按价格分 bucket
            bins = np.linspace(p_min - 1e-10, p_max + 1e-10, n_levels + 1)
            bar_df = bar_df.copy()
            bar_df["level"] = np.digitize(prices, bins) - 1
            bar_df["level"] = bar_df["level"].clip(0, n_levels - 1)

            level_stats = bar_df.groupby("level").agg(
                vol=("amount", "sum"),
                avg_fs=("fill_size", "mean"),
                n_trades=("amount", "count"),
            )

            total_vol = level_stats["vol"].sum()
            if total_vol > 0:
                vol_shares = level_stats["vol"] / total_vol
                row["level_vol_concentration"] = (vol_shares ** 2).sum()  # HHI
                row["level_top1_vol_share"] = vol_shares.max()
                top_level = level_stats["vol"].idxmax()
                row["level_avg_fill_size_top1"] = level_stats.loc[top_level, "avg_fs"]
            else:
                row["level_vol_concentration"] = 0
                row["level_top1_vol_share"] = 0
                row["level_avg_fill_size_top1"] = 0

            row["level_n_active"] = len(level_stats)

        results.append(row)

    out = pd.DataFrame(results).set_index("timestamp")
    return out


# ─────────────────────────────────────────────────────────
# 2. Trade side autocorrelation
# ─────────────────────────────────────────────────────────

def trade_side_autocorrelation(
    trades: pd.DataFrame,
    freq: str = "1min",
    window: int = 20,
    lag: int = 1,
) -> pd.DataFrame:
    """
    Buy/Sell 各自的成交量自相关 + 交叉相关。
    
    返回:
        buy_autocorr:  buy volume 的 lag-N 自相关
        sell_autocorr: sell volume 的 lag-N 自相关
        cross_corr:    buy vs sell volume 的同期相关
        side_run_length: 连续同侧成交的平均长度
    """
    df = trades.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    ts = df.set_index("timestamp").sort_index()

    buy_vol = ts.loc[ts["side"] == "buy", "amount"].resample(freq).sum().fillna(0)
    sell_vol = ts.loc[ts["side"] == "sell", "amount"].resample(freq).sum().fillna(0)

    # 统一 index
    idx = buy_vol.index.union(sell_vol.index)
    buy_vol = buy_vol.reindex(idx).fillna(0)
    sell_vol = sell_vol.reindex(idx).fillna(0)

    result = pd.DataFrame(index=idx)
    result["buy_autocorr"] = buy_vol.rolling(window, min_periods=5).corr(buy_vol.shift(lag))
    result["sell_autocorr"] = sell_vol.rolling(window, min_periods=5).corr(sell_vol.shift(lag))
    result["cross_corr"] = buy_vol.rolling(window, min_periods=5).corr(sell_vol)

    # Side run length: 连续同侧的平均长度
    sides = ts["side"].map({"buy": 1, "sell": -1}).fillna(0)
    side_changes = (sides != sides.shift(1)).astype(int)
    side_run_cumsum = side_changes.cumsum()
    run_lengths = side_changes.groupby(side_run_cumsum).transform("count")
    result["side_run_length"] = run_lengths.resample(freq).mean().reindex(idx).fillna(1)

    return result


# ─────────────────────────────────────────────────────────
# 3. 成交密度与 VWAP 偏离
# ─────────────────────────────────────────────────────────

def trade_intensity(trades: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """
    成交密度特征：
        n_agg_trades: 每 bar aggTrades 数量
        n_fills:      每 bar 原始成交笔数 = sum(l-f+1)
        avg_n_fills:  每笔 aggTrade 平均包含的 fill 数
        trade_rate:   成交频率 (aggTrades/second 估计)
    """
    df = trades.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    ts = df.set_index("timestamp").sort_index()

    result = pd.DataFrame()
    result["n_agg_trades"] = ts["agg_trade_id"].resample(freq).count()
    result["n_fills"] = ts["n_trades_in_agg"].resample(freq).sum()
    n_agg = result["n_agg_trades"].replace(0, np.nan)
    result["avg_n_fills"] = result["n_fills"] / n_agg
    # 假设每根 bar 60 秒
    result["trade_rate"] = result["n_agg_trades"] / 60.0

    return result.fillna(0)


def vwap_deviation(trades: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """
    VWAP 偏离度：
        vwap: bar 内量加权均价
        vwap_close_dev: (close - vwap) / close * 100 (%)
        vwap_trend: vwap 的 rolling 变化方向
    """
    df = trades.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    ts = df.set_index("timestamp").sort_index()

    cost = ts["cost"].resample(freq).sum()
    vol = ts["amount"].resample(freq).sum().replace(0, np.nan)
    close = ts["price"].resample(freq).last()

    result = pd.DataFrame(index=cost.index)
    result["vwap"] = cost / vol
    result["close"] = close
    result["vwap_close_dev"] = (close - result["vwap"]) / (close + 1e-10) * 100
    result["vwap_trend"] = result["vwap"].diff() / result["vwap"].shift(1).replace(0, np.nan) * 100

    return result.drop(columns=["close"], errors="ignore").fillna(0)


# ─────────────────────────────────────────────────────────
# 4. 综合特征计算
# ─────────────────────────────────────────────────────────

def compute_trade_microstructure(
    trades: pd.DataFrame,
    freq: str = "1min",
    rolling_windows: list = None,
) -> pd.DataFrame:
    """
    一次性计算所有 trade 微观结构特征。
    
    Args:
        trades: aggTrades DataFrame
        freq: resample 频率
        rolling_windows: 滚动窗口列表（默认 [5, 10, 20, 60]）
        
    Returns:
        DataFrame，index=timestamp，每列一个特征
    """
    if rolling_windows is None:
        rolling_windows = [5, 10, 20, 60]

    print("  Computing avg_fill_size...")
    fs = avg_fill_size(trades, freq)
    
    print("  Computing level_volume_profile...")
    lvp = level_volume_profile(trades, freq)
    
    print("  Computing trade_side_autocorrelation...")
    tsa = trade_side_autocorrelation(trades, freq)
    
    print("  Computing trade_intensity...")
    ti = trade_intensity(trades, freq)
    
    print("  Computing vwap_deviation...")
    vd = vwap_deviation(trades, freq)

    # 汇总基础特征
    base = pd.DataFrame(index=fs.index)
    base["avg_fill_size"] = fs
    for col in lvp.columns:
        base[col] = lvp[col].reindex(base.index)
    for col in tsa.columns:
        base[col] = tsa[col].reindex(base.index)
    for col in ti.columns:
        base[col] = ti[col].reindex(base.index)
    for col in vd.columns:
        base[col] = vd[col].reindex(base.index)

    # 添加滚动统计
    print("  Computing rolling features...")
    rolling_cols = [
        "avg_fill_size", "level_vol_concentration",
        "buy_autocorr", "sell_autocorr", "cross_corr",
        "side_run_length", "n_agg_trades", "trade_rate",
        "vwap_close_dev",
    ]
    
    for w in rolling_windows:
        for col in rolling_cols:
            if col in base.columns:
                base[f"{col}_ma{w}"] = base[col].rolling(w, min_periods=2).mean()
                base[f"{col}_std{w}"] = base[col].rolling(w, min_periods=2).std()

    return base.fillna(0)
