"""
成交量结构指标（利用 aggTrades 的 taker 聚合信息）。

原理：
  - taker_volume_corr: buy/sell volume 按 taker order 聚合后的 rolling corr
  - taker_volume_autocorr: 单侧 taker volume 的自相关
  - taker_volume_skewness: taker 成交量分布的偏度
  - buy_sell_volume_ratio: 买卖量比
  - large_trade_ratio: 大单占比

机制：
  corr 为正 → 买卖同步放大 → 做市商撤单 → 流动性真空
  skewness 偏高 → 少量超大 taker 单 → 信息交易者
  autocorr 高 → 同侧连续扫货 → 方向性动量
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _prepare_taker_series(
    trades: pd.DataFrame,
    resample_freq: str = "1s",
) -> pd.DataFrame:
    """
    将逐笔 aggTrades 聚合为时间桶的 buy/sell taker volume。

    如果有 agg_trade_id 列（Binance aggTrades），
    则先按 agg_trade_id 聚合后再 resample。
    """
    ts = trades.copy()
    ts["timestamp"] = pd.to_datetime(ts["timestamp"])

    # 如果有 taker 聚合信息，按 agg_trade_id 聚合
    if "agg_trade_id" in ts.columns:
        agg = ts.groupby("agg_trade_id").agg(
            timestamp=("timestamp", "first"),
            amount=("amount", "sum"),
            cost=("cost", "sum"),
            side=("side", "first"),
            n_trades=("n_trades_in_agg", "first") if "n_trades_in_agg" in ts.columns else ("amount", "count"),
        )
        ts = agg.reset_index()

    ts = ts.set_index("timestamp").sort_index()

    buy = ts.loc[ts["side"] == "buy", "amount"].resample(resample_freq).sum().fillna(0)
    sell = ts.loc[ts["side"] == "sell", "amount"].resample(resample_freq).sum().fillna(0)

    return pd.DataFrame({"buy_vol": buy, "sell_vol": sell}).fillna(0)


def taker_volume_corr(
    trades: pd.DataFrame,
    resample_freq: str = "1s",
    window: int = 60,
) -> pd.Series:
    """
    buy volume 和 sell volume 在按 taker 聚合后的滚动相关系数。

    corr > 0 → 买卖同步放大 → 可能的流动性真空
    corr < 0 → 一侧放量另一侧缩量 → 方向性信号
    """
    bv_sv = _prepare_taker_series(trades, resample_freq)
    result = bv_sv["buy_vol"].rolling(window, min_periods=10).corr(bv_sv["sell_vol"])
    result.name = "taker_vol_corr"
    return result


def taker_volume_autocorr(
    trades: pd.DataFrame,
    resample_freq: str = "1s",
    window: int = 60,
    lag: int = 1,
    side: str = "buy",
) -> pd.Series:
    """
    单侧 taker volume 的自相关系数。

    autocorr 高 → 同侧持续扫货 → 动量信号
    autocorr 突降 → 扫货停止 → 趋势可能衰竭
    """
    bv_sv = _prepare_taker_series(trades, resample_freq)
    col = "buy_vol" if side == "buy" else "sell_vol"
    series = bv_sv[col]
    lagged = series.shift(lag)
    result = series.rolling(window, min_periods=10).corr(lagged)
    result.name = f"taker_vol_autocorr_{side}"
    return result


def taker_volume_skewness(
    trades: pd.DataFrame,
    resample_freq: str = "1s",
    window: int = 120,
) -> pd.Series:
    """
    Taker trade volume 分布的滚动偏度。

    高偏度 → 少量超大 taker 单 → 可能的信息交易者
    """
    ts = trades.copy()
    ts["timestamp"] = pd.to_datetime(ts["timestamp"])

    if "agg_trade_id" in ts.columns:
        # 按 taker order 聚合
        agg = ts.groupby("agg_trade_id").agg(
            timestamp=("timestamp", "first"),
            amount=("amount", "sum"),
        ).reset_index()
        ts = agg

    ts = ts.set_index("timestamp").sort_index()
    vol = ts["amount"].resample(resample_freq).sum().fillna(0)
    result = vol.rolling(window, min_periods=20).skew()
    result.name = "taker_vol_skewness"
    return result


def buy_sell_volume_ratio(
    trades: pd.DataFrame,
    resample_freq: str = "1s",
    window: int = 60,
) -> pd.Series:
    """
    买卖成交量滚动比值：log(BuyVol / SellVol)。
    """
    bv_sv = _prepare_taker_series(trades, resample_freq)
    bv = bv_sv["buy_vol"].rolling(window, min_periods=5).sum() + 1e-10
    sv = bv_sv["sell_vol"].rolling(window, min_periods=5).sum() + 1e-10
    result = np.log(bv / sv)
    result.name = "buy_sell_vol_ratio"
    return result


def large_trade_ratio(
    trades: pd.DataFrame,
    resample_freq: str = "5s",
    window: int = 60,
    quantile_threshold: float = 0.9,
) -> pd.Series:
    """
    大单成交占总成交量比例。

    大单定义：单笔 taker volume > 历史 quantile_threshold 分位数。
    """
    ts = trades.copy()
    ts["timestamp"] = pd.to_datetime(ts["timestamp"])

    if "agg_trade_id" in ts.columns:
        agg = ts.groupby("agg_trade_id").agg(
            timestamp=("timestamp", "first"),
            amount=("amount", "sum"),
        ).reset_index()
        ts = agg

    ts = ts.set_index("timestamp").sort_index()
    vol = ts["amount"]

    # Rolling quantile
    threshold = vol.rolling(1000, min_periods=50).quantile(quantile_threshold)
    is_large = (vol >= threshold).astype(float)
    large_vol = (vol * is_large).resample(resample_freq).sum()
    total_vol = vol.resample(resample_freq).sum() + 1e-10

    result = (large_vol / total_vol).rolling(window, min_periods=5).mean()
    result.name = "large_trade_ratio"
    return result
