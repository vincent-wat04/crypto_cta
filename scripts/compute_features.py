#!/usr/bin/env python3
"""
特征计算 Pipeline：加载 trades → 计算全部特征 → 按天存储 parquet。

存储格式: data/features/{symbol}/bar_features/{date}.parquet
每个 parquet 包含一天的所有 bar-level 特征，index=timestamp。

Usage:
    # 计算单日
    python scripts/compute_features.py --symbol BTC/USDT --start-date 2022-10-01 --end-date 2022-10-02
    
    # 计算多天（自动分天存储）
    python scripts/compute_features.py --symbol BTC/USDT --start-date 2022-10-01 --end-date 2022-10-04
    
    # 自定义频率
    python scripts/compute_features.py --symbol BTC/USDT --start-date 2022-10-01 --end-date 2022-10-02 --freq 5min
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utilities.binance_loader import load_agg_trades, resample_trades_to_ohlcv
from utilities.paths import DataPaths
from indicators.base.taker_flow import group_taker_orders
from indicators.base.trade_microstructure import compute_trade_microstructure
from indicators.base.price_impact import tick_price_impact, volume_weighted_impact
from indicators.base.spread import trade_diff_spread, roll_spread
from indicators.base.order_flow import trade_imbalance, flow_toxicity
from indicators.base.volume_profile import (
    taker_volume_corr, taker_volume_autocorr, taker_volume_skewness,
    buy_sell_volume_ratio, large_trade_ratio,
)
from indicators.base.taker_flow import (
    avg_levels_swept, taker_imbalance as taker_imb_fn,
    sweep_depth_autocorr, impact_efficiency, taker_arrival_rate,
)
from indicators.regime.volatility_regime import (
    realized_volatility, parkinson_volatility, garch_like_vol,
)
from indicators.regime.trend_strength import (
    adx_indicator, efficiency_ratio,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def compute_ohlcv_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """从 OHLCV 计算 bar-level 基础特征。"""
    f = pd.DataFrame(index=ohlcv.index)

    c = ohlcv["close"]
    o = ohlcv["open"]
    h = ohlcv["high"]
    lo = ohlcv["low"]
    v = ohlcv["volume"]

    # Returns
    f["return_1"] = c.pct_change()
    f["log_return"] = np.log(c / c.shift(1))

    # Bar structure
    body = (c - o).abs()
    hl = h - lo
    f["bar_range"] = hl / (c + 1e-10) * 100
    f["body_ratio"] = body / (hl + 1e-10)
    f["upper_wick"] = (h - c.clip(lower=o).clip(upper=h)) / (hl + 1e-10)
    f["lower_wick"] = (c.clip(upper=o).clip(lower=lo) - lo) / (hl + 1e-10)

    # Volume
    f["volume"] = v
    f["buy_volume"] = ohlcv.get("buy_volume", 0)
    f["sell_volume"] = ohlcv.get("sell_volume", 0)
    f["volume_imbalance"] = (f["buy_volume"] - f["sell_volume"]) / (v + 1e-10)
    f["n_trades"] = ohlcv.get("n_trades", 0)

    # Rolling features
    for w in [5, 10, 20, 60]:
        f[f"return_ma{w}"] = f["return_1"].rolling(w, min_periods=2).mean()
        f[f"return_std{w}"] = f["return_1"].rolling(w, min_periods=2).std()
        f[f"volume_ma{w}"] = f["volume"].rolling(w, min_periods=2).mean()
        f[f"volume_ratio_{w}"] = f["volume"] / (f[f"volume_ma{w}"] + 1e-10)
        f[f"bar_range_ma{w}"] = f["bar_range"].rolling(w, min_periods=2).mean()
        f[f"imbalance_ma{w}"] = f["volume_imbalance"].rolling(w, min_periods=2).mean()

    return f


def compute_regime_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """计算 market regime 特征。"""
    f = pd.DataFrame(index=ohlcv.index)

    f["realized_vol"] = realized_volatility(ohlcv["close"], window=60)
    f["parkinson_vol"] = parkinson_volatility(ohlcv, window=60)
    f["garch_vol"] = garch_like_vol(ohlcv["close"])

    adx_df = adx_indicator(ohlcv, period=14)
    f["adx"] = adx_df["adx"]
    f["di_plus"] = adx_df["di_plus"]
    f["di_minus"] = adx_df["di_minus"]
    f["di_spread"] = adx_df["di_plus"] - adx_df["di_minus"]

    f["eff_ratio"] = efficiency_ratio(ohlcv["close"], window=20)

    ema20 = ohlcv["close"].ewm(span=20, adjust=False).mean()
    ema60 = ohlcv["close"].ewm(span=60, adjust=False).mean()
    f["ema_slope_20"] = (ema20 - ema20.shift(5)) / (ema20.shift(5) + 1e-10) * 100
    f["ema_cross"] = (ema20 - ema60) / (ema60 + 1e-10) * 100

    return f


def compute_tick_features(trades: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """从逐笔 trades 计算 tick-level 特征（resample 到 bar 级别）。"""
    f = pd.DataFrame()

    logger.info("  tick_price_impact...")
    impact_5s = tick_price_impact(trades, window_ms=5000)
    impact_10s = tick_price_impact(trades, window_ms=10000)
    vw_impact = volume_weighted_impact(trades, window_ms=10000)

    logger.info("  spread...")
    spread = trade_diff_spread(trades, window_trades=100)
    r_spread = roll_spread(trades, window_trades=200)

    logger.info("  order_flow...")
    imbal = trade_imbalance(trades, window_trades=100)
    tox = flow_toxicity(trades, window_trades=200)

    logger.info("  volume_profile...")
    vol_corr = taker_volume_corr(trades, resample_freq="5s", window=60)
    buy_ac = taker_volume_autocorr(trades, resample_freq="5s", window=60, side="buy")
    sell_ac = taker_volume_autocorr(trades, resample_freq="5s", window=60, side="sell")
    vol_skew = taker_volume_skewness(trades, resample_freq="5s", window=120)
    bs_ratio = buy_sell_volume_ratio(trades, resample_freq="5s", window=60)
    lg_ratio = large_trade_ratio(trades, resample_freq="5s", window=60)

    # Resample all to bar freq
    series_map = {
        "tick_impact_5s": impact_5s,
        "tick_impact_10s": impact_10s,
        "vw_impact": vw_impact,
        "trade_diff_spread": spread,
        "roll_spread": r_spread,
        "trade_imbalance": imbal,
        "flow_toxicity": tox,
        "taker_vol_corr": vol_corr,
        "buy_vol_autocorr": buy_ac,
        "sell_vol_autocorr": sell_ac,
        "taker_vol_skewness": vol_skew,
        "bs_vol_ratio": bs_ratio,
        "large_trade_ratio": lg_ratio,
    }

    for name, s in series_map.items():
        resampled = s.resample(freq).last()
        f[name] = resampled

    return f


def compute_taker_features(
    taker_orders: pd.DataFrame,
    freq: str = "1min",
) -> pd.DataFrame:
    """从 grouped taker orders 计算特征。"""
    f = pd.DataFrame()

    logger.info("  taker_flow features...")
    lvls = avg_levels_swept(taker_orders, resample_freq="5s", window=60)
    imb = taker_imb_fn(taker_orders, resample_freq="5s", window=60)
    sweep_ac = sweep_depth_autocorr(taker_orders, resample_freq="5s", window=60)
    imp_eff = impact_efficiency(taker_orders, resample_freq="10s", window=30)
    arrival = taker_arrival_rate(taker_orders, resample_freq="5s", window=60)

    series_map = {
        "taker_levels_swept": lvls,
        "taker_imbalance": imb,
        "sweep_autocorr": sweep_ac,
        "impact_efficiency": imp_eff,
        "taker_arrival_rate": arrival,
    }

    for name, s in series_map.items():
        f[name] = s.resample(freq).last()

    return f


def compute_all_features_for_day(
    symbol: str,
    dt: date,
    freq: str = "1min",
) -> pd.DataFrame:
    """
    计算单日全部特征并返回 DataFrame。
    
    Pipeline:
    1. 加载 trades (from cache)
    2. Resample to OHLCV
    3. Group taker orders
    4. Compute all feature categories
    5. Merge into single DataFrame
    """
    logger.info(f"═══ Computing features: {symbol} {dt} ═══")

    # 加载 trades
    logger.info("[1/5] Loading trades...")
    trades = load_agg_trades(symbol, dt, dt + timedelta(days=1))
    if trades.empty:
        logger.warning(f"  No trades for {dt}")
        return pd.DataFrame()
    logger.info(f"  {len(trades)} aggTrades")

    # OHLCV
    logger.info("[2/5] Building OHLCV...")
    ohlcv = resample_trades_to_ohlcv(trades, freq)
    logger.info(f"  {len(ohlcv)} bars")

    # Taker orders
    logger.info("[3/5] Grouping taker orders...")
    taker_orders = group_taker_orders(trades)
    logger.info(f"  {len(taker_orders)} taker orders")

    # Features
    logger.info("[4/5] Computing features...")

    logger.info("  OHLCV features...")
    f_ohlcv = compute_ohlcv_features(ohlcv)

    logger.info("  Regime features...")
    f_regime = compute_regime_features(ohlcv)

    logger.info("  Tick features...")
    f_tick = compute_tick_features(trades, freq)

    logger.info("  Taker features...")
    f_taker = compute_taker_features(taker_orders, freq)

    logger.info("  Trade microstructure features...")
    f_micro = compute_trade_microstructure(trades, freq)

    # Merge
    logger.info("[5/5] Merging features...")
    # Prefix each category for clarity
    f_regime = f_regime.add_prefix("regime_")
    f_tick = f_tick.add_prefix("tick_")
    f_taker = f_taker.add_prefix("taker_")
    f_micro = f_micro.add_prefix("micro_")

    # 使用 pd.concat 一次性合并（避免 fragmentation）
    all_parts = [f_ohlcv, f_regime, f_tick, f_taker, f_micro]
    reindexed = [p.reindex(f_ohlcv.index) for p in all_parts]
    result = pd.concat(reindexed, axis=1)
    result = result.fillna(0).replace([np.inf, -np.inf], 0).copy()
    logger.info(f"  Final: {result.shape[0]} bars × {result.shape[1]} features")

    return result


def main():
    parser = argparse.ArgumentParser(description="Compute and save bar-level features")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--freq", default="1min")
    args = parser.parse_args()

    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    current = start
    while current < end:
        # Check if already computed
        save_path = DataPaths.features(args.symbol, "bar_features", current)
        if save_path.exists():
            logger.info(f"[Skip] Features exist: {save_path}")
            current += timedelta(days=1)
            continue

        features = compute_all_features_for_day(args.symbol, current, args.freq)
        if features.empty:
            current += timedelta(days=1)
            continue

        features.to_parquet(save_path)
        logger.info(f"  Saved: {save_path}")
        current += timedelta(days=1)

    logger.info("Done.")


if __name__ == "__main__":
    main()
