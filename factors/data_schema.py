"""
Input data schemas and primitive computation for factor research.

Primitives are atomic per-bar quantities computed from raw data.
They are NOT factors themselves — factors are formed by applying
TS operators to primitives (handled by expressions.py / scanner.py).

Two data sources:
1. Perpetual aggTrades (from Binance FAPI)
2. Orderbook snapshots (from LakeAPI or similar)

Primitive categories:
  A) OHLCV & basic:        close, volume, return_1, ...
  B) Volume decomposition:  buy_volume, sell_volume, trade_imbalance, ...
  C) Trade microstructure:  avg_trade_size, n_trades, fills_per_agg, ...
  D) Price impact:          price_impact, small_vol_pi_ratio, ...
  E) Volume distribution:   volume_high_ratio, vwap_deviation, ...
  F) Orderbook (when avail): spread, depth_imbalance, book_skew, ...
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════
# Schema Definitions
# ══════════════════════════════════════════════════════════

AGG_TRADES_SCHEMA = {
    "timestamp": "datetime64[ns, UTC]",
    "price": "float64",
    "amount": "float64",
    "side": "str",         # "buy" or "sell"
    "cost": "float64",     # price * amount
    "agg_trade_id": "int64",
    "first_trade_id": "int64",
    "last_trade_id": "int64",
    "n_trades_in_agg": "int64",
}

ORDERBOOK_SCHEMA = {
    "timestamp": "datetime64[ns, UTC]",
    # Bid levels: bid_1_price, bid_1_size, ..., bid_N_price, bid_N_size
    # Ask levels: ask_1_price, ask_1_size, ..., ask_N_price, ask_N_size
    "mid_price": "float64",
    "spread": "float64",
    "bid_depth_5": "float64",
    "ask_depth_5": "float64",
}


# ══════════════════════════════════════════════════════════
# Trade Primitives
# ══════════════════════════════════════════════════════════

def compute_trade_primitives(trades: pd.DataFrame, freq: str = "1s") -> pd.DataFrame:
    """
    Compute ALL primitive fields from aggTrades at target frequency.

    Every column in the output is a primitive — an atomic per-bar quantity.
    Factors are formed by applying TS operators (ts_mean, delta, ts_rank, ...)
    to these columns via the expression engine.
    """
    ts = trades.set_index("timestamp") if "timestamp" in trades.columns else trades
    ts = ts[~ts.index.duplicated(keep="last")]

    # ── A) OHLCV basics ──
    ohlcv = ts["price"].resample(freq).ohlc()
    ohlcv["volume"] = ts["amount"].resample(freq).sum()
    ohlcv["cost"] = ts["cost"].resample(freq).sum()
    ohlcv = ohlcv.dropna(subset=["open"]).ffill().fillna(0)

    close = ohlcv["close"]
    ohlcv["return_1"] = close.pct_change() * 10000  # bps
    ohlcv["log_return"] = np.log(close / close.shift(1))
    ohlcv["price_range"] = (ohlcv["high"] - ohlcv["low"]) / close.clip(lower=1e-10) * 10000

    # ── B) Volume decomposition ──
    buy_mask = ts["side"] == "buy"
    ohlcv["buy_volume"] = ts.loc[buy_mask, "amount"].resample(freq).sum().reindex(ohlcv.index).fillna(0)
    ohlcv["sell_volume"] = ts.loc[~buy_mask, "amount"].resample(freq).sum().reindex(ohlcv.index).fillna(0)

    total_vol = ohlcv["buy_volume"] + ohlcv["sell_volume"]
    ohlcv["trade_imbalance"] = (ohlcv["buy_volume"] - ohlcv["sell_volume"]) / total_vol.clip(lower=1e-10)
    ohlcv["abs_imbalance"] = ohlcv["trade_imbalance"].abs()
    ohlcv["buy_volume_pct"] = ohlcv["buy_volume"] / total_vol.clip(lower=1e-10)

    # Signed (net) volume: buy_vol - sell_vol — key for Kyle's lambda regression
    ohlcv["signed_volume"] = ohlcv["buy_volume"] - ohlcv["sell_volume"]

    # ── C) Trade microstructure ──
    ohlcv["n_agg_trades"] = ts["agg_trade_id"].resample(freq).count() if "agg_trade_id" in ts.columns else 0
    if "n_trades_in_agg" in ts.columns:
        ohlcv["n_trades"] = ts["n_trades_in_agg"].resample(freq).sum()
    else:
        ohlcv["n_trades"] = ohlcv["n_agg_trades"]

    ohlcv["avg_trade_size"] = ohlcv["volume"] / ohlcv["n_trades"].clip(lower=1)
    ohlcv["fills_per_agg"] = ohlcv["n_trades"] / ohlcv["n_agg_trades"].clip(lower=1)

    # ── D) Price impact primitives ──
    # PI = |return| / volume — the atomic building block
    ohlcv["price_impact"] = ohlcv["return_1"].abs() / ohlcv["volume"].clip(lower=1e-10)

    # Small-volume PI ratio: compare PI of below-median-volume bars vs above
    # This is a rolling primitive (needs a fixed short window for the median split)
    _pi = ohlcv["price_impact"]
    _vol = ohlcv["volume"]
    _vol_median = _vol.rolling(20, min_periods=1).median()
    _is_small = _vol < _vol_median
    _small_pi = (_pi * _is_small.astype(float)).rolling(20, min_periods=1).mean()
    _large_pi = (_pi * (~_is_small).astype(float)).rolling(20, min_periods=1).mean()
    ohlcv["small_vol_pi_ratio"] = _small_pi / _large_pi.clip(lower=1e-12)

    # Signed price impact: return / volume (preserves direction)
    ohlcv["signed_price_impact"] = ohlcv["return_1"] / ohlcv["volume"].clip(lower=1e-10)

    # ── E) Volume distribution primitives ──
    # VWAP
    ohlcv["vwap"] = (ohlcv["cost"] / ohlcv["volume"]).where(ohlcv["volume"] > 0, ohlcv["close"])
    # VWAP deviation from close (bps) — measures where volume concentrates vs close
    ohlcv["vwap_deviation"] = (ohlcv["vwap"] - ohlcv["close"]) / ohlcv["close"].clip(lower=1e-10) * 10000

    # Volume-at-high ratio: 1 if close > (H+L)/2 else 0, then volume-weighted
    mid_hl = (ohlcv["high"] + ohlcv["low"]) / 2
    ohlcv["volume_at_high"] = (ohlcv["close"] > mid_hl).astype(float)

    # Bar efficiency: |close - open| / (high - low)  — directional conviction
    bar_range = (ohlcv["high"] - ohlcv["low"]).clip(lower=1e-10)
    ohlcv["bar_efficiency"] = (ohlcv["close"] - ohlcv["open"]).abs() / bar_range

    # Body direction: sign(close - open)
    ohlcv["body_direction"] = np.sign(ohlcv["close"] - ohlcv["open"])

    # Upper/lower wick ratios
    ohlcv["upper_wick_pct"] = (ohlcv["high"] - ohlcv[["open", "close"]].max(axis=1)) / bar_range
    ohlcv["lower_wick_pct"] = (ohlcv[["open", "close"]].min(axis=1) - ohlcv["low"]) / bar_range

    # Volume per range: how much volume to move 1 bps
    ohlcv["volume_per_range"] = ohlcv["volume"] / ohlcv["price_range"].clip(lower=1e-10)

    # ── F) Tick-level microstructure (computed per-bar from raw trades) ──
    # These require per-trade data within each bar, so computed during resampling
    if "timestamp" in ts.columns:
        _ts_raw = trades.set_index("timestamp") if "timestamp" in trades.columns else trades
    else:
        _ts_raw = ts

    _bar_col = _ts_raw.index.floor(pd.Timedelta(freq))

    # Distinct price count per bar (spread/fragmentation proxy)
    ohlcv["tick_n_prices"] = _ts_raw.groupby(_bar_col)["price"].nunique().reindex(ohlcv.index).fillna(1)

    # Max single trade size per bar (whale detection)
    ohlcv["tick_max_trade"] = _ts_raw.groupby(_bar_col)["amount"].max().reindex(ohlcv.index).fillna(0)

    # Trade size coefficient of variation per bar
    _bar_std = _ts_raw.groupby(_bar_col)["amount"].std().fillna(0)
    _bar_mean = _ts_raw.groupby(_bar_col)["amount"].mean().clip(lower=1e-10)
    ohlcv["tick_size_cv"] = (_bar_std / _bar_mean).reindex(ohlcv.index).fillna(0)

    # Buy/sell average size per bar
    _buy_ts = _ts_raw[_ts_raw["side"] == "buy"] if "side" in _ts_raw.columns else _ts_raw
    _sell_ts = _ts_raw[_ts_raw["side"] == "sell"] if "side" in _ts_raw.columns else _ts_raw
    ohlcv["buy_avg_size"] = _buy_ts.groupby(_buy_ts.index.floor(pd.Timedelta(freq)))["amount"].mean().reindex(ohlcv.index).fillna(0)
    ohlcv["sell_avg_size"] = _sell_ts.groupby(_sell_ts.index.floor(pd.Timedelta(freq)))["amount"].mean().reindex(ohlcv.index).fillna(0)
    _total_avg = ohlcv["buy_avg_size"] + ohlcv["sell_avg_size"]
    ohlcv["size_imbalance"] = (ohlcv["buy_avg_size"] - ohlcv["sell_avg_size"]) / _total_avg.clip(lower=1e-10)

    # Large trade percentage (trades > 2× bar mean)
    _bar_threshold = _ts_raw.groupby(_bar_col)["amount"].transform("mean") * 2
    _is_large = _ts_raw["amount"] > _bar_threshold
    _large_vol = _ts_raw.loc[_is_large].groupby(_is_large.index[_is_large].floor(pd.Timedelta(freq)))["amount"].sum()
    _total_bar_vol = _ts_raw.groupby(_bar_col)["amount"].sum().clip(lower=1e-10)
    ohlcv["large_trade_pct"] = (_large_vol / _total_bar_vol).reindex(ohlcv.index).fillna(0)

    # ── G) Spread proxies (from trades, no orderbook needed) ──
    # Roll spread: 2 * sqrt(max(0, -Cov(Δp[t], Δp[t-1])))
    _dp = ohlcv["close"].diff()
    _dp_lag = _dp.shift(1)
    _cov_dp = _dp.rolling(60, min_periods=10).cov(_dp_lag)
    ohlcv["roll_spread"] = 2 * np.sqrt(np.maximum(0, -_cov_dp))

    # Buy/sell log volume ratio
    ohlcv["buy_sell_log_ratio"] = np.log(ohlcv["buy_volume"].clip(lower=1e-10) / ohlcv["sell_volume"].clip(lower=1e-10))

    # Volume ratio: current volume / rolling mean
    _vol_ma_20 = ohlcv["volume"].rolling(20, min_periods=1).mean().clip(lower=1e-10)
    ohlcv["volume_ratio"] = ohlcv["volume"] / _vol_ma_20

    # Trade rate: trades per second (approximate)
    _freq_sec = max(1.0, pd.Timedelta(freq).total_seconds())
    ohlcv["trade_rate"] = ohlcv["n_agg_trades"] / _freq_sec

    # ── H) Trend/regime primitives ──
    # Parkinson volatility: sqrt(ln(H/L)^2 / 4ln2)
    _hl_log = np.log(ohlcv["high"] / ohlcv["low"].clip(lower=1e-10))
    ohlcv["parkinson_vol_raw"] = _hl_log ** 2 / (4 * np.log(2))

    # Efficiency ratio (multi-bar): |P[t]-P[t-w]| / Σ|ΔP| over w=20 bars
    _net_20 = (ohlcv["close"] - ohlcv["close"].shift(20)).abs()
    _path_20 = ohlcv["close"].diff().abs().rolling(20, min_periods=1).sum().clip(lower=1e-10)
    ohlcv["efficiency_ratio"] = _net_20 / _path_20

    # ── I) Small-volume trade primitives (for Kyle's lambda on small orders) ──
    # Split trades at per-trade median size, resample the small-trade subset
    # This avoids manual thresholds: "small" = below rolling bar-level median
    _vol_med_20 = ohlcv["volume"].rolling(20, min_periods=1).median()
    _is_small_bar = ohlcv["volume"] < _vol_med_20

    # Small-bar signed volume and return — regression on these gives small-trade lambda
    ohlcv["small_vol_signed_volume"] = ohlcv["signed_volume"] * _is_small_bar.astype(float)
    ohlcv["small_vol_return"] = ohlcv["return_1"] * _is_small_bar.astype(float)
    ohlcv["small_vol_volume"] = ohlcv["volume"] * _is_small_bar.astype(float)

    # ── Cleanup ──
    ohlcv = ohlcv.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    ohlcv.drop(columns=["cost"], inplace=True, errors="ignore")

    return ohlcv


# ══════════════════════════════════════════════════════════
# Orderbook Primitives
# ══════════════════════════════════════════════════════════

def compute_orderbook_primitives(book: pd.DataFrame, n_levels: int = 10) -> pd.DataFrame:
    """
    Compute primitive fields from orderbook snapshots.

    Expected input: columns like bid_1_price, bid_1_size, ask_1_price, ask_1_size, ...
    or pre-computed mid_price, spread, bid_depth_5, ask_depth_5.
    """
    df = pd.DataFrame(index=book.index)

    if "bid_1_price" in book.columns:
        bid_prices = [book[f"bid_{i}_price"] for i in range(1, n_levels + 1) if f"bid_{i}_price" in book.columns]
        bid_sizes = [book[f"bid_{i}_size"] for i in range(1, n_levels + 1) if f"bid_{i}_size" in book.columns]
        ask_prices = [book[f"ask_{i}_price"] for i in range(1, n_levels + 1) if f"ask_{i}_price" in book.columns]
        ask_sizes = [book[f"ask_{i}_size"] for i in range(1, n_levels + 1) if f"ask_{i}_size" in book.columns]

        df["mid_price"] = (bid_prices[0] + ask_prices[0]) / 2
        df["spread"] = ask_prices[0] - bid_prices[0]
        df["spread_bps"] = df["spread"] / df["mid_price"] * 10000

        for n in [1, 3, 5, 10]:
            if n <= len(bid_sizes):
                df[f"bid_depth_{n}"] = sum(bid_sizes[:n])
                df[f"ask_depth_{n}"] = sum(ask_sizes[:n])

        df["top_bid_size"] = bid_sizes[0]
        df["top_ask_size"] = ask_sizes[0]
    elif "mid_price" in book.columns:
        for col in book.columns:
            df[col] = book[col]
    else:
        return df

    # Derived orderbook primitives
    if "bid_depth_5" in df.columns:
        total = df["bid_depth_5"] + df["ask_depth_5"]
        df["depth_imbalance"] = (df["bid_depth_5"] - df["ask_depth_5"]) / total.clip(lower=1e-10)
        df["total_depth_5"] = total

    if "top_bid_size" in df.columns:
        total_top = df["top_bid_size"] + df["top_ask_size"]
        df["book_skew"] = (df["top_bid_size"] - df["top_ask_size"]) / total_top.clip(lower=1e-10)

    if "bid_depth_1" in df.columns and "bid_depth_5" in df.columns:
        df["top_concentration_bid"] = df["bid_depth_1"] / df["bid_depth_5"].clip(lower=1e-10)
        df["top_concentration_ask"] = df["ask_depth_1"] / df["ask_depth_5"].clip(lower=1e-10)

    df = df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    return df


# ══════════════════════════════════════════════════════════
# Primitive Registries
# ══════════════════════════════════════════════════════════

TRADE_PRIMITIVES = [
    # A) OHLCV
    "close", "open", "high", "low", "volume",
    "return_1", "log_return", "price_range",
    # B) Volume decomposition
    "buy_volume", "sell_volume", "signed_volume",
    "trade_imbalance", "abs_imbalance", "buy_volume_pct",
    "buy_sell_log_ratio", "volume_ratio",
    # C) Trade microstructure
    "n_trades", "n_agg_trades", "avg_trade_size", "fills_per_agg", "trade_rate",
    # D) Price impact
    "price_impact", "small_vol_pi_ratio", "signed_price_impact",
    # E) Volume distribution / bar structure
    "vwap", "vwap_deviation", "volume_at_high",
    "bar_efficiency", "body_direction",
    "upper_wick_pct", "lower_wick_pct", "volume_per_range",
    # F) Tick-level microstructure (per-bar from raw trades)
    "tick_n_prices", "tick_max_trade", "tick_size_cv",
    "buy_avg_size", "sell_avg_size", "size_imbalance",
    "large_trade_pct",
    # G) Spread proxies
    "roll_spread",
    # H) Trend/regime
    "parkinson_vol_raw", "efficiency_ratio",
    # I) Small-volume trade (for small-trade Kyle's lambda regression)
    "small_vol_signed_volume", "small_vol_return", "small_vol_volume",
]

ORDERBOOK_PRIMITIVES = [
    "mid_price", "spread", "spread_bps",
    "bid_depth_1", "ask_depth_1",
    "bid_depth_5", "ask_depth_5",
    "total_depth_5",
    "depth_imbalance", "book_skew",
    "top_bid_size", "top_ask_size",
    "top_concentration_bid", "top_concentration_ask",
]
