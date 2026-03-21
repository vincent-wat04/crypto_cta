"""
Data loader for rule-based backtesting.

Architecture
────────────
  trade_source:  "raw"  (raw aggTrades)
                 "merged" (multi-fill taker orders combined)
  bar_mode:      "time"  (fixed-interval)
                 "trade_count" (N trades per bar)
                 "volume" (target traded volume per bar)

  → 6 combos:  raw/merged × time/trade_count/volume

Two factor registries
─────────────────────
  RAW_FACTOR_REGISTRY    – needs only basic OHLCV bars
  MERGED_FACTOR_REGISTRY – needs extended bars with merge-specific columns

All factor functions receive ONLY bars (no trades parameter).

Naming convention
─────────────────
  volume  = trade quantity (previously "amount")
  value   = price × volume  (previously "cost")
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
_DATA_ROOT = Path(os.environ.get("MR_DATA_ROOT", str(ROOT / "data")))
CACHE_DIR = _DATA_ROOT / "cache"


# ═══════════════════════════════════════════════════════════
# Raw data loading (renames: amount→volume, cost→value)
# ═══════════════════════════════════════════════════════════

def load_cached_trades(
    symbol: str = "SOL/USDC",
    cache_dir: Path = CACHE_DIR,
    cache_subdir: str = "trades_perp",
) -> pd.DataFrame:
    sym_dir = cache_dir / symbol.replace("/", "_") / cache_subdir
    if not sym_dir.exists():
        raise FileNotFoundError(f"Cache directory not found: {sym_dir}")

    files = sorted(f for f in sym_dir.glob("*.parquet") if not f.name.startswith("._"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {sym_dir}")

    logger.info("Loading %d cached files from %s", len(files), sym_dir)
    dfs = [pd.read_parquet(f) for f in files]
    trades = pd.concat(dfs, ignore_index=True)
    trades["timestamp"] = pd.to_datetime(trades["timestamp"])
    trades.sort_values("timestamp", inplace=True)
    trades.reset_index(drop=True, inplace=True)

    # Unified column names
    trades.rename(columns={"amount": "volume", "cost": "value"}, inplace=True)

    logger.info("Loaded %d trades, %s → %s",
                len(trades), trades["timestamp"].iloc[0], trades["timestamp"].iloc[-1])
    return trades


# ═══════════════════════════════════════════════════════════
# AggTrade merging
# ═══════════════════════════════════════════════════════════

def merge_aggtrades(
    trades: pd.DataFrame,
    max_gap_ms: float = 100.0,
) -> pd.DataFrame:
    """
    Merge consecutive aggTrades belonging to the same taker order.

    Continuation if ALL:
      1. Same side
      2. Consecutive trade IDs (first_trade_id == prev.last_trade_id + 1)
      3. Timestamp gap < max_gap_ms
      4. Monotonic price (buy: non-decreasing, sell: non-increasing)

    Output columns:
      timestamp, price (VWAP), volume (sum), value (sum), side,
      n_fills, n_raw_trades, first_price, last_price, price_range,
      first_trade_id, last_trade_id
    """
    if trades.empty:
        return pd.DataFrame()

    ts = pd.to_datetime(trades["timestamp"])
    side = trades["side"].values
    price = trades["price"].values
    vol = trades["volume"].values
    val = trades["value"].values if "value" in trades.columns else price * vol
    first_tid = trades["first_trade_id"].values if "first_trade_id" in trades.columns else np.arange(len(trades))
    last_tid = trades["last_trade_id"].values if "last_trade_id" in trades.columns else np.arange(len(trades))
    n_in_agg = trades["n_trades_in_agg"].values if "n_trades_in_agg" in trades.columns else np.ones(len(trades))

    n = len(trades)
    dt_ms = np.zeros(n)
    dt_ms[1:] = (ts.values[1:] - ts.values[:-1]).astype("timedelta64[ms]").astype(float)

    same_side = np.zeros(n, dtype=bool)
    same_side[1:] = side[1:] == side[:-1]

    consec_id = np.zeros(n, dtype=bool)
    consec_id[1:] = first_tid[1:] == last_tid[:-1] + 1

    price_mono = np.zeros(n, dtype=bool)
    price_diff = np.diff(price, prepend=price[0])
    for i in range(1, n):
        if side[i] == "buy":
            price_mono[i] = price_diff[i] >= 0
        else:
            price_mono[i] = price_diff[i] <= 0

    is_continuation = same_side & consec_id & (dt_ms < max_gap_ms) & price_mono
    is_continuation[0] = False
    group_ids = (~is_continuation).cumsum()

    df = pd.DataFrame({
        "group": group_ids,
        "timestamp": ts,
        "price": price,
        "volume": vol,
        "value": val,
        "side": side,
        "n_raw_trades": n_in_agg,
        "first_trade_id": first_tid,
        "last_trade_id": last_tid,
    })

    merged = df.groupby("group", sort=True).agg(
        timestamp=("timestamp", "first"),
        volume=("volume", "sum"),
        side=("side", "first"),
        value=("value", "sum"),
        n_fills=("price", "count"),
        n_raw_trades=("n_raw_trades", "sum"),
        first_price=("price", "first"),
        last_price=("price", "last"),
        high_price=("price", "max"),
        low_price=("price", "min"),
        first_trade_id=("first_trade_id", "first"),
        last_trade_id=("last_trade_id", "last"),
    )

    merged["price"] = merged["value"] / (merged["volume"] + 1e-15)
    merged["price_range"] = merged["high_price"] - merged["low_price"]
    merged.drop(columns=["high_price", "low_price"], inplace=True)
    merged.reset_index(drop=True, inplace=True)

    n_reduced = len(trades) - len(merged)
    logger.info("Merged %d aggTrades → %d taker orders (%.1f%% reduction, %d multi-fill groups)",
                len(trades), len(merged),
                n_reduced / len(trades) * 100,
                (merged["n_fills"] > 1).sum())
    return merged


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════

def _ols_beta(y: np.ndarray, x: np.ndarray) -> float:
    """OLS regression slope: cov(y, x) / var(x).  Returns 0.0 if underdetermined."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return 0.0
    xm = x.mean()
    var = ((x - xm) ** 2).sum()
    if var < 1e-15:
        return 0.0
    return float(((y - y.mean()) * (x - xm)).sum() / var)


def _safe_log(arr: np.ndarray) -> np.ndarray:
    """log(x) with floor at 1e-15 to avoid -inf."""
    return np.log(np.maximum(arr, 1e-15))


def _safe_skew(arr: np.ndarray) -> float:
    if len(arr) < 3:
        return 0.0
    return float(pd.Series(arr).skew())


def estimate_avg_bar_seconds(bars: pd.DataFrame) -> float:
    if "bar_duration_sec" in bars.columns:
        return float(bars["bar_duration_sec"].median())
    if len(bars) < 2:
        return 60.0
    diffs = bars.index.to_series().diff().dropna().dt.total_seconds()
    return float(diffs.median())


# ═══════════════════════════════════════════════════════════
# Bar builders
# ═══════════════════════════════════════════════════════════

def _build_event_bar_slices(
    event_sizes: np.ndarray,
    threshold: float,
) -> list[tuple[int, int]]:
    """
    Slice a stream of events into sequential bars.

    Each slice closes once cumulative event size reaches/exceeds `threshold`.
    The last partial slice is preserved so tail events are not dropped.
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")

    n = len(event_sizes)
    if n == 0:
        return []

    slices: list[tuple[int, int]] = []
    start = 0
    acc = 0.0

    for i, size in enumerate(event_sizes):
        acc += float(size)
        if acc >= threshold:
            slices.append((start, i + 1))
            start = i + 1
            acc = 0.0

    if start < n:
        slices.append((start, n))

    return slices


def _bar_duration_sec(ts: np.ndarray, start: int, end: int) -> float:
    return max((ts[end - 1] - ts[start]) / np.timedelta64(1, "s"), 0.001)

def build_time_bars(trades_df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """
    Basic time-based OHLCV bars.  Works with raw OR merged trades.
    Output: open, high, low, close, volume, n_trades, buy_volume, sell_volume.
    """
    ts = trades_df.copy()
    ts["timestamp"] = pd.to_datetime(ts["timestamp"])
    ts = ts.set_index("timestamp").sort_index()

    bars = pd.DataFrame()
    bars["open"] = ts["price"].resample(freq).first()
    bars["high"] = ts["price"].resample(freq).max()
    bars["low"] = ts["price"].resample(freq).min()
    bars["close"] = ts["price"].resample(freq).last()
    bars["volume"] = ts["volume"].resample(freq).sum()
    bars["n_trades"] = ts["price"].resample(freq).count()

    buy = ts["side"] == "buy"
    bars["buy_volume"] = ts.loc[buy, "volume"].resample(freq).sum().reindex(bars.index).fillna(0)
    bars["sell_volume"] = ts.loc[~buy, "volume"].resample(freq).sum().reindex(bars.index).fillna(0)

    bars = bars.dropna(subset=["close"]).ffill()
    return bars


def build_trade_count_bars_raw(
    trades_df: pd.DataFrame,
    trades_per_bar: int = 200,
) -> pd.DataFrame:
    """
    Trade-count bars from raw (or merged) trades — basic OHLCV only.
    """
    n = len(trades_df)
    if n == 0:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume",
                                      "n_trades", "buy_volume", "sell_volume", "bar_duration_sec"])

    ts = pd.to_datetime(trades_df["timestamp"]).values
    price = trades_df["price"].values
    vol = trades_df["volume"].values
    side = trades_df["side"].values

    slices = _build_event_bar_slices(np.ones(n, dtype=float), trades_per_bar)
    records = []
    for s, e in slices:
        p, v, sd = price[s:e], vol[s:e], side[s:e]
        bm = sd == "buy"
        dur = _bar_duration_sec(ts, s, e)
        records.append({
            "timestamp": pd.Timestamp(ts[s]),
            "open": p[0], "high": p.max(), "low": p.min(), "close": p[-1],
            "volume": v.sum(), "n_trades": len(p),
            "buy_volume": v[bm].sum() if bm.any() else 0.0,
            "sell_volume": v[~bm].sum() if (~bm).any() else 0.0,
            "bar_duration_sec": dur,
        })
    return pd.DataFrame(records).set_index("timestamp")


def build_volume_bars_raw(
    trades_df: pd.DataFrame,
    volume_per_bar: float,
) -> pd.DataFrame:
    """
    Volume bars from raw (or merged) trades — basic OHLCV only.

    Bars close once cumulative traded volume reaches/exceeds `volume_per_bar`.
    Because the last trade is not split, realized bar volume may slightly exceed
    the target.
    """
    n = len(trades_df)
    if n == 0:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume",
                                      "n_trades", "buy_volume", "sell_volume", "bar_duration_sec"])

    ts = pd.to_datetime(trades_df["timestamp"]).values
    price = trades_df["price"].values
    vol = trades_df["volume"].values
    side = trades_df["side"].values

    slices = _build_event_bar_slices(vol, volume_per_bar)
    records = []
    for s, e in slices:
        p, v, sd = price[s:e], vol[s:e], side[s:e]
        bm = sd == "buy"
        dur = _bar_duration_sec(ts, s, e)
        records.append({
            "timestamp": pd.Timestamp(ts[s]),
            "open": p[0], "high": p.max(), "low": p.min(), "close": p[-1],
            "volume": v.sum(), "n_trades": len(p),
            "buy_volume": v[bm].sum() if bm.any() else 0.0,
            "sell_volume": v[~bm].sum() if (~bm).any() else 0.0,
            "bar_duration_sec": dur,
        })
    return pd.DataFrame(records).set_index("timestamp")


def build_time_bars_merged(merged_df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """
    Time-based bars from merged trades — extended columns.
    Includes OHLCV + merge-specific aggregates needed by MERGED_FACTOR_REGISTRY.
    """
    ts = merged_df.copy()
    ts["timestamp"] = pd.to_datetime(ts["timestamp"])
    ts = ts.set_index("timestamp").sort_index()

    # Floor to time bucket for grouping
    ts["bar_ts"] = ts.index.floor(freq)

    records = []
    for bar_ts, grp in ts.groupby("bar_ts"):
        rec = _compute_merged_bar(grp, bar_ts)
        if rec is not None:
            records.append(rec)

    if not records:
        return pd.DataFrame()
    bars = pd.DataFrame(records).set_index("timestamp")
    return bars.ffill()


def build_trade_count_bars_merged(
    merged_df: pd.DataFrame,
    trades_per_bar: int = 200,
) -> pd.DataFrame:
    """
    Trade-count bars from merged trades — extended columns.
    """
    n = len(merged_df)
    if n == 0:
        return pd.DataFrame()

    ts = pd.to_datetime(merged_df["timestamp"]).values
    slices = _build_event_bar_slices(np.ones(n, dtype=float), trades_per_bar)

    records = []
    for s, e in slices:
        grp = merged_df.iloc[s:e]
        bar_ts = pd.Timestamp(ts[s])
        rec = _compute_merged_bar(grp, bar_ts)
        if rec is not None:
            dur = _bar_duration_sec(ts, s, e)
            rec["bar_duration_sec"] = dur
            records.append(rec)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).set_index("timestamp")


def build_volume_bars_merged(
    merged_df: pd.DataFrame,
    volume_per_bar: float,
) -> pd.DataFrame:
    """
    Volume bars from merged trades — extended columns.

    Bars close once cumulative merged-trade volume reaches/exceeds
    `volume_per_bar`. Because a merged trade is not split, realized bar volume
    may slightly exceed the target.
    """
    n = len(merged_df)
    if n == 0:
        return pd.DataFrame()

    ts = pd.to_datetime(merged_df["timestamp"]).values
    vol = merged_df["volume"].values
    slices = _build_event_bar_slices(vol, volume_per_bar)

    records = []
    for s, e in slices:
        grp = merged_df.iloc[s:e]
        bar_ts = pd.Timestamp(ts[s])
        rec = _compute_merged_bar(grp, bar_ts)
        if rec is not None:
            dur = _bar_duration_sec(ts, s, e)
            rec["bar_duration_sec"] = dur
            records.append(rec)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).set_index("timestamp")


def _compute_merged_bar(grp: pd.DataFrame, bar_ts) -> dict | None:
    """Compute one bar's full metric set from a group of merged trades."""
    if len(grp) == 0:
        return None

    price = grp["price"].values
    vol = grp["volume"].values
    val = grp["value"].values if "value" in grp.columns else price * vol
    side = grp["side"].values
    n_fills = grp["n_fills"].values if "n_fills" in grp.columns else np.ones(len(grp))
    first_price = grp["first_price"].values if "first_price" in grp.columns else price
    price_range = grp["price_range"].values if "price_range" in grp.columns else np.zeros(len(grp))
    vwap_dist = price - first_price

    bm = side == "buy"
    sm = ~bm
    mf = n_fills > 1              # multi-fill mask
    mf_buy = mf & bm
    mf_sell = mf & sm

    log_vol = _safe_log(vol)

    # -- Regressions (within this bar's trades) --
    # vwap_dist = VWAP - first_price  (how far from top-of-book the order settled)
    vwap_dist_buy = price[bm] - first_price[bm]  if bm.any() else np.array([])
    vwap_dist_sell = first_price[sm] - price[sm]  if sm.any() else np.array([])

    buy_vwap_dist_reg_beta = _ols_beta(vwap_dist_buy, log_vol[bm]) if bm.sum() >= 3 else 0.0
    sell_vwap_dist_reg_beta = _ols_beta(vwap_dist_sell, log_vol[sm]) if sm.sum() >= 3 else 0.0
    buy_pr_reg_beta = _ols_beta(price_range[bm], log_vol[bm]) if bm.sum() >= 3 else 0.0
    sell_pr_reg_beta = _ols_beta(price_range[sm], log_vol[sm]) if sm.sum() >= 3 else 0.0

    # -- Multi-fill volume median ratio:  log(median_all / median_multifill) --
    # Positive → multi-fill trades are smaller than average (more "informed")
    buy_mf_med_ratio = 0.0
    if bm.any() and mf_buy.any():
        med_all = np.median(vol[bm])
        med_mf = np.median(vol[mf_buy])
        if med_mf > 1e-15 and med_all > 1e-15:
            buy_mf_med_ratio = float(np.log(med_all / med_mf))

    sell_mf_med_ratio = 0.0
    if sm.any() and mf_sell.any():
        med_all = np.median(vol[sm])
        med_mf = np.median(vol[mf_sell])
        if med_mf > 1e-15 and med_all > 1e-15:
            sell_mf_med_ratio = float(np.log(med_all / med_mf))

    # -- Off-top volume: how much extra cost beyond first_price × volume --
    # Buy: total_value - Σ(volume_i × first_price_i)  (paid MORE than top-of-book)
    # Sell: Σ(volume_i × first_price_i) - total_value  (received LESS than top-of-book)
    buy_off_top = float((val[bm] - vol[bm] * first_price[bm]).sum()) if bm.any() else 0.0
    sell_off_top = float((vol[sm] * first_price[sm] - val[sm]).sum()) if sm.any() else 0.0

    return {
        "timestamp": bar_ts,
        "open": price[0], "high": price.max(), "low": price.min(), "close": price[-1],
        "volume": vol.sum(),
        "n_trades": len(price),
        "buy_volume": float(vol[bm].sum()) if bm.any() else 0.0,
        "sell_volume": float(vol[sm].sum()) if sm.any() else 0.0,
        "buy_trade_count": int(bm.sum()),
        "sell_trade_count": int(sm.sum()),
        # vwap dist sums
        # traders 认为公允价值偏差的累积值
        "buy_vwap_dist_sum": float(vwap_dist_buy.sum()) if len(vwap_dist_buy) else 0.0,
        "sell_vwap_dist_sum": float(vwap_dist_sell.sum()) if len(vwap_dist_sell) else 0.0,
        # multi-fill counts & volumes
        "buy_multi_fill_count": int(mf_buy.sum()),
        "sell_multi_fill_count": int(mf_sell.sum()),
        "buy_multi_fill_volume": float(vol[mf_buy].sum()) if mf_buy.any() else 0.0,
        "sell_multi_fill_volume": float(vol[mf_sell].sum()) if mf_sell.any() else 0.0,
        # off-top (aggressiveness)
        "buy_off_top_volume": buy_off_top,
        "sell_off_top_volume": sell_off_top,
        # multi-fill median ratio
        "buy_mf_vol_median_ratio": buy_mf_med_ratio,
        "sell_mf_vol_median_ratio": sell_mf_med_ratio,
        # within-bar regressions
        "buy_vwap_dist_reg_beta": buy_vwap_dist_reg_beta,
        "sell_vwap_dist_reg_beta": sell_vwap_dist_reg_beta,
        "buy_pr_reg_beta": buy_pr_reg_beta,
        "sell_pr_reg_beta": sell_pr_reg_beta,
        # volume skew
        "buy_volume_skew": _safe_skew(vol[bm]) if bm.sum() >= 3 else 0.0,
        "sell_volume_skew": _safe_skew(vol[sm]) if sm.sum() >= 3 else 0.0,
        # 加权价格偏离
        "weighted_vwap_dist": (vwap_dist * vol).sum()
    }


# ═══════════════════════════════════════════════════════════
# RAW factors  (need only basic OHLCV bars)
# ═══════════════════════════════════════════════════════════

def _raw_trade_imbalance(bars: pd.DataFrame, **_kw) -> pd.Series:
    """(BuyVol − SellVol) / TotalVol."""
    total = bars["buy_volume"] + bars["sell_volume"] + 1e-10
    s = (bars["buy_volume"] - bars["sell_volume"]) / total
    s.name = "trade_imbalance"
    return s


def _raw_buy_sell_ratio(bars: pd.DataFrame, **_kw) -> pd.Series:
    s = np.log((bars["buy_volume"] + 1e-10) / (bars["sell_volume"] + 1e-10))
    s.name = "buy_sell_ratio"
    return s


def _raw_volume_zscore(bars: pd.DataFrame, **kw) -> pd.Series:
    w = kw.get("window", 60)
    mu = bars["volume"].rolling(w, min_periods=max(5, w // 4)).mean()
    sigma = bars["volume"].rolling(w, min_periods=max(5, w // 4)).std()
    s = (bars["volume"] - mu) / (sigma + 1e-10)
    s.name = "volume_zscore"
    return s


def _raw_kyles_lambda(bars: pd.DataFrame, **kw) -> pd.Series:
    w = kw.get("window", 60)
    ret = bars["close"].pct_change() * 10000
    signed_vol = bars["buy_volume"] - bars["sell_volume"]
    cov = ret.rolling(w, min_periods=max(5, w // 4)).cov(signed_vol)
    var = signed_vol.rolling(w, min_periods=max(5, w // 4)).var() + 1e-10
    s = cov / var
    s.name = "kyles_lambda"
    return s


def _raw_taker_vol_corr(bars: pd.DataFrame, **kw) -> pd.Series:
    w = kw.get("window", 60)
    s = bars["buy_volume"].rolling(w, min_periods=max(10, w // 4)).corr(bars["sell_volume"])
    s.name = "taker_vol_corr"
    return s


def _raw_volume_acceleration(bars: pd.DataFrame, **kw) -> pd.Series:
    fast, slow = kw.get("fast", 5), kw.get("slow", 20)
    s = bars["volume"].diff(fast).diff(slow)
    s.name = "volume_acceleration"
    return s


def _raw_price_impact(bars: pd.DataFrame, **_kw) -> pd.Series:
    s = (bars["close"] - bars["open"]).abs() / (bars["volume"] + 1e-10)
    s.name = "price_impact"
    return s


def _raw_taker_vol_skew(bars: pd.DataFrame, **kw) -> pd.Series:
    w = kw.get("window", 60)
    s = bars["volume"].rolling(w, min_periods=max(10, w // 4)).skew()
    s.name = "taker_vol_skew"
    return s


RAW_FACTOR_REGISTRY: Dict[str, Callable] = {
    "trade_imbalance": _raw_trade_imbalance,
    "buy_sell_ratio": _raw_buy_sell_ratio,
    "volume_zscore": _raw_volume_zscore,
    "kyles_lambda": _raw_kyles_lambda,
    "taker_vol_corr": _raw_taker_vol_corr,
    "volume_acceleration": _raw_volume_acceleration,
    "price_impact": _raw_price_impact,
    "taker_vol_skew": _raw_taker_vol_skew,
}


# ═══════════════════════════════════════════════════════════
# MERGED factors (need extended bar columns)
#
# Convention:  imbalance = (buy − sell) / (|buy| + |sell| + ε)
#              ratio     = buy / (sell + ε)
#              delta     = pct_change  then  buy_delta − sell_delta
# ═══════════════════════════════════════════════════════════

def _norm_imbalance(buy: pd.Series, sell: pd.Series) -> pd.Series:
    """Normalised imbalance: (buy − sell) / (|buy| + |sell| + ε)."""
    return (buy - sell) / (buy.abs() + sell.abs() + 1e-10)


def _merged_vwap_dist_sum_imbalance(bars: pd.DataFrame, **kw) -> pd.Series:
    w = kw.get("window", 10)
    b = bars["buy_vwap_dist_sum"].rolling(w, min_periods=max(3, w // 4)).sum()
    s = bars["sell_vwap_dist_sum"].rolling(w, min_periods=max(3, w // 4)).sum()
    out = _norm_imbalance(b, s)
    out.name = "vwap_dist_sum_imbalance"
    return out


def _merged_vwap_dist_sum_delta_imbalance(bars: pd.DataFrame, **_kw) -> pd.Series:
    """First-order difference of buy/sell vwap dist sums (pct change difference)."""
    eps = 1e-10
    bd = bars["buy_vwap_dist_sum"].diff() / (bars["buy_vwap_dist_sum"].shift(1).abs() + eps)
    sd = bars["sell_vwap_dist_sum"].diff() / (bars["sell_vwap_dist_sum"].shift(1).abs() + eps)
    out = bd - sd
    out.name = "vwap_dist_sum_delta_imbalance"
    return out


def _merged_multi_fill_count_imbalance(bars: pd.DataFrame, **kw) -> pd.Series:
    # 反应 informed trades 数量
    w = kw.get("window", 10)
    b = bars["buy_multi_fill_count"].rolling(w, min_periods=max(3, w // 4)).sum()
    s = bars["sell_multi_fill_count"].rolling(w, min_periods=max(3, w // 4)).sum()
    out = _norm_imbalance(b, s)
    out.name = "multi_fill_count_imbalance"
    return out


def _merged_multi_fill_volume_imbalance(bars: pd.DataFrame, **kw) -> pd.Series:
    # informed trades volume 的累积量差异
    w = kw.get("window", 10)
    b = bars["buy_multi_fill_volume"].rolling(w, min_periods=max(3, w // 4)).sum()
    s = bars["sell_multi_fill_volume"].rolling(w, min_periods=max(3, w // 4)).sum()
    out = _norm_imbalance(b, s)
    out.name = "multi_fill_volume_imbalance"
    return out


def _merged_mf_vol_median_ratio_imbalance(bars: pd.DataFrame, **kw) -> pd.Series:
    """Imbalance of multi-fill vs all volume median ratio (informed-trade proxy)."""
    w = kw.get("window", 10)
    b = bars["buy_mf_vol_median_ratio"].rolling(w, min_periods=max(3, w // 4)).mean()
    s = bars["sell_mf_vol_median_ratio"].rolling(w, min_periods=max(3, w // 4)).mean()
    out = _norm_imbalance(b, s)
    out.name = "mf_vol_median_ratio_imbalance"
    return out


def _merged_off_top_volume_imbalance(bars: pd.DataFrame, **kw) -> pd.Series:
    # 反应 aggressiveness 的累积量差异
    w = kw.get("window", 10)
    b = bars["buy_off_top_volume"].rolling(w, min_periods=max(3, w // 4)).sum()
    s = bars["sell_off_top_volume"].rolling(w, min_periods=max(3, w // 4)).sum()
    out = _norm_imbalance(b, s)
    out.name = "off_top_volume_imbalance"
    return out


def _merged_off_top_volume_delta_imbalance(bars: pd.DataFrame, **_kw) -> pd.Series:
    eps = 1e-10
    bd = bars["buy_off_top_volume"].diff() / (bars["buy_off_top_volume"].shift(1).abs() + eps)
    sd = bars["sell_off_top_volume"].diff() / (bars["sell_off_top_volume"].shift(1).abs() + eps)
    out = bd - sd
    out.name = "off_top_volume_delta_imbalance"
    return out


def _merged_volume_asymmetry(bars: pd.DataFrame, **kw) -> pd.Series:
    """pressure × covariance_sign  (see docstring in original)."""
    w = kw.get("window", 10)
    buy, sell = bars["buy_volume"], bars["sell_volume"]
    total = buy + sell + 1e-10
    pressure = (buy - sell) / total
    buy_dm = buy - buy.rolling(w, min_periods=3).mean()
    sell_dm = sell - sell.rolling(w, min_periods=3).mean()
    cov_sign = np.sign((buy_dm * sell_dm).rolling(max(3, w // 3), min_periods=2).mean())
    out = (pressure * cov_sign).rolling(max(3, w // 3), min_periods=2).mean()
    out.name = "volume_asymmetry"
    return out


def _merged_volume_autocorr_imbalance(bars: pd.DataFrame, **kw) -> pd.Series:
    w = kw.get("window", 10)
    ba = bars["buy_volume"].rolling(w, min_periods=max(5, w // 4)).corr(bars["buy_volume"].shift(1))
    sa = bars["sell_volume"].rolling(w, min_periods=max(5, w // 4)).corr(bars["sell_volume"].shift(1))
    out = _norm_imbalance(ba, sa)
    out.name = "volume_autocorr_imbalance"
    return out


def _merged_vwap_dist_reg_imbalance(bars: pd.DataFrame, **kw) -> pd.Series:
    w = kw.get("window", 10)
    b = bars["buy_vwap_dist_reg_beta"].rolling(w, min_periods=max(3, w // 4)).mean()
    s = bars["sell_vwap_dist_reg_beta"].rolling(w, min_periods=max(3, w // 4)).mean()
    out = _norm_imbalance(b, s)
    out.name = "vwap_dist_reg_imbalance"
    return out


def _merged_price_range_reg_imbalance(bars: pd.DataFrame, **kw) -> pd.Series:
    w = kw.get("window", 10)
    b = bars["buy_pr_reg_beta"].rolling(w, min_periods=max(3, w // 4)).mean()
    s = bars["sell_pr_reg_beta"].rolling(w, min_periods=max(3, w // 4)).mean()
    out = _norm_imbalance(b, s)
    out.name = "price_range_reg_imbalance"
    return out


def _merged_volume_skew_imbalance(bars: pd.DataFrame, **kw) -> pd.Series:
    w = kw.get("window", 10)
    b = bars["buy_volume_skew"].rolling(w, min_periods=max(3, w // 4)).mean()
    s = bars["sell_volume_skew"].rolling(w, min_periods=max(3, w // 4)).mean()
    out = _norm_imbalance(b, s)
    out.name = "volume_skew_imbalance"
    return out

def _merged_weighted_vwap_dist_sum(bars: pd.DataFrame, **kw) -> pd.Series:
    # 衡量过去一段时间内的volume加权的交易成本价偏差总和
    w = kw.get("window", 10)
    out = bars["weighted_vwap_dist"].rolling(w, min_periods=max(3, w // 4)).sum()
    out.name = "weighted_vwap_dist_sum"
    return out

MERGED_FACTOR_REGISTRY: Dict[str, Callable] = {
    "vwap_dist_sum_imbalance": _merged_vwap_dist_sum_imbalance,
    "vwap_dist_sum_delta_imbalance": _merged_vwap_dist_sum_delta_imbalance,
    "multi_fill_count_imbalance": _merged_multi_fill_count_imbalance,
    "multi_fill_volume_imbalance": _merged_multi_fill_volume_imbalance,
    "mf_vol_median_ratio_imbalance": _merged_mf_vol_median_ratio_imbalance,
    "off_top_volume_imbalance": _merged_off_top_volume_imbalance,
    "off_top_volume_delta_imbalance": _merged_off_top_volume_delta_imbalance,
    "volume_asymmetry": _merged_volume_asymmetry,
    "volume_autocorr_imbalance": _merged_volume_autocorr_imbalance,
    "vwap_dist_reg_imbalance": _merged_vwap_dist_reg_imbalance,
    "price_range_reg_imbalance": _merged_price_range_reg_imbalance,
    "volume_skew_imbalance": _merged_volume_skew_imbalance,
    "weighted_vwap_dist_sum": _merged_weighted_vwap_dist_sum,
}

# Combined registry for factor lookup
FACTOR_REGISTRY: Dict[str, Callable] = {**RAW_FACTOR_REGISTRY, **MERGED_FACTOR_REGISTRY}


# ═══════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════

def compute_factor(factor_name: str, bars: pd.DataFrame, **kwargs) -> pd.Series:
    """Compute a named factor.  No trades parameter — everything comes from bars."""
    if factor_name not in FACTOR_REGISTRY:
        raise ValueError(f"Unknown factor '{factor_name}'. "
                         f"Raw: {list(RAW_FACTOR_REGISTRY)}  "
                         f"Merged: {list(MERGED_FACTOR_REGISTRY)}")
    return FACTOR_REGISTRY[factor_name](bars, **kwargs)


def build_bars(
    trades_df: pd.DataFrame,
    trade_source: str,    # "raw" or "merged"
    bar_mode: str,        # "time" or "trade_count" or "volume"
    freq: str = "1min",
    trades_per_bar: int = 200,
    volume_per_bar: float = 0.0,
) -> pd.DataFrame:
    """
    Unified bar builder dispatching to the correct function.

    trade_source × bar_mode → builder:
      raw    + time        → build_time_bars
      raw    + trade_count → build_trade_count_bars_raw
      raw    + volume      → build_volume_bars_raw
      merged + time        → build_time_bars_merged
      merged + trade_count → build_trade_count_bars_merged
      merged + volume      → build_volume_bars_merged
    """
    if trade_source not in {"raw", "merged"}:
        raise ValueError(f"Unsupported trade_source: {trade_source}")
    if bar_mode not in {"time", "trade_count", "volume"}:
        raise ValueError(f"Unsupported bar_mode: {bar_mode}")

    if trade_source == "raw":
        if bar_mode == "trade_count":
            return build_trade_count_bars_raw(trades_df, trades_per_bar)
        if bar_mode == "volume":
            return build_volume_bars_raw(trades_df, volume_per_bar)
        return build_time_bars(trades_df, freq)
    else:
        if bar_mode == "trade_count":
            return build_trade_count_bars_merged(trades_df, trades_per_bar)
        if bar_mode == "volume":
            return build_volume_bars_merged(trades_df, volume_per_bar)
        return build_time_bars_merged(trades_df, freq)


def load_and_prepare(
    symbol: str = "SOL/USDC",
    trade_source: str = "raw",
    bar_mode: str = "time",
    freq: str = "1min",
    trades_per_bar: int = 200,
    volume_per_bar: float = 0.0,
    factor_name: str = "trade_imbalance",
    **factor_kwargs,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    End-to-end: load → (merge) → build bars → compute factor.
    Returns (bars, factor).
    """
    raw = load_cached_trades(symbol)

    if trade_source == "merged":
        trades_df = merge_aggtrades(raw)
    else:
        trades_df = raw

    bars = build_bars(
        trades_df,
        trade_source,
        bar_mode,
        freq,
        trades_per_bar,
        volume_per_bar,
    )
    factor = compute_factor(factor_name, bars, **factor_kwargs)
    return bars, factor
