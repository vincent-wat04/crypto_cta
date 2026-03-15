"""
Factor quality evaluation metrics.

For each factor, compute:
  1. IC (Information Coefficient): rank correlation with future returns
  2. IC half-life: how fast IC decays over time
  3. IC autocorrelation: stability of IC across time
  4. Decile analysis: bucket factor values into 10 groups, measure mean future return per bucket
  5. Turnover: how much the factor ranking changes between periods

All metrics evaluated at 1min and 5min forward return horizons.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def compute_factor_ic(
    factor_values: pd.Series,
    forward_returns: pd.Series,
    method: str = "spearman",
) -> float:
    """
    Information Coefficient: rank correlation between factor and future returns.

    method: "spearman" (rank) or "pearson" (linear)
    """
    mask = ~(factor_values.isna() | forward_returns.isna())
    if mask.sum() < 10:
        return np.nan

    f = factor_values[mask].values
    r = forward_returns[mask].values

    if method == "spearman":
        ic, _ = sp_stats.spearmanr(f, r)
    else:
        ic, _ = sp_stats.pearsonr(f, r)
    return ic


def compute_rolling_ic(
    factor_values: pd.Series,
    forward_returns: pd.Series,
    window: int = 100,
    method: str = "spearman",
) -> pd.Series:
    """Rolling IC over time windows."""
    ics = pd.Series(np.nan, index=factor_values.index)
    for i in range(window, len(factor_values)):
        f = factor_values.iloc[i - window:i]
        r = forward_returns.iloc[i - window:i]
        mask = ~(f.isna() | r.isna())
        if mask.sum() > 10:
            if method == "spearman":
                ic, _ = sp_stats.spearmanr(f[mask].values, r[mask].values)
            else:
                ic, _ = sp_stats.pearsonr(f[mask].values, r[mask].values)
            ics.iloc[i] = ic
    return ics


def compute_ic_half_life(rolling_ic: pd.Series) -> float:
    """
    Estimate IC half-life: how many periods until IC autocorrelation drops to 0.5.

    Uses exponential decay fit on IC autocorrelation function.
    """
    ic_clean = rolling_ic.dropna()
    if len(ic_clean) < 20:
        return np.nan

    max_lag = min(50, len(ic_clean) // 4)
    autocorrs = [ic_clean.autocorr(lag=lag) for lag in range(1, max_lag + 1)]
    autocorrs = np.array(autocorrs)

    # Find first crossing below 0.5
    below = np.where(autocorrs < 0.5)[0]
    if len(below) > 0:
        return float(below[0] + 1)

    # Exponential fit: autocorr(lag) = exp(-lag / half_life)
    valid = autocorrs > 0
    if valid.sum() < 3:
        return float(max_lag)
    lags = np.arange(1, max_lag + 1)[valid]
    log_ac = np.log(autocorrs[valid])
    slope, _, _, _, _ = sp_stats.linregress(lags, log_ac)
    if slope >= 0:
        return float(max_lag)
    return float(-np.log(2) / slope)


def compute_ic_stability(rolling_ic: pd.Series) -> Dict[str, float]:
    """
    IC stability metrics.

    Returns:
      - ic_mean: mean IC
      - ic_std: IC standard deviation
      - ic_ir: IC information ratio (mean/std)
      - ic_positive_pct: % of periods with positive IC
      - ic_autocorr_1: first-order autocorrelation of IC
    """
    ic_clean = rolling_ic.dropna()
    if len(ic_clean) < 5:
        return {"ic_mean": np.nan, "ic_std": np.nan, "ic_ir": np.nan,
                "ic_positive_pct": np.nan, "ic_autocorr_1": np.nan}

    return {
        "ic_mean": float(ic_clean.mean()),
        "ic_std": float(ic_clean.std()),
        "ic_ir": float(ic_clean.mean() / (ic_clean.std() + 1e-10)),
        "ic_positive_pct": float((ic_clean > 0).mean() * 100),
        "ic_autocorr_1": float(ic_clean.autocorr(lag=1)) if len(ic_clean) > 5 else np.nan,
    }


def compute_decile_returns(
    factor_values: pd.Series,
    forward_returns: pd.Series,
    n_buckets: int = 10,
) -> pd.DataFrame:
    """
    Bucket factor values into n_buckets, compute mean future return per bucket.

    Returns DataFrame with columns:
      - bucket (1 to n_buckets, 1=lowest factor, n_buckets=highest)
      - count
      - mean_return
      - std_return
      - t_stat
    """
    mask = ~(factor_values.isna() | forward_returns.isna())
    f = factor_values[mask]
    r = forward_returns[mask]

    if len(f) < n_buckets * 5:
        return pd.DataFrame()

    try:
        buckets = pd.qcut(f, n_buckets, labels=False, duplicates="drop") + 1
    except ValueError:
        return pd.DataFrame()

    results = []
    for b in sorted(buckets.unique()):
        b_returns = r[buckets == b]
        n = len(b_returns)
        if n < 2:
            continue
        m = float(b_returns.mean())
        s = float(b_returns.std())
        t = m / (s / np.sqrt(n) + 1e-12)
        results.append({
            "bucket": int(b),
            "count": n,
            "mean_return": m,
            "std_return": s,
            "t_stat": t,
        })

    return pd.DataFrame(results)


def compute_factor_turnover(factor_values: pd.Series, d: int = 1) -> float:
    """
    Factor turnover: mean absolute change in factor rank between periods.

    Low turnover = stable signal = lower trading costs.
    """
    ranks = factor_values.rank(pct=True)
    rank_change = (ranks - ranks.shift(d)).abs()
    return float(rank_change.mean()) if not rank_change.isna().all() else np.nan


def evaluate_factor(
    factor_values: pd.Series,
    returns_1min: pd.Series,
    returns_5min: pd.Series,
    ic_window: int = 100,
) -> Dict[str, Any]:
    """
    Comprehensive factor evaluation at both 1min and 5min horizons.

    Returns dict with all metrics for both horizons.
    """
    result = {}

    for horizon_name, fwd_ret in [("1min", returns_1min), ("5min", returns_5min)]:
        # Align
        common_idx = factor_values.index.intersection(fwd_ret.index)
        f = factor_values.reindex(common_idx)
        r = fwd_ret.reindex(common_idx)

        # Overall IC
        ic = compute_factor_ic(f, r)
        result[f"ic_{horizon_name}"] = ic

        # Rolling IC
        rolling_ic = compute_rolling_ic(f, r, window=ic_window)

        # IC stability
        stability = compute_ic_stability(rolling_ic)
        for k, v in stability.items():
            result[f"{k}_{horizon_name}"] = v

        # IC half-life
        half_life = compute_ic_half_life(rolling_ic)
        result[f"ic_half_life_{horizon_name}"] = half_life

        # Decile analysis
        deciles = compute_decile_returns(f, r)
        if not deciles.empty:
            result[f"decile_spread_{horizon_name}"] = float(
                deciles["mean_return"].iloc[-1] - deciles["mean_return"].iloc[0]
            )
            result[f"decile_monotonicity_{horizon_name}"] = float(
                sp_stats.spearmanr(
                    deciles["bucket"].values,
                    deciles["mean_return"].values,
                )[0]
            ) if len(deciles) > 2 else np.nan
            result[f"decile_top_return_{horizon_name}"] = float(deciles["mean_return"].iloc[-1])
            result[f"decile_bottom_return_{horizon_name}"] = float(deciles["mean_return"].iloc[0])
        else:
            result[f"decile_spread_{horizon_name}"] = np.nan
            result[f"decile_monotonicity_{horizon_name}"] = np.nan

    # Factor turnover
    result["turnover"] = compute_factor_turnover(factor_values)

    return result
