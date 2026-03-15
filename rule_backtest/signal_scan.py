"""
Signal-only evaluation framework.

Phase 1 of the strategy pipeline:
  Factor × Signal parameter scan → rank by predictive quality + stability.
  No position sizing / stop-loss / exit — just raw signal quality.

Evaluation metrics
──────────────────
  ic           : Pearson correlation (continuous factor vs fwd return)
  rank_ic      : Spearman rank correlation (continuous factor vs fwd return)
  IC_mean      : Mean IC across rolling windows (robustness)
  IC_IR        : IC_mean / IC_std  (information ratio)
  triple_acc   : Accuracy of signal direction vs triple-barrier label
  hit_long     : P(fwd_ret > 0 | signal == +1)
  hit_short    : P(fwd_ret < 0 | signal == -1)
  signal_auto  : Lag-1 autocorrelation of signal (stability / turnover proxy)
  change_rate  : Fraction of bars where signal changes
  avg_duration : Average bars per signal regime
  long_frac    : Fraction of bars with signal == +1
  short_frac   : Fraction of bars with signal == -1
  flat_frac    : Fraction of bars with signal ==  0
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .signals import SIGNAL_CLASSES, BaseSignal
from .data_loader import (
    load_cached_trades,
    merge_aggtrades,
    build_bars,
    compute_factor,
    estimate_avg_bar_seconds,
    RAW_FACTOR_REGISTRY,
    MERGED_FACTOR_REGISTRY,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# Metric computation
# ═══════════════════════════════════════════════════════════

def evaluate_signal(
    factor: pd.Series,
    signal: pd.Series,
    bars: pd.DataFrame,
    flat_quantile: float = 0.33,
) -> Dict[str, float]:
    """
    Evaluate a signal against 1-bar forward returns.

    flat_quantile: quantile of |return| below which the bar is labelled 'flat'.
                   0.33 → roughly 1/3 up, 1/3 flat, 1/3 down.
    """
    close = bars["close"]
    fwd_ret = close.pct_change().shift(-1)  # return of NEXT bar

    # Align and drop NaN
    df = pd.DataFrame({"factor": factor, "signal": signal, "fwd_ret": fwd_ret}).dropna()
    if len(df) < 30:
        return _empty_metrics()

    sig = df["signal"].values
    ret = df["fwd_ret"].values
    fac = df["factor"].values

    # ── IC (continuous) ──
    ic, _ = stats.pearsonr(fac, ret)
    rank_ic, _ = stats.spearmanr(fac, ret)

    # Rolling IC (100-bar windows) for IC_IR
    ic_series = df["factor"].rolling(100, min_periods=50).corr(df["fwd_ret"])
    ic_mean = float(ic_series.mean()) if ic_series.notna().sum() > 10 else ic
    ic_std = float(ic_series.std()) if ic_series.notna().sum() > 10 else 1e-10
    ic_ir = ic_mean / (ic_std + 1e-10)

    # ── Triple-barrier direction labels ──
    abs_ret = np.abs(ret)
    threshold = np.quantile(abs_ret, flat_quantile)
    direction = np.where(ret > threshold, 1, np.where(ret < -threshold, -1, 0))

    # Accuracy (only where both signal and direction are non-zero)
    active = (sig != 0) & (direction != 0)
    if active.sum() > 0:
        triple_acc = float((sig[active] == direction[active]).mean())
    else:
        triple_acc = 0.0

    # Hit rates
    long_mask = sig == 1
    short_mask = sig == -1
    hit_long = float((ret[long_mask] > 0).mean()) if long_mask.sum() > 5 else 0.0
    hit_short = float((ret[short_mask] < 0).mean()) if short_mask.sum() > 5 else 0.0

    # Long / short mean returns (bps)
    long_ret_mean = float(ret[long_mask].mean() * 10000) if long_mask.sum() > 5 else 0.0
    short_ret_mean = float(ret[short_mask].mean() * 10000) if short_mask.sum() > 5 else 0.0

    # ── Signal stability ──
    sig_s = pd.Series(sig)
    signal_auto = float(sig_s.autocorr(1)) if len(sig_s) > 10 else 0.0
    changes = (sig_s.diff().abs() > 0).sum()
    change_rate = float(changes / len(sig_s))
    n_regimes = max(1, changes)
    avg_duration = float(len(sig_s) / n_regimes)

    # Fractions
    n = len(sig)
    long_frac = float((sig == 1).sum() / n)
    short_frac = float((sig == -1).sum() / n)
    flat_frac = float((sig == 0).sum() / n)

    return {
        "ic": round(ic, 6),
        "rank_ic": round(rank_ic, 6),
        "ic_mean": round(ic_mean, 6),
        "ic_ir": round(ic_ir, 4),
        "triple_acc": round(triple_acc, 4),
        "hit_long": round(hit_long, 4),
        "hit_short": round(hit_short, 4),
        "long_ret_bps": round(long_ret_mean, 2),
        "short_ret_bps": round(short_ret_mean, 2),
        "signal_auto": round(signal_auto, 4),
        "change_rate": round(change_rate, 4),
        "avg_duration": round(avg_duration, 1),
        "long_frac": round(long_frac, 3),
        "short_frac": round(short_frac, 3),
        "flat_frac": round(flat_frac, 3),
        "n_eval_bars": len(df),
    }


def _empty_metrics() -> Dict[str, float]:
    return {k: 0.0 for k in [
        "ic", "rank_ic", "ic_mean", "ic_ir", "triple_acc", "hit_long", "hit_short",
        "long_ret_bps", "short_ret_bps", "signal_auto", "change_rate",
        "avg_duration", "long_frac", "short_frac", "flat_frac", "n_eval_bars",
    ]}


# ═══════════════════════════════════════════════════════════
# Search-space definitions  (scale with bar granularity)
# ═══════════════════════════════════════════════════════════

# trades_per_bar levels + matching parameter ranges
# Raw ≈ 4.7 trades/s → 150/bar ≈ 32s, 500 ≈ 1.8min, 2000 ≈ 7min, 6000 ≈ 21min
# Merged ≈ 3.5 trades/s → same N gives longer bars

TRADE_COUNT_LEVELS = [150, 500, 2000, 6000]

# Search-space per level:  { trades_per_bar → (factor_windows, signal_windows) }
# Principle: max(signal_window) * trades_per_bar should leave ≥ 60% of data for eval.
# Principle: factor_window should be small (within-bar lookback for cross-bar aggregation).

_SEARCH_BY_LEVEL: Dict[int, Dict[str, Any]] = {
    150: {
        "factor_windows": [3, 5, 10, 20],
        "signal_cfgs": [
            *[{"type": "zscore", "window": w, "threshold": t}
              for w in [10, 20, 40, 60, 120] for t in [1.0, 1.5, 2.0]],
            *[{"type": "quantile", "window": w, "upper_q": q, "lower_q": 1 - q}
              for w in [10, 20, 40, 60, 120] for q in [0.8, 0.9]],
            *[{"type": "ma_cross", "fast_window": f, "slow_window": s}
              for f, s in [(3, 10), (5, 20), (5, 40), (10, 60)]],
            *[{"type": "rank", "window": w} for w in [20, 40, 60, 120]],
            *[{"type": "delta", "lookback": lb} for lb in [3, 5, 10, 20]],
            *[{"type": "bollinger", "window": w, "n_std": n}
              for w in [20, 40, 60] for n in [1.5, 2.0]],
        ],
    },
    500: {
        "factor_windows": [3, 5, 10, 15],
        "signal_cfgs": [
            *[{"type": "zscore", "window": w, "threshold": t}
              for w in [10, 20, 30, 60] for t in [1.0, 1.5, 2.0]],
            *[{"type": "quantile", "window": w, "upper_q": q, "lower_q": 1 - q}
              for w in [10, 20, 30, 60] for q in [0.8, 0.9]],
            *[{"type": "ma_cross", "fast_window": f, "slow_window": s}
              for f, s in [(3, 10), (5, 20), (5, 30), (10, 60)]],
            *[{"type": "rank", "window": w} for w in [10, 20, 30, 60]],
            *[{"type": "delta", "lookback": lb} for lb in [3, 5, 10]],
            *[{"type": "bollinger", "window": w, "n_std": n}
              for w in [10, 20, 30] for n in [1.5, 2.0]],
        ],
    },
    2000: {
        "factor_windows": [3, 5, 10],
        "signal_cfgs": [
            *[{"type": "zscore", "window": w, "threshold": t}
              for w in [5, 10, 20, 30] for t in [1.0, 1.5, 2.0]],
            *[{"type": "quantile", "window": w, "upper_q": q, "lower_q": 1 - q}
              for w in [5, 10, 20, 30] for q in [0.8, 0.9]],
            *[{"type": "ma_cross", "fast_window": f, "slow_window": s}
              for f, s in [(2, 5), (3, 10), (5, 20)]],
            *[{"type": "rank", "window": w} for w in [5, 10, 20, 30]],
            *[{"type": "delta", "lookback": lb} for lb in [2, 3, 5]],
            *[{"type": "bollinger", "window": w, "n_std": n}
              for w in [5, 10, 20] for n in [1.5, 2.0]],
        ],
    },
    6000: {
        "factor_windows": [3, 5, 8],
        "signal_cfgs": [
            *[{"type": "zscore", "window": w, "threshold": t}
              for w in [3, 5, 10, 15] for t in [1.0, 1.5, 2.0]],
            *[{"type": "quantile", "window": w, "upper_q": q, "lower_q": 1 - q}
              for w in [3, 5, 10, 15] for q in [0.8, 0.9]],
            *[{"type": "ma_cross", "fast_window": f, "slow_window": s}
              for f, s in [(2, 5), (3, 10), (3, 15)]],
            *[{"type": "rank", "window": w} for w in [3, 5, 10, 15]],
            *[{"type": "delta", "lookback": lb} for lb in [2, 3, 5]],
            *[{"type": "bollinger", "window": w, "n_std": n}
              for w in [3, 5, 10] for n in [1.5, 2.0]],
        ],
    },
}


def get_search_space(trades_per_bar: int) -> Dict[str, Any]:
    if trades_per_bar in _SEARCH_BY_LEVEL:
        return _SEARCH_BY_LEVEL[trades_per_bar]
    # Fall back to nearest level
    nearest = min(TRADE_COUNT_LEVELS, key=lambda x: abs(x - trades_per_bar))
    return _SEARCH_BY_LEVEL[nearest]


def build_signal_from_cfg(cfg: Dict[str, Any]) -> BaseSignal:
    c = dict(cfg)
    t = c.pop("type")
    return SIGNAL_CLASSES[t](**c)


# ═══════════════════════════════════════════════════════════
# Main scan runner
# ═══════════════════════════════════════════════════════════

def run_signal_scan(
    symbol: str = "SOL/USDC",
    trade_source: str = "raw",
    trades_per_bar: int = 500,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run signal scan for one trade_source × trades_per_bar combo.

    Returns DataFrame with all parameter combos and their evaluation metrics.
    """
    logger.info("═══ Signal scan: %s | source=%s | tpb=%d ═══",
                symbol, trade_source, trades_per_bar)

    # Load and build bars
    raw = load_cached_trades(symbol)
    if trade_source == "merged":
        trades_df = merge_aggtrades(raw)
    else:
        trades_df = raw

    bars = build_bars(trades_df, trade_source, "trade_count",
                      trades_per_bar=trades_per_bar)
    avg_sec = estimate_avg_bar_seconds(bars)
    logger.info("Built %d bars (~%.0fs avg duration)", len(bars), avg_sec)

    # Determine which factors are available
    if trade_source == "merged":
        factor_names = list(RAW_FACTOR_REGISTRY.keys()) + list(MERGED_FACTOR_REGISTRY.keys())
    else:
        factor_names = list(RAW_FACTOR_REGISTRY.keys())

    space = get_search_space(trades_per_bar)
    factor_windows = space["factor_windows"]
    signal_cfgs = space["signal_cfgs"]

    results: List[Dict[str, Any]] = []
    combo_count = 0

    for fn in factor_names:
        for fw in factor_windows:
            try:
                factor = compute_factor(fn, bars, window=fw)
            except Exception as e:
                logger.debug("Factor %s w=%d failed: %s", fn, fw, e)
                continue

            valid_pct = factor.dropna().shape[0] / len(bars)
            if valid_pct < 0.3:
                continue

            for sc in signal_cfgs:
                try:
                    sig_obj = build_signal_from_cfg(sc)
                    signal = sig_obj.generate(factor)
                except Exception:
                    continue

                metrics = evaluate_signal(factor, signal, bars)
                combo_count += 1

                row = {
                    "trade_source": trade_source,
                    "trades_per_bar": trades_per_bar,
                    "bar_duration_sec": round(avg_sec, 1),
                    "factor": fn,
                    "factor_window": fw,
                    **sc,   # signal params (type, window, threshold, etc.)
                    **metrics,
                }
                results.append(row)

    logger.info("Evaluated %d combos", combo_count)

    df = pd.DataFrame(results)
    if df.empty:
        return df

    df["abs_ic_ir"] = df["ic_ir"].abs()
    df.sort_values("abs_ic_ir", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Save
    if output_dir:
        from pathlib import Path as P
        out = P(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        tag = f"{trade_source}_tpb{trades_per_bar}"
        csv_path = out / f"signal_scan_{tag}.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Saved → %s  (%d rows)", csv_path, len(df))

    return df


def run_full_scan(
    symbol: str = "SOL/USDC",
    output_dir: str = "",
) -> pd.DataFrame:
    """
    Run scan across all trade_source × trades_per_bar combos.
    """
    all_dfs = []
    for tpb in TRADE_COUNT_LEVELS:
        for src in ["raw", "merged"]:
            try:
                df = run_signal_scan(symbol, src, tpb, output_dir)
                all_dfs.append(df)
            except Exception as e:
                logger.error("Scan failed: src=%s tpb=%d: %s", src, tpb, e)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined["abs_ic_ir"] = combined["ic_ir"].abs()
    combined.sort_values("abs_ic_ir", ascending=False, inplace=True)
    combined.reset_index(drop=True, inplace=True)

    if output_dir:
        from pathlib import Path as P
        out = P(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out / "signal_scan_combined.csv", index=False)
        _generate_summary(combined, out)
        logger.info("Combined results: %d rows → %s", len(combined), out)

    return combined


def _generate_summary(df: pd.DataFrame, out_dir) -> None:
    """Write a human-readable top-N summary."""
    import numpy as _np
    from pathlib import Path as P
    out = P(out_dir)

    if "abs_ic_ir" not in df.columns:
        df = df.copy()
        df["abs_ic_ir"] = df["ic_ir"].abs()

    lines = [
        "=" * 80,
        "SIGNAL SCAN SUMMARY",
        "=" * 80,
        f"Total combos evaluated: {len(df)}",
        f"Trade sources: {df['trade_source'].unique().tolist()}",
        f"Trades/bar levels: {sorted(df['trades_per_bar'].unique().tolist())}",
        f"Factors tested: {sorted(df['factor'].unique().tolist())}",
        "",
    ]

    show_cols = ["trade_source", "trades_per_bar", "factor", "factor_window",
                 "type", "window", "ic", "rank_ic", "ic_mean", "ic_ir", "abs_ic_ir", "triple_acc",
                 "hit_long", "hit_short", "long_ret_bps", "short_ret_bps",
                 "signal_auto", "change_rate"]
    present = [c for c in show_cols if c in df.columns]

    # Top 30 by |IC_IR| (captures both directional and inverse predictors)
    lines.append("── Top 30 by |IC_IR| ──")
    top = df.nlargest(30, "abs_ic_ir")
    lines.append(top[present].to_string(index=False))
    lines.append("")

    # Top 15 positive IC_IR
    lines.append("── Top 15 positive IC_IR (directional predictors) ──")
    pos = df[df["ic_ir"] > 0].nlargest(15, "ic_ir")
    lines.append(pos[present].to_string(index=False))
    lines.append("")

    # Top 15 negative IC_IR (inverse predictors — flip signal for profit)
    lines.append("── Top 15 negative IC_IR (inverse predictors — flip signal) ──")
    neg = df[df["ic_ir"] < 0].nsmallest(15, "ic_ir")
    lines.append(neg[present].to_string(index=False))
    lines.append("")

    # Best |IC_IR| per factor
    lines.append("── Best |IC_IR| per factor ──")
    best_idx = df.groupby("factor")["abs_ic_ir"].idxmax()
    best_per_factor = df.loc[best_idx].sort_values("abs_ic_ir", ascending=False)
    lines.append(best_per_factor[present].to_string(index=False))
    lines.append("")

    # Inverse predictor summary
    lines.append("── Factors with strong inverse prediction (mean IC < -0.02) ──")
    grp = df.groupby(["factor", "trade_source", "trades_per_bar"]).agg(
        mean_ic=("ic", "mean"), count=("ic", "size")
    ).reset_index()
    inv = grp[grp["mean_ic"] < -0.02].sort_values("mean_ic")
    lines.append(inv.to_string(index=False))
    lines.append("")

    # Raw vs merged comparison
    lines.append("── Raw vs Merged (mean |IC_IR| by factor) ──")
    pivot = df.groupby(["factor", "trade_source"])["abs_ic_ir"].mean().unstack(fill_value=0)
    if "merged" in pivot.columns and "raw" in pivot.columns:
        pivot["delta_merged_minus_raw"] = pivot["merged"] - pivot["raw"]
        pivot = pivot.sort_values("delta_merged_minus_raw", ascending=False)
    lines.append(pivot.to_string())
    lines.append("=" * 80)

    (out / "signal_scan_summary.txt").write_text("\n".join(lines), encoding="utf-8")
