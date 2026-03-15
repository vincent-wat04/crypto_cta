#!/usr/bin/env python3
"""
Full rule-backtest for shortlisted signal-scan candidates.

Runs each candidate with:
  - maker_fill_rate ∈ [0.5, 0.7, 0.9]  (execution cost sensitivity)
  - Multiple stop-loss / exit / sizer combos
  - signal_flip for inverse-predictor factors
  - Asymmetric thresholds where signal scan shows long/short hit-rate skew

Output:
  {DATA_ROOT}/full_backtest_results/{symbol}/{candidate_tag}/{fill_tag}/
"""
from __future__ import annotations

import argparse
import itertools
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rule_backtest.data_loader import (
    load_cached_trades, merge_aggtrades, build_bars, compute_factor,
    estimate_avg_bar_seconds,
)
from rule_backtest.signals import (
    ZScoreSignal, QuantileSignal, MACrossSignal, BollingerSignal,
    RankSignal, DeltaSignal, BaseSignal, RawReversalSignal, RawSignal
)
from rule_backtest.position_sizing import FixedSizer, VolTargetSizer
from rule_backtest.stop_loss import (
    NoStopLoss, FixedStopLoss, TrailingStopLoss, TimeStopLoss,
)
from rule_backtest.exit_rules import SignalReversalExit, SignalNeutralExit, TimeDecayExit
from rule_backtest.execution import MakerFirstExecution
from rule_backtest.engine import BacktestConfig, run_backtest
from rule_backtest.report import generate_report

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# Candidate definitions
# ═══════════════════════════════════════════════════════════

@dataclass
class Candidate:
    tag: str
    factor: str
    factor_window: int
    trades_per_bar: int
    signal: BaseSignal
    signal_flip: bool = False
    tier: int = 1


def define_candidates() -> List[Candidate]:
    """Shortlisted from signal scan: merged trades, tpb=2000/6000."""
    return [
        # ── Tier 1: tpb=6000 merged (highest IC_IR) ──
        # Candidate(
        #     tag="vwap_dist_sum_imb__ma3_10",
        #     factor="vwap_dist_sum_imbalance", factor_window=3,
        #     trades_per_bar=6000,
        #     signal=MACrossSignal(fast_window=3, slow_window=10),
        #     tier=1,
        # ),
        # Candidate(
        #     tag="vwap_dist_sum_imb__q15_08",
        #     factor="vwap_dist_sum_imbalance", factor_window=3,
        #     trades_per_bar=6000,
        #     signal=QuantileSignal(window=15, upper_q=0.8, lower_q=0.2),
        #     tier=1,
        # ),
        # Candidate(
        #     tag="vwap_dist_sum_imb__z15_asym",
        #     factor="vwap_dist_sum_imbalance", factor_window=3,
        #     trades_per_bar=6000,
        #     signal=ZScoreSignal(window=15, long_threshold=1.5, short_threshold=1.0),
        #     tier=1,
        # ),
        # Candidate(
        #     tag="off_top_vol_imb__z10_15",
        #     factor="off_top_volume_imbalance", factor_window=3,
        #     trades_per_bar=6000,
        #     signal=ZScoreSignal(window=10, threshold=1.5),
        #     tier=1,
        # ),
        # Candidate(
        #     tag="off_top_vol_imb__z10_asym",
        #     factor="off_top_volume_imbalance", factor_window=3,
        #     trades_per_bar=6000,
        #     signal=ZScoreSignal(window=10, long_threshold=1.5, short_threshold=1.0),
        #     tier=1,
        # ),
        # Candidate(
        #     tag="off_top_vol_imb__q15_09",
        #     factor="off_top_volume_imbalance", factor_window=3,
        #     trades_per_bar=6000,
        #     signal=QuantileSignal(window=15, upper_q=0.9, lower_q=0.1),
        #     tier=1,
        # ),
        # Candidate(
        #     tag="vol_zscore_FLIP__boll10",
        #     factor="volume_zscore", factor_window=8,
        #     trades_per_bar=6000,
        #     signal=BollingerSignal(window=10, n_std=1.5),
        #     signal_flip=True,
        #     tier=1,
        # ),
        # # ── Tier 2: diversified factors ──
        # Candidate(
        #     tag="weighted_vwap__ma3_10",
        #     factor="weighted_vwap_dist_sum", factor_window=3,
        #     trades_per_bar=6000,
        #     signal=MACrossSignal(fast_window=3, slow_window=10),
        #     tier=2,
        # ),
        # Candidate(
        #     tag="price_range_reg__z10_15",
        #     factor="price_range_reg_imbalance", factor_window=3,
        #     trades_per_bar=6000,
        #     signal=ZScoreSignal(window=10, threshold=1.5),
        #     tier=2,
        # ),
        # Candidate(
        #     tag="multi_fill_vol__rank10",
        #     factor="multi_fill_volume_imbalance", factor_window=3,
        #     trades_per_bar=6000,
        #     signal=RankSignal(window=10, upper_pct=0.8, lower_pct=0.2),
        #     tier=2,
        # ),
        # Candidate(
        #     tag="vwap_dist_reg__ma3_15",
        #     factor="vwap_dist_reg_imbalance", factor_window=3,
        #     trades_per_bar=6000,
        #     signal=MACrossSignal(fast_window=3, slow_window=15),
        #     tier=2,
        # ),
        # # ── Tier 2: tpb=2000 merged (more data points) ──
        # Candidate(
        #     tag="vol_autocorr_FLIP__z10_20_t2k",
        #     factor="volume_autocorr_imbalance", factor_window=10,
        #     trades_per_bar=2000,
        #     signal=ZScoreSignal(window=10, threshold=2.0),
        #     signal_flip=True,
        #     tier=2,
        # ),
        # Candidate(
        #     tag="vol_skew_imb__q5_08_t2k",
        #     factor="volume_skew_imbalance", factor_window=3,
        #     trades_per_bar=2000,
        #     signal=QuantileSignal(window=5, upper_q=0.8, lower_q=0.2),
        #     tier=2,
        # ),
        # Candidate(
        #     tag="off_top_vol_delta_FLIP__d5_t2k",
        #     factor="off_top_volume_delta_imbalance", factor_window=10,
        #     trades_per_bar=2000,
        #     signal=DeltaSignal(lookback=5),
        #     signal_flip=True,
        #     tier=2,
        # ),
        # Candidate(
        #     tag="vwap_dist_sum_imbalance__raw_reversal",
        #     factor="vwap_dist_sum_imbalance", factor_window=5,
        #     trades_per_bar=6000,
        #     signal=RawReversalSignal(dead_zone=0.1),
        #     tier=1,
        # ),
        # Candidate(
        #     tag="vwap_dist_sum_imbalance__ma1_10",
        #     factor="vwap_dist_sum_imbalance", factor_window=5,
        #     trades_per_bar=2000,
        #     signal=MACrossSignal(fast_window=1, slow_window=10),
        #     tier=1,
        # ),
        # Candidate(
        #     tag="off_top_volume_delta_imbalance__raw_reversal",
        #     factor="off_top_volume_delta_imbalance", factor_window=5,
        #     trades_per_bar=6000,
        #     signal=RawReversalSignal(), 
        #     tier=1,
        # ),
        # Candidate(
        #     tag="off_top_volume_delta_imbalance__ma1_5",
        #     factor="off_top_volume_delta_imbalance", factor_window=5,
        #     trades_per_bar=6000,
        #     signal=MACrossSignal(fast_window=1, slow_window=5),     
        #     tier=1,
        # ),
        # Candidate(
        #     tag="off_top_volume_delta_imbalance__raw_reversal_tps500",
        #     factor="off_top_volume_delta_imbalance", factor_window=5,
        #     trades_per_bar=500,
        #     signal=RawReversalSignal(),     
        #     tier=1,
        # ),
        Candidate(
            tag="off_top_volume_imbalance__raw_reversal_tpb500",
            factor="off_top_volume_imbalance", factor_window=5,
            trades_per_bar=500,
            signal=RawReversalSignal(),     
            tier=1,
        ),
    ]


# ═══════════════════════════════════════════════════════════
# Strategy parameter grid (per candidate)
# ═══════════════════════════════════════════════════════════

MAKER_FILL_RATES = [0.5, 0.7, 0.9]

SIZER_GRID = [
    FixedSizer(size_units=1.0),
    VolTargetSizer(target_vol_bps=50.0, vol_window=30, max_leverage=3.0),
]

STOPLOSS_GRID = [
    NoStopLoss(),
    FixedStopLoss(max_loss_bps=30.0),
    FixedStopLoss(max_loss_bps=50.0),
    TrailingStopLoss(trail_bps=25.0),
    TrailingStopLoss(trail_bps=40.0),
    TimeStopLoss(max_bars=10),
    TimeStopLoss(max_bars=20),
]

EXIT_GRID = [
    SignalReversalExit(),
    SignalNeutralExit(),
    TimeDecayExit(max_hold_bars=10),
    TimeDecayExit(max_hold_bars=20),
]


def run_candidate(
    candidate: Candidate,
    bars: pd.DataFrame,
    bar_sec: int,
    output_root: Path,
    initial_capital: float = 100_000.0,
    quick: bool = False,
) -> pd.DataFrame:
    """Run full grid for one candidate. Returns results DataFrame."""

    factor = compute_factor(candidate.factor, bars, window=candidate.factor_window)

    if quick:
        sizers = [FixedSizer(size_units=1.0)]
        stoplosses = [NoStopLoss(), FixedStopLoss(max_loss_bps=40.0), TrailingStopLoss(trail_bps=30.0)]
        exits = [SignalReversalExit(), SignalNeutralExit()]
    else:
        sizers = SIZER_GRID
        stoplosses = STOPLOSS_GRID
        exits = EXIT_GRID

    combos = list(itertools.product(MAKER_FILL_RATES, sizers, stoplosses, exits))
    logger.info("  %s: %d combos (flip=%s)", candidate.tag, len(combos), candidate.signal_flip)

    rows: List[Dict[str, Any]] = []
    best_sharpe = -1e18
    best_result = None
    best_fill = 0.7

    for fill_rate, sizer, stoploss, exit_rule in combos:
        execution = MakerFirstExecution(
            maker_fee_bps=2.0, taker_fee_bps=5.0,
            maker_fill_rate=fill_rate,
        )
        config = BacktestConfig(
            signal=candidate.signal,
            sizer=sizer,
            stop_loss=stoploss,
            exit_rule=exit_rule,
            execution=execution,
            initial_capital=initial_capital,
            bar_seconds=bar_sec,
            bar_mode="trade_count",
            trades_per_bar=candidate.trades_per_bar,
            trade_source="merged",
            signal_flip=candidate.signal_flip,
        )

        try:
            stoploss.reset()
            result = run_backtest(bars, factor, config,
                                  factor_name=candidate.factor)
        except (ValueError, KeyError, ZeroDivisionError, RuntimeError) as e:
            logger.debug("  combo failed: %s", e)
            continue

        row = {
            "candidate": candidate.tag,
            "tier": candidate.tier,
            "factor": candidate.factor,
            "factor_window": candidate.factor_window,
            "signal_flip": candidate.signal_flip,
            "tpb": candidate.trades_per_bar,
            **result.params,
            **result.metrics,
        }
        rows.append(row)

        sharpe = result.metrics.get("sharpe_ratio", -1e18)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_result = result
            best_fill = fill_rate

    # Save best report
    if best_result is not None:
        fill_tag = f"fill{best_fill:.2f}"
        out_dir = output_root / candidate.tag / fill_tag / "best"
        generate_report(
            best_result.trades, best_result.metrics, best_result.params,
            initial_capital=initial_capital, output_dir=out_dir,
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        csv_path = output_root / candidate.tag / "grid_results.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)

    return df


# ═══════════════════════════════════════════════════════════
# Summary across all candidates
# ═══════════════════════════════════════════════════════════

def generate_cross_candidate_summary(
    all_results: pd.DataFrame,
    output_root: Path,
) -> None:
    """Write top results and fill-rate sensitivity report."""
    lines = [
        "=" * 100,
        "FULL BACKTEST SUMMARY — ALL CANDIDATES",
        "=" * 100,
        f"Total evaluations: {len(all_results)}",
        f"Candidates: {all_results['candidate'].nunique()}",
        "",
    ]

    # Top 30 by Sharpe
    lines.append("── Top 30 by Sharpe Ratio ──")
    show_cols = [
        "candidate", "tier", "signal_flip", "maker_fill_rate",
        "sharpe_ratio", "sortino_ratio", "total_return_pct",
        "max_drawdown_pct", "win_rate_pct", "profit_factor",
        "net_pnl_bps", "long_pnl_bps", "short_pnl_bps",
        "total_cost_bps", "n_round_trips", "avg_hold_bars",
    ]
    present = [c for c in show_cols if c in all_results.columns]
    top = all_results.nlargest(30, "sharpe_ratio")
    lines.append(top[present].to_string(index=False))
    lines.append("")

    # Best per candidate
    lines.append("── Best Sharpe per Candidate ──")
    best_idx = all_results.groupby("candidate")["sharpe_ratio"].idxmax()
    best = all_results.loc[best_idx].sort_values("sharpe_ratio", ascending=False)
    lines.append(best[present].to_string(index=False))
    lines.append("")

    # Fill-rate sensitivity
    lines.append("── Maker Fill Rate Sensitivity (mean Sharpe per candidate × fill_rate) ──")
    if "maker_fill_rate" in all_results.columns:
        pivot = all_results.groupby(
            ["candidate", "maker_fill_rate"]
        )["sharpe_ratio"].mean().unstack(fill_value=0)
        lines.append(pivot.to_string())
    lines.append("")

    # Long vs Short PnL breakdown
    lines.append("── Long vs Short PnL (best Sharpe per candidate) ──")
    ls_cols = ["candidate", "long_pnl_bps", "short_pnl_bps", "net_pnl_bps",
               "total_cost_bps", "sharpe_ratio"]
    ls_present = [c for c in ls_cols if c in best.columns]
    lines.append(best[ls_present].to_string(index=False))
    lines.append("=" * 100)

    summary_path = output_root / "full_backtest_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Summary → %s", summary_path)

    csv_path = output_root / "full_backtest_combined.csv"
    all_results.to_csv(csv_path, index=False)
    logger.info("Combined CSV → %s", csv_path)


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Full rule-backtest for shortlisted candidates")
    p.add_argument("--symbol", default="SOL/USDC")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--initial_capital", type=float, default=100_000.0)
    p.add_argument("--tier", type=int, default=0,
                   help="Only run tier N candidates (0 = all)")
    p.add_argument("--candidate", default=None,
                   help="Run a single candidate by tag name")
    p.add_argument("--quick", action="store_true",
                   help="Reduced grid for faster testing")
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    import os
    data_root = Path(os.environ.get("MR_DATA_ROOT", str(ROOT / "data")))
    sym_tag = args.symbol.replace("/", "_")
    output_root = Path(args.output_dir) if args.output_dir else (
        data_root / "full_backtest_results" / sym_tag
    )
    output_root.mkdir(parents=True, exist_ok=True)

    candidates = define_candidates()
    if args.candidate:
        candidates = [c for c in candidates if c.tag == args.candidate]
    if args.tier > 0:
        candidates = [c for c in candidates if c.tier == args.tier]

    if not candidates:
        print("No matching candidates found.")
        return

    # Group by tpb to reuse bars
    from collections import defaultdict
    by_tpb: Dict[int, List[Candidate]] = defaultdict(list)
    for c in candidates:
        by_tpb[c.trades_per_bar].append(c)

    print(f"\n{'='*70}")
    print(f"Full Rule Backtest — {args.symbol}")
    print(f"  Candidates: {len(candidates)}")
    print(f"  TPB levels: {sorted(by_tpb.keys())}")
    print(f"  Fill rates: {MAKER_FILL_RATES}")
    n_combos = len(candidates) * len(MAKER_FILL_RATES) * (
        6 if args.quick else len(SIZER_GRID) * len(STOPLOSS_GRID) * len(EXIT_GRID))
    print(f"  Est. combos: ~{n_combos}")
    print(f"  Output: {output_root}")
    print(f"{'='*70}\n")

    t0 = time.time()
    all_dfs = []

    for tpb, cands in sorted(by_tpb.items()):
        print(f"\n── Loading merged bars: tpb={tpb} ──")
        raw = load_cached_trades(args.symbol)
        merged = merge_aggtrades(raw)
        bars = build_bars(merged, "merged", "trade_count", trades_per_bar=tpb)
        bar_sec = int(estimate_avg_bar_seconds(bars))
        print(f"  Built {len(bars)} bars (~{bar_sec}s avg)")

        for c in cands:
            ct0 = time.time()
            df = run_candidate(c, bars, bar_sec, output_root,
                               args.initial_capital, args.quick)
            ct1 = time.time()
            if not df.empty:
                all_dfs.append(df)
                best = df.nlargest(1, "sharpe_ratio").iloc[0]
                print(f"  [{c.tag}] {len(df)} combos in {ct1-ct0:.1f}s "
                      f"| best Sharpe={best['sharpe_ratio']:.3f} "
                      f"| ret={best.get('total_return_pct',0):.2f}% "
                      f"| DD={best.get('max_drawdown_pct',0):.2f}%")
            else:
                print(f"  [{c.tag}] no valid results")

    elapsed = time.time() - t0

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.sort_values("sharpe_ratio", ascending=False, inplace=True)
        combined.reset_index(drop=True, inplace=True)
        generate_cross_candidate_summary(combined, output_root)

        print(f"\n{'='*70}")
        print(f"DONE: {len(combined)} evaluations in {elapsed:.0f}s ({elapsed/60:.1f} min)")
        print(f"Results: {output_root / 'full_backtest_combined.csv'}")
        print(f"Summary: {output_root / 'full_backtest_summary.txt'}")
        print(f"{'='*70}")
    else:
        print(f"\nNo results. Elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
