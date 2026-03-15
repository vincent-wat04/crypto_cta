#!/usr/bin/env python3
"""
Run signal-quality scan: Factor × Signal parameter search.

Phase 1 of strategy pipeline — evaluate signal predictive power and stability
before layering on position sizing / stop-loss / exit rules.

Usage:
  # Full scan: all trade_source × trades_per_bar combos
  python scripts/run_signal_scan.py

  # Single combo
  python scripts/run_signal_scan.py --trade_source merged --trades_per_bar 500

  # Only specific levels
  python scripts/run_signal_scan.py --levels 150 500
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rule_backtest.signal_scan import (
    run_signal_scan, TRADE_COUNT_LEVELS,
    get_search_space, RAW_FACTOR_REGISTRY, MERGED_FACTOR_REGISTRY,
)

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Signal-quality parameter scan")
    p.add_argument("--symbol", default="SOL/USDC")
    p.add_argument("--trade_source", choices=["raw", "merged", "both"], default="both")
    p.add_argument("--levels", nargs="*", type=int, default=None,
                   help=f"Trades-per-bar levels to scan (default: all {TRADE_COUNT_LEVELS})")
    p.add_argument("--trades_per_bar", type=int, default=None,
                   help="Single trades_per_bar (overrides --levels)")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def _estimate_combos(levels, sources):
    """Print estimated combo count and runtime."""
    total = 0
    for tpb in levels:
        space = get_search_space(tpb)
        n_sig = len(space["signal_cfgs"])
        n_fw = len(space["factor_windows"])
        for src in sources:
            n_fac = len(RAW_FACTOR_REGISTRY) + (len(MERGED_FACTOR_REGISTRY) if src == "merged" else 0)
            combos = n_fac * n_fw * n_sig
            total += combos
            print(f"  {src:7s} tpb={tpb:5d}: {n_fac} factors × {n_fw} fwindows × {n_sig} signals = {combos:,} combos")
    print(f"  Total: {total:,} evaluations")
    est_minutes = total * 0.002 / 60 + len(levels) * len(sources) * 0.5
    print(f"  Estimated runtime: {est_minutes:.1f} - {est_minutes * 2:.1f} minutes")
    return total


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    sym_tag = args.symbol.replace("/", "_")
    out_dir = args.output_dir or str(ROOT / "data" / "signal_scan_results" / sym_tag)

    sources = ["raw", "merged"] if args.trade_source == "both" else [args.trade_source]

    if args.trades_per_bar:
        levels = [args.trades_per_bar]
    elif args.levels:
        levels = args.levels
    else:
        levels = TRADE_COUNT_LEVELS

    print(f"\n{'='*60}")
    print(f"Signal Scan: {args.symbol}")
    print(f"Sources: {sources}  |  Levels: {levels}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")
    _estimate_combos(levels, sources)
    print()

    t0 = time.time()
    all_dfs = []

    for tpb in levels:
        for src in sources:
            try:
                df = run_signal_scan(args.symbol, src, tpb, out_dir)
                all_dfs.append(df)
                if not df.empty:
                    top = df.head(5)
                    print(f"\n  Top 5 for {src} tpb={tpb}:")
                    show = ["factor", "factor_window", "type", "ic_ir", "triple_acc",
                            "hit_long", "hit_short", "signal_auto"]
                    present = [c for c in show if c in top.columns]
                    print(top[present].to_string(index=False))
            except Exception as e:
                logger.error("Failed: %s tpb=%d: %s", src, tpb, e)

    elapsed = time.time() - t0

    if all_dfs:
        import pandas as pd
        combined = pd.concat(all_dfs, ignore_index=True)
        combined["abs_ic_ir"] = combined["ic_ir"].abs()
        combined.sort_values("abs_ic_ir", ascending=False, inplace=True)
        combined.reset_index(drop=True, inplace=True)
        csv = Path(out_dir) / "signal_scan_combined.csv"
        combined.to_csv(csv, index=False)

        from rule_backtest.signal_scan import _generate_summary
        _generate_summary(combined, out_dir)

        print(f"\n{'='*60}")
        print(f"DONE: {len(combined):,} combos in {elapsed:.0f}s ({elapsed/60:.1f} min)")
        print(f"Results: {csv}")
        print(f"Summary: {Path(out_dir) / 'signal_scan_summary.txt'}")
        print(f"{'='*60}")
    else:
        print(f"\nNo results. Elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
