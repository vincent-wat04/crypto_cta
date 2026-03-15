#!/usr/bin/env python3
"""
Run rule-based single-factor backtest with 4 data combos and optional optimization.

Data architecture:
  trade_source:  raw     (raw aggTrades)
                 merged  (multi-fill taker orders combined)
  bar_mode:      time        (fixed-interval, e.g. 1min)
                 trade_count (N trades per bar)

  → 4 combos:  raw_time, raw_trade_count, merged_time, merged_trade_count

Factor registries:
  Raw factors    — need only basic OHLCV bars
  Merged factors — need extended bars with merge-specific columns

Output folder structure:
  data/rule_backtest_results/{symbol}/{factor}/{source}_{bar_mode}/fill{rate}/

Usage:
  # Default: raw + time bars
  python scripts/run_rule_backtest.py --factor trade_imbalance

  # Merged + trade_count bars with specific fill rate
  python scripts/run_rule_backtest.py --factor vwap_dist_sum_imbalance \\
      --trade_source merged --bar_mode trade_count --maker_fill_rate 0.7

  # Grid search over all 4 combos
  python scripts/run_rule_backtest.py --factor trade_imbalance --optimize --all_combos

  # All merged factors at once
  python scripts/run_rule_backtest.py --factor all_merged --trade_source merged
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rule_backtest.data_loader import (
    load_cached_trades, merge_aggtrades, build_bars, compute_factor,
    estimate_avg_bar_seconds,
    RAW_FACTOR_REGISTRY, MERGED_FACTOR_REGISTRY, FACTOR_REGISTRY,
)
from rule_backtest.signals import SIGNAL_CLASSES
from rule_backtest.position_sizing import SIZER_CLASSES
from rule_backtest.stop_loss import STOPLOSS_CLASSES
from rule_backtest.exit_rules import EXIT_CLASSES
from rule_backtest.execution import MakerFirstExecution
from rule_backtest.engine import BacktestConfig, run_backtest
from rule_backtest.report import generate_report
from rule_backtest.optimizer import run_grid_search

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Rule-based single-factor backtest (4 data combos)")

    p.add_argument("--symbol", default="SOL/USDC")
    p.add_argument("--trade_source", choices=["raw", "merged"], default="raw",
                   help="Trade source: 'raw' aggTrades or 'merged' taker orders")
    p.add_argument("--bar_mode", choices=["time", "trade_count"], default="time")
    p.add_argument("--freq", default="1min", help="Bar frequency for time bars")
    p.add_argument("--trades_per_bar", type=int, default=200,
                   help="Trades per bar (trade_count mode)")
    p.add_argument("--all_combos", action="store_true",
                   help="Run all 4 trade_source × bar_mode combinations")
    p.add_argument("--factor", nargs="+", default=["trade_imbalance"],
                   help="Factor(s) to backtest. Use 'all_raw', 'all_merged', or 'all'.")

    # Signal
    p.add_argument("--signal", default="zscore",
                   help=f"Signal: {list(SIGNAL_CLASSES.keys())}")
    p.add_argument("--signal_window", type=int, default=60)
    p.add_argument("--signal_threshold", type=float, default=1.5)

    # Position sizing / stop / exit
    p.add_argument("--sizer", default="fixed")
    p.add_argument("--stoploss", default="none")
    p.add_argument("--stoploss_bps", type=float, default=30.0)
    p.add_argument("--exit_rule", default="signal_reversal")

    # Execution / cost sensitivity
    p.add_argument("--maker_fill_rate", type=float, default=0.7,
                   help="Maker fill probability (0-1) — also used in folder names for sensitivity")
    p.add_argument("--maker_fee_bps", type=float, default=2.0)
    p.add_argument("--taker_fee_bps", type=float, default=5.0)

    # Optimization
    p.add_argument("--optimize", action="store_true")
    p.add_argument("--sort_by", default="sharpe_ratio")
    p.add_argument("--top_n", type=int, default=20)

    # General
    p.add_argument("--initial_capital", type=float, default=100_000.0)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--log_level", default="INFO")

    return p.parse_args()


# ──────────────────────────────────────────────────────────
# Component builders
# ──────────────────────────────────────────────────────────

def build_signal(args):
    t = args.signal
    if t == "zscore":
        return SIGNAL_CLASSES[t](window=args.signal_window, threshold=args.signal_threshold)
    if t == "quantile":
        return SIGNAL_CLASSES[t](window=args.signal_window)
    if t == "ma_cross":
        return SIGNAL_CLASSES[t](fast_window=5, slow_window=args.signal_window)
    if t == "bollinger":
        return SIGNAL_CLASSES[t](window=args.signal_window, n_std=args.signal_threshold)
    if t == "rank":
        return SIGNAL_CLASSES[t](window=args.signal_window)
    if t == "delta":
        return SIGNAL_CLASSES[t](lookback=args.signal_window)
    if t == "dual_ma_band":
        return SIGNAL_CLASSES[t](fast_span=5, slow_span=args.signal_window)
    return SIGNAL_CLASSES[t]()


def build_sizer(args):
    return SIZER_CLASSES[args.sizer]()


def build_stoploss(args):
    t = args.stoploss
    if t == "fixed":
        return STOPLOSS_CLASSES[t](max_loss_bps=args.stoploss_bps)
    if t == "trailing":
        return STOPLOSS_CLASSES[t](trail_bps=args.stoploss_bps)
    if t == "time":
        return STOPLOSS_CLASSES[t](max_bars=int(args.stoploss_bps))
    return STOPLOSS_CLASSES.get(t, STOPLOSS_CLASSES["none"])()


def build_exit(args):
    return EXIT_CLASSES.get(args.exit_rule, EXIT_CLASSES["signal_reversal"])()


def bar_seconds_from_freq(freq: str) -> int:
    freq = freq.lower()
    if "min" in freq:
        return int(freq.replace("min", "")) * 60
    if "s" in freq:
        return int(freq.replace("s", ""))
    if "h" in freq:
        return int(freq.replace("h", "")) * 3600
    return 60


# ──────────────────────────────────────────────────────────
# Output directory naming
# ──────────────────────────────────────────────────────────

def make_output_dir(args, factor_name: str, trade_source: str, bar_mode: str,
                    suffix: str = "") -> Path:
    """
    Folder naming:
      {root}/data/rule_backtest_results/{symbol}/{factor}/{source}_{bar_mode}/fill{rate}/{suffix}
    """
    if args.output_dir:
        return Path(args.output_dir)

    sym = args.symbol.replace("/", "_")
    combo = f"{trade_source}_{bar_mode}"
    fill_tag = f"fill{args.maker_fill_rate:.2f}"

    base = ROOT / "data" / "rule_backtest_results" / sym / factor_name / combo / fill_tag
    if suffix:
        base = base / suffix
    return base


# ──────────────────────────────────────────────────────────
# Resolve factor list
# ──────────────────────────────────────────────────────────

def resolve_factors(factor_args: list, trade_source: str) -> list:
    if factor_args == ["all_raw"]:
        return list(RAW_FACTOR_REGISTRY.keys())
    if factor_args == ["all_merged"]:
        return list(MERGED_FACTOR_REGISTRY.keys())
    if factor_args == ["all"]:
        if trade_source == "merged":
            return list(FACTOR_REGISTRY.keys())
        return list(RAW_FACTOR_REGISTRY.keys())
    return factor_args


# ──────────────────────────────────────────────────────────
# Run modes
# ──────────────────────────────────────────────────────────

def run_single(args, factor_name: str, bars: pd.DataFrame, factor: pd.Series,
               bar_sec: int, trade_source: str, bar_mode: str):
    execution = MakerFirstExecution(
        maker_fee_bps=args.maker_fee_bps,
        taker_fee_bps=args.taker_fee_bps,
        maker_fill_rate=args.maker_fill_rate,
    )
    config = BacktestConfig(
        signal=build_signal(args),
        sizer=build_sizer(args),
        stop_loss=build_stoploss(args),
        exit_rule=build_exit(args),
        execution=execution,
        initial_capital=args.initial_capital,
        bar_seconds=bar_sec,
        bar_mode=bar_mode,
        trades_per_bar=args.trades_per_bar,
    )

    t0 = time.time()
    result = run_backtest(bars, factor, config, factor_name)
    elapsed = time.time() - t0

    print(f"\n{'='*70}")
    print(f"Factor: {factor_name} | Source: {trade_source} | Bar: {bar_mode} | "
          f"Fill: {args.maker_fill_rate} | {elapsed:.1f}s")
    print(f"{'='*70}")
    for k, v in result.metrics.items():
        print(f"  {k:30s}: {v}")

    out_dir = make_output_dir(args, factor_name, trade_source, bar_mode)
    generate_report(result.trades, result.metrics, result.params,
                    initial_capital=args.initial_capital, output_dir=out_dir)
    return result


def run_optimize(args, factor_name: str, bars: pd.DataFrame, factor: pd.Series,
                 bar_sec: int, trade_source: str, bar_mode: str):
    out_dir = make_output_dir(args, factor_name, trade_source, bar_mode, suffix="optimize")

    t0 = time.time()
    top_results = run_grid_search(
        bars=bars, factor=factor, factor_name=factor_name,
        sort_by=args.sort_by, top_n=args.top_n,
        bar_seconds=bar_sec, initial_capital=args.initial_capital,
        output_dir=out_dir,
    )
    elapsed = time.time() - t0

    print(f"\n{'='*70}")
    print(f"Grid: {factor_name} | {trade_source}_{bar_mode} | {len(top_results)} results | {elapsed:.1f}s")
    print(f"{'='*70}")
    if not top_results.empty:
        show_cols = ["signal", "maker_fill_rate", "sharpe_ratio",
                     "total_return_pct", "max_drawdown_pct"]
        present = [c for c in show_cols if c in top_results.columns]
        print(top_results[present].head(10).to_string(index=False))
    return top_results


# ──────────────────────────────────────────────────────────
# Data pipeline
# ──────────────────────────────────────────────────────────

def prepare_data(args, trade_source: str, bar_mode: str):
    """Load raw trades, optionally merge, build bars."""
    raw = load_cached_trades(args.symbol)

    if trade_source == "merged":
        trades_df = merge_aggtrades(raw)
    else:
        trades_df = raw

    bars = build_bars(trades_df, trade_source, bar_mode, args.freq, args.trades_per_bar)

    if bar_mode == "trade_count":
        bar_sec = int(estimate_avg_bar_seconds(bars))
        logger.info("Built %d trade-count bars (%d trades/bar, ~%ds avg)",
                     len(bars), args.trades_per_bar, bar_sec)
    else:
        bar_sec = bar_seconds_from_freq(args.freq)
        logger.info("Built %d time bars (%s)", len(bars), args.freq)

    return bars, bar_sec


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    combos = []
    if args.all_combos:
        combos = [
            ("raw", "time"), ("raw", "trade_count"),
            ("merged", "time"), ("merged", "trade_count"),
        ]
    else:
        combos = [(args.trade_source, args.bar_mode)]

    for trade_source, bar_mode in combos:
        logger.info("══ Data combo: %s + %s ══", trade_source, bar_mode)
        bars, bar_sec = prepare_data(args, trade_source, bar_mode)

        # Validate factors against the combo
        factors = resolve_factors(args.factor, trade_source)
        valid_registry = FACTOR_REGISTRY if trade_source == "merged" else RAW_FACTOR_REGISTRY
        for f in factors:
            if f not in valid_registry:
                logger.warning("Factor '%s' not available for trade_source='%s', skipping.", f, trade_source)
                continue

            logger.info("Computing factor: %s", f)
            factor = compute_factor(f, bars)

            if args.optimize:
                run_optimize(args, f, bars, factor, bar_sec, trade_source, bar_mode)
            else:
                run_single(args, f, bars, factor, bar_sec, trade_source, bar_mode)


if __name__ == "__main__":
    main()
