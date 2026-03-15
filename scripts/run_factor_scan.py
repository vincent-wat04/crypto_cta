#!/usr/bin/env python3
"""
Factor scanning and evaluation runner.

Usage:
  # Basic scan (1st-order only, quick):
  python scripts/run_factor_scan.py --symbol SOL/USDT --days 30 --max-order 1

  # Full scan (1st + 2nd order):
  python scripts/run_factor_scan.py --symbol SOL/USDT --days 365 --max-order 2

  # With orderbook data:
  python scripts/run_factor_scan.py --symbol SOL/USDT --days 365 --orderbook-path data/orderbook/

  # Limit expression count (for testing):
  python scripts/run_factor_scan.py --symbol SOL/USDT --days 7 --max-expressions 500
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from factors.data_schema import (
    TRADE_PRIMITIVES,
    ORDERBOOK_PRIMITIVES,
    compute_trade_primitives,
    compute_orderbook_primitives,
)
from factors.scanner import (
    generate_first_order_expressions,
    generate_second_order_expressions,
    generate_orderbook_expressions,
    scan_factors,
    ALL_LOOKBACKS,
)

logger = logging.getLogger("factor_scan")


# ══════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════

def load_trade_data(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Load perpetual aggTrades for date range."""
    from utilities.binance_loader import load_agg_trades_perpetual

    logger.info("Loading %s perpetual aggTrades: %s to %s", symbol, start_date, end_date)
    trades = load_agg_trades_perpetual(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
    )
    logger.info("Loaded %d trades", len(trades))
    return trades


def load_orderbook_data(path: str) -> pd.DataFrame:
    """Load orderbook data from parquet files."""
    ob_path = Path(path)
    if ob_path.is_dir():
        files = sorted(ob_path.glob("*.parquet"))
        if not files:
            logger.warning("No parquet files found in %s", path)
            return pd.DataFrame()
        dfs = [pd.read_parquet(f) for f in files]
        return pd.concat(dfs, ignore_index=True)
    elif ob_path.suffix == ".parquet":
        return pd.read_parquet(ob_path)
    else:
        logger.warning("Unknown orderbook format: %s", path)
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════
# Pipeline
# ══════════════════════════════════════════════════════════

def run_scan(args: argparse.Namespace) -> pd.DataFrame:
    """Execute the full factor scan pipeline."""
    end_dt = date.today()
    start_dt = end_dt - timedelta(days=args.days)

    # 1. Load raw data
    trades = load_trade_data(args.symbol, start_dt, end_dt)
    if trades.empty:
        logger.error("No trade data loaded")
        return pd.DataFrame()

    # 2. Compute primitives
    logger.info("Computing trade primitives at %s frequency...", args.freq)
    primitives = compute_trade_primitives(trades, freq=args.freq)
    logger.info("Primitives shape: %s, columns: %d", primitives.shape, len(primitives.columns))
    del trades  # free memory

    if len(primitives) < 500:
        logger.error("Too few bars (%d), need at least 500", len(primitives))
        return pd.DataFrame()

    # 3. Forward returns
    cum_ret = primitives["return_1"].cumsum()
    freq_sec = pd.Timedelta(args.freq).total_seconds()
    bars_1min = max(1, int(60 / freq_sec))
    bars_5min = max(1, int(300 / freq_sec))

    returns_1min = cum_ret.shift(-bars_1min) - cum_ret
    returns_5min = cum_ret.shift(-bars_5min) - cum_ret
    logger.info("Forward returns: 1min=%d bars, 5min=%d bars", bars_1min, bars_5min)

    # 4. Build expression list
    avail_prims = [p for p in TRADE_PRIMITIVES if p in primitives.columns]
    logger.info("Available primitives: %d / %d", len(avail_prims), len(TRADE_PRIMITIVES))

    expressions = generate_first_order_expressions(avail_prims)
    logger.info("1st-order expressions: %d", len(expressions))

    if args.max_order >= 2:
        second_order = generate_second_order_expressions(avail_prims)
        logger.info("2nd-order expressions: %d", len(second_order))
        expressions.extend(second_order)

    # 5. Orderbook data (optional)
    data = primitives
    if args.orderbook_path:
        ob_raw = load_orderbook_data(args.orderbook_path)
        if not ob_raw.empty:
            logger.info("Computing orderbook primitives...")
            ob_prims = compute_orderbook_primitives(ob_raw)
            ob_prims = ob_prims.reindex(primitives.index, method="ffill")
            data = pd.concat([primitives, ob_prims], axis=1)

            avail_ob = [p for p in ORDERBOOK_PRIMITIVES if p in ob_prims.columns]
            ob_exprs = generate_orderbook_expressions(avail_ob, avail_prims)
            logger.info("Orderbook expressions: %d", len(ob_exprs))
            expressions.extend(ob_exprs)
            del ob_raw, ob_prims

    logger.info("Total expressions to scan: %d", len(expressions))

    # 6. Scan
    results = scan_factors(
        data=data,
        returns_1min=returns_1min,
        returns_5min=returns_5min,
        expressions=expressions,
        primitives=avail_prims,
        max_order=args.max_order,
        ic_window=args.ic_window,
        min_ic_abs=args.min_ic,
        max_expressions=args.max_expressions,
        verbose=True,
    )

    return results


# ══════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════

def save_results(results: pd.DataFrame, out_dir: Path, args: argparse.Namespace):
    """Save scan results and summary report."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Full results
    results_path = out_dir / "factor_scan_results.parquet"
    results.to_parquet(results_path, index=False)
    logger.info("Full results saved to %s", results_path)

    # CSV for readability
    csv_path = out_dir / "factor_scan_results.csv"
    results.to_csv(csv_path, index=False, float_format="%.6f")

    # Top factors summary
    n_top = min(50, len(results))
    top = results.head(n_top)
    summary_path = out_dir / "top_factors_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Factor Scan Summary\n")
        f.write(f"{'=' * 80}\n")
        f.write(f"Symbol: {args.symbol}\n")
        f.write(f"Days: {args.days}\n")
        f.write(f"Frequency: {args.freq}\n")
        f.write(f"Max order: {args.max_order}\n")
        f.write(f"Max expressions: {args.max_expressions}\n")
        f.write(f"Min |IC|: {args.min_ic}\n")
        f.write(f"Total passed: {len(results)}\n")
        f.write(f"{'=' * 80}\n\n")

        f.write(f"Top {n_top} Factors by |IC_5min|:\n")
        f.write(f"{'-' * 80}\n")
        for i, row in top.iterrows():
            f.write(f"\n#{i+1}: {row['expression']}\n")
            f.write(f"  Order: {row.get('order', '?')}  |  Fields: {row.get('fields', '?')}\n")
            f.write(f"  IC_1min: {row.get('ic_1min', 0):.4f}  |  IC_5min: {row.get('ic_5min', 0):.4f}\n")
            ic_ir_5 = row.get('ic_ir_5min', np.nan)
            ic_hl_5 = row.get('ic_half_life_5min', np.nan)
            f.write(f"  IC_IR_5min: {ic_ir_5:.3f}  |  IC_half_life_5min: {ic_hl_5:.1f}\n")
            ds_1 = row.get('decile_spread_1min', np.nan)
            ds_5 = row.get('decile_spread_5min', np.nan)
            dm_5 = row.get('decile_monotonicity_5min', np.nan)
            f.write(f"  Decile_spread_1min: {ds_1:.2f}  |  Decile_spread_5min: {ds_5:.2f}\n")
            f.write(f"  Decile_monotonicity_5min: {dm_5:.3f}  |  Turnover: {row.get('turnover', 0):.4f}\n")

        # Aggregate stats
        f.write(f"\n{'=' * 80}\n")
        f.write(f"Aggregate Statistics\n")
        f.write(f"{'-' * 80}\n")
        if "order" in results.columns:
            for order in sorted(results["order"].unique()):
                subset = results[results["order"] == order]
                f.write(f"\nOrder {order}: {len(subset)} factors\n")
                f.write(f"  Median |IC_5min|: {subset['ic_5min'].abs().median():.4f}\n")
                f.write(f"  Max |IC_5min|: {subset['ic_5min'].abs().max():.4f}\n")

        # Top fields
        if "fields" in results.columns:
            field_counts = {}
            for fields in results.head(100)["fields"]:
                for field in str(fields).split(","):
                    field = field.strip()
                    if field:
                        field_counts[field] = field_counts.get(field, 0) + 1
            f.write(f"\nMost frequent primitives in top-100:\n")
            for field, count in sorted(field_counts.items(), key=lambda x: -x[1])[:15]:
                f.write(f"  {field}: {count}\n")

    logger.info("Summary saved to %s", summary_path)

    # Config
    config = {
        "symbol": args.symbol,
        "days": args.days,
        "freq": args.freq,
        "max_order": args.max_order,
        "max_expressions": args.max_expressions,
        "min_ic": args.min_ic,
        "ic_window": args.ic_window,
        "n_results": len(results),
    }
    with open(out_dir / "scan_config.json", "w") as f:
        json.dump(config, f, indent=2)


# ══════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Factor Scan & Evaluation")
    parser.add_argument("--symbol", default="SOL/USDT", help="Trading pair")
    parser.add_argument("--days", type=int, default=365, help="History lookback in days")
    parser.add_argument("--freq", default="1s", help="Bar frequency (1s, 5s, 1min)")
    parser.add_argument("--max-order", type=int, default=2, help="Max operator nesting (1 or 2)")
    parser.add_argument("--max-expressions", type=int, default=5000, help="Max expressions to evaluate")
    parser.add_argument("--min-ic", type=float, default=0.01, help="Min |IC| threshold for full eval")
    parser.add_argument("--ic-window", type=int, default=100, help="Rolling IC window size")
    parser.add_argument("--orderbook-path", default=None, help="Path to orderbook parquet data")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: data/factor_scan_results/)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("Starting factor scan: %s, %d days, max_order=%d", args.symbol, args.days, args.max_order)
    t0 = time.time()

    results = run_scan(args)

    elapsed = time.time() - t0
    logger.info("Scan completed in %.1f seconds, %d factors passed", elapsed, len(results))

    if results.empty:
        logger.warning("No factors passed the threshold")
        return

    # Save
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = ROOT / "data" / "factor_scan_results" / args.symbol.replace("/", "_")
    save_results(results, out_dir, args)

    # Print top 10
    print(f"\n{'=' * 80}")
    print(f"Top 10 Factors by |IC_5min| ({len(results)} total)")
    print(f"{'=' * 80}")
    for i, row in results.head(10).iterrows():
        print(f"  #{i+1}: {row['expression']}")
        print(f"       IC_1min={row.get('ic_1min',0):.4f}  IC_5min={row.get('ic_5min',0):.4f}  "
              f"Turnover={row.get('turnover',0):.4f}")
    print()


if __name__ == "__main__":
    main()
