#!/usr/bin/env python3
"""
Fetch Binance USDC-M perpetual aggTrades and cache to parquet.

Uses MR_DATA_ROOT (see utilities.paths.DataPaths) for cache layout:
  $MR_DATA_ROOT/cache/{SYMBOL}/trades_perp/{YYYY-MM-DD}.parquet

Usage:
  python scripts/fetch_perpetual_trades.py --symbol SOL/USDC --days 14
  python scripts/fetch_perpetual_trades.py --start-date 2026-01-01 \\
      --symbols SOL/USDC,BTC/USDC,ETH/USDC

Note: end_date is exclusive in the loader; default end is tomorrow so \"today\" is included.

macOS background (no Terminal): scripts/fetch_perp_background.sh (see scripts/fetch_perp_stop.sh,
optional scripts/fetch_perp_background.launchd.plist.example for launchd).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utilities.binance_loader import load_agg_trades_perpetual

logger = logging.getLogger(__name__)


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default=None, help="Single symbol, e.g. SOL/USDC")
    p.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbols (overrides --symbol if set)",
    )
    p.add_argument("--days", type=int, default=None, help="Rolling window ending today (optional)")
    p.add_argument(
        "--start-date",
        default=None,
        help="Inclusive start (YYYY-MM-DD). Implies calendar range mode.",
    )
    p.add_argument(
        "--end-date",
        default=None,
        help="Exclusive end (YYYY-MM-DD). Default: tomorrow so today is fully fetched.",
    )
    p.add_argument(
        "--pause-between-symbols",
        type=float,
        default=2.0,
        help="Seconds to sleep between symbols (reduces 429 bursts).",
    )
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    elif args.symbol:
        symbols = [args.symbol]
    else:
        symbols = ["SOL/USDC"]

    if args.start_date is not None:
        start_d = _parse_date(args.start_date)
        if args.end_date is not None:
            end_d = _parse_date(args.end_date)
        else:
            end_d = date.today() + timedelta(days=1)
        range_desc = f"{start_d.isoformat()} .. (exclusive) {end_d.isoformat()}"
        use_days = False
    elif args.days is not None:
        start_d = None
        end_d = None
        use_days = True
        range_desc = f"last {args.days} days"
    else:
        args.days = 14
        start_d = None
        end_d = None
        use_days = True
        range_desc = f"last {args.days} days"

    print(f"Symbols: {symbols}")
    print(f"Range: {range_desc}")
    print()

    for sym in symbols:
        print(f"=== {sym} ===")
        t0 = time.time()
        if use_days:
            df = load_agg_trades_perpetual(symbol=sym, days=args.days)
        else:
            df = load_agg_trades_perpetual(
                symbol=sym,
                start_date=start_d,
                end_date=end_d,
            )
        elapsed = time.time() - t0

        if df.empty:
            print("  No rows loaded (missing cache + fetch failed or empty range).\n")
        else:
            print(
                f"  Rows: {len(df):,}  elapsed {elapsed:.0f}s ({elapsed/60:.1f} min)\n"
                f"  Period: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}\n"
            )
        if args.pause_between_symbols > 0 and sym != symbols[-1]:
            time.sleep(args.pause_between_symbols)


if __name__ == "__main__":
    main()
