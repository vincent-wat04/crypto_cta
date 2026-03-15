#!/usr/bin/env python3
"""
Fetch Binance USDC-M perpetual aggTrades and cache to parquet.

Usage:
  python scripts/fetch_perpetual_trades.py --symbol SOL/USDC --days 14
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utilities.binance_loader import load_agg_trades_perpetual

logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="SOL/USDC")
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"Fetching {args.days} days of {args.symbol} perpetual aggTrades...")
    print(f"Estimated time: ~{args.days * 2}-{args.days * 3} minutes")
    print()

    t0 = time.time()
    df = load_agg_trades_perpetual(symbol=args.symbol, days=args.days)
    elapsed = time.time() - t0

    if df.empty:
        print("No data fetched!")
        return

    print(f"\nFetched {len(df):,} trades in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Period: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")


if __name__ == "__main__":
    main()
