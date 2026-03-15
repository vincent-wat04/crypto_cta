#!/usr/bin/env python3
"""
Launch rule-based strategy on live FAPI data.

Three workflows:
  1. From optimizer CSV:  load best row from grid_search_results.csv
  2. From manual args:    specify signal/factor/stoploss via CLI

Usage:
  # From optimizer best result (simulate mode)
  python scripts/run_rule_live.py --from_csv data/.../grid_search_results.csv

  # Manual params (simulate mode)
  python scripts/run_rule_live.py --factor trade_imbalance --signal zscore --signal_window 60 --signal_threshold 1.5

  # Live mode: specify position sizing and API keys
  python scripts/run_rule_live.py --from_csv .../grid_search_results.csv --mode live \
      --api_key YOUR_KEY --api_secret YOUR_SECRET \
      --leverage 2 --max_position_pct 0.05

  # Fixed USD notional position
  python scripts/run_rule_live.py --from_csv .../grid_search_results.csv --mode live \
      --position_size_usd 500 --api_key KEY --api_secret SECRET
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trading.config import RuleRunnerConfig
from trading.rule_strategy_runner import RuleStrategyRunner

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Rule strategy runner on live FAPI data")

    # Source of strategy params
    g = p.add_argument_group("Strategy source")
    g.add_argument("--from_csv", type=str, default=None,
                   help="Path to grid_search_results.csv; uses row 0 (best)")
    g.add_argument("--csv_row", type=int, default=0)

    # Manual strategy params (used when --from_csv is not provided)
    g2 = p.add_argument_group("Manual strategy params")
    g2.add_argument("--factor", default="trade_imbalance")
    g2.add_argument("--signal", default="zscore")
    g2.add_argument("--signal_window", type=int, default=60)
    g2.add_argument("--signal_threshold", type=float, default=1.5)
    g2.add_argument("--sizer", default="fixed")
    g2.add_argument("--stoploss", default="none")
    g2.add_argument("--stoploss_bps", type=float, default=30.0)
    g2.add_argument("--exit_rule", default="signal_reversal")

    # Trading config (maps to RuleRunnerConfig fields)
    g3 = p.add_argument_group("Trading config")
    g3.add_argument("--mode", choices=["simulate", "live"], default="simulate")
    g3.add_argument("--symbol", default="SOL/USDT")
    g3.add_argument("--api_key", default="")
    g3.add_argument("--api_secret", default="")
    g3.add_argument("--leverage", type=int, default=1)
    g3.add_argument("--max_position_pct", type=float, default=0.10,
                    help="Max position as %% of equity (e.g. 0.10 = 10%%)")
    g3.add_argument("--position_size_usd", type=float, default=0.0,
                    help="Fixed USD notional per signal unit (overrides pct if > 0)")
    g3.add_argument("--maker_fill_rate", type=float, default=0.7)

    # Runner config
    g4 = p.add_argument_group("Runner config")
    g4.add_argument("--bar_seconds", type=int, default=60)
    g4.add_argument("--warmup_bars", type=int, default=120)
    g4.add_argument("--duration_sec", type=int, default=0,
                    help="Run for N seconds then stop (0 = indefinite)")

    # Output
    p.add_argument("--output_dir", default=None)
    p.add_argument("--log_level", default="INFO")

    return p.parse_args()


def load_strategy_params(args) -> dict:
    """Load strategy params from CSV or CLI args."""
    if args.from_csv:
        df = pd.read_csv(args.from_csv)
        if args.csv_row >= len(df):
            raise ValueError(f"Row {args.csv_row} out of range (CSV has {len(df)} rows)")
        row = df.iloc[args.csv_row].to_dict()
        return {k: v for k, v in row.items() if pd.notna(v)}

    return {
        "factor": args.factor,
        "signal": args.signal,
        "window": args.signal_window,
        "threshold": args.signal_threshold,
        "sizer": args.sizer,
        "size_units": 1.0,
        "stoploss": args.stoploss,
        "max_loss_bps": args.stoploss_bps,
        "exit": args.exit_rule,
        "execution": "maker_first",
        "maker_fee": -2.0,
        "taker_fee": 4.0,
        "fill_rate": args.maker_fill_rate,
        "bar_seconds": args.bar_seconds,
    }


def build_config(args) -> RuleRunnerConfig:
    """Build unified RuleRunnerConfig from CLI args."""
    return RuleRunnerConfig(
        mode=args.mode,
        symbol=args.symbol,
        api_key=args.api_key,
        api_secret=args.api_secret,
        leverage=args.leverage,
        max_position_pct=args.max_position_pct,
        position_size_usd=args.position_size_usd,
        maker_fill_rate=args.maker_fill_rate,
        bar_seconds=args.bar_seconds,
        warmup_bars=args.warmup_bars,
    )


def on_signal_callback(record: dict):
    """Print each trade as it happens."""
    contracts = record.get("contracts", 0)
    print(
        f"  [{record['action']:6s}] pos={record['old_pos']:+.1f}→{record['new_pos']:+.1f} "
        f"({contracts:.4f} contracts) @ {record['price']:.4f} | "
        f"net={record['net_pnl_bps']:+.1f} bps | cum={record['cum_pnl_bps']:+.1f} bps"
    )


async def run_with_duration(runner: RuleStrategyRunner, duration_sec: int):
    if duration_sec > 0:
        try:
            await asyncio.wait_for(runner.run(), timeout=duration_sec)
        except asyncio.TimeoutError:
            logger.info("Duration reached (%d sec), stopping.", duration_sec)
            runner.stop()
    else:
        await runner.run()


def save_results(runner: RuleStrategyRunner, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    results = runner.get_results()

    trade_df = runner.get_trade_log_df()
    if not trade_df.empty:
        csv_path = output_dir / "live_trades.csv"
        trade_df.to_csv(csv_path, index=False)
        logger.info("Trade log saved to %s", csv_path)

    metrics_path = output_dir / "live_metrics.json"
    metrics = {k: v for k, v in results.items() if k != "trade_log"}
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Metrics saved to %s", metrics_path)


def main():
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    strategy_params = load_strategy_params(args)
    config = build_config(args)

    logger.info("Strategy params: %s", strategy_params)
    logger.info("Config: mode=%s, symbol=%s, leverage=%d, max_pos_pct=%.0f%%, pos_usd=%.0f",
                config.mode, config.symbol, config.leverage,
                config.max_position_pct * 100, config.position_size_usd)

    runner = RuleStrategyRunner.from_backtest_params(
        params=strategy_params,
        config=config,
        on_signal=on_signal_callback,
    )

    # Graceful shutdown
    loop = asyncio.new_event_loop()

    def _shutdown(sig, _frame):
        logger.info("Caught signal %s, shutting down...", sig)
        runner.stop()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    factor = strategy_params.get("factor", "?")
    sig = strategy_params.get("signal", "?")
    print(f"\n{'='*60}")
    print(f"Rule Strategy Runner [{args.mode.upper()} mode]")
    print(f"  Symbol:      {args.symbol}")
    print(f"  Factor:      {factor}")
    print(f"  Signal:      {sig}")
    print(f"  Bar:         {args.bar_seconds}s")
    print(f"  Warmup:      {args.warmup_bars} bars")
    print(f"  Leverage:    {args.leverage}x")
    if args.position_size_usd > 0:
        print(f"  Position:    ${args.position_size_usd:.0f} USD notional per unit")
    else:
        print(f"  Position:    {args.max_position_pct*100:.0f}% of equity")
    if args.duration_sec > 0:
        print(f"  Duration:    {args.duration_sec}s")
    else:
        print("  Duration:    indefinite (Ctrl+C to stop)")
    print(f"{'='*60}\n")

    try:
        loop.run_until_complete(run_with_duration(runner, args.duration_sec))
    finally:
        results = runner.get_results()
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"  Bars processed:    {results['n_bars']}")
        print(f"  Signals generated: {results['n_signals']}")
        print(f"  Trades executed:   {results['n_trades']}")
        print(f"  Current position:  {results['current_position']}")
        print(f"  Cumulative PnL:    {results['cumulative_pnl_bps']:+.2f} bps")
        acct = results.get("account", {})
        if acct.get("balance_usdt"):
            print(f"  Account balance:   {acct['balance_usdt']:.2f} USDT")
            print(f"  Position size:     {acct['position_size_contracts']:.4f} contracts")
        if results.get("metrics"):
            m = results["metrics"]
            print(f"  Sharpe:            {m.get('sharpe_ratio', 0):.4f}")
            print(f"  Win Rate:          {m.get('win_rate', 0):.1f}%")
        print(f"{'='*60}")

        if args.output_dir:
            out = Path(args.output_dir)
        else:
            out = ROOT / "data" / "rule_live_results" / args.symbol.replace("/", "_")
        save_results(runner, out)


if __name__ == "__main__":
    main()
