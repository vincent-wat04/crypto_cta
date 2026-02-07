#!/usr/bin/env python3
"""
微观结构反转策略回测。

运行：
    .venv/bin/python scripts/run_microstructure_backtest.py --days 5 --symbol SOL/USDT
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from utilities.binance_loader import load_agg_trades
from utilities.paths import DataPaths
from cta.microstructure.price_impact import compute_tick_price_impact, detect_large_impact_events
from cta.microstructure.spread_estimator import estimate_spread_from_trades, compute_spread_percentile
from cta.microstructure.order_flow import compute_trade_imbalance, compute_vpin, detect_aggressive_flow
from cta.microstructure.liquidity_detector import generate_reversal_signals


def compute_microstructure_features(trades: pd.DataFrame, window_ms: int = 5000) -> pd.DataFrame:
    """计算所有微观结构特征。"""
    print("[Features] Computing price impact...")
    impact = compute_tick_price_impact(trades, window_ms=window_ms)

    print("[Features] Estimating spread...")
    spread = estimate_spread_from_trades(trades, method="trade_diff", window_trades=100)
    spread_pct = compute_spread_percentile(spread, lookback=500)

    print("[Features] Computing order flow...")
    imbalance = compute_trade_imbalance(trades, window_trades=100)
    vpin_s = compute_vpin(trades, bucket_size=1000, n_buckets=50)

    if impact.empty:
        return pd.DataFrame()

    impact = impact.set_index("timestamp")
    impact = impact[~impact.index.duplicated(keep="last")]

    for s in [spread, spread_pct, imbalance, vpin_s]:
        s.drop(s.index[s.index.duplicated(keep="last")], inplace=True)

    impact["spread"] = spread.reindex(impact.index, method="ffill")
    impact["spread_percentile"] = spread_pct.reindex(impact.index, method="ffill")
    impact["imbalance"] = imbalance.reindex(impact.index, method="ffill")
    impact["vpin"] = vpin_s.reindex(impact.index, method="ffill")

    return impact


def backtest_microstructure(
    trades: pd.DataFrame, params: Dict[str, Any], hold_ms: int = 60000,
) -> Dict[str, Any]:
    """回测微观结构反转策略。"""
    print(f"\n[Backtest] Params: {params}")
    signals_df = generate_reversal_signals(trades, params)

    if signals_df.empty:
        print("[Backtest] No signals!")
        return {"n_signals": 0, "win_rate": 0, "total_return": 0, "sharpe": 0}

    print(f"[Backtest] {len(signals_df)} signals")

    trades_idx = trades.set_index("timestamp")
    trades_idx = trades_idx[~trades_idx.index.duplicated(keep="last")]
    results = []

    for _, sig in signals_df.iterrows():
        entry_t = sig["timestamp"]
        direction = sig["direction"]
        exit_t = entry_t + pd.Timedelta(milliseconds=hold_ms)

        after_entry = trades_idx.loc[trades_idx.index >= entry_t]
        if after_entry.empty:
            continue
        entry_p = after_entry["price"].iloc[0]

        after_exit = trades_idx.loc[trades_idx.index >= exit_t]
        exit_p = after_exit["price"].iloc[0] if not after_exit.empty else after_entry["price"].iloc[-1]

        ret = ((exit_p - entry_p) / entry_p * 100) if direction == 1 else ((entry_p - exit_p) / entry_p * 100)
        results.append({"entry_time": entry_t, "direction": direction,
                        "entry_price": entry_p, "exit_price": exit_p,
                        "return_pct": ret, "is_win": ret > 0})

    if not results:
        return {"n_signals": len(signals_df), "win_rate": 0, "total_return": 0, "sharpe": 0}

    rdf = pd.DataFrame(results)
    n = len(rdf)
    wins = rdf[rdf["is_win"]]["return_pct"]
    losses = rdf[~rdf["is_win"]]["return_pct"]

    return {
        "n_signals": n,
        "n_wins": int(rdf["is_win"].sum()),
        "win_rate": round(rdf["is_win"].mean() * 100, 2),
        "total_return": round(rdf["return_pct"].sum(), 4),
        "avg_return": round(rdf["return_pct"].mean(), 4),
        "avg_win": round(wins.mean(), 4) if len(wins) > 0 else 0,
        "avg_loss": round(losses.mean(), 4) if len(losses) > 0 else 0,
        "profit_factor": round(abs(wins.sum() / (losses.sum() + 1e-10)), 2) if len(losses) > 0 else float("inf"),
        "sharpe": round(rdf["return_pct"].mean() / (rdf["return_pct"].std() + 1e-10) * np.sqrt(n), 3),
        "results_df": rdf,
    }


def main():
    parser = argparse.ArgumentParser(description="Microstructure Reversal Backtest")
    parser.add_argument("--days", type=int, default=5)
    parser.add_argument("--symbol", default="SOL/USDT")
    parser.add_argument("--hold-ms", type=int, default=60000)
    parser.add_argument("--optimize", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("MICROSTRUCTURE REVERSAL STRATEGY")
    print("=" * 70)
    print(f"Symbol: {args.symbol}, Days: {args.days}")

    print("\n[Data] Loading Binance aggTrades...")
    trades = load_agg_trades(symbol=args.symbol, days=args.days)
    if trades.empty:
        print("ERROR: No trades!"); return 1
    print(f"  {len(trades):,} trades, {trades['timestamp'].min()} ~ {trades['timestamp'].max()}")

    print("\n[Analysis] Computing features...")
    features = compute_microstructure_features(trades)
    if not features.empty:
        print(f"  Impact(mean): {features['impact'].mean():.6f}")
        print(f"  Spread%(mean): {features['spread_percentile'].mean():.2f}")
        print(f"  Imbalance(abs mean): {features['imbalance'].abs().mean():.4f}")

    if args.optimize:
        param_sets = [
            {"impact_threshold_pct": 0.3, "spread_percentile_threshold": 85.0,
             "imbalance_threshold": 0.6, "refill_window_ms": 30000,
             "spread_recovery_threshold": 50.0, "price_hold_tolerance_pct": 0.15},
            {"impact_threshold_pct": 0.15, "spread_percentile_threshold": 70.0,
             "imbalance_threshold": 0.4, "refill_window_ms": 30000,
             "spread_recovery_threshold": 60.0, "price_hold_tolerance_pct": 0.2},
            {"impact_threshold_pct": 0.1, "spread_percentile_threshold": 65.0,
             "imbalance_threshold": 0.35, "refill_window_ms": 15000,
             "spread_recovery_threshold": 65.0, "price_hold_tolerance_pct": 0.3},
        ]
        best_score, best_result = -999, None
        for params in param_sets:
            result = backtest_microstructure(trades, params, hold_ms=args.hold_ms)
            score = result["win_rate"] / 100 * 0.4 + max(0, result.get("sharpe", 0)) * 0.3
            if score > best_score:
                best_score, best_result = score, result
    else:
        params = {"impact_threshold_pct": 0.15, "spread_percentile_threshold": 70.0,
                  "imbalance_threshold": 0.4, "refill_window_ms": 30000,
                  "spread_recovery_threshold": 60.0, "price_hold_tolerance_pct": 0.2}
        best_result = backtest_microstructure(trades, params, hold_ms=args.hold_ms)

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    for k in ["n_signals", "win_rate", "total_return", "avg_return", "sharpe", "profit_factor"]:
        print(f"  {k}: {best_result.get(k, 0)}")

    if "results_df" in best_result and not best_result["results_df"].empty:
        p = DataPaths.backtest_result("microstructure", "latest")
        best_result["results_df"].to_csv(p, index=False)
        print(f"\n  Saved to {p}")

    return 0


if __name__ == "__main__":
    exit(main() or 0)
