#!/usr/bin/env python3
"""
SOL V5 Perpetual — Train GBM on FAPI perpetual aggTrades data.

Replicates the Sharpe 2.66 model architecture (5min return regression, GBM)
using Binance USDT-M Perpetual data instead of spot.

Includes:
  - Comprehensive investment metrics (returns, ann. returns, max DD, Sharpe, Sortino, Calmar)
  - Maker-first-then-taker cost simulation
  - Model persistence for real-time deployment

Usage:
    python scripts/train_sol_v5_perpetual.py
    python scripts/train_sol_v5_perpetual.py --symbol BTC/USDT
    python scripts/train_sol_v5_perpetual.py --maker-fill 0.5
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import pickle
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trading.cost_model import CostModel
from trading.metrics import compute_investment_metrics, format_metrics_report
from trading.paper_backtest import backtest_maker_first_taker
from utilities.binance_loader import load_agg_trades_perpetual, resample_trades_to_ohlcv
from utilities.paths import DataPaths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ─── Config (same as V5 Sharpe 2.66 winning architecture) ───
TRAIN_DATES = [
    "2026-01-22", "2026-01-23", "2026-01-24", "2026-01-25",
    "2026-01-26", "2026-01-27", "2026-01-28", "2026-01-29",
    "2026-01-30", "2026-01-31", "2026-02-01", "2026-02-02", "2026-02-03",
]
TEST_DATES = ["2026-02-04", "2026-02-05", "2026-02-06"]
INTERVAL = 300  # 5min
FEATURE_WINDOWS = [3, 5, 10, 20]


def load_features_v5_perpetual(symbol: str, dates: list[str]) -> pd.DataFrame:
    """Load V5 features from FAPI perpetual aggTrades (real data, not testnet)."""
    from scripts.train_sol_v5 import compute_all_features_v5

    all_dfs = []
    for d_str in dates:
        dt = date.fromisoformat(d_str)
        cache_path = DataPaths.features(symbol, "1s_features_v5_perp", dt)
        if cache_path.exists():
            logger.info("  [Cache] %s (perp)", d_str)
            all_dfs.append(pd.read_parquet(cache_path))
        else:
            logger.info("  [Compute] %s perp...", d_str)
            trades = load_agg_trades_perpetual(
                symbol, start_date=dt, end_date=dt + timedelta(days=1)
            )
            if trades.empty:
                logger.warning("  [Skip] %s — no perp data", d_str)
                continue
            logger.info("    %d aggTrades loaded", len(trades))
            df = compute_all_features_v5(trades, "1s")
            if df.empty:
                del trades
                gc.collect()
                continue
            df.to_parquet(cache_path)
            all_dfs.append(df)
            del trades
            gc.collect()
    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs).sort_index()


def main():
    parser = argparse.ArgumentParser(description="Train V5 GBM on perpetual data")
    parser.add_argument("--symbol", default="SOL/USDT")
    parser.add_argument("--maker-fill", type=float, default=0.7,
                        help="Maker fill rate for cost simulation (0-1)")
    parser.add_argument("--maker-bps", type=float, default=-2.0,
                        help="Maker fee in bps (negative = rebate)")
    parser.add_argument("--taker-bps", type=float, default=4.0,
                        help="Taker fee in bps")
    parser.add_argument("--top-features", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Signal threshold for position entry")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("SOL V5 Perpetual — GBM 5min Return Regression")
    logger.info("  Data: Binance FAPI perpetual aggTrades (real)")
    logger.info("  Architecture: Sharpe 2.66 replica (GBM, 5min interval)")
    logger.info("  Maker rebate: %.1f bps, Taker fee: %.1f bps", args.maker_bps, args.taker_bps)
    logger.info("  Maker fill rate: %.0f%%", args.maker_fill * 100)
    logger.info("=" * 70)

    # ── 1. Load features ──
    logger.info("\n[1] Loading V5 features from perpetual aggTrades...")
    t0 = time.time()
    train_1s = load_features_v5_perpetual(args.symbol, TRAIN_DATES)
    test_1s = load_features_v5_perpetual(args.symbol, TEST_DATES)

    if train_1s.empty or test_1s.empty:
        logger.error("Failed to load features!")
        return 1

    logger.info("  Train: %d bars x %d cols (%.1fs)", len(train_1s), train_1s.shape[1], time.time() - t0)
    logger.info("  Test:  %d bars x %d cols", len(test_1s), test_1s.shape[1])

    # ── 2. Build targets ──
    logger.info("\n[2] Building 5min return targets...")
    train_positions = np.arange(0, len(train_1s), INTERVAL)
    test_positions = np.arange(0, len(test_1s), INTERVAL)

    cum_train = train_1s["return_1"].cumsum().values
    cum_test = test_1s["return_1"].cumsum().values
    y_train = np.array([
        cum_train[min(i + INTERVAL, len(cum_train) - 1)] - cum_train[i]
        for i in train_positions
    ])
    y_test_actual = np.array([
        cum_test[min(i + INTERVAL, len(cum_test) - 1)] - cum_test[i]
        for i in test_positions
    ])
    y_train[train_positions + INTERVAL >= len(cum_train)] = np.nan
    y_test_actual[test_positions + INTERVAL >= len(cum_test)] = np.nan

    train_mask = ~np.isnan(y_train)
    test_mask = ~np.isnan(y_test_actual)
    y_tr = y_train[train_mask]
    y_te_actual = y_test_actual[test_mask]
    test_ts = test_1s.index[test_positions[test_mask]]

    logger.info("  Train samples: %d, Test samples: %d", len(y_tr), len(y_te_actual))

    # ── 3. Feature selection ──
    logger.info("\n[3] Feature selection...")
    exclude = {"return_1", "log_return", "close"}
    feat_cols = [
        c for c in train_1s.columns
        if c not in exclude
        and train_1s[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]
        and train_1s[c].std() > 1e-10
    ]
    logger.info("  Candidate features: %d", len(feat_cols))

    from scripts.train_sol_v5 import select_features
    selected, importance = select_features(
        train_1s.iloc[train_positions[train_mask]][feat_cols].values,
        y_tr.astype(float),
        feat_cols,
        top_n=args.top_features,
        task="reg",
    )
    logger.info("  Selected features: %d", len(selected))

    X_train = train_1s.iloc[train_positions[train_mask]][selected].values
    X_test = test_1s.iloc[test_positions[test_mask]][selected].values

    # ── 4. Train GBM ──
    logger.info("\n[4] Training HistGradientBoostingRegressor...")
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingRegressor

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    model = HistGradientBoostingRegressor(
        max_iter=300, max_depth=3, learning_rate=0.03,
        min_samples_leaf=20, l2_regularization=5.0,
        early_stopping=True, validation_fraction=0.2,
        n_iter_no_change=20, random_state=42,
    ).fit(Xtr, y_tr)

    pred_tr = model.predict(Xtr)
    ss_res = np.sum((y_tr - pred_tr) ** 2)
    ss_tot = np.sum((y_tr - y_tr.mean()) ** 2)
    r2_train = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    signal = model.predict(Xte)
    ic_mask = ~(np.isnan(signal) | np.isnan(y_te_actual))
    ic = np.corrcoef(signal[ic_mask], y_te_actual[ic_mask])[0, 1] if ic_mask.sum() > 10 else 0

    logger.info("  Train R²: %.4f", r2_train)
    logger.info("  Test IC:  %.4f", ic)
    logger.info("  Signal:   mu=%.4f sigma=%.4f", signal.mean(), signal.std())

    # ── 5. Backtest with multiple cost scenarios ──
    logger.info("\n[5] Backtesting with maker-first-then-taker costs...")
    cost_model = CostModel(maker_fee_bps=args.maker_bps, taker_fee_bps=args.taker_bps)
    periods_per_year = 365 * 24 * 3600 / INTERVAL

    scenarios = [
        ("maker_70pct", 0.7, 0.0),
        ("maker_50pct", 0.5, 0.0),
        ("taker_only", 0.0, 0.0),
        ("maker_70pct_thr0.5", 0.7, 0.5),
        ("maker_70pct_thr1.0", 0.7, 1.0),
    ]

    all_results = {}
    for name, mfr, thr in scenarios:
        df = backtest_maker_first_taker(
            signal=signal, actual_return=y_te_actual, timestamps=test_ts,
            cost_model=cost_model, threshold=thr, maker_fill_rate=mfr,
        )
        metrics = compute_investment_metrics(df["net_pnl"], periods_per_year=periods_per_year)
        all_results[name] = metrics

        logger.info(
            "  %-25s  Net=%+8.0f bps  SR=%+6.2f  WR=%.1f%%  MaxDD=%.2f%%  Ann=%.2f%%",
            name, metrics["net_pnl_bps"], metrics["sharpe_ratio"],
            metrics["win_rate"], metrics["max_drawdown_pct"],
            metrics["annualized_return_pct"],
        )

    # Best scenario
    best_name = max(all_results, key=lambda k: all_results[k]["sharpe_ratio"])
    best_metrics = all_results[best_name]
    logger.info("\n  ★ Best: %s (Sharpe=%.2f)", best_name, best_metrics["sharpe_ratio"])
    logger.info("\n" + format_metrics_report(best_metrics))

    # ── 6. Save model + artifacts ──
    logger.info("\n[6] Saving model and artifacts...")
    out_dir = DataPaths.backtest_dir("sol_v5_perpetual")
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "model": model,
        "scaler": scaler,
        "selected_features": selected,
        "feature_importance": importance,
        "config": {
            "symbol": args.symbol,
            "interval": INTERVAL,
            "feature_windows": FEATURE_WINDOWS,
            "threshold": args.threshold,
            "train_dates": TRAIN_DATES,
            "test_dates": TEST_DATES,
            "maker_fee_bps": args.maker_bps,
            "taker_fee_bps": args.taker_bps,
        },
    }

    model_path = out_dir / "model_artifacts.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(artifacts, f)
    logger.info("  Model saved: %s", model_path)

    # Save backtest trades for best scenario
    best_mfr = dict(scenarios)[best_name.rsplit("_thr", 1)[0]] if "_thr" in best_name else 0.7
    best_thr = float(best_name.split("thr")[-1]) if "thr" in best_name else 0.0
    # Re-run best for trade log
    df_best = backtest_maker_first_taker(
        signal=signal, actual_return=y_te_actual, timestamps=test_ts,
        cost_model=cost_model, threshold=best_thr, maker_fill_rate=0.7,
    )
    df_best.to_csv(out_dir / "trades.csv", index=False)

    # Save all metrics
    results_json = {
        "train_r2": float(r2_train),
        "test_ic": float(ic),
        "n_train": len(y_tr),
        "n_test": len(y_te_actual),
        "n_features": len(selected),
        "best_scenario": best_name,
        "scenarios": {k: v for k, v in all_results.items()},
        "top_features": importance,
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, default=str)
    logger.info("  Results saved: %s", out_dir / "results.json")

    logger.info("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
