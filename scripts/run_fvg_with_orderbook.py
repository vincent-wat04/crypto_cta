#!/usr/bin/env python3
"""
FVG V2 策略：方向预测 + Orderbook 特征 + 网格参数搜索。

数据来源：
  - Trades:    Binance aggTrades（含 taker 聚合信息）
  - Orderbook: lake-api BTC-USDT sample (2022-10-01 ~ 2022-10-03)

运行：
    .venv/bin/python scripts/run_fvg_with_orderbook.py
    .venv/bin/python scripts/run_fvg_with_orderbook.py --use-binance --symbol SOL/USDT --days 3
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from utilities.binance_loader import load_agg_trades, resample_trades_to_ohlcv
from utilities.lakeapi_loader import load_lakeapi_orderbook, load_lakeapi_trades
from utilities.paths import DataPaths
from cta.fvg.detector import detect_fvg, FVGStatus
from cta.fvg.features import build_fvg_feature_matrix, get_feature_columns
from cta.fvg.orderbook_features import add_orderbook_features, get_ob_feature_columns
from cta.backtest import param_grid_search


def main():
    parser = argparse.ArgumentParser(description="FVG V2 with Orderbook")
    parser.add_argument("--use-binance", action="store_true",
                        help="Use Binance aggTrades instead of lake-api trades")
    parser.add_argument("--symbol", default="SOL/USDT")
    parser.add_argument("--days", type=int, default=3)
    args = parser.parse_args()

    print("=" * 70)
    print("FVG STRATEGY V2 — OPTIMIZED")
    print("  Target: price direction prediction")
    print("  Enhancements: stop-loss, take-profit, cooldown, param search")
    print("=" * 70)

    # ─── 1. 加载数据 ───
    if args.use_binance:
        print(f"\n[1] Loading Binance aggTrades ({args.symbol}, {args.days}d)...")
        trades = load_agg_trades(symbol=args.symbol, days=args.days)
        print(f"  {len(trades):,} aggTrades")
        if "n_trades_in_agg" in trades.columns:
            print(f"  Taker info: mean={trades['n_trades_in_agg'].mean():.1f} trades/agg")
        book = None
    else:
        print("\n[1] Loading lake-api data (BTC-USDT sample)...")
        trades = load_lakeapi_trades(days=args.days, symbol="BTC-USDT")
        book = load_lakeapi_orderbook(days=args.days, symbol="BTC-USDT")
        print(f"  Trades: {len(trades):,}, Book: {len(book):,} snapshots")

    # ─── 2. Resample OHLCV ───
    print("\n[2] Resampling to OHLCV...")
    timeframes = {"1min": "1min", "3min": "3min", "5min": "5min"}
    ohlcv_dict = {}
    for label, freq in timeframes.items():
        ohlcv_dict[label] = resample_trades_to_ohlcv(trades, freq=freq)
        print(f"  {label}: {len(ohlcv_dict[label])} bars")

    # ─── 3. 多频率 FVG 检测 ───
    print("\n[3] Detecting FVGs across timeframes...")
    best_tf, best_gap, best_count = None, None, 0

    for tf, ohlcv_tf in ohlcv_dict.items():
        for min_gap in [0.01, 0.02, 0.03]:
            fvgs = detect_fvg(ohlcv_tf, min_gap_pct=min_gap)
            filled = sum(1 for f in fvgs if f.status == FVGStatus.FILLED)
            print(f"  {tf}, gap>{min_gap}%: {len(fvgs)} FVGs ({filled} filled)")
            if len(fvgs) > best_count and len(fvgs) >= 30:
                best_count = len(fvgs)
                best_tf, best_gap = tf, min_gap

    if best_tf is None:
        best_tf, best_gap = "1min", 0.01

    ohlcv = ohlcv_dict[best_tf]
    fvgs = detect_fvg(ohlcv, min_gap_pct=best_gap)
    print(f"\n  Selected: {best_tf}, gap>{best_gap}%, {len(fvgs)} FVGs")

    # ─── 4. 构建特征 ───
    print("\n[4] Building trade features...")
    feat_df = build_fvg_feature_matrix(fvgs, trades, ohlcv)
    print(f"  Feature matrix: {feat_df.shape}")

    if len(feat_df) < 30:
        print("  Too few samples!")
        return 1

    # ─── 5. Orderbook 特征（如可用）───
    if book is not None:
        print("\n[5] Adding orderbook features...")
        feat_df_ob = add_orderbook_features(feat_df, book, window_before_ms=10000, window_after_ms=5000)
        print(f"  Enhanced: {feat_df_ob.shape}")
    else:
        print("\n[5] No orderbook data — skipping OB features")
        feat_df_ob = feat_df

    # 缓存
    cache_path = DataPaths.data / "cache" / "fvg_features_with_ob.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    feat_df_ob.to_csv(cache_path, index=False)

    # ─── 6. Walk-forward 多目标搜索 ───
    print("\n[6] Walk-forward optimization...")
    split_idx = int(len(feat_df_ob) * 0.7)
    train = feat_df_ob.iloc[:split_idx]
    test = feat_df_ob.iloc[split_idx:]
    print(f"  Train: {len(train)}, Test: {len(test)}")

    original_cols = get_feature_columns()
    ob_cols = get_ob_feature_columns() if book is not None else []
    all_cols = (
        [c for c in original_cols if c in train.columns]
        + [c for c in ob_cols if c in train.columns]
    )

    targets = {
        "direction_5":  "label_direction_5",
        "direction_10": "label_direction_10",
        "direction_20": "label_direction_20",
        "fill":         "label_filled",
    }

    all_results = {}
    for target_name, target_col in targets.items():
        if target_col not in train.columns:
            print(f"\n  SKIP {target_name}: column missing")
            continue

        print(f"\n  --- Target: {target_name} ({target_col}) ---")
        result = param_grid_search(train, test, all_cols, ohlcv, target_col)

        if "error" in result:
            print(f"    FAILED: {result['error']}")
            continue

        report = result["report"]
        bt = result["best_result"]

        print(f"    Train accuracy: {report.get('train_accuracy', 0)}%")
        print(f"    CV accuracy:    {report.get('cv_accuracy', 0)}%")
        print(f"    Best params:    {result['best_params']}")
        print(f"    Signals: {bt.get('n_signals', 0)}, Win rate: {bt.get('win_rate', 0)}%")
        print(f"    Return: {bt.get('total_return', 0)}%, Sharpe: {bt.get('sharpe', 0)}")
        print(f"    Max DD: {bt.get('max_drawdown', 0)}%, PF: {bt.get('profit_factor', 0)}")

        if "results_df" in bt and not bt["results_df"].empty:
            rdf = bt["results_df"]
            print(f"    Exit reasons: {dict(rdf['exit_reason'].value_counts())}")

        all_results[target_name] = result

    # ─── 7. 最终对比 ───
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    header = f"{'Target':<18} {'Signals':>8} {'WinRate':>8} {'Return':>8} {'Sharpe':>8} {'MaxDD':>8} {'PF':>8}"
    print(header)
    print("-" * len(header))

    best_target, best_sharpe = None, -999
    for tname, res in all_results.items():
        bt = res["best_result"]
        sharpe = bt.get("sharpe", 0)
        print(
            f"{tname:<18} {bt.get('n_signals', 0):>8} "
            f"{bt.get('win_rate', 0):>7}% {bt.get('total_return', 0):>7}% "
            f"{sharpe:>8} {bt.get('max_drawdown', 0):>7}% {bt.get('profit_factor', 0):>8}"
        )
        if sharpe > best_sharpe:
            best_sharpe, best_target = sharpe, tname

    # ─── 8. 保存最优模型 ───
    if best_target and best_target in all_results:
        winner = all_results[best_target]
        print(f"\n  BEST: {best_target} (Sharpe={best_sharpe})")
        print(f"  Params: {winner['best_params']}")

        model_path = DataPaths.model("fvg_v2", "best_model.pkl")
        winner["model"].save(model_path)
        print(f"  Model saved: {model_path}")

        bt = winner["best_result"]
        if "results_df" in bt and not bt["results_df"].empty:
            bt_path = DataPaths.backtest_result("fvg_v2", "latest")
            bt["results_df"].to_csv(bt_path, index=False)

        if winner.get("search_log") is not None and len(winner["search_log"]) > 0:
            log_path = DataPaths.backtest_result("fvg_v2", "param_search")
            winner["search_log"].sort_values("score", ascending=False).to_csv(log_path, index=False)

    return 0


if __name__ == "__main__":
    exit(main() or 0)
