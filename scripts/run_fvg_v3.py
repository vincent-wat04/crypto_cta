#!/usr/bin/env python3
"""
FVG V3 策略：全面指标 + Triple Barrier + Grid Search

Pipeline:
  1. 加载数据 (aggTrades + klines + 可选 orderbook)
  2. Taker order 分组
  3. FVG 检测 (多频段)
  4. V3 特征工程 (indicators/ 库)
  5. Triple Barrier 标签
  6. 模型训练 (聚类 + GBM)
  7. Grid Search 参数优化
  8. 回测 + 保存结果

Usage:
    # Triple Barrier 目标（默认）
    python scripts/run_fvg_v3.py --symbol BTC/USDT --start-date 2022-10-01 --end-date 2022-10-04 --use-orderbook
    # 原始标签（适合 TB 标签不均衡时）
    python scripts/run_fvg_v3.py --symbol BTC/USDT --start-date 2022-10-01 --end-date 2022-10-04 --use-orderbook --target fill_direction
    python scripts/run_fvg_v3.py --symbol BTC/USDT --start-date 2022-10-01 --end-date 2022-10-04 --use-orderbook --target future_ret_10
    # Grid search
    python scripts/run_fvg_v3.py --symbol BTC/USDT --start-date 2022-10-01 --end-date 2022-10-04 --use-orderbook --grid-search
    # 近期数据（无 orderbook）
    python scripts/run_fvg_v3.py --symbol SOL/USDT --days 3 --freq 1min
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, timedelta
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utilities.binance_loader import load_agg_trades, resample_trades_to_ohlcv
from utilities.paths import DataPaths
from indicators.base.taker_flow import group_taker_orders
from cta.fvg.detector import detect_fvg
from cta.fvg.features_v3 import (
    build_fvg_features_v3,
    add_triple_barrier_labels,
    get_v3_feature_columns,
)
from backtest.metrics import compute_backtest_metrics
from backtest.labeling import validate_labels
from backtest.dynamic_exit import atr_exit_params, simulate_exit

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Optional
try:
    from utilities.lakeapi_loader import load_lakeapi_orderbook
    HAS_LAKEAPI = True
except ImportError:
    HAS_LAKEAPI = False


# ─────────────────────────────────────────────────────────
# Model: V3 uses GBM with auto feature selection
# ─────────────────────────────────────────────────────────

def train_v3_model(feature_df, target_col, feature_cols, log_dir=None):
    """
    Train GBM classifier with:
      - 降低的模型复杂度（防过拟合）
      - Train/Validation 时序分割
      - Early stopping（通过 n_iter_no_change）
      - 概率校准（CalibratedClassifierCV）
      - 训练日志保存
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import (
        classification_report, log_loss, roc_auc_score, brier_score_loss,
    )
    from sklearn.calibration import CalibratedClassifierCV

    X = feature_df[feature_cols].fillna(0).values
    y = feature_df[target_col].values.astype(int)

    # Remove constant / NaN-heavy columns
    valid_cols = []
    for i in range(len(feature_cols)):
        if np.std(X[:, i]) > 1e-10 and np.isfinite(X[:, i]).mean() > 0.5:
            valid_cols.append(i)
    X = X[:, valid_cols]
    valid_feature_names = [feature_cols[i] for i in valid_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Train / Validation split (时序，最后 20% 作 val) ──
    val_size = max(int(len(X) * 0.2), 10)
    X_train, X_val = X_scaled[:-val_size], X_scaled[-val_size:]
    y_train, y_val = y[:-val_size], y[-val_size:]
    logger.info(f"  Train: {len(X_train)}, Validation: {len(X_val)}")

    # ── 降低模型复杂度 + early stopping ──
    model = GradientBoostingClassifier(
        n_estimators=300,           # 上限，配合 early stopping
        max_depth=3,                # 降低 (was 5)
        learning_rate=0.05,         # 降低 (was 0.1)
        subsample=0.7,              # 降低 (was 0.8)
        min_samples_leaf=20,        # 新增：防过拟合
        min_samples_split=30,       # 新增
        max_features="sqrt",        # 新增：随机特征子集
        n_iter_no_change=15,        # Early stopping: 15 轮无改善则停止
        validation_fraction=0.15,   # 用于 early stopping 的验证集比例
        tol=1e-4,
        random_state=42,
    )

    # ── CV (时序) ──
    n_splits = min(5, len(X_train) // 20)
    cv_scores = {"accuracy": [], "log_loss": []}
    if n_splits >= 2:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for fold_i, (tr_idx, te_idx) in enumerate(tscv.split(X_train)):
            fold_model = GradientBoostingClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.7, min_samples_leaf=20, min_samples_split=30,
                max_features="sqrt", n_iter_no_change=15,
                validation_fraction=0.15, tol=1e-4, random_state=42,
            )
            fold_model.fit(X_train[tr_idx], y_train[tr_idx])
            fold_pred = fold_model.predict(X_train[te_idx])
            fold_proba = fold_model.predict_proba(X_train[te_idx])
            acc = (fold_pred == y_train[te_idx]).mean()
            cv_scores["accuracy"].append(acc)
            try:
                ll = log_loss(y_train[te_idx], fold_proba)
                cv_scores["log_loss"].append(ll)
            except Exception:
                pass

        cv_acc = np.mean(cv_scores["accuracy"])
        cv_std = np.std(cv_scores["accuracy"])
        logger.info(f"  CV Accuracy: {cv_acc:.4f} (+/- {cv_std:.4f})")
        if cv_scores["log_loss"]:
            logger.info(f"  CV Log-loss: {np.mean(cv_scores['log_loss']):.4f}")
    else:
        cv_acc = 0
        cv_std = 0

    # ── Fit on full train set ──
    model.fit(X_train, y_train)
    actual_n_estimators = model.n_estimators_
    logger.info(f"  Stopped at {actual_n_estimators} trees (max 300)")

    # ── Validation set evaluation ──
    val_preds = model.predict(X_val)
    val_proba = model.predict_proba(X_val)
    val_acc = (val_preds == y_val).mean()
    train_preds = model.predict(X_train)
    train_acc = (train_preds == y_train).mean()

    logger.info(f"  Train Accuracy: {train_acc:.4f}")
    logger.info(f"  Val Accuracy:   {val_acc:.4f}")
    logger.info(f"  Overfit gap:    {train_acc - val_acc:.4f}")

    # ── 概率校准（在 validation set 上）──
    try:
        cal_model = CalibratedClassifierCV(model, cv="prefit", method="isotonic")
        cal_model.fit(X_val, y_val)
        use_calibrated = True
        logger.info("  Probability calibration: isotonic (on val set)")
    except Exception:
        cal_model = model
        use_calibrated = False
        logger.info("  Probability calibration: skipped")

    # ── 全量预测（用于回测）──
    if use_calibrated:
        all_preds = cal_model.predict(X_scaled)
        all_proba = cal_model.predict_proba(X_scaled)
    else:
        all_preds = model.predict(X_scaled)
        all_proba = model.predict_proba(X_scaled)

    train_report = classification_report(y, all_preds, output_dict=True)

    # ── Validation metrics ──
    val_metrics = {"val_accuracy": round(val_acc, 4)}
    try:
        val_proba_cal = cal_model.predict_proba(X_val) if use_calibrated else val_proba
        if val_proba_cal.shape[1] == 2:
            val_metrics["val_roc_auc"] = round(
                roc_auc_score(y_val, val_proba_cal[:, 1]), 4
            )
            val_metrics["val_brier"] = round(
                brier_score_loss(y_val, val_proba_cal[:, 1]), 4
            )
        val_metrics["val_log_loss"] = round(log_loss(y_val, val_proba_cal), 4)
    except Exception:
        pass

    logger.info(f"  Val metrics: {val_metrics}")

    # ── Feature importance ──
    importances = model.feature_importances_
    feat_imp = sorted(
        zip(valid_feature_names, importances),
        key=lambda x: x[1], reverse=True,
    )[:30]
    logger.info("  Top features:")
    for name, imp in feat_imp[:10]:
        logger.info(f"    {name}: {imp:.4f}")

    # Regime feature importance ratio
    regime_imp = sum(
        imp for name, imp in zip(valid_feature_names, importances)
        if name.startswith("regime_")
    )
    total_imp = sum(importances)
    regime_ratio = regime_imp / (total_imp + 1e-10)
    logger.info(f"  Regime features importance: {regime_ratio:.2%}")

    # ── Save training log ──
    training_log = {
        "n_samples": len(X),
        "n_features_input": len(feature_cols),
        "n_features_valid": len(valid_cols),
        "target_col": target_col,
        "label_distribution": {
            str(k): int(v)
            for k, v in pd.Series(y).value_counts().items()
        },
        "model_params": {
            "n_estimators_max": 300,
            "n_estimators_actual": actual_n_estimators,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.7,
            "min_samples_leaf": 20,
        },
        "train_accuracy": round(train_acc, 4),
        "val_accuracy": round(val_acc, 4),
        "cv_accuracy": round(cv_acc, 4),
        "cv_std": round(cv_std, 4),
        "overfit_gap": round(train_acc - val_acc, 4),
        "calibrated": use_calibrated,
        "val_metrics": val_metrics,
        "regime_importance_ratio": round(regime_ratio, 4),
        "top_features": [(n, round(float(v), 6)) for n, v in feat_imp[:20]],
        "proba_stats": {
            "mean": round(float(all_proba.max(axis=1).mean()), 4),
            "std": round(float(all_proba.max(axis=1).std()), 4),
            "min": round(float(all_proba.max(axis=1).min()), 4),
            "max": round(float(all_proba.max(axis=1).max()), 4),
        },
    }

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        ts_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"train_{ts_str}.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(training_log, f, indent=2, default=str)
        logger.info(f"  Training log saved: {log_file}")

    return {
        "model": cal_model if use_calibrated else model,
        "raw_model": model,
        "scaler": scaler,
        "valid_cols": valid_cols,
        "valid_feature_names": valid_feature_names,
        "cv_accuracy": cv_acc,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "train_report": train_report,
        "top_features": feat_imp,
        "predictions": all_preds,
        "probabilities": all_proba,
        "regime_importance_ratio": regime_ratio,
        "val_metrics": val_metrics,
        "training_log": training_log,
    }


# ─────────────────────────────────────────────────────────
# Backtest
# ─────────────────────────────────────────────────────────

def backtest_fvg_v3(
    feature_df, ohlcv, predictions, probabilities,
    min_proba=0.6, hold_bars=20,
    stop_loss_pct=0.3, take_profit_pct=0.5,
    cooldown_bars=5,
    dynamic_exit=True,
):
    """
    Backtest FVG V3 predictions.

    Args:
        dynamic_exit: 使用 ATR 动态止损止盈（只用历史数据，无未来信息泄漏）
    """
    df = feature_df.copy()
    df["pred"] = predictions
    df["proba"] = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]

    signals = df[df["proba"] >= min_proba].copy()
    if signals.empty:
        return {"metrics": compute_backtest_metrics(pd.DataFrame()), "results_df": pd.DataFrame()}

    results = []
    last_exit = -cooldown_bars

    for _, row in signals.iterrows():
        ts = pd.to_datetime(row["timestamp"])
        # 统一时区
        if ohlcv.index.tz is None and getattr(ts, "tzinfo", None) is not None:
            ts = ts.tz_localize(None)
        elif ohlcv.index.tz is not None and getattr(ts, "tzinfo", None) is None:
            ts = ts.tz_localize(ohlcv.index.tz)

        if ts not in ohlcv.index:
            continue

        entry_i = ohlcv.index.get_loc(ts)
        if entry_i - last_exit < cooldown_bars:
            continue

        entry_price = float(ohlcv.iloc[entry_i]["close"])
        fvg_dir = int(row["fvg_type"])
        direction = fvg_dir if row["pred"] == 1 else -fvg_dir

        # Dynamic exit: ATR-based SL/TP (只用 entry_i 及之前的数据)
        if dynamic_exit:
            ep = atr_exit_params(
                ohlcv, entry_i, direction,
                atr_period=14, sl_atr_mult=2.0, tp_atr_mult=3.0,
                max_hold_bars=hold_bars,
            )
        else:
            from backtest.dynamic_exit import ExitParams
            ep = ExitParams(
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                max_hold_bars=hold_bars,
            )

        exit_result = simulate_exit(ohlcv, entry_i, direction, ep)
        last_exit = exit_result["exit_idx"]

        results.append({
            "entry_time": ts, "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_result["exit_price"],
            "return_pct": exit_result["return_pct"],
            "is_win": exit_result["return_pct"] > 0,
            "exit_reason": exit_result["exit_reason"],
            "hold_bars": exit_result["hold_bars"],
            "fvg_type": row["fvg_type"],
            "pred_proba": row["proba"],
            "sl_pct": ep.stop_loss_pct,
            "tp_pct": ep.take_profit_pct,
        })

    rdf = pd.DataFrame(results) if results else pd.DataFrame()
    return {
        "metrics": compute_backtest_metrics(rdf),
        "results_df": rdf,
        "n_signals": len(signals),
        "n_trades": len(rdf),
    }


# ─────────────────────────────────────────────────────────
# Grid Search
# ─────────────────────────────────────────────────────────

def grid_search(feature_df, ohlcv, trained_model_result):
    """Grid search over backtest params."""
    params_grid = {
        "min_proba": [0.5, 0.6, 0.7],
        "hold_bars": [5, 10, 20],
        "stop_loss_pct": [0.2, 0.3, 0.5],
        "take_profit_pct": [0.3, 0.5, 1.0],
        "cooldown_bars": [2, 3, 5],
    }

    keys = list(params_grid.keys())
    values = list(params_grid.values())
    all_combos = list(product(*values))
    logger.info(f"Grid search: {len(all_combos)} combinations")

    preds = trained_model_result["predictions"]
    probas = trained_model_result["probabilities"]

    results = []
    for i, combo in enumerate(all_combos):
        params = dict(zip(keys, combo))
        bt = backtest_fvg_v3(
            feature_df, ohlcv, preds, probas, **params,
        )
        m = bt["metrics"]
        results.append({**params, **m, "n_trades": bt["n_trades"]})

        if (i + 1) % 50 == 0:
            logger.info(f"  [{i+1}/{len(all_combos)}]")

    rdf = pd.DataFrame(results)
    rdf = rdf.sort_values("sharpe", ascending=False)
    return rdf


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FVG V3 Strategy")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--start-date", type=str, default=None, help="Explicit start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="Explicit end date YYYY-MM-DD")
    parser.add_argument("--freq", default="1min", help="OHLCV resample frequency")
    parser.add_argument("--min-gap-pct", type=float, default=0.02, help="Min FVG gap %")
    parser.add_argument("--tb-upper", type=float, default=0.5, help="Triple Barrier upper %% (对称)")
    parser.add_argument("--tb-lower", type=float, default=0.5, help="Triple Barrier lower %% (默认与 upper 相同)")
    parser.add_argument("--tb-horizons", nargs="+", type=int, default=[10, 30, 60])
    parser.add_argument("--target-horizon", type=int, default=30, help="Which TB horizon to predict")
    parser.add_argument("--grid-search", action="store_true")
    parser.add_argument("--use-orderbook", action="store_true")
    parser.add_argument(
        "--target", default="tb_direction",
        choices=[
            "tb_direction",     # Triple Barrier: 预测上涨方向 (binary: label=1 vs rest)
            "tb_trend",         # Triple Barrier: 预测趋势 (binary: |label|=1 vs 0)
            "tb_multiclass",    # Triple Barrier: 三分类 (-1/0/1)
            "fill_direction",   # 原始标签: FVG 回填后价格方向 (binary)
            "future_ret_5",     # 原始标签: 未来5根K线涨跌方向 (binary)
            "future_ret_10",    # 原始标签: 未来10根K线涨跌方向 (binary)
            "future_ret_20",    # 原始标签: 未来20根K线涨跌方向 (binary)
        ],
        help="预测目标类型：tb_* 使用 Triple Barrier，其余使用原始标签（适合标签不均衡时）",
    )
    args = parser.parse_args()

    # Date range
    if args.start_date and args.end_date:
        start_date = date.fromisoformat(args.start_date)
        end_date = date.fromisoformat(args.end_date)
    else:
        end_date = date.today()
        start_date = end_date - timedelta(days=args.days)

    n_days = (end_date - start_date).days
    sym_slug = args.symbol.replace("/", "_")

    # ── 1. Load data ──
    logger.info("═══ FVG V3 Strategy ═══")
    logger.info(f"Symbol: {args.symbol} | {start_date} -> {end_date} ({n_days}d)")

    logger.info("[1/7] Loading aggTrades...")
    trades = load_agg_trades(args.symbol, start_date, end_date)
    if trades.empty:
        logger.error("No trades data")
        return
    logger.info(f"  Loaded {len(trades):,} aggTrades")

    logger.info("[1/7] Resampling to OHLCV...")
    ohlcv = resample_trades_to_ohlcv(trades, freq=args.freq)
    logger.info(f"  {len(ohlcv):,} bars ({args.freq})")

    # Optional: orderbook (lake-api sample: BTC-USDT 2022-10-01 ~ 2022-10-03)
    book = None
    if args.use_orderbook and HAS_LAKEAPI:
        logger.info("[1/7] Loading orderbook (lake-api sample)...")
        try:
            # lake-api symbol format: BTC-USDT
            lake_symbol = args.symbol.replace("/", "-")
            book = load_lakeapi_orderbook(
                days=n_days, symbol=lake_symbol,
                resample_freq="1s", depth_levels=10,
            )
            if book is not None and not book.empty:
                # 确保 orderbook 有时间索引
                if "origin_time" in book.columns:
                    book = book.set_index(pd.to_datetime(book["origin_time"])).sort_index()
                    book = book.drop(columns=["origin_time"], errors="ignore")
                logger.info(f"  Loaded {len(book):,} orderbook snapshots")
            else:
                logger.warning("  Orderbook empty")
                book = None
        except Exception as e:
            logger.warning(f"  Orderbook unavailable: {e}")
            book = None

    # ── 2. Taker order grouping ──
    logger.info("[2/7] Grouping taker orders...")
    taker_orders = group_taker_orders(trades)
    logger.info(f"  {len(taker_orders):,} taker orders from {len(trades):,} aggTrades")
    logger.info(f"  Avg levels swept: {taker_orders['levels_swept'].mean():.2f}")
    logger.info(f"  Multi-level (>=2): {(taker_orders['levels_swept'] >= 2).mean()*100:.1f}%")
    logger.info(f"  Aggressive (>=3): {(taker_orders['levels_swept'] >= 3).mean()*100:.1f}%")

    # ── 3. Detect FVGs ──
    logger.info(f"[3/7] Detecting FVGs (min gap: {args.min_gap_pct}%)...")
    fvgs = detect_fvg(ohlcv, min_gap_pct=args.min_gap_pct)
    logger.info(f"  Found {len(fvgs)} FVGs")
    if len(fvgs) == 0:
        logger.error("No FVGs found. Try lowering --min-gap-pct")
        return

    n_bullish = sum(1 for f in fvgs if f.fvg_type.value == "bullish")
    n_bearish = len(fvgs) - n_bullish
    logger.info(f"  Bullish: {n_bullish} | Bearish: {n_bearish}")

    # ── 4. Build V3 features ──
    logger.info("[4/7] Building V3 feature matrix...")
    feat_df = build_fvg_features_v3(
        fvgs, trades, ohlcv,
        taker_orders=taker_orders,
        book=book,
    )
    logger.info(f"  Feature matrix: {feat_df.shape}")

    # ── 5. Labels ──
    logger.info("[5/7] Building labels...")

    # Always compute Triple Barrier labels with improved implementation
    # 使用 OHLCV high/low 判断屏障触及（更真实）
    feat_df = add_triple_barrier_labels(
        feat_df, ohlcv,
        horizons=args.tb_horizons,
        upper_pct=args.tb_upper,
        lower_pct=args.tb_lower,
    )

    tb_col = f"tb_label_{args.target_horizon}"

    # 验证 Triple Barrier 标签质量
    if tb_col in feat_df.columns:
        from backtest.labeling import validate_labels
        tb_labels = feat_df[tb_col]
        val_result = validate_labels(tb_labels, ohlcv["close"])
        logger.info(f"  TB Label validation: trend={val_result['expected_trend']}, "
                     f"consistent={val_result['is_consistent']}")
        if val_result["warnings"]:
            for w in val_result["warnings"]:
                logger.warning(f"  ⚠ {w}")

    # Derived columns (always created for downstream use)
    if tb_col in feat_df.columns:
        feat_df["target_trend"] = (feat_df[tb_col] != 0).astype(int)
        feat_df["target_direction"] = feat_df[tb_col].map({1: 1, 0: 0, -1: 0}).fillna(0).astype(int)
        feat_df["target_multiclass"] = (feat_df[tb_col] + 1).astype(int)  # 0/1/2

    # ── Select target based on --target flag ──
    target_name = args.target
    if target_name == "tb_direction":
        train_target = "target_direction"
    elif target_name == "tb_trend":
        train_target = "target_trend"
    elif target_name == "tb_multiclass":
        train_target = "target_multiclass"
    elif target_name == "fill_direction":
        # 原始标签：FVG 回填后方向
        train_target = "label_direction_10"
        if train_target not in feat_df.columns:
            # fallback
            for h in [5, 10, 20]:
                c = f"label_direction_{h}"
                if c in feat_df.columns:
                    train_target = c
                    break
    elif target_name.startswith("future_ret_"):
        horizon = target_name.split("_")[-1]
        train_target = f"label_direction_{horizon}"
    else:
        train_target = "target_direction"

    if train_target not in feat_df.columns:
        logger.error(f"Target column '{train_target}' not found in features. "
                     f"Available: {[c for c in feat_df.columns if 'label' in c or 'target' in c]}")
        return

    label_dist = feat_df[train_target].value_counts().to_dict()
    logger.info(f"  Target: --target={target_name} -> column='{train_target}'")
    logger.info(f"  Label distribution: {label_dist}")

    # 检查是否严重不均衡并提示
    if len(label_dist) >= 2:
        majority = max(label_dist.values())
        minority = min(label_dist.values())
        ratio = majority / (minority + 1)
        if ratio > 5:
            logger.warning(f"  ⚠ Label imbalance ratio: {ratio:.1f}x — "
                           f"consider --target fill_direction or future_ret_10")

    feature_cols = get_v3_feature_columns(feat_df)
    logger.info(f"  Using {len(feature_cols)} features")

    # ── 6. Train model ──
    logger.info("[6/7] Training model...")
    logger.info(f"  Target column: {train_target}")
    log_dir = DataPaths.data / "training_logs" / "fvg_v3" / sym_slug
    trained = train_v3_model(feat_df, train_target, feature_cols, log_dir=log_dir)

    # ── 7. Backtest ──
    logger.info("[7/7] Backtesting (dynamic ATR exit)...")
    preds = trained["predictions"]
    probas = trained["probabilities"]

    bt = backtest_fvg_v3(
        feat_df, ohlcv, preds, probas,
        min_proba=0.5, hold_bars=20,
        cooldown_bars=5,
        dynamic_exit=True,
    )

    metrics = bt["metrics"]
    logger.info("═══ Backtest Results ═══")
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")

    # ── Grid search ──
    grid_df = None
    if args.grid_search:
        logger.info("═══ Grid Search ═══")
        grid_df = grid_search(feat_df, ohlcv, trained)
        logger.info(f"\nTop 10 parameter sets:")
        print(grid_df.head(10).to_string())

    # ── Save results ──
    # 按 币种 / 日期范围_目标 分目录，避免实验互相覆盖
    run_id = f"{start_date}_{end_date}_{args.target}"
    results_dir = (
        DataPaths.data / "backtest_results" / "fvg_v3" / sym_slug / run_id
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    feat_df.to_parquet(results_dir / "features_v3.parquet", index=False)
    if bt["results_df"] is not None and not bt["results_df"].empty:
        bt["results_df"].to_csv(results_dir / "backtest_trades.csv", index=False)
    if grid_df is not None:
        grid_df.to_csv(results_dir / "grid_search.csv", index=False)

    # Save summary
    summary = {
        "symbol": args.symbol,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "freq": args.freq,
        "target": args.target,
        "train_target_column": train_target,
        "label_distribution": {str(k): int(v) for k, v in label_dist.items()},
        "n_agg_trades": len(trades),
        "n_taker_orders": len(taker_orders),
        "n_fvgs": len(fvgs),
        "n_fvgs_bullish": n_bullish,
        "n_fvgs_bearish": n_bearish,
        "n_features": len(feature_cols),
        "has_orderbook": book is not None,
        "model": {
            "cv_accuracy": trained["cv_accuracy"],
            "train_accuracy": trained.get("train_accuracy", 0),
            "val_accuracy": trained.get("val_accuracy", 0),
            "overfit_gap": round(
                trained.get("train_accuracy", 0) - trained.get("val_accuracy", 0), 4
            ),
            "n_estimators_actual": trained["training_log"]["model_params"]["n_estimators_actual"],
            "calibrated": trained["training_log"]["calibrated"],
            "val_metrics": trained.get("val_metrics", {}),
        },
        "regime_importance_ratio": trained.get("regime_importance_ratio", 0),
        "proba_stats": trained["training_log"]["proba_stats"],
        "backtest_metrics": metrics,
        "backtest_config": {
            "dynamic_exit": True,
            "min_proba": 0.5,
            "hold_bars": 20,
            "cooldown_bars": 5,
        },
        "top_features": [(n, float(v)) for n, v in trained["top_features"][:15]],
    }
    if args.grid_search and grid_df is not None and not grid_df.empty:
        best_row = grid_df.iloc[0].to_dict()
        summary["grid_search_best"] = {
            k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v)
            for k, v in best_row.items()
        }

    with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
