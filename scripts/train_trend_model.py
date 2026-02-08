#!/usr/bin/env python3
"""
趋势预测模型训练：Triple Barrier + 多 Regime 平衡训练。

思路：
  1. 从三种市场环境（上升/下跌/震荡）各选 N 天数据
  2. 计算特征（从 parquet 加载或现场计算）
  3. 对每根 bar 标注 Triple Barrier 标签
  4. 训练 GBM 和 LSTM 两种模型
  5. 在三种 regime 的测试集上分别评估

Usage:
    # 使用默认 BTC 日期配置
    python scripts/train_trend_model.py

    # 自定义日期
    python scripts/train_trend_model.py --symbol BTC/USDT --config custom_dates.json

    # 只训练 GBM
    python scripts/train_trend_model.py --model gbm

    # 只训练 LSTM
    python scripts/train_trend_model.py --model lstm --lookback 60
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utilities.binance_loader import load_agg_trades, resample_trades_to_ohlcv
from utilities.paths import DataPaths
from backtest.labeling import triple_barrier_labels, validate_labels
from backtest.metrics import compute_backtest_metrics
from backtest.dynamic_exit import atr_exit_params, simulate_exit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# 默认日期配置（BTC/USDT 2022）
# ─────────────────────────────────────────────────────────

# BTC/USDT 2022 October（确认 Binance API 可用）
# Oct 4-6: 小幅上涨 ($19.3k → $20.3k)
# Oct 13-15: 明显下跌 ($19.1k → $18.3k)
# Oct 1-3: 窄幅震荡 (~$19.3k)
DEFAULT_REGIME_DATES = {
    "train": {
        "up": ["2022-10-04", "2022-10-05", "2022-10-06"],
        "down": ["2022-10-13", "2022-10-14", "2022-10-15"],
        "sideways": ["2022-10-01", "2022-10-02", "2022-10-03"],
    },
    "test": {
        "up": ["2022-10-07"],
        "down": ["2022-10-16"],
        "sideways": ["2022-10-08"],
    },
}


# ─────────────────────────────────────────────────────────
# 特征加载
# ─────────────────────────────────────────────────────────

def load_features_for_dates(
    symbol: str,
    dates: List[str],
    compute_if_missing: bool = True,
) -> pd.DataFrame:
    """加载多天特征（从 parquet 或现场计算）。"""
    from scripts.compute_features import compute_all_features_for_day

    all_dfs = []
    for d_str in dates:
        dt = date.fromisoformat(d_str)
        path = DataPaths.features(symbol, "bar_features", dt)

        if path.exists():
            logger.info(f"  [Feature] Loading cached: {d_str}")
            df = pd.read_parquet(path)
        elif compute_if_missing:
            logger.info(f"  [Feature] Computing: {d_str}")
            df = compute_all_features_for_day(symbol, dt)
            if not df.empty:
                df.to_parquet(path)
                logger.info(f"  [Feature] Saved: {path}")
        else:
            logger.warning(f"  [Feature] Missing: {d_str}, skipping")
            continue

        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    result = pd.concat(all_dfs)
    result = result.sort_index()
    return result


# ─────────────────────────────────────────────────────────
# 标签生成
# ─────────────────────────────────────────────────────────

def add_labels_to_features(
    features: pd.DataFrame,
    ohlcv: pd.DataFrame,
    upper_pct: float = 0.3,
    lower_pct: float = 0.3,
    max_bars: int = 30,
) -> pd.DataFrame:
    """
    为每根 bar 添加 Triple Barrier 标签。
    使用 OHLCV high/low 判断屏障触及。
    """
    tb = triple_barrier_labels(
        ohlcv["close"],
        upper_pct=upper_pct,
        lower_pct=lower_pct,
        max_bars=max_bars,
        use_high_low=True,
        ohlcv=ohlcv,
    )

    # 合并到 features
    features = features.copy()
    common_idx = features.index.intersection(tb.index)
    features = features.loc[common_idx]

    features["tb_label"] = tb.loc[common_idx, "label"]
    features["tb_barrier"] = tb.loc[common_idx, "barrier_hit"]
    features["tb_bars"] = tb.loc[common_idx, "bars_to_hit"]
    features["tb_return"] = tb.loc[common_idx, "return_at_hit"]

    # Binary target: direction (1 = up, 0 = down/neutral)
    features["target_direction"] = (features["tb_label"] == 1).astype(int)
    # Binary target: trend (1 = strong move, 0 = neutral)
    features["target_trend"] = (features["tb_label"] != 0).astype(int)
    # Multiclass: 0=down, 1=neutral, 2=up
    features["target_multiclass"] = (features["tb_label"] + 1).astype(int)

    return features


# ─────────────────────────────────────────────────────────
# 回测辅助
# ─────────────────────────────────────────────────────────

def _run_backtest(
    feature_df: pd.DataFrame,
    ohlcv: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    min_proba: float = 0.4,
    cooldown_bars: int = 5,
    hold_bars: int = 20,
    target_mode: str = "multiclass",
) -> pd.DataFrame:
    """
    模型预测 → 实际交易回测。

    三分类策略 (target_multiclass):
      - pred=2 (up) → 做多
      - pred=0 (down) → 做空
      - pred=1 (neutral) → 不交易
    二分类兼容 (target_trend/target_direction):
      - 同之前逻辑
    """
    df = feature_df.copy()
    df["pred"] = predictions

    # 构建信号
    if target_mode == "multiclass":
        # 三分类：pred=2 做多，pred=0 做空，pred=1 不交易
        # 取预测类别对应的概率作为置信度
        if probabilities.ndim > 1 and probabilities.shape[1] >= 3:
            df["proba"] = probabilities.max(axis=1)
        else:
            df["proba"] = 1.0

        signals = df[(df["pred"] != 1) & (df["proba"] >= min_proba)]
    else:
        # 二分类兼容
        if probabilities.ndim > 1 and probabilities.shape[1] > 1:
            df["proba"] = probabilities[:, 1]
        else:
            df["proba"] = probabilities[:, 0]
        signals = df[(df["pred"] == 1) & (df["proba"] >= min_proba)]

    if signals.empty:
        return pd.DataFrame()

    results = []
    last_exit = -cooldown_bars

    for idx_label, row in signals.iterrows():
        ts = idx_label

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
        if entry_i < 5 or entry_i >= len(ohlcv) - 2:
            continue

        entry_price = float(ohlcv.iloc[entry_i]["close"])

        # 方向：三分类直接由模型决定
        if target_mode == "multiclass":
            direction = 1 if row["pred"] == 2 else -1
        else:
            # 二分类：用近 5 bar 动量推断方向
            recent_ret = (ohlcv.iloc[entry_i]["close"] - ohlcv.iloc[entry_i - 5]["close"])
            direction = 1 if recent_ret > 0 else -1

        # ATR 动态止损
        ep = atr_exit_params(
            ohlcv, entry_i, direction,
            atr_period=14, sl_atr_mult=2.0, tp_atr_mult=3.0,
            max_hold_bars=hold_bars,
        )
        exit_result = simulate_exit(ohlcv, entry_i, direction, ep)
        last_exit = exit_result["exit_idx"]

        results.append({
            "entry_time": ts,
            "direction": "long" if direction == 1 else "short",
            "entry_price": entry_price,
            "exit_price": exit_result["exit_price"],
            "return_pct": exit_result["return_pct"],
            "is_win": exit_result["return_pct"] > 0,
            "exit_reason": exit_result["exit_reason"],
            "hold_bars": exit_result["hold_bars"],
            "pred_proba": row["proba"],
            "pred_class": int(row["pred"]),
            "sl_pct": ep.stop_loss_pct,
            "tp_pct": ep.take_profit_pct,
        })

    return pd.DataFrame(results) if results else pd.DataFrame()


def _select_top_features(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    top_n: int = 30,
) -> List[str]:
    """
    基于 permutation importance 选出 top_n 特征 + 全部 regime 特征。
    """
    from sklearn.inspection import permutation_importance

    logger.info(f"  [Feature Selection] Computing permutation importance...")
    perm = permutation_importance(
        model, X_val, y_val,
        n_repeats=5, random_state=42, n_jobs=-1, scoring="accuracy",
    )
    imp = perm.importances_mean

    # 排序
    ranked = sorted(
        zip(feature_names, imp), key=lambda x: x[1], reverse=True,
    )

    # Top N 特征（排除 regime，regime 单独加）
    top_non_regime = [
        name for name, _ in ranked
        if not name.startswith("regime_")
    ][:top_n]

    # 全部 regime 特征
    regime_feats = [name for name in feature_names if name.startswith("regime_")]

    selected = list(set(top_non_regime + regime_feats))
    logger.info(f"  [Feature Selection] Top {top_n} features + {len(regime_feats)} regime = {len(selected)} total")
    logger.info(f"  [Feature Selection] Top 10:")
    for name, v in ranked[:10]:
        logger.info(f"    {name}: {v:.4f}")

    return selected, ranked


# ─────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train trend prediction model")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--model", default="both", choices=["gbm", "lstm", "both"])
    parser.add_argument("--target", default="target_multiclass",
                        choices=["target_direction", "target_trend", "target_multiclass"])
    parser.add_argument("--top-features", type=int, default=30,
                        help="Top N features to select (+ all regime features)")
    parser.add_argument("--tb-upper", type=float, default=0.3, help="TB upper barrier %%")
    parser.add_argument("--tb-lower", type=float, default=0.3, help="TB lower barrier %%")
    parser.add_argument("--tb-horizon", type=int, default=30, help="TB max bars")
    parser.add_argument("--lookback", type=int, default=60, help="LSTM lookback")
    parser.add_argument("--freq", default="1min")
    parser.add_argument("--config", type=str, default=None,
                        help="JSON file with regime date config")
    args = parser.parse_args()

    sym_slug = args.symbol.replace("/", "_")
    log_dir = DataPaths.data / "training_logs" / "trend" / sym_slug
    results_dir = DataPaths.data / "backtest_results" / "trend" / sym_slug
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load date config
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            regime_dates = json.load(f)
    else:
        regime_dates = DEFAULT_REGIME_DATES

    logger.info("═══════════════════════════════════════════════")
    logger.info("  Trend Prediction Model Training")
    logger.info(f"  Symbol: {args.symbol}")
    logger.info(f"  Target: {args.target}")
    logger.info(f"  TB Barriers: ±{args.tb_upper}%, horizon={args.tb_horizon}")
    logger.info(f"  Model: {args.model}")
    logger.info("═══════════════════════════════════════════════")

    # ── 1. Load and compute features ──
    logger.info("\n[1/5] Loading features for each regime...")

    train_dfs = {}
    for regime, dates in regime_dates["train"].items():
        logger.info(f"\n  ── {regime.upper()} regime (train) ──")
        features = load_features_for_dates(args.symbol, dates)
        if features.empty:
            logger.warning(f"  No features for {regime} train dates!")
            continue
        train_dfs[regime] = features
        logger.info(f"  {regime}: {len(features)} bars, {features.shape[1]} features")

    test_dfs = {}
    for regime, dates in regime_dates["test"].items():
        logger.info(f"\n  ── {regime.upper()} regime (test) ──")
        features = load_features_for_dates(args.symbol, dates)
        if features.empty:
            logger.warning(f"  No features for {regime} test dates!")
            continue
        test_dfs[regime] = features
        logger.info(f"  {regime}: {len(features)} bars, {features.shape[1]} features")

    if not train_dfs:
        logger.error("No training data! Run compute_features.py first or ensure trades are available.")
        return

    # ── 2. Add labels ──
    logger.info("\n[2/5] Adding Triple Barrier labels...")

    labeled_train = {}
    for regime, feat_df in train_dfs.items():
        logger.info(f"  Labeling {regime} regime...")
        # 需要 OHLCV 来计算 TB 标签 — 从 trades 重建
        all_dates = regime_dates["train"][regime]
        ohlcv = _load_ohlcv_for_dates(args.symbol, all_dates, args.freq)
        if ohlcv is None or ohlcv.empty:
            logger.warning(f"  Cannot build OHLCV for {regime}")
            continue

        labeled = add_labels_to_features(
            feat_df, ohlcv,
            upper_pct=args.tb_upper,
            lower_pct=args.tb_lower,
            max_bars=args.tb_horizon,
        )
        val = validate_labels(labeled["tb_label"], ohlcv["close"])
        logger.info(f"  {regime}: labels {val['label_distribution']}, "
                     f"trend={val['expected_trend']}, consistent={val['is_consistent']}")
        if val["warnings"]:
            for w in val["warnings"]:
                logger.warning(f"    ⚠ {w}")
        labeled_train[regime] = labeled

    labeled_test = {}
    for regime, feat_df in test_dfs.items():
        all_dates = regime_dates["test"][regime]
        ohlcv = _load_ohlcv_for_dates(args.symbol, all_dates, args.freq)
        if ohlcv is None or ohlcv.empty:
            continue
        labeled = add_labels_to_features(
            feat_df, ohlcv,
            upper_pct=args.tb_upper,
            lower_pct=args.tb_lower,
            max_bars=args.tb_horizon,
        )
        labeled_test[regime] = labeled

    # ── 3. Combine training data (balanced) ──
    logger.info("\n[3/5] Combining regime-balanced training data...")

    # 每个 regime 取最小数量做平衡
    min_n = min(len(df) for df in labeled_train.values())
    logger.info(f"  Min regime size: {min_n}")

    balanced_dfs = []
    for regime, df in labeled_train.items():
        # 随机下采样到 min_n
        sampled = df.sample(n=min_n, random_state=42) if len(df) > min_n else df
        sampled = sampled.copy()
        sampled["regime"] = regime
        balanced_dfs.append(sampled)
        logger.info(f"  {regime}: {len(sampled)} bars")

    train_all = pd.concat(balanced_dfs).sort_index()

    # Target
    target_col = args.target
    if target_col not in train_all.columns:
        logger.error(f"Target '{target_col}' not found! Available: "
                     f"{[c for c in train_all.columns if 'target' in c]}")
        return

    # Get feature columns (exclude targets, labels, metadata, absolute price features)
    exclude_prefixes = (
        "tb_", "target_", "regime", "timestamp",
        "label_", "pred", "proba",
    )
    # 排除绝对价格特征（会泄漏跨期价格水平信息）
    absolute_price_cols = {"micro_vwap", "vwap", "close", "open", "high", "low"}
    feature_cols = [
        c for c in train_all.columns
        if not c.startswith(exclude_prefixes)
        and c not in absolute_price_cols
        and train_all[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]
    ]
    logger.info(f"  Feature columns: {len(feature_cols)} (excluded absolute price features)")
    logger.info(f"  Total training bars: {len(train_all)}")

    y_train_raw = train_all[target_col].values.astype(int)
    X_train_raw = train_all[feature_cols].fillna(0).values.astype(np.float32)

    label_dist_raw = pd.Series(y_train_raw).value_counts().to_dict()
    logger.info(f"  Raw label distribution: {label_dist_raw}")

    # 三分类平衡：下采样多数类到少数类的数量
    if target_col == "target_multiclass" and len(set(y_train_raw)) > 2:
        min_class_n = min(pd.Series(y_train_raw).value_counts())
        balanced_idx = []
        for cls in sorted(set(y_train_raw)):
            cls_idx = np.where(y_train_raw == cls)[0]
            if len(cls_idx) > min_class_n:
                cls_idx = np.random.RandomState(42).choice(
                    cls_idx, min_class_n, replace=False,
                )
            balanced_idx.extend(cls_idx)
        balanced_idx = sorted(balanced_idx)
        X_train = X_train_raw[balanced_idx]
        y_train = y_train_raw[balanced_idx]
        label_dist = pd.Series(y_train).value_counts().to_dict()
        logger.info(f"  Balanced label distribution: {label_dist}")
        logger.info(f"  Balanced training bars: {len(y_train)}")
    else:
        X_train = X_train_raw
        y_train = y_train_raw
        label_dist = label_dist_raw

    # ── Prepare test data ──
    test_data = {}
    for regime, df in labeled_test.items():
        y = df[target_col].values.astype(int)
        X = df[feature_cols].fillna(0).values.astype(np.float32)
        test_data[regime] = (X, y, df)

    # ── 4. Feature Selection + Training ──
    logger.info("\n[4/6] Feature selection & model training...")

    # Train/val split (last 20%)
    val_size = max(int(len(train_all) * 0.2), 50)
    X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
    y_tr, y_val = y_train[:-val_size], y_train[-val_size:]

    all_results = {}

    # ── Phase 1: Quick GBM for feature selection ──
    if args.model in ("gbm", "both"):
        logger.info("\n  ═══ Phase 1: Feature Selection (quick GBM) ═══")
        from models.gbm_model import TrendGBM

        gbm_quick = TrendGBM(
            max_iter=100, max_depth=3, learning_rate=0.1,
            min_samples_leaf=30, max_features=0.5,
            l2_regularization=1.0, calibrate=False,
        )
        gbm_quick.fit(X_tr, y_tr, X_val, y_val, feature_cols)

        # 获取 raw model 用于 permutation importance
        selected_feats, full_ranking = _select_top_features(
            gbm_quick._raw_model,
            gbm_quick.scaler.transform(X_val[:, gbm_quick.valid_cols]),
            y_val, gbm_quick.feature_names,
            top_n=args.top_features,
        )

        # 从 feature_cols 中过滤出 selected 的列索引
        selected_idx = [i for i, c in enumerate(feature_cols) if c in selected_feats]
        selected_cols = [feature_cols[i] for i in selected_idx]
        logger.info(f"  Selected {len(selected_cols)} features for final training")

        # 重建 X 矩阵
        X_tr_sel = X_tr[:, selected_idx]
        X_val_sel = X_val[:, selected_idx]

        # 重建 test data
        test_data_sel = {}
        for regime, (X_test, y_test, df) in test_data.items():
            test_data_sel[regime] = (X_test[:, selected_idx], y_test, df)

        # ── Phase 2: Final GBM with selected features ──
        # 使用更简单的配置减少过拟合
        logger.info("\n  ═══ Phase 2: Final GBM (selected features) ═══")
        gbm = TrendGBM(
            max_iter=80, max_depth=2, learning_rate=0.05,
            min_samples_leaf=50, max_features=0.5,
            l2_regularization=5.0, calibrate=True,
        )
        gbm_log = gbm.fit(X_tr_sel, y_tr, X_val_sel, y_val, selected_cols)
        gbm.save_log(log_dir, prefix="gbm")

        # Evaluate on each regime
        logger.info("\n  ── GBM Test Results by Regime ──")
        gbm_test_results = {}
        for regime, (X_test, y_test, _) in test_data_sel.items():
            preds, _proba = gbm.predict(X_test)
            acc = (preds == y_test).mean()
            dist = pd.Series(y_test).value_counts().to_dict()
            pred_dist = pd.Series(preds).value_counts().to_dict()
            logger.info(f"    {regime:>10}: acc={acc:.4f}, "
                         f"true={dist}, pred={pred_dist}")
            gbm_test_results[regime] = {
                "accuracy": round(acc, 4),
                "n_samples": len(y_test),
                "label_dist": {str(k): int(v) for k, v in dist.items()},
                "pred_dist": {str(k): int(v) for k, v in pred_dist.items()},
            }

        all_results["gbm"] = {
            "training": gbm_log,
            "test_by_regime": gbm_test_results,
            "selected_features": selected_cols,
            "full_feature_ranking": [(n, round(float(v), 6)) for n, v in full_ranking[:50]],
        }

    # ── LSTM (with selected features) ──
    if args.model in ("lstm", "both"):
        logger.info("\n  ═══ LSTM Model (selected features) ═══")
        try:
            from models.temporal_model import TrendLSTM, create_sequences

            _X_tr = X_tr_sel if "selected_idx" in dir() else X_tr
            _X_val = X_val_sel if "selected_idx" in dir() else X_val
            _test_data = test_data_sel if "test_data_sel" in dir() else test_data
            _feat_cols = selected_cols if "selected_cols" in dir() else feature_cols

            lstm = TrendLSTM(
                lookback=args.lookback,
                hidden_size=64,
                num_layers=2,
                dropout=0.3,
                cell_type="lstm",
                learning_rate=1e-3,
                batch_size=64,
                max_epochs=100,
                patience=15,
            )

            X_tr_seq, y_tr_seq = create_sequences(_X_tr, y_tr, lookback=args.lookback)
            X_val_seq, y_val_seq = create_sequences(_X_val, y_val, lookback=args.lookback)

            if len(X_tr_seq) < 10 or len(X_val_seq) < 5:
                logger.warning("  Not enough data for LSTM sequences, skipping")
            else:
                lstm_log = lstm.fit(X_tr_seq, y_tr_seq, X_val_seq, y_val_seq, _feat_cols)
                lstm.save_log(log_dir, prefix="lstm")

                logger.info("\n  ── LSTM Test Results by Regime ──")
                lstm_test_results = {}
                for regime, (X_test, y_test, _) in _test_data.items():
                    X_test_seq, y_test_seq = create_sequences(
                        X_test, y_test, lookback=args.lookback,
                    )
                    if len(X_test_seq) < 1:
                        logger.info(f"    {regime:>10}: too few samples for LSTM")
                        continue
                    preds, proba = lstm.predict(X_test_seq)
                    acc = (preds == y_test_seq).mean()
                    dist = pd.Series(y_test_seq).value_counts().to_dict()
                    pred_dist = pd.Series(preds).value_counts().to_dict()
                    logger.info(f"    {regime:>10}: acc={acc:.4f}, "
                                 f"true={dist}, pred={pred_dist}")
                    lstm_test_results[regime] = {
                        "accuracy": round(acc, 4),
                        "n_samples": len(y_test_seq),
                        "label_dist": {str(k): int(v) for k, v in dist.items()},
                        "pred_dist": {str(k): int(v) for k, v in pred_dist.items()},
                    }

                all_results["lstm"] = {
                    "training": lstm_log,
                    "test_by_regime": lstm_test_results,
                }

        except ImportError as e:
            logger.warning(f"  LSTM skipped: {e}")

    # ── 5. Backtest on test data ──
    logger.info("\n[5/7] Running backtest on test data...")

    # 确定使用哪个 test_data（带特征筛选的）
    _bt_test_data = test_data_sel if "test_data_sel" in dir() else test_data
    target_mode = "multiclass" if target_col == "target_multiclass" else "binary"

    backtest_results = {}
    all_trades = []
    for regime, (X_test, y_test, test_df) in _bt_test_data.items():
        logger.info(f"\n  ── Backtest: {regime.upper()} regime ──")

        # 加载该 regime 的 OHLCV
        test_dates = regime_dates["test"][regime]
        ohlcv = _load_ohlcv_for_dates(args.symbol, test_dates, args.freq)
        if ohlcv is None or ohlcv.empty:
            logger.warning(f"  No OHLCV for {regime} test, skipping backtest")
            continue

        # GBM 预测回测
        if "gbm" in all_results:
            preds, proba = gbm.predict(X_test)
            bt = _run_backtest(
                test_df, ohlcv, preds, proba,
                min_proba=0.4, cooldown_bars=5, hold_bars=20,
                target_mode=target_mode,
            )
            metrics = compute_backtest_metrics(bt)

            # 统计多空分布
            if not bt.empty:
                long_n = (bt["direction"] == "long").sum()
                short_n = (bt["direction"] == "short").sum()
                dir_str = f"long={long_n}, short={short_n}"
            else:
                dir_str = "none"

            logger.info(f"  [GBM] {regime}: signals={metrics['n_signals']} ({dir_str}), "
                         f"win_rate={metrics['win_rate']}%, "
                         f"sharpe={metrics['sharpe']}, "
                         f"PF={metrics['profit_factor']}, "
                         f"total_ret={metrics['total_return']}%")
            if regime not in backtest_results:
                backtest_results[regime] = {}
            backtest_results[regime]["gbm"] = metrics

            if not bt.empty:
                bt["regime"] = regime
                all_trades.append(bt)
                bt_path = results_dir / f"trades_{regime}_gbm.csv"
                bt.to_csv(bt_path, index=False)

    # 汇总全部 regime 的交易
    if all_trades:
        all_trades_df = pd.concat(all_trades, ignore_index=True)
        overall_metrics = compute_backtest_metrics(all_trades_df)
        backtest_results["overall"] = overall_metrics
        logger.info(f"\n  ── Overall Backtest ──")
        logger.info(f"  signals={overall_metrics['n_signals']}, "
                     f"win_rate={overall_metrics['win_rate']}%, "
                     f"sharpe={overall_metrics['sharpe']}, "
                     f"PF={overall_metrics['profit_factor']}, "
                     f"total_ret={overall_metrics['total_return']}%, "
                     f"max_dd={overall_metrics['max_drawdown']}%")

        all_bt_path = results_dir / "trades_all_gbm.csv"
        all_trades_df.to_csv(all_bt_path, index=False)

    all_results["backtest_by_regime"] = backtest_results

    # ── 6. Save summary ──
    logger.info("\n[6/7] Saving results...")

    ts_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "timestamp": ts_str,
        "symbol": args.symbol,
        "freq": args.freq,
        "target": target_col,
        "tb_params": {
            "upper_pct": args.tb_upper,
            "lower_pct": args.tb_lower,
            "max_bars": args.tb_horizon,
        },
        "regime_dates": regime_dates,
        "train_label_distribution": {str(k): int(v) for k, v in label_dist.items()},
        "n_features": len(feature_cols),
        "results": all_results,
    }

    summary_path = results_dir / f"summary_{ts_str}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\nResults saved: {summary_path}")
    logger.info("Done.")


def _load_ohlcv_for_dates(
    symbol: str, dates: List[str], freq: str,
) -> Optional[pd.DataFrame]:
    """为指定日期加载 OHLCV。"""
    all_dfs = []
    for d_str in dates:
        dt = date.fromisoformat(d_str)
        trades = load_agg_trades(symbol, dt, dt + timedelta(days=1))
        if trades.empty:
            continue
        ohlcv = resample_trades_to_ohlcv(trades, freq)
        all_dfs.append(ohlcv)

    if not all_dfs:
        return None
    return pd.concat(all_dfs).sort_index()


if __name__ == "__main__":
    main()
