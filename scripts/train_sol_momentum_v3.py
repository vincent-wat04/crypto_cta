#!/usr/bin/env python3
"""
SOL Momentum / Reversal — V3  (Unified Experiment Framework)

V3 核心设计:
  1. 特征在 1s 粒度计算，保持物理窗口 [3, 5, 10, 20]s
  2. 每个交易周期开头取最新 1s 特征做一次预测，周期内仓位不变
     → 彻底消除过度交易问题
  3. 3 种交易间隔: 1s / 1min / 5min
  4. 3 种预测目标:
     (a) return   — 未来 N 秒的连续收益率 (regression)
     (b) binary   — 未来收益 >0 为 1, ≤0 为 0 (binary classification)
     (c) tbm      — Triple Barrier Method: +1(上轨), -1(下轨), 0(超时)
  5. 模型: Ridge/Logistic + GBM (regression & classification variants)
  6. 所有实验结果放入统一 summary 进行对比

Usage:
    python scripts/train_sol_momentum_v3.py                       # 跑全部 9 组
    python scripts/train_sol_momentum_v3.py --interval 300        # 只跑 5min
    python scripts/train_sol_momentum_v3.py --interval 60 --target tbm
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utilities.binance_loader import load_agg_trades, resample_trades_to_ohlcv
from utilities.paths import DataPaths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ─── Config ───
SOL_TRAIN_DATES = [
    "2026-01-28", "2026-01-29", "2026-01-30", "2026-01-31",
    "2026-02-01", "2026-02-02", "2026-02-03",
]
SOL_TEST_DATES = [
    "2026-02-04", "2026-02-05", "2026-02-06",
]
FEATURE_WINDOWS = [3, 5, 10, 20]  # 秒 — 微观结构物理窗口


# ═══════════════════════════════════════════════════════════
# 1. Feature Computation (1s bars)
# ═══════════════════════════════════════════════════════════

def compute_1s_features(trades: pd.DataFrame) -> pd.DataFrame:
    """aggTrades → 1s OHLCV → 特征 (窗口 [3,5,10,20]s)。"""
    ohlcv = resample_trades_to_ohlcv(trades, "1s")
    if len(ohlcv) < 100:
        return pd.DataFrame()

    f = pd.DataFrame(index=ohlcv.index)
    c = ohlcv["close"].astype(float)
    o = ohlcv["open"].astype(float)
    h = ohlcv["high"].astype(float)
    lo = ohlcv["low"].astype(float)
    v = ohlcv["volume"].astype(float)
    bv = ohlcv["buy_volume"].astype(float)
    sv = ohlcv["sell_volume"].astype(float)
    nt = ohlcv.get("n_trades", pd.Series(0, index=ohlcv.index)).astype(float)

    # ── Return ──
    f["return_1"] = c.pct_change() * 10000
    f["log_return"] = np.log(c / c.shift(1))
    f["close"] = c  # 保留 close 用于 TBM

    for w in FEATURE_WINDOWS:
        f[f"return_sum_{w}"] = f["return_1"].rolling(w, min_periods=1).sum()
        f[f"return_std_{w}"] = f["return_1"].rolling(w, min_periods=2).std()
        f[f"return_skew_{w}"] = f["return_1"].rolling(w, min_periods=3).skew()

    # ── Volume ──
    f["volume"] = v
    f["buy_volume"] = bv
    f["sell_volume"] = sv
    f["volume_imbalance"] = (bv - sv) / (v + 1e-10)
    f["n_trades"] = nt

    for w in FEATURE_WINDOWS:
        f[f"volume_ma_{w}"] = v.rolling(w, min_periods=1).mean()
        f[f"volume_ratio_{w}"] = v / (f[f"volume_ma_{w}"] + 1e-10)
        f[f"imbalance_ma_{w}"] = f["volume_imbalance"].rolling(w, min_periods=1).mean()
        signed_vol = bv - sv
        f[f"signed_flow_{w}"] = signed_vol.rolling(w, min_periods=1).sum()
        f[f"signed_flow_norm_{w}"] = f[f"signed_flow_{w}"] / (v.rolling(w, min_periods=1).sum() + 1e-10)

    # ── Bar structure ──
    hl = h - lo
    f["bar_range_bps"] = hl / (c + 1e-10) * 10000
    f["body_ratio"] = (c - o).abs() / (hl + 1e-10)
    f["bar_direction"] = np.sign(c - o)

    for w in FEATURE_WINDOWS:
        f[f"bar_range_ma_{w}"] = f["bar_range_bps"].rolling(w, min_periods=1).mean()
        f[f"trades_ma_{w}"] = nt.rolling(w, min_periods=1).mean()

    # ── Spread / Impact proxy ──
    price_diff = c.diff().abs()
    for w in FEATURE_WINDOWS:
        f[f"spread_proxy_{w}"] = price_diff.rolling(w, min_periods=1).mean()
        cum_move = price_diff.rolling(w, min_periods=1).sum()
        cum_vol = v.rolling(w, min_periods=1).sum() + 1e-10
        f[f"impact_proxy_{w}"] = cum_move / cum_vol

    # ── Volume autocorrelation ──
    for w in [5, 10, 20]:
        f[f"buy_vol_autocorr_{w}"] = bv.rolling(w, min_periods=3).corr(bv.shift(1))
        f[f"sell_vol_autocorr_{w}"] = sv.rolling(w, min_periods=3).corr(sv.shift(1))
        f[f"bv_sv_corr_{w}"] = bv.rolling(w, min_periods=3).corr(sv)

    f = f.replace([np.inf, -np.inf], np.nan).fillna(0)
    return f


def add_taker_features(f: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """taker order 聚合 → 1s bars。"""
    from indicators.base.taker_flow import group_taker_orders

    taker = group_taker_orders(trades)
    if taker.empty or len(taker) < 10:
        return f
    taker_ts = taker.set_index("timestamp").sort_index()
    taker_ts = taker_ts[~taker_ts.index.duplicated(keep="last")]

    t_levels = taker_ts["levels_swept"].resample("1s").mean()
    t_volume = taker_ts["total_volume"].resample("1s").sum()
    t_impact = taker_ts["price_impact_pct"].resample("1s").mean()
    t_count = taker_ts["levels_swept"].resample("1s").count()
    buy_t = taker_ts.loc[taker_ts["side"] == "buy", "total_volume"].resample("1s").sum()
    sell_t = taker_ts.loc[taker_ts["side"] == "sell", "total_volume"].resample("1s").sum()
    buy_t = buy_t.reindex(t_levels.index).fillna(0)
    sell_t = sell_t.reindex(t_levels.index).fillna(0)

    f["taker_levels_swept"] = t_levels.reindex(f.index).fillna(0)
    f["taker_volume"] = t_volume.reindex(f.index).fillna(0)
    f["taker_impact_pct"] = t_impact.reindex(f.index).fillna(0)
    f["taker_count"] = t_count.reindex(f.index).fillna(0)
    f["taker_imbalance"] = ((buy_t - sell_t) / (buy_t + sell_t + 1e-10)).reindex(f.index).fillna(0)

    for w in [3, 5, 10]:
        f[f"taker_levels_ma_{w}"] = f["taker_levels_swept"].rolling(w, min_periods=1).mean()
        f[f"taker_imb_ma_{w}"] = f["taker_imbalance"].rolling(w, min_periods=1).mean()

    f = f.replace([np.inf, -np.inf], np.nan).fillna(0)
    return f


# ═══════════════════════════════════════════════════════════
# 2. Data Loading
# ═══════════════════════════════════════════════════════════

def load_1s_features(symbol: str, dates: List[str]) -> pd.DataFrame:
    all_dfs = []
    for d_str in dates:
        dt = date.fromisoformat(d_str)
        cache_path = DataPaths.features(symbol, "1s_features_v3", dt)
        if cache_path.exists():
            logger.info(f"  [Cache] {d_str}")
            all_dfs.append(pd.read_parquet(cache_path))
        else:
            logger.info(f"  [Compute] {d_str}...")
            trades = load_agg_trades(symbol, dt, dt + timedelta(days=1))
            if trades.empty:
                continue
            logger.info(f"    {len(trades):,} aggTrades")
            df = compute_1s_features(trades)
            if df.empty:
                del trades; gc.collect(); continue
            df = add_taker_features(df, trades)
            df.to_parquet(cache_path)
            logger.info(f"    → {len(df):,} bars × {df.shape[1]} features")
            all_dfs.append(df)
            del trades; gc.collect()
    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs).sort_index()


# ═══════════════════════════════════════════════════════════
# 3. Target Computation
# ═══════════════════════════════════════════════════════════

def make_target_return(df: pd.DataFrame, interval: int) -> pd.Series:
    """连续收益率 (bps): cumulative return over next `interval` seconds."""
    cum = df["return_1"].cumsum()
    return cum.shift(-interval) - cum


def make_target_binary(df: pd.DataFrame, interval: int) -> pd.Series:
    """二分类: 1 if forward return > 0, else 0."""
    fwd = make_target_return(df, interval)
    return (fwd > 0).astype(int)


def make_target_tbm(df: pd.DataFrame, interval: int,
                    tp_mult: float = 1.5, sl_mult: float = 1.5) -> pd.Series:
    """
    Triple Barrier Method:
      +1: 先触碰上轨 (tp)
      -1: 先触碰下轨 (sl)
       0: 超时未触碰
    上下轨 = rolling std × multiplier (自适应波动率)
    """
    close = df["close"].values
    ret_std = df["return_1"].rolling(max(20, interval), min_periods=10).std().values
    n = len(close)
    labels = np.full(n, np.nan)

    for i in range(n - interval):
        if np.isnan(ret_std[i]) or ret_std[i] < 1e-10:
            labels[i] = 0
            continue
        # 自适应 barrier 基于近期波动
        vol = ret_std[i] * np.sqrt(interval)  # bps, scaled by sqrt(horizon)
        tp_bps = vol * tp_mult
        sl_bps = vol * sl_mult
        p0 = close[i]
        if p0 < 1e-10:
            labels[i] = 0
            continue

        hit = 0
        for j in range(1, interval + 1):
            ret = (close[i + j] - p0) / p0 * 10000
            if ret >= tp_bps:
                hit = 1; break
            elif ret <= -sl_bps:
                hit = -1; break
        labels[i] = hit

    return pd.Series(labels, index=df.index)


# ═══════════════════════════════════════════════════════════
# 4. Subsample to Trading Intervals
# ═══════════════════════════════════════════════════════════

def subsample_to_interval(df: pd.DataFrame, interval: int) -> pd.DataFrame:
    """
    从 1s bars 中每隔 interval 秒取一个样本。
    保留该时刻的特征作为交易决策输入。
    """
    return df.iloc[::interval].copy()


# ═══════════════════════════════════════════════════════════
# 5. Feature Selection
# ═══════════════════════════════════════════════════════════

def select_features(X, y, names, top_n=40, task="reg"):
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    Xn = sc.fit_transform(X)

    if task == "reg":
        from sklearn.linear_model import Ridge
        from sklearn.inspection import permutation_importance
        m = Ridge(alpha=1.0).fit(Xn, y)
        val_n = max(200, len(X) // 5)
        perm = permutation_importance(m, Xn[-val_n:], y[-val_n:],
                                       n_repeats=5, random_state=42, scoring="r2")
    else:
        from sklearn.linear_model import LogisticRegression
        from sklearn.inspection import permutation_importance
        m = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        m.fit(Xn, y)
        val_n = max(200, len(X) // 5)
        scoring = "accuracy" if len(np.unique(y)) == 2 else "balanced_accuracy"
        perm = permutation_importance(m, Xn[-val_n:], y[-val_n:],
                                       n_repeats=5, random_state=42, scoring=scoring)

    imp = perm.importances_mean
    top_idx = np.argsort(imp)[::-1][:top_n]
    selected = [names[i] for i in top_idx]

    logger.info(f"  Top features ({task}):")
    for rank, i in enumerate(top_idx[:10]):
        logger.info(f"    {rank+1:>3}. {names[i]:<30} imp={imp[i]:.6f}")
    return selected


# ═══════════════════════════════════════════════════════════
# 6. Models
# ═══════════════════════════════════════════════════════════

def train_regression(X_tr, y_tr, X_te, model_type="ridge"):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_tr)
    Xte = sc.transform(X_te)

    if model_type == "ridge":
        from sklearn.linear_model import Ridge
        m = Ridge(alpha=10.0).fit(Xtr, y_tr)
    else:  # gbm
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(
            max_iter=300, max_depth=3, learning_rate=0.03,
            min_samples_leaf=20, l2_regularization=5.0,
            early_stopping=True, validation_fraction=0.2,
            n_iter_no_change=20, random_state=42,
        ).fit(Xtr, y_tr)

    pred_tr = m.predict(Xtr)
    pred_te = m.predict(Xte)
    ss_res = np.sum((y_tr - pred_tr) ** 2)
    ss_tot = np.sum((y_tr - y_tr.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    name = model_type.upper()
    if hasattr(m, "n_iter_"):
        logger.info(f"  [{name}] Train R²={r2:.4f}, n_iter={m.n_iter_}")
    else:
        logger.info(f"  [{name}] Train R²={r2:.4f}")
    return pred_te


def train_classifier(X_tr, y_tr, X_te, model_type="logistic"):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_tr)
    Xte = sc.transform(X_te)

    if model_type == "logistic":
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(C=0.1, max_iter=2000, random_state=42,
                               multi_class="multinomial" if len(np.unique(y_tr)) > 2 else "auto")
        m.fit(Xtr, y_tr)
    else:  # gbm
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(
            max_iter=300, max_depth=3, learning_rate=0.03,
            min_samples_leaf=20, l2_regularization=5.0,
            early_stopping=True, validation_fraction=0.2,
            n_iter_no_change=20, random_state=42,
        ).fit(Xtr, y_tr)

    acc_tr = m.score(Xtr, y_tr)
    acc_te = m.score(Xte, y_tr[:len(Xte)] if len(y_tr) >= len(Xte) else y_tr)
    name = model_type.upper()
    logger.info(f"  [{name}] Train acc={acc_tr:.4f}")

    # 返回 signed signal: +1 for long, -1 for short, 0 for flat
    classes = m.classes_
    proba = m.predict_proba(Xte)

    if len(classes) == 2:
        # binary: P(up) - 0.5 → positive = long, negative = short
        up_idx = np.where(classes == 1)[0][0]
        signal = proba[:, up_idx] - 0.5
    else:
        # TBM 3-class: signal = P(+1) - P(-1)
        p_up = proba[:, np.where(classes == 1)[0][0]] if 1 in classes else 0
        p_dn = proba[:, np.where(classes == -1)[0][0]] if -1 in classes else 0
        signal = p_up - p_dn

    return signal


# ═══════════════════════════════════════════════════════════
# 7. Fixed-Interval Backtest
# ═══════════════════════════════════════════════════════════

def backtest_fixed_interval(
    signal: np.ndarray,
    actual_return: np.ndarray,
    timestamps,
    cost_bps: float = 4.0,
    threshold: float = 0.0,
    task: str = "reg",
) -> pd.DataFrame:
    """
    固定间隔回测：每个周期开头下单，周期内仓位不变。

    For regression: position = sign(signal) if |signal| > threshold
    For classification: position = sign(signal) if |signal| > threshold

    actual_return = 该周期的实际 return (bps)
    """
    n = len(signal)
    rt_cost = cost_bps * 2

    trades = []
    cum_pnl = 0.0
    prev_pos = 0

    for i in range(n):
        s = signal[i]
        if np.isnan(s):
            pos = 0
        elif abs(s) > threshold:
            pos = 1 if s > 0 else -1
        else:
            pos = 0

        gross = actual_return[i] * pos if not np.isnan(actual_return[i]) else 0.0
        cost = rt_cost if pos != prev_pos and (pos != 0 or prev_pos != 0) else 0.0
        net = gross - cost
        cum_pnl += net

        trades.append({
            "ts": timestamps[i] if i < len(timestamps) else None,
            "position": pos,
            "signal": float(s) if not np.isnan(s) else 0.0,
            "actual_return": float(actual_return[i]) if not np.isnan(actual_return[i]) else 0.0,
            "gross_pnl": float(gross),
            "cost": float(cost),
            "net_pnl": float(net),
            "cum_pnl": float(cum_pnl),
        })
        prev_pos = pos

    return pd.DataFrame(trades)


def compute_bt_metrics(df: pd.DataFrame) -> Dict:
    """从逐周期回测记录计算指标。"""
    if df.empty or len(df) < 2:
        return {"n_periods": 0, "n_trades": 0, "total_pnl": 0, "gross_pnl": 0,
                "sharpe": 0, "win_rate": 0, "profit_factor": 0, "max_dd": 0,
                "avg_ret_per_trade": 0, "pct_in_market": 0}

    active = df[df["position"] != 0]
    n_trades = int((df["position"].diff().fillna(0) != 0).sum())
    pnl = df["net_pnl"]
    gross = df["gross_pnl"]

    # Sharpe: annualize based on number of periods
    ret_mean = pnl.mean()
    ret_std = pnl.std() + 1e-10
    sr = ret_mean / ret_std * np.sqrt(len(pnl))

    # Win rate (on active periods)
    if len(active) > 0:
        wins = active[active["net_pnl"] > 0]
        wr = len(wins) / len(active)
    else:
        wr = 0

    # Profit factor
    pos_pnl = pnl[pnl > 0].sum()
    neg_pnl = pnl[pnl < 0].abs().sum() + 1e-10
    pf = pos_pnl / neg_pnl

    # Max drawdown
    cum = df["cum_pnl"]
    dd = cum - cum.cummax()

    return {
        "n_periods": len(df),
        "n_trades": n_trades,
        "n_active": len(active),
        "pct_in_market": len(active) / len(df) if len(df) > 0 else 0,
        "total_pnl": float(pnl.sum()),
        "gross_pnl": float(gross.sum()),
        "total_cost": float(df["cost"].sum()),
        "sharpe": float(sr),
        "win_rate": float(wr),
        "profit_factor": float(pf),
        "max_dd": float(dd.min()),
        "avg_ret_per_trade": float(active["net_pnl"].mean()) if len(active) > 0 else 0,
    }


# ═══════════════════════════════════════════════════════════
# 8. Single Experiment
# ═══════════════════════════════════════════════════════════

def run_experiment(
    train_1s: pd.DataFrame,
    test_1s: pd.DataFrame,
    interval: int,
    target_type: str,
    cost_bps: float,
    top_features: int = 40,
) -> Dict:
    """
    运行一组实验:
      interval: 交易间隔 (秒)
      target_type: "return" / "binary" / "tbm"
    """
    interval_str = {1: "1s", 60: "1min", 300: "5min"}.get(interval, f"{interval}s")
    logger.info(f"\n{'─'*60}")
    logger.info(f"  Experiment: interval={interval_str}, target={target_type}")
    logger.info(f"{'─'*60}")

    # ── Compute targets on 1s data ──
    if target_type == "return":
        train_1s["_target"] = make_target_return(train_1s, interval)
        test_1s["_target"] = make_target_return(test_1s, interval)
        task = "reg"
    elif target_type == "binary":
        train_1s["_target"] = make_target_binary(train_1s, interval)
        test_1s["_target"] = make_target_binary(test_1s, interval)
        task = "cls"
    elif target_type == "tbm":
        logger.info("  Computing TBM labels (may take a moment)...")
        train_1s["_target"] = make_target_tbm(train_1s, interval)
        test_1s["_target"] = make_target_tbm(test_1s, interval)
        task = "cls"
    else:
        raise ValueError(f"Unknown target: {target_type}")

    # ── Subsample to trading interval ──
    train_sub = subsample_to_interval(train_1s, interval)
    test_sub = subsample_to_interval(test_1s, interval)

    train_sub = train_sub.dropna(subset=["_target"])
    test_sub = test_sub.dropna(subset=["_target"])

    y_train = train_sub["_target"].values
    y_test = test_sub["_target"].values

    logger.info(f"  Train: {len(train_sub):,} samples, Test: {len(test_sub):,} samples")

    if task == "cls":
        unique, counts = np.unique(y_train, return_counts=True)
        dist = {int(u): int(c) for u, c in zip(unique, counts)}
        logger.info(f"  Label distribution (train): {dist}")

    # ── Actual interval return for backtest ──
    actual_ret_train = make_target_return(train_1s, interval)
    actual_ret_test = make_target_return(test_1s, interval)
    actual_ret_test_sub = actual_ret_test.iloc[::interval].values[:len(test_sub)]

    # ── Feature columns ──
    exclude = {"_target", "return_1", "log_return", "close"}
    feat_cols = [
        c for c in train_sub.columns
        if c not in exclude
        and train_sub[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]
        and train_sub[c].std() > 1e-10
    ]
    logger.info(f"  Features: {len(feat_cols)}")

    # ── Feature selection ──
    selected = select_features(
        train_sub[feat_cols].values, y_train.astype(float), feat_cols,
        top_n=top_features, task=task,
    )

    X_train = train_sub[selected].values
    X_test = test_sub[selected].values

    # ── Run models ──
    results = {}

    model_pairs = [("ridge", "gbm")] if task == "reg" else [("logistic", "gbm")]

    for m1, m2 in model_pairs:
        for mtype in [m1, m2]:
            logger.info(f"\n  Training {mtype.upper()} ({task})...")
            t0 = time.time()

            if task == "reg":
                signal = train_regression(X_train, y_train, X_test, model_type=mtype)
            else:
                signal = train_classifier(X_train, y_train, X_test, model_type=mtype)

            dt_train = time.time() - t0
            logger.info(f"    Time: {dt_train:.1f}s")

            # Signal stats
            logger.info(f"    Signal: mean={signal.mean():.4f}, std={signal.std():.4f}")

            # IC (for regression)
            if task == "reg":
                mask = ~(np.isnan(signal) | np.isnan(y_test))
                if mask.sum() > 10:
                    ic = np.corrcoef(signal[mask], y_test[mask])[0, 1]
                    logger.info(f"    IC={ic:.4f}")

            # ── Backtest with multiple thresholds ──
            if task == "reg":
                thresholds = [0, 0.5, 1.0, 2.0]
            else:
                thresholds = [0, 0.05, 0.1, 0.2]

            for cost in [cost_bps, 1.0]:  # taker and maker
                cost_label = "taker" if cost == cost_bps else "maker"
                for thr in thresholds:
                    bt = backtest_fixed_interval(
                        signal, actual_ret_test_sub[:len(signal)],
                        test_sub.index[:len(signal)],
                        cost_bps=cost, threshold=thr, task=task,
                    )
                    met = compute_bt_metrics(bt)
                    key = f"{mtype}_{cost_label}_thr{thr}"
                    results[key] = met

                    logger.info(
                        f"    {key:<30} N={met['n_trades']:>4} Act={met['n_active']:>5} "
                        f"Gross={met['gross_pnl']:>+7.0f} Net={met['total_pnl']:>+7.0f} "
                        f"WR={met['win_rate']:.1%} SR={met['sharpe']:>+6.2f} "
                        f"PF={met['profit_factor']:.2f}"
                    )

    # Clean up
    train_1s.drop(columns=["_target"], inplace=True, errors="ignore")
    test_1s.drop(columns=["_target"], inplace=True, errors="ignore")

    # Best config
    best_key = max(results, key=lambda k: results[k]["sharpe"])
    best_met = results[best_key]
    logger.info(f"\n  ★ Best: {best_key} (Sharpe={best_met['sharpe']:.2f})")

    return {
        "interval": interval,
        "interval_str": interval_str,
        "target": target_type,
        "train_samples": len(train_sub),
        "test_samples": len(test_sub),
        "best_config": best_key,
        "best_sharpe": best_met["sharpe"],
        "results": results,
    }


# ═══════════════════════════════════════════════════════════
# 9. Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SOL/USDT")
    parser.add_argument("--interval", type=int, default=0,
                        help="Trading interval in seconds (0=all: 1,60,300)")
    parser.add_argument("--target", default="all",
                        choices=["return", "binary", "tbm", "all"])
    parser.add_argument("--cost-bps", type=float, default=4.0)
    parser.add_argument("--top-features", type=int, default=40)
    args = parser.parse_args()

    intervals = [1, 60, 300] if args.interval == 0 else [args.interval]
    targets = ["return", "binary", "tbm"] if args.target == "all" else [args.target]

    logger.info("=" * 65)
    logger.info("SOL V3 — Unified Experiment Framework")
    logger.info(f"  Intervals: {intervals} seconds")
    logger.info(f"  Targets: {targets}")
    logger.info(f"  Feature windows: {FEATURE_WINDOWS}s (physical, fixed)")
    logger.info(f"  Cost: {args.cost_bps} bps/leg")
    logger.info(f"  Train: {len(SOL_TRAIN_DATES)} days, Test: {len(SOL_TEST_DATES)} days")
    logger.info("=" * 65)

    # ── Load 1s features ──
    logger.info("\n[1] Loading 1s features...")
    train_1s = load_1s_features(args.symbol, SOL_TRAIN_DATES)
    test_1s = load_1s_features(args.symbol, SOL_TEST_DATES)
    logger.info(f"  Train: {len(train_1s):,} × {train_1s.shape[1]} cols")
    logger.info(f"  Test:  {len(test_1s):,} × {test_1s.shape[1]} cols")

    # ── Run all experiments ──
    all_experiments = []

    for interval in intervals:
        for target in targets:
            exp = run_experiment(
                train_1s, test_1s, interval, target,
                cost_bps=args.cost_bps, top_features=args.top_features,
            )
            all_experiments.append(exp)

    # ═══ Grand Summary ═══
    logger.info(f"\n{'='*80}")
    logger.info("GRAND SUMMARY — All Experiments")
    logger.info(f"{'='*80}")
    logger.info(f"{'Interval':<8} {'Target':<8} {'Best Config':<35} {'SR':>6} {'Net PnL':>8} {'Trades':>6} {'WR':>5}")
    logger.info(f"{'-'*80}")

    summary_rows = []
    for exp in all_experiments:
        best = exp["results"][exp["best_config"]]
        row = {
            "interval": exp["interval_str"],
            "target": exp["target"],
            "best_config": exp["best_config"],
            "sharpe": best["sharpe"],
            "net_pnl": best["total_pnl"],
            "n_trades": best["n_trades"],
            "win_rate": best["win_rate"],
            "gross_pnl": best["gross_pnl"],
            "profit_factor": best["profit_factor"],
            "max_dd": best["max_dd"],
        }
        summary_rows.append(row)
        logger.info(
            f"{row['interval']:<8} {row['target']:<8} {row['best_config']:<35} "
            f"{row['sharpe']:>+6.2f} {row['net_pnl']:>+8.0f} {row['n_trades']:>6} "
            f"{row['win_rate']:>4.1%}"
        )

    # Also include V1 results for comparison
    logger.info(f"\n{'─'*80}")
    logger.info("For reference — V1 (5min bars, feature windows [2,3,6,12]×5min):")
    logger.info("  Ridge maker_q60_h12:  SR=+1.23, Net=+1817, N=161, WR=54.0%")
    logger.info("  GBM   maker_q80_h24:  SR=+1.17, Net=+2054, N= 67, WR=46.3%")
    logger.info(f"{'─'*80}")

    # Save
    save_data = {
        "config": {
            "version": "v3",
            "symbol": args.symbol,
            "intervals": intervals,
            "targets": targets,
            "feature_windows": FEATURE_WINDOWS,
            "cost_bps": args.cost_bps,
            "train_dates": SOL_TRAIN_DATES,
            "test_dates": SOL_TEST_DATES,
        },
        "experiments": [{
            "interval": e["interval_str"],
            "target": e["target"],
            "best_config": e["best_config"],
            "best_sharpe": e["best_sharpe"],
            "train_samples": e["train_samples"],
            "test_samples": e["test_samples"],
            "all_results": e["results"],
        } for e in all_experiments],
        "summary": summary_rows,
    }

    sp = DataPaths.backtest_dir("sol_v3") / "summary.json"
    with open(sp, "w") as fp:
        json.dump(save_data, fp, indent=2, default=str)
    logger.info(f"\nSaved: {sp}")


if __name__ == "__main__":
    main()
