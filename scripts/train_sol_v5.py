#!/usr/bin/env python3
"""
SOL Unified Experiment — V5 (Full Indicators Library)

V5 核心改进:
  1. 特征全面使用 indicators/ 库 (base + regime)，而非手写 inline 特征
     → price_impact, spread, order_flow, volume_profile, taker_flow
     → trade_microstructure, returns_momentum, bar_structure, tick_statistics
     → regime: realized_vol, parkinson_vol, garch_vol, adx, efficiency_ratio
     → regime: amihud, turnover
  2. 物理窗口 [3,5,10,20]s 不变
  3. 1s/1min/5min 预测间隔，仓位周期内不变
  4. 目标: return regression + TBM (tp/sl 参数可调)
  5. 模型: Lasso/Ridge/GBM (regression), Logistic/GBM (classification)

Usage:
    python scripts/train_sol_v5.py                     # 全部实验
    python scripts/train_sol_v5.py --interval 60       # 只跑 1min
    python scripts/train_sol_v5.py --target tbm        # 只跑 TBM
    python scripts/train_sol_v5.py --quick              # 快速模式 (1min + 5min)
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
from typing import Dict, List

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
TRAIN_DATES = [
    "2026-01-22", "2026-01-23", "2026-01-24", "2026-01-25",
    "2026-01-26", "2026-01-27", "2026-01-28", "2026-01-29",
    "2026-01-30", "2026-01-31", "2026-02-01", "2026-02-02", "2026-02-03",
]
TEST_DATES = ["2026-02-04", "2026-02-05", "2026-02-06"]

FEATURE_WINDOWS = [3, 5, 10, 20]  # 秒 — 物理窗口

# TBM 参数 (V4 最优组合)
TBM_CONFIGS = [
    (1.0, 0.5),   # V4 best PnL: 正常止盈 + 快速止损
    (1.0, 1.0),   # V4 best Sharpe: 对称中等
    (0.5, 0.5),   # 宽松对称: 更多趋势信号
]


# ═══════════════════════════════════════════════════════════
# 1. Comprehensive Feature Computation — 使用 indicators 库
# ═══════════════════════════════════════════════════════════

def compute_all_features_v5(trades: pd.DataFrame, freq: str = "1s") -> pd.DataFrame:
    """
    从原始 aggTrades 计算所有特征，全面使用 indicators/ 库。

    Feature categories:
      A) Returns & Momentum (returns_momentum.py)
      B) Bar Structure (bar_structure.py)
      C) Tick Statistics (tick_statistics.py)
      D) Trade Microstructure (trade_microstructure.py)
      E) Price Impact (price_impact.py)
      F) Spread (spread.py)
      G) Order Flow (order_flow.py)
      H) Volume Profile (volume_profile.py)
      I) Taker Flow (taker_flow.py)
      J) Regime (volatility, trend, liquidity)

    Returns: DataFrame at `freq` frequency with 200+ features
    """
    windows = FEATURE_WINDOWS

    # ── OHLCV at target frequency (computed once) ──
    ohlcv = resample_trades_to_ohlcv(trades, freq)
    if len(ohlcv) < 100:
        return pd.DataFrame()

    close = ohlcv["close"].astype(float)
    f = pd.DataFrame(index=ohlcv.index)
    f["close"] = close  # needed for TBM

    # ══════ A) Returns & Momentum ══════
    logger.info("    [A] Returns & momentum...")
    from indicators.base.returns_momentum import compute_returns
    ret = compute_returns(ohlcv, windows)
    for col in ret.columns:
        f[col] = ret[col].reindex(f.index)

    # ══════ B) Bar Structure ══════
    logger.info("    [B] Bar structure...")
    from indicators.base.bar_structure import compute_bar_structure
    bar = compute_bar_structure(ohlcv, windows)
    for col in bar.columns:
        f[col] = bar[col].reindex(f.index)

    # ══════ C) Tick-Level Statistics ══════
    logger.info("    [C] Tick-level stats...")
    from indicators.base.tick_statistics import compute_tick_stats
    tick = compute_tick_stats(trades, freq, windows)
    for col in tick.columns:
        f[col] = tick[col].reindex(f.index)

    # ══════ D) Trade Microstructure ══════
    logger.info("    [D] Trade microstructure...")
    from indicators.base.trade_microstructure import (
        avg_fill_size, trade_side_autocorrelation, trade_intensity, vwap_deviation,
    )
    # avg_fill_size
    fs = avg_fill_size(trades, freq)
    f["avg_fill_size"] = fs.reindex(f.index).fillna(0)
    for w in windows:
        f[f"avg_fill_size_ma_{w}"] = f["avg_fill_size"].rolling(w, min_periods=1).mean()

    # trade_side_autocorrelation
    tsa = trade_side_autocorrelation(trades, freq, window=20, lag=1)
    for col in tsa.columns:
        s = tsa[col].reindex(f.index).fillna(0)
        f[f"tsa_{col}"] = s
        for w in [5, 10, 20]:
            f[f"tsa_{col}_ma_{w}"] = s.rolling(w, min_periods=1).mean()

    # trade_intensity
    ti = trade_intensity(trades, freq)
    for col in ti.columns:
        f[f"ti_{col}"] = ti[col].reindex(f.index).fillna(0)

    # vwap_deviation
    vd = vwap_deviation(trades, freq)
    for col in vd.columns:
        s = vd[col].reindex(f.index).fillna(0)
        f[f"vd_{col}"] = s
        for w in [5, 10]:
            f[f"vd_{col}_ma_{w}"] = s.rolling(w, min_periods=1).mean()

    # ══════ E) Price Impact ══════
    logger.info("    [E] Price impact...")
    from indicators.base.price_impact import tick_price_impact, volume_weighted_impact

    pi = tick_price_impact(trades, window_ms=3000)
    pi_resampled = pi[~pi.index.duplicated(keep="last")].resample(freq).last()
    f["tick_price_impact"] = pi_resampled.reindex(f.index).fillna(0)

    vwi = volume_weighted_impact(trades, window_ms=5000)
    vwi_resampled = vwi[~vwi.index.duplicated(keep="last")].resample(freq).last()
    f["vw_impact"] = vwi_resampled.reindex(f.index).fillna(0)

    for col in ["tick_price_impact", "vw_impact"]:
        for w in windows:
            f[f"{col}_ma_{w}"] = f[col].rolling(w, min_periods=1).mean()

    # ══════ F) Spread ══════
    logger.info("    [F] Spread estimators...")
    from indicators.base.spread import trade_diff_spread, roll_spread

    tds = trade_diff_spread(trades, window_trades=50)
    tds_r = tds[~tds.index.duplicated(keep="last")].resample(freq).last()
    f["trade_diff_spread"] = tds_r.reindex(f.index).fillna(0)

    rs = roll_spread(trades, window_trades=100)
    rs_r = rs[~rs.index.duplicated(keep="last")].resample(freq).last()
    f["roll_spread"] = rs_r.reindex(f.index).fillna(0)

    for col in ["trade_diff_spread", "roll_spread"]:
        for w in windows:
            f[f"{col}_ma_{w}"] = f[col].rolling(w, min_periods=1).mean()

    # Spread proxy from close (补充)
    price_diff = close.diff().abs()
    for w in windows:
        f[f"spread_proxy_{w}"] = price_diff.rolling(w, min_periods=1).mean()
        cum_move = price_diff.rolling(w, min_periods=1).sum()
        cum_vol = ohlcv["volume"].rolling(w, min_periods=1).sum() + 1e-10
        f[f"impact_proxy_{w}"] = cum_move / cum_vol

    # ══════ G) Order Flow ══════
    logger.info("    [G] Order flow...")
    from indicators.base.order_flow import trade_imbalance, flow_toxicity

    timb = trade_imbalance(trades, window_trades=50)
    timb_r = timb[~timb.index.duplicated(keep="last")].resample(freq).last()
    f["trade_imbalance_lib"] = timb_r.reindex(f.index).fillna(0)

    ft = flow_toxicity(trades, window_trades=100)
    ft_r = ft[~ft.index.duplicated(keep="last")].resample(freq).last()
    f["flow_toxicity"] = ft_r.reindex(f.index).fillna(0)

    for col in ["trade_imbalance_lib", "flow_toxicity"]:
        for w in windows:
            f[f"{col}_ma_{w}"] = f[col].rolling(w, min_periods=1).mean()

    # ══════ H) Volume Profile ══════
    logger.info("    [H] Volume profile...")
    from indicators.base.volume_profile import (
        taker_volume_corr, taker_volume_autocorr,
        taker_volume_skewness, buy_sell_volume_ratio,
    )

    tvc = taker_volume_corr(trades, resample_freq=freq, window=20)
    f["taker_vol_corr"] = tvc.reindex(f.index).fillna(0)

    for side in ["buy", "sell"]:
        tva = taker_volume_autocorr(trades, resample_freq=freq, window=20, side=side)
        f[f"taker_vol_autocorr_{side}"] = tva.reindex(f.index).fillna(0)

    tvs = taker_volume_skewness(trades, resample_freq=freq, window=60)
    f["taker_vol_skewness"] = tvs.reindex(f.index).fillna(0)

    bsvr = buy_sell_volume_ratio(trades, resample_freq=freq, window=20)
    f["buy_sell_vol_ratio"] = bsvr.reindex(f.index).fillna(0)

    vp_cols = ["taker_vol_corr", "taker_vol_autocorr_buy", "taker_vol_autocorr_sell",
               "taker_vol_skewness", "buy_sell_vol_ratio"]
    for col in vp_cols:
        if col in f.columns:
            for w in [5, 10]:
                f[f"{col}_ma_{w}"] = f[col].rolling(w, min_periods=1).mean()

    # ══════ I) Taker Flow ══════
    logger.info("    [I] Taker flow...")
    from indicators.base.taker_flow import (
        group_taker_orders, avg_levels_swept, large_taker_ratio,
        taker_imbalance, sweep_depth_autocorr, impact_efficiency,
        taker_arrival_rate, taker_size_skewness,
    )

    taker_orders = group_taker_orders(trades)
    if not taker_orders.empty and len(taker_orders) >= 10:
        als = avg_levels_swept(taker_orders, resample_freq=freq, window=20)
        f["avg_levels_swept"] = als.reindex(f.index).fillna(0)

        ltr = large_taker_ratio(taker_orders, resample_freq=freq, window=20, levels_threshold=3)
        f["large_taker_ratio"] = ltr.reindex(f.index).fillna(0)

        tim = taker_imbalance(taker_orders, resample_freq=freq, window=20)
        f["taker_imbalance_lib"] = tim.reindex(f.index).fillna(0)

        sda = sweep_depth_autocorr(taker_orders, resample_freq=freq, window=20)
        f["sweep_autocorr"] = sda.reindex(f.index).fillna(0)

        ie = impact_efficiency(taker_orders, resample_freq=freq, window=20)
        f["impact_efficiency"] = ie.reindex(f.index).fillna(0)

        tar = taker_arrival_rate(taker_orders, resample_freq=freq, window=20)
        f["taker_arrival_rate"] = tar.reindex(f.index).fillna(0)

        tss = taker_size_skewness(taker_orders, resample_freq=freq, window=30)
        f["taker_size_skewness_lib"] = tss.reindex(f.index).fillna(0)

        taker_cols = [
            "avg_levels_swept", "large_taker_ratio", "taker_imbalance_lib",
            "sweep_autocorr", "impact_efficiency", "taker_arrival_rate",
            "taker_size_skewness_lib",
        ]
        for col in taker_cols:
            if col in f.columns:
                for w in [5, 10]:
                    f[f"{col}_ma_{w}"] = f[col].rolling(w, min_periods=1).mean()

    # ══════ J) Regime ══════
    logger.info("    [J] Regime features...")
    from indicators.regime import (
        realized_volatility, parkinson_volatility, garch_like_vol,
        adx_indicator, efficiency_ratio,
        amihud_illiquidity, turnover_ratio,
    )

    f["realized_vol"] = realized_volatility(close, window=60)
    f["parkinson_vol"] = parkinson_volatility(ohlcv, window=60)
    f["garch_vol"] = garch_like_vol(close)

    adx_df = adx_indicator(ohlcv, period=14)
    for col in adx_df.columns:
        f[f"regime_{col}"] = adx_df[col].reindex(f.index)

    f["regime_efficiency"] = efficiency_ratio(close, window=20)
    f["regime_amihud"] = amihud_illiquidity(ohlcv, window=20)
    f["regime_turnover"] = turnover_ratio(ohlcv, window=20)

    # ── Final cleanup ──
    f = f.replace([np.inf, -np.inf], np.nan)
    # Forward-fill for indicators that produce staircase patterns
    f = f.ffill().fillna(0)

    logger.info(f"    → {len(f):,} bars × {f.shape[1]} features")
    return f


# ═══════════════════════════════════════════════════════════
# 2. Data Loading
# ═══════════════════════════════════════════════════════════

def load_features_v5(symbol: str, dates: List[str]) -> pd.DataFrame:
    """加载 V5 全特征 (indicators 库)。"""
    all_dfs = []
    for d_str in dates:
        dt = date.fromisoformat(d_str)
        cache_path = DataPaths.features(symbol, "1s_features_v5", dt)
        if cache_path.exists():
            logger.info(f"  [Cache] {d_str}")
            all_dfs.append(pd.read_parquet(cache_path))
        else:
            logger.info(f"  [Compute] {d_str}...")
            trades = load_agg_trades(symbol, dt, dt + timedelta(days=1))
            if trades.empty:
                logger.warning(f"  [Skip] {d_str} — no data")
                continue
            logger.info(f"    {len(trades):,} aggTrades")
            df = compute_all_features_v5(trades, "1s")
            if df.empty:
                del trades; gc.collect(); continue
            df.to_parquet(cache_path)
            all_dfs.append(df)
            del trades; gc.collect()
    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs).sort_index()


# ═══════════════════════════════════════════════════════════
# 3. Target Computation
# ═══════════════════════════════════════════════════════════

def make_target_return(df: pd.DataFrame, interval: int) -> pd.Series:
    """连续收益率 (bps)。"""
    cum = df["return_1"].cumsum()
    return cum.shift(-interval) - cum


def make_target_tbm_at_positions(
    close: np.ndarray, ret_std: np.ndarray, interval: int,
    positions: np.ndarray, tp_mult: float, sl_mult: float,
) -> np.ndarray:
    """仅在指定位置计算 TBM 标签。"""
    n_close = len(close)
    labels = np.full(len(positions), np.nan)
    for idx, i in enumerate(positions):
        if i + interval >= n_close:
            continue
        if np.isnan(ret_std[i]) or ret_std[i] < 1e-10:
            labels[idx] = 0; continue
        vol = ret_std[i] * np.sqrt(interval)
        tp_bps = vol * tp_mult
        sl_bps = vol * sl_mult
        p0 = close[i]
        if p0 < 1e-10:
            labels[idx] = 0; continue
        fwd = close[i+1:i+interval+1]
        rets = (fwd - p0) / p0 * 10000
        up_h = np.where(rets >= tp_bps)[0]
        dn_h = np.where(rets <= -sl_bps)[0]
        fu = up_h[0] if len(up_h) else interval + 1
        fd = dn_h[0] if len(dn_h) else interval + 1
        if fu < fd:
            labels[idx] = 1
        elif fd < fu:
            labels[idx] = -1
        else:
            labels[idx] = 0
    return labels


# ═══════════════════════════════════════════════════════════
# 4. Feature Selection
# ═══════════════════════════════════════════════════════════

def select_features(X, y, names, top_n=50, task="reg"):
    from sklearn.preprocessing import StandardScaler
    from sklearn.inspection import permutation_importance

    sc = StandardScaler()
    Xn = sc.fit_transform(X)
    val_n = max(200, len(X) // 5)

    if task == "reg":
        from sklearn.linear_model import Ridge
        m = Ridge(alpha=1.0).fit(Xn, y)
        perm = permutation_importance(m, Xn[-val_n:], y[-val_n:],
                                      n_repeats=5, random_state=42, scoring="r2")
    else:
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        m.fit(Xn, y)
        scoring = "balanced_accuracy" if len(np.unique(y)) > 2 else "accuracy"
        perm = permutation_importance(m, Xn[-val_n:], y[-val_n:],
                                      n_repeats=5, random_state=42, scoring=scoring)

    imp = perm.importances_mean
    top_idx = np.argsort(imp)[::-1][:top_n]
    selected = [names[i] for i in top_idx]

    logger.info(f"  Top features ({task}):")
    for rank, i in enumerate(top_idx[:15]):
        logger.info(f"    {rank+1:>3}. {names[i]:<35} imp={imp[i]:.6f}")
    return selected, {names[i]: float(imp[i]) for i in top_idx[:20]}


# ═══════════════════════════════════════════════════════════
# 5. Models
# ═══════════════════════════════════════════════════════════

def train_regression(X_tr, y_tr, X_te, model_type="ridge"):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_tr)
    Xte = sc.transform(X_te)

    if model_type == "ridge":
        from sklearn.linear_model import Ridge
        m = Ridge(alpha=10.0).fit(Xtr, y_tr)
    elif model_type == "lasso":
        from sklearn.linear_model import LassoCV
        m = LassoCV(n_alphas=50, max_iter=5000, cv=3, random_state=42).fit(Xtr, y_tr)
        n_active = np.sum(m.coef_ != 0)
        logger.info(f"    Lasso: alpha={m.alpha_:.6f}, active={n_active}/{Xtr.shape[1]}")
    else:  # gbm
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(
            max_iter=300, max_depth=3, learning_rate=0.03,
            min_samples_leaf=20, l2_regularization=5.0,
            early_stopping=True, validation_fraction=0.2,
            n_iter_no_change=20, random_state=42,
        ).fit(Xtr, y_tr)

    pred_tr = m.predict(Xtr)
    ss_res = np.sum((y_tr - pred_tr) ** 2)
    ss_tot = np.sum((y_tr - y_tr.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    logger.info(f"  [{model_type.upper()}] Train R²={r2:.4f}")
    return m.predict(Xte)


def train_classifier(X_tr, y_tr, X_te, model_type="logistic"):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_tr)
    Xte = sc.transform(X_te)

    if model_type == "logistic":
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(
            C=0.1, max_iter=2000, random_state=42, multi_class="multinomial",
        ).fit(Xtr, y_tr)
    else:  # gbm
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(
            max_iter=300, max_depth=3, learning_rate=0.03,
            min_samples_leaf=20, l2_regularization=5.0,
            early_stopping=True, validation_fraction=0.2,
            n_iter_no_change=20, random_state=42,
        ).fit(Xtr, y_tr)

    acc = m.score(Xtr, y_tr)
    logger.info(f"  [{model_type.upper()}] Train acc={acc:.4f}")

    classes = m.classes_
    proba = m.predict_proba(Xte)
    if len(classes) == 2:
        up_idx = np.where(classes == 1)[0][0]
        signal = proba[:, up_idx] - 0.5
    else:
        p_up = proba[:, np.where(classes == 1)[0][0]] if 1 in classes else np.zeros(len(Xte))
        p_dn = proba[:, np.where(classes == -1)[0][0]] if -1 in classes else np.zeros(len(Xte))
        signal = p_up - p_dn
    return signal


# ═══════════════════════════════════════════════════════════
# 6. Backtest
# ═══════════════════════════════════════════════════════════

def backtest_fixed_interval(signal, actual_return, timestamps, cost_bps=4.0, threshold=0.0):
    n = len(signal)
    rt_cost = cost_bps * 2
    trades, cum_pnl, prev_pos = [], 0.0, 0
    for i in range(n):
        s = signal[i]
        pos = 0 if np.isnan(s) else (1 if s > threshold else (-1 if s < -threshold else 0))
        gross = actual_return[i] * pos if not np.isnan(actual_return[i]) else 0.0
        cost = rt_cost if pos != prev_pos and (pos != 0 or prev_pos != 0) else 0.0
        net = gross - cost
        cum_pnl += net
        trades.append({"ts": timestamps[i] if i < len(timestamps) else None,
                        "position": pos, "signal": float(s) if not np.isnan(s) else 0,
                        "actual_return": float(actual_return[i]) if not np.isnan(actual_return[i]) else 0,
                        "gross_pnl": gross, "cost": cost, "net_pnl": net, "cum_pnl": cum_pnl})
        prev_pos = pos
    return pd.DataFrame(trades)


def compute_metrics(df):
    if df.empty or len(df) < 2:
        return {"n_trades": 0, "total_pnl": 0, "sharpe": 0, "win_rate": 0, "profit_factor": 0}
    active = df[df["position"] != 0]
    n_trades = int((df["position"].diff().fillna(0) != 0).sum())
    pnl = df["net_pnl"]
    sr = pnl.mean() / (pnl.std() + 1e-10) * np.sqrt(len(pnl))
    wr = len(active[active["net_pnl"] > 0]) / len(active) if len(active) > 0 else 0
    pos_pnl = pnl[pnl > 0].sum()
    neg_pnl = pnl[pnl < 0].abs().sum() + 1e-10
    dd = (df["cum_pnl"] - df["cum_pnl"].cummax()).min()
    return {"n_trades": n_trades, "n_active": len(active), "total_pnl": float(pnl.sum()),
            "gross_pnl": float(df["gross_pnl"].sum()), "sharpe": float(sr),
            "win_rate": float(wr), "profit_factor": float(pos_pnl / neg_pnl), "max_dd": float(dd)}


# ═══════════════════════════════════════════════════════════
# 7. Single Experiment
# ═══════════════════════════════════════════════════════════

def run_experiment(
    train_1s, test_1s, interval, target_type,
    cost_bps=4.0, top_features=50, tp_mult=1.0, sl_mult=0.5,
):
    interval_str = {1: "1s", 60: "1min", 300: "5min"}.get(interval, f"{interval}s")
    tbm_label = f"tp{tp_mult}_sl{sl_mult}" if target_type == "tbm" else ""
    logger.info(f"\n{'─'*70}")
    logger.info(f"  Experiment: interval={interval_str}, target={target_type} {tbm_label}")
    logger.info(f"{'─'*70}")

    # ── Subsample positions ──
    train_positions = np.arange(0, len(train_1s), interval)
    test_positions = np.arange(0, len(test_1s), interval)

    # ── Targets ──
    if target_type == "return":
        task = "reg"
        cum_train = train_1s["return_1"].cumsum().values
        cum_test = test_1s["return_1"].cumsum().values
        y_train = np.array([cum_train[min(i+interval, len(cum_train)-1)] - cum_train[i]
                            for i in train_positions])
        y_test_actual = np.array([cum_test[min(i+interval, len(cum_test)-1)] - cum_test[i]
                                  for i in test_positions])
        y_train[train_positions + interval >= len(cum_train)] = np.nan
        y_test_actual[test_positions + interval >= len(cum_test)] = np.nan

    elif target_type == "tbm":
        task = "cls"
        close_train = train_1s["close"].values
        close_test = test_1s["close"].values
        ret_std_train = train_1s["return_1"].rolling(max(20, interval), min_periods=10).std().values
        ret_std_test = test_1s["return_1"].rolling(max(20, interval), min_periods=10).std().values

        y_train = make_target_tbm_at_positions(
            close_train, ret_std_train, interval, train_positions, tp_mult, sl_mult)
        y_test_labels = make_target_tbm_at_positions(
            close_test, ret_std_test, interval, test_positions, tp_mult, sl_mult)

        # Actual return for PnL
        cum_test = test_1s["return_1"].cumsum().values
        y_test_actual = np.array([
            cum_test[min(i+interval, len(cum_test)-1)] - cum_test[i]
            for i in test_positions])
        y_test_actual[test_positions + interval >= len(cum_test)] = np.nan
    else:
        raise ValueError(f"Unknown target: {target_type}")

    # ── Filter NaN ──
    train_mask = ~np.isnan(y_train)
    test_mask = ~np.isnan(y_test_actual)
    if target_type == "tbm":
        test_mask &= ~np.isnan(y_test_labels)

    y_tr = y_train[train_mask]
    y_te_actual = y_test_actual[test_mask]
    if target_type == "tbm":
        y_tr = y_tr.astype(int)

    logger.info(f"  Train: {len(y_tr):,}, Test: {test_mask.sum():,}")
    if task == "cls":
        unique, counts = np.unique(y_tr, return_counts=True)
        dist = {int(u): int(c) for u, c in zip(unique, counts)}
        total = sum(dist.values())
        trend_pct = (dist.get(1, 0) + dist.get(-1, 0)) / total * 100 if total > 0 else 0
        logger.info(f"  Label distribution: {dist} → trend={trend_pct:.1f}%")

    if len(y_tr) < 100:
        logger.warning("  [Skip] Too few samples")
        return None

    # ── Feature columns ──
    exclude = {"return_1", "log_return", "close"}
    feat_cols = [
        c for c in train_1s.columns
        if c not in exclude
        and train_1s[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]
        and train_1s[c].std() > 1e-10
    ]
    logger.info(f"  Features: {len(feat_cols)}")

    X_train_all = train_1s.iloc[train_positions[train_mask]][feat_cols].values
    X_test_all = test_1s.iloc[test_positions[test_mask]][feat_cols].values
    test_ts = test_1s.index[test_positions[test_mask]]

    # ── Feature selection ──
    selected, importance = select_features(
        X_train_all, y_tr.astype(float), feat_cols, top_n=top_features, task=task)

    X_train = train_1s.iloc[train_positions[train_mask]][selected].values
    X_test = test_1s.iloc[test_positions[test_mask]][selected].values

    # ── Models ──
    results = {}
    if task == "reg":
        model_types = ["ridge", "lasso", "gbm"]
    else:
        model_types = ["logistic", "gbm"]

    for mtype in model_types:
        logger.info(f"\n  Training {mtype.upper()} ({task})...")
        t0 = time.time()

        if task == "reg":
            # For lasso, use all features (L1 does its own selection)
            if mtype == "lasso":
                signal = train_regression(X_train_all, y_tr, X_test_all, model_type=mtype)
            else:
                signal = train_regression(X_train, y_tr, X_test, model_type=mtype)
        else:
            signal = train_classifier(X_train, y_tr, X_test, model_type=mtype)

        logger.info(f"    Time: {time.time()-t0:.1f}s | Signal: μ={signal.mean():.4f} σ={signal.std():.4f}")

        # IC for regression
        if task == "reg":
            mask = ~(np.isnan(signal) | np.isnan(y_te_actual))
            if mask.sum() > 10:
                ic = np.corrcoef(signal[mask], y_te_actual[mask])[0, 1]
                logger.info(f"    IC={ic:.4f}")

        # Backtest sweeps
        thresholds = [0, 0.5, 1.0, 2.0] if task == "reg" else [0, 0.05, 0.1, 0.15, 0.2, 0.3]
        for cost, clbl in [(cost_bps, "taker"), (1.0, "maker")]:
            for thr in thresholds:
                bt = backtest_fixed_interval(signal, y_te_actual, test_ts, cost_bps=cost, threshold=thr)
                met = compute_metrics(bt)
                key = f"{mtype}_{clbl}_thr{thr}"
                results[key] = met
                logger.info(
                    f"    {key:<35} N={met['n_trades']:>4} Act={met.get('n_active',0):>5} "
                    f"Gross={met['gross_pnl']:>+7.0f} Net={met['total_pnl']:>+7.0f} "
                    f"WR={met['win_rate']:.1%} SR={met['sharpe']:>+6.2f} PF={met['profit_factor']:.2f}"
                )

    best_key = max(results, key=lambda k: results[k]["sharpe"]) if results else "N/A"
    best_met = results.get(best_key, {})
    logger.info(f"\n  ★ Best: {best_key} (Sharpe={best_met.get('sharpe',0):.2f})")

    return {
        "interval": interval_str, "target": target_type,
        "tbm_params": {"tp_mult": tp_mult, "sl_mult": sl_mult} if target_type == "tbm" else None,
        "train_samples": len(y_tr), "test_samples": int(test_mask.sum()),
        "n_features_total": len(feat_cols), "n_features_selected": len(selected),
        "top_features": importance,
        "best_config": best_key, "best_sharpe": best_met.get("sharpe", -999),
        "results": results,
        "label_dist": dist if task == "cls" else None,
    }


# ═══════════════════════════════════════════════════════════
# 8. Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SOL/USDT")
    parser.add_argument("--interval", type=int, default=0, help="0=all: 1,60,300")
    parser.add_argument("--target", default="all", choices=["return", "tbm", "all"])
    parser.add_argument("--cost-bps", type=float, default=4.0)
    parser.add_argument("--top-features", type=int, default=50)
    parser.add_argument("--quick", action="store_true", help="Skip 1s interval")
    args = parser.parse_args()

    intervals = [60, 300] if args.quick else ([1, 60, 300] if args.interval == 0 else [args.interval])
    targets = ["return", "tbm"] if args.target == "all" else [args.target]

    logger.info("=" * 75)
    logger.info("SOL V5 — Full Indicators Library Experiment")
    logger.info(f"  Intervals: {intervals}s")
    logger.info(f"  Targets: {targets}")
    logger.info(f"  TBM configs: {TBM_CONFIGS}")
    logger.info(f"  Feature source: indicators/ library (ALL base + regime)")
    logger.info(f"  Feature windows: {FEATURE_WINDOWS}s")
    logger.info(f"  Models: Ridge/Lasso/GBM (reg), Logistic/GBM (cls)")
    logger.info(f"  Train: {len(TRAIN_DATES)} days, Test: {len(TEST_DATES)} days")
    logger.info("=" * 75)

    # ── Load features ──
    logger.info("\n[1] Loading V5 features (full indicators library)...")
    train_1s = load_features_v5(args.symbol, TRAIN_DATES)
    test_1s = load_features_v5(args.symbol, TEST_DATES)
    if train_1s.empty or test_1s.empty:
        logger.error("Failed to load features!"); return

    logger.info(f"  Train: {len(train_1s):,} bars × {train_1s.shape[1]} features")
    logger.info(f"  Test:  {len(test_1s):,} bars × {test_1s.shape[1]} features")

    # Feature inventory
    exclude = {"return_1", "log_return", "close"}
    all_feat = [c for c in train_1s.columns if c not in exclude
                and train_1s[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]
                and train_1s[c].std() > 1e-10]
    logger.info(f"\n[2] Feature inventory: {len(all_feat)} effective features")

    # ── Run experiments ──
    logger.info(f"\n[3] Running experiments...")
    all_experiments = []

    for interval in intervals:
        for target in targets:
            if target == "return":
                exp = run_experiment(
                    train_1s, test_1s, interval, "return",
                    cost_bps=args.cost_bps, top_features=args.top_features)
                if exp: all_experiments.append(exp)
            elif target == "tbm":
                for tp, sl in TBM_CONFIGS:
                    exp = run_experiment(
                        train_1s, test_1s, interval, "tbm",
                        cost_bps=args.cost_bps, top_features=args.top_features,
                        tp_mult=tp, sl_mult=sl)
                    if exp: all_experiments.append(exp)

    # ═══ Grand Summary ═══
    logger.info(f"\n{'='*95}")
    logger.info("GRAND SUMMARY — V5 Full Indicators Experiments")
    logger.info(f"{'='*95}")
    logger.info(
        f"{'Interval':<8} {'Target':<12} {'TBM':>10} "
        f"{'Best Config':<40} {'SR':>6} {'Net':>8} {'N':>5} {'WR':>5}"
    )
    logger.info(f"{'-'*95}")

    summary_rows = []
    for exp in all_experiments:
        best = exp["results"].get(exp["best_config"], {})
        tbm_str = f"tp{exp['tbm_params']['tp_mult']}_sl{exp['tbm_params']['sl_mult']}" if exp.get("tbm_params") else "-"
        row = {
            "interval": exp["interval"], "target": exp["target"],
            "tbm_params": tbm_str,
            "best_config": exp["best_config"],
            "sharpe": best.get("sharpe", -999),
            "net_pnl": best.get("total_pnl", 0),
            "n_trades": best.get("n_trades", 0),
            "win_rate": best.get("win_rate", 0),
            "n_features": exp.get("n_features_total", 0),
        }
        summary_rows.append(row)
        logger.info(
            f"{row['interval']:<8} {row['target']:<12} {row['tbm_params']:>10} "
            f"{row['best_config']:<40} {row['sharpe']:>+6.2f} {row['net_pnl']:>+8.0f} "
            f"{row['n_trades']:>5} {row['win_rate']:>4.1%}"
        )

    # Overall best
    if all_experiments:
        best_exp = max(all_experiments, key=lambda e: e["best_sharpe"])
        logger.info(f"\n★ OVERALL BEST: {best_exp['interval']} {best_exp['target']} → "
                     f"{best_exp['best_config']} (Sharpe={best_exp['best_sharpe']:.2f})")
        if best_exp.get("top_features"):
            logger.info("  Top features:")
            for feat, imp in list(best_exp["top_features"].items())[:10]:
                logger.info(f"    {feat:<35} imp={imp:.6f}")

    # Compare baselines
    logger.info(f"\n{'─'*95}")
    logger.info("Baselines:")
    logger.info("  V3 1min TBM:  logistic_maker_thr0.2 → SR=+1.52, Net=+358, N=26")
    logger.info("  V4 1min TBM:  logistic_maker_thr0.3 → SR=+1.88, Net=+452, N=14 (tp=1.0,sl=1.0)")
    logger.info("  V4 1min TBM:  logistic_maker_thr0.05→ SR=+1.59, Net=+2606, N=10 (tp=1.0,sl=0.5)")
    logger.info("  V3 5min Ret:  gbm_maker_thr0        → SR=+1.22, Net=+2000")
    logger.info(f"{'─'*95}")

    # Save
    save_data = {
        "config": {
            "version": "v5_full_indicators",
            "symbol": args.symbol,
            "intervals": intervals,
            "targets": targets,
            "tbm_configs": [{"tp": t, "sl": s} for t, s in TBM_CONFIGS],
            "feature_windows": FEATURE_WINDOWS,
            "cost_bps": args.cost_bps,
            "train_dates": TRAIN_DATES,
            "test_dates": TEST_DATES,
        },
        "experiments": [{
            "interval": e["interval"], "target": e["target"],
            "tbm_params": e.get("tbm_params"),
            "train_samples": e.get("train_samples"),
            "test_samples": e.get("test_samples"),
            "n_features_total": e.get("n_features_total"),
            "n_features_selected": e.get("n_features_selected"),
            "top_features": e.get("top_features"),
            "best_config": e["best_config"],
            "best_sharpe": e["best_sharpe"],
            "all_results": e.get("results"),
            "label_dist": e.get("label_dist"),
        } for e in all_experiments],
        "summary": summary_rows,
    }
    sp = DataPaths.backtest_dir("sol_v5") / "summary.json"
    with open(sp, "w") as fp:
        json.dump(save_data, fp, indent=2, default=str)
    logger.info(f"\nSaved: {sp}")


if __name__ == "__main__":
    main()
