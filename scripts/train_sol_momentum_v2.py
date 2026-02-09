#!/usr/bin/env python3
"""
SOL Momentum / Reversal Prediction  —  V2

V2 核心改进（vs V1）:
  1. 采样粒度 = 1s bar（~86K/天），而非 5min bar（~288/天）
     → 总训练样本 ~600K vs V1 的 ~2K
  2. 特征窗口保持物理意义 [3, 5, 10, 20]s
     → 不随预测区间缩放（特征描述微观结构，其时间尺度是固定的）
  3. 仅改变预测目标 horizon:
       --horizon 300  →  next 5min return (动量)
       --horizon 60   →  next 1min return (均值回复)
  4. LSTM 终于有足够训练数据 (600K+ samples)
  5. Lasso 在更大样本上做有效 L1 特征选择
  6. 训练采用 stride 防止重叠 target 过拟合

Models:
  - Ridge:  L2, 预选 top-N 特征
  - GBM:    HistGradientBoosting, 预选 top-N 特征
  - Lasso:  LassoCV, 全部特征, L1 自动稀疏
  - LSTM:   逐 bar 原始特征 (不做 rolling), lookback=30s

Usage:
    python scripts/train_sol_momentum_v2.py --horizon 300          # 5min 动量
    python scripts/train_sol_momentum_v2.py --horizon 60           # 1min 反转
    python scripts/train_sol_momentum_v2.py --horizon 300 --model lstm
    python scripts/train_sol_momentum_v2.py --horizon 60 --model lasso
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
SOL_TRAIN_DATES = [
    "2026-01-28", "2026-01-29", "2026-01-30", "2026-01-31",
    "2026-02-01", "2026-02-02", "2026-02-03",
]
SOL_TEST_DATES = [
    "2026-02-04", "2026-02-05", "2026-02-06",
]

# 特征窗口：基于微观结构物理特性，不随预测区间改变
FEATURE_WINDOWS = [3, 5, 10, 20]  # 秒

# LSTM 设置
LSTM_LOOKBACK = 30  # 30 bars = 30s（微观结构记忆尺度）
LSTM_RAW_COLS = [
    "return_1", "volume", "buy_volume", "sell_volume",
    "volume_imbalance", "n_trades",
    "bar_range_bps", "body_ratio", "bar_direction",
    "taker_levels_swept", "taker_volume", "taker_impact_pct",
    "taker_count", "taker_imbalance", "taker_large_ratio",
]
LSTM_DIFF_COLS = [
    "return_1", "volume", "buy_volume", "sell_volume",
    "n_trades", "taker_count", "taker_imbalance",
]


# ─────────────────────────────────────────────────────────
# 1. Feature Computation (1s bars, short windows)
# ─────────────────────────────────────────────────────────

def compute_1s_features(trades: pd.DataFrame) -> pd.DataFrame:
    """
    aggTrades → 1s OHLCV → 特征。
    窗口 = [3, 5, 10, 20] 秒，保持微观结构物理意义。
    """
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

    # ── A. Return ──
    f["return_1"] = c.pct_change() * 10000  # bps
    f["log_return"] = np.log(c / c.shift(1))

    for w in FEATURE_WINDOWS:
        f[f"return_sum_{w}"] = f["return_1"].rolling(w, min_periods=1).sum()
        f[f"return_std_{w}"] = f["return_1"].rolling(w, min_periods=2).std()
        f[f"return_skew_{w}"] = f["return_1"].rolling(w, min_periods=3).skew()

    f["return_sign"] = np.sign(f["return_1"])
    sign_change = (f["return_sign"] != f["return_sign"].shift(1))
    run_group = sign_change.cumsum()
    f["return_run_len"] = run_group.groupby(run_group).cumcount() + 1
    f.drop(columns=["return_sign"], inplace=True)

    # ── B. Volume ──
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

    # ── C. Micro-structure ──
    hl = h - lo
    f["bar_range_bps"] = hl / (c + 1e-10) * 10000
    f["body_ratio"] = (c - o).abs() / (hl + 1e-10)
    f["upper_wick"] = (h - c.clip(lower=o).clip(upper=h)) / (hl + 1e-10)
    f["lower_wick"] = (c.clip(upper=o).clip(lower=lo) - lo) / (hl + 1e-10)
    f["bar_direction"] = np.sign(c - o)

    for w in FEATURE_WINDOWS:
        f[f"bar_range_ma_{w}"] = f["bar_range_bps"].rolling(w, min_periods=1).mean()
        f[f"trades_ma_{w}"] = nt.rolling(w, min_periods=1).mean()
        f[f"trades_ratio_{w}"] = nt / (f[f"trades_ma_{w}"] + 1e-10)

    # ── D. Spread & Impact proxy ──
    price_diff = c.diff().abs()
    for w in FEATURE_WINDOWS:
        f[f"spread_proxy_{w}"] = price_diff.rolling(w, min_periods=1).mean()
        f[f"spread_proxy_std_{w}"] = price_diff.rolling(w, min_periods=2).std()
        cum_move = price_diff.rolling(w, min_periods=1).sum()
        cum_vol = v.rolling(w, min_periods=1).sum() + 1e-10
        f[f"impact_proxy_{w}"] = cum_move / cum_vol

    # ── E. Volume autocorrelation ──
    for w in [5, 10, 20]:
        f[f"buy_vol_autocorr_{w}"] = bv.rolling(w, min_periods=3).corr(bv.shift(1))
        f[f"sell_vol_autocorr_{w}"] = sv.rolling(w, min_periods=3).corr(sv.shift(1))
        f[f"bv_sv_corr_{w}"] = bv.rolling(w, min_periods=3).corr(sv)

    f = f.replace([np.inf, -np.inf], np.nan).fillna(0)
    return f


def add_taker_features(f: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """taker order 聚合特征 → 1s bars。"""
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

    large_mask = (taker_ts["levels_swept"] >= 3).astype(float)
    large_vol = (taker_ts["total_volume"] * large_mask).resample("1s").sum()
    total_t = taker_ts["total_volume"].resample("1s").sum() + 1e-10

    f["taker_levels_swept"] = t_levels.reindex(f.index).fillna(0)
    f["taker_volume"] = t_volume.reindex(f.index).fillna(0)
    f["taker_impact_pct"] = t_impact.reindex(f.index).fillna(0)
    f["taker_count"] = t_count.reindex(f.index).fillna(0)
    f["taker_imbalance"] = ((buy_t - sell_t) / (buy_t + sell_t + 1e-10)).reindex(f.index).fillna(0)
    f["taker_large_ratio"] = (large_vol / total_t).reindex(f.index).fillna(0)

    for w in [3, 5, 10]:
        f[f"taker_levels_ma_{w}"] = f["taker_levels_swept"].rolling(w, min_periods=1).mean()
        f[f"taker_imb_ma_{w}"] = f["taker_imbalance"].rolling(w, min_periods=1).mean()
        f[f"taker_count_ma_{w}"] = f["taker_count"].rolling(w, min_periods=1).mean()

    f = f.replace([np.inf, -np.inf], np.nan).fillna(0)
    return f


# ─────────────────────────────────────────────────────────
# 2. Data Loading
# ─────────────────────────────────────────────────────────

def load_1s_features(symbol: str, dates: List[str]) -> pd.DataFrame:
    """逐天加载 → 1s 特征 → concat。缓存为 parquet。"""
    all_dfs = []
    for d_str in dates:
        dt = date.fromisoformat(d_str)
        cache_path = DataPaths.features(symbol, "1s_features_v2", dt)

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
                del trades; gc.collect()
                continue
            df = add_taker_features(df, trades)
            df.to_parquet(cache_path)
            logger.info(f"    → {len(df):,} bars × {df.shape[1]} features")
            all_dfs.append(df)
            del trades; gc.collect()

    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs).sort_index()


# ─────────────────────────────────────────────────────────
# 3. Target Computation
# ─────────────────────────────────────────────────────────

def compute_forward_return(df: pd.DataFrame, horizon: int) -> pd.Series:
    """
    计算 horizon 秒后的累计收益（bps）。
    使用 log return cumsum 近似，误差极小。
    """
    cum = df["return_1"].cumsum()
    target = cum.shift(-horizon) - cum
    return target


# ─────────────────────────────────────────────────────────
# 4. Feature Selection
# ─────────────────────────────────────────────────────────

def select_features(X, y, names, top_n=40, stride=1):
    from sklearn.linear_model import Ridge
    from sklearn.inspection import permutation_importance
    from sklearn.preprocessing import StandardScaler

    # Subsample for speed
    Xs = X[::stride]
    ys = y[::stride]
    logger.info(f"  Feature selection on {Xs.shape} (stride={stride})...")

    sc = StandardScaler()
    Xn = sc.fit_transform(Xs)
    m = Ridge(alpha=1.0).fit(Xn, ys)

    val_n = max(500, len(Xs) // 5)
    perm = permutation_importance(m, Xn[-val_n:], ys[-val_n:],
                                   n_repeats=5, random_state=42, scoring="r2")
    imp = perm.importances_mean
    top_idx = np.argsort(imp)[::-1][:top_n]
    selected = [names[i] for i in top_idx]

    logger.info(f"  Top {min(top_n, 15)} features:")
    for rank, i in enumerate(top_idx[:15]):
        logger.info(f"    {rank+1:>3}. {names[i]:<35} imp={imp[i]:.6f}")
    if top_n > 15:
        logger.info(f"    ... ({top_n - 15} more)")
    return selected


# ─────────────────────────────────────────────────────────
# 5. Models
# ─────────────────────────────────────────────────────────

def train_ridge(X_tr, y_tr, X_te, stride=1):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    Xs, ys = X_tr[::stride], y_tr[::stride]
    sc = StandardScaler()
    Xn = sc.fit_transform(Xs)
    m = Ridge(alpha=10.0).fit(Xn, ys)

    pred = m.predict(sc.transform(X_te))
    r2_tr = 1 - np.sum((ys - m.predict(Xn))**2) / np.sum((ys - ys.mean())**2)
    logger.info(f"  [Ridge] Train R²={r2_tr:.4f} (on {len(Xs):,} samples, stride={stride})")
    return pred, m, sc


def train_gbm(X_tr, y_tr, X_te, stride=1):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler

    Xs, ys = X_tr[::stride], y_tr[::stride]
    sc = StandardScaler()
    Xn = sc.fit_transform(Xs)

    m = HistGradientBoostingRegressor(
        max_iter=500, max_depth=4, learning_rate=0.02,
        min_samples_leaf=50, l2_regularization=3.0,
        early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=30, random_state=42,
    )
    m.fit(Xn, ys)

    pred = m.predict(sc.transform(X_te))
    tr_pred = m.predict(Xn)
    r2 = 1 - np.sum((ys - tr_pred)**2) / np.sum((ys - ys.mean())**2)
    logger.info(f"  [GBM] Train R²={r2:.4f}, n_iter={m.n_iter_} (stride={stride})")
    return pred, m, sc


def train_lasso(X_tr, y_tr, X_te, stride=1):
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler

    Xs, ys = X_tr[::stride], y_tr[::stride]
    sc = StandardScaler()
    Xn = sc.fit_transform(Xs)

    m = LassoCV(
        cv=5, n_alphas=50, max_iter=10000,
        random_state=42, n_jobs=-1,
    )
    m.fit(Xn, ys)

    coefs = m.coef_
    n_active = int(np.sum(np.abs(coefs) > 1e-8))
    active_idx = np.where(np.abs(coefs) > 1e-8)[0]
    logger.info(f"  [Lasso] alpha={m.alpha_:.6f}, active={n_active}/{Xn.shape[1]}")

    if n_active > 0:
        sorted_active = active_idx[np.argsort(np.abs(coefs[active_idx]))[::-1]]
        logger.info(f"  Active features (top 20):")
        for rank, i in enumerate(sorted_active[:20]):
            logger.info(f"    {rank+1:>3}. feat_{i:<4} coef={coefs[i]:+.6f}")

    pred = m.predict(sc.transform(X_te))
    tr_pred = m.predict(Xn)
    r2 = 1 - np.sum((ys - tr_pred)**2) / np.sum((ys - ys.mean())**2)
    logger.info(f"  [Lasso] Train R²={r2:.4f} (stride={stride})")
    return pred, m, sc


def train_lstm(train_df, test_df, horizon, feat_cols, lookback=LSTM_LOOKBACK, stride=5):
    """
    LSTM 回归：逐 bar 原始特征，不做 rolling。
    lookback = 30 bars = 30s（不随 horizon 改变）。
    stride 控制训练样本密度。
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    # ── 准备 raw features (不做 rolling，让 LSTM 学时序) ──
    raw_cols = [c for c in LSTM_RAW_COLS if c in train_df.columns]
    diff_cols = [c for c in LSTM_DIFF_COLS if c in train_df.columns]

    def build_feat_matrix(df):
        feat = df[raw_cols].copy()
        for c in diff_cols:
            feat[f"{c}_d1"] = df[c].diff()
        feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0)
        return feat

    logger.info(f"  [LSTM] lookback={lookback}s, stride={stride}")
    train_feat = build_feat_matrix(train_df)
    test_feat = build_feat_matrix(test_df)
    feat_names = list(train_feat.columns)
    n_feat = len(feat_names)
    logger.info(f"    Raw features: {n_feat} ({feat_names[:8]}...)")

    # Target
    y_train_all = compute_forward_return(train_df, horizon).values
    y_test_all = compute_forward_return(test_df, horizon).values

    # ── Normalization (train stats) ──
    tr_vals = train_feat.values
    mu = tr_vals.mean(axis=0).astype(np.float32)
    sd = tr_vals.std(axis=0).astype(np.float32) + 1e-8
    tr_norm = ((tr_vals - mu) / sd).astype(np.float32)
    te_norm = ((test_feat.values - mu) / sd).astype(np.float32)

    y_mu = float(np.nanmean(y_train_all))
    y_sd = float(np.nanstd(y_train_all)) + 1e-8

    # ── Sliding windows with stride ──
    def make_windows(data, targets, lb, s):
        X, y, idx = [], [], []
        for i in range(lb, len(data) - 1, s):
            if np.isnan(targets[i]):
                continue
            X.append(data[i - lb:i])
            y.append((targets[i] - y_mu) / y_sd)
            idx.append(i)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(idx)

    X_tr, y_tr, idx_tr = make_windows(tr_norm, y_train_all, lookback, stride)
    # Test: stride=1 for full predictions
    X_te, y_te, idx_te = make_windows(te_norm, y_test_all, lookback, 1)
    logger.info(f"    Train windows: {X_tr.shape} (stride={stride})")
    logger.info(f"    Test  windows: {X_te.shape} (stride=1)")

    # Validation split
    val_n = max(500, len(X_tr) // 5)
    X_val, y_val = X_tr[-val_n:], y_tr[-val_n:]
    X_tr2, y_tr2 = X_tr[:-val_n], y_tr[:-val_n]

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    logger.info(f"    Device: {device}")

    # ── Model ──
    class LSTMReg(nn.Module):
        def __init__(self, inp, hid=64, layers=2, drop=0.2):
            super().__init__()
            self.lstm = nn.LSTM(inp, hid, layers, batch_first=True,
                                dropout=drop if layers > 1 else 0)
            self.drop = nn.Dropout(drop)
            self.fc = nn.Linear(hid, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(self.drop(out[:, -1, :])).squeeze(-1)

    model = LSTMReg(n_feat, hid=64, layers=2, drop=0.2).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"    Model: LSTM(64, 2 layers), {n_params:,} params")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    tr_ds = TensorDataset(torch.from_numpy(X_tr2), torch.from_numpy(y_tr2))
    tr_dl = DataLoader(tr_ds, batch_size=256, shuffle=True)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=1e-3, epochs=50, steps_per_epoch=len(tr_dl))

    val_xt = torch.from_numpy(X_val).to(device)
    val_yt = torch.from_numpy(y_val).to(device)

    # ── Training ──
    best_val = 1e10
    patience, wait = 12, 0
    best_state = None

    for ep in range(50):
        model.train()
        ep_loss = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); scheduler.step()
            ep_loss += loss.item() * len(xb)
        ep_loss /= len(X_tr2)

        model.eval()
        with torch.no_grad():
            vp = model(val_xt)
            vl = loss_fn(vp, val_yt).item()

        if ep % 5 == 0 or ep < 3:
            logger.info(f"    Epoch {ep:>3}: train_loss={ep_loss:.6f}, val_loss={vl:.6f}")

        if vl < best_val:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info(f"    Early stop at epoch {ep} (best val_loss={best_val:.6f})")
                break

    model.load_state_dict(best_state)
    model.eval()

    # ── Predict (in batches) ──
    preds = []
    batch_sz = 2048
    with torch.no_grad():
        for s in range(0, len(X_te), batch_sz):
            batch = torch.from_numpy(X_te[s:s+batch_sz]).to(device)
            p = model(batch).cpu().numpy()
            preds.append(p)
    pred_n = np.concatenate(preds)
    pred_bps = pred_n * y_sd + y_mu

    # Train R²
    tr_preds = []
    with torch.no_grad():
        for s in range(0, len(X_tr), batch_sz):
            batch = torch.from_numpy(X_tr[s:s+batch_sz]).to(device)
            p = model(batch).cpu().numpy()
            tr_preds.append(p)
    tr_p = np.concatenate(tr_preds) * y_sd + y_mu
    y_tr_raw = y_train_all[idx_tr]
    mask = ~np.isnan(y_tr_raw)
    r2 = 1 - np.sum((y_tr_raw[mask] - tr_p[mask])**2) / np.sum((y_tr_raw[mask] - np.nanmean(y_tr_raw))**2)
    logger.info(f"  [LSTM] Train R²={r2:.4f}, best_val_loss={best_val:.6f}")

    return pred_bps, idx_te, model, (mu, sd, y_mu, y_sd)


# ─────────────────────────────────────────────────────────
# 6. Position-Based Backtest
# ─────────────────────────────────────────────────────────

def backtest_position(
    pred, y_actual, timestamps,
    cost_bps=4.0, smooth_window=10,
    entry_quantile=0.7, max_hold_bars=300,
):
    """持仓模型回测。y_actual = 逐 bar (1s) 的 return_1。"""
    n = len(pred)
    rt_cost = cost_bps * 2

    sig = pd.Series(pred).ewm(span=smooth_window, adjust=False).mean().values

    warm = max(100, n // 20)
    abs_s = np.abs(sig[:warm])
    pos_abs = abs_s[abs_s > 0]
    thresh = np.quantile(pos_abs, entry_quantile) if len(pos_abs) > 10 else 1.0

    trades = []
    cum_pnl = 0.0
    pos = 0
    entry_bar = 0

    for i in range(warm, n):
        if i > warm + 500 and i % 500 == 0:
            lb = min(i, 5000)
            a = np.abs(sig[i - lb:i])
            a = a[a > 0]
            if len(a) > 10:
                thresh = np.quantile(a, entry_quantile)

        want = 0
        if sig[i] > thresh:
            want = 1
        elif sig[i] < -thresh:
            want = -1

        if pos != 0 and (i - entry_bar) >= max_hold_bars:
            want = 0
        if pos > 0 and sig[i] < 0:
            want = -1 if sig[i] < -thresh else 0
        elif pos < 0 and sig[i] > 0:
            want = 1 if sig[i] > thresh else 0

        if want != pos:
            if pos != 0:
                gross = float(np.sum(y_actual[entry_bar + 1:i + 1]) * pos)
                net = gross - rt_cost
                cum_pnl += net
                trades.append({
                    "entry_ts": timestamps[entry_bar],
                    "exit_ts": timestamps[i],
                    "direction": pos,
                    "bars_held": i - entry_bar,
                    "gross_pnl_bps": gross,
                    "net_pnl_bps": net,
                    "cum_pnl_bps": cum_pnl,
                })
            if want != 0:
                entry_bar = i
            pos = want

    if pos != 0:
        gross = float(np.sum(y_actual[entry_bar + 1:n]) * pos)
        net = gross - rt_cost
        cum_pnl += net
        trades.append({
            "entry_ts": timestamps[entry_bar],
            "exit_ts": timestamps[-1],
            "direction": pos,
            "bars_held": n - 1 - entry_bar,
            "gross_pnl_bps": gross,
            "net_pnl_bps": net,
            "cum_pnl_bps": cum_pnl,
        })

    return pd.DataFrame(trades)


def compute_metrics(df):
    if df.empty:
        return {"n_trades": 0, "total_pnl": 0, "gross_pnl": 0,
                "win_rate": 0, "sharpe": 0, "max_dd": 0,
                "profit_factor": 0, "avg_hold": 0, "avg_gross": 0, "avg_net": 0}
    pnl = df["net_pnl_bps"]
    gross = df["gross_pnl_bps"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    cum = df["cum_pnl_bps"]
    dd = cum - cum.cummax()
    tw = wins.sum() if len(wins) else 0
    tl = losses.abs().sum() if len(losses) else 1e-10
    sr = pnl.mean() / (pnl.std() + 1e-10) * np.sqrt(len(pnl)) if len(pnl) > 1 else 0
    return {
        "n_trades": len(df),
        "n_long": int((df["direction"] == 1).sum()),
        "n_short": int((df["direction"] == -1).sum()),
        "total_pnl": float(pnl.sum()),
        "gross_pnl": float(gross.sum()),
        "avg_gross": float(gross.mean()),
        "avg_net": float(pnl.mean()),
        "win_rate": float(len(wins) / len(pnl)) if len(pnl) else 0,
        "sharpe": float(sr),
        "max_dd": float(dd.min()),
        "profit_factor": float(tw / tl),
        "avg_hold": float(df["bars_held"].mean()),
    }


# ─────────────────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SOL/USDT")
    parser.add_argument("--horizon", type=int, default=300,
                        help="Forward return horizon in seconds (300=5min, 60=1min)")
    parser.add_argument("--model", default="all",
                        choices=["ridge", "gbm", "lasso", "lstm", "all"])
    parser.add_argument("--cost-bps", type=float, default=4.0)
    parser.add_argument("--top-features", type=int, default=40)
    args = parser.parse_args()

    horizon = args.horizon
    horizon_str = f"{horizon//60}min" if horizon >= 60 else f"{horizon}s"
    # Training stride: subsample to reduce overlap in targets
    train_stride = max(1, horizon // 10)
    lstm_stride = max(1, horizon // 30)

    logger.info("=" * 65)
    logger.info(f"SOL Momentum/Reversal V2  —  Horizon = {horizon_str} ({horizon}s)")
    logger.info(f"  特征采样: 1s bars, 窗口 = {FEATURE_WINDOWS}s (物理不变)")
    logger.info(f"  预测目标: next_{horizon_str}_return (累计 {horizon} bars)")
    logger.info(f"  训练 stride: {train_stride} (非重叠 target)")
    logger.info(f"  LSTM stride: {lstm_stride}, lookback: {LSTM_LOOKBACK}s")
    logger.info(f"  Cost: {args.cost_bps} bps/leg")
    logger.info(f"  Train: {SOL_TRAIN_DATES} ({len(SOL_TRAIN_DATES)} days)")
    logger.info(f"  Test:  {SOL_TEST_DATES} ({len(SOL_TEST_DATES)} days)")
    logger.info("=" * 65)

    # ── Load 1s features ──
    logger.info("\n[1] Loading train features (1s bars)...")
    train_df = load_1s_features(args.symbol, SOL_TRAIN_DATES)
    logger.info(f"  Train: {len(train_df):,} bars × {train_df.shape[1]} features")

    logger.info("\n[2] Loading test features (1s bars)...")
    test_df = load_1s_features(args.symbol, SOL_TEST_DATES)
    logger.info(f"  Test:  {len(test_df):,} bars × {test_df.shape[1]} features")

    # ── Target ──
    logger.info(f"\n[3] Target: next_{horizon_str}_return (bps)...")
    train_df["target"] = compute_forward_return(train_df, horizon)
    test_df["target"] = compute_forward_return(test_df, horizon)

    train_df.dropna(subset=["target"], inplace=True)
    test_df.dropna(subset=["target"], inplace=True)

    y_train_all = train_df["target"].values
    y_test_all = test_df["target"].values
    y_1s_test = test_df["return_1"].values  # for backtest PnL

    logger.info(f"  Train target: mean={np.nanmean(y_train_all):.2f}, std={np.nanstd(y_train_all):.2f} bps")
    logger.info(f"  Test  target: mean={np.nanmean(y_test_all):.2f}, std={np.nanstd(y_test_all):.2f} bps")

    # ── Feature columns ──
    exclude = {"target", "return_1", "log_return", "return_run_len"}
    feat_cols = [
        c for c in train_df.columns
        if c not in exclude
        and train_df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]
        and train_df[c].std() > 1e-10
    ]
    logger.info(f"  Candidate features: {len(feat_cols)}")

    # ── Feature selection (for Ridge/GBM) ──
    logger.info("\n[4] Feature selection...")
    selected = select_features(
        train_df[feat_cols].values, y_train_all, feat_cols,
        top_n=args.top_features, stride=train_stride,
    )

    X_train_sel = train_df[selected].values
    X_test_sel = test_df[selected].values
    X_train_all_feat = train_df[feat_cols].values
    X_test_all_feat = test_df[feat_cols].values

    # ── Models ──
    if args.model == "all":
        models = ["ridge", "gbm", "lasso", "lstm"]
    else:
        models = [args.model]

    # Backtest configs (scaled to 1s bars)
    if horizon >= 300:
        configs = [
            # name,             cost,          eq,   smooth, max_hold
            ("taker_q60_h5m",   args.cost_bps, 0.60, 15,    300),
            ("taker_q70_h5m",   args.cost_bps, 0.70, 15,    300),
            ("taker_q70_h10m",  args.cost_bps, 0.70, 30,    600),
            ("taker_q80_h10m",  args.cost_bps, 0.80, 30,    600),
            ("maker_q60_h5m",   1.0,           0.60, 15,    300),
            ("maker_q70_h5m",   1.0,           0.70, 15,    300),
            ("maker_q70_h10m",  1.0,           0.70, 30,    600),
            ("maker_q80_h10m",  1.0,           0.80, 30,    600),
        ]
    else:  # 1min
        configs = [
            ("taker_q60_h1m",   args.cost_bps, 0.60, 5,     60),
            ("taker_q60_h2m",   args.cost_bps, 0.60, 10,    120),
            ("taker_q70_h1m",   args.cost_bps, 0.70, 5,     60),
            ("taker_q70_h2m",   args.cost_bps, 0.70, 10,    120),
            ("taker_q80_h2m",   args.cost_bps, 0.80, 10,    120),
            ("maker_q60_h1m",   1.0,           0.60, 5,     60),
            ("maker_q60_h2m",   1.0,           0.60, 10,    120),
            ("maker_q70_h2m",   1.0,           0.70, 10,    120),
            ("maker_q80_h2m",   1.0,           0.80, 10,    120),
        ]

    all_results = {}

    for mname in models:
        logger.info(f"\n{'='*60}")
        logger.info(f"[5] Training: {mname.upper()}")
        logger.info(f"{'='*60}")

        t0 = time.time()

        if mname == "lstm":
            pred_bps, pred_idx, _, _ = train_lstm(
                train_df, test_df, horizon, feat_cols,
                lookback=LSTM_LOOKBACK, stride=lstm_stride,
            )
            y_bt = test_df["return_1"].values[pred_idx]
            ts_bt = test_df.index[pred_idx]
            y_fwd_bt = y_test_all[pred_idx]
            pred = pred_bps
        elif mname == "lasso":
            pred, _, _ = train_lasso(X_train_all_feat, y_train_all,
                                      X_test_all_feat, stride=train_stride)
            y_bt = y_1s_test
            ts_bt = test_df.index
            y_fwd_bt = y_test_all
        elif mname == "ridge":
            pred, _, _ = train_ridge(X_train_sel, y_train_all,
                                      X_test_sel, stride=train_stride)
            y_bt = y_1s_test
            ts_bt = test_df.index
            y_fwd_bt = y_test_all
        else:  # gbm
            pred, _, _ = train_gbm(X_train_sel, y_train_all,
                                    X_test_sel, stride=train_stride)
            y_bt = y_1s_test
            ts_bt = test_df.index
            y_fwd_bt = y_test_all

        logger.info(f"  Time: {time.time()-t0:.1f}s")
        logger.info(f"  Pred: mean={pred.mean():.2f}, std={pred.std():.2f}")

        # IC
        mask = ~(np.isnan(pred) | np.isnan(y_fwd_bt))
        ic = np.corrcoef(pred[mask], y_fwd_bt[mask])[0, 1]
        ric = pd.Series(pred[mask]).corr(pd.Series(y_fwd_bt[mask]), method="spearman")
        logger.info(f"  IC={ic:.4f}, RankIC={ric:.4f}")

        # Backtest sweep
        logger.info(f"\n  Backtesting (position-based)...")
        best_sr = -999
        best_cfg = ""
        model_results = {}

        for cname, cost, eq, sw, mh in configs:
            tr = backtest_position(
                pred, y_bt, ts_bt,
                cost_bps=cost, smooth_window=sw,
                entry_quantile=eq, max_hold_bars=mh,
            )
            met = compute_metrics(tr)
            model_results[cname] = met

            rt = cost * 2
            hold_str = f"{met['avg_hold']:.0f}s" if met['avg_hold'] > 0 else "0s"
            logger.info(
                f"    {cname:<20} RT={rt:.0f}bps | N={met['n_trades']:>4} "
                f"Gross={met['gross_pnl']:>+8.0f} Net={met['total_pnl']:>+8.0f} "
                f"WR={met['win_rate']:.1%} SR={met['sharpe']:>+6.2f} "
                f"PF={met['profit_factor']:.2f} Hold={hold_str}"
            )

            if met['sharpe'] > best_sr:
                best_sr = met['sharpe']
                best_cfg = cname

        all_results[mname] = model_results
        logger.info(f"\n  ★ Best: {best_cfg} (Sharpe={best_sr:.2f})")

        # Save best trades
        bc = [(c,e,s,m) for n,c,e,s,m in configs if n == best_cfg][0]
        tr = backtest_position(pred, y_bt, ts_bt,
                               cost_bps=bc[0], smooth_window=bc[2],
                               entry_quantile=bc[1], max_hold_bars=bc[3])
        sdir = DataPaths.backtest_dir(f"sol_{horizon_str}_v2/{mname}")
        if not tr.empty:
            tr.to_csv(sdir / "trades.csv", index=False)

    # ── Summary ──
    logger.info(f"\n{'='*65}")
    logger.info(f"SUMMARY  —  SOL {horizon_str} V2")
    logger.info(f"{'='*65}")

    for mname, cfg_res in all_results.items():
        logger.info(f"\n  {mname.upper()}:")
        logger.info(f"  {'Config':<20} | {'N':>4} | {'Gross':>7} | {'Net':>7} | {'WR':>5} | {'SR':>6} | {'PF':>5} | {'Hold':>6}")
        logger.info(f"  {'-'*75}")
        for cn, m in cfg_res.items():
            hold_s = f"{m['avg_hold']:.0f}s" if m['avg_hold'] > 0 else "0s"
            logger.info(
                f"  {cn:<20} | {m['n_trades']:>4} | {m['gross_pnl']:>+7.0f} | "
                f"{m['total_pnl']:>+7.0f} | {m['win_rate']:>4.1%} | "
                f"{m['sharpe']:>+6.2f} | {m['profit_factor']:>5.2f} | {hold_s:>6}"
            )

    summary = {
        "config": {
            "version": "v2",
            "symbol": args.symbol,
            "horizon": horizon,
            "horizon_str": horizon_str,
            "bar_freq": "1s",
            "feature_windows_s": FEATURE_WINDOWS,
            "lstm_lookback_s": LSTM_LOOKBACK,
            "train_stride": train_stride,
            "lstm_stride": lstm_stride,
            "n_features_all": len(feat_cols),
            "n_features_selected": len(selected),
            "top_features": selected[:15],
            "train_dates": SOL_TRAIN_DATES,
            "test_dates": SOL_TEST_DATES,
            "train_bars": len(train_df),
            "test_bars": len(test_df),
        },
        "results": all_results,
    }
    sp = DataPaths.backtest_dir(f"sol_{horizon_str}_v2") / "summary.json"
    with open(sp, "w") as fp:
        json.dump(summary, fp, indent=2, default=str)
    logger.info(f"\nSaved: {sp}")


if __name__ == "__main__":
    main()
