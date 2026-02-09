#!/usr/bin/env python3
"""
SOL 5min Momentum Prediction

自相关分析：SOL 5min lag-1 ACF = +0.0404（动量），daily t-stat = +2.02(**)
5min return std ≈ 42 bps → 预期条件收益 ≈ 0.04 * 42 ≈ 1.7 bps/bar
多 bar 持仓预期收益可覆盖交易成本。

Models:
  - Ridge:  L2 regularization, 预选 top-N 特征
  - GBM:    HistGradientBoosting, 预选 top-N 特征
  - Lasso:  L1 regularization (LassoCV), 用全部特征, 自动稀疏化
  - LSTM:   逐 bar 原始特征 (不做 rolling), lookback window = 6 bars (30min)
            避免 rolling 窗口引入 LSTM 窗口之外的噪音

特征窗口设计 (Ridge/GBM/Lasso):
  - 2 bars  (10 min) — 最近动量信号
  - 3 bars  (15 min) — 短期趋势
  - 6 bars  (30 min) — 中期趋势
  - 12 bars (1 hr)   — 最长窗口（不超过此值，因 15min/30min ACF 转负）

LSTM 特征设计:
  - 逐 bar 原始值: return, volume, buy_vol, sell_vol, imbalance, n_trades,
    bar_range, body_ratio, taker_levels, taker_imbalance 等 ~15 个
  - 仅保留 1-bar diff (momentum), 不做 rolling
  - LSTM lookback = 6 bars (30 min), 让网络自己学时序依赖

Usage:
    python scripts/train_sol_5min_momentum.py --model all
    python scripts/train_sol_5min_momentum.py --model lasso
    python scripts/train_sol_5min_momentum.py --model lstm
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

# 5min 动量窗口：2, 3, 6, 12 bars  =  10min, 15min, 30min, 1hr
# 不超过 12 bars（15min/30min 尺度 SOL ACF 转负，长窗口引入反向噪音）
FEATURE_WINDOWS = [2, 3, 6, 12]
FREQ = "5min"


# ─────────────────────────────────────────────────────────
# 1. Feature Computation (5min bars)
# ─────────────────────────────────────────────────────────

def compute_5min_features(trades: pd.DataFrame) -> pd.DataFrame:
    """
    从 aggTrades → 5min OHLCV bar 特征。
    窗口严格匹配 5min 动量信号物理特性。
    """
    ohlcv = resample_trades_to_ohlcv(trades, FREQ)
    if len(ohlcv) < 30:
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

    # ── A. Return & Momentum ──
    f["return_1"] = c.pct_change() * 10000  # bps
    f["log_return"] = np.log(c / c.shift(1))

    for w in FEATURE_WINDOWS:
        f[f"return_sum_{w}"] = f["return_1"].rolling(w, min_periods=1).sum()
        f[f"return_mean_{w}"] = f["return_1"].rolling(w, min_periods=1).mean()
        f[f"return_std_{w}"] = f["return_1"].rolling(w, min_periods=2).std()

    # Return momentum（当前 bar return vs 滚动均值）
    for w in [3, 6, 12]:
        f[f"return_mom_{w}"] = f["return_1"] - f[f"return_mean_{w}"]

    # 连续同向 bar 计数
    sign = np.sign(f["return_1"])
    sign_change = (sign != sign.shift(1))
    run_group = sign_change.cumsum()
    f["return_run_len"] = run_group.groupby(run_group).cumcount() + 1

    # ── B. Volume & Order Flow ──
    f["volume"] = v
    f["buy_volume"] = bv
    f["sell_volume"] = sv
    f["volume_imbalance"] = (bv - sv) / (v + 1e-10)
    f["n_trades"] = nt

    for w in FEATURE_WINDOWS:
        f[f"volume_ma_{w}"] = v.rolling(w, min_periods=1).mean()
        f[f"volume_ratio_{w}"] = v / (f[f"volume_ma_{w}"] + 1e-10)
        f[f"imbalance_ma_{w}"] = f["volume_imbalance"].rolling(w, min_periods=1).mean()
        # Signed flow
        signed_vol = bv - sv
        f[f"signed_flow_{w}"] = signed_vol.rolling(w, min_periods=1).sum()
        f[f"signed_flow_norm_{w}"] = f[f"signed_flow_{w}"] / (v.rolling(w, min_periods=1).sum() + 1e-10)

    # Buy/sell volume autocorrelation (动量持续性)
    for w in [3, 6, 12]:
        f[f"buy_vol_autocorr_{w}"] = bv.rolling(w, min_periods=3).corr(bv.shift(1))
        f[f"sell_vol_autocorr_{w}"] = sv.rolling(w, min_periods=3).corr(sv.shift(1))
        f[f"bv_sv_corr_{w}"] = bv.rolling(w, min_periods=3).corr(sv)

    # Volume skewness
    for w in [6, 12]:
        f[f"volume_skew_{w}"] = v.rolling(w, min_periods=4).skew()

    # ── C. Bar Structure ──
    hl = h - lo
    f["bar_range_bps"] = hl / (c + 1e-10) * 10000
    f["body_ratio"] = (c - o).abs() / (hl + 1e-10)
    f["upper_wick"] = (h - c.clip(lower=o).clip(upper=h)) / (hl + 1e-10)
    f["lower_wick"] = (c.clip(upper=o).clip(lower=lo) - lo) / (hl + 1e-10)
    # bar direction: +1 bullish, -1 bearish
    f["bar_direction"] = np.sign(c - o)

    for w in FEATURE_WINDOWS:
        f[f"bar_range_ma_{w}"] = f["bar_range_bps"].rolling(w, min_periods=1).mean()
        f[f"trades_ma_{w}"] = nt.rolling(w, min_periods=1).mean()
        f[f"trades_ratio_{w}"] = nt / (f[f"trades_ma_{w}"] + 1e-10)

    # ── D. Spread & Impact 代理 ──
    price_diff = c.diff().abs()
    for w in FEATURE_WINDOWS:
        f[f"spread_proxy_{w}"] = price_diff.rolling(w, min_periods=1).mean()
        cum_move = price_diff.rolling(w, min_periods=1).sum()
        cum_vol = v.rolling(w, min_periods=1).sum() + 1e-10
        f[f"impact_proxy_{w}"] = cum_move / cum_vol

    # ── E. Volatility Ratios ──
    for w in [3, 6, 12]:
        short_vol = f["return_1"].rolling(w, min_periods=2).std()
        long_vol = f["return_1"].rolling(w * 2, min_periods=3).std()
        f[f"vol_ratio_{w}"] = short_vol / (long_vol + 1e-10)

    f = f.replace([np.inf, -np.inf], np.nan).fillna(0)
    return f


def add_taker_features(f: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """添加 taker order 层面特征到 5min bars。"""
    from indicators.base.taker_flow import group_taker_orders

    taker = group_taker_orders(trades)
    if taker.empty or len(taker) < 10:
        return f

    taker_ts = taker.set_index("timestamp").sort_index()
    taker_ts = taker_ts[~taker_ts.index.duplicated(keep="last")]

    # Resample to 5min
    t_levels = taker_ts["levels_swept"].resample(FREQ).mean()
    t_volume = taker_ts["total_volume"].resample(FREQ).sum()
    t_impact = taker_ts["price_impact_pct"].resample(FREQ).mean()
    t_count = taker_ts["levels_swept"].resample(FREQ).count()

    # Buy/sell taker
    t_buy = taker_ts.loc[taker_ts["side"] == "buy", "total_volume"].resample(FREQ).sum()
    t_sell = taker_ts.loc[taker_ts["side"] == "sell", "total_volume"].resample(FREQ).sum()
    t_buy = t_buy.reindex(t_levels.index).fillna(0)
    t_sell = t_sell.reindex(t_levels.index).fillna(0)

    # Large taker ratio
    large = (taker_ts["levels_swept"] >= 3).astype(float)
    large_vol = (taker_ts["total_volume"] * large).resample(FREQ).sum()
    total_vol_t = taker_ts["total_volume"].resample(FREQ).sum() + 1e-10

    f["taker_levels"] = t_levels.reindex(f.index).fillna(0)
    f["taker_volume"] = t_volume.reindex(f.index).fillna(0)
    f["taker_impact_pct"] = t_impact.reindex(f.index).fillna(0)
    f["taker_count"] = t_count.reindex(f.index).fillna(0)
    f["taker_imbalance"] = ((t_buy - t_sell) / (t_buy + t_sell + 1e-10)).reindex(f.index).fillna(0)
    f["taker_large_ratio"] = (large_vol / total_vol_t).reindex(f.index).fillna(0)

    for w in [2, 3, 6]:
        f[f"taker_levels_ma_{w}"] = f["taker_levels"].rolling(w, min_periods=1).mean()
        f[f"taker_imb_ma_{w}"] = f["taker_imbalance"].rolling(w, min_periods=1).mean()
        f[f"taker_count_ma_{w}"] = f["taker_count"].rolling(w, min_periods=1).mean()
        f[f"taker_impact_ma_{w}"] = f["taker_impact_pct"].rolling(w, min_periods=1).mean()

    f = f.replace([np.inf, -np.inf], np.nan).fillna(0)
    return f


# ─────────────────────────────────────────────────────────
# 2. Data Loading
# ─────────────────────────────────────────────────────────

def load_5min_features(symbol: str, dates: List[str]) -> pd.DataFrame:
    """逐天加载 → 计算 5min 特征 → concat。"""
    all_dfs = []
    for d_str in dates:
        dt = date.fromisoformat(d_str)
        cache_path = DataPaths.features(symbol, "5min_features", dt)

        if cache_path.exists():
            logger.info(f"  [Cache] {d_str}")
            all_dfs.append(pd.read_parquet(cache_path))
        else:
            logger.info(f"  [Compute] {d_str}...")
            trades = load_agg_trades(symbol, dt, dt + timedelta(days=1))
            if trades.empty:
                continue
            logger.info(f"    {len(trades):,} aggTrades")
            df = compute_5min_features(trades)
            if df.empty:
                del trades; gc.collect()
                continue
            df = add_taker_features(df, trades)
            df.to_parquet(cache_path)
            logger.info(f"    Saved: {len(df)} bars × {df.shape[1]} features")
            all_dfs.append(df)
            del trades; gc.collect()

    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs).sort_index()


# ─────────────────────────────────────────────────────────
# 3. Feature Selection
# ─────────────────────────────────────────────────────────

def select_features(X, y, names, top_n=30):
    from sklearn.linear_model import Ridge
    from sklearn.inspection import permutation_importance
    from sklearn.preprocessing import StandardScaler

    logger.info(f"  Fitting quick Ridge on {X.shape}...")
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    m = Ridge(alpha=1.0).fit(Xs, y)

    val_n = max(200, len(X) // 5)
    perm = permutation_importance(m, Xs[-val_n:], y[-val_n:],
                                   n_repeats=10, random_state=42, scoring="r2")
    imp = perm.importances_mean
    top_idx = np.argsort(imp)[::-1][:top_n]
    selected = [names[i] for i in top_idx]

    logger.info(f"  Top {min(top_n, 15)} features:")
    for rank, i in enumerate(top_idx[:15]):
        logger.info(f"    {rank+1:>3}. {names[i]:<30} imp={imp[i]:.6f}")
    if top_n > 15:
        logger.info(f"    ... ({top_n - 15} more)")
    return selected


# ─────────────────────────────────────────────────────────
# 4. Models
# ─────────────────────────────────────────────────────────

def train_ridge(X_tr, y_tr, X_te, names):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    Xtr = sc.fit_transform(X_tr)
    Xte = sc.transform(X_te)
    m = Ridge(alpha=10.0).fit(Xtr, y_tr)
    p_tr = m.predict(Xtr)
    p_te = m.predict(Xte)
    r2 = 1 - np.sum((y_tr - p_tr)**2) / np.sum((y_tr - y_tr.mean())**2)
    logger.info(f"  [Ridge] Train R²={r2:.4f}")
    return p_te, m, sc


def train_gbm(X_tr, y_tr, X_te, names):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    Xtr = sc.fit_transform(X_tr)
    Xte = sc.transform(X_te)

    # 5min 数据量小，需要更强正则化防止过拟合
    m = HistGradientBoostingRegressor(
        max_iter=300,
        max_depth=3,
        learning_rate=0.03,
        min_samples_leaf=20,
        l2_regularization=5.0,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=20,
        random_state=42,
    )
    m.fit(Xtr, y_tr)
    p_tr = m.predict(Xtr)
    p_te = m.predict(Xte)
    r2 = 1 - np.sum((y_tr - p_tr)**2) / np.sum((y_tr - y_tr.mean())**2)
    logger.info(f"  [GBM] Train R²={r2:.4f}, n_iter={m.n_iter_}")
    return p_te, m, sc


def train_lasso(X_tr, y_tr, X_te, names):
    """
    LassoCV: L1 正则化自动特征选择。
    用全部候选特征 → L1 把无用特征系数压为零。
    """
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    Xtr = sc.fit_transform(X_tr)
    Xte = sc.transform(X_te)

    m = LassoCV(
        cv=5,
        n_alphas=100,
        max_iter=5000,
        random_state=42,
        n_jobs=-1,
    )
    m.fit(Xtr, y_tr)

    # 活跃特征
    coefs = m.coef_
    n_active = int(np.sum(np.abs(coefs) > 1e-8))
    active_idx = np.where(np.abs(coefs) > 1e-8)[0]
    sorted_active = active_idx[np.argsort(np.abs(coefs[active_idx]))[::-1]]

    logger.info(f"  [Lasso] alpha={m.alpha_:.6f}, "
                f"active features: {n_active}/{len(names)}")
    logger.info(f"  Top active features:")
    for rank, i in enumerate(sorted_active[:20]):
        logger.info(f"    {rank+1:>3}. {names[i]:<30} coef={coefs[i]:+.6f}")

    p_tr = m.predict(Xtr)
    p_te = m.predict(Xte)
    r2 = 1 - np.sum((y_tr - p_tr)**2) / np.sum((y_tr - y_tr.mean())**2)
    logger.info(f"  [Lasso] Train R²={r2:.4f}")
    return p_te, m, sc


# ─────────────────────────────────────────────────────────
# 4b. LSTM Model
# ─────────────────────────────────────────────────────────

# LSTM 使用逐 bar 原始特征（不做 rolling），避免窗口噪音叠加
LSTM_RAW_FEATURES = [
    # 逐 bar 原始值
    "return_1", "volume", "buy_volume", "sell_volume",
    "volume_imbalance", "n_trades",
    "bar_range_bps", "body_ratio", "bar_direction",
    # taker 聚合值（5min bar 内的聚合，不是 rolling）
    "taker_levels", "taker_volume", "taker_impact_pct",
    "taker_count", "taker_imbalance", "taker_large_ratio",
]
# 仅保留 1-bar diff 作为 "momentum" 信号（不用 rolling）
LSTM_DIFF_FEATURES = [
    "return_1", "volume", "buy_volume", "sell_volume",
    "n_trades", "taker_count", "taker_imbalance",
]

LSTM_LOOKBACK = 6  # 6 bars = 30 min（匹配 ACF 有效范围）


def prepare_lstm_data(df: pd.DataFrame, lookback: int = LSTM_LOOKBACK):
    """
    从 5min bar DataFrame 构造 LSTM 输入。
    
    每个样本 = lookback 个连续 bar 的原始特征矩阵 (lookback, n_features)。
    target = 下一个 bar 的 return。
    
    特征 = raw values + 1-bar diff（不用 rolling，避免窗口外噪音）。
    """
    # 构建特征矩阵
    raw_cols = [c for c in LSTM_RAW_FEATURES if c in df.columns]
    diff_cols = [c for c in LSTM_DIFF_FEATURES if c in df.columns]

    feat = df[raw_cols].copy()
    for c in diff_cols:
        feat[f"{c}_diff1"] = df[c].diff()
    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0)

    feature_names = list(feat.columns)
    values = feat.values  # (T, F)

    # 归一化：用训练集的 mean/std（调用方负责 split）
    # 这里只返回 raw data, 由 train_lstm 做归一化

    # 构造滑动窗口
    X, y, idx = [], [], []
    target = df["target"].values
    for i in range(lookback, len(values)):
        if np.isnan(target[i]):
            continue
        X.append(values[i - lookback:i])  # (lookback, F)
        y.append(target[i])
        idx.append(i)

    X = np.array(X, dtype=np.float32)  # (N, lookback, F)
    y = np.array(y, dtype=np.float32)
    idx = np.array(idx)
    return X, y, idx, feature_names


def train_lstm(train_df, test_df, lookback=LSTM_LOOKBACK):
    """
    训练 LSTM 回归模型。
    
    架构: LSTM(hidden=32, layers=1) → Dropout → Linear → 1
    用 AdamW + OneCycleLR，early stopping on validation loss。
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    logger.info(f"  [LSTM] Preparing data (lookback={lookback} bars = {lookback*5}min)...")

    X_tr, y_tr, idx_tr, feat_names = prepare_lstm_data(train_df, lookback)
    X_te, y_te, idx_te, _ = prepare_lstm_data(test_df, lookback)

    logger.info(f"    Train: {X_tr.shape} ({X_tr.shape[0]} samples × {lookback} steps × {X_tr.shape[2]} features)")
    logger.info(f"    Test:  {X_te.shape}")
    logger.info(f"    Features ({len(feat_names)}): {feat_names[:10]}...")

    # 归一化（用训练集统计量, 沿 sample+time 维度）
    tr_flat = X_tr.reshape(-1, X_tr.shape[2])
    mu = tr_flat.mean(axis=0)
    sd = tr_flat.std(axis=0) + 1e-8
    X_tr = (X_tr - mu) / sd
    X_te = (X_te - mu) / sd

    # Target 归一化
    y_mu, y_sd = y_tr.mean(), y_tr.std() + 1e-8
    y_tr_n = (y_tr - y_mu) / y_sd
    y_te_n = (y_te - y_mu) / y_sd

    # Validation split (last 20% of train)
    val_n = max(50, len(X_tr) // 5)
    X_val, y_val_n = X_tr[-val_n:], y_tr_n[-val_n:]
    X_tr2, y_tr2_n = X_tr[:-val_n], y_tr_n[:-val_n]

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    logger.info(f"    Device: {device}")

    # Model
    class LSTMReg(nn.Module):
        def __init__(self, input_dim, hidden=32, layers=1, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden, layers,
                                batch_first=True, dropout=dropout if layers > 1 else 0)
            self.drop = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(self.drop(out[:, -1, :])).squeeze(-1)

    n_feat = X_tr.shape[2]
    model = LSTMReg(n_feat, hidden=32, layers=1, dropout=0.15).to(device)
    logger.info(f"    Model params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    tr_ds = TensorDataset(torch.from_numpy(X_tr2), torch.from_numpy(y_tr2_n))
    tr_dl = DataLoader(tr_ds, batch_size=64, shuffle=True, drop_last=False)

    val_t = torch.from_numpy(X_val).to(device)
    val_y = torch.from_numpy(y_val_n).to(device)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=1e-3, epochs=80, steps_per_epoch=len(tr_dl))

    # Training
    best_val_loss = 1e10
    patience, wait = 15, 0
    best_state = None

    for epoch in range(80):
        model.train()
        ep_loss = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            ep_loss += loss.item() * len(xb)
        ep_loss /= len(X_tr2)

        model.eval()
        with torch.no_grad():
            val_pred = model(val_t)
            val_loss = loss_fn(val_pred, val_y).item()

        if epoch % 5 == 0 or epoch < 3:
            logger.info(f"    Epoch {epoch:>3}: train_loss={ep_loss:.6f}, val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info(f"    Early stop at epoch {epoch} (best val_loss={best_val_loss:.6f})")
                break

    # Load best
    model.load_state_dict(best_state)
    model.eval()

    # Predict
    with torch.no_grad():
        te_t = torch.from_numpy(X_te).to(device)
        pred_n = model(te_t).cpu().numpy()

    # 反归一化
    pred_bps = pred_n * y_sd + y_mu

    # Train R²
    with torch.no_grad():
        all_tr = torch.from_numpy(X_tr).to(device)
        tr_pred_n = model(all_tr).cpu().numpy()
    tr_pred_bps = tr_pred_n * y_sd + y_mu
    r2 = 1 - np.sum((y_tr - tr_pred_bps)**2) / np.sum((y_tr - y_tr.mean())**2)
    logger.info(f"  [LSTM] Train R²={r2:.4f}, best_val_loss={best_val_loss:.6f}")

    return pred_bps, idx_te, model, (mu, sd, y_mu, y_sd)


# ─────────────────────────────────────────────────────────
# 5. Position-Based Backtest
# ─────────────────────────────────────────────────────────

def backtest_position(
    pred: np.ndarray,
    y_actual: np.ndarray,
    timestamps: pd.DatetimeIndex,
    cost_bps: float = 4.0,
    smooth_window: int = 2,
    entry_quantile: float = 0.6,
    max_hold_bars: int = 12,
) -> pd.DataFrame:
    """
    持仓模型回测：
      - EMA 平滑预测信号
      - 信号强度超过分位阈值时入场（方向 = sign）
      - 持仓直到信号翻转或超时
      - 成本仅在持仓变化时收取
    """
    n = len(pred)
    rt_cost = cost_bps * 2

    # 平滑信号
    sig = pd.Series(pred).ewm(span=smooth_window, adjust=False).mean().values

    # 入场阈值（用前 20% 数据 warm-up）
    warm = max(50, n // 10)
    abs_s = np.abs(sig[:warm])
    pos_abs = abs_s[abs_s > 0]
    thresh = np.quantile(pos_abs, entry_quantile) if len(pos_abs) > 5 else 0.5

    trades = []
    cum_pnl = 0.0
    pos = 0
    entry_bar = 0

    for i in range(warm, n):
        # 动态阈值更新
        if i > warm + 50 and i % 50 == 0:
            lb = min(i, 500)
            a = np.abs(sig[i - lb:i])
            a = a[a > 0]
            if len(a) > 5:
                thresh = np.quantile(a, entry_quantile)

        want = 0
        if sig[i] > thresh:
            want = 1
        elif sig[i] < -thresh:
            want = -1

        # 超时
        if pos != 0 and (i - entry_bar) >= max_hold_bars:
            want = 0
        # 信号反转
        if pos > 0 and sig[i] < 0:
            want = -1 if sig[i] < -thresh else 0
        elif pos < 0 and sig[i] > 0:
            want = 1 if sig[i] > thresh else 0

        if want != pos:
            if pos != 0:
                gross = sum(y_actual[entry_bar + 1:i + 1]) * pos
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

    # 关闭尾仓
    if pos != 0:
        gross = sum(y_actual[entry_bar + 1:n]) * pos
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


def compute_metrics(df: pd.DataFrame) -> Dict:
    if df.empty:
        return {"n_trades": 0, "total_pnl_bps": 0, "gross_pnl_bps": 0,
                "win_rate": 0, "sharpe": 0, "max_dd_bps": 0,
                "profit_factor": 0, "avg_hold": 0, "n_long": 0, "n_short": 0,
                "avg_gross": 0, "avg_net": 0}
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
        "total_pnl_bps": float(pnl.sum()),
        "gross_pnl_bps": float(gross.sum()),
        "avg_gross": float(gross.mean()),
        "avg_net": float(pnl.mean()),
        "win_rate": float(len(wins) / len(pnl)) if len(pnl) else 0,
        "sharpe": float(sr),
        "max_dd_bps": float(dd.min()),
        "profit_factor": float(tw / tl),
        "avg_hold": float(df["bars_held"].mean()),
    }


# ─────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SOL/USDT")
    parser.add_argument("--model", default="all", choices=["ridge", "gbm", "lasso", "lstm", "all"])
    parser.add_argument("--cost-bps", type=float, default=4.0)
    parser.add_argument("--top-features", type=int, default=30)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("SOL 5min Momentum Prediction")
    logger.info(f"  ACF basis: lag-1 = +0.0404 (momentum, t-stat=+2.02)")
    logger.info(f"  Feature windows: {FEATURE_WINDOWS} bars = {[w*5 for w in FEATURE_WINDOWS]} min")
    logger.info(f"  Max window: {max(FEATURE_WINDOWS)} bars = {max(FEATURE_WINDOWS)*5} min")
    logger.info(f"  Cost: {args.cost_bps} bps/leg")
    logger.info(f"  Train: {SOL_TRAIN_DATES} (7 days)")
    logger.info(f"  Test:  {SOL_TEST_DATES} (3 days)")
    logger.info("=" * 60)

    # ── Load ──
    logger.info("\n[1] Loading training features...")
    train_df = load_5min_features(args.symbol, SOL_TRAIN_DATES)
    logger.info(f"  Train: {len(train_df):,} bars × {train_df.shape[1]} features")

    logger.info("\n[2] Loading test features...")
    test_df = load_5min_features(args.symbol, SOL_TEST_DATES)
    logger.info(f"  Test:  {len(test_df):,} bars × {test_df.shape[1]} features")

    # ── Target ──
    logger.info("\n[3] Target: next_5min_return (bps)...")
    for df in [train_df, test_df]:
        df["target"] = df["return_1"].shift(-1)
    train_df.dropna(subset=["target"], inplace=True)
    test_df.dropna(subset=["target"], inplace=True)

    # ── Features ──
    exclude = {"target", "return_1", "log_return", "return_run_len"}
    feat_cols = [
        c for c in train_df.columns
        if c not in exclude
        and train_df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]
        and train_df[c].std() > 1e-10
    ]
    logger.info(f"  Candidate features: {len(feat_cols)}")

    y_train = train_df["target"].values
    y_test = test_df["target"].values

    logger.info(f"  Train target: mean={y_train.mean():.2f} bps, std={y_train.std():.2f} bps")
    logger.info(f"  Test  target: mean={y_test.mean():.2f} bps, std={y_test.std():.2f} bps")

    # ── Feature selection (for Ridge/GBM only) ──
    logger.info("\n[4] Feature selection...")
    selected = select_features(
        train_df[feat_cols].values, y_train, feat_cols,
        top_n=args.top_features,
    )

    X_train_sel = train_df[selected].values
    X_test_sel = test_df[selected].values

    # Full feature matrix (for Lasso — no pre-selection)
    X_train_all = train_df[feat_cols].values
    X_test_all = test_df[feat_cols].values

    # ── Train & Backtest ──
    if args.model == "all":
        models = ["ridge", "gbm", "lasso", "lstm"]
    else:
        models = [args.model]
    all_results = {}

    # Backtest config sweep
    configs = [
        # name,        cost, entry_q, smooth, max_hold
        ("taker_q60_h6",   args.cost_bps, 0.60, 2,  6),
        ("taker_q60_h12",  args.cost_bps, 0.60, 2,  12),
        ("taker_q70_h12",  args.cost_bps, 0.70, 2,  12),
        ("taker_q70_h24",  args.cost_bps, 0.70, 3,  24),
        ("taker_q80_h12",  args.cost_bps, 0.80, 2,  12),
        ("taker_q80_h24",  args.cost_bps, 0.80, 3,  24),
        ("maker_q60_h12",  1.0,           0.60, 2,  12),
        ("maker_q70_h12",  1.0,           0.70, 2,  12),
        ("maker_q70_h24",  1.0,           0.70, 3,  24),
        ("maker_q80_h24",  1.0,           0.80, 3,  24),
    ]

    for mname in models:
        logger.info(f"\n{'='*55}")
        logger.info(f"[5] Training: {mname.upper()}")
        logger.info(f"{'='*55}")

        t0 = time.time()

        if mname == "lstm":
            # LSTM 用独立的 raw 特征 pipeline
            pred_bps, pred_idx, lstm_model, lstm_params = train_lstm(train_df, test_df)
            # pred_idx 是 test_df 中的 row 索引（由于 lookback 裁剪）
            y_test_lstm = test_df["target"].values[pred_idx]
            ts_lstm = test_df.index[pred_idx]
            pred = pred_bps
            y_bt = y_test_lstm
            ts_bt = ts_lstm
        elif mname == "lasso":
            # Lasso 用全部候选特征
            pred, model_obj, scaler = train_lasso(
                X_train_all, y_train, X_test_all, feat_cols)
            y_bt = y_test
            ts_bt = test_df.index
        elif mname == "ridge":
            pred, model_obj, scaler = train_ridge(
                X_train_sel, y_train, X_test_sel, selected)
            y_bt = y_test
            ts_bt = test_df.index
        else:  # gbm
            pred, model_obj, scaler = train_gbm(
                X_train_sel, y_train, X_test_sel, selected)
            y_bt = y_test
            ts_bt = test_df.index

        logger.info(f"  Time: {time.time()-t0:.1f}s")
        logger.info(f"  Pred: mean={pred.mean():.2f}, std={pred.std():.2f}, "
                     f"min={pred.min():.2f}, max={pred.max():.2f}")

        # IC (information coefficient)
        ic = np.corrcoef(pred, y_bt)[0, 1]
        rank_ic = pd.Series(pred).corr(pd.Series(y_bt), method="spearman")
        logger.info(f"  IC={ic:.4f}, RankIC={rank_ic:.4f}")

        # 多组参数扫描
        logger.info(f"\n  Backtesting (parameter sweep)...")
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
            logger.info(
                f"    {cname:<18} RT={rt:.0f}bps | N={met['n_trades']:>4} "
                f"Gross={met['gross_pnl_bps']:>+8.0f} Net={met['total_pnl_bps']:>+8.0f} "
                f"WR={met['win_rate']:.1%} SR={met['sharpe']:>+6.2f} "
                f"PF={met['profit_factor']:.2f} Hold={met['avg_hold']:.1f}bars"
            )

            if met['sharpe'] > best_sr:
                best_sr = met['sharpe']
                best_cfg = cname

        all_results[mname] = model_results
        logger.info(f"\n  ★ Best: {best_cfg} (Sharpe={best_sr:.2f})")

        # Save trades for best config
        bc = [(c,e,s,m) for n,c,e,s,m in configs if n == best_cfg][0]
        tr = backtest_position(pred, y_bt, ts_bt,
                               cost_bps=bc[0], smooth_window=bc[2],
                               entry_quantile=bc[1], max_hold_bars=bc[3])
        sdir = DataPaths.backtest_dir(f"sol_5min_momentum/{mname}")
        if not tr.empty:
            tr.to_csv(sdir / "trades.csv", index=False)

    # ── Summary ──
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")

    for mname, cfg_res in all_results.items():
        logger.info(f"\n  {mname.upper()}:")
        logger.info(f"  {'Config':<18} | {'N':>4} | {'Gross':>7} | {'Net':>7} | {'WR':>5} | {'SR':>6} | {'PF':>5} | {'Hold':>5}")
        logger.info(f"  {'-'*72}")
        for cn, m in cfg_res.items():
            logger.info(
                f"  {cn:<18} | {m['n_trades']:>4} | {m['gross_pnl_bps']:>+7.0f} | "
                f"{m['total_pnl_bps']:>+7.0f} | {m['win_rate']:>4.1%} | "
                f"{m['sharpe']:>+6.2f} | {m['profit_factor']:>5.2f} | {m['avg_hold']:>4.1f}b"
            )

    summary = {
        "config": {
            "symbol": args.symbol, "freq": FREQ,
            "signal_type": "momentum (lag-1 ACF = +0.0404, t-stat=+2.02)",
            "feature_windows_bars": FEATURE_WINDOWS,
            "feature_windows_min": [w * 5 for w in FEATURE_WINDOWS],
            "n_features_ridge_gbm": len(selected),
            "n_features_lasso": len(feat_cols),
            "lstm_lookback": LSTM_LOOKBACK,
            "lstm_raw_features": LSTM_RAW_FEATURES,
            "top_features_ridge_gbm": selected[:15],
            "train_dates": SOL_TRAIN_DATES,
            "test_dates": SOL_TEST_DATES,
            "train_bars": len(train_df),
            "test_bars": len(test_df),
            "models_run": models,
        },
        "results": all_results,
    }
    sp = DataPaths.backtest_dir("sol_5min_momentum") / "summary.json"
    with open(sp, "w") as fp:
        json.dump(summary, fp, indent=2, default=str)
    logger.info(f"\nSaved: {sp}")


if __name__ == "__main__":
    main()
