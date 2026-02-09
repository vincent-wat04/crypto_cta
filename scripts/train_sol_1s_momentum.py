#!/usr/bin/env python3
"""
SOL 1s Momentum Prediction

基于自相关分析，SOL 在 1s 频率表现出显著正 lag-1 ACF (+0.035)，
即短期动量效应。本脚本：
  1. 逐天加载 SOL aggTrades → 1s bars
  2. 使用极短窗口（3-20 bars = 3-20s）计算特征
  3. 预测目标：next_1s_return (bps)
  4. 模型：Ridge / GBM Regressor
  5. 按交易成本阈值过滤信号，回测

关键：窗口设计匹配 1s 动量信号物理特性，避免长窗口引入反向噪音。

Usage:
    python scripts/train_sol_1s_momentum.py
    python scripts/train_sol_1s_momentum.py --model all --cost-bps 4
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


# ─────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────

SOL_TRAIN_DATES = [
    "2026-01-28", "2026-01-29", "2026-01-30", "2026-01-31",
    "2026-02-01", "2026-02-02", "2026-02-03",
]
SOL_TEST_DATES = [
    "2026-02-04", "2026-02-05", "2026-02-06",
]

# 1s 动量特征窗口：极短，匹配信号物理特性
# 避免 60+ bar 窗口（在 1min 尺度 ACF 已转负，会引入反向噪音）
FEATURE_WINDOWS = [3, 5, 10, 20]  # 3s, 5s, 10s, 20s


# ─────────────────────────────────────────────────────────
# 1. 1s Bar Feature Computation
# ─────────────────────────────────────────────────────────

def compute_1s_features(trades: pd.DataFrame) -> pd.DataFrame:
    """
    从 aggTrades 计算 1s bar 特征，窗口设计匹配动量信号。

    特征分组：
      A. Return 特征（核心动量信号）
      B. Volume 特征（交易强度/方向）
      C. Trade 微观结构（市场微观状态）
      D. Spread 代理（流动性状态）
    """
    freq = "1s"

    # ── 0. Build 1s OHLCV ──
    ohlcv = resample_trades_to_ohlcv(trades, freq)
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

    # ── A. Return 特征 ──
    f["return_1"] = c.pct_change() * 10000  # bps
    f["log_return"] = np.log(c / c.shift(1))

    for w in FEATURE_WINDOWS:
        f[f"return_sum_{w}"] = f["return_1"].rolling(w, min_periods=1).sum()
        f[f"return_std_{w}"] = f["return_1"].rolling(w, min_periods=2).std()
        f[f"return_skew_{w}"] = f["return_1"].rolling(w, min_periods=3).skew()

    # 连续同向 return 计数
    f["return_sign"] = np.sign(f["return_1"])
    sign_change = (f["return_sign"] != f["return_sign"].shift(1))
    run_group = sign_change.cumsum()
    f["return_run_len"] = run_group.groupby(run_group).cumcount() + 1
    f.drop(columns=["return_sign"], inplace=True)

    # ── B. Volume 特征 ──
    f["volume"] = v
    f["buy_volume"] = bv
    f["sell_volume"] = sv
    f["volume_imbalance"] = (bv - sv) / (v + 1e-10)
    f["n_trades"] = ohlcv.get("n_trades", pd.Series(0, index=ohlcv.index)).astype(float)

    for w in FEATURE_WINDOWS:
        f[f"volume_ma_{w}"] = v.rolling(w, min_periods=1).mean()
        f[f"volume_ratio_{w}"] = v / (f[f"volume_ma_{w}"] + 1e-10)
        f[f"imbalance_ma_{w}"] = f["volume_imbalance"].rolling(w, min_periods=1).mean()
        # Signed volume flow (累积方向量)
        signed_vol = bv - sv
        f[f"signed_flow_{w}"] = signed_vol.rolling(w, min_periods=1).sum()
        f[f"signed_flow_norm_{w}"] = f[f"signed_flow_{w}"] / (v.rolling(w, min_periods=1).sum() + 1e-10)

    # ── C. Trade 微观结构 ──
    # Bar range (波动性)
    hl = h - lo
    f["bar_range_bps"] = hl / (c + 1e-10) * 10000
    f["body_ratio"] = (c - o).abs() / (hl + 1e-10)
    # 上下影线
    f["upper_wick"] = (h - c.clip(lower=o).clip(upper=h)) / (hl + 1e-10)
    f["lower_wick"] = (c.clip(upper=o).clip(lower=lo) - lo) / (hl + 1e-10)

    for w in FEATURE_WINDOWS:
        f[f"bar_range_ma_{w}"] = f["bar_range_bps"].rolling(w, min_periods=1).mean()
        f[f"trades_ma_{w}"] = f["n_trades"].rolling(w, min_periods=1).mean()
        f[f"trades_ratio_{w}"] = f["n_trades"] / (f[f"trades_ma_{w}"] + 1e-10)

    # ── D. Spread & Impact 代理（从 trade 数据推导）──
    # trade_diff = |p[t] - p[t-1]| 的均值作为 spread proxy
    price_diff = c.diff().abs()
    for w in FEATURE_WINDOWS:
        f[f"spread_proxy_{w}"] = price_diff.rolling(w, min_periods=1).mean()
        f[f"spread_proxy_std_{w}"] = price_diff.rolling(w, min_periods=2).std()

    # Volume-weighted price movement (impact 代理)
    price_move = (c - c.shift(1)).abs()
    for w in FEATURE_WINDOWS:
        cum_move = price_move.rolling(w, min_periods=1).sum()
        cum_vol = v.rolling(w, min_periods=1).sum() + 1e-10
        f[f"impact_proxy_{w}"] = cum_move / cum_vol

    # ── E. 额外衍生：buy/sell volume 自相关 ──
    for w in [5, 10, 20]:
        if w > 3:
            f[f"buy_vol_autocorr_{w}"] = bv.rolling(w, min_periods=3).corr(bv.shift(1))
            f[f"sell_vol_autocorr_{w}"] = sv.rolling(w, min_periods=3).corr(sv.shift(1))
            f[f"bv_sv_corr_{w}"] = bv.rolling(w, min_periods=3).corr(sv)

    # ── 清理 ──
    f = f.replace([np.inf, -np.inf], np.nan).fillna(0)
    return f


def compute_1s_features_from_trades_inplace(
    trades: pd.DataFrame,
) -> pd.DataFrame:
    """
    从原始 trades 计算 1s 特征 + taker order 衍生特征。
    taker 特征也使用短窗口。
    """
    from indicators.base.taker_flow import group_taker_orders

    f = compute_1s_features(trades)
    if f.empty:
        return f

    # Taker order 聚合特征
    logger.info("  Grouping taker orders...")
    taker = group_taker_orders(trades)
    if taker.empty or len(taker) < 10:
        return f

    taker_ts = taker.set_index("timestamp").sort_index()
    taker_ts = taker_ts[~taker_ts.index.duplicated(keep="last")]

    # Resample taker features to 1s
    t_levels = taker_ts["levels_swept"].resample("1s").mean()
    t_volume = taker_ts["total_volume"].resample("1s").sum()
    t_impact = taker_ts["price_impact_pct"].resample("1s").mean()
    t_count = taker_ts["levels_swept"].resample("1s").count()

    # Buy/sell taker volume
    buy_taker = taker_ts.loc[taker_ts["side"] == "buy", "total_volume"]
    sell_taker = taker_ts.loc[taker_ts["side"] == "sell", "total_volume"]
    t_buy = buy_taker.resample("1s").sum().reindex(t_levels.index).fillna(0)
    t_sell = sell_taker.resample("1s").sum().reindex(t_levels.index).fillna(0)

    # Large taker (>= 3 levels) ratio
    large_mask = (taker_ts["levels_swept"] >= 3).astype(float)
    large_vol = (taker_ts["total_volume"] * large_mask).resample("1s").sum()
    total_vol_t = taker_ts["total_volume"].resample("1s").sum() + 1e-10

    # Align to feature index
    f["taker_levels_swept"] = t_levels.reindex(f.index).fillna(0)
    f["taker_volume"] = t_volume.reindex(f.index).fillna(0)
    f["taker_impact_pct"] = t_impact.reindex(f.index).fillna(0)
    f["taker_count"] = t_count.reindex(f.index).fillna(0)
    f["taker_imbalance"] = ((t_buy - t_sell) / (t_buy + t_sell + 1e-10)).reindex(f.index).fillna(0)
    f["taker_large_ratio"] = (large_vol / total_vol_t).reindex(f.index).fillna(0)

    # Short-window rolling on taker features
    for w in [3, 5, 10]:
        f[f"taker_levels_ma_{w}"] = f["taker_levels_swept"].rolling(w, min_periods=1).mean()
        f[f"taker_imb_ma_{w}"] = f["taker_imbalance"].rolling(w, min_periods=1).mean()
        f[f"taker_count_ma_{w}"] = f["taker_count"].rolling(w, min_periods=1).mean()

    f = f.replace([np.inf, -np.inf], np.nan).fillna(0)
    return f


# ─────────────────────────────────────────────────────────
# 2. Data Loading
# ─────────────────────────────────────────────────────────

def load_1s_features_for_dates(
    symbol: str, dates: List[str],
) -> pd.DataFrame:
    """逐天加载 trades → 计算 1s 特征 → concat。内存友好。"""
    all_dfs = []
    for d_str in dates:
        dt = date.fromisoformat(d_str)
        cache_key = f"1s_features"
        cache_path = DataPaths.features(symbol, cache_key, dt)

        if cache_path.exists():
            logger.info(f"  [Cache hit] {d_str} 1s features")
            df = pd.read_parquet(cache_path)
            all_dfs.append(df)
        else:
            logger.info(f"  [Computing] {d_str} 1s features...")
            trades = load_agg_trades(symbol, dt, dt + timedelta(days=1))
            if trades.empty:
                continue
            logger.info(f"    {len(trades):,} aggTrades → computing features...")
            df = compute_1s_features_from_trades_inplace(trades)
            if df.empty:
                del trades
                gc.collect()
                continue
            df.to_parquet(cache_path)
            logger.info(f"    Saved: {cache_path} ({len(df):,} bars × {df.shape[1]} features)")
            all_dfs.append(df)
            del trades
            gc.collect()

    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs).sort_index()


# ─────────────────────────────────────────────────────────
# 3. Feature Selection
# ─────────────────────────────────────────────────────────

def select_features(
    X_train: np.ndarray, y_train: np.ndarray,
    feature_names: List[str], top_n: int = 40,
) -> List[str]:
    """使用 permutation importance 选取 top N 特征。"""
    from sklearn.linear_model import Ridge
    from sklearn.inspection import permutation_importance
    from sklearn.preprocessing import StandardScaler

    logger.info(f"  Feature selection: fitting quick Ridge on {X_train.shape}...")
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_train)

    model = Ridge(alpha=1.0)
    model.fit(X_s, y_train)

    # 用训练集末尾 20% 做 importance
    val_n = max(1000, len(X_train) // 5)
    X_val = X_s[-val_n:]
    y_val = y_train[-val_n:]

    perm = permutation_importance(
        model, X_val, y_val,
        n_repeats=5, random_state=42, scoring="r2",
    )
    importances = perm.importances_mean
    top_idx = np.argsort(importances)[::-1][:top_n]
    selected = [feature_names[i] for i in top_idx]

    logger.info(f"  Top {top_n} features selected:")
    for rank, i in enumerate(top_idx[:15]):
        logger.info(f"    {rank+1:>3}. {feature_names[i]:<35} imp={importances[i]:.6f}")
    if top_n > 15:
        logger.info(f"    ... ({top_n - 15} more)")

    return selected


# ─────────────────────────────────────────────────────────
# 4. Models
# ─────────────────────────────────────────────────────────

def train_ridge(X_train, y_train, X_test, feature_names):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    model = Ridge(alpha=10.0)
    model.fit(X_tr, y_train)

    pred_train = model.predict(X_tr)
    pred_test = model.predict(X_te)

    logger.info(f"  [Ridge] Train R²={1 - np.sum((y_train-pred_train)**2)/np.sum((y_train-y_train.mean())**2):.4f}")
    return pred_test, model, scaler


def train_gbm(X_train, y_train, X_test, feature_names):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    # 1s 数据量大，用浅树 + 小学习率
    val_n = max(2000, len(X_train) // 5)
    X_fit = X_tr[:-val_n]
    y_fit = y_train[:-val_n]

    model = HistGradientBoostingRegressor(
        max_iter=500,
        max_depth=4,
        learning_rate=0.02,
        min_samples_leaf=100,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        random_state=42,
    )
    model.fit(X_fit, y_fit)

    pred_train = model.predict(X_tr)
    pred_test = model.predict(X_te)

    r2 = 1 - np.sum((y_train - pred_train) ** 2) / np.sum((y_train - y_train.mean()) ** 2)
    logger.info(f"  [GBM] Train R²={r2:.4f}, n_iter={model.n_iter_}")
    return pred_test, model, scaler


# ─────────────────────────────────────────────────────────
# 5. Backtest — Position-Based Model
# ─────────────────────────────────────────────────────────

def backtest_predictions(
    pred: np.ndarray,
    y_actual: np.ndarray,
    timestamps: pd.DatetimeIndex,
    cost_bps: float = 4.0,
    smoothing_window: int = 5,
    entry_quantile: float = 0.7,
    max_hold_bars: int = 30,
) -> pd.DataFrame:
    """
    持仓模型回测：

    1s 频率下单笔 return 极小，不可能让每笔预测都超过交易成本。
    正确做法：
      - 平滑预测信号（消除噪音）
      - 当信号强度超过历史分位数时入场
      - 持仓直到信号翻转或超时
      - 成本仅在持仓方向改变时收取（一次开+关 = 8bps）

    参数：
      smoothing_window: 对预测信号做 EMA 平滑（消除单 bar 噪音）
      entry_quantile: 信号绝对值超过此分位时入场
      max_hold_bars: 单笔最长持仓
    """
    n = len(pred)
    rt_cost = cost_bps * 2  # 8 bps round-trip

    # Step 1: 平滑信号
    pred_series = pd.Series(pred)
    smoothed = pred_series.ewm(span=smoothing_window, adjust=False).mean().values

    # Step 2: 计算入场阈值（用训练集的分位数，这里用前 30% 数据作为 warm-up）
    warm_up = max(100, n // 10)
    abs_signal = np.abs(smoothed[:warm_up])
    entry_thresh = np.quantile(abs_signal[abs_signal > 0], entry_quantile) if len(abs_signal[abs_signal > 0]) > 10 else 0.1

    # Step 3: 生成持仓信号
    # position: +1 (long), -1 (short), 0 (flat)
    position = np.zeros(n, dtype=int)
    trades = []
    cum_pnl = 0.0

    current_pos = 0
    entry_bar = 0
    trade_count = 0

    for i in range(warm_up, n):
        sig = smoothed[i]

        # 更新动态阈值（滚动分位数）
        if i > warm_up + 100 and i % 100 == 0:
            lookback = min(i, 3000)
            abs_recent = np.abs(smoothed[i - lookback:i])
            abs_recent = abs_recent[abs_recent > 0]
            if len(abs_recent) > 10:
                entry_thresh = np.quantile(abs_recent, entry_quantile)

        desired_pos = 0
        if sig > entry_thresh:
            desired_pos = 1
        elif sig < -entry_thresh:
            desired_pos = -1

        # 超时平仓
        if current_pos != 0 and (i - entry_bar) >= max_hold_bars:
            desired_pos = 0

        # 信号反转时立即翻转
        if current_pos > 0 and sig < 0:
            desired_pos = -1 if sig < -entry_thresh else 0
        elif current_pos < 0 and sig > 0:
            desired_pos = 1 if sig > entry_thresh else 0

        position[i] = desired_pos

        # 记录交易（持仓变化时）
        if desired_pos != current_pos:
            # 关闭旧仓位
            if current_pos != 0:
                gross_pnl = sum(y_actual[entry_bar + 1:i + 1]) * current_pos
                net_pnl = gross_pnl - rt_cost
                cum_pnl += net_pnl
                trades.append({
                    "entry_ts": timestamps[entry_bar] if entry_bar < len(timestamps) else None,
                    "exit_ts": timestamps[i] if i < len(timestamps) else None,
                    "direction": current_pos,
                    "bars_held": i - entry_bar,
                    "gross_pnl_bps": gross_pnl,
                    "net_pnl_bps": net_pnl,
                    "cum_pnl_bps": cum_pnl,
                    "signal_at_entry": smoothed[entry_bar],
                    "signal_at_exit": sig,
                })
                trade_count += 1

            # 开新仓
            if desired_pos != 0:
                entry_bar = i

            current_pos = desired_pos

    # 关闭最后的仓位
    if current_pos != 0:
        gross_pnl = sum(y_actual[entry_bar + 1:n]) * current_pos
        net_pnl = gross_pnl - rt_cost
        cum_pnl += net_pnl
        trades.append({
            "entry_ts": timestamps[entry_bar],
            "exit_ts": timestamps[-1],
            "direction": current_pos,
            "bars_held": n - 1 - entry_bar,
            "gross_pnl_bps": gross_pnl,
            "net_pnl_bps": net_pnl,
            "cum_pnl_bps": cum_pnl,
            "signal_at_entry": smoothed[entry_bar],
            "signal_at_exit": smoothed[-1],
        })

    return pd.DataFrame(trades)


def compute_metrics(trades_df: pd.DataFrame) -> Dict:
    """从交易记录计算回测指标。"""
    if trades_df.empty:
        return {
            "n_trades": 0, "total_pnl_bps": 0, "win_rate": 0,
            "sharpe": 0, "max_drawdown_bps": 0, "profit_factor": 0,
            "avg_bars_held": 0, "n_long": 0, "n_short": 0,
        }

    pnl = trades_df["net_pnl_bps"]
    gross = trades_df["gross_pnl_bps"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    # Sharpe (per-trade)
    sharpe = pnl.mean() / (pnl.std() + 1e-10) * np.sqrt(len(pnl))

    # Max drawdown
    cum = trades_df["cum_pnl_bps"]
    peak = cum.cummax()
    dd = cum - peak
    max_dd = dd.min()

    # Profit factor
    total_wins = wins.sum() if len(wins) > 0 else 0
    total_losses = losses.abs().sum() if len(losses) > 0 else 1e-10
    pf = total_wins / total_losses

    return {
        "n_trades": len(trades_df),
        "n_long": int((trades_df["direction"] == 1).sum()),
        "n_short": int((trades_df["direction"] == -1).sum()),
        "total_pnl_bps": float(pnl.sum()),
        "avg_pnl_bps": float(pnl.mean()),
        "win_rate": float(len(wins) / len(pnl)) if len(pnl) > 0 else 0,
        "sharpe": float(sharpe),
        "max_drawdown_bps": float(max_dd),
        "profit_factor": float(pf),
        "avg_bars_held": float(trades_df["bars_held"].mean()),
        "avg_gross_pnl": float(gross.mean()),
    }


# ─────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SOL/USDT")
    parser.add_argument("--model", default="all", choices=["ridge", "gbm", "all"])
    parser.add_argument("--cost-bps", type=float, default=4.0)
    parser.add_argument("--top-features", type=int, default=40)
    parser.add_argument("--smooth-window", type=int, default=5,
                        help="EMA smoothing window for prediction signal")
    parser.add_argument("--entry-quantile", type=float, default=0.7,
                        help="Signal quantile threshold for entry")
    parser.add_argument("--max-hold", type=int, default=30,
                        help="Max hold bars before forced exit")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("SOL 1s Momentum Prediction (Position-Based)")
    logger.info(f"  Symbol: {args.symbol}")
    logger.info(f"  Cost: {args.cost_bps} bps/leg ({args.cost_bps * 2} bps round-trip)")
    logger.info(f"  Signal smoothing: EMA({args.smooth_window})")
    logger.info(f"  Entry quantile: {args.entry_quantile}")
    logger.info(f"  Max hold: {args.max_hold} bars ({args.max_hold}s)")
    logger.info(f"  Train: {SOL_TRAIN_DATES}")
    logger.info(f"  Test:  {SOL_TEST_DATES}")
    logger.info("=" * 60)

    # ── Load ──
    logger.info("\n[1] Loading training features...")
    train_df = load_1s_features_for_dates(args.symbol, SOL_TRAIN_DATES)
    logger.info(f"  Train: {len(train_df):,} bars × {train_df.shape[1]} features")

    logger.info("\n[2] Loading test features...")
    test_df = load_1s_features_for_dates(args.symbol, SOL_TEST_DATES)
    logger.info(f"  Test:  {len(test_df):,} bars × {test_df.shape[1]} features")

    # ── Target ──
    logger.info("\n[3] Computing target: next_1s_return (bps)...")
    # 使用 close 的下一根 bar return
    for df in [train_df, test_df]:
        if "return_1" in df.columns:
            df["target"] = df["return_1"].shift(-1)  # next bar return
        else:
            # fallback
            df["target"] = df.iloc[:, 0].shift(-1)

    train_df.dropna(subset=["target"], inplace=True)
    test_df.dropna(subset=["target"], inplace=True)

    # ── Feature columns ──
    exclude_cols = {"target", "return_1", "log_return", "return_run_len"}
    feature_cols = [
        c for c in train_df.columns
        if c not in exclude_cols
        and train_df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]
        and train_df[c].std() > 1e-10
    ]
    logger.info(f"  Candidate features: {len(feature_cols)}")

    X_train_all = train_df[feature_cols].values
    y_train = train_df["target"].values
    X_test_all = test_df[feature_cols].values
    y_test = test_df["target"].values

    # ── Feature selection ──
    logger.info("\n[4] Feature selection (permutation importance)...")
    selected = select_features(
        X_train_all, y_train, feature_cols, top_n=args.top_features,
    )

    X_train = train_df[selected].values
    X_test = test_df[selected].values

    # ── Train & Backtest ──
    models_to_run = (
        ["ridge", "gbm"] if args.model == "all"
        else [args.model]
    )

    results = {}
    for model_name in models_to_run:
        logger.info(f"\n{'='*50}")
        logger.info(f"[5] Training: {model_name.upper()}")
        logger.info(f"{'='*50}")

        t0 = time.time()
        if model_name == "ridge":
            pred, model, scaler = train_ridge(X_train, y_train, X_test, selected)
        elif model_name == "gbm":
            pred, model, scaler = train_gbm(X_train, y_train, X_test, selected)
        else:
            continue
        train_time = time.time() - t0
        logger.info(f"  Training time: {train_time:.1f}s")

        # Prediction stats
        logger.info(f"  Pred stats: mean={pred.mean():.2f}, std={pred.std():.2f}, "
                     f"min={pred.min():.2f}, max={pred.max():.2f}")

        # Backtest — 多组参数扫描
        logger.info(f"\n  Backtesting (position-based, multiple configs)...")

        configs = [
            # (name, cost_bps, entry_q, smooth, max_hold)
            ("taker_q90_hold60",  args.cost_bps, 0.90, 5,  60),
            ("taker_q95_hold60",  args.cost_bps, 0.95, 5,  60),
            ("taker_q95_hold120", args.cost_bps, 0.95, 10, 120),
            ("maker_q90_hold60",  1.0,           0.90, 5,  60),
            ("maker_q95_hold60",  1.0,           0.95, 5,  60),
            ("maker_q95_hold120", 1.0,           0.95, 10, 120),
        ]

        best_sharpe = -999
        best_config = ""
        model_results = {}

        for cfg_name, cost, eq, sw, mh in configs:
            trades = backtest_predictions(
                pred, y_test, test_df.index,
                cost_bps=cost,
                smoothing_window=sw,
                entry_quantile=eq,
                max_hold_bars=mh,
            )
            metrics = compute_metrics(trades)
            model_results[cfg_name] = metrics

            logger.info(f"\n    {cfg_name} (cost={cost*2}bps RT, q={eq}, hold≤{mh}s):")
            logger.info(f"      Trades={metrics['n_trades']:>5}  Gross={metrics.get('avg_gross_pnl',0):>+6.2f}  "
                         f"Net={metrics['avg_pnl_bps']:>+6.2f}  WR={metrics['win_rate']:.1%}  "
                         f"Sharpe={metrics['sharpe']:>+6.2f}  PF={metrics['profit_factor']:.2f}")

            if metrics['sharpe'] > best_sharpe:
                best_sharpe = metrics['sharpe']
                best_config = cfg_name

        results[model_name] = model_results
        logger.info(f"\n  Best config for {model_name}: {best_config} (Sharpe={best_sharpe:.2f})")

        # Save best config trades
        best_cost, best_eq, best_sw, best_mh = [(c,e,s,m) for n,c,e,s,m in configs if n == best_config][0]
        trades = backtest_predictions(
            pred, y_test, test_df.index,
            cost_bps=best_cost, smoothing_window=best_sw,
            entry_quantile=best_eq, max_hold_bars=best_mh,
        )
        save_dir = DataPaths.backtest_dir(f"sol_1s_momentum/{model_name}")
        if not trades.empty:
            trades.to_csv(save_dir / "trades.csv", index=False)
            logger.info(f"  Saved trades: {save_dir / 'trades.csv'}")

    # ── Summary ──
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")

    summary = {
        "config": {
            "symbol": args.symbol,
            "freq": "1s",
            "signal_type": "momentum (positive lag-1 ACF = +0.035)",
            "backtest_model": "position-based (cost only at position change)",
            "cost_bps_per_leg": args.cost_bps,
            "cost_bps_roundtrip": args.cost_bps * 2,
            "signal_smoothing": f"EMA({args.smooth_window})",
            "entry_quantile": args.entry_quantile,
            "max_hold_bars": args.max_hold,
            "feature_windows": FEATURE_WINDOWS,
            "n_features_selected": len(selected),
            "selected_features": selected[:20],
            "train_dates": SOL_TRAIN_DATES,
            "test_dates": SOL_TEST_DATES,
            "train_bars": len(train_df),
            "test_bars": len(test_df),
        },
        "results": results,
    }

    for model_name, cfg_results in results.items():
        logger.info(f"\n  {model_name.upper()}:")
        for cfg_name, m in cfg_results.items():
            pnl_str = f"{m['total_pnl_bps']:>+8.0f}" if m['n_trades'] > 0 else "       0"
            logger.info(
                f"    {cfg_name:<25} | N={m['n_trades']:>5} | "
                f"PnL={pnl_str} | WR={m['win_rate']:.1%} | "
                f"SR={m['sharpe']:>+6.2f} | PF={m['profit_factor']:.2f} | "
                f"Hold={m['avg_bars_held']:.0f}s"
            )

    # Save summary
    summary_path = DataPaths.backtest_dir("sol_1s_momentum") / "summary.json"
    with open(summary_path, "w") as fp:
        json.dump(summary, fp, indent=2, default=str)
    logger.info(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
