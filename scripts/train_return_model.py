#!/usr/bin/env python3
"""
收益率回归预测：预测下一分钟收益率 → 按交易成本阈值过滤信号。

思路：
  1. 加载已缓存特征 (parquet)
  2. y = next_1min_return = (close[t+1] - close[t]) / close[t]
  3. 特征筛选 (permutation importance → top N)
  4. 训练 Ridge / GBM Regressor / LSTM Regressor
  5. 回测：|pred| > trading_cost → 交易，方向由 pred 符号决定
  6. 报告 Sharpe / PF / drawdown 等

Usage:
    python scripts/train_return_model.py
    python scripts/train_return_model.py --model all --top-features 30
    python scripts/train_return_model.py --cost-bps 8 --model ridge
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utilities.binance_loader import load_agg_trades, resample_trades_to_ohlcv
from utilities.paths import DataPaths
from backtest.metrics import compute_backtest_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# 默认日期配置
# ─────────────────────────────────────────────────────────

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
# 数据加载
# ─────────────────────────────────────────────────────────

def load_features_for_dates(symbol: str, dates: List[str]) -> pd.DataFrame:
    """加载多天已缓存的 parquet 特征。"""
    from scripts.compute_features import compute_all_features_for_day

    dfs = []
    for d_str in dates:
        dt = date.fromisoformat(d_str)
        path = DataPaths.features(symbol, "bar_features", dt)
        if path.exists():
            logger.info(f"  [Feature] Cached: {d_str}")
            dfs.append(pd.read_parquet(path))
        else:
            logger.info(f"  [Feature] Computing: {d_str}")
            df = compute_all_features_for_day(symbol, dt)
            if not df.empty:
                df.to_parquet(path)
            dfs.append(df)
    return pd.concat(dfs).sort_index() if dfs else pd.DataFrame()


def load_ohlcv(symbol: str, dates: List[str], freq: str = "1min") -> pd.DataFrame:
    """从缓存 trades 构建 OHLCV。"""
    dfs = []
    for d_str in dates:
        dt = date.fromisoformat(d_str)
        trades = load_agg_trades(symbol, dt, dt + timedelta(days=1))
        if not trades.empty:
            dfs.append(resample_trades_to_ohlcv(trades, freq))
    return pd.concat(dfs).sort_index() if dfs else pd.DataFrame()


def add_return_target(features: pd.DataFrame, ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    添加 y = 下一分钟收益率 (bps)。
    return_1m = (close[t+1] - close[t]) / close[t] * 10000  (bps)
    """
    features = features.copy()
    common = features.index.intersection(ohlcv.index)
    features = features.loc[common]

    close = ohlcv.loc[common, "close"].astype(float)
    ret = close.pct_change().shift(-1) * 10000  # bps
    features["return_1m"] = ret
    # 去掉最后一行 (NaN)
    features = features.dropna(subset=["return_1m"])
    return features


# ─────────────────────────────────────────────────────────
# 特征列选择
# ─────────────────────────────────────────────────────────

EXCLUDE_PREFIXES = (
    "tb_", "target_", "regime", "timestamp", "label_", "pred", "proba", "return_1m",
)
ABS_PRICE_COLS = {"micro_vwap", "vwap", "close", "open", "high", "low"}


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """从 DataFrame 中选出有效特征列。"""
    return [
        c for c in df.columns
        if not c.startswith(EXCLUDE_PREFIXES)
        and c not in ABS_PRICE_COLS
        and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]
    ]


# ─────────────────────────────────────────────────────────
# 模型
# ─────────────────────────────────────────────────────────

def train_ridge(X_tr, y_tr, X_val, y_val, feat_names):
    """Ridge 线性回归。"""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    model = Ridge(alpha=1.0)
    model.fit(X_tr_s, y_tr)

    tr_pred = model.predict(X_tr_s)
    val_pred = model.predict(X_val_s)
    tr_mse = ((tr_pred - y_tr) ** 2).mean()
    val_mse = ((val_pred - y_val) ** 2).mean()
    tr_corr = np.corrcoef(tr_pred, y_tr)[0, 1] if len(y_tr) > 1 else 0
    val_corr = np.corrcoef(val_pred, y_val)[0, 1] if len(y_val) > 1 else 0

    # 特征重要性 = |系数|
    coefs = np.abs(model.coef_)
    ranked = sorted(zip(feat_names, coefs), key=lambda x: x[1], reverse=True)

    log = {
        "model_type": "Ridge",
        "alpha": 1.0,
        "n_features": len(feat_names),
        "train_mse": round(float(tr_mse), 6),
        "val_mse": round(float(val_mse), 6),
        "train_corr": round(float(tr_corr), 4),
        "val_corr": round(float(val_corr), 4),
        "top_features": [(n, round(float(v), 6)) for n, v in ranked[:20]],
    }

    logger.info(f"  [Ridge] Train MSE={tr_mse:.4f}, Val MSE={val_mse:.4f}")
    logger.info(f"  [Ridge] Train corr={tr_corr:.4f}, Val corr={val_corr:.4f}")

    return model, scaler, log, ranked


def train_gbm_regressor(X_tr, y_tr, X_val, y_val, feat_names):
    """HistGradientBoostingRegressor（浅层，强正则化）。"""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.inspection import permutation_importance

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    model = HistGradientBoostingRegressor(
        max_iter=100, max_depth=2, learning_rate=0.05,
        min_samples_leaf=50, max_features=0.5,
        l2_regularization=5.0,
        early_stopping=True, n_iter_no_change=15,
        validation_fraction=0.15, random_state=42,
    )
    model.fit(X_tr_s, y_tr)

    tr_pred = model.predict(X_tr_s)
    val_pred = model.predict(X_val_s)
    tr_mse = ((tr_pred - y_tr) ** 2).mean()
    val_mse = ((val_pred - y_val) ** 2).mean()
    tr_corr = np.corrcoef(tr_pred, y_tr)[0, 1] if len(y_tr) > 1 else 0
    val_corr = np.corrcoef(val_pred, y_val)[0, 1] if len(y_val) > 1 else 0

    # Permutation importance
    perm = permutation_importance(
        model, X_val_s, y_val, n_repeats=5, random_state=42, n_jobs=-1,
    )
    ranked = sorted(
        zip(feat_names, perm.importances_mean),
        key=lambda x: x[1], reverse=True,
    )

    log = {
        "model_type": "HistGradientBoostingRegressor",
        "n_iter_actual": model.n_iter_,
        "n_features": len(feat_names),
        "train_mse": round(float(tr_mse), 6),
        "val_mse": round(float(val_mse), 6),
        "train_corr": round(float(tr_corr), 4),
        "val_corr": round(float(val_corr), 4),
        "top_features": [(n, round(float(v), 6)) for n, v in ranked[:20]],
    }

    logger.info(f"  [GBM] iters={model.n_iter_}, Train MSE={tr_mse:.4f}, Val MSE={val_mse:.4f}")
    logger.info(f"  [GBM] Train corr={tr_corr:.4f}, Val corr={val_corr:.4f}")

    return model, scaler, log, ranked


def train_lstm_regressor(X_tr, y_tr, X_val, y_val, feat_names,
                         lookback=30, hidden=32, layers=2, dropout=0.2,
                         lr=1e-3, batch_size=128, max_epochs=60, patience=10):
    """LSTM 回归模型（PyTorch）。"""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        logger.warning("  [LSTM] PyTorch not available, skipping")
        return None, None, None, None

    # 构建序列
    def make_sequences(X, y, lb):
        Xs, ys = [], []
        for i in range(lb, len(X)):
            Xs.append(X[i - lb:i])
            ys.append(y[i])
        return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    X_tr_seq, y_tr_seq = make_sequences(X_tr, y_tr, lookback)
    X_val_seq, y_val_seq = make_sequences(X_val, y_val, lookback)

    if len(X_tr_seq) < 50 or len(X_val_seq) < 10:
        logger.warning("  [LSTM] Not enough sequences, skipping")
        return None, None, None, None

    n_features = X_tr_seq.shape[2]

    # Normalize per-feature
    mean = X_tr_seq.reshape(-1, n_features).mean(axis=0)
    std = X_tr_seq.reshape(-1, n_features).std(axis=0) + 1e-8
    X_tr_seq = (X_tr_seq - mean) / std
    X_val_seq = (X_val_seq - mean) / std

    # Normalize y
    y_mean = y_tr_seq.mean()
    y_std = y_tr_seq.std() + 1e-8
    y_tr_norm = (y_tr_seq - y_mean) / y_std
    y_val_norm = (y_val_seq - y_mean) / y_std

    # Model
    class LSTMRegressor(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, dropout):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size, hidden_size, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0,
            )
            self.bn = nn.BatchNorm1d(hidden_size)
            self.head = nn.Sequential(
                nn.Linear(hidden_size, 16),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(16, 1),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            last = self.bn(last)
            return self.head(last).squeeze(-1)

    device = torch.device("cpu")
    model = LSTMRegressor(n_features, hidden, layers, dropout).to(device)

    train_ds = TensorDataset(
        torch.FloatTensor(X_tr_seq), torch.FloatTensor(y_tr_norm),
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_v = torch.FloatTensor(X_val_seq).to(device)
    y_v = torch.FloatTensor(y_val_norm).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    best_val_loss = float("inf")
    best_state = None
    wait = 0
    epoch_logs = []

    logger.info(f"  [LSTM] Sequences: train={len(X_tr_seq)}, val={len(X_val_seq)}")
    logger.info(f"  [LSTM] lookback={lookback}, hidden={hidden}, layers={layers}")
    logger.info("  ┌────────┬──────────┬──────────┬──────────┬──────────┬────────┐")
    logger.info("  │ Epoch  │ Tr Loss  │ Val Loss │ Tr Corr  │ Val Corr │   LR   │")
    logger.info("  ├────────┼──────────┼──────────┼──────────┼──────────┼────────┤")

    t0 = time.time()
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / n_batches

        model.eval()
        with torch.no_grad():
            val_pred = model(X_v)
            val_loss = criterion(val_pred, y_v).item()

            # 反归一化计算 corr
            tr_all_pred = model(torch.FloatTensor(X_tr_seq)).cpu().numpy() * y_std + y_mean
            vl_pred_np = val_pred.cpu().numpy() * y_std + y_mean
            tr_corr = np.corrcoef(tr_all_pred, y_tr_seq)[0, 1] if len(y_tr_seq) > 1 else 0
            vl_corr = np.corrcoef(vl_pred_np, y_val_seq)[0, 1] if len(y_val_seq) > 1 else 0

        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        logger.info(
            f"  │  {epoch+1:>3}   │ {train_loss:>8.5f} │ {val_loss:>8.5f} │"
            f"  {tr_corr:>+.4f} │  {vl_corr:>+.4f} │{cur_lr:>7.5f}│"
        )
        epoch_logs.append({
            "epoch": epoch + 1, "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "train_corr": round(float(tr_corr), 4),
            "val_corr": round(float(vl_corr), 4), "lr": cur_lr,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info(f"  └─── Early stopped at epoch {epoch+1} ──────────────────────┘")
                break
    else:
        logger.info(f"  └─── Completed {max_epochs} epochs ───────────────────────┘")

    elapsed = time.time() - t0
    if best_state:
        model.load_state_dict(best_state)

    # Final val corr
    model.eval()
    with torch.no_grad():
        vp = model(X_v).cpu().numpy() * y_std + y_mean
    final_corr = np.corrcoef(vp, y_val_seq)[0, 1] if len(y_val_seq) > 1 else 0

    log = {
        "model_type": "LSTM_Regressor",
        "lookback": lookback, "hidden": hidden, "layers": layers,
        "n_features": n_features,
        "best_epoch": len(epoch_logs) - wait,
        "total_epochs": len(epoch_logs),
        "best_val_loss": round(best_val_loss, 5),
        "final_val_corr": round(float(final_corr), 4),
        "training_time_s": round(elapsed, 1),
        "epoch_history": epoch_logs,
    }
    logger.info(f"  [LSTM] Best val loss: {best_val_loss:.5f}, final corr: {final_corr:.4f} ({elapsed:.1f}s)")

    # 封装预测函数需要的参数
    predictor = {
        "model": model, "mean": mean, "std": std,
        "y_mean": y_mean, "y_std": y_std, "lookback": lookback,
    }
    return predictor, None, log, None


def predict_lstm(predictor, X_raw):
    """用 LSTM predictor 做序列预测。"""
    import torch
    lb = predictor["lookback"]
    mean, std = predictor["mean"], predictor["std"]
    y_mean, y_std = predictor["y_mean"], predictor["y_std"]
    model = predictor["model"]

    # 构建序列
    seqs = []
    for i in range(lb, len(X_raw)):
        seqs.append(X_raw[i - lb:i])
    if not seqs:
        return np.array([]), np.arange(lb, len(X_raw))
    X_seq = np.array(seqs, dtype=np.float32)
    X_seq = (X_seq - mean) / (std + 1e-8)

    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_seq)).cpu().numpy() * y_std + y_mean
    indices = np.arange(lb, len(X_raw))  # 对应 feature_df 中的行号
    return preds, indices


# ─────────────────────────────────────────────────────────
# 回测
# ─────────────────────────────────────────────────────────

def backtest_predictions(
    pred_returns: np.ndarray,
    actual_returns: np.ndarray,
    cost_bps: float = 8.0,
    min_signal_bps: float = None,
    cooldown: int = 1,
) -> pd.DataFrame:
    """
    回归预测 → 交易回测。

    Args:
        pred_returns: 预测的下一分钟收益率 (bps)
        actual_returns: 实际的下一分钟收益率 (bps)
        cost_bps: 单边交易成本 (bps), 往返 = 2 * cost_bps
        min_signal_bps: 最小信号阈值 (bps), None 则使用 cost * 2
        cooldown: 交易冷却 bar 数

    Returns:
        DataFrame of trades
    """
    if min_signal_bps is None:
        min_signal_bps = cost_bps * 2  # 默认：需要覆盖往返成本

    round_trip_cost = cost_bps * 2  # 往返成本

    trades = []
    last_trade = -cooldown

    for i in range(len(pred_returns)):
        if i - last_trade < cooldown:
            continue

        pred = pred_returns[i]
        actual = actual_returns[i]

        if abs(pred) < min_signal_bps:
            continue

        direction = "long" if pred > 0 else "short"
        sign = 1 if pred > 0 else -1
        gross_ret = sign * actual  # bps
        net_ret = gross_ret - round_trip_cost  # 扣除往返成本

        trades.append({
            "bar_idx": i,
            "direction": direction,
            "pred_return_bps": round(float(pred), 2),
            "actual_return_bps": round(float(actual), 2),
            "gross_return_bps": round(float(gross_ret), 2),
            "net_return_bps": round(float(net_ret), 2),
            "return_pct": round(float(net_ret / 100), 4),  # bps → %
        })
        last_trade = i

    df = pd.DataFrame(trades)
    if not df.empty:
        df["is_win"] = df["net_return_bps"] > 0
    return df


def backtest_metrics_from_trades(trades_df: pd.DataFrame) -> dict:
    """从交易 df 计算回测统计。"""
    if trades_df.empty or "net_return_bps" not in trades_df.columns:
        return {
            "n_signals": 0, "n_long": 0, "n_short": 0,
            "win_rate": 0, "total_return_bps": 0, "total_return_pct": 0,
            "avg_return_bps": 0, "avg_win_bps": 0, "avg_loss_bps": 0,
            "profit_factor": 0, "sharpe": 0, "sortino": 0,
            "max_drawdown_bps": 0, "median_return_bps": 0, "test_corr": 0,
        }

    r = trades_df["net_return_bps"]
    n = len(r)
    wins = r[r > 0]
    losses = r[r <= 0]

    cum = r.cumsum()
    peak = cum.cummax()
    dd = cum - peak

    ret_std = r.std() + 1e-10
    down_std = r[r < 0].std() if len(r[r < 0]) > 1 else 1e-10

    return {
        "n_signals": n,
        "n_long": int((trades_df["direction"] == "long").sum()),
        "n_short": int((trades_df["direction"] == "short").sum()),
        "win_rate": round((r > 0).mean() * 100, 2),
        "total_return_bps": round(float(r.sum()), 2),
        "total_return_pct": round(float(r.sum() / 100), 4),
        "avg_return_bps": round(float(r.mean()), 2),
        "avg_win_bps": round(float(wins.mean()), 2) if len(wins) > 0 else 0,
        "avg_loss_bps": round(float(losses.mean()), 2) if len(losses) > 0 else 0,
        "profit_factor": round(abs(float(wins.sum() / (losses.sum() + 1e-10))), 2),
        "sharpe": round(float(r.mean() / ret_std * np.sqrt(n)), 3),
        "sortino": round(float(r.mean() / (down_std + 1e-10) * np.sqrt(n)), 3),
        "max_drawdown_bps": round(float(dd.min()), 2),
        "median_return_bps": round(float(r.median()), 2),
    }


# ─────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Return regression model")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--model", default="all",
                        choices=["ridge", "gbm", "lstm", "all"])
    parser.add_argument("--top-features", type=int, default=30)
    parser.add_argument("--cost-bps", type=float, default=4.0,
                        help="Single-leg trading cost in bps (default 4 = 0.04%%)")
    parser.add_argument("--signal-mult", type=float, default=1.0,
                        help="Signal threshold = round_trip_cost * signal_mult")
    parser.add_argument("--lookback", type=int, default=30, help="LSTM lookback")
    parser.add_argument("--freq", default="1min")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    sym_slug = args.symbol.replace("/", "_")
    results_dir = DataPaths.data / "backtest_results" / "return" / sym_slug
    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir = DataPaths.data / "training_logs" / "return" / sym_slug
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            regime_dates = json.load(f)
    else:
        regime_dates = DEFAULT_REGIME_DATES

    round_trip = args.cost_bps * 2
    min_signal = round_trip * args.signal_mult  # 至少覆盖往返成本

    logger.info("═══════════════════════════════════════════════")
    logger.info("  Return Regression Model")
    logger.info(f"  Symbol: {args.symbol}")
    logger.info(f"  Cost: {args.cost_bps} bps/leg, round-trip: {round_trip} bps")
    logger.info(f"  Signal threshold: |pred| > {min_signal} bps")
    logger.info(f"  Model: {args.model}")
    logger.info("═══════════════════════════════════════════════")

    # ── 1. Load features + target ──
    logger.info("\n[1/5] Loading features & computing target...")

    all_train_dates = []
    for dates in regime_dates["train"].values():
        all_train_dates.extend(dates)
    all_test_dates = {}
    for regime, dates in regime_dates["test"].items():
        all_test_dates[regime] = dates

    # Train
    logger.info("  Train dates:")
    train_feat = load_features_for_dates(args.symbol, all_train_dates)
    train_ohlcv = load_ohlcv(args.symbol, all_train_dates, args.freq)
    train_df = add_return_target(train_feat, train_ohlcv)
    logger.info(f"  Train: {len(train_df)} bars, {train_df.shape[1]} cols")

    # Target stats
    y_stats = train_df["return_1m"].describe()
    logger.info(f"  Return stats (bps): mean={y_stats['mean']:.2f}, "
                 f"std={y_stats['std']:.2f}, "
                 f"min={y_stats['min']:.2f}, max={y_stats['max']:.2f}")

    # Test (per regime)
    test_dfs = {}
    for regime, dates in all_test_dates.items():
        logger.info(f"  Test {regime}:")
        feat = load_features_for_dates(args.symbol, dates)
        ohlcv = load_ohlcv(args.symbol, dates, args.freq)
        test_dfs[regime] = add_return_target(feat, ohlcv)
        logger.info(f"    {regime}: {len(test_dfs[regime])} bars")

    # ── 2. Feature selection ──
    logger.info("\n[2/5] Feature selection...")

    feature_cols = get_feature_cols(train_df)
    logger.info(f"  Available: {len(feature_cols)} features")

    y_train_all = train_df["return_1m"].values.astype(np.float32)
    X_train_all = train_df[feature_cols].fillna(0).values.astype(np.float32)

    # Time-series split
    val_size = max(int(len(y_train_all) * 0.2), 100)
    X_tr = X_train_all[:-val_size]
    X_val = X_train_all[-val_size:]
    y_tr = y_train_all[:-val_size]
    y_val = y_train_all[-val_size:]

    # Quick GBM for feature selection
    logger.info("  Quick GBM for feature ranking...")
    _, _, _, full_ranking = train_gbm_regressor(X_tr, y_tr, X_val, y_val, feature_cols)

    # Select top N
    selected = [name for name, imp in full_ranking if imp > 0][:args.top_features]
    if len(selected) < 10:
        # 如果有效特征太少，取 top N 不管正负
        selected = [name for name, _ in full_ranking[:args.top_features]]
    selected_idx = [feature_cols.index(c) for c in selected if c in feature_cols]
    selected_cols = [feature_cols[i] for i in selected_idx]

    logger.info(f"  Selected {len(selected_cols)} features:")
    for n, v in full_ranking[:10]:
        logger.info(f"    {n}: {v:.4f}")

    # Rebuild arrays with selected features
    X_tr_sel = X_tr[:, selected_idx]
    X_val_sel = X_val[:, selected_idx]

    # ── 3. Train models ──
    logger.info("\n[3/5] Training models...")

    models_results = {}

    # Ridge
    if args.model in ("ridge", "all"):
        logger.info("\n  ═══ Ridge ═══")
        ridge_model, ridge_scaler, ridge_log, _ = train_ridge(
            X_tr_sel, y_tr, X_val_sel, y_val, selected_cols,
        )
        models_results["ridge"] = {
            "model": ridge_model, "scaler": ridge_scaler,
            "log": ridge_log, "type": "sklearn",
        }

    # GBM
    if args.model in ("gbm", "all"):
        logger.info("\n  ═══ GBM Regressor ═══")
        gbm_model, gbm_scaler, gbm_log, _ = train_gbm_regressor(
            X_tr_sel, y_tr, X_val_sel, y_val, selected_cols,
        )
        models_results["gbm"] = {
            "model": gbm_model, "scaler": gbm_scaler,
            "log": gbm_log, "type": "sklearn",
        }

    # LSTM
    if args.model in ("lstm", "all"):
        logger.info("\n  ═══ LSTM Regressor ═══")
        lstm_pred, _, lstm_log, _ = train_lstm_regressor(
            X_tr_sel, y_tr, X_val_sel, y_val, selected_cols,
            lookback=args.lookback, hidden=32, layers=2, dropout=0.2,
            lr=1e-3, batch_size=128, max_epochs=60, patience=10,
        )
        if lstm_pred is not None:
            models_results["lstm"] = {
                "predictor": lstm_pred, "log": lstm_log, "type": "lstm",
            }

    # ── 4. Backtest ──
    logger.info("\n[4/5] Backtesting...")

    backtest_all = {}
    for regime, tdf in test_dfs.items():
        logger.info(f"\n  ── {regime.upper()} regime ──")
        y_actual = tdf["return_1m"].values.astype(np.float32)
        X_test = tdf[selected_cols].fillna(0).values.astype(np.float32)

        regime_results = {}
        for mname, minfo in models_results.items():
            if minfo["type"] == "sklearn":
                X_scaled = minfo["scaler"].transform(X_test)
                pred = minfo["model"].predict(X_scaled)
                actual = y_actual
            elif minfo["type"] == "lstm":
                pred, idx = predict_lstm(minfo["predictor"], X_test)
                if len(pred) == 0:
                    logger.info(f"    {mname}: no predictions (too few bars)")
                    continue
                actual = y_actual[idx]
            else:
                continue

            # Correlation
            if len(pred) > 1:
                corr = np.corrcoef(pred, actual)[0, 1]
            else:
                corr = 0

            # Backtest
            bt = backtest_predictions(
                pred, actual,
                cost_bps=args.cost_bps,
                min_signal_bps=min_signal,
                cooldown=1,
            )
            metrics = backtest_metrics_from_trades(bt)
            metrics["test_corr"] = round(float(corr), 4)

            logger.info(
                f"    [{mname:>5}] corr={corr:+.4f}, "
                f"signals={metrics['n_signals']} "
                f"(L={metrics.get('n_long', 0)}/S={metrics.get('n_short', 0)}), "
                f"win={metrics['win_rate']}%, "
                f"sharpe={metrics['sharpe']}, "
                f"PF={metrics['profit_factor']}, "
                f"ret={metrics['total_return_bps']}bps"
            )

            regime_results[mname] = metrics

            # Save trades
            if not bt.empty:
                bt["regime"] = regime
                bt_path = results_dir / f"trades_{regime}_{mname}.csv"
                bt.to_csv(bt_path, index=False)

        backtest_all[regime] = regime_results

    # Overall across regimes
    logger.info("\n  ── Overall ──")
    for mname in models_results:
        all_trades = []
        for regime in backtest_all:
            bt_path = results_dir / f"trades_{regime}_{mname}.csv"
            if bt_path.exists():
                all_trades.append(pd.read_csv(bt_path))
        if all_trades:
            combined = pd.concat(all_trades, ignore_index=True)
            combined["return_pct"] = combined["net_return_bps"] / 100
            combined["is_win"] = combined["net_return_bps"] > 0
            overall = backtest_metrics_from_trades(combined)
            backtest_all.setdefault("overall", {})[mname] = overall
            logger.info(
                f"    [{mname:>5}] signals={overall['n_signals']}, "
                f"win={overall['win_rate']}%, "
                f"sharpe={overall['sharpe']}, "
                f"PF={overall['profit_factor']}, "
                f"ret={overall['total_return_bps']}bps, "
                f"dd={overall['max_drawdown_bps']}bps"
            )

    # ── 5. Save ──
    logger.info("\n[5/5] Saving results...")

    ts_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "timestamp": ts_str,
        "symbol": args.symbol,
        "freq": args.freq,
        "target": "return_1m (bps)",
        "cost_bps": args.cost_bps,
        "round_trip_bps": round_trip,
        "signal_threshold_bps": min_signal,
        "regime_dates": regime_dates,
        "n_features_selected": len(selected_cols),
        "selected_features": selected_cols,
        "feature_ranking": [(n, round(float(v), 6)) for n, v in full_ranking[:30]],
        "models": {
            k: v.get("log", v.get("predictor", {}).get("log", {}))
            for k, v in models_results.items()
            if "log" in v
        },
        "backtest": backtest_all,
    }

    path = results_dir / f"summary_{ts_str}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\nResults saved: {path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
