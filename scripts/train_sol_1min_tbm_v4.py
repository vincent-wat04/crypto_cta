#!/usr/bin/env python3
"""
SOL 1min TBM Explorer — V4

聚焦 1min Triple Barrier Method 策略的深度探索。

V4 核心改进:
  1. TBM 栏栅参数网格搜索: (tp_mult, sl_mult) 多组组合
     → 通过放宽/收紧 barrier 调整趋势 vs 震荡标签比例
  2. 扩展训练数据: Jan 22 → Feb 3 (13 天训练), Feb 4-6 (3 天测试)
  3. 精细化 tick-level 微观特征: 从原始 aggTrades 提取秒内统计
     → window 不变 [3,5,10,20]s，但每个 1s bar 信息更丰富
  4. 高效 TBM 计算: 仅在 subsample 位置计算标签 (60× 加速)
  5. 全面的特征报告 + 自动化结果对比

Usage:
    python scripts/train_sol_1min_tbm_v4.py                 # 全部 grid
    python scripts/train_sol_1min_tbm_v4.py --quick          # 仅跑 4 组快速验证
    python scripts/train_sol_1min_tbm_v4.py --fetch-extra    # 先拉取额外训练数据
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
# 扩展训练集: 13 天 (原 V3 只有 7 天)
EXTENDED_TRAIN_DATES = [
    "2026-01-22", "2026-01-23", "2026-01-24", "2026-01-25",
    "2026-01-26", "2026-01-27",
    "2026-01-28", "2026-01-29", "2026-01-30", "2026-01-31",
    "2026-02-01", "2026-02-02", "2026-02-03",
]

# 原始 V3 训练集 (fallback)
ORIGINAL_TRAIN_DATES = [
    "2026-01-28", "2026-01-29", "2026-01-30", "2026-01-31",
    "2026-02-01", "2026-02-02", "2026-02-03",
]

TEST_DATES = ["2026-02-04", "2026-02-05", "2026-02-06"]

FEATURE_WINDOWS = [3, 5, 10, 20]  # 秒 — 物理窗口不变
INTERVAL = 60  # 1min 固定

# TBM 栏栅 grid search
TBM_GRID_FULL = [
    # symmetric — 主要维度：barrier 松紧
    (0.5, 0.5),    # 很松 → 更多 ±1 信号
    (0.75, 0.75),  # 松
    (1.0, 1.0),    # 中
    (1.25, 1.25),  # 略紧
    (1.5, 1.5),    # V3 默认 (紧)
    (2.0, 2.0),    # 很紧
    # asymmetric — 不对称策略
    (1.5, 0.75),   # 让利润跑 + 快速止损
    (0.75, 1.5),   # 快速止盈 + 给亏损更多空间
    (1.0, 0.5),    # 中等止盈 + 紧止损
    (0.5, 1.0),    # 宽止盈 + 中等止损
]

TBM_GRID_QUICK = [
    (0.5, 0.5),
    (0.75, 0.75),
    (1.0, 1.0),
    (1.5, 1.5),
]


# ═══════════════════════════════════════════════════════════
# 1. Tick-Level Micro Features (NEW in V4)
# ═══════════════════════════════════════════════════════════

def compute_tick_features(trades: pd.DataFrame) -> pd.DataFrame:
    """
    从原始 aggTrades 提取每秒内的 tick-level 微观统计。

    这些特征捕捉了 1s OHLCV bar 无法表达的秒内微观结构:
      - n_unique_prices:   每秒交易了多少个不同价位 (spread/深度碎片化)
      - max_single_trade:  最大单笔成交量 (鲸鱼检测)
      - avg_trade_size:    平均成交量
      - trade_size_cv:     成交量变异系数 (交易异质性)
      - vwap_close_dev:    VWAP 与 close 偏差 bps (执行压力方向)
      - buy_avg_size:      买方平均成交 size
      - sell_avg_size:     卖方平均成交 size
      - size_imbalance:    买卖方 size 不对称度
      - large_trade_pct:   大单成交量占比 (>2x均值)
      - range_per_trade:   价格范围 / 交易笔数 (每笔对价格的影响)
      - n_agg_trades:      aggTrade 笔数 (不同于 n_trades_in_agg 的总和)
    """
    t = trades.copy()
    t["second"] = t["timestamp"].dt.floor("1s")

    g = t.groupby("second")
    tf = pd.DataFrame(index=sorted(g.groups.keys()))
    tf.index.name = "timestamp"

    # ── Distinct price levels ──
    tf["tick_n_prices"] = g["price"].nunique()

    # ── Trade size stats ──
    tf["tick_max_trade"] = g["amount"].max()
    tf["tick_avg_trade"] = g["amount"].mean()
    qty_std = g["amount"].std().fillna(0)
    qty_mean = g["amount"].mean()
    tf["tick_size_cv"] = qty_std / (qty_mean + 1e-10)

    # ── VWAP deviation (bps) ──
    t["pv"] = t["price"] * t["amount"]
    pv_sum = t.groupby("second")["pv"].sum()
    v_sum = g["amount"].sum()
    vwap = pv_sum / (v_sum + 1e-10)
    last_price = g["price"].last()
    tf["tick_vwap_dev"] = (vwap - last_price) / (last_price + 1e-10) * 10000

    # ── Buy vs Sell avg size ──
    is_buy = t["side"] == "buy"
    buy_sum = t[is_buy].groupby("second")["amount"].sum()
    buy_cnt = t[is_buy].groupby("second")["amount"].count()
    sell_sum = t[~is_buy].groupby("second")["amount"].sum()
    sell_cnt = t[~is_buy].groupby("second")["amount"].count()

    buy_avg = (buy_sum / (buy_cnt + 1e-10)).reindex(tf.index).fillna(0)
    sell_avg = (sell_sum / (sell_cnt + 1e-10)).reindex(tf.index).fillna(0)
    tf["tick_buy_avg_size"] = buy_avg
    tf["tick_sell_avg_size"] = sell_avg
    tf["tick_size_imbalance"] = (buy_avg - sell_avg) / (buy_avg + sell_avg + 1e-10)

    # ── Large trade ratio (volume from trades > 2× mean) ──
    overall_mean = t.groupby("second")["amount"].transform("mean")
    t["_is_large"] = t["amount"] > 2 * overall_mean
    large_vol = t[t["_is_large"]].groupby("second")["amount"].sum()
    total_vol = g["amount"].sum()
    tf["tick_large_pct"] = (large_vol / (total_vol + 1e-10)).reindex(tf.index).fillna(0)

    # ── Range per trade (bps) ──
    p_range = g["price"].max() - g["price"].min()
    p_count = g["price"].count()
    p_mean = g["price"].mean()
    tf["tick_range_per_trade"] = p_range / (p_count + 1e-10) / (p_mean + 1e-10) * 10000

    # ── Number of aggTrades per second ──
    tf["tick_n_agg"] = g["price"].count()

    tf = tf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return tf


# ═══════════════════════════════════════════════════════════
# 2. Feature Computation (1s bars + tick enrichment)
# ═══════════════════════════════════════════════════════════

def compute_1s_features_v4(trades: pd.DataFrame) -> pd.DataFrame:
    """
    aggTrades → 1s OHLCV + tick 微观特征 → rolling features。

    V4 相比 V3:
      - 保持所有 V3 特征 (OHLCV-based)
      - 新增 12 个 tick-level 原始特征
      - 新增 tick 特征的 rolling 聚合 (mean/std)
      - 总特征数: ~110+ (vs V3 的 65)
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

    # ── Tick-level micro features (NEW V4) ──
    logger.info("    Computing tick-level micro features...")
    tick_f = compute_tick_features(trades)
    # Align to OHLCV index
    tick_cols = [c for c in tick_f.columns if c.startswith("tick_")]
    for col in tick_cols:
        f[col] = tick_f[col].reindex(f.index).fillna(0)

    # Rolling aggregates of tick features
    tick_roll_cols = [
        "tick_n_prices", "tick_max_trade", "tick_size_cv",
        "tick_vwap_dev", "tick_size_imbalance", "tick_large_pct",
        "tick_range_per_trade", "tick_n_agg",
    ]
    for col in tick_roll_cols:
        if col in f.columns:
            for w in FEATURE_WINDOWS:
                f[f"{col}_ma_{w}"] = f[col].rolling(w, min_periods=1).mean()

    # ── Taker features ──
    from indicators.base.taker_flow import group_taker_orders
    taker = group_taker_orders(trades)
    if not taker.empty and len(taker) >= 10:
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
# 3. Data Loading
# ═══════════════════════════════════════════════════════════

def load_1s_features_v4(symbol: str, dates: List[str]) -> pd.DataFrame:
    """加载 V4 1s 特征 (含 tick-level 微观统计)。"""
    all_dfs = []
    for d_str in dates:
        dt = date.fromisoformat(d_str)
        cache_path = DataPaths.features(symbol, "1s_features_v4", dt)
        if cache_path.exists():
            logger.info(f"  [Cache] {d_str}")
            all_dfs.append(pd.read_parquet(cache_path))
        else:
            logger.info(f"  [Compute] {d_str}...")
            trades = load_agg_trades(symbol, dt, dt + timedelta(days=1))
            if trades.empty:
                logger.warning(f"  [Skip] {d_str} — no trades data")
                continue
            logger.info(f"    {len(trades):,} aggTrades")
            df = compute_1s_features_v4(trades)
            if df.empty:
                del trades; gc.collect(); continue
            df.to_parquet(cache_path)
            logger.info(f"    → {len(df):,} bars × {df.shape[1]} features")
            all_dfs.append(df)
            del trades; gc.collect()
    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs).sort_index()


# ═══════════════════════════════════════════════════════════
# 4. Efficient TBM at Subsample Positions
# ═══════════════════════════════════════════════════════════

def make_target_tbm_at_positions(
    close: np.ndarray,
    ret_std: np.ndarray,
    interval: int,
    positions: np.ndarray,
    tp_mult: float,
    sl_mult: float,
) -> np.ndarray:
    """
    仅在指定的 positions 处计算 TBM 标签 (大幅加速)。

    Args:
        close: 完整 1s close 价格序列
        ret_std: rolling return_1 std
        interval: 时间窗口 (秒)
        positions: 需要计算标签的 bar 索引 (e.g., 每 60 个取一个)
        tp_mult: 上轨乘数
        sl_mult: 下轨乘数

    Returns:
        labels: 与 positions 等长的标签数组
    """
    n_close = len(close)
    labels = np.full(len(positions), np.nan)

    for idx, i in enumerate(positions):
        if i + interval >= n_close:
            continue
        if np.isnan(ret_std[i]) or ret_std[i] < 1e-10:
            labels[idx] = 0
            continue

        vol = ret_std[i] * np.sqrt(interval)
        tp_bps = vol * tp_mult
        sl_bps = vol * sl_mult
        p0 = close[i]
        if p0 < 1e-10:
            labels[idx] = 0
            continue

        # Vectorized inner check
        fwd_prices = close[i + 1: i + interval + 1]
        rets = (fwd_prices - p0) / p0 * 10000

        up_hits = np.where(rets >= tp_bps)[0]
        dn_hits = np.where(rets <= -sl_bps)[0]

        first_up = up_hits[0] if len(up_hits) > 0 else interval + 1
        first_dn = dn_hits[0] if len(dn_hits) > 0 else interval + 1

        if first_up < first_dn:
            labels[idx] = 1
        elif first_dn < first_up:
            labels[idx] = -1
        else:
            labels[idx] = 0

    return labels


def make_target_return_at_positions(
    cum_return: np.ndarray,
    positions: np.ndarray,
    interval: int,
) -> np.ndarray:
    """在 subsample 位置计算 forward return (用于回测的 actual return)。"""
    n = len(cum_return)
    ret = np.full(len(positions), np.nan)
    for idx, i in enumerate(positions):
        if i + interval < n:
            ret[idx] = cum_return[i + interval] - cum_return[i]
    return ret


# ═══════════════════════════════════════════════════════════
# 5. Feature Selection
# ═══════════════════════════════════════════════════════════

def select_features(X, y, names, top_n=50):
    """分类任务的 permutation importance feature selection。"""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.inspection import permutation_importance

    sc = StandardScaler()
    Xn = sc.fit_transform(X)

    m = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    m.fit(Xn, y)
    val_n = max(200, len(X) // 5)
    scoring = "balanced_accuracy" if len(np.unique(y)) > 2 else "accuracy"
    perm = permutation_importance(
        m, Xn[-val_n:], y[-val_n:],
        n_repeats=5, random_state=42, scoring=scoring,
    )

    imp = perm.importances_mean
    top_idx = np.argsort(imp)[::-1][:top_n]
    selected = [names[i] for i in top_idx]

    logger.info(f"  Top features (balanced_accuracy):")
    for rank, i in enumerate(top_idx[:15]):
        logger.info(f"    {rank+1:>3}. {names[i]:<35} imp={imp[i]:.6f}")

    return selected, {names[i]: float(imp[i]) for i in top_idx}


# ═══════════════════════════════════════════════════════════
# 6. Models
# ═══════════════════════════════════════════════════════════

def train_classifier(X_tr, y_tr, X_te, model_type="logistic"):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_tr)
    Xte = sc.transform(X_te)

    if model_type == "logistic":
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(
            C=0.1, max_iter=2000, random_state=42,
            multi_class="multinomial",
        )
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
    name = model_type.upper()
    logger.info(f"  [{name}] Train acc={acc_tr:.4f}")

    # 返回 signed signal: P(+1) - P(-1)
    classes = m.classes_
    proba = m.predict_proba(Xte)

    if len(classes) == 2:
        up_idx = np.where(classes == 1)[0][0]
        signal = proba[:, up_idx] - 0.5
    else:
        # TBM 3-class: signal = P(+1) - P(-1)
        p_up = proba[:, np.where(classes == 1)[0][0]] if 1 in classes else np.zeros(len(Xte))
        p_dn = proba[:, np.where(classes == -1)[0][0]] if -1 in classes else np.zeros(len(Xte))
        signal = p_up - p_dn

    return signal, m


# ═══════════════════════════════════════════════════════════
# 7. Fixed-Interval Backtest
# ═══════════════════════════════════════════════════════════

def backtest_fixed_interval(
    signal: np.ndarray,
    actual_return: np.ndarray,
    timestamps,
    cost_bps: float = 4.0,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """固定间隔回测：每个周期开头下单，周期内仓位不变。"""
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

    ret_mean = pnl.mean()
    ret_std = pnl.std() + 1e-10
    sr = ret_mean / ret_std * np.sqrt(len(pnl))

    if len(active) > 0:
        wins = active[active["net_pnl"] > 0]
        wr = len(wins) / len(active)
    else:
        wr = 0

    pos_pnl = pnl[pnl > 0].sum()
    neg_pnl = pnl[pnl < 0].abs().sum() + 1e-10
    pf = pos_pnl / neg_pnl

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
# 8. Single TBM Experiment
# ═══════════════════════════════════════════════════════════

def run_tbm_experiment(
    train_1s: pd.DataFrame,
    test_1s: pd.DataFrame,
    tp_mult: float,
    sl_mult: float,
    cost_bps: float,
    top_features: int = 50,
) -> Dict:
    """
    运行一组 1min TBM 实验。

    高效流程:
      1. 仅在 subsample 位置 (每 60s) 计算 TBM 标签
      2. 训练 logistic + GBM 分类器
      3. 多阈值回测
    """
    label = f"tp={tp_mult:.2f}_sl={sl_mult:.2f}"
    logger.info(f"\n{'─'*60}")
    logger.info(f"  TBM Grid: {label}")
    logger.info(f"{'─'*60}")

    interval = INTERVAL
    close_train = train_1s["close"].values
    close_test = test_1s["close"].values
    ret_std_train = train_1s["return_1"].rolling(max(20, interval), min_periods=10).std().values
    ret_std_test = test_1s["return_1"].rolling(max(20, interval), min_periods=10).std().values

    # Subsample positions (每 60 个 bar 取一个)
    train_positions = np.arange(0, len(train_1s), interval)
    test_positions = np.arange(0, len(test_1s), interval)

    # 高效 TBM 标签计算
    t0 = time.time()
    y_train = make_target_tbm_at_positions(
        close_train, ret_std_train, interval, train_positions, tp_mult, sl_mult,
    )
    y_test = make_target_tbm_at_positions(
        close_test, ret_std_test, interval, test_positions, tp_mult, sl_mult,
    )
    dt_tbm = time.time() - t0
    logger.info(f"  TBM labels computed in {dt_tbm:.1f}s")

    # Forward return (用于回测)
    cum_return_train = train_1s["return_1"].cumsum().values
    cum_return_test = test_1s["return_1"].cumsum().values
    actual_ret_test = make_target_return_at_positions(cum_return_test, test_positions, interval)

    # 过滤 NaN labels
    train_mask = ~np.isnan(y_train)
    test_mask = ~np.isnan(y_test) & ~np.isnan(actual_ret_test)

    y_train_clean = y_train[train_mask].astype(int)
    y_test_clean = y_test[test_mask].astype(int)
    actual_ret_test_clean = actual_ret_test[test_mask]

    # Label distribution
    unique, counts = np.unique(y_train_clean, return_counts=True)
    dist = {int(u): int(c) for u, c in zip(unique, counts)}
    total = sum(dist.values())
    trend_pct = (dist.get(1, 0) + dist.get(-1, 0)) / total * 100 if total > 0 else 0
    logger.info(f"  Train: {total:,} samples")
    logger.info(f"  Label distribution: {dist}")
    logger.info(f"  Trend signals: {trend_pct:.1f}% (±1), Sideways: {100-trend_pct:.1f}% (0)")

    if total < 100 or len(np.unique(y_train_clean)) < 2:
        logger.warning(f"  [Skip] Insufficient training data")
        return {"tp_mult": tp_mult, "sl_mult": sl_mult, "label": label,
                "train_dist": dist, "results": {}, "best_config": "N/A", "best_sharpe": -999}

    # ── Feature columns ──
    exclude = {"_target", "return_1", "log_return", "close"}
    feat_cols = [
        c for c in train_1s.columns
        if c not in exclude
        and train_1s[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]
        and train_1s[c].std() > 1e-10
    ]

    # Extract features at subsample positions
    X_train_all = train_1s.iloc[train_positions[train_mask]][feat_cols].values
    X_test_all = test_1s.iloc[test_positions[test_mask]][feat_cols].values
    test_timestamps = test_1s.index[test_positions[test_mask]]

    logger.info(f"  Features: {len(feat_cols)}")

    # ── Feature selection ──
    selected, importance = select_features(
        X_train_all, y_train_clean.astype(float), feat_cols, top_n=top_features,
    )

    X_train = train_1s.iloc[train_positions[train_mask]][selected].values
    X_test = test_1s.iloc[test_positions[test_mask]][selected].values

    # ── Run models ──
    results = {}

    for mtype in ["logistic", "gbm"]:
        logger.info(f"\n  Training {mtype.upper()}...")
        t0 = time.time()
        signal, model = train_classifier(X_train, y_train_clean, X_test, model_type=mtype)
        dt_train = time.time() - t0
        logger.info(f"    Time: {dt_train:.1f}s")
        logger.info(f"    Signal: mean={signal.mean():.4f}, std={signal.std():.4f}")

        # 多阈值回测
        thresholds = [0, 0.05, 0.1, 0.15, 0.2, 0.3]

        for cost, cost_label in [(cost_bps, "taker"), (1.0, "maker")]:
            for thr in thresholds:
                bt = backtest_fixed_interval(
                    signal, actual_ret_test_clean,
                    test_timestamps, cost_bps=cost, threshold=thr,
                )
                met = compute_bt_metrics(bt)
                key = f"{mtype}_{cost_label}_thr{thr}"
                results[key] = met

                logger.info(
                    f"    {key:<35} N={met['n_trades']:>4} Act={met['n_active']:>5} "
                    f"Gross={met['gross_pnl']:>+7.0f} Net={met['total_pnl']:>+7.0f} "
                    f"WR={met['win_rate']:.1%} SR={met['sharpe']:>+6.2f} "
                    f"PF={met['profit_factor']:.2f}"
                )

    # Best config
    best_key = max(results, key=lambda k: results[k]["sharpe"]) if results else "N/A"
    best_met = results.get(best_key, {})
    best_sr = best_met.get("sharpe", -999)
    logger.info(f"\n  ★ Best: {best_key} (Sharpe={best_sr:.2f})")

    return {
        "tp_mult": tp_mult,
        "sl_mult": sl_mult,
        "label": label,
        "train_dist": dist,
        "trend_pct": trend_pct,
        "train_samples": total,
        "test_samples": int(test_mask.sum()),
        "n_features_total": len(feat_cols),
        "n_features_selected": len(selected),
        "top_features": {k: v for k, v in list(importance.items())[:15]},
        "results": results,
        "best_config": best_key,
        "best_sharpe": best_sr,
    }


# ═══════════════════════════════════════════════════════════
# 9. Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SOL/USDT")
    parser.add_argument("--cost-bps", type=float, default=4.0)
    parser.add_argument("--top-features", type=int, default=50)
    parser.add_argument("--quick", action="store_true",
                        help="Only run 4 symmetric grid points")
    parser.add_argument("--use-original-dates", action="store_true",
                        help="Use only 7 original train dates (skip extra)")
    args = parser.parse_args()

    tbm_grid = TBM_GRID_QUICK if args.quick else TBM_GRID_FULL
    train_dates = ORIGINAL_TRAIN_DATES if args.use_original_dates else EXTENDED_TRAIN_DATES

    logger.info("=" * 70)
    logger.info("SOL 1min TBM Explorer — V4")
    logger.info(f"  Interval: 1min (60s) fixed")
    logger.info(f"  TBM grid: {len(tbm_grid)} combinations")
    logger.info(f"  Feature windows: {FEATURE_WINDOWS}s (physical, fixed)")
    logger.info(f"  Feature enrichment: tick-level micro stats (NEW)")
    logger.info(f"  Cost: {args.cost_bps} bps/leg (taker), 1.0 bps/leg (maker)")
    logger.info(f"  Train: {len(train_dates)} days ({train_dates[0]} → {train_dates[-1]})")
    logger.info(f"  Test:  {len(TEST_DATES)} days ({TEST_DATES[0]} → {TEST_DATES[-1]})")
    logger.info(f"  Top features: {args.top_features}")
    logger.info("=" * 70)

    # ── Load V4 features ──
    logger.info("\n[1] Loading V4 1s features (with tick-level micro stats)...")
    train_1s = load_1s_features_v4(args.symbol, train_dates)
    test_1s = load_1s_features_v4(args.symbol, TEST_DATES)

    if train_1s.empty or test_1s.empty:
        logger.error("Failed to load features!")
        return

    logger.info(f"  Train: {len(train_1s):,} bars × {train_1s.shape[1]} features")
    logger.info(f"  Test:  {len(test_1s):,} bars × {test_1s.shape[1]} features")

    # ── Report all feature columns ──
    exclude = {"return_1", "log_return", "close"}
    all_feat_cols = [
        c for c in train_1s.columns
        if c not in exclude
        and train_1s[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]
        and train_1s[c].std() > 1e-10
    ]
    logger.info(f"\n[2] Feature inventory: {len(all_feat_cols)} effective features")
    # Categorize
    cats = {
        "Return": [c for c in all_feat_cols if c.startswith("return_")],
        "Volume": [c for c in all_feat_cols if any(c.startswith(p) for p in ["volume", "buy_volume", "sell_volume", "volume_", "signed_flow", "imbalance_ma"])],
        "Bar Structure": [c for c in all_feat_cols if any(c.startswith(p) for p in ["bar_", "body_", "n_trades", "trades_ma"])],
        "Spread/Impact": [c for c in all_feat_cols if any(c.startswith(p) for p in ["spread_proxy", "impact_proxy"])],
        "Autocorrelation": [c for c in all_feat_cols if "autocorr" in c or "bv_sv_corr" in c],
        "Taker": [c for c in all_feat_cols if c.startswith("taker_")],
        "Tick Micro (NEW)": [c for c in all_feat_cols if c.startswith("tick_")],
    }
    for cat, cols in cats.items():
        if cols:
            logger.info(f"    {cat}: {len(cols)} features")
            for c in cols[:5]:
                logger.info(f"      - {c}")
            if len(cols) > 5:
                logger.info(f"      ... and {len(cols) - 5} more")

    # ── Run TBM grid search ──
    logger.info(f"\n[3] Running TBM grid search ({len(tbm_grid)} combos)...")
    all_experiments = []

    for tp_mult, sl_mult in tbm_grid:
        exp = run_tbm_experiment(
            train_1s, test_1s, tp_mult, sl_mult,
            cost_bps=args.cost_bps, top_features=args.top_features,
        )
        all_experiments.append(exp)

    # ═══ Grand Summary ═══
    logger.info(f"\n{'='*90}")
    logger.info("GRAND SUMMARY — TBM Grid Search (1min)")
    logger.info(f"{'='*90}")
    logger.info(
        f"{'tp_mult':>7} {'sl_mult':>7} {'Trend%':>6} {'Train':>6} "
        f"{'Best Config':<40} {'SR':>6} {'Net PnL':>8} {'Trades':>6} {'WR':>5}"
    )
    logger.info(f"{'-'*90}")

    summary_rows = []
    for exp in all_experiments:
        best = exp["results"].get(exp["best_config"], {})
        row = {
            "tp_mult": exp["tp_mult"],
            "sl_mult": exp["sl_mult"],
            "trend_pct": exp.get("trend_pct", 0),
            "train_samples": exp.get("train_samples", 0),
            "best_config": exp["best_config"],
            "sharpe": best.get("sharpe", -999),
            "net_pnl": best.get("total_pnl", 0),
            "n_trades": best.get("n_trades", 0),
            "win_rate": best.get("win_rate", 0),
            "gross_pnl": best.get("gross_pnl", 0),
            "profit_factor": best.get("profit_factor", 0),
            "max_dd": best.get("max_dd", 0),
            "train_dist": exp.get("train_dist", {}),
        }
        summary_rows.append(row)
        logger.info(
            f"{row['tp_mult']:>7.2f} {row['sl_mult']:>7.2f} {row['trend_pct']:>5.1f}% "
            f"{row['train_samples']:>6} "
            f"{row['best_config']:<40} "
            f"{row['sharpe']:>+6.2f} {row['net_pnl']:>+8.0f} {row['n_trades']:>6} "
            f"{row['win_rate']:>4.1%}"
        )

    # Overall best
    best_exp = max(all_experiments, key=lambda e: e["best_sharpe"])
    logger.info(f"\n{'─'*90}")
    logger.info(
        f"★ OVERALL BEST: tp={best_exp['tp_mult']:.2f}, sl={best_exp['sl_mult']:.2f} → "
        f"{best_exp['best_config']} (Sharpe={best_exp['best_sharpe']:.2f})"
    )
    logger.info(f"  Label distribution: {best_exp.get('train_dist', {})}")
    logger.info(f"  Trend signal %: {best_exp.get('trend_pct', 0):.1f}%")
    if best_exp.get("top_features"):
        logger.info(f"  Top features:")
        for feat, imp in list(best_exp["top_features"].items())[:10]:
            logger.info(f"    {feat:<35} imp={imp:.6f}")
    logger.info(f"{'─'*90}")

    # Compare with V3 baseline
    logger.info(f"\nV3 Baseline (tp=1.5, sl=1.5):")
    logger.info(f"  logistic_maker_thr0.2: Sharpe=+1.52, Net=+358, N=26, WR=69.2%")

    # ── Save ──
    save_data = {
        "config": {
            "version": "v4_1min_tbm",
            "symbol": args.symbol,
            "interval": "1min",
            "target": "tbm",
            "feature_windows": FEATURE_WINDOWS,
            "cost_bps": args.cost_bps,
            "train_dates": train_dates,
            "test_dates": TEST_DATES,
            "tbm_grid": [{"tp_mult": tp, "sl_mult": sl} for tp, sl in tbm_grid],
            "top_features_n": args.top_features,
        },
        "experiments": [{
            "tp_mult": e["tp_mult"],
            "sl_mult": e["sl_mult"],
            "label": e["label"],
            "train_dist": e.get("train_dist"),
            "trend_pct": e.get("trend_pct"),
            "train_samples": e.get("train_samples"),
            "test_samples": e.get("test_samples"),
            "n_features_total": e.get("n_features_total"),
            "n_features_selected": e.get("n_features_selected"),
            "top_features": e.get("top_features"),
            "best_config": e["best_config"],
            "best_sharpe": e["best_sharpe"],
            "all_results": e.get("results"),
        } for e in all_experiments],
        "summary": summary_rows,
        "overall_best": {
            "tp_mult": best_exp["tp_mult"],
            "sl_mult": best_exp["sl_mult"],
            "best_config": best_exp["best_config"],
            "best_sharpe": best_exp["best_sharpe"],
        },
    }

    sp = DataPaths.backtest_dir("sol_v4_1min_tbm") / "summary.json"
    with open(sp, "w") as fp:
        json.dump(save_data, fp, indent=2, default=str)
    logger.info(f"\nSaved: {sp}")


if __name__ == "__main__":
    main()
