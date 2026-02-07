"""
统一回测引擎。

提供：
- compute_stats: 计算回测统计指标
- backtest_fvg: FVG 策略回测（带止损/止盈/冷却期）
- param_grid_search: 网格参数搜索
"""
from __future__ import annotations

from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from .fvg.model import FVGModel


# ─────────────────────────────────────────────────────────
# 统计量计算
# ─────────────────────────────────────────────────────────

def compute_stats(rdf: pd.DataFrame) -> Dict[str, Any]:
    """从交易结果 DataFrame 计算标准回测统计量。"""
    empty = {
        "n_signals": 0, "n_wins": 0, "win_rate": 0, "total_return": 0,
        "avg_return": 0, "avg_win": 0, "avg_loss": 0, "profit_factor": 0,
        "sharpe": 0, "max_drawdown": 0,
    }
    if rdf.empty:
        return empty

    n = len(rdf)
    wins = rdf[rdf["is_win"]]
    losses = rdf[~rdf["is_win"]]

    cum = rdf["return_pct"].cumsum()
    peak = cum.cummax()
    dd = cum - peak

    return {
        "n_signals": n,
        "n_wins": int(rdf["is_win"].sum()),
        "win_rate": round(rdf["is_win"].mean() * 100, 2),
        "total_return": round(rdf["return_pct"].sum(), 4),
        "avg_return": round(rdf["return_pct"].mean(), 4),
        "avg_win": round(wins["return_pct"].mean(), 4) if len(wins) > 0 else 0,
        "avg_loss": round(losses["return_pct"].mean(), 4) if len(losses) > 0 else 0,
        "profit_factor": (
            round(abs(wins["return_pct"].sum() / (losses["return_pct"].sum() + 1e-10)), 2)
            if len(losses) > 0 else float("inf")
        ),
        "sharpe": round(
            rdf["return_pct"].mean() / (rdf["return_pct"].std() + 1e-10) * np.sqrt(n), 3
        ),
        "max_drawdown": round(dd.min(), 4),
        "results_df": rdf,
    }


# ─────────────────────────────────────────────────────────
# FVG 回测
# ─────────────────────────────────────────────────────────

def backtest_fvg(
    feature_df: pd.DataFrame,
    model: FVGModel,
    ohlcv: pd.DataFrame,
    hold_bars: int = 10,
    min_proba: float = 0.6,
    stop_loss_pct: float = 0.2,
    take_profit_pct: float = 0.4,
    cooldown_bars: int = 3,
) -> Dict[str, Any]:
    """
    FVG 策略回测引擎。

    特性：
    1. 预测价格方向 (pred_fill: 1=涨, 0=跌)
    2. 止损/止盈 逐根 K 线模拟
    3. 信号冷却期（避免同一区域反复开仓）
    """
    predictions = model.predict(feature_df)
    signals = predictions[predictions["pred_proba"] >= min_proba].copy()

    if signals.empty:
        return compute_stats(pd.DataFrame())

    results = []
    last_exit_idx = -cooldown_bars

    for _, row in signals.iterrows():
        ts = pd.to_datetime(row["timestamp"])
        pred_direction = row["pred_fill"]  # 1=涨, 0=跌

        if ts not in ohlcv.index:
            continue

        entry_idx = ohlcv.index.get_loc(ts)

        # Cooldown
        if entry_idx - last_exit_idx < cooldown_bars:
            continue

        entry_price = ohlcv.iloc[entry_idx]["close"]
        direction = 1 if pred_direction == 1 else -1

        # Bar-by-bar simulation
        exit_price = None
        exit_reason = "hold"
        actual_exit_idx = min(entry_idx + hold_bars, len(ohlcv) - 1)

        for j in range(entry_idx + 1, min(entry_idx + hold_bars + 1, len(ohlcv))):
            bar = ohlcv.iloc[j]

            if direction == 1:  # Long
                worst = (bar["low"] - entry_price) / entry_price * 100
                if worst <= -stop_loss_pct:
                    exit_price = entry_price * (1 - stop_loss_pct / 100)
                    exit_reason = "stop_loss"
                    actual_exit_idx = j
                    break
                best = (bar["high"] - entry_price) / entry_price * 100
                if best >= take_profit_pct:
                    exit_price = entry_price * (1 + take_profit_pct / 100)
                    exit_reason = "take_profit"
                    actual_exit_idx = j
                    break
            else:  # Short
                worst = (bar["high"] - entry_price) / entry_price * 100
                if worst >= stop_loss_pct:
                    exit_price = entry_price * (1 + stop_loss_pct / 100)
                    exit_reason = "stop_loss"
                    actual_exit_idx = j
                    break
                best = (entry_price - bar["low"]) / entry_price * 100
                if best >= take_profit_pct:
                    exit_price = entry_price * (1 - take_profit_pct / 100)
                    exit_reason = "take_profit"
                    actual_exit_idx = j
                    break

        if exit_price is None:
            exit_price = ohlcv.iloc[actual_exit_idx]["close"]

        last_exit_idx = actual_exit_idx

        if direction == 1:
            ret_pct = (exit_price - entry_price) / entry_price * 100
        else:
            ret_pct = (entry_price - exit_price) / entry_price * 100

        results.append({
            "entry_time": ts,
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "return_pct": ret_pct,
            "is_win": ret_pct > 0,
            "exit_reason": exit_reason,
            "cluster": row.get("cluster", -1),
            "pred_proba": row["pred_proba"],
            "hold_bars": actual_exit_idx - entry_idx,
        })

    return compute_stats(pd.DataFrame(results) if results else pd.DataFrame())


# ─────────────────────────────────────────────────────────
# 参数网格搜索
# ─────────────────────────────────────────────────────────

def param_grid_search(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: List[str],
    ohlcv: pd.DataFrame,
    target_col: str,
    n_clusters: int = 4,
) -> Dict[str, Any]:
    """
    网格搜索最优回测参数组合。

    Returns:
        {"model", "report", "best_params", "best_result", "best_score", "search_log"}
    """
    param_grid = {
        "hold_bars": [5, 10, 15, 20],
        "stop_loss_pct": [0.1, 0.15, 0.2, 0.3],
        "take_profit_pct": [0.2, 0.3, 0.5, 0.8],
        "min_proba": [0.5, 0.55, 0.6, 0.65],
        "cooldown_bars": [2, 3, 5],
    }

    model = FVGModel(n_clusters=n_clusters)
    model._feature_cols = feature_cols[:]
    report = model.fit(train, target_col=target_col)
    if "error" in report:
        return {"error": "train_failed"}

    best_score = -999
    best_params: Dict[str, Any] = {}
    best_result: Dict[str, Any] = {}
    results_log = []

    for hold in param_grid["hold_bars"]:
        for sl in param_grid["stop_loss_pct"]:
            for tp in param_grid["take_profit_pct"]:
                if tp <= sl:
                    continue
                for mp in param_grid["min_proba"]:
                    for cd in param_grid["cooldown_bars"]:
                        bt = backtest_fvg(
                            test, model, ohlcv,
                            hold_bars=hold,
                            min_proba=mp,
                            stop_loss_pct=sl,
                            take_profit_pct=tp,
                            cooldown_bars=cd,
                        )

                        n_sig = bt.get("n_signals", 0)
                        if n_sig < 5:
                            continue

                        sharpe = bt.get("sharpe", 0)
                        wr = bt.get("win_rate", 0)
                        pf = bt.get("profit_factor", 0)
                        total_ret = bt.get("total_return", 0)

                        score = sharpe + 0.02 * wr + 0.1 * min(pf, 5) + 0.05 * total_ret

                        results_log.append({
                            "hold": hold, "sl": sl, "tp": tp, "mp": mp, "cd": cd,
                            "signals": n_sig, "wr": wr, "sharpe": sharpe,
                            "total_ret": total_ret, "pf": pf, "score": score,
                        })

                        if score > best_score:
                            best_score = score
                            best_params = {
                                "hold_bars": hold, "stop_loss_pct": sl,
                                "take_profit_pct": tp, "min_proba": mp,
                                "cooldown_bars": cd,
                            }
                            best_result = bt

    return {
        "model": model,
        "report": report,
        "best_params": best_params,
        "best_result": best_result,
        "best_score": best_score,
        "search_log": pd.DataFrame(results_log),
    }
