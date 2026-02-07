"""
回测统计指标计算。
"""
from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd


def compute_backtest_metrics(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    从交易结果 DataFrame 计算标准回测统计量。

    期望列: return_pct, is_win（至少）
    """
    if results_df.empty or "return_pct" not in results_df.columns:
        return {
            "n_signals": 0, "n_wins": 0, "win_rate": 0, "total_return": 0,
            "avg_return": 0, "avg_win": 0, "avg_loss": 0,
            "profit_factor": 0, "sharpe": 0, "sortino": 0, "max_drawdown": 0,
        }

    r = results_df["return_pct"]
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
        "n_wins": int((r > 0).sum()),
        "win_rate": round((r > 0).mean() * 100, 2),
        "total_return": round(r.sum(), 4),
        "avg_return": round(r.mean(), 4),
        "avg_win": round(wins.mean(), 4) if len(wins) > 0 else 0,
        "avg_loss": round(losses.mean(), 4) if len(losses) > 0 else 0,
        "profit_factor": round(abs(wins.sum() / (losses.sum() + 1e-10)), 2) if len(losses) > 0 else float("inf"),
        "sharpe": round(r.mean() / ret_std * np.sqrt(n), 3),
        "sortino": round(r.mean() / (down_std + 1e-10) * np.sqrt(n), 3),
        "max_drawdown": round(dd.min(), 4),
        "median_return": round(r.median(), 4),
        "skewness": round(r.skew(), 4) if n > 2 else 0,
    }
