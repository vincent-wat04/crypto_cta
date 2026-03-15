"""
Paper backtest with maker-first-then-taker cost simulation.

Simulates: place limit at mid (maker), if not filled within timeout → market (taker).
Cost model: maker rebate (-2 bps), taker fee (4 bps).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .cost_model import CostModel
from .metrics import compute_investment_metrics, format_metrics_report


def backtest_maker_first_taker(
    signal: np.ndarray,
    actual_return: np.ndarray,
    timestamps: pd.DatetimeIndex,
    cost_model: CostModel,
    threshold: float = 0.0,
    maker_fill_rate: float = 0.7,
) -> pd.DataFrame:
    """
    Backtest with maker-first-then-taker cost simulation.

    signal: model signal (regression: continuous; classification: prob diff)
    actual_return: realized return in bps for each interval
    timestamps: index for each bar
    cost_model: CostModel with maker/taker fees
    threshold: signal threshold for position (0 = all non-zero signals trade)
    maker_fill_rate: fraction of orders filled as maker (0–1). Rest are taker.
    """
    n = len(signal)
    trades = []
    cum_pnl = 0.0
    prev_pos = 0

    for i in range(n):
        s = signal[i]
        pos = 0
        if not np.isnan(s):
            if s > threshold:
                pos = 1
            elif s < -threshold:
                pos = -1

        gross = actual_return[i] * pos if not np.isnan(actual_return[i]) else 0.0

        # Cost: only when position changes
        cost_bps = 0.0
        if pos != prev_pos and (pos != 0 or prev_pos != 0):
            # Round-trip: 2 legs. Simulate maker_fill_rate as maker.
            cost_bps = 2 * (
                maker_fill_rate * cost_model.maker_fee_bps
                + (1 - maker_fill_rate) * cost_model.taker_fee_bps
            )

        net = gross - cost_bps
        cum_pnl += net

        trades.append({
            "ts": timestamps[i] if i < len(timestamps) else None,
            "position": pos,
            "signal": float(s) if not np.isnan(s) else 0,
            "actual_return": float(actual_return[i]) if not np.isnan(actual_return[i]) else 0,
            "gross_pnl": gross,
            "cost": cost_bps,
            "net_pnl": net,
            "cum_pnl": cum_pnl,
        })
        prev_pos = pos

    return pd.DataFrame(trades)


def run_backtest_with_metrics(
    signal: np.ndarray,
    actual_return: np.ndarray,
    timestamps: pd.DatetimeIndex,
    cost_model: Optional[CostModel] = None,
    threshold: float = 0.0,
    maker_fill_rate: float = 0.7,
    interval_sec: int = 300,
) -> tuple[pd.DataFrame, dict]:
    """
    Run paper backtest and compute comprehensive metrics.

    interval_sec: bar interval in seconds (300 = 5min) for annualization.
    Returns (trades_df, metrics_dict).
    """
    if cost_model is None:
        cost_model = CostModel()

    df = backtest_maker_first_taker(
        signal=signal,
        actual_return=actual_return,
        timestamps=timestamps,
        cost_model=cost_model,
        threshold=threshold,
        maker_fill_rate=maker_fill_rate,
    )

    # PnL series for metrics (only active bars for return calc)
    pnl = df["net_pnl"]
    periods_per_year = 365 * 24 * 3600 / interval_sec  # 5min -> ~105120
    metrics = compute_investment_metrics(
        pnl,
        initial_capital=1.0,
        periods_per_year=periods_per_year,
    )

    return df, metrics
