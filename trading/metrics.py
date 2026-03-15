"""
Investment management metrics for backtest results.

Includes: returns, annualized returns, max drawdown, Sharpe, Sortino, Calmar,
profit factor, win rate, etc.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def compute_investment_metrics(
    pnl_series: pd.Series,
    initial_capital: float = 1.0,
    periods_per_year: float = 252 * 24 * 12,  # 5min bars
) -> Dict[str, Any]:
    """
    Compute comprehensive investment metrics from PnL series (in bps or %).

    pnl_series: per-period PnL (e.g. bps or %)
    initial_capital: starting capital for return calculation
    periods_per_year: for annualization (e.g. 5min bars ≈ 252*24*12)
    """
    if pnl_series.empty or len(pnl_series) < 2:
        return _empty_metrics()

    pnl = pnl_series.dropna()
    if len(pnl) < 2:
        return _empty_metrics()

    # Returns (treat pnl as bps; convert to decimal for returns)
    returns_bps = pnl
    returns_pct = returns_bps / 10000.0  # bps -> decimal
    cum_returns = (1 + returns_pct).cumprod()
    total_return_pct = (cum_returns.iloc[-1] - 1) * 100

    # Annualized return
    n_periods = len(pnl)
    years = n_periods / periods_per_year
    if years > 0:
        annualized_return = (cum_returns.iloc[-1] ** (1 / years) - 1) * 100
    else:
        annualized_return = 0.0

    # Max drawdown
    cum = cum_returns
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    max_drawdown_pct = drawdown.min() * 100

    # Volatility (annualized)
    vol_ann = returns_pct.std() * np.sqrt(periods_per_year) * 100 if returns_pct.std() > 1e-12 else 0

    # Sharpe (assuming risk-free = 0)
    mean_ret = returns_pct.mean()
    std_ret = returns_pct.std() + 1e-12
    sharpe = mean_ret / std_ret * np.sqrt(periods_per_year)

    # Sortino (downside deviation)
    downside = returns_pct[returns_pct < 0]
    down_std = downside.std() if len(downside) > 1 else 1e-12
    sortino = mean_ret / down_std * np.sqrt(periods_per_year) if down_std > 0 else 0

    # Calmar (return / max drawdown)
    calmar = annualized_return / abs(max_drawdown_pct) if max_drawdown_pct != 0 else 0

    # Win rate
    wins = (returns_pct > 0).sum()
    total = len(returns_pct)
    win_rate = wins / total if total > 0 else 0

    # Profit factor
    gross_profit = returns_pct[returns_pct > 0].sum()
    gross_loss = abs(returns_pct[returns_pct < 0].sum())
    profit_factor = gross_profit / (gross_loss + 1e-12) if gross_loss > 0 else float("inf")

    # Number of trades (position changes)
    n_trades = int((pnl != 0).sum())

    return {
        "total_return_pct": round(total_return_pct, 4),
        "annualized_return_pct": round(annualized_return, 4),
        "max_drawdown_pct": round(max_drawdown_pct, 4),
        "volatility_ann_pct": round(vol_ann, 4),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "calmar_ratio": round(calmar, 4),
        "win_rate": round(win_rate * 100, 2),
        "profit_factor": round(profit_factor, 4),
        "n_trades": n_trades,
        "n_periods": n_periods,
        "net_pnl_bps": round(pnl.sum(), 2),
        "gross_pnl_bps": round(pnl[pnl > 0].sum(), 2),
        "loss_pnl_bps": round(pnl[pnl < 0].sum(), 2),
    }


def _empty_metrics() -> Dict[str, Any]:
    return {
        "total_return_pct": 0,
        "annualized_return_pct": 0,
        "max_drawdown_pct": 0,
        "volatility_ann_pct": 0,
        "sharpe_ratio": 0,
        "sortino_ratio": 0,
        "calmar_ratio": 0,
        "win_rate": 0,
        "profit_factor": 0,
        "n_trades": 0,
        "n_periods": 0,
        "net_pnl_bps": 0,
        "gross_pnl_bps": 0,
        "loss_pnl_bps": 0,
    }


def format_metrics_report(metrics: Dict[str, Any]) -> str:
    """Format metrics as readable report."""
    lines = [
        "=" * 50,
        "Backtest Metrics",
        "=" * 50,
        f"  Net PnL (bps):        {metrics['net_pnl_bps']:>12.2f}",
        f"  Total Return:         {metrics['total_return_pct']:>12.2f}%",
        f"  Annualized Return:    {metrics['annualized_return_pct']:>12.2f}%",
        f"  Max Drawdown:         {metrics['max_drawdown_pct']:>12.2f}%",
        f"  Volatility (ann):     {metrics['volatility_ann_pct']:>12.2f}%",
        f"  Sharpe Ratio:         {metrics['sharpe_ratio']:>12.2f}",
        f"  Sortino Ratio:        {metrics['sortino_ratio']:>12.2f}",
        f"  Calmar Ratio:         {metrics['calmar_ratio']:>12.2f}",
        f"  Win Rate:             {metrics['win_rate']:>12.2f}%",
        f"  Profit Factor:        {metrics['profit_factor']:>12.2f}",
        f"  N Trades:             {metrics['n_trades']:>12}",
        "=" * 50,
    ]
    return "\n".join(lines)
