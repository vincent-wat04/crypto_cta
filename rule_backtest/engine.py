"""
Per-bar backtest engine for rule-based strategies.

Orchestrates: Signal → Position Sizing → Stop-Loss → Exit → Execution cost → PnL tracking.

Produces a BacktestResult containing:
  - Per-bar trade log (DataFrame)
  - Aggregate metrics (dict)
  - Component parameters (for reproducibility)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from .signals import BaseSignal
from .position_sizing import BasePositionSizer, FixedSizer
from .stop_loss import BaseStopLoss, NoStopLoss, ATRStopLoss
from .exit_rules import BaseExitRule, SignalReversalExit
from .execution import BaseExecution, MakerFirstExecution


@dataclass
class BacktestResult:
    """Container for full backtest output."""
    trades: pd.DataFrame
    metrics: Dict[str, Any]
    params: Dict[str, Any]
    factor_name: str = ""


@dataclass
class BacktestConfig:
    """Configuration bundle for a single backtest run."""
    signal: BaseSignal
    sizer: BasePositionSizer = field(default_factory=FixedSizer)
    stop_loss: BaseStopLoss = field(default_factory=NoStopLoss)
    exit_rule: BaseExitRule = field(default_factory=SignalReversalExit)
    execution: BaseExecution = field(default_factory=MakerFirstExecution)
    initial_capital: float = 100_000.0
    bar_seconds: int = 60
    bar_mode: str = "time"         # "time" or "trade_count"
    trades_per_bar: int = 200
    trade_source: str = "raw"      # "raw" or "merged"
    signal_flip: bool = False       # flip signal direction for inverse predictors
    hold_through_flat: bool = False  # if True, signal=0 means "keep current position"
                                     # (only exit via reversal, stop-loss, or exit rule)


def run_backtest(
    bars: pd.DataFrame,
    factor: pd.Series,
    config: BacktestConfig,
    factor_name: str = "factor",
) -> BacktestResult:
    """
    Run a single backtest.

    Args:
        bars: OHLCV bars with columns [open, high, low, close, volume].
              Index = DatetimeIndex (bar timestamps).
        factor: indicator / factor Series aligned to bars index.
        config: BacktestConfig with all component choices.
        factor_name: name for logging.

    Returns:
        BacktestResult with trades DataFrame, metrics dict, and params.
    """
    bars = bars.copy()
    factor = factor.reindex(bars.index)

    sig_series = config.signal.generate(factor)
    if config.signal_flip:
        sig_series = -sig_series
    pos_series = config.sizer.size(sig_series, bars["close"])

    close = bars["close"].values.astype(float)
    n = len(bars)

    # Pre-compute ATR if ATRStopLoss is used
    atr_series = None
    if isinstance(config.stop_loss, ATRStopLoss):
        tr = bars["high"] - bars["low"]
        atr_series = tr.rolling(14, min_periods=3).mean() / bars["close"] * 10000  # bps

    # Per-bar loop with stop-loss and exit-rule checks
    records: List[Dict] = []
    position = 0.0          # current signed position
    entry_price = 0.0
    entry_bar = 0
    cum_pnl_bps = 0.0
    trade_pnl_bps = 0.0     # pnl since entry for current trade

    for i in range(n):
        desired_pos = float(pos_series.iloc[i]) if not np.isnan(pos_series.iloc[i]) else 0.0
        signal_now = int(sig_series.iloc[i]) if not np.isnan(sig_series.iloc[i]) else 0
        factor_now = float(factor.iloc[i]) if not np.isnan(factor.iloc[i]) else np.nan
        price = close[i]
        bars_held = i - entry_bar if position != 0 else 0

        # Unrealized PnL for the current trade
        if position != 0 and entry_price > 0:
            if position > 0:
                trade_pnl_bps = (price / entry_price - 1) * 10000 * abs(position)
            else:
                trade_pnl_bps = (1 - price / entry_price) * 10000 * abs(position)
        else:
            trade_pnl_bps = 0.0

        # Check stop-loss
        stop_triggered = False
        if position != 0:
            stop_triggered = config.stop_loss.check(i, entry_price, price, trade_pnl_bps, bars_held)

        # Check exit rule
        exit_triggered = False
        if position != 0 and not stop_triggered:
            exit_triggered = config.exit_rule.should_exit(
                i, int(np.sign(position)), signal_now, factor_now, bars_held)

        # Determine actual new position
        if config.hold_through_flat and desired_pos == 0.0 and position != 0.0:
            new_position = position
        else:
            new_position = desired_pos

        if stop_triggered:
            new_position = 0.0
        elif exit_triggered:
            new_position = 0.0

        # Compute PnL for this bar (mark-to-market)
        bar_return_bps = 0.0
        if i > 0 and position != 0:
            bar_return_bps = (close[i] / close[i - 1] - 1) * 10000 * position

        # Compute execution cost if position changes
        cost_bps = 0.0
        pos_delta = new_position - position
        if abs(pos_delta) > 1e-8:
            if abs(new_position) > abs(position):
                cost_bps = config.execution.entry_cost_bps() * abs(pos_delta)
            elif abs(new_position) < abs(position):
                cost_bps = config.execution.exit_cost_bps() * abs(pos_delta)
            else:
                # Flip direction: close old + open new
                cost_bps = (config.execution.exit_cost_bps() * abs(position)
                            + config.execution.entry_cost_bps() * abs(new_position))

        net_bps = bar_return_bps - cost_bps
        cum_pnl_bps += net_bps

        records.append({
            "timestamp": bars.index[i],
            "close": price,
            "factor": factor_now,
            "signal": signal_now,
            "desired_pos": desired_pos,
            "position": new_position,
            "bar_return_bps": bar_return_bps,
            "cost_bps": cost_bps,
            "net_pnl_bps": net_bps,
            "cum_pnl_bps": cum_pnl_bps,
            "stop_triggered": stop_triggered,
            "exit_triggered": exit_triggered,
        })

        # Update state for next bar
        if abs(new_position) > 1e-8 and abs(position) < 1e-8:
            # New entry
            entry_price = price
            entry_bar = i
            config.stop_loss.reset()
            if isinstance(config.stop_loss, ATRStopLoss) and atr_series is not None:
                config.stop_loss.set_atr(float(atr_series.iloc[i]) if not np.isnan(atr_series.iloc[i]) else 0)
        elif abs(new_position) < 1e-8:
            entry_price = 0.0
            entry_bar = i
            trade_pnl_bps = 0.0
        position = new_position

    df = pd.DataFrame(records)
    if not df.empty:
        df.set_index("timestamp", inplace=True)

    # For trade-count bars, compute actual avg bar_seconds from data
    effective_bar_seconds = config.bar_seconds
    if config.bar_mode == "trade_count" and "bar_duration_sec" in bars.columns:
        effective_bar_seconds = int(bars["bar_duration_sec"].median())
    metrics = _compute_metrics(df, config, effective_bar_seconds)
    all_params = {
        "factor": factor_name,
        "signal_flip": config.signal_flip,
        **config.signal.params(),
        **config.sizer.params(),
        **config.stop_loss.params(),
        **config.exit_rule.params(),
        **config.execution.params(),
        "initial_capital": config.initial_capital,
        "bar_seconds": effective_bar_seconds,
        "bar_mode": config.bar_mode,
        "trade_source": config.trade_source,
        "trades_per_bar": config.trades_per_bar if config.bar_mode == "trade_count" else 0,
    }

    return BacktestResult(trades=df, metrics=metrics, params=all_params, factor_name=factor_name)


def _compute_metrics(df: pd.DataFrame, config: BacktestConfig,
                     effective_bar_seconds: int = 60) -> Dict[str, Any]:
    """Compute comprehensive investment metrics from trade log."""
    if df.empty or len(df) < 2:
        return _empty_metrics()

    pnl = df["net_pnl_bps"]
    returns_pct = pnl / 10000.0  # bps → decimal

    cum = (1 + returns_pct).cumprod()
    total_return_pct = (cum.iloc[-1] - 1) * 100

    n = len(pnl)
    periods_per_year = 365 * 24 * 3600 / effective_bar_seconds
    years = n / periods_per_year

    ann_return = (cum.iloc[-1] ** (1 / max(years, 1e-6)) - 1) * 100 if cum.iloc[-1] > 0 else -100.0

    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    max_dd_pct = drawdown.min() * 100

    std_ret = returns_pct.std()
    mean_ret = returns_pct.mean()
    sharpe = mean_ret / (std_ret + 1e-12) * np.sqrt(periods_per_year)

    downside = returns_pct[returns_pct < 0]
    down_std = downside.std() if len(downside) > 1 else 1e-12
    sortino = mean_ret / (down_std + 1e-12) * np.sqrt(periods_per_year)

    calmar = ann_return / (abs(max_dd_pct) + 1e-12) if max_dd_pct != 0 else 0

    wins = (returns_pct > 0).sum()
    win_rate = wins / n * 100

    gross_profit = returns_pct[returns_pct > 0].sum()
    gross_loss = abs(returns_pct[returns_pct < 0].sum())
    profit_factor = gross_profit / (gross_loss + 1e-12)

    # Position-based stats
    pos = df["position"]
    pos_changes = (pos.diff().abs() > 1e-8).sum()
    long_bars = (pos > 1e-8).sum()
    short_bars = (pos < -1e-8).sum()
    flat_bars = n - long_bars - short_bars

    # Long / Short PnL decomposition
    long_pnl = df.loc[df["position"] > 1e-8, "net_pnl_bps"].sum()
    short_pnl = df.loc[df["position"] < -1e-8, "net_pnl_bps"].sum()

    # Total cost
    total_cost = df["cost_bps"].sum()

    # Turnover: mean(|position_change|)
    turnover = pos.diff().abs().mean() * 100  # as percentage

    # Holding period statistics
    hold_stats = _compute_holding_stats(df, effective_bar_seconds)

    return {
        "total_return_pct": round(total_return_pct, 4),
        "annualized_return_pct": round(ann_return, 4),
        "max_drawdown_pct": round(max_dd_pct, 4),
        "volatility_ann_pct": round(std_ret * np.sqrt(periods_per_year) * 100, 4),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "calmar_ratio": round(calmar, 4),
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(profit_factor, 4),
        "n_bars": n,
        "n_position_changes": int(pos_changes),
        "long_bars": int(long_bars),
        "short_bars": int(short_bars),
        "flat_bars": int(flat_bars),
        "net_pnl_bps": round(pnl.sum(), 2),
        "long_pnl_bps": round(long_pnl, 2),
        "short_pnl_bps": round(short_pnl, 2),
        "total_cost_bps": round(total_cost, 2),
        "avg_turnover_pct": round(turnover, 2),
        **hold_stats,
    }


def _compute_holding_stats(df: pd.DataFrame, bar_seconds: int) -> Dict[str, Any]:
    """Compute per-trade holding period statistics."""
    pos = df["position"]
    is_active = pos.abs() > 1e-8

    # Identify contiguous holding stretches
    switches = is_active.astype(int).diff().fillna(0)
    entries = switches == 1
    exits = switches == -1

    entry_indices = df.index[entries].tolist()
    exit_indices = df.index[exits].tolist()

    # If currently in a position at end, treat last bar as exit
    if is_active.iloc[-1]:
        exit_indices.append(df.index[-1])

    n_trades_round = min(len(entry_indices), len(exit_indices))
    if n_trades_round == 0:
        return {
            "n_round_trips": 0,
            "avg_hold_bars": 0,
            "avg_hold_minutes": 0.0,
            "median_hold_bars": 0,
            "max_hold_bars": 0,
            "min_hold_bars": 0,
        }

    hold_bars = []
    for i in range(n_trades_round):
        entry_t = entry_indices[i]
        exit_t = exit_indices[i]
        n_bars_held = len(df.loc[entry_t:exit_t])
        hold_bars.append(n_bars_held)

    hold_arr = np.array(hold_bars)
    bar_min = bar_seconds / 60.0

    return {
        "n_round_trips": n_trades_round,
        "avg_hold_bars": round(float(hold_arr.mean()), 1),
        "avg_hold_minutes": round(float(hold_arr.mean()) * bar_min, 1),
        "median_hold_bars": int(np.median(hold_arr)),
        "max_hold_bars": int(hold_arr.max()),
        "min_hold_bars": int(hold_arr.min()),
    }


def _empty_metrics() -> Dict[str, Any]:
    keys = [
        "total_return_pct", "annualized_return_pct", "max_drawdown_pct",
        "volatility_ann_pct", "sharpe_ratio", "sortino_ratio", "calmar_ratio",
        "win_rate_pct", "profit_factor", "n_bars", "n_position_changes",
        "long_bars", "short_bars", "flat_bars", "net_pnl_bps",
        "long_pnl_bps", "short_pnl_bps", "total_cost_bps", "avg_turnover_pct",
        "n_round_trips", "avg_hold_bars", "avg_hold_minutes",
        "median_hold_bars", "max_hold_bars", "min_hold_bars",
    ]
    return {k: 0 for k in keys}
