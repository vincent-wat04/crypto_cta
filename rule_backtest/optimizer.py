"""
Grid search optimizer: enumerate combinations of signal / position-sizing /
stop-loss / exit / execution parameters and run backtests in parallel.

Results are sorted by Sharpe ratio (or user-chosen metric).
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .signals import SIGNAL_CLASSES, BaseSignal
from .position_sizing import SIZER_CLASSES, BasePositionSizer
from .stop_loss import STOPLOSS_CLASSES, BaseStopLoss
from .exit_rules import EXIT_CLASSES, BaseExitRule
from .execution import EXECUTION_CLASSES, BaseExecution
from .engine import BacktestConfig, BacktestResult, run_backtest
from .report import generate_report

logger = logging.getLogger(__name__)


@dataclass
class SearchSpace:
    """Parameter grid for a single component."""
    signals: List[Dict[str, Any]] = field(default_factory=list)
    sizers: List[Dict[str, Any]] = field(default_factory=list)
    stop_losses: List[Dict[str, Any]] = field(default_factory=list)
    exit_rules: List[Dict[str, Any]] = field(default_factory=list)
    executions: List[Dict[str, Any]] = field(default_factory=list)


def default_search_space() -> SearchSpace:
    """Sensible default parameter grid for HF volume factors.

    Windows span 60 → 900 (1min bars → 15h lookback) to capture
    both short-term microstructure and longer mean-reversion regimes.
    """
    _WINDOWS = [60, 120, 300, 600, 900]

    # ZScore: window × threshold
    zscore_signals = [
        {"type": "zscore", "window": w, "threshold": t}
        for w in _WINDOWS for t in [1.5, 2.0]
    ]
    # Quantile: window × quantile pairs
    quantile_signals = [
        {"type": "quantile", "window": w, "upper_q": uq, "lower_q": 1 - uq}
        for w in _WINDOWS for uq in [0.8, 0.9]
    ]
    # MA Cross: fast/slow combos (fast ≈ 1/4–1/6 of slow)
    ma_signals = [
        {"type": "ma_cross", "fast_window": f, "slow_window": s}
        for f, s in [(5, 20), (10, 60), (20, 120), (30, 300), (60, 600)]
    ]
    # Bollinger: window × n_std
    boll_signals = [
        {"type": "bollinger", "window": w, "n_std": n}
        for w in [60, 120, 300, 600] for n in [1.5, 2.0]
    ]
    # Rank
    rank_signals = [
        {"type": "rank", "window": w, "upper_pct": 0.8, "lower_pct": 0.2}
        for w in _WINDOWS
    ]
    # Delta
    delta_signals = [
        {"type": "delta", "lookback": lb}
        for lb in [5, 10, 30, 60]
    ]
    # Dual MA Band
    dmab_signals = [
        {"type": "dual_ma_band", "fast_span": f, "slow_span": s,
         "band_window": bw, "band_mult": 1.0}
        for f, s, bw in [(5, 20, 40), (10, 60, 120), (20, 120, 300)]
    ]

    return SearchSpace(
        signals=(zscore_signals + quantile_signals + ma_signals
                 + boll_signals + rank_signals + delta_signals + dmab_signals),
        sizers=[
            {"type": "fixed", "size_units": 1.0},
            {"type": "vol_target", "target_vol_bps": 50.0, "vol_window": 60},
        ],
        stop_losses=[
            {"type": "none"},
            {"type": "fixed", "max_loss_bps": 30.0},
            {"type": "fixed", "max_loss_bps": 50.0},
            {"type": "trailing", "trail_bps": 20.0},
            {"type": "trailing", "trail_bps": 40.0},
            {"type": "time", "max_bars": 60},
            {"type": "time", "max_bars": 120},
        ],
        exit_rules=[
            {"type": "signal_reversal"},
            {"type": "signal_neutral"},
            {"type": "time_decay", "max_hold_bars": 30},
            {"type": "time_decay", "max_hold_bars": 60},
        ],
        executions=[
            {"type": "maker_first", "maker_fee_bps": 2.0, "taker_fee_bps": 5.0, "maker_fill_rate": fr}
            for fr in [0.5, 0.7, 0.9]
        ],
    )


def _build_signal(cfg: Dict[str, Any]) -> BaseSignal:
    t = cfg.pop("type")
    return SIGNAL_CLASSES[t](**cfg)


def _build_sizer(cfg: Dict[str, Any]) -> BasePositionSizer:
    t = cfg.pop("type")
    return SIZER_CLASSES[t](**cfg)


def _build_stoploss(cfg: Dict[str, Any]) -> BaseStopLoss:
    t = cfg.pop("type")
    return STOPLOSS_CLASSES[t](**cfg)


def _build_exit(cfg: Dict[str, Any]) -> BaseExitRule:
    t = cfg.pop("type")
    return EXIT_CLASSES[t](**cfg)


def _build_execution(cfg: Dict[str, Any]) -> BaseExecution:
    t = cfg.pop("type")
    return EXECUTION_CLASSES[t](**cfg)


def run_grid_search(
    bars: pd.DataFrame,
    factor: pd.Series,
    factor_name: str,
    space: Optional[SearchSpace] = None,
    sort_by: str = "sharpe_ratio",
    top_n: int = 20,
    bar_seconds: int = 60,
    initial_capital: float = 100_000.0,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Exhaustive grid search over all parameter combinations.

    Returns DataFrame of top_n results sorted by `sort_by` metric (descending).
    Also saves top-1 report if output_dir is provided.
    """
    if space is None:
        space = default_search_space()

    combos = list(itertools.product(
        space.signals, space.sizers, space.stop_losses,
        space.exit_rules, space.executions,
    ))
    logger.info("Grid search: %d combinations", len(combos))

    results: List[Dict[str, Any]] = []
    best_result: Optional[BacktestResult] = None
    best_metric: float = -1e18

    for idx, (sig_cfg, sz_cfg, sl_cfg, ex_cfg, exec_cfg) in enumerate(combos):
        # Deep-copy dicts to avoid mutation
        s = _build_signal(dict(sig_cfg))
        z = _build_sizer(dict(sz_cfg))
        sl = _build_stoploss(dict(sl_cfg))
        er = _build_exit(dict(ex_cfg))
        ec = _build_execution(dict(exec_cfg))

        config = BacktestConfig(
            signal=s, sizer=z, stop_loss=sl, exit_rule=er, execution=ec,
            initial_capital=initial_capital, bar_seconds=bar_seconds,
        )

        try:
            result = run_backtest(bars, factor, config, factor_name)
        except Exception as e:
            logger.warning("Combo #%d failed: %s", idx, e)
            continue

        row = {**result.params, **result.metrics}
        results.append(row)

        metric_val = result.metrics.get(sort_by, -1e18)
        if metric_val > best_metric:
            best_metric = metric_val
            best_result = result

        if (idx + 1) % 100 == 0:
            logger.info("  ... completed %d / %d", idx + 1, len(combos))

    if not results:
        logger.warning("No valid results from grid search.")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df.sort_values(sort_by, ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Save results
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / "grid_search_results.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Grid results saved to %s", csv_path)

        # Generate report for the best combo
        if best_result is not None:
            best_dir = output_dir / "best"
            generate_report(
                best_result.trades, best_result.metrics, best_result.params,
                initial_capital=initial_capital, output_dir=best_dir,
            )

    return df.head(top_n)
