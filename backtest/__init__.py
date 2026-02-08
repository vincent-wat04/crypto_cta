"""
回测框架：Triple Barrier 标签 + 通用回测引擎 + 动态退出。
"""
from .labeling import triple_barrier_labels, validate_labels
from .engine import backtest_single_indicator, backtest_composite
from .metrics import compute_backtest_metrics
from .dynamic_exit import (
    atr_exit_params,
    volatility_percentile_exit,
    chandelier_exit_params,
    simulate_exit,
    ExitParams,
)
