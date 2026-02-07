"""
回测框架：Triple Barrier 标签 + 通用回测引擎。
"""
from .labeling import triple_barrier_labels
from .engine import backtest_single_indicator, backtest_composite
from .metrics import compute_backtest_metrics
