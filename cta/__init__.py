"""
CTA 策略模块。

子模块：
- fvg: Fair Value Gap 策略（检测、特征 V1/V3、模型）
- microstructure: 微观结构高级检测器（基于 indicators/ 基础指标）

回测和标签相关功能已迁移至：
- backtest/metrics.py: 回测统计指标
- backtest/labeling.py: Triple Barrier 等标签方法
- backtest/engine.py: 通用回测引擎
"""
from .microstructure import (
    compute_tick_price_impact,
    compute_volume_weighted_impact,
    detect_large_impact_events,
    estimate_spread_from_trades,
    compute_spread_percentile,
    compute_trade_imbalance,
    compute_vpin,
    detect_aggressive_flow,
    detect_liquidity_shock,
    generate_reversal_signals,
)

__all__ = [
    "compute_tick_price_impact",
    "compute_volume_weighted_impact",
    "detect_large_impact_events",
    "estimate_spread_from_trades",
    "compute_spread_percentile",
    "compute_trade_imbalance",
    "compute_vpin",
    "detect_aggressive_flow",
    "detect_liquidity_shock",
    "generate_reversal_signals",
]
