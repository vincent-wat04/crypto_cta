"""
CTA 策略模块。

子模块：
- fvg: Fair Value Gap 策略（检测、特征、模型）
- microstructure: 微观结构指标（price impact、spread、order flow、liquidity）
- metrics: 准确性评估
- backtest: 统一回测引擎
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
