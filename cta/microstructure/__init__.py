"""
微观结构分析模块：基于逐笔成交和订单簿数据。

NOTE: 底层指标已迁移到 indicators/base/。
此模块保留高层封装（流动性检测、反转信号生成）。
底层函数从 indicators/ 重新导出，保持向后兼容。

核心流程：
1. Price Impact: indicators.base.price_impact
2. Spread: indicators.base.spread
3. Order Flow: indicators.base.order_flow
4. Taker Flow: indicators.base.taker_flow
5. Liquidity Shock: cta.microstructure.liquidity_detector
"""

# ── 从 indicators/ 重新导出底层函数（向后兼容） ──
from indicators.base.price_impact import tick_price_impact, volume_weighted_impact, kyles_lambda
from indicators.base.spread import trade_diff_spread, roll_spread, effective_spread, spread_percentile
from indicators.base.order_flow import trade_imbalance, vpin, flow_toxicity
from indicators.base.taker_flow import group_taker_orders

# ── 保留 cta 特有的高层检测器 ──
# 旧的低层实现保留以兼容现有脚本
from .price_impact import (
    compute_tick_price_impact,
    compute_volume_weighted_impact,
    compute_kyle_lambda,
    detect_large_impact_events,
)
from .spread_estimator import (
    estimate_spread_from_trades,
    compute_spread_percentile,
    detect_spread_expansion,
)
from .order_flow import (
    compute_trade_imbalance,
    compute_vpin,
    detect_aggressive_flow,
)
from .liquidity_detector import (
    detect_liquidity_shock,
    detect_mm_refill,
    analyze_liquidity_cycle,
    generate_reversal_signals,
    LiquidityState,
)

__all__ = [
    # indicators/ re-exports
    "tick_price_impact", "volume_weighted_impact", "kyles_lambda",
    "trade_diff_spread", "roll_spread", "effective_spread", "spread_percentile",
    "trade_imbalance", "vpin", "flow_toxicity",
    "group_taker_orders",
    # legacy
    "compute_tick_price_impact", "compute_volume_weighted_impact",
    "compute_kyle_lambda", "detect_large_impact_events",
    "estimate_spread_from_trades", "compute_spread_percentile", "detect_spread_expansion",
    "compute_trade_imbalance", "compute_vpin", "detect_aggressive_flow",
    # high-level detectors
    "detect_liquidity_shock", "detect_mm_refill",
    "analyze_liquidity_cycle", "generate_reversal_signals",
    "LiquidityState",
]
