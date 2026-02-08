"""
基础高频指标。

每个模块专注一类指标，返回 pd.Series（或 pd.DataFrame）。
所有指标接受统一的 trades DataFrame（列：timestamp, price, amount, side, ...）。
"""
from .price_impact import (
    tick_price_impact,
    volume_weighted_impact,
    kyles_lambda,
)
from .spread import (
    trade_diff_spread,
    roll_spread,
    effective_spread,
    spread_percentile,
)
from .order_flow import (
    trade_imbalance,
    vpin,
    aggressive_flow_events,
    flow_toxicity,
)
from .volume_profile import (
    taker_volume_corr,
    taker_volume_autocorr,
    taker_volume_skewness,
    buy_sell_volume_ratio,
    large_trade_ratio,
)
from .orderbook_pressure import (
    vwap_pressure,
    depth_imbalance,
    weighted_depth_slope,
)
from .taker_flow import (
    group_taker_orders,
    avg_levels_swept,
    large_taker_ratio as taker_large_ratio,
    taker_imbalance,
    sweep_depth_autocorr,
    impact_efficiency,
    taker_arrival_rate,
    taker_size_skewness,
    iceberg_score,
)
from .orderbook_lifecycle import (
    depth_change_rate,
    refill_frequency,
    cancel_rate,
    depth_resilience,
    level_thickness_profile,
)
from .trade_microstructure import (
    avg_fill_size,
    level_volume_profile,
    trade_side_autocorrelation,
    trade_intensity,
    vwap_deviation,
    compute_trade_microstructure,
)
