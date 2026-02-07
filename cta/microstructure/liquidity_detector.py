"""
流动性检测：流动性冲击与做市商补单观测。

核心逻辑：
1. 流动性冲击：大单打出后，spread 扩大 + 价格冲击
2. 做市商补单：冲击后 spread 收窄，成交量回归正常
3. 反转信号：补单完成 + 价格未突破极值
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from .price_impact import compute_tick_price_impact, ImpactEvent
from .spread_estimator import estimate_spread_from_trades, compute_spread_percentile
from .order_flow import compute_trade_imbalance


class LiquidityState(Enum):
    """流动性状态。"""
    NORMAL = "normal"
    SHOCK = "shock"  # 流动性冲击中
    RECOVERY = "recovery"  # 恢复中（做市商补单）
    REVERSAL_READY = "reversal_ready"  # 准备反转


@dataclass
class LiquidityShockEvent:
    """流动性冲击事件。"""
    start_time: pd.Timestamp
    end_time: Optional[pd.Timestamp]
    direction: int  # 1 = 买方冲击（向上）, -1 = 卖方冲击（向下）
    peak_price: float
    price_change_pct: float
    peak_spread_percentile: float
    volume: float
    state: LiquidityState
    # 做市商补单信息
    refill_detected: bool = False
    refill_time: Optional[pd.Timestamp] = None
    spread_normalized: bool = False


def detect_liquidity_shock(
    trades: pd.DataFrame,
    impact_threshold_pct: float = 0.3,
    spread_percentile_threshold: float = 85.0,
    imbalance_threshold: float = 0.6,
    window_ms: int = 5000,
) -> List[LiquidityShockEvent]:
    """
    检测流动性冲击事件。
    
    条件（需同时满足）：
    1. 价格变动 > impact_threshold_pct
    2. Spread 扩大至历史 > spread_percentile_threshold 分位
    3. 订单流不平衡 > imbalance_threshold
    
    Args:
        trades: 逐笔成交数据
        impact_threshold_pct: 价格冲击阈值（%）
        spread_percentile_threshold: spread 分位阈值
        imbalance_threshold: 订单流不平衡阈值
        window_ms: 时间窗口（毫秒）
    
    Returns:
        List of LiquidityShockEvent
    """
    if trades.empty or len(trades) < 100:
        return []
    
    # 计算各指标
    impact_df = compute_tick_price_impact(trades, window_ms=window_ms)
    if impact_df.empty:
        return []
    
    spread = estimate_spread_from_trades(trades, method="trade_diff", window_trades=50)
    spread_pct = compute_spread_percentile(spread, lookback=500)
    imbalance = compute_trade_imbalance(trades, window_trades=50)
    
    # 合并指标
    impact_df = impact_df.set_index("timestamp")
    # 去除重复索引
    impact_df = impact_df[~impact_df.index.duplicated(keep='last')]
    spread = spread[~spread.index.duplicated(keep='last')]
    spread_pct = spread_pct[~spread_pct.index.duplicated(keep='last')]
    imbalance = imbalance[~imbalance.index.duplicated(keep='last')]
    
    # 对齐时间索引（使用最近的值）
    spread_aligned = spread.reindex(impact_df.index, method="ffill")
    spread_pct_aligned = spread_pct.reindex(impact_df.index, method="ffill")
    imbalance_aligned = imbalance.reindex(impact_df.index, method="ffill")
    
    events = []
    for ts, row in impact_df.iterrows():
        price_change = abs(row["price_change_pct"])
        current_spread_pct = spread_pct_aligned.get(ts, 0)
        current_imbalance = abs(imbalance_aligned.get(ts, 0))
        
        if (
            price_change >= impact_threshold_pct
            and current_spread_pct >= spread_percentile_threshold
            and current_imbalance >= imbalance_threshold
        ):
            direction = 1 if row["price_change_pct"] > 0 else -1
            events.append(LiquidityShockEvent(
                start_time=ts,
                end_time=None,
                direction=direction,
                peak_price=row.get("price", 0),  # 需要从 trades 获取
                price_change_pct=row["price_change_pct"],
                peak_spread_percentile=current_spread_pct,
                volume=row["volume"],
                state=LiquidityState.SHOCK,
            ))
    
    return events


def detect_mm_refill(
    trades: pd.DataFrame,
    shock_event: LiquidityShockEvent,
    refill_window_ms: int = 30000,
    spread_recovery_threshold: float = 50.0,
    price_hold_tolerance_pct: float = 0.1,
) -> LiquidityShockEvent:
    """
    检测做市商补单。
    
    条件（在 refill_window_ms 内）：
    1. Spread 分位回落至 < spread_recovery_threshold
    2. 价格未突破冲击时的极值（容忍 tolerance）
    
    Args:
        trades: 冲击后的逐笔成交数据
        shock_event: 流动性冲击事件
        refill_window_ms: 观察窗口（毫秒）
        spread_recovery_threshold: spread 恢复阈值（分位）
        price_hold_tolerance_pct: 价格容忍度（%）
    
    Returns:
        Updated LiquidityShockEvent with refill info
    """
    event = shock_event
    shock_time = event.start_time
    
    # 筛选冲击后的交易
    df = trades.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    after_shock = df[
        (df["timestamp"] > shock_time) & 
        (df["timestamp"] <= shock_time + pd.Timedelta(milliseconds=refill_window_ms))
    ]
    
    if after_shock.empty:
        return event
    
    # 计算 spread 分位
    spread = estimate_spread_from_trades(after_shock, method="trade_diff", window_trades=30)
    spread_pct = compute_spread_percentile(spread, lookback=100)
    
    # 检查 spread 是否恢复
    if spread_pct.empty:
        return event
    
    min_spread_pct = spread_pct.min()
    spread_recovered = min_spread_pct < spread_recovery_threshold
    
    # 检查价格是否突破极值
    prices = after_shock["price"]
    if event.direction == 1:  # 上冲击
        # 检查是否创新高
        tolerance = event.peak_price * (1 + price_hold_tolerance_pct / 100)
        price_held = prices.max() <= tolerance
    else:  # 下冲击
        # 检查是否创新低
        tolerance = event.peak_price * (1 - price_hold_tolerance_pct / 100)
        price_held = prices.min() >= tolerance
    
    if spread_recovered and price_held:
        event.refill_detected = True
        event.refill_time = spread_pct.idxmin() if not spread_pct.empty else None
        event.spread_normalized = True
        event.state = LiquidityState.REVERSAL_READY
    elif spread_recovered:
        event.spread_normalized = True
        event.state = LiquidityState.RECOVERY
    
    return event


def analyze_liquidity_cycle(
    trades: pd.DataFrame,
    impact_threshold_pct: float = 0.3,
    spread_percentile_threshold: float = 85.0,
    imbalance_threshold: float = 0.6,
    refill_window_ms: int = 30000,
    spread_recovery_threshold: float = 50.0,
    price_hold_tolerance_pct: float = 0.1,
) -> List[LiquidityShockEvent]:
    """
    分析完整的流动性周期：冲击 -> 补单 -> 可能反转。
    
    Returns:
        List of LiquidityShockEvent with full cycle info
    """
    # 1. 检测冲击
    shocks = detect_liquidity_shock(
        trades,
        impact_threshold_pct=impact_threshold_pct,
        spread_percentile_threshold=spread_percentile_threshold,
        imbalance_threshold=imbalance_threshold,
    )
    
    if not shocks:
        return []
    
    # 2. 对每个冲击检测补单
    results = []
    for shock in shocks:
        updated = detect_mm_refill(
            trades,
            shock,
            refill_window_ms=refill_window_ms,
            spread_recovery_threshold=spread_recovery_threshold,
            price_hold_tolerance_pct=price_hold_tolerance_pct,
        )
        results.append(updated)
    
    return results


def generate_reversal_signals(
    trades: pd.DataFrame,
    params: dict = None,
) -> pd.DataFrame:
    """
    生成反转信号。
    
    信号条件：
    1. 检测到流动性冲击
    2. 做市商补单完成
    3. 价格未突破极值
    
    Returns:
        DataFrame with columns [timestamp, signal, direction, confidence]
        signal: 1 = 反转信号, 0 = 无信号
        direction: 1 = 看涨（冲击后做空反转）, -1 = 看跌
    """
    params = params or {}
    
    # 分析流动性周期
    cycles = analyze_liquidity_cycle(
        trades,
        impact_threshold_pct=params.get("impact_threshold_pct", 0.3),
        spread_percentile_threshold=params.get("spread_percentile_threshold", 85.0),
        imbalance_threshold=params.get("imbalance_threshold", 0.6),
        refill_window_ms=params.get("refill_window_ms", 30000),
        spread_recovery_threshold=params.get("spread_recovery_threshold", 50.0),
        price_hold_tolerance_pct=params.get("price_hold_tolerance_pct", 0.1),
    )
    
    signals = []
    for cycle in cycles:
        if cycle.state == LiquidityState.REVERSAL_READY:
            signals.append({
                "timestamp": cycle.refill_time or cycle.start_time,
                "signal": 1,
                "direction": -cycle.direction,  # 反向
                "confidence": min(1.0, cycle.peak_spread_percentile / 100),
                "shock_time": cycle.start_time,
                "price_change_pct": cycle.price_change_pct,
            })
    
    if not signals:
        return pd.DataFrame(columns=["timestamp", "signal", "direction", "confidence"])
    
    return pd.DataFrame(signals)
