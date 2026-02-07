"""
实时交易引擎：高频数据处理与信号生成。

设计原则：
1. 低延迟：使用环形缓冲区避免内存分配
2. 异步处理：WebSocket 数据流与策略计算分离
3. 增量计算：避免重复计算历史数据
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional, Callable, Dict, Any, List
from collections import deque
import json

from core.types import (
    Trade, OrderBook, OrderBookLevel, Signal, SignalType,
    MarketState, RingBuffer, Side,
)
from core.config import Config, StrategyConfig

logger = logging.getLogger(__name__)

try:
    import websockets
except ImportError:
    websockets = None


class IncrementalStats:
    """增量统计计算（避免每次遍历整个缓冲区）。"""
    
    def __init__(self, max_count: int):
        self._sum = 0.0
        self._sum_sq = 0.0
        self._count = 0
        self._max_count = max_count
        self._values = deque(maxlen=max_count)
    
    def update(self, value: float) -> None:
        """添加新值，移除旧值（如果超出窗口）。"""
        if len(self._values) == self._max_count:
            old = self._values[0]
            self._sum -= old
            self._sum_sq -= old * old
        
        self._values.append(value)
        self._sum += value
        self._sum_sq += value * value
        self._count = len(self._values)
    
    @property
    def mean(self) -> float:
        return self._sum / self._count if self._count > 0 else 0.0
    
    @property
    def variance(self) -> float:
        if self._count < 2:
            return 0.0
        mean = self.mean
        return self._sum_sq / self._count - mean * mean
    
    @property
    def std(self) -> float:
        return self.variance ** 0.5
    
    def percentile_rank(self, value: float) -> float:
        """计算 value 在历史数据中的分位。"""
        if self._count == 0:
            return 50.0
        below = sum(1 for v in self._values if v < value)
        return below / self._count * 100


class MicrostructureCalculator:
    """
    微观结构指标计算器（增量版本）。
    
    优化：
    1. 使用增量统计避免全量计算
    2. 使用环形缓冲区限制内存
    3. 预分配内存避免 GC
    """
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        
        # 环形缓冲区存储最近交易
        buffer_size = max(1000, config.spread_lookback * 2)
        self._trades = RingBuffer(buffer_size)
        
        # 增量统计
        self._spread_stats = IncrementalStats(config.spread_lookback)
        self._impact_stats = IncrementalStats(config.imbalance_window)
        self._imbalance_stats = IncrementalStats(config.imbalance_window)
        
        # 窗口内累计
        self._window_buy_vol = 0.0
        self._window_sell_vol = 0.0
        self._window_price_start = 0.0
        self._window_start_ns = 0
        
        # 最新状态
        self._last_price = 0.0
        self._last_spread = 0.0
        self._last_orderbook: Optional[OrderBook] = None
    
    def on_trade(self, trade: Trade) -> None:
        """处理新的逐笔成交。"""
        self._trades.append(trade)
        self._last_price = trade.price
        
        # 更新窗口统计
        window_ms = self.config.impact_window_ms
        if trade.timestamp_ns - self._window_start_ns > window_ms * 1_000_000:
            # 新窗口
            self._reset_window(trade)
        else:
            # 累计到当前窗口
            if trade.side == Side.BUY:
                self._window_buy_vol += trade.amount
            else:
                self._window_sell_vol += trade.amount
    
    def on_orderbook(self, ob: OrderBook) -> None:
        """处理新的订单簿快照。"""
        self._last_orderbook = ob
        
        if ob.spread is not None:
            self._last_spread = ob.spread
            self._spread_stats.update(ob.spread)
    
    def _reset_window(self, trade: Trade) -> None:
        """重置统计窗口。"""
        # 计算上一窗口的指标
        if self._window_start_ns > 0:
            total_vol = self._window_buy_vol + self._window_sell_vol
            if total_vol > 0:
                imbalance = (self._window_buy_vol - self._window_sell_vol) / total_vol
                self._imbalance_stats.update(imbalance)
                
                price_change = (self._last_price - self._window_price_start) / (self._window_price_start + 1e-10)
                impact = abs(price_change) / (total_vol + 1e-10)
                self._impact_stats.update(impact)
        
        # 重置
        self._window_start_ns = trade.timestamp_ns
        self._window_price_start = trade.price
        self._window_buy_vol = trade.amount if trade.side == Side.BUY else 0
        self._window_sell_vol = trade.amount if trade.side == Side.SELL else 0
    
    def get_market_state(self) -> MarketState:
        """获取当前市场状态。"""
        total_vol = self._window_buy_vol + self._window_sell_vol
        imbalance = (self._window_buy_vol - self._window_sell_vol) / (total_vol + 1e-10) if total_vol > 0 else 0
        
        price_change = (self._last_price - self._window_price_start) / (self._window_price_start + 1e-10) if self._window_price_start > 0 else 0
        impact = abs(price_change) / (total_vol + 1e-10) if total_vol > 0 else 0
        
        spread_pct = self._spread_stats.percentile_rank(self._last_spread)
        
        return MarketState(
            timestamp_ns=time.time_ns(),
            price=self._last_price,
            spread=self._last_spread,
            spread_percentile=spread_pct,
            order_flow_imbalance=imbalance,
            price_impact=impact,
            vpin=abs(imbalance),  # 简化的 VPIN
            bid_depth=self._last_orderbook.bid_depth() if self._last_orderbook else 0,
            ask_depth=self._last_orderbook.ask_depth() if self._last_orderbook else 0,
        )


class ReversalSignalGenerator:
    """
    反转信号生成器。
    
    基于微观结构状态判断反转条件：
    1. 价格冲击 + 放量
    2. Spread 扩大
    3. 订单流不平衡
    4. 做市商补单（可选）
    """
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self._shock_detected = False
        self._shock_direction = 0
        self._shock_time_ns = 0
        self._shock_price = 0.0
    
    def evaluate(self, state: MarketState) -> Optional[Signal]:
        """评估是否产生信号。"""
        # 检测冲击
        shock = self._detect_shock(state)
        
        if shock:
            self._shock_detected = True
            self._shock_direction = shock
            self._shock_time_ns = state.timestamp_ns
            self._shock_price = state.price
            return None  # 冲击时不直接入场
        
        # 检查是否在冲击恢复期
        if self._shock_detected:
            signal = self._check_reversal(state)
            if signal:
                self._shock_detected = False
                return signal
            
            # 超时重置
            if state.timestamp_ns - self._shock_time_ns > self.config.refill_window_ms * 1_000_000:
                self._shock_detected = False
        
        return None
    
    def _detect_shock(self, state: MarketState) -> int:
        """检测流动性冲击。返回方向：1=上冲击，-1=下冲击，0=无。"""
        # 条件1：Spread 扩大
        if state.spread_percentile < self.config.spread_percentile_threshold:
            return 0
        
        # 条件2：订单流不平衡
        if abs(state.order_flow_imbalance) < self.config.imbalance_threshold:
            return 0
        
        # 条件3：价格冲击（通过 impact 判断）
        # 这里简化：如果 spread 扩大 + 不平衡高，就认为是冲击
        direction = 1 if state.order_flow_imbalance > 0 else -1
        
        logger.debug(f"Shock detected: dir={direction}, spread_pct={state.spread_percentile:.1f}, imb={state.order_flow_imbalance:.3f}")
        
        return direction
    
    def _check_reversal(self, state: MarketState) -> Optional[Signal]:
        """检查是否可以反转入场。"""
        # 条件1：Spread 恢复
        if state.spread_percentile > self.config.spread_recovery_threshold:
            return None
        
        # 条件2：价格未突破极值
        tolerance = self._shock_price * self.config.price_hold_tolerance_pct / 100
        if self._shock_direction == 1:  # 上冲击
            if state.price > self._shock_price + tolerance:
                return None  # 价格继续上涨，不反转
        else:  # 下冲击
            if state.price < self._shock_price - tolerance:
                return None  # 价格继续下跌，不反转
        
        # 产生反转信号
        signal_type = SignalType.SHORT if self._shock_direction == 1 else SignalType.LONG
        confidence = min(1.0, (100 - state.spread_percentile) / 50)  # Spread 恢复越多，置信度越高
        
        logger.info(f"Reversal signal: {signal_type.name}, confidence={confidence:.2f}")
        
        return Signal(
            timestamp_ns=state.timestamp_ns,
            signal_type=signal_type,
            confidence=confidence,
            price=state.price,
            reason=f"Reversal after {'-' if self._shock_direction < 0 else '+'}shock",
            metadata={
                "shock_price": self._shock_price,
                "shock_direction": self._shock_direction,
            },
        )


class RealtimeEngine:
    """
    实时交易引擎主类。
    
    功能：
    1. WebSocket 数据接收
    2. 微观结构计算
    3. 信号生成
    4. 回调通知
    """
    
    def __init__(
        self,
        config: Config,
        on_signal: Optional[Callable[[Signal], None]] = None,
    ):
        self.config = config
        self.on_signal = on_signal
        
        self._calculator = MicrostructureCalculator(config.strategy)
        self._signal_gen = ReversalSignalGenerator(config.strategy)
        
        self._running = False
        self._trade_count = 0
        self._signal_count = 0
    
    async def run(self):
        """运行实时引擎。"""
        if websockets is None:
            raise ImportError("websockets not installed")
        
        symbol = self.config.exchange.symbol.replace("/", "").lower()
        
        # 同时订阅 trades 和 depth
        streams = [
            f"{symbol}@aggTrade",
            f"{symbol}@depth20@100ms",
        ]
        ws_url = f"{self.config.exchange.ws_url}/{'/'.join(streams)}"
        
        self._running = True
        logger.info(f"Starting realtime engine: {ws_url}")
        
        while self._running:
            try:
                async with websockets.connect(ws_url) as ws:
                    logger.info("WebSocket connected")
                    
                    async for msg in ws:
                        if not self._running:
                            break
                        
                        await self._process_message(msg)
                        
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if self._running:
                    await asyncio.sleep(5)  # 重连延迟
    
    async def _process_message(self, msg: str):
        """处理 WebSocket 消息。"""
        try:
            data = json.loads(msg)
            
            if "e" not in data:
                return
            
            event_type = data["e"]
            
            if event_type == "aggTrade":
                trade = Trade(
                    timestamp_ns=data["T"] * 1_000_000,  # ms -> ns
                    price=float(data["p"]),
                    amount=float(data["q"]),
                    side=Side.SELL if data["m"] else Side.BUY,
                    trade_id=str(data["a"]),
                )
                self._on_trade(trade)
                
            elif event_type == "depthUpdate":
                ob = OrderBook(
                    timestamp_ns=data["E"] * 1_000_000,
                    bids=[OrderBookLevel(float(p), float(q)) for p, q in data.get("b", [])],
                    asks=[OrderBookLevel(float(p), float(q)) for p, q in data.get("a", [])],
                )
                self._on_orderbook(ob)
                
        except Exception as e:
            logger.error(f"Message processing error: {e}")
    
    def _on_trade(self, trade: Trade):
        """处理逐笔成交。"""
        self._trade_count += 1
        self._calculator.on_trade(trade)
        
        # 每 500 笔交易输出状态
        if self._trade_count % 500 == 0:
            state = self._calculator.get_market_state()
            logger.info(
                f"📊 Stats: trades={self._trade_count}, price={state.price:.2f}, "
                f"spread_pct={state.spread_percentile:.1f}%, imbalance={state.order_flow_imbalance:.3f}"
            )
        
        # 每笔交易后评估信号
        state = self._calculator.get_market_state()
        signal = self._signal_gen.evaluate(state)
        
        if signal and signal.confidence >= self.config.strategy.min_confidence:
            self._signal_count += 1
            if self.on_signal:
                self.on_signal(signal)
    
    def _on_orderbook(self, ob: OrderBook):
        """处理订单簿更新。"""
        self._calculator.on_orderbook(ob)
    
    def stop(self):
        """停止引擎。"""
        self._running = False
        logger.info(f"Engine stopped. Trades: {self._trade_count}, Signals: {self._signal_count}")
    
    @property
    def stats(self) -> Dict[str, Any]:
        """获取统计信息。"""
        return {
            "trade_count": self._trade_count,
            "signal_count": self._signal_count,
            "running": self._running,
        }
