"""
类型定义：用于高频交易系统的数据结构。

兼容 Python 3.9+
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
import time


class Side(Enum):
    """交易方向。"""
    BUY = "buy"
    SELL = "sell"


class SignalType(Enum):
    """信号类型。"""
    LONG = 1   # 做多
    SHORT = -1  # 做空
    FLAT = 0   # 平仓/无信号


class OrderStatus(Enum):
    """订单状态。"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Trade:
    """逐笔成交数据。"""
    timestamp_ns: int  # 纳秒时间戳
    price: float
    amount: float
    side: Side
    trade_id: Optional[str] = None
    
    @property
    def timestamp_ms(self) -> int:
        return self.timestamp_ns // 1_000_000
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Trade":
        return cls(
            timestamp_ns=int(d.get("timestamp_ns", d.get("timestamp", 0) * 1_000_000)),
            price=float(d["price"]),
            amount=float(d["amount"]),
            side=Side(d["side"]) if isinstance(d["side"], str) else d["side"],
            trade_id=d.get("trade_id"),
        )


@dataclass
class OrderBookLevel:
    """订单簿单层。"""
    price: float
    quantity: float


@dataclass
class OrderBook:
    """订单簿快照。"""
    timestamp_ns: int
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    
    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None
    
    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None
    
    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None
    
    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None
    
    def bid_depth(self, levels: int = 5) -> float:
        return sum(lv.quantity for lv in self.bids[:levels])
    
    def ask_depth(self, levels: int = 5) -> float:
        return sum(lv.quantity for lv in self.asks[:levels])


@dataclass
class Signal:
    """交易信号。"""
    timestamp_ns: int
    signal_type: SignalType
    confidence: float
    price: float
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_long(self) -> bool:
        return self.signal_type == SignalType.LONG
    
    @property
    def is_short(self) -> bool:
        return self.signal_type == SignalType.SHORT


@dataclass
class Order:
    """订单。"""
    order_id: str
    symbol: str
    side: Side
    price: float
    quantity: float
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    created_at_ns: int = field(default_factory=lambda: time.time_ns())
    updated_at_ns: int = field(default_factory=lambda: time.time_ns())


@dataclass
class Position:
    """持仓。"""
    symbol: str
    side: Side
    quantity: float
    entry_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    created_at_ns: int = field(default_factory=lambda: time.time_ns())


@dataclass
class MarketState:
    """市场状态快照。"""
    timestamp_ns: int
    price: float
    spread: float
    spread_percentile: float
    order_flow_imbalance: float
    price_impact: float
    vpin: float
    bid_depth: float = 0.0
    ask_depth: float = 0.0


class RingBuffer:
    """环形缓冲区。"""
    
    def __init__(self, size: int):
        self._buffer = [None] * size
        self._size = size
        self._index = 0
        self._count = 0
    
    def append(self, item: Any) -> None:
        self._buffer[self._index] = item
        self._index = (self._index + 1) % self._size
        if self._count < self._size:
            self._count += 1
    
    def __len__(self) -> int:
        return self._count
    
    def __getitem__(self, idx: int) -> Any:
        if idx < 0:
            idx = self._count + idx
        if idx < 0 or idx >= self._count:
            raise IndexError("Index out of range")
        actual_idx = (self._index - self._count + idx) % self._size
        return self._buffer[actual_idx]
    
    def get_last(self, n: int) -> List[Any]:
        n = min(n, self._count)
        return [self[self._count - n + i] for i in range(n)]
    
    def clear(self) -> None:
        self._index = 0
        self._count = 0
