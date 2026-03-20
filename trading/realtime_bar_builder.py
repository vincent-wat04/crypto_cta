"""
Real-time bar builder: accumulate FAPI aggTrades into OHLCV bars
with buy/sell volume decomposition.

Supports two modes:
  - time:        bar closes every `bar_seconds` (wall-clock)
  - trade_count: bar closes every `trades_per_bar` merged taker orders

Before counting, incoming aggTrades are streamed through a real-time
merge filter that combines multi-fill taker orders (same side,
consecutive, tight timestamps, monotonic prices).
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Deque, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Bar:
    """A single OHLCV bar with buy/sell decomposition and merged-trade metrics."""
    timestamp: datetime
    open: float = 0.0
    high: float = -np.inf
    low: float = np.inf
    close: float = 0.0
    volume: float = 0.0
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    n_trades: int = 0
    # Merged-trade metrics: populated once StreamingMerger is active
    buy_vwap_dist_sum: float = 0.0   # Σ(vwap - first_price) for buy orders
    sell_vwap_dist_sum: float = 0.0  # Σ(first_price - vwap) for sell orders
    buy_multi_fill_count: int = 0    # buy merged orders with >1 raw fill
    sell_multi_fill_count: int = 0   # sell merged orders with >1 raw fill

    def update(
        self,
        price: float,
        amount: float,
        side: str,
        vwap_dist: float = 0.0,
        n_fills: int = 1,
    ) -> None:
        if self.n_trades == 0:
            self.open = price
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += amount
        if side == "buy":
            self.buy_volume += amount
            self.buy_vwap_dist_sum += vwap_dist
            if n_fills > 1:
                self.buy_multi_fill_count += 1
        else:
            self.sell_volume += amount
            self.sell_vwap_dist_sum += vwap_dist
            if n_fills > 1:
                self.sell_multi_fill_count += 1
        self.n_trades += 1

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "buy_volume": self.buy_volume,
            "sell_volume": self.sell_volume,
            "n_trades": self.n_trades,
            "buy_vwap_dist_sum": self.buy_vwap_dist_sum,
            "sell_vwap_dist_sum": self.sell_vwap_dist_sum,
            "buy_multi_fill_count": self.buy_multi_fill_count,
            "sell_multi_fill_count": self.sell_multi_fill_count,
        }


# ═══════════════════════════════════════════════════════════
# Streaming aggTrade merger
# ═══════════════════════════════════════════════════════════

@dataclass
class _PendingMerge:
    """State for the in-progress taker order being assembled."""
    side: str = ""
    total_cost: float = 0.0
    total_amount: float = 0.0
    first_price: float = 0.0   # first fill price (proxy for top-of-book)
    last_price: float = 0.0
    last_timestamp: datetime = field(default_factory=lambda: datetime.min.replace(tzinfo=timezone.utc))
    first_timestamp: datetime = field(default_factory=lambda: datetime.min.replace(tzinfo=timezone.utc))
    n_fills: int = 0

    def reset(self) -> None:
        self.side = ""
        self.total_cost = 0.0
        self.total_amount = 0.0
        self.first_price = 0.0
        self.last_price = 0.0
        self.n_fills = 0

    @property
    def vwap(self) -> float:
        return self.total_cost / (self.total_amount + 1e-15) if self.total_amount > 0 else self.last_price


class StreamingMerger:
    """
    Real-time aggTrade merger.

    Buffers consecutive same-side trades with monotonic prices and tight
    timestamps.  When the chain breaks, emits the merged taker order
    (VWAP price, summed amount).
    """

    def __init__(self, max_gap_ms: float = 100.0):
        self.max_gap_ms = max_gap_ms
        self._pending = _PendingMerge()

    def on_trade(
        self, timestamp: datetime, price: float, amount: float, side: str,
    ) -> Optional[Dict]:
        """
        Feed a raw aggTrade.  Returns a merged-trade dict when a taker
        order boundary is detected; None if the trade was absorbed into
        the current pending group.
        """
        emit = None

        if self._pending.n_fills > 0:
            gap_ms = (timestamp - self._pending.last_timestamp).total_seconds() * 1000
            is_continuation = (
                side == self._pending.side
                and gap_ms < self.max_gap_ms
                and self._price_ok(side, price)
            )
            if not is_continuation:
                emit = self._flush()

        if self._pending.n_fills == 0:
            self._pending.side = side
            self._pending.first_timestamp = timestamp
            self._pending.first_price = price

        self._pending.total_cost += price * amount     # "cost" is internal only (→ value in output)
        self._pending.total_amount += amount
        self._pending.last_price = price
        self._pending.last_timestamp = timestamp
        self._pending.n_fills += 1

        return emit

    def flush(self) -> Optional[Dict]:
        """Force-emit whatever is buffered (e.g. at shutdown)."""
        if self._pending.n_fills > 0:
            return self._flush()
        return None

    def _flush(self) -> Dict:
        p = self._pending
        vwap = p.vwap
        # vwap_dist: how far the order's VWAP deviated from its first fill price
        # buy: paid more than initial = positive; sell: received less = positive
        if p.side == "buy":
            vwap_dist = vwap - p.first_price
        else:
            vwap_dist = p.first_price - vwap
        result = {
            "timestamp": p.first_timestamp,
            "price": vwap,
            "volume": p.total_amount,
            "side": p.side,
            "n_fills": p.n_fills,
            "first_price": p.first_price,
            "vwap_dist": vwap_dist,
        }
        p.reset()
        return result

    def _price_ok(self, side: str, price: float) -> bool:
        if side == "buy":
            return price >= self._pending.last_price
        return price <= self._pending.last_price


# ═══════════════════════════════════════════════════════════
# RealtimeBarBuilder (supports both time and trade-count)
# ═══════════════════════════════════════════════════════════

class RealtimeBarBuilder:
    """
    Accumulates trades into OHLCV bars.

    Modes:
      - "time":        bar closes every `bar_seconds`
      - "trade_count": bar closes every `trades_per_bar` merged taker orders

    Usage:
        builder = RealtimeBarBuilder(mode="trade_count", trades_per_bar=200)
        builder.on_bar_close = my_callback
        builder.on_trade(timestamp, price, amount, side)
    """

    def __init__(
        self,
        mode: str = "time",
        bar_seconds: int = 60,
        trades_per_bar: int = 200,
        history_size: int = 500,
    ):
        self.mode = mode
        self.bar_seconds = bar_seconds
        self.trades_per_bar = trades_per_bar
        self.history_size = history_size

        self._merger = StreamingMerger()
        self._current_bar: Optional[Bar] = None
        self._current_bar_epoch: int = 0
        self._merged_count_in_bar: int = 0
        self._history: Deque[Dict] = deque(maxlen=history_size)

        self.on_bar_close: Optional[Callable[[pd.DataFrame], None]] = None

    @property
    def n_bars(self) -> int:
        return len(self._history)

    @property
    def merged_count_in_current_bar(self) -> int:
        """
        Number of merged trades accumulated in the currently open trade-count bar.
        For time bars, this value is informational only.
        """
        return self._merged_count_in_bar

    @property
    def trades_per_bar_target(self) -> int:
        """Configured trades-per-bar threshold used to close a trade-count bar."""
        return self.trades_per_bar

    def on_trade(self, timestamp: datetime, price: float, amount: float, side: str) -> None:
        """
        Process a single raw aggTrade.

        The merger buffers it; when a merged taker order boundary is detected,
        the merged trade is fed into the bar builder.
        """
        merged = self._merger.on_trade(timestamp, price, amount, side)
        if merged is not None:
            self._on_merged_trade(merged)

    def _on_merged_trade(self, mt: Dict) -> None:
        """Process a completed merged taker order."""
        ts: datetime = mt["timestamp"]
        price: float = mt["price"]
        amount: float = mt["volume"]
        side: str = mt["side"]
        vwap_dist: float = mt.get("vwap_dist", 0.0)
        n_fills: int = mt.get("n_fills", 1)

        if self.mode == "time":
            self._handle_time_bar(ts, price, amount, side, vwap_dist, n_fills)
        else:
            self._handle_trade_count_bar(ts, price, amount, side, vwap_dist, n_fills)

    def _handle_time_bar(
        self, ts: datetime, price: float, amount: float, side: str,
        vwap_dist: float = 0.0, n_fills: int = 1,
    ) -> None:
        ts_epoch = int(ts.timestamp())
        bar_epoch = ts_epoch // self.bar_seconds * self.bar_seconds

        if self._current_bar is None:
            bar_dt = datetime.fromtimestamp(bar_epoch, tz=timezone.utc)
            self._current_bar = Bar(timestamp=bar_dt)
            self._current_bar_epoch = bar_epoch

        if bar_epoch > self._current_bar_epoch:
            self._close_bar()
            bar_dt = datetime.fromtimestamp(bar_epoch, tz=timezone.utc)
            self._current_bar = Bar(timestamp=bar_dt)
            self._current_bar_epoch = bar_epoch

        self._current_bar.update(price, amount, side, vwap_dist, n_fills)

    def _handle_trade_count_bar(
        self, ts: datetime, price: float, amount: float, side: str,
        vwap_dist: float = 0.0, n_fills: int = 1,
    ) -> None:
        if self._current_bar is None:
            self._current_bar = Bar(timestamp=ts)
            self._merged_count_in_bar = 0

        self._current_bar.update(price, amount, side, vwap_dist, n_fills)
        self._merged_count_in_bar += 1

        if self._merged_count_in_bar >= self.trades_per_bar:
            self._close_bar()
            self._current_bar = None
            self._merged_count_in_bar = 0

    def _close_bar(self) -> None:
        """Finalize the current bar and add to history."""
        if self._current_bar is None or self._current_bar.n_trades == 0:
            return

        bar_dict = self._current_bar.to_dict()
        self._history.append(bar_dict)

        if self.on_bar_close is not None:
            bars_df = self.get_bars()
            self.on_bar_close(bars_df)

    def get_bars(self) -> pd.DataFrame:
        """Return historical bars as a DataFrame."""
        if not self._history:
            return pd.DataFrame(columns=[
                "open", "high", "low", "close", "volume",
                "buy_volume", "sell_volume", "n_trades",
            ])
        df = pd.DataFrame(list(self._history))
        df.set_index("timestamp", inplace=True)
        return df

    def get_last_bar(self) -> Optional[Dict]:
        if self._history:
            return self._history[-1]
        return None
