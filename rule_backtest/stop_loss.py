"""
Stop-loss mechanisms applied per-bar during the backtest loop.

Each StopLoss checks whether a currently held position should be force-closed.
Returns True (stop triggered) / False per bar.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any


class BaseStopLoss(ABC):
    """Interface for stop-loss rules."""

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state when a new position is opened."""

    @abstractmethod
    def check(self, bar_idx: int, entry_price: float, current_price: float,
              current_pnl_bps: float, bars_held: int) -> bool:
        """Return True if stop should be triggered."""

    @abstractmethod
    def params(self) -> Dict[str, Any]: ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ═══════════════════════════════════════════════════════════
# No stop-loss (always False)
# ═══════════════════════════════════════════════════════════

@dataclass
class NoStopLoss(BaseStopLoss):
    def reset(self) -> None:
        pass

    def check(self, bar_idx, entry_price, current_price, current_pnl_bps, bars_held) -> bool:
        return False

    def params(self) -> Dict[str, Any]:
        return {"stoploss": "none"}


# ═══════════════════════════════════════════════════════════
# Fixed stop-loss: trigger when unrealized loss > threshold bps
# ═══════════════════════════════════════════════════════════

@dataclass
class FixedStopLoss(BaseStopLoss):
    max_loss_bps: float = 30.0

    def reset(self) -> None:
        pass

    def check(self, bar_idx, entry_price, current_price, current_pnl_bps, bars_held) -> bool:
        return current_pnl_bps < -self.max_loss_bps

    def params(self) -> Dict[str, Any]:
        return {"stoploss": "fixed", "max_loss_bps": self.max_loss_bps}


# ═══════════════════════════════════════════════════════════
# Trailing stop-loss: trigger when price retraces from peak
# ═══════════════════════════════════════════════════════════

@dataclass
class TrailingStopLoss(BaseStopLoss):
    trail_bps: float = 20.0
    _peak_pnl: float = field(default=0.0, init=False, repr=False)

    def reset(self) -> None:
        self._peak_pnl = 0.0

    def check(self, bar_idx, entry_price, current_price, current_pnl_bps, bars_held) -> bool:
        if current_pnl_bps > self._peak_pnl:
            self._peak_pnl = current_pnl_bps
        drawdown = self._peak_pnl - current_pnl_bps
        return drawdown > self.trail_bps

    def params(self) -> Dict[str, Any]:
        return {"stoploss": "trailing", "trail_bps": self.trail_bps}


# ═══════════════════════════════════════════════════════════
# Time stop-loss: force close after N bars
# ═══════════════════════════════════════════════════════════

@dataclass
class TimeStopLoss(BaseStopLoss):
    max_bars: int = 60

    def reset(self) -> None:
        pass

    def check(self, bar_idx, entry_price, current_price, current_pnl_bps, bars_held) -> bool:
        return bars_held >= self.max_bars

    def params(self) -> Dict[str, Any]:
        return {"stoploss": "time", "max_bars": self.max_bars}


# ═══════════════════════════════════════════════════════════
# ATR-based stop-loss: trigger when loss exceeds N * ATR
# ═══════════════════════════════════════════════════════════

@dataclass
class ATRStopLoss(BaseStopLoss):
    """
    Requires external ATR series to be injected via `set_atr`.
    Triggers when unrealized loss (bps) > multiplier * ATR_bps at entry bar.
    """
    multiplier: float = 2.0
    _atr_at_entry: float = field(default=0.0, init=False, repr=False)

    def set_atr(self, atr_bps: float) -> None:
        self._atr_at_entry = atr_bps

    def reset(self) -> None:
        self._atr_at_entry = 0.0

    def check(self, bar_idx, entry_price, current_price, current_pnl_bps, bars_held) -> bool:
        if self._atr_at_entry <= 0:
            return False
        return current_pnl_bps < -(self.multiplier * self._atr_at_entry)

    def params(self) -> Dict[str, Any]:
        return {"stoploss": "atr", "multiplier": self.multiplier}


# ═══════════════════════════════════════════════════════════
# Composite: any of multiple stops triggers
# ═══════════════════════════════════════════════════════════

@dataclass
class CompositeStopLoss(BaseStopLoss):
    """OR-combine multiple stop-loss rules."""
    stops: list = field(default_factory=list)

    def reset(self) -> None:
        for s in self.stops:
            s.reset()

    def check(self, bar_idx, entry_price, current_price, current_pnl_bps, bars_held) -> bool:
        return any(s.check(bar_idx, entry_price, current_price, current_pnl_bps, bars_held)
                   for s in self.stops)

    def params(self) -> Dict[str, Any]:
        return {"stoploss": "composite", "components": [s.params() for s in self.stops]}


STOPLOSS_CLASSES = {
    "none": NoStopLoss,
    "fixed": FixedStopLoss,
    "trailing": TrailingStopLoss,
    "time": TimeStopLoss,
    "atr": ATRStopLoss,
    "composite": CompositeStopLoss,
}
