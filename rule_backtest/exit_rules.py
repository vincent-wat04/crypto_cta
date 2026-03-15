"""
Exit rules: determine when to close an existing position (beyond stop-loss).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np


class BaseExitRule(ABC):
    @abstractmethod
    def should_exit(self, bar_idx: int, position: int, signal_now: int,
                    factor_now: float, bars_held: int) -> bool:
        """Return True if the current position should be closed."""

    @abstractmethod
    def params(self) -> Dict[str, Any]: ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ═══════════════════════════════════════════════════════════
# Signal Reversal: exit when signal flips sign
# ═══════════════════════════════════════════════════════════

@dataclass
class SignalReversalExit(BaseExitRule):
    """Close position when signal changes direction (or goes to zero)."""

    def should_exit(self, bar_idx, position, signal_now, factor_now, bars_held) -> bool:
        if position == 0:
            return False
        return (position > 0 and signal_now < 0) or (position < 0 and signal_now > 0)

    def params(self) -> Dict[str, Any]:
        return {"exit": "signal_reversal"}


# ═══════════════════════════════════════════════════════════
# Signal Neutral: exit only when signal goes to zero
# ═══════════════════════════════════════════════════════════

@dataclass
class SignalNeutralExit(BaseExitRule):
    """Close only when signal is exactly zero; stay through flips."""

    def should_exit(self, bar_idx, position, signal_now, factor_now, bars_held) -> bool:
        if position == 0:
            return False
        return signal_now == 0

    def params(self) -> Dict[str, Any]:
        return {"exit": "signal_neutral"}


# ═══════════════════════════════════════════════════════════
# Time Decay: exit after N bars (complement to TimeStopLoss)
# ═══════════════════════════════════════════════════════════

@dataclass
class TimeDecayExit(BaseExitRule):
    max_hold_bars: int = 30

    def should_exit(self, bar_idx, position, signal_now, factor_now, bars_held) -> bool:
        if position == 0:
            return False
        return bars_held >= self.max_hold_bars

    def params(self) -> Dict[str, Any]:
        return {"exit": "time_decay", "max_hold_bars": self.max_hold_bars}


# ═══════════════════════════════════════════════════════════
# Factor Mean-Reversion: exit when factor returns to neutral zone
# ═══════════════════════════════════════════════════════════

@dataclass
class MeanReversionExit(BaseExitRule):
    """
    Exit when the factor value crosses back toward the mean.
    For longs: exit when factor drops below exit_threshold * sigma above mean.
    For shorts: exit when factor rises above -exit_threshold * sigma below mean.
    Uses running z-score of the factor.
    """
    exit_zscore: float = 0.0

    def should_exit(self, bar_idx, position, signal_now, factor_now, bars_held) -> bool:
        if position == 0:
            return False
        if np.isnan(factor_now):
            return False
        if position > 0:
            return factor_now <= self.exit_zscore
        return factor_now >= -self.exit_zscore

    def params(self) -> Dict[str, Any]:
        return {"exit": "mean_reversion", "exit_zscore": self.exit_zscore}


# ═══════════════════════════════════════════════════════════
# No explicit exit (rely on signal generator + position sizing + stop-loss)
# ═══════════════════════════════════════════════════════════

@dataclass
class NoExit(BaseExitRule):
    def should_exit(self, bar_idx, position, signal_now, factor_now, bars_held) -> bool:
        return False

    def params(self) -> Dict[str, Any]:
        return {"exit": "none"}


EXIT_CLASSES = {
    "signal_reversal": SignalReversalExit,
    "signal_neutral": SignalNeutralExit,
    "time_decay": TimeDecayExit,
    "mean_reversion": MeanReversionExit,
    "none": NoExit,
}
