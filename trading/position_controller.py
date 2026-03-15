"""
Position controllers: determine position size based on signal.

- FixedPositionController: constant position size when signal exceeds threshold
- SignalAdjustedController: scale position by signal strength (with threshold gate)

All controllers share the same interface for extensibility.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class PositionTarget:
    """Output of position controller."""
    size: float        # position size (0 to max_position)
    direction: int     # +1 long, -1 short, 0 flat
    reason: str = ""


class BasePositionController(ABC):
    """Abstract position controller."""

    def __init__(self, max_position: float = 1.0, threshold: float = 0.0):
        self.max_position = max_position
        self.threshold = threshold

    @abstractmethod
    def compute_target(self, signal: float, current_position: float = 0.0) -> PositionTarget:
        """
        Compute target position given signal value.

        signal: model output (regression: bps predicted; classification: prob diff)
        current_position: current position (-1 to 1 normalized)
        """
        ...


class FixedPositionController(BasePositionController):
    """
    Fixed position sizing: full position when signal exceeds threshold, flat otherwise.
    No adjustment based on signal magnitude.
    """

    def compute_target(self, signal: float, current_position: float = 0.0) -> PositionTarget:
        if signal > self.threshold:
            return PositionTarget(size=self.max_position, direction=1, reason="signal > threshold")
        elif signal < -self.threshold:
            return PositionTarget(size=self.max_position, direction=-1, reason="signal < -threshold")
        else:
            return PositionTarget(size=0.0, direction=0, reason="signal within threshold")


class SignalAdjustedController(BasePositionController):
    """
    Signal-adjusted position sizing: position scaled by signal strength,
    but only when signal exceeds threshold (prevents noise-driven adjustments).

    Position = min(1, |signal| / scale_factor) * max_position
    Direction = sign(signal)
    """

    def __init__(
        self,
        max_position: float = 1.0,
        threshold: float = 0.0,
        scale_factor: float = 5.0,
        min_position: float = 0.1,
    ):
        super().__init__(max_position, threshold)
        self.scale_factor = scale_factor
        self.min_position = min_position

    def compute_target(self, signal: float, current_position: float = 0.0) -> PositionTarget:
        if abs(signal) <= self.threshold:
            return PositionTarget(size=0.0, direction=0, reason="signal within threshold")

        # Scale position by signal strength
        raw_size = min(1.0, abs(signal) / self.scale_factor)
        size = max(self.min_position, raw_size) * self.max_position
        direction = 1 if signal > 0 else -1

        return PositionTarget(
            size=size,
            direction=direction,
            reason=f"scaled: |signal|={abs(signal):.3f}, size={size:.3f}",
        )
