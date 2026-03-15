"""
Position sizing strategies: map raw signal {-1,0,+1} to actual position size.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any

import pandas as pd


class BasePositionSizer(ABC):
    @abstractmethod
    def size(self, signal: pd.Series, close: pd.Series) -> pd.Series:
        """Return position sizes (signed float, e.g. -1.0 to +1.0) aligned to index."""

    @abstractmethod
    def params(self) -> Dict[str, Any]: ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ═══════════════════════════════════════════════════════════
# Fixed position: always ±1.0 when signal is active
# ═══════════════════════════════════════════════════════════

@dataclass
class FixedSizer(BasePositionSizer):
    size_units: float = 1.0

    def size(self, signal: pd.Series, close: pd.Series) -> pd.Series:
        return signal.astype(float) * self.size_units

    def params(self) -> Dict[str, Any]:
        return {"sizer": "fixed", "size_units": self.size_units}


# ═══════════════════════════════════════════════════════════
# Volatility-target: inverse volatility sizing
# ═══════════════════════════════════════════════════════════

@dataclass
class VolTargetSizer(BasePositionSizer):
    """
    Size = target_vol / realized_vol.
    Keeps dollar risk roughly constant across volatility regimes.
    """
    target_vol_bps: float = 50.0
    vol_window: int = 60
    max_leverage: float = 3.0

    def size(self, signal: pd.Series, close: pd.Series) -> pd.Series:
        ret = close.pct_change() * 10000  # bps
        rvol = ret.rolling(self.vol_window, min_periods=max(5, self.vol_window // 4)).std()
        rvol = rvol.clip(lower=1.0)
        raw = self.target_vol_bps / rvol
        raw = raw.clip(upper=self.max_leverage)
        return signal.astype(float) * raw

    def params(self) -> Dict[str, Any]:
        return {"sizer": "vol_target", "target_vol_bps": self.target_vol_bps,
                "vol_window": self.vol_window, "max_leverage": self.max_leverage}


# ═══════════════════════════════════════════════════════════
# Signal-proportional: size proportional to signal magnitude (for continuous signals)
# ═══════════════════════════════════════════════════════════

@dataclass
class SignalProportionalSizer(BasePositionSizer):
    """
    For continuous-valued signals (before discretizing to {-1,0,1}).
    Size = clip(signal_raw * scale, -max_size, max_size).
    Requires the signal generator to store raw_signal in a side channel.
    Falls back to sign(signal) * max_size if signal is discrete.
    """
    scale: float = 1.0
    max_size: float = 1.0

    def size(self, signal: pd.Series, close: pd.Series) -> pd.Series:
        scaled = signal.astype(float) * self.scale
        return scaled.clip(-self.max_size, self.max_size)

    def params(self) -> Dict[str, Any]:
        return {"sizer": "signal_prop", "scale": self.scale, "max_size": self.max_size}


SIZER_CLASSES = {
    "fixed": FixedSizer,
    "vol_target": VolTargetSizer,
    "signal_prop": SignalProportionalSizer,
}
