"""
Entry signal generators: convert a factor Series into directional signals {-1, 0, +1}.

Each signal class inherits from BaseSignal and implements `generate()`.
All signals are stateless (pure functions of the factor history up to each bar).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd


class BaseSignal(ABC):
    """Interface for signal generators."""

    @abstractmethod
    def generate(self, factor: pd.Series) -> pd.Series:
        """Return Series of {-1, 0, +1} aligned with factor index."""

    @abstractmethod
    def params(self) -> Dict[str, Any]:
        """Return dict of parameters for logging / optimization."""

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ═══════════════════════════════════════════════════════════
# Raw Factor Signal (sign of the factor value itself)
# ═══════════════════════════════════════════════════════════

@dataclass
class RawSignal(BaseSignal):
    """Use sign(factor) directly: positive → long, negative → short, ~0 → flat.

    dead_zone: absolute factor values below this are treated as flat (signal=0).
               Set to 0 for pure sign-based signal with no dead zone.
    """

    dead_zone: float = 0.0

    def generate(self, factor: pd.Series) -> pd.Series:
        sig = pd.Series(0, index=factor.index, dtype=np.int8)
        sig[factor > self.dead_zone] = 1
        sig[factor < -self.dead_zone] = -1
        return sig

    def params(self) -> Dict[str, Any]:
        return {"signal": "raw", "dead_zone": self.dead_zone}


@dataclass
class RawReversalSignal(BaseSignal):
    """Mean-reversion: positive factor → short, negative → long.

    dead_zone: absolute factor values below this are treated as flat (signal=0).
    """

    dead_zone: float = 0.0

    def generate(self, factor: pd.Series) -> pd.Series:
        sig = pd.Series(0, index=factor.index, dtype=np.int8)
        sig[factor > self.dead_zone] = -1
        sig[factor < -self.dead_zone] = 1
        return sig

    def params(self) -> Dict[str, Any]:
        return {"signal": "raw_reversal", "dead_zone": self.dead_zone}


# ═══════════════════════════════════════════════════════════
# Z-Score Signal
# ═══════════════════════════════════════════════════════════

@dataclass
class ZScoreSignal(BaseSignal):
    """Long when zscore > long_threshold, short when < -short_threshold.

    Supports asymmetric thresholds: if long_threshold / short_threshold
    are set they override the symmetric `threshold`.
    """

    window: int = 60
    threshold: float = 1.5
    long_threshold: float | None = None
    short_threshold: float | None = None

    def generate(self, factor: pd.Series) -> pd.Series:
        mu = factor.rolling(self.window, min_periods=max(5, self.window // 4)).mean()
        sigma = factor.rolling(self.window, min_periods=max(5, self.window // 4)).std()
        z = (factor - mu) / (sigma + 1e-12)
        lt = self.long_threshold if self.long_threshold is not None else self.threshold
        st = self.short_threshold if self.short_threshold is not None else self.threshold
        sig = pd.Series(0, index=factor.index, dtype=np.int8)
        sig[z > lt] = 1
        sig[z < -st] = -1
        return sig

    def params(self) -> Dict[str, Any]:
        lt = self.long_threshold if self.long_threshold is not None else self.threshold
        st = self.short_threshold if self.short_threshold is not None else self.threshold
        d: Dict[str, Any] = {"signal": "zscore", "window": self.window}
        if lt == st:
            d["threshold"] = lt
        else:
            d["long_threshold"] = lt
            d["short_threshold"] = st
        return d


# ═══════════════════════════════════════════════════════════
# Quantile Signal
# ═══════════════════════════════════════════════════════════

@dataclass
class QuantileSignal(BaseSignal):
    """Long when factor > upper rolling quantile, short when < lower."""

    window: int = 120
    upper_q: float = 0.8
    lower_q: float = 0.2

    def generate(self, factor: pd.Series) -> pd.Series:
        roll = factor.rolling(self.window, min_periods=max(10, self.window // 4))
        upper = roll.quantile(self.upper_q)
        lower = roll.quantile(self.lower_q)
        sig = pd.Series(0, index=factor.index, dtype=np.int8)
        sig[factor > upper] = 1
        sig[factor < lower] = -1
        return sig

    def params(self) -> Dict[str, Any]:
        return {"signal": "quantile", "window": self.window,
                "upper_q": self.upper_q, "lower_q": self.lower_q}


# ═══════════════════════════════════════════════════════════
# MA Crossover Signal
# ═══════════════════════════════════════════════════════════

@dataclass
class MACrossSignal(BaseSignal):
    """Long when fast MA > slow MA, short otherwise."""

    fast_window: int = 5
    slow_window: int = 20

    def generate(self, factor: pd.Series) -> pd.Series:
        fast = factor.rolling(self.fast_window, min_periods=1).mean()
        slow = factor.rolling(self.slow_window, min_periods=max(3, self.slow_window // 4)).mean()
        sig = pd.Series(0, index=factor.index, dtype=np.int8)
        sig[fast > slow] = 1
        sig[fast < slow] = -1
        return sig

    def params(self) -> Dict[str, Any]:
        return {"signal": "ma_cross", "fast": self.fast_window, "slow": self.slow_window}


# ═══════════════════════════════════════════════════════════
# Bollinger Band Signal (mean-reversion)
# ═══════════════════════════════════════════════════════════

@dataclass
class BollingerSignal(BaseSignal):
    """Short when above upper band, long when below lower band (mean reversion).

    Supports asymmetric bands: long_n_std / short_n_std override n_std.
    """

    window: int = 20
    n_std: float = 2.0
    long_n_std: float | None = None
    short_n_std: float | None = None

    def generate(self, factor: pd.Series) -> pd.Series:
        mu = factor.rolling(self.window, min_periods=max(5, self.window // 4)).mean()
        sigma = factor.rolling(self.window, min_periods=max(5, self.window // 4)).std()
        ln = self.long_n_std if self.long_n_std is not None else self.n_std
        sn = self.short_n_std if self.short_n_std is not None else self.n_std
        upper = mu + sn * sigma
        lower = mu - ln * sigma
        sig = pd.Series(0, index=factor.index, dtype=np.int8)
        sig[factor > upper] = -1   # mean-revert: short above upper
        sig[factor < lower] = 1    # long below lower
        return sig

    def params(self) -> Dict[str, Any]:
        ln = self.long_n_std if self.long_n_std is not None else self.n_std
        sn = self.short_n_std if self.short_n_std is not None else self.n_std
        d: Dict[str, Any] = {"signal": "bollinger", "window": self.window}
        if ln == sn:
            d["n_std"] = ln
        else:
            d["long_n_std"] = ln
            d["short_n_std"] = sn
        return d


# ═══════════════════════════════════════════════════════════
# Rolling Rank Signal
# ═══════════════════════════════════════════════════════════

@dataclass
class RankSignal(BaseSignal):
    """Long when rolling percentile rank > upper, short when < lower."""

    window: int = 60
    upper_pct: float = 0.8
    lower_pct: float = 0.2

    def generate(self, factor: pd.Series) -> pd.Series:
        rank = factor.rolling(self.window, min_periods=max(5, self.window // 4)).rank(pct=True)
        sig = pd.Series(0, index=factor.index, dtype=np.int8)
        sig[rank > self.upper_pct] = 1
        sig[rank < self.lower_pct] = -1
        return sig

    def params(self) -> Dict[str, Any]:
        return {"signal": "rank", "window": self.window,
                "upper_pct": self.upper_pct, "lower_pct": self.lower_pct}


# ═══════════════════════════════════════════════════════════
# Delta (Momentum) Signal
# ═══════════════════════════════════════════════════════════

@dataclass
class DeltaSignal(BaseSignal):
    """Long if factor increased over lookback, short if decreased."""

    lookback: int = 5

    def generate(self, factor: pd.Series) -> pd.Series:
        delta = factor.diff(self.lookback)
        sig = pd.Series(0, index=factor.index, dtype=np.int8)
        sig[delta > 0] = 1
        sig[delta < 0] = -1
        return sig

    def params(self) -> Dict[str, Any]:
        return {"signal": "delta", "lookback": self.lookback}


# ═══════════════════════════════════════════════════════════
# Threshold Signal (simple above/below static levels)
# ═══════════════════════════════════════════════════════════

@dataclass
class ThresholdSignal(BaseSignal):
    """Long when factor > upper_thr, short when < lower_thr."""

    upper_thr: float = 0.5
    lower_thr: float = -0.5

    def generate(self, factor: pd.Series) -> pd.Series:
        sig = pd.Series(0, index=factor.index, dtype=np.int8)
        sig[factor > self.upper_thr] = 1
        sig[factor < self.lower_thr] = -1
        return sig

    def params(self) -> Dict[str, Any]:
        return {"signal": "threshold", "upper": self.upper_thr, "lower": self.lower_thr}


# ═══════════════════════════════════════════════════════════
# Dual MA of Factor with Threshold Band
# ═══════════════════════════════════════════════════════════

@dataclass
class DualMABandSignal(BaseSignal):
    """
    Compute fast and slow EMA of factor; trade when the spread exceeds a
    rolling-std band.  Combines trend-following with noise filtering.
    """

    fast_span: int = 5
    slow_span: int = 20
    band_window: int = 40
    band_mult: float = 1.0

    def generate(self, factor: pd.Series) -> pd.Series:
        fast = factor.ewm(span=self.fast_span, min_periods=self.fast_span).mean()
        slow = factor.ewm(span=self.slow_span, min_periods=self.slow_span).mean()
        spread = fast - slow
        spread_std = spread.rolling(self.band_window, min_periods=max(5, self.band_window // 4)).std()
        band = self.band_mult * spread_std

        sig = pd.Series(0, index=factor.index, dtype=np.int8)
        sig[spread > band] = 1
        sig[spread < -band] = -1
        return sig

    def params(self) -> Dict[str, Any]:
        return {"signal": "dual_ma_band", "fast_span": self.fast_span,
                "slow_span": self.slow_span, "band_window": self.band_window,
                "band_mult": self.band_mult}


@dataclass
class V41Signal(BaseSignal):
    """
    Dual-factor delta-reversal signal from vwap_dist_sum_v41 strategy.

    Operates on bars that contain `buy_vwap_dist_sum` and `sell_vwap_dist_sum`.
    Uses two EMA-smoothed imbalance factors (short window fw1, long window fw2);
    generates signals on delta sign reversals filtered by adaptive MAD thresholds.

    - Long  entry: delta1 crosses from neg→pos AND |delta1| > threshold1
    - Short entry: delta2 crosses from pos→neg AND |delta2| > threshold2
    - Signal holds until opposite entry or explicit exit.

    The standard `generate(factor)` interface is NOT meaningful for this signal.
    Use `generate_from_bars(bars_df)` instead.
    """
    fw1: int = 8
    ema_w1: int = 10
    fw2: int = 10
    ema_w2: int = 10
    mad_window: int = 100
    mad_k1: float = 1.5   # threshold multiplier for long signal
    mad_k2: float = 1.0   # threshold multiplier for short signal

    name: str = "v41"

    def _compute_factors(self, bars: pd.DataFrame):
        """Return (factor1_ema, delta1, factor2_ema, delta2)."""
        bvd = bars["buy_vwap_dist_sum"]
        svd = bars["sell_vwap_dist_sum"]
        eps = 1e-10

        b1 = bvd.rolling(self.fw1).mean()
        s1 = svd.rolling(self.fw1).mean()
        f1 = (b1 - s1) / (b1.abs() + s1.abs() + eps)
        f1_ema = f1.ewm(span=self.ema_w1, min_periods=1).mean()
        d1 = f1_ema.diff()

        b2 = bvd.rolling(self.fw2).mean()
        s2 = svd.rolling(self.fw2).mean()
        f2 = (b2 - s2) / (b2.abs() + s2.abs() + eps)
        f2_ema = f2.ewm(span=self.ema_w2, min_periods=1).mean()
        d2 = f2_ema.diff()

        return f1_ema, d1, f2_ema, d2

    def _mad_threshold(self, delta: pd.Series, k: float, vol: pd.Series) -> pd.Series:
        w = self.mad_window
        med = delta.rolling(w).median()
        thr = (delta - med).abs().rolling(w).median() * k
        local_vol = vol.rolling(10).mean()
        avg_vol = local_vol.rolling(w).mean()
        thr = thr * np.sqrt(local_vol / (avg_vol + 1e-10))
        return thr.ffill().bfill()

    def generate_from_bars(self, bars: pd.DataFrame) -> pd.Series:
        """Primary interface for V41Signal — requires full bars DataFrame."""
        _, d1, _, d2 = self._compute_factors(bars)
        vol = bars.get("volume", pd.Series(1.0, index=bars.index))

        thr1 = self._mad_threshold(d1, self.mad_k1, vol)
        thr2 = self._mad_threshold(d2, self.mad_k2, vol)

        sign1 = np.sign(d1).replace(0, np.nan).ffill()
        sign2 = np.sign(d2).replace(0, np.nan).ffill()

        long_entry = (sign1.shift(1) == -1) & (sign1 == 1) & (d1.abs() > thr1)
        short_entry = (sign2.shift(1) == 1) & (sign2 == -1) & (d2.abs() > thr2)

        # Hold-forward position
        raw_pos = pd.Series(np.nan, index=bars.index)
        raw_pos[long_entry] = 1.0
        raw_pos[short_entry] = -1.0
        sig = raw_pos.ffill().fillna(0).astype(np.int8)
        return sig

    def generate(self, factor: pd.Series) -> pd.Series:
        """Fallback: not meaningful for V41, returns zeros. Use generate_from_bars."""
        return pd.Series(0, index=factor.index, dtype=np.int8)

    def params(self) -> Dict[str, Any]:
        return {
            "signal": "v41",
            "fw1": self.fw1, "ema_w1": self.ema_w1,
            "fw2": self.fw2, "ema_w2": self.ema_w2,
            "mad_window": self.mad_window,
            "mad_k1": self.mad_k1, "mad_k2": self.mad_k2,
        }


# ═══════════════════════════════════════════════════════════
# Registry for optimizer
# ═══════════════════════════════════════════════════════════

SIGNAL_CLASSES = {
    "raw": RawSignal,
    "raw_reversal": RawReversalSignal,
    "zscore": ZScoreSignal,
    "quantile": QuantileSignal,
    "ma_cross": MACrossSignal,
    "bollinger": BollingerSignal,
    "rank": RankSignal,
    "delta": DeltaSignal,
    "threshold": ThresholdSignal,
    "dual_ma_band": DualMABandSignal,
    "v41": V41Signal,
}
