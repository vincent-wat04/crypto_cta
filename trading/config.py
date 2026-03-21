"""
Unified trading configuration.

TradingConfig: base config for exchange connection, fees, execution, position sizing.
RuleRunnerConfig: extends TradingConfig with strategy/runner-specific fields.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TradingConfig:
    """
    Base configuration for Binance Perpetual trading.

    Covers: credentials, fees, execution, position sizing, leverage.
    Used by all trading components (live runner, simulator, executor).
    """

    # ── Mode ──
    mode: str = "simulate"   # "simulate" (paper) | "live" (real orders)

    # ── API credentials ──
    api_key: str = ""
    api_secret: str = ""

    # ── Symbol ──
    symbol: str = "SOL/USDC"

    # ── Fee structure (Binance USDC-M, VIP 0) ──
    maker_fee_bps: float = 2.0    # positive fee at VIP 0 (no rebate)
    taker_fee_bps: float = 5.0

    # ── Execution ──
    maker_fill_rate: float = 0.7   # simulated maker fill probability
    maker_timeout_sec: float = 5.0 # limit order timeout before switching to market

    # ── Leverage & Position sizing ──
    leverage: int = 1
    max_position_pct: float = 0.10  # max position as fraction of equity (0.10 = 10%)
    position_size_usd: float = 0.0  # explicit USD notional (overrides pct if > 0)

    # ── WebSocket ──
    ws_url: str = "wss://fstream.binance.com/ws"

    @property
    def is_live(self) -> bool:
        return self.mode == "live"

    @property
    def is_simulate(self) -> bool:
        return self.mode == "simulate"

    @property
    def blended_fee_bps(self) -> float:
        """Weighted average fee assuming maker_fill_rate."""
        return (self.maker_fill_rate * self.maker_fee_bps
                + (1 - self.maker_fill_rate) * self.taker_fee_bps)


@dataclass
class RuleRunnerConfig(TradingConfig):
    """
    Config for the rule-based strategy runner.
    Inherits all TradingConfig fields and adds strategy/runner specifics.
    """

    # ── Strategy ──
    factor_name: str = "vwap_dist_sum_imbalance"

    # ── Bar / timing ──
    bar_mode: str = "time"         # "time" or "trade_count" or "volume"
    bar_seconds: int = 60          # for time-based bars
    trades_per_bar: int = 200      # for trade-count bars
    volume_per_bar: float = 0.0    # for volume bars
    warmup_bars: int = 120
    history_bars: int = 500

    # ── Signal options ──
    signal_flip: bool = False       # invert signal for inverse predictors
    hold_through_flat: bool = False  # keep position when signal=0
