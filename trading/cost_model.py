"""
Trading cost model for Binance USDC-M Perpetual Futures.

Fee schedule (VIP 0, lowest tier, standard):
  Maker:  0.0200%  =  2.0 bps   (positive — NOT a rebate at VIP 0)
  Taker:  0.0500%  =  5.0 bps

With BNB payment discount (10% off):
  Maker:  0.0180%  =  1.8 bps
  Taker:  0.0450%  =  4.5 bps

Source: https://www.binance.com/en/fee/futureFee (USDS-M Futures, VIP 0)

Note: VIP 4+ may receive maker rebates (negative fees).
      Promotions may temporarily set maker = 0%.
      This model defaults to VIP 0 standard rates.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostModel:
    """Binance USDC-M Perpetual fee model."""

    maker_fee_bps: float = 2.0     # VIP 0: positive fee (not rebate)
    taker_fee_bps: float = 5.0     # VIP 0

    def blended_fee_bps(self, maker_fill_rate: float = 0.7) -> float:
        """Weighted fee assuming a fraction of legs are maker."""
        return maker_fill_rate * self.maker_fee_bps + (1 - maker_fill_rate) * self.taker_fee_bps

    def round_trip_maker(self) -> float:
        return 2 * self.maker_fee_bps

    def round_trip_taker(self) -> float:
        return 2 * self.taker_fee_bps

    def round_trip_blended(self, maker_fill_rate: float = 0.7) -> float:
        return 2 * self.blended_fee_bps(maker_fill_rate)
