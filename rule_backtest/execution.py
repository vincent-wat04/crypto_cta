"""
Execution cost simulation.

Binance USDC-M Perpetual Futures (VIP 0):
  Maker:  0.0200%  =  2.0 bps  (positive fee, no rebate at VIP 0)
  Taker:  0.0500%  =  5.0 bps
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any


class BaseExecution(ABC):
    @abstractmethod
    def entry_cost_bps(self) -> float: ...

    @abstractmethod
    def exit_cost_bps(self) -> float: ...

    def round_trip_cost_bps(self) -> float:
        return self.entry_cost_bps() + self.exit_cost_bps()

    @abstractmethod
    def params(self) -> Dict[str, Any]: ...


@dataclass
class MakerFirstExecution(BaseExecution):
    """
    Attempt limit (maker) first; if not filled, taker.
    maker_fill_rate controls the blended cost.
    """
    maker_fee_bps: float = 2.0     # VIP 0 positive fee
    taker_fee_bps: float = 5.0     # VIP 0
    maker_fill_rate: float = 0.7

    def _blended_fee(self) -> float:
        return (self.maker_fill_rate * self.maker_fee_bps
                + (1 - self.maker_fill_rate) * self.taker_fee_bps)

    def entry_cost_bps(self) -> float:
        return self._blended_fee()

    def exit_cost_bps(self) -> float:
        return self._blended_fee()

    def params(self) -> Dict[str, Any]:
        return {
            "execution": "maker_first",
            "maker_fee_bps": self.maker_fee_bps,
            "taker_fee_bps": self.taker_fee_bps,
            "maker_fill_rate": self.maker_fill_rate,
        }


@dataclass
class PureTakerExecution(BaseExecution):
    taker_fee_bps: float = 5.0

    def entry_cost_bps(self) -> float:
        return self.taker_fee_bps

    def exit_cost_bps(self) -> float:
        return self.taker_fee_bps

    def params(self) -> Dict[str, Any]:
        return {"execution": "pure_taker", "taker_fee_bps": self.taker_fee_bps}


EXECUTION_CLASSES = {
    "maker_first": MakerFirstExecution,
    "pure_taker": PureTakerExecution,
}
