"""
Maker-first-then-taker order execution.

1. Place limit order at mid price (maker)
2. Wait maker_timeout_sec
3. If not filled, cancel and place market order (taker)
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Literal, Optional

from .binance_perpetual_client import BinancePerpetualClient
from .cost_model import CostModel

logger = logging.getLogger(__name__)


class MakerFirstExecutor:
    """
    Execute orders: maker first at mid, then taker if not filled.
    """

    def __init__(
        self,
        client: BinancePerpetualClient,
        cost_model: CostModel,
        maker_timeout_sec: float = 5.0,
    ):
        self.client = client
        self.cost_model = cost_model
        self.maker_timeout_sec = maker_timeout_sec

    def execute(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        amount: float,
        reduce_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute order: limit at mid first, then market if not filled.

        Returns dict with:
          - filled: bool
          - filled_amount: float
          - avg_price: float
          - is_maker: bool (True if filled as maker)
          - order_id: str
          - cost_bps: float
        """
        mid = self.client.get_mid_price(symbol)
        if mid <= 0:
            return {"filled": False, "filled_amount": 0, "error": "no mid price"}

        # 1. Place limit order at mid
        limit_order = self.client.create_limit_order(
            symbol=symbol,
            side=side,
            amount=amount,
            price=mid,
            reduce_only=reduce_only,
        )
        order_id = limit_order.get("id")
        logger.info(f"Limit order placed: {order_id} {side} {amount} @ {mid}")

        # 2. Wait for fill
        start = time.time()
        while time.time() - start < self.maker_timeout_sec:
            status = self.client.fetch_order(order_id, symbol)
            filled = float(status.get("filled", 0) or 0)
            remaining = float(status.get("remaining", amount) or amount)
            if remaining <= 0 or status.get("status") == "closed":
                # Filled as maker
                avg_price = float(status.get("average", mid) or mid)
                cost_bps = self.cost_model.maker_cost_bps()
                logger.info(f"Filled as maker: {filled} @ {avg_price}, cost={cost_bps} bps")
                return {
                    "filled": True,
                    "filled_amount": filled,
                    "avg_price": avg_price,
                    "is_maker": True,
                    "order_id": order_id,
                    "cost_bps": cost_bps,
                }
            time.sleep(0.5)

        # 3. Cancel and place market order
        try:
            self.client.cancel_order(order_id, symbol)
        except Exception as e:
            logger.warning(f"Cancel failed (may be filled): {e}")

        # Check if partially filled
        status = self.client.fetch_order(order_id, symbol)
        filled = float(status.get("filled", 0) or 0)
        remaining = float(status.get("remaining", amount) or amount)

        if remaining <= 0:
            avg_price = float(status.get("average", mid) or mid)
            return {
                "filled": True,
                "filled_amount": filled,
                "avg_price": avg_price,
                "is_maker": True,
                "order_id": order_id,
                "cost_bps": self.cost_model.maker_cost_bps(),
            }

        # Place market order for remaining
        market_order = self.client.create_market_order(
            symbol=symbol,
            side=side,
            amount=remaining,
            reduce_only=reduce_only,
        )
        m_filled = float(market_order.get("filled", remaining) or remaining)
        m_avg = float(market_order.get("average", mid) or mid)

        total_filled = filled + m_filled
        # Mixed cost: maker portion + taker portion
        maker_pct = filled / total_filled if total_filled > 0 else 0
        cost_bps = (
            maker_pct * self.cost_model.maker_cost_bps()
            + (1 - maker_pct) * self.cost_model.taker_cost_bps()
        )

        logger.info(
            f"Filled: maker={filled}, taker={m_filled}, total={total_filled}, "
            f"cost_bps={cost_bps:.2f}"
        )

        return {
            "filled": True,
            "filled_amount": total_filled,
            "avg_price": (filled * (status.get("average") or mid) + m_filled * m_avg) / total_filled
            if total_filled > 0
            else mid,
            "is_maker": False,
            "order_id": order_id,
            "cost_bps": cost_bps,
        }
