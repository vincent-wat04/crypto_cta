"""
Maker-first-then-taker order execution.

1. Place limit order at same-side best price (maker)
     buy  → best_bid  (joins bid queue, does not cross spread)
     sell → best_ask  (joins ask queue, does not cross spread)
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
        best_bid, best_ask = self.client.get_best_prices(symbol)
        if best_bid <= 0 or best_ask <= 0:
            return {"filled": False, "filled_amount": 0, "error": "no best price"}

        # Limit price: same side as the order so we rest inside the spread (maker)
        #   buy  → best_bid:  below best_ask, will not cross
        #   sell → best_ask:  above best_bid, will not cross
        limit_price = best_bid if side == "buy" else best_ask
        mid = (best_bid + best_ask) / 2.0   # kept for fallback avg_price references only

        # 1. Place limit order at same-side best price
        limit_order = self.client.create_limit_order(
            symbol=symbol,
            side=side,
            amount=amount,
            price=limit_price,
            reduce_only=reduce_only,
        )
        order_id = limit_order.get("id")
        logger.info(
            f"Limit order placed: {order_id} {side} {amount} @ {limit_price} "
            f"(bid={best_bid} ask={best_ask})"
        )

        # 2. Wait for fill
        start = time.time()
        while time.time() - start < self.maker_timeout_sec:
            status = self.client.fetch_order(order_id, symbol)
            filled = float(status.get("filled", 0) or 0)
            remaining = float(status.get("remaining", amount) or amount)
            if remaining <= 0 or status.get("status") == "closed":
                # Filled as maker
                avg_price = float(status.get("average", limit_price) or limit_price)
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
            avg_price = float(status.get("average", limit_price) or limit_price)
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
        m_avg = float(market_order.get("average", limit_price) or limit_price)

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
            "avg_price": (filled * (status.get("average") or limit_price) + m_filled * m_avg) / total_filled
            if total_filled > 0
            else limit_price,
            "is_maker": False,
            "order_id": order_id,
            "cost_bps": cost_bps,
        }
