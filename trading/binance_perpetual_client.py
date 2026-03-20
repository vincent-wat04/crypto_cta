"""
Binance USDT-M Perpetual exchange client (demo testnet + live).
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import ccxt
except ImportError:
    ccxt = None


class BinancePerpetualClient:
    """
    CCXT wrapper for Binance USDT-M Perpetual (futures).

    - Demo: testnet.binancefuture.com (Futures testnet)
    - Live: fapi.binance.com
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        demo: bool = True,
    ):
        if ccxt is None:
            raise ImportError("pip install ccxt")
        self.api_key = api_key
        self.api_secret = api_secret
        self.demo = demo

        self._exchange: Optional[Any] = None

    def _get_exchange(self):
        if self._exchange is None:
            self._exchange = ccxt.binanceusdm({
                "apiKey": self.api_key or None,
                "secret": self.api_secret or None,
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            })
            if self.demo:
                self._exchange.set_sandbox_mode(True)
                logger.info("Binance Perpetual: using demo/testnet")
            else:
                logger.info("Binance Perpetual: using live")
            self._exchange.load_markets()
        return self._exchange

    @property
    def exchange(self):
        return self._get_exchange()

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker (bid, ask, last)."""
        return self.exchange.fetch_ticker(symbol)

    def fetch_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Get order book for mid price."""
        return self.exchange.fetch_order_book(symbol, limit)

    def get_best_prices(self, symbol: str) -> tuple[float, float]:
        """
        Return (best_bid, best_ask) from the top of the order book.

        These are the correct prices for maker limit orders:
          - buy  limit → best_bid  (queues behind existing bids, does not cross spread)
          - sell limit → best_ask  (queues behind existing asks, does not cross spread)

        Placing at mid price risks rounding onto the opposite side on a
        1-tick spread (very common for SOL/USDC perp), resulting in an
        immediate taker fill.
        """
        ob = self.fetch_order_book(symbol, 1)
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        if not bids or not asks:
            ticker = self.fetch_ticker(symbol)
            last = float(ticker.get("last", 0))
            return last, last
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        return best_bid, best_ask

    def get_mid_price(self, symbol: str) -> float:
        """Mid price for reference only — do NOT use for limit order placement."""
        best_bid, best_ask = self.get_best_prices(symbol)
        return (best_bid + best_ask) / 2.0

    def create_limit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        reduce_only: bool = False,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Place limit order (maker)."""
        return self.exchange.create_order(
            symbol=symbol,
            type="limit",
            side=side,
            amount=amount,
            price=price,
            params={"reduceOnly": reduce_only, **(params or {})},
        )

    def create_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        reduce_only: bool = False,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Place market order (taker)."""
        return self.exchange.create_order(
            symbol=symbol,
            type="market",
            side=side,
            amount=amount,
            params={"reduceOnly": reduce_only, **(params or {})},
        )

    def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Cancel an order."""
        return self.exchange.cancel_order(order_id, symbol)

    def fetch_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Fetch order status."""
        return self.exchange.fetch_order(order_id, symbol)

    def fetch_agg_trades(
        self,
        symbol: str,
        limit: int = 1000,
        start_time_ms: Optional[int] = None,
        end_time_ms: Optional[int] = None,
        from_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch raw Futures aggTrades directly from Binance FAPI.

        Returned dicts are the native Binance payloads:
          a, p, q, f, l, T, m
        """
        params: Dict[str, Any] = {
            "symbol": self.exchange.market(symbol)["id"],
            "limit": limit,
        }
        if start_time_ms is not None:
            params["startTime"] = int(start_time_ms)
        if end_time_ms is not None:
            params["endTime"] = int(end_time_ms)
        if from_id is not None:
            params["fromId"] = int(from_id)
        return self.exchange.fapiPublicGetAggTrades(params)

    def fetch_historical_agg_trades_range(
        self,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int = 1000,
        sleep_every_n_requests: int = 10,
        sleep_seconds: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Fetch all Futures aggTrades in [start_time_ms, end_time_ms).

        Strategy:
        1. startTime to find the first batch in range
        2. paginate forward with fromId (much faster than repeated startTime scans)
        """
        all_records: List[Dict[str, Any]] = []
        n_requests = 0

        try:
            first_batch = self.fetch_agg_trades(
                symbol=symbol,
                start_time_ms=start_time_ms,
                limit=limit,
            )
            n_requests += 1
        except Exception as e:
            logger.warning("Error fetching first aggTrade batch: %s", e)
            return []

        if not first_batch:
            return []

        for record in first_batch:
            ts = int(record["T"])
            if ts >= end_time_ms:
                break
            all_records.append(record)

        last_id = int(first_batch[-1]["a"])

        while True:
            try:
                batch = self.fetch_agg_trades(
                    symbol=symbol,
                    from_id=last_id + 1,
                    limit=limit,
                )
                n_requests += 1
            except Exception as e:
                logger.warning("Error fetching aggTrades fromId=%s: %s", last_id + 1, e)
                time.sleep(1.0)
                continue

            if not batch:
                break

            hit_end = False
            for record in batch:
                ts = int(record["T"])
                if ts >= end_time_ms:
                    hit_end = True
                    break
                all_records.append(record)

            last_id = int(batch[-1]["a"])

            if hit_end or len(batch) < limit:
                break

            if sleep_every_n_requests > 0 and n_requests % sleep_every_n_requests == 0:
                time.sleep(sleep_seconds)

        logger.info(
            "Fetched %d historical perp aggTrades in %d requests [%s, %s)",
            len(all_records), n_requests, start_time_ms, end_time_ms,
        )
        return all_records

    def fetch_order_book_trades(self, symbol: str) -> List[Dict]:
        """Fetch recent trades (for aggTrades-like data)."""
        return self.exchange.fetch_trades(symbol, limit=1000)

    def fetch_balance(self) -> Dict[str, Any]:
        """Fetch account balance."""
        return self.exchange.fetch_balance()

    def fetch_positions(self, symbols: Optional[List[str]] = None) -> List[Dict]:
        """Fetch open positions."""
        return self.exchange.fetch_positions(symbols)

    def set_leverage(self, leverage: int, symbol: str) -> Dict[str, Any]:
        """Set leverage for symbol."""
        return self.exchange.set_leverage(leverage, symbol)
