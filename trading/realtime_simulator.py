"""
Real-time trading simulator using Binance FAPI WebSocket.

Subscribes to perpetual aggTrades, computes V5 features incrementally,
generates signals at each 5min boundary, and simulates maker-first-then-taker.

Usage:
    simulator = RealtimeSimulator(config)
    await simulator.run()
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .cost_model import CostModel
from .position_controller import BasePositionController, FixedPositionController, PositionTarget

logger = logging.getLogger(__name__)

try:
    import websockets
except ImportError:
    websockets = None


@dataclass
class SimulatorConfig:
    """Configuration for real-time simulator."""
    symbol: str = "SOL/USDT"
    interval_sec: int = 300  # signal evaluation interval
    ws_url: str = "wss://fstream.binance.com/ws"

    # Model artifacts path (pickle with model, scaler, features)
    model_path: str = ""

    # Cost model
    maker_fee_bps: float = -2.0
    taker_fee_bps: float = 4.0
    maker_fill_rate: float = 0.7

    # Trade buffer for feature computation
    trade_buffer_size: int = 100_000  # keep last N trades for features
    warmup_bars: int = 60  # minimum 1s bars before generating signals


@dataclass
class SimulatorState:
    """Internal state of the simulator."""
    current_position: int = 0       # +1, -1, 0
    position_size: float = 0.0
    entry_price: float = 0.0
    cumulative_pnl: float = 0.0
    n_trades: int = 0
    n_signals: int = 0
    last_signal_time: float = 0.0
    trade_log: List[Dict] = field(default_factory=list)
    pnl_series: List[float] = field(default_factory=list)


class RealtimeSimulator:
    """
    Real-time trading simulator.

    Flow:
    1. WebSocket → aggTrade stream
    2. Buffer trades in deque
    3. Every interval_sec: resample → compute features → model predict → signal
    4. Position controller → execution simulation → PnL tracking
    """

    def __init__(
        self,
        config: SimulatorConfig,
        position_controller: Optional[BasePositionController] = None,
        on_signal: Optional[Callable] = None,
    ):
        if websockets is None:
            raise ImportError("pip install websockets")

        self.config = config
        self.position_controller = position_controller or FixedPositionController(
            max_position=1.0, threshold=0.0,
        )
        self.on_signal = on_signal
        self.cost_model = CostModel(
            maker_fee_bps=config.maker_fee_bps,
            taker_fee_bps=config.taker_fee_bps,
        )

        self._state = SimulatorState()
        self._trade_buffer: deque = deque(maxlen=config.trade_buffer_size)
        self._running = False
        self._last_bar_time = 0.0

        # Model artifacts (loaded lazily)
        self._model = None
        self._scaler = None
        self._selected_features: List[str] = []

    def load_model(self, path: Optional[str] = None):
        """Load trained model artifacts."""
        import pickle
        path = path or self.config.model_path
        if not path:
            raise ValueError("No model path specified")
        with open(path, "rb") as f:
            artifacts = pickle.load(f)
        self._model = artifacts["model"]
        self._scaler = artifacts["scaler"]
        self._selected_features = artifacts["selected_features"]
        logger.info("Model loaded: %d features", len(self._selected_features))

    async def run(self):
        """Main event loop: connect WebSocket, process trades."""
        if self._model is None:
            self.load_model()

        symbol_ws = self.config.symbol.replace("/", "").lower()
        ws_url = f"{self.config.ws_url}/{symbol_ws}@aggTrade"

        self._running = True
        logger.info("Starting real-time simulator: %s", ws_url)

        while self._running:
            try:
                async with websockets.connect(ws_url) as ws:
                    logger.info("WebSocket connected")
                    async for msg in ws:
                        if not self._running:
                            break
                        self._process_trade_message(msg)
            except Exception as e:
                logger.error("WebSocket error: %s", e)
                if self._running:
                    await asyncio.sleep(5)

    def _process_trade_message(self, msg: str):
        """Parse aggTrade message and buffer it."""
        try:
            data = json.loads(msg)
            if data.get("e") != "aggTrade":
                return

            trade = {
                "timestamp": pd.Timestamp(int(data["T"]), unit="ms", tz="UTC"),
                "price": float(data["p"]),
                "amount": float(data["q"]),
                "side": "sell" if data["m"] else "buy",
                "cost": float(data["p"]) * float(data["q"]),
                "agg_trade_id": int(data["a"]),
                "first_trade_id": int(data["f"]),
                "last_trade_id": int(data["l"]),
                "n_trades_in_agg": int(data["l"]) - int(data["f"]) + 1,
            }
            self._trade_buffer.append(trade)

            # Check if we've crossed a bar boundary
            trade_ts = trade["timestamp"].timestamp()
            bar_boundary = trade_ts // self.config.interval_sec * self.config.interval_sec

            if bar_boundary > self._last_bar_time and self._last_bar_time > 0:
                self._on_bar_complete(bar_boundary)

            if self._last_bar_time == 0:
                self._last_bar_time = bar_boundary

        except Exception as e:
            logger.error("Trade parse error: %s", e)

    def _on_bar_complete(self, bar_time: float):
        """Called when a 5min bar completes. Compute features, predict, execute."""
        self._last_bar_time = bar_time

        if len(self._trade_buffer) < 1000:
            return

        trades_df = pd.DataFrame(list(self._trade_buffer))

        # Compute V5 features
        try:
            from scripts.train_sol_v5 import compute_all_features_v5
            features_1s = compute_all_features_v5(trades_df, "1s")
        except Exception as e:
            logger.error("Feature computation error: %s", e)
            return

        if len(features_1s) < self.config.warmup_bars:
            logger.info("Warming up: %d / %d bars", len(features_1s), self.config.warmup_bars)
            return

        # Take last row as current features
        latest = features_1s.iloc[-1:]
        available = [f for f in self._selected_features if f in latest.columns]
        if len(available) < len(self._selected_features) * 0.8:
            logger.warning("Missing features: %d / %d", len(available), len(self._selected_features))
            return

        X = latest[available].values
        # Pad missing features with 0
        if len(available) < len(self._selected_features):
            full_X = np.zeros((1, len(self._selected_features)))
            for i, f in enumerate(self._selected_features):
                if f in available:
                    full_X[0, i] = X[0, available.index(f)]
            X = full_X

        X_scaled = self._scaler.transform(X)
        signal = float(self._model.predict(X_scaled)[0])

        self._state.n_signals += 1
        self._state.last_signal_time = bar_time

        # Position controller
        target = self.position_controller.compute_target(signal, self._state.current_position)

        # Execute position change
        current_price = float(trades_df["price"].iloc[-1])
        self._execute_position_change(target, current_price, bar_time, signal)

        if self.on_signal:
            self.on_signal({
                "time": bar_time,
                "signal": signal,
                "target": target,
                "price": current_price,
                "position": self._state.current_position,
                "cum_pnl": self._state.cumulative_pnl,
            })

    def _execute_position_change(
        self, target: PositionTarget, price: float, bar_time: float, signal: float
    ):
        """Simulate position change with maker-first-then-taker costs."""
        new_pos = target.direction
        old_pos = self._state.current_position

        if new_pos == old_pos:
            return

        # Close existing position PnL
        if old_pos != 0 and self._state.entry_price > 0:
            pnl_bps = (price - self._state.entry_price) / self._state.entry_price * 10000 * old_pos
        else:
            pnl_bps = 0.0

        # Cost for position change
        cost_bps = 0.0
        if new_pos != old_pos and (new_pos != 0 or old_pos != 0):
            cost_bps = 2 * (
                self.config.maker_fill_rate * self.cost_model.maker_fee_bps
                + (1 - self.config.maker_fill_rate) * self.cost_model.taker_fee_bps
            )

        net_pnl = pnl_bps - cost_bps
        self._state.cumulative_pnl += net_pnl
        self._state.pnl_series.append(net_pnl)

        if new_pos != old_pos:
            self._state.n_trades += 1

        trade_record = {
            "time": datetime.fromtimestamp(bar_time, tz=timezone.utc).isoformat(),
            "signal": signal,
            "old_pos": old_pos,
            "new_pos": new_pos,
            "price": price,
            "pnl_bps": pnl_bps,
            "cost_bps": cost_bps,
            "net_pnl": net_pnl,
            "cum_pnl": self._state.cumulative_pnl,
        }
        self._state.trade_log.append(trade_record)

        logger.info(
            "Trade #%d: %+d→%+d @ %.4f | signal=%.3f | pnl=%+.1f cost=%.1f net=%+.1f | cum=%+.1f",
            self._state.n_trades, old_pos, new_pos, price,
            signal, pnl_bps, cost_bps, net_pnl, self._state.cumulative_pnl,
        )

        self._state.current_position = new_pos
        self._state.entry_price = price if new_pos != 0 else 0.0

    def stop(self):
        """Stop the simulator."""
        self._running = False

    def get_results(self) -> Dict[str, Any]:
        """Get simulation results."""
        from .metrics import compute_investment_metrics
        pnl = pd.Series(self._state.pnl_series)
        periods_per_year = 365 * 24 * 3600 / self.config.interval_sec
        metrics = compute_investment_metrics(pnl, periods_per_year=periods_per_year)
        return {
            "metrics": metrics,
            "trade_log": self._state.trade_log,
            "n_trades": self._state.n_trades,
            "n_signals": self._state.n_signals,
            "cumulative_pnl": self._state.cumulative_pnl,
        }
