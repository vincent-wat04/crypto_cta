#!/usr/bin/env python3
"""
Live / paper-trade runner for the vwap_dist_sum_v41 strategy.

Best in-sample params:  fw1=8, ema_w1=10, fw2=10, ema_w2=10
Bar type:               trade_count, tpb=500 (merged trades)
Symbol:                 SOL/USDC perpetual

Usage (paper trade, no API keys):
    python scripts/run_live_vwap_v41.py --mode simulate

Usage (live trade on EC2):
    export BINANCE_API_KEY=xxx
    export BINANCE_API_SECRET=yyy
    python scripts/run_live_vwap_v41.py --mode live --leverage 2 --position_usd 500

EC2 deployment:
    nohup python scripts/run_live_vwap_v41.py --mode live 2>&1 | tee v41_live.log &
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trading.config import RuleRunnerConfig
from trading.realtime_bar_builder import RealtimeBarBuilder
from trading.cost_model import CostModel
from trading.telegram_notifier import TelegramNotifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vwap_v41_runner")

try:
    import websockets
except ImportError:
    websockets = None


# ═══════════════════════════════════════════════════════════
# Fill statistics tracker
# ═══════════════════════════════════════════════════════════

@dataclass
class FillStats:
    """
    Tracks actual vs estimated execution costs across all live orders.

    Updated asynchronously when each order's thread-pool future resolves.
    """
    n_orders: int = 0           # total completed orders
    n_maker: int = 0            # fully filled as maker (limit, no fallback)
    n_taker: int = 0            # fell back entirely to market order
    n_partial: int = 0          # partial maker + market
    n_failed: int = 0           # order returned error / exception
    total_actual_cost_bps: float = 0.0
    total_estimated_cost_bps: float = 0.0

    @property
    def actual_maker_rate(self) -> float:
        return self.n_maker / self.n_orders if self.n_orders else 0.0

    @property
    def avg_actual_cost_bps(self) -> float:
        return self.total_actual_cost_bps / self.n_orders if self.n_orders else 0.0

    @property
    def avg_estimated_cost_bps(self) -> float:
        return self.total_estimated_cost_bps / self.n_orders if self.n_orders else 0.0

    @property
    def cost_slippage_bps(self) -> float:
        """Actual − estimated per order (positive = paid more than expected)."""
        return self.avg_actual_cost_bps - self.avg_estimated_cost_bps

    def summary_line(self) -> str:
        return (
            f"FillStats | orders={self.n_orders} "
            f"maker={self.n_maker}({self.actual_maker_rate:.0%}) "
            f"taker={self.n_taker} partial={self.n_partial} failed={self.n_failed} | "
            f"avg_cost: actual={self.avg_actual_cost_bps:.2f}bps "
            f"est={self.avg_estimated_cost_bps:.2f}bps "
            f"slip={self.cost_slippage_bps:+.2f}bps"
        )


# ═══════════════════════════════════════════════════════════
# V41-specific runner (subclasses base pattern but is self-contained)
# ═══════════════════════════════════════════════════════════

class VwapV41Runner:
    """
    Real-time runner for the vwap_dist_sum_v41 dual-factor delta reversal strategy.

    Key differences from generic RuleStrategyRunner:
    - Uses V41Signal.generate_from_bars(bars_df) instead of factor → signal pipeline
    - Bar builder outputs buy_vwap_dist_sum / sell_vwap_dist_sum via merged trades
    - Implements hold_through_flat natively
    """

    def __init__(self, config: RuleRunnerConfig):
        if websockets is None:
            raise ImportError("pip install websockets")

        self.config = config
        self.cost_model = CostModel(
            maker_fee_bps=config.maker_fee_bps,
            taker_fee_bps=config.taker_fee_bps,
        )

        # Build trade-count bar builder (merged mode)
        self.bar_builder = RealtimeBarBuilder(
            mode="trade_count",
            trades_per_bar=config.trades_per_bar,
            history_size=config.history_bars,
        )
        self.bar_builder.on_bar_close = self._on_bar_close

        # Load V41 signal
        from rule_backtest.signals import V41Signal
        self.signal = V41Signal(
            fw1=config.v41_fw1,
            ema_w1=config.v41_ema_w1,
            fw2=config.v41_fw2,
            ema_w2=config.v41_ema_w2,
        )

        # Position state
        self._position: float = 0.0
        self._entry_price: float = 0.0
        self._bar_count: int = 0

        # PnL tracking (two tracks: estimated=immediate, actual=updated on fill)
        self._cum_pnl_estimated: float = 0.0   # uses blended_fee_bps, available instantly
        self._cum_pnl_actual: float = 0.0      # uses real fill result, async update

        # Trade log: list of dicts; live entries are patched in-place by fill callback
        self._trade_log: list = []

        # Execution statistics (live mode only)
        self._fill_stats: FillStats = FillStats()

        # Live exchange objects
        self._client = None
        self._executor = None
        self._running = False
        self._account_free: float = 0.0

        # Telegram notifier (optional — None means notifications disabled)
        self._tg: Optional[TelegramNotifier] = TelegramNotifier.from_env_optional()

        # Observability counters / timestamps
        self._ws_reconnect_count: int = 0
        self._msg_count: int = 0
        self._first_msg_seen: bool = False
        self._last_msg_ts: Optional[datetime] = None
        self._last_bar_ts: Optional[datetime] = None
        self._last_tg_heartbeat_ts: Optional[datetime] = None
        self._stale_alert_sent: bool = False
        self._heartbeat_task: Optional[asyncio.Task] = None

        # Warmup acceleration state:
        # - WS starts immediately and buffers live raw aggTrades
        # - a background REST loader fetches historical aggTrades to prewarm
        # - once historical warmup finishes, buffered WS trades are replayed in order
        self._run_start_ts: Optional[datetime] = None
        self._warmup_prefill_done: bool = False
        self._ws_buffer: List[Dict[str, Any]] = []
        self._buffered_raw_trade_count: int = 0
        self._historical_raw_trade_count: int = 0
        self._historical_merged_bar_count: int = 0
        self._prefill_task: Optional[asyncio.Task] = None

    # ─── Live init ───

    def _init_live(self) -> None:
        from trading.binance_perpetual_client import BinancePerpetualClient
        from trading.maker_first_executor import MakerFirstExecutor

        self._client = BinancePerpetualClient(
            api_key=self.config.api_key,
            api_secret=self.config.api_secret,
            demo=False,
        )
        self._executor = MakerFirstExecutor(
            client=self._client,
            cost_model=self.cost_model,
            maker_timeout_sec=self.config.maker_timeout_sec,
        )

        try:
            self._client.set_leverage(self.config.leverage, self.config.symbol)
            logger.info("Leverage set to %dx", self.config.leverage)
        except Exception as e:
            logger.warning("set_leverage failed: %s", e)

        # Fetch USDC or USDT balance
        try:
            balance = self._client.fetch_balance()
            for asset in ("USDC", "USDT"):
                info = balance.get(asset)
                if info:
                    total = float(info.get("total", 0) or 0)
                    free = float(info.get("free", 0) or 0)
                    logger.info("Balance (%s): total=%.2f free=%.2f", asset, total, free)
                    self._account_free = free
                    break
        except Exception as e:
            logger.error("fetch_balance failed: %s", e)
            self._account_free = 0.0

        # Check for existing positions
        try:
            positions = self._client.fetch_positions([self.config.symbol])
            for p in positions:
                contracts = float(p.get("contracts", 0) or 0)
                if contracts != 0:
                    side = p.get("side", "")
                    logger.warning(
                        "⚠ EXISTING POSITION: %s %.4f contracts (%s) — "
                        "runner will start with position=0 and may flip immediately",
                        self.config.symbol, contracts, side,
                    )
        except Exception as e:
            logger.warning("fetch_positions failed: %s", e)

    # ─── Main event loop ───

    async def run(self) -> None:
        if self.config.is_live:
            self._init_live()

        self._run_start_ts = datetime.now(timezone.utc)
        symbol_ws = self.config.symbol.replace("/", "").lower()
        ws_url = f"{self.config.ws_url}/{symbol_ws}@aggTrade"

        cfg_summary = (
            f"symbol={self.config.symbol}  mode={self.config.mode}\n"
            f"tpb={self.config.trades_per_bar}  warmup={self.config.warmup_bars}\n"
            f"fw1={self.config.v41_fw1} ema1={self.config.v41_ema_w1} "
            f"fw2={self.config.v41_fw2} ema2={self.config.v41_ema_w2}\n"
            f"leverage={self.config.leverage}  "
            f"pos_usd={self.config.position_size_usd or f'{self.config.max_position_pct:.0%} equity'}"
        )
        logger.info("Starting VwapV41Runner [%s mode]", self.config.mode)
        logger.info("WS: %s | tpb=%d | warmup=%d bars",
                    ws_url, self.config.trades_per_bar, self.config.warmup_bars)
        logger.info("Params: fw1=%d ema_w1=%d fw2=%d ema_w2=%d",
                    self.config.v41_fw1, self.config.v41_ema_w1,
                    self.config.v41_fw2, self.config.v41_ema_w2)

        if self._tg:
            await self._tg.send_startup(cfg_summary)

        self._running = True
        self._warmup_notified = False
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._prefill_task = asyncio.create_task(self._prefill_warmup_from_rest())

        while self._running:
            try:
                self._ws_reconnect_count += 1
                async with websockets.connect(
                    ws_url,
                    ping_interval=20,
                    ping_timeout=30,
                    close_timeout=5,
                ) as ws:
                    logger.info("WebSocket connected (attempt=%d)", self._ws_reconnect_count)
                    if self._tg:
                        await self._tg.send(
                            f"🔌 <b>WebSocket connected</b> "
                            f"(attempt={self._ws_reconnect_count})\n"
                            f"<code>{ws_url}</code>"
                        )
                    async for msg in ws:
                        if not self._running:
                            break
                        self._process_message(msg)
            except Exception as e:
                logger.error("WebSocket error: %s — reconnecting in 5s", e)
                if self._tg:
                    await self._tg.send_error("WebSocket", str(e))
                if self._running:
                    await asyncio.sleep(5)
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        if self._prefill_task:
            self._prefill_task.cancel()
            try:
                await self._prefill_task
            except asyncio.CancelledError:
                pass

    def _process_message(self, msg: str) -> None:
        try:
            data = json.loads(msg)
            if data.get("e") != "aggTrade":
                return
            self._msg_count += 1
            self._last_msg_ts = datetime.now(timezone.utc)
            if not self._first_msg_seen:
                self._first_msg_seen = True
                logger.info("First aggTrade received; stream is live")
                if self._tg:
                    asyncio.get_event_loop().create_task(
                        self._tg.send("✅ <b>First aggTrade received</b> — market data stream is live.")
                    )
            if self._stale_alert_sent:
                self._stale_alert_sent = False
                logger.info("Market data stream recovered")
                if self._tg:
                    asyncio.get_event_loop().create_task(
                        self._tg.send("🟢 <b>Market data recovered</b> — aggTrade stream resumed.")
                    )
            ts = datetime.fromtimestamp(int(data["T"]) / 1000, tz=timezone.utc)
            price = float(data["p"])
            amount = float(data["q"])
            side = "sell" if data["m"] else "buy"
            raw_trade = {
                "timestamp": ts,
                "price": price,
                "volume": amount,
                "side": side,
            }
            if self._warmup_prefill_done:
                self.bar_builder.on_trade(ts, price, amount, side)
            else:
                # Keep buffering live trades until REST prewarm has built the history.
                self._ws_buffer.append(raw_trade)
                self._buffered_raw_trade_count += 1
        except Exception as e:
            logger.error("parse error: %s", e)

    async def _prefill_warmup_from_rest(self) -> None:
        """
        Accelerate warmup using Binance historical aggTrades via REST in a background thread.

        Design:
        1. WS starts immediately and buffers fresh raw aggTrades.
        2. Historical aggTrades older than runner start time are fetched via REST.
        3. Historical trades are replayed into the same bar builder.
        4. Buffered WS trades are then replayed in timestamp order and live flow resumes.

        This avoids needing an external DB / queue. We just need:
        - a chronological cut-off (`self._run_start_ts`)
        - an in-memory WS buffer during prewarm
        """
        if self._run_start_ts is None:
            return

        # simulate mode may not have a client yet; build a lightweight public client if needed
        if self._client is None:
            from trading.binance_perpetual_client import BinancePerpetualClient
            self._client = BinancePerpetualClient(demo=False)

        target_bars = self.config.warmup_bars
        target_merged = target_bars * self.config.trades_per_bar
        # Heuristic from observed SOL/USDC flow on this project:
        # merged/raw ≈ 0.75-0.85. Use 1.35x raw buffer to improve odds of covering warmup.
        estimated_raw_needed = int(target_merged * 1.35)
        estimated_raw_per_hour = 6_000
        estimated_hours = max(2, min(24, estimated_raw_needed // estimated_raw_per_hour + 2))

        end_ms = int(self._run_start_ts.timestamp() * 1000)
        start_ms = int((self._run_start_ts - timedelta(hours=estimated_hours)).timestamp() * 1000)

        logger.info(
            "Starting REST prewarm: target_bars=%d target_merged≈%d estimated_raw≈%d lookback≈%dh",
            target_bars, target_merged, estimated_raw_needed, estimated_hours,
        )
        if self._tg:
            await self._tg.send(
                "⏪ <b>REST prewarm started</b>\n"
                f"target_bars=<code>{target_bars}</code>  "
                f"target_merged≈<code>{target_merged}</code>\n"
                f"estimated_raw≈<code>{estimated_raw_needed}</code>  "
                f"lookback≈<code>{estimated_hours}h</code>"
            )

        loop = asyncio.get_event_loop()
        records = await loop.run_in_executor(
            None,
            lambda: self._client.fetch_historical_agg_trades_range(
                symbol=self.config.symbol,
                start_time_ms=start_ms,
                end_time_ms=end_ms,
            ),
        )

        self._historical_raw_trade_count = len(records)
        if not records:
            logger.warning("REST prewarm returned no historical aggTrades; falling back to pure WS warmup")
            if self._tg:
                await self._tg.send(
                    "⚠️ <b>REST prewarm returned no data</b>\n"
                    "Falling back to pure WebSocket warmup."
                )
            self._warmup_prefill_done = True # so that all ws trades will go into bar builder
            return

        # Replay historical trades in strict time order.
        for record in records:
            ts = datetime.fromtimestamp(int(record["T"]) / 1000, tz=timezone.utc)
            price = float(record["p"])
            volume = float(record["q"])
            side = "sell" if record["m"] else "buy"
            self.bar_builder.on_trade(ts, price, volume, side)

        self._historical_merged_bar_count = self.bar_builder.n_bars

        # Replay buffered websocket trades newer than run start time.
        self._ws_buffer.sort(key=lambda x: x["timestamp"])
        replayed = 0
        for trade in self._ws_buffer:
            if trade["timestamp"] < self._run_start_ts:
                continue
            self.bar_builder.on_trade(
                trade["timestamp"], trade["price"], trade["volume"], trade["side"]
            )
            replayed += 1

        logger.info(
            "REST prewarm complete | historical_raw=%d historical_bars=%d replayed_ws_raw=%d final_bars=%d",
            self._historical_raw_trade_count,
            self._historical_merged_bar_count,
            replayed,
            self.bar_builder.n_bars,
        )

        # reset state and clear buffer before await for _tg
        # during await, ws协程重新工作，读取tcp缓冲数据
        # 要保证缓冲区数据直接进入 bar builder
        self._ws_buffer.clear()
        self._warmup_prefill_done = True

        if self._tg:
            await self._tg.send(
                "✅ <b>REST prewarm complete</b>\n"
                f"historical_raw=<code>{self._historical_raw_trade_count}</code>\n"
                f"historical_bars=<code>{self._historical_merged_bar_count}</code>\n"
                f"replayed_ws_raw=<code>{replayed}</code>\n"
                f"final_bars=<code>{self.bar_builder.n_bars}</code>"
            )

        if self.bar_builder.n_bars >= self.config.warmup_bars:
            logger.info(
                "Warmup satisfied immediately after REST prefill | bars=%d/%d",
                self.bar_builder.n_bars,
                self.config.warmup_bars,
            )
            if self._tg and not getattr(self, "_warmup_notified", False):
                self._warmup_notified = True
                await self._tg.send(
                    "🔥 <b>Warmup complete (REST prefill)</b>\n"
                    f"symbol=<code>{self.config.symbol}</code> bars=<code>{self.bar_builder.n_bars}</code>\n"
                    f"historical_raw=<code>{self._historical_raw_trade_count}</code> "
                    f"replayed_ws=<code>{replayed}</code>\n"
                    f"merged_progress=<code>{self.bar_builder.merged_count_in_current_bar}/"
                    f"{self.bar_builder.trades_per_bar_target}</code>"
                )

    def _on_bar_close(self, bars_df: pd.DataFrame) -> None:
        self._bar_count += 1
        self._last_bar_ts = datetime.now(timezone.utc)

        # Historical prefill is still building the bar history. Do not generate
        # signals / orders until both historical replay and buffered WS replay finish.
        if not self._warmup_prefill_done:
            if self._bar_count % 50 == 0:
                logger.info(
                    "Prefill: bars=%d/%d | hist_raw=%d buffered_ws=%d | merged_in_bar=%d/%d",
                    self._bar_count,
                    self.config.warmup_bars,
                    self._historical_raw_trade_count,
                    self._buffered_raw_trade_count,
                    self.bar_builder.merged_count_in_current_bar,
                    self.bar_builder.trades_per_bar_target,
                )
            return

        if self._bar_count < self.config.warmup_bars:
            if self._bar_count % 50 == 0:
                logger.info(
                    "Warmup: %d / %d | raw_trades=%d buffered_ws=%d hist_raw=%d "
                    "| merged_in_bar=%d/%d",
                    self._bar_count,
                    self.config.warmup_bars,
                    self._msg_count,
                    self._buffered_raw_trade_count,
                    self._historical_raw_trade_count,
                    self.bar_builder.merged_count_in_current_bar,
                    self.bar_builder.trades_per_bar_target,
                )
            return

        # Notify once when warmup completes
        if self._tg and not getattr(self, "_warmup_notified", False):
            self._warmup_notified = True
            asyncio.get_event_loop().create_task(
                self._tg.send(
                    "🔥 <b>Warmup complete</b>\n"
                    f"symbol=<code>{self.config.symbol}</code> bars=<code>{self._bar_count}</code>\n"
                    f"raw_trades=<code>{self._msg_count}</code> "
                    f"historical_raw=<code>{self._historical_raw_trade_count}</code> "
                    f"buffered_ws=<code>{self._buffered_raw_trade_count}</code>\n"
                    f"merged_progress=<code>{self.bar_builder.merged_count_in_current_bar}/"
                    f"{self.bar_builder.trades_per_bar_target}</code>"
                )
            )

        # Validate required columns
        if "buy_vwap_dist_sum" not in bars_df.columns:
            logger.error("Missing buy_vwap_dist_sum in bars_df — check bar builder mode")
            return

        try:
            sig_series = self.signal.generate_from_bars(bars_df)
        except Exception as e:
            logger.error("Signal computation error: %s", e)
            return

        signal_now = int(sig_series.iloc[-1]) if not np.isnan(sig_series.iloc[-1]) else 0
        price_now = float(bars_df["close"].iloc[-1])

        # Desired position from signal
        desired_pos = float(signal_now)

        # Hold-through-flat: keep position when signal=0
        if desired_pos == 0.0 and self._position != 0.0:
            new_position = self._position
        else:
            new_position = desired_pos

        self._execute_position(new_position, price_now, signal_now)

    def _execute_position(self, new_pos: float, price: float, signal: int) -> None:
        old_pos = self._position
        delta = new_pos - old_pos
        if abs(delta) < 1e-8:
            return

        # ── Gross PnL from the closing leg ──
        close_pnl_bps = 0.0
        if old_pos != 0 and self._entry_price > 0:
            if old_pos > 0:
                close_pnl_bps = (price / self._entry_price - 1) * 10000 * abs(old_pos)
            else:
                close_pnl_bps = (1 - price / self._entry_price) * 10000 * abs(old_pos)

        # ── Fee estimate (used immediately for simulate mode and cumulative tracking) ──
        est_fee_bps = abs(delta) * self.cost_model.blended_fee_bps(self.config.maker_fill_rate)
        est_net_pnl = close_pnl_bps - est_fee_bps
        self._cum_pnl_estimated += est_net_pnl
        # Actual cumulative starts equal to estimated; patched later for live orders
        self._cum_pnl_actual += est_net_pnl

        action = "LONG" if delta > 0 else "SHORT" if delta < 0 else "FLAT"
        fee_source = "estimate"

        logger.info(
            "[%s] pos %.1f→%.1f @ %.4f | close_pnl=%+.1f est_fee=%.2f est_net=%+.1f "
            "| cum_est=%+.1f bps",
            action, old_pos, new_pos, price,
            close_pnl_bps, est_fee_bps, est_net_pnl, self._cum_pnl_estimated,
        )

        # Trade log entry — live orders will be patched in-place by _on_fill_complete
        trade_idx = len(self._trade_log)
        self._trade_log.append({
            "trade_idx": trade_idx,
            "time": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "old_pos": old_pos,
            "new_pos": new_pos,
            "price": price,
            "signal": signal,
            "close_pnl_bps": round(close_pnl_bps, 3),
            # Fee fields — updated to actuals when live fill completes
            "fee_source": fee_source,
            "fee_bps": round(est_fee_bps, 3),
            "net_pnl_bps": round(est_net_pnl, 3),
            "cum_pnl_estimated": round(self._cum_pnl_estimated, 3),
            "cum_pnl_actual": round(self._cum_pnl_actual, 3),
            # Fill detail — populated for live orders after execution
            "fill_status": "simulate" if not self.config.is_live else "pending",
            "is_maker": None,
            "avg_fill_price": None,
            "filled_contracts": None,
        })

        # ── Telegram: trade signal ──
        if self._tg:
            asyncio.get_event_loop().create_task(
                self._tg.send_trade(
                    action=action,
                    old_pos=old_pos, new_pos=new_pos, price=price,
                    close_pnl_bps=close_pnl_bps,
                    est_fee_bps=est_fee_bps,
                    est_net_pnl=est_net_pnl,
                    cum_pnl_estimated=self._cum_pnl_estimated,
                    symbol=self.config.symbol,
                )
            )

        # ── Submit live order and register fill callback ──
        if self.config.is_live:
            self._send_live_order(delta, price, trade_idx, est_fee_bps)

        # Update position state
        self._position = new_pos
        if abs(new_pos) > 1e-8 and abs(old_pos) < 1e-8:
            self._entry_price = price
        elif abs(new_pos) < 1e-8:
            self._entry_price = 0.0

    def _send_live_order(
        self, delta: float, price: float, trade_idx: int, est_fee_bps: float
    ) -> None:
        """Submit order to thread pool; register callback to patch trade log on fill."""
        if self._client is None or self._executor is None:
            return

        side = "buy" if delta > 0 else "sell"
        contracts = abs(delta) * self._compute_contracts(price)
        logger.info("LIVE ORDER: %s %.4f %s", side, contracts, self.config.symbol)

        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            None,
            lambda: self._executor.execute(
                symbol=self.config.symbol,
                side=side,
                amount=contracts,
            ),
        )

        # done_callback fires in the thread-pool thread; use call_soon_threadsafe
        # to marshal the update back to the asyncio event loop thread safely.
        def _on_done(fut):
            loop.call_soon_threadsafe(
                self._on_fill_complete, trade_idx, est_fee_bps, contracts, fut
            )

        future.add_done_callback(_on_done)

    def _on_fill_complete(
        self,
        trade_idx: int,
        est_fee_bps: float,
        contracts: float,
        future,
    ) -> None:
        """
        Called (on the event-loop thread) when the executor.execute() future resolves.

        Patches the trade log entry with actual fill data and updates FillStats
        and the actual cumulative PnL.
        """
        entry = self._trade_log[trade_idx]

        try:
            result: Dict[str, Any] = future.result()
        except Exception as exc:
            logger.error("Order execution raised exception: %s", exc)
            entry["fill_status"] = "error"
            entry["fill_error"] = str(exc)
            self._fill_stats.n_failed += 1
            self._fill_stats.n_orders += 1
            logger.warning("FILL STATS | %s", self._fill_stats.summary_line())
            return

        if not result.get("filled"):
            err = result.get("error", "unknown")
            logger.error("Order not filled: %s", err)
            entry["fill_status"] = "failed"
            entry["fill_error"] = err
            self._fill_stats.n_failed += 1
            self._fill_stats.n_orders += 1
            logger.warning("FILL STATS | %s", self._fill_stats.summary_line())
            return

        # ── Compute actual fee in bps (same unit as est_fee_bps) ──
        # result["cost_bps"] is the per-leg fee rate (e.g. 2.0 maker, 5.0 taker, or blended)
        # We multiply by |delta|=1 because size is already captured in contracts.
        actual_fee_rate_bps = float(result.get("cost_bps", self.cost_model.maker_fee_bps))
        actual_fee_bps = actual_fee_rate_bps   # delta=1 unit; contracts handles size

        is_maker: bool = result.get("is_maker", False)
        avg_fill_price: float = float(result.get("avg_price", entry["price"]))
        filled_contracts: float = float(result.get("filled_amount", contracts))

        # ── Determine fill category ──
        if is_maker:
            fill_status = "maker"
            self._fill_stats.n_maker += 1
        else:
            # Check if partially filled as maker (mixed cost signals partial)
            if actual_fee_rate_bps < self.cost_model.taker_fee_bps:
                fill_status = "partial"
                self._fill_stats.n_partial += 1
            else:
                fill_status = "taker"
                self._fill_stats.n_taker += 1

        # ── Update cumulative actual PnL (correct the estimate we added earlier) ──
        fee_correction = actual_fee_bps - est_fee_bps   # positive = paid more than estimated
        self._cum_pnl_actual -= fee_correction

        # ── Patch trade log entry in-place ──
        actual_net_pnl = entry["close_pnl_bps"] - actual_fee_bps
        entry.update({
            "fee_source": "actual",
            "fee_bps": round(actual_fee_bps, 3),
            "net_pnl_bps": round(actual_net_pnl, 3),
            "cum_pnl_actual": round(self._cum_pnl_actual, 3),
            "fill_status": fill_status,
            "is_maker": is_maker,
            "avg_fill_price": round(avg_fill_price, 6),
            "filled_contracts": round(filled_contracts, 6),
            "fee_rate_bps": round(actual_fee_rate_bps, 3),
        })

        # ── Update FillStats ──
        self._fill_stats.n_orders += 1
        self._fill_stats.total_actual_cost_bps += actual_fee_bps
        self._fill_stats.total_estimated_cost_bps += est_fee_bps

        logger.info(
            "FILL #%d [%s] contracts=%.4f @ %.6f | "
            "fee: actual=%.2f est=%.2f diff=%+.2f bps | "
            "net_pnl=%+.2f | cum_actual=%+.1f bps",
            trade_idx, fill_status.upper(), filled_contracts, avg_fill_price,
            actual_fee_bps, est_fee_bps, fee_correction,
            actual_net_pnl, self._cum_pnl_actual,
        )
        logger.info("FILL STATS | %s", self._fill_stats.summary_line())

        # ── Telegram: fill confirmation ──
        if self._tg:
            asyncio.get_event_loop().create_task(
                self._tg.send_fill(
                    trade_idx=trade_idx,
                    fill_status=fill_status,
                    contracts=filled_contracts,
                    avg_fill_price=avg_fill_price,
                    actual_fee_bps=actual_fee_bps,
                    est_fee_bps=est_fee_bps,
                    net_pnl=actual_net_pnl,
                    cum_pnl_actual=self._cum_pnl_actual,
                    fill_stats_line=self._fill_stats.summary_line(),
                )
            )

    def _compute_contracts(self, price: float) -> float:
        if self.config.position_size_usd > 0:
            return self.config.position_size_usd / max(price, 1e-8)
        free = getattr(self, "_account_free", 0.0)
        if free > 0 and self.config.max_position_pct > 0:
            notional = free * self.config.max_position_pct * self.config.leverage
            return notional / max(price, 1e-8)
        return 1.0

    def stop(self) -> None:
        logger.info("Stopping runner...")
        self._running = False
        if self.config.is_live and self._fill_stats.n_orders > 0:
            logger.info("Final %s", self._fill_stats.summary_line())
        logger.info(
            "PnL summary | estimated=%+.1f bps | actual=%+.1f bps",
            self._cum_pnl_estimated, self._cum_pnl_actual,
        )
        if self._tg:
            self._tg.send_sync(
                f"🛑 Runner stopped | trades={len(self._trade_log)} | "
                f"PnL est={self._cum_pnl_estimated:+.1f} actual={self._cum_pnl_actual:+.1f} bps\n"
                + (self._fill_stats.summary_line() if self._fill_stats.n_orders > 0 else "")
            )

    async def _heartbeat_loop(self) -> None:
        """
        Periodic health reporting:
        - INFO log every minute
        - Telegram heartbeat every 5 minutes
        - Telegram stale-data alert if no aggTrade > 90s
        """
        while self._running:
            await asyncio.sleep(60)
            now = datetime.now(timezone.utc)
            since_msg = (
                (now - self._last_msg_ts).total_seconds()
                if self._last_msg_ts is not None
                else None
            )
            since_bar = (
                (now - self._last_bar_ts).total_seconds()
                if self._last_bar_ts is not None
                else None
            )

            logger.info(
                "HEARTBEAT | raw_trades=%d hist_raw=%d buffered_ws=%d bars=%d trades=%d "
                "merged_in_bar=%d/%d pos=%+.1f "
                "pnl_est=%+.1f pnl_actual=%+.1f | last_msg=%s last_bar=%s",
                self._msg_count,  # raw aggTrade message count
                self._historical_raw_trade_count,
                self._buffered_raw_trade_count,
                self._bar_count,
                len(self._trade_log),
                self.bar_builder.merged_count_in_current_bar,
                self.bar_builder.trades_per_bar_target,
                self._position,
                self._cum_pnl_estimated,
                self._cum_pnl_actual,
                f"{since_msg:.0f}s" if since_msg is not None else "n/a",
                f"{since_bar:.0f}s" if since_bar is not None else "n/a",
            )

            # Data-stale alert: no aggTrade for > 90s
            if since_msg is not None and since_msg > 90 and not self._stale_alert_sent:
                self._stale_alert_sent = True
                if self._tg:
                    await self._tg.send(
                        f"🟠 <b>Data stale alert</b>\n"
                        f"No aggTrade for <code>{since_msg:.0f}s</code>.\n"
                        f"raw_trades=<code>{self._msg_count}</code> "
                        f"hist_raw=<code>{self._historical_raw_trade_count}</code> "
                        f"buffered_ws=<code>{self._buffered_raw_trade_count}</code>\n"
                        f"bars=<code>{self._bar_count}</code> "
                        f"merged_progress=<code>{self.bar_builder.merged_count_in_current_bar}/"
                        f"{self.bar_builder.trades_per_bar_target}</code> "
                        f"reconnects=<code>{self._ws_reconnect_count}</code>"
                    )

            # Telegram heartbeat every 5 minutes
            should_send_hb = (
                self._tg is not None and
                (
                    self._last_tg_heartbeat_ts is None or
                    now - self._last_tg_heartbeat_ts >= timedelta(minutes=5)
                )
            )
            if should_send_hb:
                self._last_tg_heartbeat_ts = now
                await self._tg.send(
                    "💓 <b>Runner heartbeat</b>\n"
                    f"raw_trades=<code>{self._msg_count}</code>  "
                    f"hist_raw=<code>{self._historical_raw_trade_count}</code>  "
                    f"buffered_ws=<code>{self._buffered_raw_trade_count}</code>\n"
                    f"bars=<code>{self._bar_count}</code>  "
                    f"trades=<code>{len(self._trade_log)}</code>\n"
                    f"merged_progress=<code>{self.bar_builder.merged_count_in_current_bar}/"
                    f"{self.bar_builder.trades_per_bar_target}</code>\n"
                    f"pos=<code>{self._position:+.1f}</code>  "
                    f"pnl_est=<code>{self._cum_pnl_estimated:+.1f}</code>  "
                    f"pnl_actual=<code>{self._cum_pnl_actual:+.1f}</code>\n"
                    f"reconnects=<code>{self._ws_reconnect_count}</code>"
                )

    def get_trade_log(self) -> list:
        return self._trade_log

    def get_fill_stats(self) -> FillStats:
        return self._fill_stats

    def save_trade_log(self, path: str) -> None:
        if self._trade_log:
            df = pd.DataFrame(self._trade_log)
            df.to_csv(path, index=False)
            logger.info("Trade log saved → %s", path)
            if self.config.is_live:
                logger.info("  Columns include fee_source, fill_status, is_maker, "
                            "avg_fill_price, fee_rate_bps, cum_pnl_actual")


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Live/paper runner for vwap_dist_sum_v41")
    p.add_argument("--mode", default="simulate", choices=["simulate", "live"])
    p.add_argument("--symbol", default="SOL/USDC")
    p.add_argument("--tpb", type=int, default=500, help="Trades per bar")
    p.add_argument("--warmup", type=int, default=150, help="Warmup bars before trading")
    p.add_argument("--history", type=int, default=200, help="Max bars to keep in memory")
    p.add_argument("--fw1", type=int, default=8)
    p.add_argument("--ema_w1", type=int, default=10)
    p.add_argument("--fw2", type=int, default=10)
    p.add_argument("--ema_w2", type=int, default=10)
    p.add_argument("--leverage", type=int, default=2)
    p.add_argument("--position_usd", type=float, default=0.0,
                   help="Fixed USD notional per trade; 0 = use max_position_pct")
    p.add_argument("--max_pct", type=float, default=0.30,
                   help="Max fraction of free balance per position")
    p.add_argument("--maker_fill_rate", type=float, default=0.7)
    p.add_argument("--log_dir", default="logs")
    return p.parse_args()


def main():
    args = parse_args()

    # Extend RuleRunnerConfig with v41 params via subclass attributes
    cfg = RuleRunnerConfig(
        mode=args.mode,
        symbol=args.symbol,
        api_key=os.environ.get("BINANCE_API_KEY", ""),
        api_secret=os.environ.get("BINANCE_API_SECRET", ""),
        bar_mode="trade_count",
        trades_per_bar=args.tpb,
        warmup_bars=args.warmup,
        history_bars=args.history,
        leverage=args.leverage,
        position_size_usd=args.position_usd,
        max_position_pct=args.max_pct,
        maker_fill_rate=args.maker_fill_rate,
    )
    # Attach v41-specific params as dynamic attributes
    cfg.v41_fw1 = args.fw1
    cfg.v41_ema_w1 = args.ema_w1
    cfg.v41_fw2 = args.fw2
    cfg.v41_ema_w2 = args.ema_w2

    runner = VwapV41Runner(cfg)

    # Graceful shutdown on SIGTERM / SIGINT
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"v41_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    def _shutdown(sig, frame):
        logger.info("Shutdown signal received — stopping runner")
        runner.stop()
        runner.save_trade_log(str(log_file))
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        asyncio.run(runner.run())
    finally:
        runner.save_trade_log(str(log_file))


if __name__ == "__main__":
    main()
