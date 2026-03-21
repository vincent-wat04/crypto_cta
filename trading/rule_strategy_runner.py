"""
Real-time rule-strategy runner for Binance USDT-M Perpetual.

Accepts best params from rule_backtest optimizer → reconstructs the
signal/sizer/stoploss/exit/execution pipeline → connects to FAPI
aggTrade WebSocket → generates signals bar-by-bar → paper-trades
or live-trades via MakerFirstExecutor.

Two modes:
  - simulate: paper-trade using simulated fills (no API keys needed)
  - live:     execute real orders via BinancePerpetualClient + MakerFirstExecutor

Binance FAPI WebSocket:
  URL:    wss://fstream.binance.com/ws/<symbol>@aggTrade
  Stream: solusdt@aggTrade
  Rate:   real-time, fires on every aggTrade (~100-500/s for SOL)
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import RuleRunnerConfig
from .realtime_bar_builder import RealtimeBarBuilder
from .cost_model import CostModel

logger = logging.getLogger(__name__)

try:
    import websockets
except ImportError:
    websockets = None


# ═══════════════════════════════════════════════════════════
# Account Info (populated at startup for live mode)
# ═══════════════════════════════════════════════════════════

@dataclass
class AccountInfo:
    """Snapshot of account state at startup."""
    balance_usdt: float = 0.0
    available_usdt: float = 0.0
    leverage: int = 1
    position_size_contracts: float = 0.0  # computed trade size
    mark_price: float = 0.0


@dataclass
class RunnerState:
    """Mutable state of the runner."""
    position: float = 0.0          # normalized signal units (-1 to +1)
    position_contracts: float = 0.0  # actual contract quantity
    entry_price: float = 0.0
    trade_pnl_bps: float = 0.0
    cum_pnl_bps: float = 0.0
    n_signals: int = 0
    n_trades: int = 0
    trade_log: List[Dict] = field(default_factory=list)
    pnl_series: List[float] = field(default_factory=list)
    bars_since_entry: int = 0


# ═══════════════════════════════════════════════════════════
# Main Runner
# ═══════════════════════════════════════════════════════════

class RuleStrategyRunner:
    """
    Connects rule_backtest components to real-time FAPI data.

    Usage (from optimizer best result):
        runner = RuleStrategyRunner.from_backtest_params(best_params)
        await runner.run()
        results = runner.get_results()

    Usage (manual):
        cfg = RuleRunnerConfig(mode="simulate", symbol="SOL/USDT", ...)
        runner = RuleStrategyRunner(cfg, signal_gen, sizer, stop_loss, exit_rule)
        await runner.run()
    """

    def __init__(
        self,
        config: RuleRunnerConfig,
        signal_gen,              # rule_backtest.signals.BaseSignal
        sizer,                   # rule_backtest.position_sizing.BasePositionSizer
        stop_loss,               # rule_backtest.stop_loss.BaseStopLoss
        exit_rule,               # rule_backtest.exit_rules.BaseExitRule
        on_signal: Optional[Callable] = None,
    ):
        if websockets is None:
            raise ImportError("pip install websockets")

        self.config = config
        self.signal_gen = signal_gen
        self.sizer = sizer
        self.stop_loss = stop_loss
        self.exit_rule = exit_rule
        self.on_signal = on_signal

        self.cost_model = CostModel(
            maker_fee_bps=config.maker_fee_bps,
            taker_fee_bps=config.taker_fee_bps,
        )

        self.bar_builder = RealtimeBarBuilder(
            mode=config.bar_mode,
            bar_seconds=config.bar_seconds,
            trades_per_bar=config.trades_per_bar,
            volume_per_bar=config.volume_per_bar,
            history_size=config.history_bars,
        )
        self.bar_builder.on_bar_close = self._on_bar_complete

        self._state = RunnerState()
        self._account = AccountInfo()
        self._running = False
        self._bar_count = 0

        # Live execution objects (lazy init in _init_live)
        self._client = None
        self._executor = None

    # ─── Construction from backtest params ───

    @classmethod
    def from_backtest_params(
        cls,
        params: Dict[str, Any],
        config: Optional[RuleRunnerConfig] = None,
        on_signal: Optional[Callable] = None,
    ) -> "RuleStrategyRunner":
        """
        Reconstruct a RuleStrategyRunner from a BacktestResult.params dict
        (as produced by the optimizer or single-run).

        Args:
            params: dict from BacktestResult.params or optimizer CSV row.
            config: optional pre-built RuleRunnerConfig (for API keys, mode, etc.).
                    Fields in `params` fill any defaults not set in `config`.
            on_signal: callback fired on each trade.
        """
        from rule_backtest.signals import SIGNAL_CLASSES
        from rule_backtest.position_sizing import SIZER_CLASSES
        from rule_backtest.stop_loss import STOPLOSS_CLASSES
        from rule_backtest.exit_rules import EXIT_CLASSES

        # Reconstruct strategy components
        sig_type = params.get("signal", "zscore")
        signal_gen = SIGNAL_CLASSES[sig_type](**_extract_signal_kwargs(sig_type, params))

        sz_type = params.get("sizer", "fixed")
        sizer = SIZER_CLASSES[sz_type](**_extract_sizer_kwargs(sz_type, params))

        sl_type = params.get("stoploss", "none")
        stop_loss = STOPLOSS_CLASSES[sl_type](**_extract_stoploss_kwargs(sl_type, params))

        ex_type = params.get("exit", "signal_reversal")
        exit_rule = EXIT_CLASSES[ex_type](**_extract_exit_kwargs(ex_type, params))

        # Build config: start from provided or defaults, overlay with params
        if config is None:
            config = RuleRunnerConfig()

        config.factor_name = params.get("factor", config.factor_name)
        config.bar_seconds = int(params.get("bar_seconds", config.bar_seconds))
        if "bar_mode" in params and str(params["bar_mode"]) != "nan":
            config.bar_mode = str(params["bar_mode"])
        if "trades_per_bar" in params and float(params.get("trades_per_bar", 0)) > 0:
            config.trades_per_bar = int(params["trades_per_bar"])
        if "volume_per_bar" in params and float(params.get("volume_per_bar", 0)) > 0:
            config.volume_per_bar = float(params["volume_per_bar"])
        if "maker_fee" in params:
            config.maker_fee_bps = float(params["maker_fee"])
        if "taker_fee" in params:
            config.taker_fee_bps = float(params["taker_fee"])
        if "fill_rate" in params:
            config.maker_fill_rate = float(params["fill_rate"])

        # Auto-compute warmup_bars from the largest lookback in strategy params
        max_lookback = _infer_max_lookback(params)
        config.warmup_bars = max(config.warmup_bars, int(max_lookback * 1.5))
        config.history_bars = max(config.history_bars, config.warmup_bars + 200)

        return cls(
            config=config,
            signal_gen=signal_gen,
            sizer=sizer,
            stop_loss=stop_loss,
            exit_rule=exit_rule,
            on_signal=on_signal,
        )

    # ─── Account Initialization ───

    def _init_live(self) -> None:
        """
        Initialize live trading: connect exchange, check balance,
        set leverage, compute position size.
        """
        from .binance_perpetual_client import BinancePerpetualClient
        from .maker_first_executor import MakerFirstExecutor

        logger.info("Initializing live connection...")
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

        # Set leverage
        try:
            self._client.set_leverage(self.config.leverage, self.config.symbol)
            logger.info("Leverage set to %dx for %s", self.config.leverage, self.config.symbol)
        except Exception as e:
            logger.warning("Failed to set leverage: %s", e)

        # Fetch balance — support USDC-M or USDT-M wallets
        try:
            balance = self._client.fetch_balance()
            # Try USDC first (for USDC-M perp), then fall back to USDT
            for asset in ("USDC", "USDT"):
                asset_info = balance.get(asset)
                if asset_info:
                    break
            else:
                asset_info = balance.get("total", {})
            if isinstance(asset_info, dict):
                self._account.balance_usdt = float(asset_info.get("total", 0) or 0)
                self._account.available_usdt = float(asset_info.get("free", 0) or 0)
            else:
                self._account.balance_usdt = float(asset_info or 0)
                self._account.available_usdt = self._account.balance_usdt
            logger.info("Account balance: %.2f (available: %.2f)",
                        self._account.balance_usdt, self._account.available_usdt)
        except Exception as e:
            logger.error("Failed to fetch balance: %s", e)

        # Fetch current price for position sizing
        try:
            ticker = self._client.fetch_ticker(self.config.symbol)
            self._account.mark_price = float(ticker.get("last", 0) or 0)
            logger.info("Current price: %.4f", self._account.mark_price)
        except Exception as e:
            logger.warning("Failed to fetch price: %s", e)

        # Compute position size
        self._account.leverage = self.config.leverage
        self._compute_position_size()

        # Check existing positions
        try:
            positions = self._client.fetch_positions([self.config.symbol])
            for p in positions:
                contracts = float(p.get("contracts", 0) or 0)
                if contracts != 0:
                    side = p.get("side", "")
                    logger.warning("Existing position: %s %.4f contracts (%s)",
                                   self.config.symbol, contracts, side)
        except Exception as e:
            logger.warning("Failed to check positions: %s", e)

    def _compute_position_size(self) -> None:
        """
        Compute the contract quantity for one unit of signal.

        Priority:
          1. config.position_size_usd > 0  →  usd / price
          2. config.max_position_pct > 0    →  equity * pct * leverage / price
          3. fallback = 1.0 contract
        """
        price = self._account.mark_price
        if price <= 0:
            logger.warning("No price available, using 1.0 contract as default")
            self._account.position_size_contracts = 1.0
            return

        if self.config.position_size_usd > 0:
            self._account.position_size_contracts = self.config.position_size_usd / price
            logger.info("Position size: %.4f contracts (from %.2f USD notional)",
                        self._account.position_size_contracts, self.config.position_size_usd)
        elif self.config.max_position_pct > 0 and self._account.balance_usdt > 0:
            notional = self._account.balance_usdt * self.config.max_position_pct * self.config.leverage
            self._account.position_size_contracts = notional / price
            logger.info("Position size: %.4f contracts (%.2f USDT * %.0f%% * %dx leverage / %.2f)",
                        self._account.position_size_contracts,
                        self._account.balance_usdt,
                        self.config.max_position_pct * 100,
                        self.config.leverage, price)
        else:
            self._account.position_size_contracts = 1.0
            logger.info("Position size: 1.0 contract (default)")

    # ─── Run ───

    async def run(self):
        """Main event loop: connect FAPI WebSocket, process aggTrades."""
        # Init live connection if needed
        if self.config.is_live:
            self._init_live()

        symbol_ws = self.config.symbol.replace("/", "").lower()
        ws_url = f"{self.config.ws_url}/{symbol_ws}@aggTrade"

        self._running = True
        self._print_startup_banner(ws_url)

        while self._running:
            try:
                async with websockets.connect(ws_url, ping_interval=20) as ws:
                    logger.info("WebSocket connected to %s", ws_url)
                    async for msg in ws:
                        if not self._running:
                            break
                        self._process_message(msg)
            except Exception as e:
                logger.error("WebSocket error: %s", e)
                if self._running:
                    logger.info("Reconnecting in 5 seconds...")
                    await asyncio.sleep(5)

    def _print_startup_banner(self, ws_url: str) -> None:
        logger.info("Rule runner starting: %s [%s mode]", ws_url, self.config.mode)
        logger.info("Factor=%s | Signal=%s | StopLoss=%s | Exit=%s",
                     self.config.factor_name, self.signal_gen.name,
                     self.stop_loss.name, self.exit_rule.name)
        logger.info("Bar=%ds | Warmup=%d bars | Fee=%.1f bps blended",
                     self.config.bar_seconds, self.config.warmup_bars,
                     self.config.blended_fee_bps)
        if self.config.is_live:
            logger.info("Balance=%.2f USDT | Leverage=%dx | Size=%.4f contracts",
                        self._account.balance_usdt, self._account.leverage,
                        self._account.position_size_contracts)

    def _process_message(self, msg: str) -> None:
        """Parse aggTrade and feed to bar builder."""
        try:
            data = json.loads(msg)
            if data.get("e") != "aggTrade":
                return
            ts = datetime.fromtimestamp(int(data["T"]) / 1000, tz=timezone.utc)
            price = float(data["p"])
            amount = float(data["q"])
            side = "sell" if data["m"] else "buy"
            self.bar_builder.on_trade(ts, price, amount, side)
        except Exception as e:
            logger.error("Parse error: %s", e)

    # ─── Bar complete callback ───

    def _on_bar_complete(self, bars_df: pd.DataFrame) -> None:
        """Called by bar builder when a bar closes."""
        self._bar_count += 1

        if self._bar_count < self.config.warmup_bars:
            if self._bar_count % 20 == 0:
                logger.info("Warming up: %d / %d bars", self._bar_count, self.config.warmup_bars)
            return

        try:
            factor = self._compute_factor(bars_df)
        except Exception as e:
            logger.error("Factor computation error: %s", e)
            return

        if factor.isna().all():
            return

        sig_series = self.signal_gen.generate(factor)
        pos_series = self.sizer.size(sig_series, bars_df["close"])

        signal_now = int(sig_series.iloc[-1]) if not np.isnan(sig_series.iloc[-1]) else 0
        desired_pos = float(pos_series.iloc[-1]) if not np.isnan(pos_series.iloc[-1]) else 0.0
        factor_now = float(factor.iloc[-1]) if not np.isnan(factor.iloc[-1]) else 0.0
        current_price = float(bars_df["close"].iloc[-1])

        self._state.n_signals += 1
        if self._state.position != 0:
            self._state.bars_since_entry += 1

        # Unrealized PnL
        if self._state.position != 0 and self._state.entry_price > 0:
            if self._state.position > 0:
                self._state.trade_pnl_bps = (current_price / self._state.entry_price - 1) * 10000
            else:
                self._state.trade_pnl_bps = (1 - current_price / self._state.entry_price) * 10000
        else:
            self._state.trade_pnl_bps = 0.0

        # Stop-loss check
        stop_triggered = False
        if self._state.position != 0:
            stop_triggered = self.stop_loss.check(
                self._bar_count, self._state.entry_price, current_price,
                self._state.trade_pnl_bps, self._state.bars_since_entry)

        # Exit rule check
        exit_triggered = False
        if self._state.position != 0 and not stop_triggered:
            exit_triggered = self.exit_rule.should_exit(
                self._bar_count, int(np.sign(self._state.position)),
                signal_now, factor_now, self._state.bars_since_entry)

        # Apply signal flip (inverse predictor)
        if self.config.signal_flip:
            signal_now = -signal_now
            desired_pos = -desired_pos

        # Hold-through-flat: if signal is 0 but we hold a position, keep it
        if self.config.hold_through_flat and desired_pos == 0.0 and self._state.position != 0.0:
            new_position = self._state.position
        else:
            new_position = desired_pos

        if stop_triggered:
            new_position = 0.0
        elif exit_triggered:
            new_position = 0.0

        self._execute(new_position, current_price, signal_now, factor_now,
                      stop_triggered, exit_triggered)

    def _compute_factor(self, bars_df: pd.DataFrame) -> pd.Series:
        from rule_backtest.data_loader import compute_factor
        return compute_factor(self.config.factor_name, bars_df)

    # ─── Execution ───

    def _execute(self, new_position: float, price: float, signal: int,
                 factor: float, stop: bool, exit_triggered: bool) -> None:
        """Execute position change (paper or live)."""
        old_pos = self._state.position
        pos_delta = new_position - old_pos

        if abs(pos_delta) < 1e-8:
            return

        # PnL for closing leg
        close_pnl_bps = 0.0
        if old_pos != 0 and self._state.entry_price > 0:
            if old_pos > 0:
                close_pnl_bps = (price / self._state.entry_price - 1) * 10000 * abs(old_pos)
            else:
                close_pnl_bps = (1 - price / self._state.entry_price) * 10000 * abs(old_pos)

        cost_bps = abs(pos_delta) * self.config.blended_fee_bps
        net_pnl = close_pnl_bps - cost_bps
        self._state.cum_pnl_bps += net_pnl
        self._state.pnl_series.append(net_pnl)
        self._state.n_trades += 1

        # Contract quantity for the live leg
        contract_delta = abs(pos_delta) * self._account.position_size_contracts

        action = "STOP" if stop else ("EXIT" if exit_triggered else "SIGNAL")
        logger.info(
            "[%s] #%d: pos %.1f→%.1f (%.4f contracts) @ %.4f | "
            "sig=%d fac=%.4f | pnl=%+.1f cost=%.1f net=%+.1f | cum=%+.1f",
            action, self._state.n_trades, old_pos, new_position,
            contract_delta, price, signal, factor,
            close_pnl_bps, cost_bps, net_pnl, self._state.cum_pnl_bps)

        trade_record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "old_pos": old_pos,
            "new_pos": new_position,
            "contracts": contract_delta,
            "price": price,
            "signal": signal,
            "factor": factor,
            "close_pnl_bps": close_pnl_bps,
            "cost_bps": cost_bps,
            "net_pnl_bps": net_pnl,
            "cum_pnl_bps": self._state.cum_pnl_bps,
        }
        self._state.trade_log.append(trade_record)

        # Live order execution
        if self.config.is_live and contract_delta > 0:
            self._live_execute(pos_delta, contract_delta, price)

        # State update
        if abs(new_position) > 1e-8 and abs(old_pos) < 1e-8:
            self._state.entry_price = price
            self._state.bars_since_entry = 0
            self._state.position_contracts = contract_delta * np.sign(new_position)
            self.stop_loss.reset()
        elif abs(new_position) < 1e-8:
            self._state.entry_price = 0.0
            self._state.bars_since_entry = 0
            self._state.trade_pnl_bps = 0.0
            self._state.position_contracts = 0.0

        self._state.position = new_position

        if self.on_signal:
            self.on_signal(trade_record)

    def _live_execute(self, pos_delta: float, contracts: float, price: float) -> None:
        """
        Execute real orders through MakerFirstExecutor.

        Note: MakerFirstExecutor.execute() is synchronous (uses time.sleep for
        maker polling). It is called from _on_bar_complete which runs in the
        asyncio event loop. To avoid blocking, we run it in a thread executor.
        The caller should await the returned coroutine if async context is needed;
        here we use run_in_executor via asyncio.get_event_loop().run_in_executor.
        """
        if self._client is None or self._executor is None:
            self._init_live()

        side = "buy" if pos_delta > 0 else "sell"

        logger.info("LIVE ORDER: %s %.4f %s @ ~%.4f",
                     side, contracts, self.config.symbol, price)
        try:
            # Run blocking executor call in a thread pool so we don't block WS loop
            loop = asyncio.get_event_loop()
            loop.run_in_executor(
                None,
                lambda: self._executor.execute(
                    symbol=self.config.symbol,
                    side=side,
                    amount=contracts,
                )
            )
            logger.info("LIVE ORDER submitted to thread pool")
        except RuntimeError:
            # No running event loop (e.g. tests) — fall back to direct call
            try:
                result = self._executor.execute(
                    symbol=self.config.symbol,
                    side=side,
                    amount=contracts,
                )
                logger.info("LIVE FILL: %s", result)
            except Exception as ex:
                logger.error("LIVE ORDER FAILED: %s", ex)
        except Exception as e:
            logger.error("LIVE ORDER FAILED: %s", e)

    # ─── Control & Results ───

    def stop(self):
        self._running = False

    def get_results(self) -> Dict[str, Any]:
        """Return simulation/live results with metrics."""
        from .metrics import compute_investment_metrics
        pnl = pd.Series(self._state.pnl_series, dtype=float)
        periods_per_year = 365 * 24 * 3600 / self.config.bar_seconds
        metrics = compute_investment_metrics(pnl, periods_per_year=periods_per_year) if len(pnl) > 1 else {}

        return {
            "metrics": metrics,
            "trade_log": self._state.trade_log,
            "n_trades": self._state.n_trades,
            "n_signals": self._state.n_signals,
            "n_bars": self._bar_count,
            "cumulative_pnl_bps": self._state.cum_pnl_bps,
            "current_position": self._state.position,
            "account": {
                "balance_usdt": self._account.balance_usdt,
                "leverage": self._account.leverage,
                "position_size_contracts": self._account.position_size_contracts,
            },
        }

    def get_trade_log_df(self) -> pd.DataFrame:
        if not self._state.trade_log:
            return pd.DataFrame()
        return pd.DataFrame(self._state.trade_log)


# ═══════════════════════════════════════════════════════════
# Parameter extraction helpers
# ═══════════════════════════════════════════════════════════

def _infer_max_lookback(params: Dict) -> int:
    """Extract the largest lookback window from strategy params."""
    candidates = [
        params.get("window", 0),
        params.get("slow_window", 0),
        params.get("slow", 0),
        params.get("fast_window", 0),
        params.get("fast", 0),
        params.get("band_window", 0),
        params.get("slow_span", 0),
        params.get("vol_window", 0),
        params.get("lookback", 0),
        params.get("max_hold_bars", 0),
        params.get("max_bars", 0),
    ]
    return max(int(c) for c in candidates if c)


def _extract_signal_kwargs(sig_type: str, params: Dict) -> Dict:
    if sig_type == "zscore":
        return {"window": params.get("window", 60), "threshold": params.get("threshold", 1.5)}
    if sig_type == "quantile":
        return {"window": params.get("window", 120),
                "upper_q": params.get("upper_q", 0.8), "lower_q": params.get("lower_q", 0.2)}
    if sig_type == "ma_cross":
        return {"fast_window": params.get("fast", params.get("fast_window", 5)),
                "slow_window": params.get("slow", params.get("slow_window", 20))}
    if sig_type == "bollinger":
        return {"window": params.get("window", 20), "n_std": params.get("n_std", 2.0)}
    if sig_type == "rank":
        return {"window": params.get("window", 60),
                "upper_pct": params.get("upper_pct", 0.8), "lower_pct": params.get("lower_pct", 0.2)}
    if sig_type == "delta":
        return {"lookback": params.get("lookback", 5)}
    if sig_type == "threshold":
        return {"upper_thr": params.get("upper", 0.5), "lower_thr": params.get("lower", -0.5)}
    if sig_type == "dual_ma_band":
        return {"fast_span": params.get("fast_span", 5), "slow_span": params.get("slow_span", 20),
                "band_window": params.get("band_window", 40), "band_mult": params.get("band_mult", 1.0)}
    return {}


def _extract_sizer_kwargs(sz_type: str, params: Dict) -> Dict:
    if sz_type == "fixed":
        return {"size_units": params.get("size_units", 1.0)}
    if sz_type == "vol_target":
        return {"target_vol_bps": params.get("target_vol_bps", 50.0),
                "vol_window": params.get("vol_window", 60)}
    if sz_type == "signal_prop":
        return {"scale": params.get("scale", 1.0), "max_size": params.get("max_size", 1.0)}
    return {}


def _extract_stoploss_kwargs(sl_type: str, params: Dict) -> Dict:
    if sl_type == "fixed":
        return {"max_loss_bps": params.get("max_loss_bps", 30.0)}
    if sl_type == "trailing":
        return {"trail_bps": params.get("trail_bps", 20.0)}
    if sl_type == "time":
        return {"max_bars": params.get("max_bars", 60)}
    if sl_type == "atr":
        return {"multiplier": params.get("multiplier", 2.0)}
    return {}


def _extract_exit_kwargs(ex_type: str, params: Dict) -> Dict:
    if ex_type == "time_decay":
        return {"max_hold_bars": params.get("max_hold_bars", 30)}
    if ex_type == "mean_reversion":
        return {"exit_zscore": params.get("exit_zscore", 0.0)}
    return {}
