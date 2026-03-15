"""
Trading module: Binance Perpetual exchange client, order execution, cost model.

- BinancePerpetualClient: ccxt wrapper for demo (testnet) and live
- MakerFirstExecutor: place limit at mid, then market if not filled
- CostModel: maker rebate (-2 bps), taker fee (4 bps)
- paper_backtest: simulate maker-first-then-taker with real costs
- RealtimeBarBuilder: accumulate aggTrades into OHLCV bars in real-time
- RuleStrategyRunner: connect rule_backtest best params → live FAPI WebSocket
"""
from .config import TradingConfig, RuleRunnerConfig
from .binance_perpetual_client import BinancePerpetualClient
from .cost_model import CostModel
from .maker_first_executor import MakerFirstExecutor
from .metrics import compute_investment_metrics, format_metrics_report
from .realtime_bar_builder import RealtimeBarBuilder
from .rule_strategy_runner import RuleStrategyRunner

__all__ = [
    "TradingConfig",
    "RuleRunnerConfig",
    "BinancePerpetualClient",
    "CostModel",
    "MakerFirstExecutor",
    "compute_investment_metrics",
    "format_metrics_report",
    "RealtimeBarBuilder",
    "RuleStrategyRunner",
]
