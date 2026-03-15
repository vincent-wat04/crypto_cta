"""
Rule-based backtesting framework for single-factor HF strategies.

Architecture:
  data_loader   - Load cached perpetual aggTrades, build OHLCV + indicator bars
  signals       - Entry signal generators (zscore, quantile, MA cross, ...)
  position_sizing - Position sizing strategies (fixed, vol-target, signal-prop)
  stop_loss     - Stop-loss mechanisms (fixed, trailing, time-based, ATR)
  exit_rules    - Exit strategies (signal reversal, mean reversion, time decay)
  execution     - Execution cost simulation (maker-first-taker, pure taker)
  engine        - Per-bar backtest engine orchestrating all components
  report        - Visualization & metrics (5-panel chart + trade log)
  optimizer     - Grid search over signal/position/stoploss parameter combos
"""
