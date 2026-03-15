"""
Factor research framework (WorldQuant Brain-like).

All factors = primitives + TS operators, composed via expression strings.
No hardcoded indicator functions — everything goes through the expression engine.

Modules:
  data_schema  - Primitives: atomic per-bar quantities from aggTrades + orderbook
                 (OHLCV, volume decomposition, price impact, microstructure, etc.)
  operators    - Time series operators (ts_mean, ts_rank, decay_linear, ts_entropy, ...)
  expressions  - Factor expression parser & evaluator (string → AST → Series)
  evaluator    - Factor quality metrics (IC, decile analysis, half-life, turnover)
  scanner      - Factor combination generator & batch evaluator (1st + 2nd order)
"""
