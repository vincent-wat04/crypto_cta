"""
高频指标库。

分层设计：
  base/       基础指标（price impact, spread, order flow, volume profile, ob pressure）
  composite/  复合指标（FVG, microstructure reversal — 由 base 指标通过模型/规则组合）
  regime/     市场状态识别（volatility regime, trend strength, liquidity regime）

所有 base 指标返回 pd.Series，index 与输入对齐。
composite 和 regime 指标可返回 DataFrame。
"""
from . import base
from . import regime
