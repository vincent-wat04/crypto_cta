"""
流动性 Regime 指标。

原理：
  - amihud_illiquidity: Amihud (2002) 非流动性比率
  - turnover_ratio: 成交额换手率
  - liquidity_regime: 基于流动性指标的 regime 分类

机制：
  Amihud 高 → 价格对成交量敏感 → 低流动性 → 反转策略更优
  Amihud 低 → 价格吸收能力强 → 高流动性 → 趋势策略更优
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def amihud_illiquidity(
    ohlcv: pd.DataFrame,
    window: int = 20,
) -> pd.Series:
    """
    Amihud 非流动性: mean(|r_t| / Volume_t)。
    """
    c = ohlcv["close"].astype(float)
    v = ohlcv["volume"].astype(float)
    abs_ret = (c / c.shift(1) - 1).abs()
    illiq = abs_ret / (v + 1e-10)
    result = illiq.rolling(window, min_periods=5).mean()
    result.name = "amihud"
    return result


def turnover_ratio(
    ohlcv: pd.DataFrame,
    window: int = 20,
) -> pd.Series:
    """
    成交量 / 滚动平均成交量。
    """
    v = ohlcv["volume"].astype(float)
    avg = v.rolling(window, min_periods=5).mean() + 1e-10
    result = v / avg
    result.name = "turnover_ratio"
    return result


def liquidity_regime(
    amihud_series: pd.Series,
    lookback: int = 500,
    thresholds: tuple = (0.25, 0.75),
) -> pd.Series:
    """
    流动性 regime:
      "liquid"   : amihud < 25th percentile（低冲击 → 高流动性）
      "normal"   : 25th ~ 75th
      "illiquid" : amihud >= 75th percentile（高冲击 → 低流动性）
    """
    q_low = amihud_series.rolling(lookback, min_periods=30).quantile(thresholds[0])
    q_high = amihud_series.rolling(lookback, min_periods=30).quantile(thresholds[1])

    result = pd.Series("normal", index=amihud_series.index)
    result[amihud_series < q_low] = "liquid"
    result[amihud_series >= q_high] = "illiquid"
    result.name = "liquidity_regime"
    return result
