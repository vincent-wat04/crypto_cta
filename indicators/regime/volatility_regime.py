"""
波动率 Regime 指标。

原理：
  - realized_volatility: 已实现波动率（对数收益率的标准差）
  - parkinson_volatility: Parkinson (1980) 高低价波动率
  - garch_like_vol: 简化 GARCH 递推波动率
  - volatility_regime: 基于 vol 分位数的 regime 分类

机制：
  高波动 regime → 止损范围扩大，信号阈值提高
  低波动 regime → 更紧的止盈止损，更积极的入场
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def realized_volatility(
    prices: pd.Series,
    window: int = 60,
) -> pd.Series:
    """
    已实现波动率 = log_return.rolling(window).std() * sqrt(window_per_day)。
    """
    log_ret = np.log(prices / prices.shift(1))
    result = log_ret.rolling(window, min_periods=10).std()
    result.name = "realized_vol"
    return result


def parkinson_volatility(
    ohlcv: pd.DataFrame,
    window: int = 60,
) -> pd.Series:
    """
    Parkinson 高低价波动率: sqrt(1/(4*ln2) * mean(ln(H/L)^2))
    比已实现波动率更高效（利用了 high/low 信息）。
    """
    h = ohlcv["high"].astype(float)
    l = ohlcv["low"].astype(float)
    hl_ratio = np.log(h / (l + 1e-10)) ** 2
    result = np.sqrt(hl_ratio.rolling(window, min_periods=10).mean() / (4 * np.log(2)))
    result.name = "parkinson_vol"
    return result


def garch_like_vol(
    prices: pd.Series,
    alpha: float = 0.06,
    beta: float = 0.93,
    omega: float = 0.01,
) -> pd.Series:
    """
    简化 GARCH(1,1) 递推波动率：
    σ²_t = ω + α * r²_{t-1} + β * σ²_{t-1}

    无需 scipy 优化，使用预设参数。
    """
    log_ret = np.log(prices / prices.shift(1)).fillna(0)
    n = len(log_ret)
    var = np.zeros(n)
    var[0] = log_ret[:20].var() if n >= 20 else 0.0001

    for i in range(1, n):
        var[i] = omega + alpha * log_ret.iloc[i - 1] ** 2 + beta * var[i - 1]

    result = pd.Series(np.sqrt(var), index=prices.index, name="garch_vol")
    return result


def volatility_regime(
    vol_series: pd.Series,
    lookback: int = 500,
    thresholds: tuple = (0.25, 0.75),
) -> pd.Series:
    """
    根据波动率分位数划分 regime:
      "low"    : vol < 25th percentile
      "medium" : 25th <= vol < 75th
      "high"   : vol >= 75th percentile
    """
    q_low = vol_series.rolling(lookback, min_periods=30).quantile(thresholds[0])
    q_high = vol_series.rolling(lookback, min_periods=30).quantile(thresholds[1])

    result = pd.Series("medium", index=vol_series.index)
    result[vol_series < q_low] = "low"
    result[vol_series >= q_high] = "high"
    result.name = "vol_regime"
    return result
