"""
趋势强度 Regime 指标。

原理：
  - adx_indicator: Average Directional Index (Wilder, 1978)
  - efficiency_ratio: Kaufman 效率比（net move / total path）
  - hurst_exponent: Hurst 指数估计（R/S 分析）
  - trend_regime: 基于趋势指标的 regime 分类

机制：
  ADX > 25 → 趋势市场 → 趋势跟踪策略
  ADX < 20 → 震荡市场 → 均值回归策略
  Hurst > 0.5 → 趋势持续 / < 0.5 → 均值回归
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def adx_indicator(
    ohlcv: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:
    """
    ADX + DI+ / DI-。

    Returns:
        DataFrame with: adx, di_plus, di_minus
    """
    h = ohlcv["high"].astype(float)
    l = ohlcv["low"].astype(float)
    c = ohlcv["close"].astype(float)

    # True Range
    tr1 = h - l
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # +DM, -DM
    up = h - h.shift(1)
    down = l.shift(1) - l
    dm_plus = pd.Series(np.where((up > down) & (up > 0), up, 0), index=ohlcv.index)
    dm_minus = pd.Series(np.where((down > up) & (down > 0), down, 0), index=ohlcv.index)

    # Smoothed
    atr = tr.ewm(span=period, adjust=False).mean()
    sdm_plus = dm_plus.ewm(span=period, adjust=False).mean()
    sdm_minus = dm_minus.ewm(span=period, adjust=False).mean()

    di_plus = 100 * sdm_plus / (atr + 1e-10)
    di_minus = 100 * sdm_minus / (atr + 1e-10)
    dx = (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-10) * 100
    adx = dx.ewm(span=period, adjust=False).mean()

    return pd.DataFrame({
        "adx": adx,
        "di_plus": di_plus,
        "di_minus": di_minus,
    }, index=ohlcv.index)


def efficiency_ratio(
    prices: pd.Series,
    window: int = 20,
) -> pd.Series:
    """
    Kaufman 效率比: |P_t - P_{t-n}| / Σ|P_i - P_{i-1}|
    范围 [0, 1]，越接近 1 趋势越强。
    """
    net_move = (prices - prices.shift(window)).abs()
    path_sum = prices.diff().abs().rolling(window).sum() + 1e-10
    result = net_move / path_sum
    result.name = "efficiency_ratio"
    return result


def hurst_exponent(
    prices: pd.Series,
    max_lag: int = 100,
) -> pd.Series:
    """
    滚动 Hurst 指数（R/S 分析简化版）。

    H > 0.5: 持续性（趋势）
    H = 0.5: 随机游走
    H < 0.5: 均值回归
    """
    log_ret = np.log(prices / prices.shift(1)).fillna(0)

    def _rs_hurst(window):
        n = len(window)
        if n < 20:
            return 0.5

        lags = np.arange(2, min(n // 2, max_lag))
        if len(lags) < 3:
            return 0.5

        rs_values = []
        for lag in lags:
            rs_list = []
            for start in range(0, n - lag, lag):
                segment = window[start:start + lag]
                mean_val = segment.mean()
                deviations = np.cumsum(segment - mean_val)
                r = deviations.max() - deviations.min()
                s = segment.std()
                if s > 1e-10:
                    rs_list.append(r / s)
            if rs_list:
                rs_values.append((np.log(lag), np.log(np.mean(rs_list))))

        if len(rs_values) < 3:
            return 0.5

        x = np.array([v[0] for v in rs_values])
        y = np.array([v[1] for v in rs_values])
        slope = np.polyfit(x, y, 1)[0]
        return np.clip(slope, 0, 1)

    result = log_ret.rolling(max_lag * 2, min_periods=50).apply(_rs_hurst, raw=True)
    result.name = "hurst"
    return result


def trend_regime(
    adx_series: pd.Series,
    hurst_series: pd.Series = None,
    adx_trend_th: float = 25,
    adx_range_th: float = 20,
) -> pd.Series:
    """
    趋势 regime 分类：
      "trending"    : ADX > 25 (or Hurst > 0.55)
      "ranging"     : ADX < 20 (or Hurst < 0.45)
      "transitional": 其他
    """
    result = pd.Series("transitional", index=adx_series.index)
    result[adx_series > adx_trend_th] = "trending"
    result[adx_series < adx_range_th] = "ranging"

    if hurst_series is not None:
        result[(hurst_series > 0.55) & (result != "trending")] = "trending"
        result[(hurst_series < 0.45) & (result != "ranging")] = "ranging"

    result.name = "trend_regime"
    return result
