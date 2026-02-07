"""
Triple Barrier Method 标签生成。

思路：
  以当前时刻为起点，设定固定未来时间窗口 + 上下轨道。
  未来价格最先触及哪个屏障，就赋予对应标签：
    触及上轨 →  1（强劲上涨）
    触及下轨 → -1（强劲下跌）
    时间屏障 →  0（震荡/无趋势）

  优点：标签反映趋势强度和方向，与固定时间窗口解耦。
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def triple_barrier_labels(
    prices: pd.Series,
    upper_pct: float = 2.0,
    lower_pct: float = 1.5,
    max_bars: int = 60,
    volatility_adjusted: bool = False,
    vol_lookback: int = 100,
    vol_multiplier_up: float = 2.0,
    vol_multiplier_down: float = 1.5,
) -> pd.DataFrame:
    """
    Triple Barrier 标签生成。

    Args:
        prices: 价格序列（close prices，index 为时间）
        upper_pct: 上轨百分比（如 2.0 表示 +2%）
        lower_pct: 下轨百分比（如 1.5 表示 -1.5%）
        max_bars: 最大持有 bar 数（时间屏障）
        volatility_adjusted: 是否根据波动率动态调整轨道
        vol_lookback: 波动率回溯周期
        vol_multiplier_up / vol_multiplier_down: 波动率乘数

    Returns:
        DataFrame with columns:
            label: 1 / -1 / 0
            barrier_hit: "upper" / "lower" / "time"
            bars_to_hit: 触发屏障所需 bar 数
            return_at_hit: 触发时的收益率 (%)
            max_favorable: 最大有利偏移 (%)
            max_adverse: 最大不利偏移 (%)
    """
    n = len(prices)
    arr = prices.values.astype(float)

    labels = np.zeros(n, dtype=int)
    barrier_hit = np.empty(n, dtype=object)
    bars_to_hit = np.zeros(n, dtype=int)
    return_at_hit = np.zeros(n, dtype=float)
    max_favorable = np.zeros(n, dtype=float)
    max_adverse = np.zeros(n, dtype=float)

    # 可选：波动率序列
    if volatility_adjusted:
        returns = prices.pct_change().fillna(0)
        vol = returns.rolling(vol_lookback, min_periods=20).std().fillna(returns.std()) * 100
    else:
        vol = None

    for i in range(n):
        entry = arr[i]
        if entry <= 0 or np.isnan(entry):
            barrier_hit[i] = "invalid"
            continue

        end = min(i + max_bars, n - 1)

        # 动态轨道
        if volatility_adjusted and vol is not None:
            v = vol.iloc[i]
            up_th = v * vol_multiplier_up if v > 0 else upper_pct
            dn_th = v * vol_multiplier_down if v > 0 else lower_pct
        else:
            up_th = upper_pct
            dn_th = lower_pct

        upper_price = entry * (1 + up_th / 100)
        lower_price = entry * (1 - dn_th / 100)

        hit = "time"
        hit_bar = end - i
        hit_ret = 0.0
        mfav = 0.0
        madv = 0.0

        for j in range(i + 1, end + 1):
            p = arr[j]
            ret_pct = (p - entry) / entry * 100

            mfav = max(mfav, ret_pct)
            madv = min(madv, ret_pct)

            if p >= upper_price:
                hit = "upper"
                hit_bar = j - i
                hit_ret = ret_pct
                break
            elif p <= lower_price:
                hit = "lower"
                hit_bar = j - i
                hit_ret = ret_pct
                break
        else:
            # Time barrier
            if end < n:
                hit_ret = (arr[end] - entry) / entry * 100

        if hit == "upper":
            labels[i] = 1
        elif hit == "lower":
            labels[i] = -1
        else:
            labels[i] = 0

        barrier_hit[i] = hit
        bars_to_hit[i] = hit_bar
        return_at_hit[i] = hit_ret
        max_favorable[i] = mfav
        max_adverse[i] = madv

    return pd.DataFrame({
        "label": labels,
        "barrier_hit": barrier_hit,
        "bars_to_hit": bars_to_hit,
        "return_at_hit": return_at_hit,
        "max_favorable": max_favorable,
        "max_adverse": max_adverse,
    }, index=prices.index)


def multi_horizon_labels(
    prices: pd.Series,
    horizons: Optional[list] = None,
    upper_pct: float = 2.0,
    lower_pct: float = 1.5,
) -> pd.DataFrame:
    """
    多时间跨度的 Triple Barrier 标签。

    Returns:
        DataFrame with columns: label_{horizon}, barrier_{horizon}, ...
    """
    if horizons is None:
        horizons = [10, 30, 60, 120]

    result = pd.DataFrame(index=prices.index)
    for h in horizons:
        tb = triple_barrier_labels(prices, upper_pct=upper_pct, lower_pct=lower_pct, max_bars=h)
        for col in tb.columns:
            result[f"{col}_{h}"] = tb[col]

    return result
