"""
Triple Barrier Method 标签生成。

参考：Marcos López de Prado, "Advances in Financial Machine Learning" (2018), Chapter 3。

三重屏障法：
  以当前时刻为起点，设定未来时间窗口 + 上下轨道。
  未来价格最先触及哪个屏障，就赋予对应标签：
    触及上轨 →  1（强劲上涨）
    触及下轨 → -1（强劲下跌）
    时间屏障 →  0（震荡/无趋势）

关键改进（相对于朴素实现）：
  1. 使用 HIGH/LOW 而非仅 CLOSE 判断触及（更真实）
  2. 默认对称上下轨（与 de Prado 一致）
  3. 支持波动率自适应轨道宽度
  4. 时间屏障到达时按最终收益方向给标签（可选）
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def triple_barrier_labels(
    prices: pd.Series,
    upper_pct: float = 0.5,
    lower_pct: float = 0.5,
    max_bars: int = 60,
    volatility_adjusted: bool = False,
    vol_lookback: int = 100,
    vol_multiplier: float = 2.0,
    use_high_low: bool = False,
    ohlcv: Optional[pd.DataFrame] = None,
    time_label_by_return: bool = False,
) -> pd.DataFrame:
    """
    Triple Barrier 标签生成。

    Args:
        prices: 价格序列（close prices，index 为时间）
        upper_pct: 上轨百分比（如 0.5 表示 +0.5%）
        lower_pct: 下轨百分比（如 0.5 表示 -0.5%）
            注意：上下轨默认对称使用同一数值，与 de Prado 一致。
        max_bars: 最大持有 bar 数（时间屏障）
        volatility_adjusted: 是否根据波动率动态调整轨道宽度
        vol_lookback: 波动率回溯周期
        vol_multiplier: 波动率乘数（上下轨 = vol * multiplier）
        use_high_low: 是否使用 OHLCV 的 high/low 判断屏障触及
            （更真实：同一根 bar 内如果 high 和 low 都触及，
             优先判断 bar 内哪个先到 — 用 close 与 open 方向推断）
        ohlcv: 当 use_high_low=True 时需要提供
        time_label_by_return: 时间屏障到期时是否按最终收益方向给标签
            True: 正收益→1, 负收益→-1
            False: 始终给 0（严格三分类）

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
    arr_close = prices.values.astype(float)

    if use_high_low and ohlcv is not None:
        arr_high = ohlcv["high"].values.astype(float)
        arr_low = ohlcv["low"].values.astype(float)
    else:
        arr_high = arr_close
        arr_low = arr_close

    labels = np.zeros(n, dtype=int)
    barrier_hit = np.empty(n, dtype=object)
    bars_to_hit = np.zeros(n, dtype=int)
    return_at_hit = np.zeros(n, dtype=float)
    max_favorable = np.zeros(n, dtype=float)
    max_adverse = np.zeros(n, dtype=float)

    # 波动率序列（用于自适应轨道）
    if volatility_adjusted:
        returns = prices.pct_change().fillna(0)
        vol = returns.rolling(vol_lookback, min_periods=20).std().fillna(
            returns.std()
        ) * 100  # 转为百分比
    else:
        vol = None

    for i in range(n):
        entry = arr_close[i]
        if entry <= 0 or np.isnan(entry):
            barrier_hit[i] = "invalid"
            continue

        end = min(i + max_bars, n - 1)

        # 动态轨道宽度
        if volatility_adjusted and vol is not None:
            v = vol.iloc[i]
            if v > 0:
                up_th = v * vol_multiplier
                dn_th = v * vol_multiplier
            else:
                up_th = upper_pct
                dn_th = lower_pct
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
            # 使用 high/low 判断是否触及屏障
            h_j = arr_high[j]
            l_j = arr_low[j]
            c_j = arr_close[j]

            ret_h = (h_j - entry) / entry * 100
            ret_l = (l_j - entry) / entry * 100
            ret_c = (c_j - entry) / entry * 100

            mfav = max(mfav, ret_h)
            madv = min(madv, ret_l)

            touched_upper = h_j >= upper_price
            touched_lower = l_j <= lower_price

            if touched_upper and touched_lower:
                # 同一根 bar 内两侧都触及，用 close vs open 方向推断先后
                # 如果 close > open，假定先触下再触上（V 型反转）
                # 如果 close < open，假定先触上再触下（倒 V）
                if c_j >= entry:
                    hit = "lower"  # 先跌后涨
                else:
                    hit = "upper"  # 先涨后跌
                hit_bar = j - i
                hit_ret = ret_c
                break
            elif touched_upper:
                hit = "upper"
                hit_bar = j - i
                hit_ret = (upper_price - entry) / entry * 100
                break
            elif touched_lower:
                hit = "lower"
                hit_bar = j - i
                hit_ret = (lower_price - entry) / entry * 100
                break
        else:
            # 时间屏障到达
            if end < n:
                hit_ret = (arr_close[end] - entry) / entry * 100

        if hit == "upper":
            labels[i] = 1
        elif hit == "lower":
            labels[i] = -1
        else:
            if time_label_by_return:
                # 时间屏障按收益方向
                if hit_ret > 0:
                    labels[i] = 1
                elif hit_ret < 0:
                    labels[i] = -1
                else:
                    labels[i] = 0
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
    upper_pct: float = 0.5,
    lower_pct: float = 0.5,
    **kwargs,
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
        tb = triple_barrier_labels(
            prices, upper_pct=upper_pct, lower_pct=lower_pct,
            max_bars=h, **kwargs,
        )
        for col in tb.columns:
            result[f"{col}_{h}"] = tb[col]

    return result


def validate_labels(
    labels: pd.Series,
    prices: pd.Series,
    expected_trend: str = "auto",
) -> dict:
    """
    验证标签质量：检查标签分布是否与实际价格趋势一致。

    Args:
        labels: Triple Barrier 标签 (1/-1/0)
        prices: 对应的价格序列
        expected_trend: "up", "down", "auto" (从价格推断)

    Returns:
        dict with validation results
    """
    total_return = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] * 100

    if expected_trend == "auto":
        if total_return > 1:
            expected_trend = "up"
        elif total_return < -1:
            expected_trend = "down"
        else:
            expected_trend = "neutral"

    dist = labels.value_counts().to_dict()
    n_up = dist.get(1, 0)
    n_down = dist.get(-1, 0)
    n_neutral = dist.get(0, 0)
    total = len(labels)

    # 一致性检查
    is_consistent = True
    warnings = []

    if expected_trend == "down" and n_up > n_down:
        is_consistent = False
        warnings.append(
            f"Price fell {total_return:.1f}% but label=1 ({n_up}) > label=-1 ({n_down}). "
            f"Barriers may be too wide or asymmetric."
        )

    if expected_trend == "up" and n_down > n_up:
        is_consistent = False
        warnings.append(
            f"Price rose {total_return:.1f}% but label=-1 ({n_down}) > label=1 ({n_up}). "
            f"Barriers may be too wide or asymmetric."
        )

    if n_neutral / (total + 1) > 0.8:
        warnings.append(
            "80%+ labels are 0 (neutral). Barriers may be too wide."
        )

    return {
        "total_return_pct": round(total_return, 2),
        "expected_trend": expected_trend,
        "label_distribution": {str(k): int(v) for k, v in dist.items()},
        "is_consistent": is_consistent,
        "warnings": warnings,
    }
