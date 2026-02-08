"""
动态止损止盈模块。

核心原则：**只使用 t 时刻及之前的数据**，无未来信息泄漏。

参考：
  - ATR-based trailing stop (Wilder)
  - Chandelier Exit (Chuck LeBeau)
  - Volatility-adjusted take profit

方法：
  1. ATR 动态止损：stop = entry ± N * ATR(t)，ATR 在入场时刻计算
  2. Chandelier Exit：trailing stop 跟踪最高/最低价
  3. 波动率分位数止损：stop = entry ± vol_percentile * price
  4. 复合退出：组合以上方法
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class ExitParams:
    """单笔交易的动态退出参数。"""
    stop_loss_pct: float    # 止损百分比
    take_profit_pct: float  # 止盈百分比
    trailing_stop: bool = False      # 是否启用移动止损
    trailing_atr_mult: float = 2.0   # 移动止损的 ATR 倍数
    max_hold_bars: int = 20          # 最大持仓 bar 数


def compute_atr(ohlcv: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range，只用历史数据。

    Args:
        ohlcv: 含 high, low, close 的 DataFrame
        period: 回溯窗口

    Returns:
        ATR Series（与 ohlcv 同 index）
    """
    h = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)
    c = ohlcv["close"].astype(float)

    tr1 = h - low
    tr2 = (h - c.shift(1)).abs()
    tr3 = (low - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(span=period, adjust=False).mean()
    atr.name = "atr"
    return atr


def atr_exit_params(
    ohlcv: pd.DataFrame,
    entry_idx: int,
    direction: int = 1,  # noqa: ARG001 — kept for API consistency
    atr_period: int = 14,
    sl_atr_mult: float = 2.0,
    tp_atr_mult: float = 3.0,
    max_hold_bars: int = 20,
    min_sl_pct: float = 0.1,
    max_sl_pct: float = 2.0,
) -> ExitParams:
    """
    基于 ATR 的动态止损止盈（只用 entry_idx 及之前的数据）。

    Args:
        ohlcv: OHLCV 数据
        entry_idx: 入场 bar 的位置索引
        direction: 1=做多, -1=做空
        atr_period: ATR 回溯周期
        sl_atr_mult: 止损 = ATR * sl_atr_mult
        tp_atr_mult: 止盈 = ATR * tp_atr_mult
        max_hold_bars: 最大持仓 bar 数
        min_sl_pct / max_sl_pct: 止损百分比的上下限

    Returns:
        ExitParams
    """
    # 计算到 entry_idx 为止的 ATR
    window = ohlcv.iloc[max(0, entry_idx - atr_period * 3):entry_idx + 1]
    if len(window) < 5:
        return ExitParams(
            stop_loss_pct=0.3, take_profit_pct=0.5,
            max_hold_bars=max_hold_bars,
        )

    atr = compute_atr(window, period=atr_period)
    current_atr = atr.iloc[-1]
    entry_price = ohlcv.iloc[entry_idx]["close"]

    # ATR 转百分比
    atr_pct = current_atr / entry_price * 100

    sl_pct = np.clip(atr_pct * sl_atr_mult, min_sl_pct, max_sl_pct)
    tp_pct = np.clip(atr_pct * tp_atr_mult, min_sl_pct * 1.5, max_sl_pct * 2)

    return ExitParams(
        stop_loss_pct=round(sl_pct, 4),
        take_profit_pct=round(tp_pct, 4),
        max_hold_bars=max_hold_bars,
    )


def volatility_percentile_exit(
    ohlcv: pd.DataFrame,
    entry_idx: int,
    vol_lookback: int = 100,
    sl_percentile: float = 0.9,
    tp_ratio: float = 1.5,
    max_hold_bars: int = 20,
    min_sl_pct: float = 0.1,
    max_sl_pct: float = 2.0,
) -> ExitParams:
    """
    基于波动率分位数的动态止损。

    思路：止损设在近期波动率的 90th percentile 处，
    即"正常波动范围的极端值"。

    Args:
        ohlcv: OHLCV
        entry_idx: 入场位置
        vol_lookback: 波动率回溯窗口
        sl_percentile: 止损的波动率分位数
        tp_ratio: 止盈 / 止损 比率
    """
    start = max(0, entry_idx - vol_lookback)
    window = ohlcv.iloc[start:entry_idx + 1]

    if len(window) < 20:
        return ExitParams(
            stop_loss_pct=0.3, take_profit_pct=0.5,
            max_hold_bars=max_hold_bars,
        )

    # 单根 bar 的波动率分布
    bar_range = (window["high"] - window["low"]) / window["close"] * 100
    sl_pct = np.clip(
        bar_range.quantile(sl_percentile),
        min_sl_pct, max_sl_pct,
    )
    tp_pct = np.clip(sl_pct * tp_ratio, min_sl_pct * 1.5, max_sl_pct * 2)

    return ExitParams(
        stop_loss_pct=round(float(sl_pct), 4),
        take_profit_pct=round(float(tp_pct), 4),
        max_hold_bars=max_hold_bars,
    )


def chandelier_exit_params(
    ohlcv: pd.DataFrame,
    entry_idx: int,
    direction: int,
    atr_period: int = 14,
    atr_mult: float = 3.0,
    max_hold_bars: int = 20,
) -> ExitParams:
    """
    Chandelier Exit：初始止损基于 ATR，回测时使用 trailing stop。

    做多: stop = highest_high - N*ATR
    做空: stop = lowest_low + N*ATR
    """
    params = atr_exit_params(
        ohlcv, entry_idx, direction,
        atr_period=atr_period,
        sl_atr_mult=atr_mult,
        tp_atr_mult=atr_mult * 1.5,
        max_hold_bars=max_hold_bars,
    )
    params.trailing_stop = True
    params.trailing_atr_mult = atr_mult
    return params


def simulate_exit(
    ohlcv: pd.DataFrame,
    entry_idx: int,
    direction: int,
    exit_params: ExitParams,
) -> dict:
    """
    模拟单笔交易的退出。

    Args:
        ohlcv: OHLCV 数据
        entry_idx: 入场 bar 位置
        direction: 1=多, -1=空
        exit_params: 退出参数

    Returns:
        dict: exit_idx, exit_price, exit_reason, return_pct, hold_bars
    """
    entry_price = ohlcv.iloc[entry_idx]["close"]
    n = len(ohlcv)
    end_idx = min(entry_idx + exit_params.max_hold_bars, n - 1)

    sl_pct = exit_params.stop_loss_pct
    tp_pct = exit_params.take_profit_pct

    # Trailing stop state
    if exit_params.trailing_stop:
        if direction == 1:
            trail_extreme = entry_price  # 跟踪最高价
        else:
            trail_extreme = entry_price  # 跟踪最低价

    for j in range(entry_idx + 1, end_idx + 1):
        bar = ohlcv.iloc[j]
        h, low = float(bar["high"]), float(bar["low"])

        if direction == 1:  # Long
            # Trailing stop update
            if exit_params.trailing_stop:
                trail_extreme = max(trail_extreme, h)
                # ATR trailing: 用入场时的 ATR%
                trail_sl_price = trail_extreme * (1 - sl_pct / 100)
                if low <= trail_sl_price:
                    exit_price = trail_sl_price
                    ret = (exit_price - entry_price) / entry_price * 100
                    return {
                        "exit_idx": j, "exit_price": exit_price,
                        "exit_reason": "trailing_stop",
                        "return_pct": ret, "hold_bars": j - entry_idx,
                    }

            # Fixed stop loss
            if (low - entry_price) / entry_price * 100 <= -sl_pct:
                exit_price = entry_price * (1 - sl_pct / 100)
                return {
                    "exit_idx": j, "exit_price": exit_price,
                    "exit_reason": "stop_loss",
                    "return_pct": -sl_pct, "hold_bars": j - entry_idx,
                }

            # Take profit
            if (h - entry_price) / entry_price * 100 >= tp_pct:
                exit_price = entry_price * (1 + tp_pct / 100)
                return {
                    "exit_idx": j, "exit_price": exit_price,
                    "exit_reason": "take_profit",
                    "return_pct": tp_pct, "hold_bars": j - entry_idx,
                }

        else:  # Short
            if exit_params.trailing_stop:
                trail_extreme = min(trail_extreme, low)
                trail_sl_price = trail_extreme * (1 + sl_pct / 100)
                if h >= trail_sl_price:
                    exit_price = trail_sl_price
                    ret = (entry_price - exit_price) / entry_price * 100
                    return {
                        "exit_idx": j, "exit_price": exit_price,
                        "exit_reason": "trailing_stop",
                        "return_pct": ret, "hold_bars": j - entry_idx,
                    }

            if (h - entry_price) / entry_price * 100 >= sl_pct:
                exit_price = entry_price * (1 + sl_pct / 100)
                return {
                    "exit_idx": j, "exit_price": exit_price,
                    "exit_reason": "stop_loss",
                    "return_pct": -sl_pct, "hold_bars": j - entry_idx,
                }

            if (entry_price - low) / entry_price * 100 >= tp_pct:
                exit_price = entry_price * (1 - tp_pct / 100)
                return {
                    "exit_idx": j, "exit_price": exit_price,
                    "exit_reason": "take_profit",
                    "return_pct": tp_pct, "hold_bars": j - entry_idx,
                }

    # Hold to end
    exit_price = ohlcv.iloc[end_idx]["close"]
    ret = (exit_price - entry_price) / entry_price * 100 * direction
    return {
        "exit_idx": end_idx, "exit_price": float(exit_price),
        "exit_reason": "hold",
        "return_pct": ret, "hold_bars": end_idx - entry_idx,
    }
