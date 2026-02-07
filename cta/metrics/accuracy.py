"""
反转预测准确性评估：胜率、精确率、盈亏比、连续盈亏等。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd


@dataclass
class TradeResult:
    """单笔交易结果。"""
    entry_time: Any
    exit_time: Any
    direction: int  # 1 = long, -1 = short
    entry_price: float
    exit_price: float
    return_pct: float
    is_win: bool
    holding_bars: int


@dataclass
class AccuracyMetrics:
    """预测准确性指标集合。"""
    # 基础统计
    total_signals: int = 0
    total_trades: int = 0

    # 胜率与准确率
    win_rate: float = 0.0
    precision_up: float = 0.0  # 预测上涨且实际上涨的比例
    precision_down: float = 0.0  # 预测下跌且实际下跌的比例
    overall_precision: float = 0.0

    # 盈亏分布
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0  # 总盈利 / 总亏损
    avg_return_pct: float = 0.0
    median_return_pct: float = 0.0

    # 连续性
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # 风险调整
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0

    # 反转幅度
    avg_reversal_magnitude_pct: float = 0.0  # 实际发生反转的平均幅度
    hit_target_rate: float = 0.0  # 达到目标反转幅度的比例

    # 时间维度
    avg_holding_bars: float = 0.0

    # 明细
    trade_returns: List[float] = field(default_factory=list)


def compute_prediction_accuracy(
    signal: pd.Series,
    bars: pd.DataFrame,
    hold_bars: int = 3,
    target_return_pct: float = 0.3,
) -> AccuracyMetrics:
    """
    计算预测准确性：
    - signal: 信号序列，+1/-1/0
    - bars: OHLCV 数据
    - hold_bars: 持仓 bar 数
    - target_return_pct: 目标反转幅度（%）
    """
    metrics = AccuracyMetrics()
    close = bars["close"].astype(float)

    signal_times = signal[signal != 0].index.tolist()
    metrics.total_signals = len(signal_times)

    if metrics.total_signals == 0:
        return metrics

    trades: List[TradeResult] = []
    up_correct = 0
    up_total = 0
    down_correct = 0
    down_total = 0

    for ts in signal_times:
        try:
            idx = bars.index.get_loc(ts)
        except KeyError:
            continue

        if idx + hold_bars >= len(bars):
            continue

        direction = int(signal.loc[ts])
        entry_price = float(close.iloc[idx + 1]) if idx + 1 < len(bars) else float(close.iloc[idx])
        exit_price = float(close.iloc[idx + hold_bars])

        # 计算收益
        if direction == 1:
            ret_pct = (exit_price - entry_price) / (entry_price + 1e-10) * 100
            up_total += 1
            if ret_pct > 0:
                up_correct += 1
        else:
            ret_pct = (entry_price - exit_price) / (entry_price + 1e-10) * 100
            down_total += 1
            if ret_pct > 0:
                down_correct += 1

        is_win = ret_pct > 0
        trades.append(TradeResult(
            entry_time=ts,
            exit_time=bars.index[idx + hold_bars],
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            return_pct=ret_pct,
            is_win=is_win,
            holding_bars=hold_bars,
        ))

    if not trades:
        return metrics

    metrics.total_trades = len(trades)
    returns = [t.return_pct for t in trades]
    metrics.trade_returns = returns

    # 胜率
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]
    metrics.win_rate = len(wins) / len(returns) if returns else 0

    # 精确率
    metrics.precision_up = up_correct / up_total if up_total > 0 else 0
    metrics.precision_down = down_correct / down_total if down_total > 0 else 0
    metrics.overall_precision = (up_correct + down_correct) / (up_total + down_total) if (up_total + down_total) > 0 else 0

    # 盈亏分布
    metrics.avg_win_pct = np.mean(wins) if wins else 0
    metrics.avg_loss_pct = np.mean(losses) if losses else 0
    total_win = sum(wins)
    total_loss = abs(sum(losses)) if losses else 1e-10
    metrics.profit_factor = total_win / total_loss if total_loss > 0 else float('inf')
    metrics.avg_return_pct = np.mean(returns)
    metrics.median_return_pct = np.median(returns)

    # 连续盈亏
    is_win_series = [r > 0 for r in returns]
    metrics.max_consecutive_wins = _max_consecutive(is_win_series, True)
    metrics.max_consecutive_losses = _max_consecutive(is_win_series, False)

    # Sharpe & Sortino
    ret_std = np.std(returns) if len(returns) > 1 else 1e-10
    metrics.sharpe = metrics.avg_return_pct / (ret_std + 1e-10) * np.sqrt(len(returns))
    downside_returns = [r for r in returns if r < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 1 else 1e-10
    metrics.sortino = metrics.avg_return_pct / (downside_std + 1e-10) * np.sqrt(len(returns))

    # Max drawdown
    cum = np.cumsum(returns)
    running_max = np.maximum.accumulate(cum)
    drawdown = running_max - cum
    metrics.max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

    # 反转幅度
    reversal_magnitudes = [abs(t.return_pct) for t in trades if t.is_win]
    metrics.avg_reversal_magnitude_pct = np.mean(reversal_magnitudes) if reversal_magnitudes else 0
    hit_target = sum(1 for t in trades if abs(t.return_pct) >= target_return_pct)
    metrics.hit_target_rate = hit_target / len(trades) if trades else 0

    # 平均持仓
    metrics.avg_holding_bars = np.mean([t.holding_bars for t in trades])

    return metrics


def _max_consecutive(series: List[bool], target: bool) -> int:
    """计算连续出现 target 的最大次数。"""
    max_count = 0
    current = 0
    for val in series:
        if val == target:
            current += 1
            max_count = max(max_count, current)
        else:
            current = 0
    return max_count


def compute_trade_metrics(
    signal: pd.Series,
    bars: pd.DataFrame,
    hold_bars_options: List[int] = None,
) -> Dict[str, AccuracyMetrics]:
    """对不同持仓周期计算指标。"""
    if hold_bars_options is None:
        hold_bars_options = [1, 3, 5, 10]

    results = {}
    for hb in hold_bars_options:
        results[f"hold_{hb}"] = compute_prediction_accuracy(signal, bars, hold_bars=hb)
    return results


def evaluate_signal_quality(
    signal: pd.Series,
    bars: pd.DataFrame,
    hold_bars: int = 3,
) -> Dict[str, Any]:
    """
    综合评估信号质量，返回易读的字典。
    """
    m = compute_prediction_accuracy(signal, bars, hold_bars=hold_bars)
    return {
        "total_signals": m.total_signals,
        "total_trades": m.total_trades,
        "win_rate": round(m.win_rate * 100, 2),  # %
        "precision_up": round(m.precision_up * 100, 2),
        "precision_down": round(m.precision_down * 100, 2),
        "overall_precision": round(m.overall_precision * 100, 2),
        "avg_win_pct": round(m.avg_win_pct, 4),
        "avg_loss_pct": round(m.avg_loss_pct, 4),
        "profit_factor": round(m.profit_factor, 2),
        "avg_return_pct": round(m.avg_return_pct, 4),
        "sharpe": round(m.sharpe, 3),
        "sortino": round(m.sortino, 3),
        "max_drawdown": round(m.max_drawdown, 4),
        "max_consecutive_wins": m.max_consecutive_wins,
        "max_consecutive_losses": m.max_consecutive_losses,
        "avg_reversal_magnitude": round(m.avg_reversal_magnitude_pct, 4),
        "hit_target_rate": round(m.hit_target_rate * 100, 2),
    }
