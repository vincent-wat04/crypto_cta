"""
通用回测引擎。

支持：
1. 单指标回测（indicator + triple barrier labels）
2. 复合策略回测（多指标组合 + 止损止盈）
"""
from __future__ import annotations

from typing import Dict, Any, Optional, Callable

import pandas as pd

from .labeling import triple_barrier_labels
from .metrics import compute_backtest_metrics


# ─────────────────────────────────────────────────────────
# 单指标回测
# ─────────────────────────────────────────────────────────

def backtest_single_indicator(
    indicator: pd.Series,
    prices: pd.Series,
    signal_fn: Optional[Callable] = None,
    long_threshold: float = 0.0,
    short_threshold: float = 0.0,
    hold_bars: int = 30,
    stop_loss_pct: float = 1.0,
    take_profit_pct: float = 2.0,
    cooldown_bars: int = 5,
    use_triple_barrier: bool = True,
    tb_upper_pct: float = 2.0,
    tb_lower_pct: float = 1.5,
    tb_max_bars: int = 60,
) -> Dict[str, Any]:
    """
    单指标回测。

    Args:
        indicator: 指标值序列（index 与 prices 对齐）
        prices: 价格序列
        signal_fn: 自定义信号函数 indicator_value -> {1, -1, 0}
                   如果为 None，使用 long_threshold / short_threshold
        long_threshold: 指标值 > 此值时做多
        short_threshold: 指标值 < 此值时做空
        hold_bars: 最大持仓 bar 数
        stop_loss_pct: 止损百分比
        take_profit_pct: 止盈百分比
        cooldown_bars: 信号冷却期
        use_triple_barrier: 是否用 Triple Barrier 作为标签评估
        tb_upper_pct / tb_lower_pct / tb_max_bars: Triple Barrier 参数

    Returns:
        Dict with metrics + results_df + label_analysis
    """
    # 对齐
    common_idx = indicator.index.intersection(prices.index)
    indicator = indicator.loc[common_idx]
    prices = prices.loc[common_idx]

    # 生成信号
    if signal_fn is not None:
        signals = indicator.apply(signal_fn)
    else:
        signals = pd.Series(0, index=common_idx)
        signals[indicator > long_threshold] = 1
        signals[indicator < short_threshold] = -1

    signal_times = signals[signals != 0]

    if signal_times.empty:
        return {"metrics": compute_backtest_metrics(pd.DataFrame()), "results_df": pd.DataFrame()}

    # Triple Barrier 评估
    tb = None
    if use_triple_barrier:
        tb = triple_barrier_labels(
            prices, upper_pct=tb_upper_pct, lower_pct=tb_lower_pct, max_bars=tb_max_bars,
        )

    # 逐信号模拟
    arr = prices.values.astype(float)
    idx_map = {t: i for i, t in enumerate(prices.index)}

    results = []
    last_exit_idx = -cooldown_bars

    for ts, direction in signal_times.items():
        if ts not in idx_map:
            continue
        entry_i = idx_map[ts]

        # Cooldown
        if entry_i - last_exit_idx < cooldown_bars:
            continue

        entry_price = arr[entry_i]
        direction = int(direction)

        # Bar-by-bar simulation
        exit_price = None
        exit_reason = "hold"
        end_i = min(entry_i + hold_bars, len(arr) - 1)

        for j in range(entry_i + 1, end_i + 1):
            p = arr[j]
            if direction == 1:
                ret = (p - entry_price) / entry_price * 100
                if ret <= -stop_loss_pct:
                    exit_price = entry_price * (1 - stop_loss_pct / 100)
                    exit_reason = "stop_loss"; break
                if ret >= take_profit_pct:
                    exit_price = entry_price * (1 + take_profit_pct / 100)
                    exit_reason = "take_profit"; break
            else:
                ret = (entry_price - p) / entry_price * 100
                if ret <= -stop_loss_pct:
                    exit_price = entry_price * (1 + stop_loss_pct / 100)
                    exit_reason = "stop_loss"; break
                if ret >= take_profit_pct:
                    exit_price = entry_price * (1 - take_profit_pct / 100)
                    exit_reason = "take_profit"; break

        if exit_price is None:
            exit_price = arr[end_i]

        last_exit_idx = end_i if exit_price else entry_i + hold_bars

        ret_pct = ((exit_price - entry_price) / entry_price * 100 * direction)

        row = {
            "entry_time": ts,
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "return_pct": ret_pct,
            "is_win": ret_pct > 0,
            "exit_reason": exit_reason,
            "indicator_value": float(indicator.loc[ts]),
        }

        # Triple Barrier alignment
        if use_triple_barrier and tb is not None and ts in tb.index:
            row["tb_label"] = int(tb.loc[ts, "label"])
            row["tb_barrier"] = tb.loc[ts, "barrier_hit"]

        results.append(row)

    rdf = pd.DataFrame(results) if results else pd.DataFrame()
    metrics = compute_backtest_metrics(rdf)

    # Label analysis
    label_analysis = {}
    if use_triple_barrier and not rdf.empty and "tb_label" in rdf.columns:
        for lbl in [1, 0, -1]:
            subset = rdf[rdf["tb_label"] == lbl]
            label_analysis[f"tb_{lbl}_count"] = len(subset)
            label_analysis[f"tb_{lbl}_avg_return"] = round(subset["return_pct"].mean(), 4) if len(subset) > 0 else 0

    return {
        "metrics": metrics,
        "results_df": rdf,
        "n_signals": len(signal_times),
        "n_trades": len(rdf),
        "label_analysis": label_analysis,
    }


# ─────────────────────────────────────────────────────────
# 复合策略回测（FVG、微观结构等）
# ─────────────────────────────────────────────────────────

def backtest_composite(
    feature_df: pd.DataFrame,
    model,
    ohlcv: pd.DataFrame,
    hold_bars: int = 10,
    min_proba: float = 0.6,
    stop_loss_pct: float = 0.2,
    take_profit_pct: float = 0.4,
    cooldown_bars: int = 3,
) -> Dict[str, Any]:
    """
    复合策略回测（如 FVG + ML 模型）。
    与 cta/backtest.py 中 backtest_fvg 等价，统一入口。
    """
    predictions = model.predict(feature_df)
    signals = predictions[predictions["pred_proba"] >= min_proba].copy()

    if signals.empty:
        return {"metrics": compute_backtest_metrics(pd.DataFrame()), "results_df": pd.DataFrame()}

    arr_close = ohlcv["close"].values if "close" in ohlcv.columns else ohlcv.iloc[:, 3].values
    arr_high = ohlcv["high"].values if "high" in ohlcv.columns else ohlcv.iloc[:, 1].values
    arr_low = ohlcv["low"].values if "low" in ohlcv.columns else ohlcv.iloc[:, 2].values

    results = []
    last_exit_idx = -cooldown_bars

    for _, row in signals.iterrows():
        ts = pd.to_datetime(row["timestamp"])
        pred_dir = row["pred_fill"]

        if ts not in ohlcv.index:
            continue

        entry_idx = ohlcv.index.get_loc(ts)
        if entry_idx - last_exit_idx < cooldown_bars:
            continue

        entry_price = arr_close[entry_idx]
        direction = 1 if pred_dir == 1 else -1

        exit_price = None
        exit_reason = "hold"
        end_idx = min(entry_idx + hold_bars, len(ohlcv) - 1)

        for j in range(entry_idx + 1, end_idx + 1):
            if direction == 1:
                if (arr_low[j] - entry_price) / entry_price * 100 <= -stop_loss_pct:
                    exit_price = entry_price * (1 - stop_loss_pct / 100)
                    exit_reason = "stop_loss"; end_idx = j; break
                if (arr_high[j] - entry_price) / entry_price * 100 >= take_profit_pct:
                    exit_price = entry_price * (1 + take_profit_pct / 100)
                    exit_reason = "take_profit"; end_idx = j; break
            else:
                if (arr_high[j] - entry_price) / entry_price * 100 >= stop_loss_pct:
                    exit_price = entry_price * (1 + stop_loss_pct / 100)
                    exit_reason = "stop_loss"; end_idx = j; break
                if (entry_price - arr_low[j]) / entry_price * 100 >= take_profit_pct:
                    exit_price = entry_price * (1 - take_profit_pct / 100)
                    exit_reason = "take_profit"; end_idx = j; break

        if exit_price is None:
            exit_price = arr_close[end_idx]

        last_exit_idx = end_idx
        ret_pct = (exit_price - entry_price) / entry_price * 100 * direction

        results.append({
            "entry_time": ts, "direction": direction,
            "entry_price": entry_price, "exit_price": exit_price,
            "return_pct": ret_pct, "is_win": ret_pct > 0,
            "exit_reason": exit_reason,
            "cluster": row.get("cluster", -1),
            "pred_proba": row["pred_proba"],
        })

    rdf = pd.DataFrame(results) if results else pd.DataFrame()
    return {"metrics": compute_backtest_metrics(rdf), "results_df": rdf}
