"""
Orderbook 生命周期指标（需要 orderbook snapshot 序列）。

保留的指标：
  - depth_change_rate: 各侧深度变化率（增加了 std/momentum 保留时序信息）
  - depth_resilience: 深度冲击后的恢复速度（仅使用历史数据，无未来泄漏）
  - level_thickness_profile: 各档位厚度占比

已移除的指标：
  - refill_frequency: ❌ 虚假指标。top-N 聚合深度在 levels 被吃后会因
    下层 levels 滑升而自动恢复，并非真正的做市商补单。
    要准确衡量补单需要追踪同一价位的订单变化，snapshot 数据无法做到。
  - cancel_rate: ❌ 无法区分撤单与被成交。当 taker 部分消耗 bid1 时，
    mid 不变、depth 下降，与撤单表现完全一致。
    没有 order-level 数据无法准确计算。

机制：
  高 depth_change_rate → 补单活跃 → 流动性充足
  高 resilience → 冲击后快速恢复 → 均值回归信号
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _get_depth(book: pd.DataFrame, side: str, levels: int) -> pd.Series:
    """获取某侧的总深度。"""
    avail = [i for i in range(levels) if f"{side}_{i}_size" in book.columns]
    if not avail:
        return pd.Series(0, index=book.index)
    return sum(book[f"{side}_{i}_size"].astype(float) for i in avail)


def depth_change_rate(
    book: pd.DataFrame,
    levels: int = 5,
    window: int = 10,
) -> pd.DataFrame:
    """
    Bid/Ask 侧深度的滚动变化率，保留时序结构信息。

    输出：
      - {side}_depth_change_rate:  pct_change 的滚动均值（方向信号）
      - {side}_depth_change_std:   pct_change 的滚动标准差（波动性）
      - {side}_depth_change_mom:   最近 1 步 vs 滚动均值的偏离（加速/减速）
      - depth_change_asymmetry:    bid 变化率 - ask 变化率（方向不平衡）
    """
    bid_depth = _get_depth(book, "bid", levels)
    ask_depth = _get_depth(book, "ask", levels)

    bid_pct = bid_depth.pct_change()
    ask_pct = ask_depth.pct_change()

    bid_mean = bid_pct.rolling(window, min_periods=2).mean()
    ask_mean = ask_pct.rolling(window, min_periods=2).mean()
    bid_std = bid_pct.rolling(window, min_periods=2).std()
    ask_std = ask_pct.rolling(window, min_periods=2).std()

    # momentum: 当前 pct_change 与滚动均值之差（正 = 加速补单/恶化）
    bid_mom = bid_pct - bid_mean
    ask_mom = ask_pct - ask_mean

    return pd.DataFrame({
        "bid_depth_change_rate": bid_mean,
        "ask_depth_change_rate": ask_mean,
        "bid_depth_change_std": bid_std,
        "ask_depth_change_std": ask_std,
        "bid_depth_change_mom": bid_mom,
        "ask_depth_change_mom": ask_mom,
        "depth_change_asymmetry": bid_mean - ask_mean,
    }, index=book.index)


def depth_resilience(
    book: pd.DataFrame,
    levels: int = 5,
    shock_window: int = 3,
    recovery_window: int = 10,
    shock_threshold: float = -0.2,
) -> pd.DataFrame:
    """
    深度韧性：深度被冲击后的恢复速度。

    **仅使用历史数据**（无未来泄漏）。

    算法：
      1. 检测冲击：rolling(shock_window).min pct_change < shock_threshold
      2. 冲击后在 recovery_window 内的恢复程度：
         recovery = (current_depth - min_depth_in_window) / depth_before_shock
      3. 滚动取 recovery 均值

    高韧性 → 做市商迅速补单 → 反转信号
    低韧性 → 单边吃穿 → 趋势继续
    """
    bid_depth = _get_depth(book, "bid", levels)
    ask_depth = _get_depth(book, "ask", levels)

    def _resilience(depth):
        # 过去 shock_window 内的最小深度变化
        pct = depth.pct_change()
        rolling_min_pct = pct.rolling(shock_window, min_periods=1).min()

        # 标记冲击发生
        is_shock = rolling_min_pct < shock_threshold

        # 冲击前深度 = shock_window 前的值
        depth_before = depth.shift(shock_window)

        # 冲击时的最低深度 = 近 shock_window 内最低
        depth_min = depth.rolling(shock_window, min_periods=1).min()

        # 恢复 = 当前深度相对最低点的恢复占冲击前深度的比例
        recovery = (depth - depth_min) / (depth_before.abs() + 1e-10)

        # 仅在冲击后才有意义，用 rolling mean 平滑
        res = pd.Series(np.nan, index=depth.index, dtype=float)
        res[is_shock] = recovery[is_shock]
        res = res.rolling(recovery_window, min_periods=1).mean()
        return res.fillna(0)

    return pd.DataFrame({
        "bid_resilience": _resilience(bid_depth),
        "ask_resilience": _resilience(ask_depth),
    }, index=book.index)


def level_thickness_profile(
    book: pd.DataFrame,
    levels: int = 10,
) -> pd.DataFrame:
    """
    各档位厚度占比，衡量流动性集中度。

    如果 top 3 档占了 >80% 的深度 → 流动性集中在近档 → 容易被冲击。
    """
    avail_bid = [i for i in range(levels) if f"bid_{i}_size" in book.columns]
    avail_ask = [i for i in range(levels) if f"ask_{i}_size" in book.columns]

    if not avail_bid or not avail_ask:
        return pd.DataFrame(index=book.index)

    bid_total = sum(book[f"bid_{i}_size"].astype(float) for i in avail_bid) + 1e-10
    ask_total = sum(book[f"ask_{i}_size"].astype(float) for i in avail_ask) + 1e-10

    # Top 3 concentration
    top3_bid = sum(book[f"bid_{i}_size"].astype(float) for i in avail_bid[:3])
    top3_ask = sum(book[f"ask_{i}_size"].astype(float) for i in avail_ask[:3])

    return pd.DataFrame({
        "bid_top3_concentration": top3_bid / bid_total,
        "ask_top3_concentration": top3_ask / ask_total,
        "bid_total_depth": bid_total,
        "ask_total_depth": ask_total,
    }, index=book.index)
