"""
Orderbook 生命周期指标（需要 orderbook snapshot 序列）。

衡量挂单的「寿命」和「补充」行为：
  - depth_change_rate: 各档位深度变化率
  - refill_frequency: 某侧被消耗后的补充速度
  - order_lifespan_proxy: 挂单存活时间的代理指标
  - cancel_rate: 撤单率（深度减少的频率）
  - depth_resilience: 深度冲击后的恢复速度

机制：
  高 refill → 做市商积极补单 → 强支撑
  高 cancel → 挂单快速撤走 → 虚假流动性
  高 resilience → 冲击后快速恢复 → 均值回归信号
"""
from __future__ import annotations

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
    Bid/Ask 侧深度的滚动变化率。

    change_rate = (depth_t - depth_{t-1}) / depth_{t-1}

    正值 → 补单（流动性增加）
    负值 → 被消耗或撤单（流动性减少）
    """
    bid_depth = _get_depth(book, "bid", levels)
    ask_depth = _get_depth(book, "ask", levels)

    bid_change = bid_depth.pct_change().rolling(window, min_periods=2).mean()
    ask_change = ask_depth.pct_change().rolling(window, min_periods=2).mean()

    return pd.DataFrame({
        "bid_depth_change_rate": bid_change,
        "ask_depth_change_rate": ask_change,
        "depth_change_asymmetry": bid_change - ask_change,
    }, index=book.index)


def refill_frequency(
    book: pd.DataFrame,
    levels: int = 5,
    window: int = 20,
    depletion_threshold: float = -0.3,
) -> pd.DataFrame:
    """
    补充频率：深度下降 >30% 后，在后续 N 个 snapshot 内恢复的比例。

    高频补充 → 做市商积极 → 价格有支撑
    低频补充 → 做市商撤退 → 流动性真空

    Args:
        depletion_threshold: 深度变化 < 此值视为 "被消耗"
    """
    bid_depth = _get_depth(book, "bid", levels)
    ask_depth = _get_depth(book, "ask", levels)

    def _refill_rate(depth_series):
        change = depth_series.pct_change()
        is_depleted = change < depletion_threshold
        # 看 depletion 后下一个 snapshot 是否恢复
        next_change = change.shift(-1)
        refills_after_depletion = (is_depleted & (next_change > 0)).astype(float)
        return refills_after_depletion.rolling(window, min_periods=3).mean()

    return pd.DataFrame({
        "bid_refill_freq": _refill_rate(bid_depth),
        "ask_refill_freq": _refill_rate(ask_depth),
    }, index=book.index)


def cancel_rate(
    book: pd.DataFrame,
    levels: int = 5,
    window: int = 20,
) -> pd.DataFrame:
    """
    撤单率：深度减少（非成交导致）的频率。

    近似方法：depth 下降但 mid price 未变化 → 可能是撤单而非成交。
    """
    bid_depth = _get_depth(book, "bid", levels)
    ask_depth = _get_depth(book, "ask", levels)

    has_mid = "bid_0_price" in book.columns and "ask_0_price" in book.columns
    if has_mid:
        mid = (book["bid_0_price"].astype(float) + book["ask_0_price"].astype(float)) / 2
        mid_unchanged = mid.diff().abs() < 1e-10
    else:
        mid_unchanged = pd.Series(True, index=book.index)

    bid_dropped = bid_depth.diff() < 0
    ask_dropped = ask_depth.diff() < 0

    # 撤单 = 深度下降 + mid 不变
    bid_cancel = (bid_dropped & mid_unchanged).astype(float)
    ask_cancel = (ask_dropped & mid_unchanged).astype(float)

    return pd.DataFrame({
        "bid_cancel_rate": bid_cancel.rolling(window, min_periods=3).mean(),
        "ask_cancel_rate": ask_cancel.rolling(window, min_periods=3).mean(),
    }, index=book.index)


def depth_resilience(
    book: pd.DataFrame,
    levels: int = 5,
    shock_window: int = 3,
    recovery_window: int = 10,
) -> pd.DataFrame:
    """
    深度韧性：深度被冲击后的恢复速度。

    resilience = (depth after recovery_window - depth at shock) / depth before shock

    高韧性 → 做市商迅速补单 → 反转信号
    低韧性 → 单边吃穿 → 趋势继续
    """
    bid_depth = _get_depth(book, "bid", levels)
    ask_depth = _get_depth(book, "ask", levels)

    def _resilience(depth):
        before = depth.shift(shock_window)
        at_shock = depth
        after = depth.shift(-recovery_window)

        shock_size = (at_shock - before) / (before + 1e-10)
        recovery = (after - at_shock) / (before + 1e-10)

        # 只在冲击发生时有意义
        is_shock = shock_size < -0.2
        res = pd.Series(0, index=depth.index, dtype=float)
        res[is_shock] = recovery[is_shock]
        return res

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
