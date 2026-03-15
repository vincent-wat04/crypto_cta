"""
Price Impact 指标。

# TODO:
这是基于 takerID 合并后的 taker trades 指标，而非基于 raw trades

原理：衡量单位成交量对价格的推动力度。
  - tick_price_impact: |ΔPrice| / Volume，在滚动窗口内计算
  - volume_weighted_impact: 成交量加权的价格冲击
  - kyles_lambda: Kyle's Lambda（价格变化对净订单流的回归系数）

机制：大量市价单成交时，价格单方向剧烈波动，impact 会飙升。
用途：高 impact → 流动性被消耗 → 可能出现反转。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def tick_price_impact(
    trades: pd.DataFrame,
    window_ms: int = 5000,
) -> pd.Series:
    """
    滚动窗口内的 tick-level price impact：|ΔPrice| / ΣVolume。

    Returns:
        Series (index=timestamp), values = impact
    """
    ts = trades.set_index("timestamp").sort_index()
    ts = ts[~ts.index.duplicated(keep="last")]

    window = pd.Timedelta(milliseconds=window_ms)
    prices = ts["price"]
    volumes = ts["amount"]

    impact = (
        prices.diff().abs().rolling(window).sum()
        / (volumes.rolling(window).sum() + 1e-10)
    )
    impact.name = "price_impact"
    return impact


def volume_weighted_impact(
    trades: pd.DataFrame,
    window_ms: int = 10000,
) -> pd.Series:
    """
    成交量加权 price impact：Σ(|ΔP_i| * V_i) / ΣV_i。
    """
    ts = trades.set_index("timestamp").sort_index()
    ts = ts[~ts.index.duplicated(keep="last")]

    window = pd.Timedelta(milliseconds=window_ms)
    abs_dp = ts["price"].diff().abs()
    weighted = (abs_dp * ts["amount"]).rolling(window).sum()
    total_vol = ts["amount"].rolling(window).sum() + 1e-10

    result = weighted / total_vol
    result.name = "vw_impact"
    return result

# TODO: vwap - starting price

# TODO: vwap dist orthogonised w volume 


def kyles_lambda(
    trades: pd.DataFrame,
    window_trades: int = 100,
) -> pd.Series:
    """
    Kyle's Lambda: 价格变化对签名订单流的回归系数。

    signed_flow = Σ (side * amount)，side: buy=+1, sell=-1
    lambda = Cov(ΔP, signed_flow) / Var(signed_flow)
    """
    ts = trades.copy().sort_values("timestamp")
    sign = ts["side"].map({"buy": 1, "sell": -1}).fillna(0)
    ts["signed_flow"] = sign * ts["amount"]
    ts["dp"] = ts["price"].diff()

    def _lambda(window):
        sf = window["signed_flow"]
        dp = window["dp"]
        if sf.var() < 1e-15:
            return 0
        return np.cov(dp, sf)[0, 1] / (sf.var() + 1e-10)

    # 手动计算滚动 lambda
    sf = ts.set_index("timestamp")["signed_flow"]
    dp = ts.set_index("timestamp")["dp"]
    sf = sf[~sf.index.duplicated(keep="last")]
    dp = dp[~dp.index.duplicated(keep="last")]

    cov = sf.rolling(window_trades, min_periods=20).cov(dp)
    var = sf.rolling(window_trades, min_periods=20).var() + 1e-10
    lam = cov / var
    lam.name = "kyles_lambda"
    return lam
