"""
收益率与动量特征（基于 OHLCV bar）。

从 OHLCV 数据计算：
  - return_1: 单 bar 收益率 (bps)
  - log_return: 对数收益率
  - return_sum_w: 累计收益 (动量)
  - return_std_w: 收益波动率
  - return_skew_w: 收益偏度
  - return_kurt_w: 收益峰度 (尾部风险)
  - return_zscore_w: 收益 z-score (均值回归信号)

所有特征在指定 rolling windows 上计算。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_returns(
    ohlcv: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    从 OHLCV 计算收益率与动量特征。

    Args:
        ohlcv: DataFrame with columns: open, high, low, close, volume
        windows: rolling 窗口列表 (bar 数)。默认 [3, 5, 10, 20]

    Returns:
        DataFrame，index 与 ohlcv 一致，每列一个特征
    """
    if windows is None:
        windows = [3, 5, 10, 20]

    c = ohlcv["close"].astype(float)

    f = pd.DataFrame(index=ohlcv.index)

    # ── 基础收益 ──
    f["return_1"] = c.pct_change() * 10000  # bps
    f["log_return"] = np.log(c / c.shift(1))

    # ── Rolling 特征 ──
    for w in windows:
        # 累计动量
        f[f"return_sum_{w}"] = f["return_1"].rolling(w, min_periods=1).sum()
        # 波动率
        f[f"return_std_{w}"] = f["return_1"].rolling(w, min_periods=2).std()
        # 偏度 (方向性倾斜)
        f[f"return_skew_{w}"] = f["return_1"].rolling(w, min_periods=3).skew()
        # 峰度 (尾部风险 / 跳跃频率)
        if w >= 5:
            f[f"return_kurt_{w}"] = f["return_1"].rolling(w, min_periods=5).kurt()
        # Z-score (均值回复信号)
        roll_mean = f["return_1"].rolling(w, min_periods=2).mean()
        roll_std = f[f"return_std_{w}"]
        f[f"return_zscore_{w}"] = (f["return_1"] - roll_mean) / (roll_std + 1e-10)

    f = f.replace([np.inf, -np.inf], np.nan).fillna(0)
    return f
