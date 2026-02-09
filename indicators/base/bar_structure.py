"""
Bar 形态结构特征（基于 OHLCV bar）。

从 OHLCV 数据提取 K 线形态信息：
  - bar_range_bps: 振幅 (bps)
  - body_ratio: 实体占比 |C-O| / (H-L)
  - bar_direction: 阴阳 sign(C-O)
  - upper_wick / lower_wick: 上下影线占比
  - bar_efficiency: |C-O| / 路径长度 (趋势效率)
  - volume_per_range: 单位振幅的成交量 (流动性指标)

Rolling 统计 (mean/std) 按指定 windows 计算。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_bar_structure(
    ohlcv: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    从 OHLCV 计算 bar 形态结构特征。

    Args:
        ohlcv: DataFrame with columns: open, high, low, close, volume
               可选: n_trades, buy_volume, sell_volume
        windows: rolling 窗口列表 (bar 数)。默认 [3, 5, 10, 20]

    Returns:
        DataFrame，index 与 ohlcv 一致
    """
    if windows is None:
        windows = [3, 5, 10, 20]

    c = ohlcv["close"].astype(float)
    o = ohlcv["open"].astype(float)
    h = ohlcv["high"].astype(float)
    lo = ohlcv["low"].astype(float)
    v = ohlcv["volume"].astype(float)
    bv = ohlcv["buy_volume"].astype(float) if "buy_volume" in ohlcv.columns else v * 0.5
    sv = ohlcv["sell_volume"].astype(float) if "sell_volume" in ohlcv.columns else v * 0.5
    nt = ohlcv["n_trades"].astype(float) if "n_trades" in ohlcv.columns else pd.Series(0, index=ohlcv.index)

    hl = h - lo
    mid = (h + lo) / 2

    f = pd.DataFrame(index=ohlcv.index)

    # ── 基础 bar 形态 ──
    f["bar_range_bps"] = hl / (mid + 1e-10) * 10000
    f["body_ratio"] = (c - o).abs() / (hl + 1e-10)
    f["bar_direction"] = np.sign(c - o)

    # 上下影线占比
    upper_wick = h - np.maximum(c, o)
    lower_wick = np.minimum(c, o) - lo
    f["upper_wick_pct"] = upper_wick / (hl + 1e-10)
    f["lower_wick_pct"] = lower_wick / (hl + 1e-10)

    # Bar 效率: |net move| / total path (趋势 vs 震荡)
    net_move = (c - o).abs()
    total_path = upper_wick + lower_wick + net_move
    f["bar_efficiency"] = net_move / (total_path + 1e-10)

    # 成交量 / 振幅 (流动性指标: 大 → 振幅每 bps 需要更多成交量)
    f["volume_per_range"] = v / (f["bar_range_bps"] + 1e-10)

    # Volume imbalance at bar level
    f["volume_imbalance"] = (bv - sv) / (v + 1e-10)

    # Trades 相关
    f["n_trades"] = nt

    # ── Rolling 统计 ──
    roll_base = [
        "bar_range_bps", "body_ratio", "bar_efficiency",
        "volume_per_range", "volume_imbalance", "n_trades",
    ]
    for w in windows:
        for col in roll_base:
            if col in f.columns:
                f[f"{col}_ma_{w}"] = f[col].rolling(w, min_periods=1).mean()

        # 额外: 方向一致性 (连续同方向 bar 的比例)
        f[f"direction_consistency_{w}"] = f["bar_direction"].rolling(w, min_periods=1).mean()

        # Volume rolling
        f[f"volume_ma_{w}"] = v.rolling(w, min_periods=1).mean()
        f[f"volume_ratio_{w}"] = v / (f[f"volume_ma_{w}"] + 1e-10)
        f[f"trades_ma_{w}"] = nt.rolling(w, min_periods=1).mean()

        # Signed flow
        signed_vol = bv - sv
        f[f"signed_flow_{w}"] = signed_vol.rolling(w, min_periods=1).sum()
        f[f"signed_flow_norm_{w}"] = f[f"signed_flow_{w}"] / (v.rolling(w, min_periods=1).sum() + 1e-10)

    f = f.replace([np.inf, -np.inf], np.nan).fillna(0)
    return f
