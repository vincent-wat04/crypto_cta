"""
FVG (Fair Value Gap) 检测器。

FVG 定义：
  三根 K 线序列中，中间一根的 body 在 candle1.high 和 candle3.low 之间
  留下了一个未被交易覆盖的价格真空区域（gap）。

类别：
  - Bullish FVG: candle3.low > candle1.high → 价格向上跳空
  - Bearish FVG: candle1.low > candle3.high → 价格向下跳空
  
最适合的频段：
  - 1min ~ 5min K 线（高频环境下 FVG 频繁出现且填充速度快）
  - 用逐笔成交 resample 得到精确的 OHLCV
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd


class FVGType(Enum):
    """FVG 类型。"""
    BULLISH = "bullish"      # 看涨 FVG（向上跳空）
    BEARISH = "bearish"      # 看跌 FVG（向下跳空）


class FVGStatus(Enum):
    """FVG 状态。"""
    OPEN = "open"            # 未填充
    PARTIAL = "partial"      # 部分填充
    FILLED = "filled"        # 完全填充
    EXPIRED = "expired"      # 超时未填充


@dataclass
class FVG:
    """单个 FVG 事件。"""
    timestamp: pd.Timestamp          # FVG 产生时间（candle3 收盘）
    fvg_type: FVGType
    gap_high: float                   # gap 上沿
    gap_low: float                    # gap 下沿
    gap_size: float                   # gap 大小 (absolute)
    gap_size_pct: float               # gap 大小 (%)
    mid_price: float                  # gap 中点
    
    # 产生 FVG 的 K 线信息
    candle1_idx: int = 0
    candle2_volume: float = 0.0       # 中间 K 线成交量
    candle2_range: float = 0.0        # 中间 K 线振幅
    
    # 后续状态
    status: FVGStatus = FVGStatus.OPEN
    fill_time: Optional[pd.Timestamp] = None
    fill_bars: int = 0                # 填充所用 K 线数
    max_extension: float = 0.0        # 填充前最大偏离
    
    # 附加特征
    metadata: Dict[str, Any] = field(default_factory=dict)


def detect_fvg(
    ohlcv: pd.DataFrame,
    min_gap_pct: float = 0.02,
    max_lookback_bars: int = 100,
) -> List[FVG]:
    """
    从 OHLCV 数据中检测所有 FVG。
    
    Args:
        ohlcv: DataFrame with columns [open, high, low, close, volume]，index 为时间
        min_gap_pct: 最小 gap 百分比阈值（过滤噪音）
        max_lookback_bars: 填充检测的最大回溯 K 线数
    
    Returns:
        List of FVG events with fill status
    """
    if len(ohlcv) < 3:
        return []
    
    df = ohlcv.copy()
    fvgs: List[FVG] = []
    
    for i in range(2, len(df)):
        c1 = df.iloc[i - 2]  # candle 1
        c2 = df.iloc[i - 1]  # candle 2 (中间)
        c3 = df.iloc[i]      # candle 3
        
        mid_price = c2["close"]
        
        # Bullish FVG: candle3.low > candle1.high
        if c3["low"] > c1["high"]:
            gap_low = c1["high"]
            gap_high = c3["low"]
            gap_size = gap_high - gap_low
            gap_pct = gap_size / mid_price * 100
            
            if gap_pct >= min_gap_pct:
                fvg = FVG(
                    timestamp=df.index[i],
                    fvg_type=FVGType.BULLISH,
                    gap_high=gap_high,
                    gap_low=gap_low,
                    gap_size=gap_size,
                    gap_size_pct=gap_pct,
                    mid_price=mid_price,
                    candle1_idx=i - 2,
                    candle2_volume=c2.get("volume", 0),
                    candle2_range=(c2["high"] - c2["low"]) / mid_price * 100,
                )
                fvgs.append(fvg)
        
        # Bearish FVG: candle1.low > candle3.high
        if c1["low"] > c3["high"]:
            gap_low = c3["high"]
            gap_high = c1["low"]
            gap_size = gap_high - gap_low
            gap_pct = gap_size / mid_price * 100
            
            if gap_pct >= min_gap_pct:
                fvg = FVG(
                    timestamp=df.index[i],
                    fvg_type=FVGType.BEARISH,
                    gap_high=gap_high,
                    gap_low=gap_low,
                    gap_size=gap_size,
                    gap_size_pct=gap_pct,
                    mid_price=mid_price,
                    candle1_idx=i - 2,
                    candle2_volume=c2.get("volume", 0),
                    candle2_range=(c2["high"] - c2["low"]) / mid_price * 100,
                )
                fvgs.append(fvg)
    
    # 检测填充状态
    _check_fill_status(fvgs, df, max_lookback_bars)
    
    return fvgs


def _check_fill_status(
    fvgs: List[FVG],
    ohlcv: pd.DataFrame,
    max_bars: int,
) -> None:
    """检测每个 FVG 是否被填充。"""
    for fvg in fvgs:
        fvg_idx = ohlcv.index.get_loc(fvg.timestamp)
        end_idx = min(fvg_idx + max_bars + 1, len(ohlcv))
        
        max_ext = 0.0
        
        for j in range(fvg_idx + 1, end_idx):
            bar = ohlcv.iloc[j]
            
            if fvg.fvg_type == FVGType.BULLISH:
                # Bullish FVG 填充：价格回落到 gap_low 以下
                if bar["low"] <= fvg.gap_low:
                    fvg.status = FVGStatus.FILLED
                    fvg.fill_time = ohlcv.index[j]
                    fvg.fill_bars = j - fvg_idx
                    break
                elif bar["low"] <= fvg.gap_high:
                    fvg.status = FVGStatus.PARTIAL
                # 最大向上延伸
                ext = (bar["high"] - fvg.gap_high) / fvg.mid_price * 100
                max_ext = max(max_ext, ext)
                
            else:  # BEARISH
                # Bearish FVG 填充：价格回升到 gap_high 以上
                if bar["high"] >= fvg.gap_high:
                    fvg.status = FVGStatus.FILLED
                    fvg.fill_time = ohlcv.index[j]
                    fvg.fill_bars = j - fvg_idx
                    break
                elif bar["high"] >= fvg.gap_low:
                    fvg.status = FVGStatus.PARTIAL
                # 最大向下延伸
                ext = (fvg.gap_low - bar["low"]) / fvg.mid_price * 100
                max_ext = max(max_ext, ext)
        
        fvg.max_extension = max_ext
        
        if fvg.status == FVGStatus.OPEN and (end_idx - fvg_idx) >= max_bars:
            fvg.status = FVGStatus.EXPIRED


def detect_fvg_multi_timeframe(
    trades: pd.DataFrame,
    timeframes: List[str] = None,
    min_gap_pct: float = 0.02,
) -> Dict[str, List[FVG]]:
    """
    多时间框架 FVG 检测。
    
    Args:
        trades: 逐笔成交数据
        timeframes: K 线频率列表，如 ["1min", "3min", "5min"]
        min_gap_pct: 最小 gap 百分比
    
    Returns:
        Dict[timeframe -> List[FVG]]
    """
    from utilities.binance_loader import resample_trades_to_ohlcv
    
    timeframes = timeframes or ["1min", "3min", "5min"]
    results = {}
    
    for tf in timeframes:
        ohlcv = resample_trades_to_ohlcv(trades, freq=tf)
        fvgs = detect_fvg(ohlcv, min_gap_pct=min_gap_pct)
        results[tf] = fvgs
    
    return results
