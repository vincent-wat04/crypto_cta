"""
指标注册表：统一发现和调用所有指标。

Usage:
    from indicators.registry import REGISTRY, compute_all_base_indicators

    # 查看所有可用指标
    for name, info in REGISTRY.items():
        print(f"{name}: {info['description']}")

    # 一键计算所有 tick-level 指标
    df = compute_all_base_indicators(trades, resample_freq='1min')
"""
from __future__ import annotations

from typing import Dict, Any

import pandas as pd

from .base import price_impact, spread, order_flow, volume_profile
from .base import taker_flow, orderbook_lifecycle, orderbook_pressure
from .base import returns_momentum, bar_structure, tick_statistics

REGISTRY: Dict[str, Dict[str, Any]] = {
    # ── Price Impact ──
    "price_impact": {
        "fn": price_impact.tick_price_impact,
        "category": "base",
        "subcategory": "price_impact",
        "input": "trades",
        "description": "|ΔPrice| / Volume 滚动窗口",
        "default_params": {"window_ms": 5000},
    },
    "vw_impact": {
        "fn": price_impact.volume_weighted_impact,
        "category": "base",
        "subcategory": "price_impact",
        "input": "trades",
        "description": "成交量加权 price impact",
        "default_params": {"window_ms": 10000},
    },
    "kyles_lambda": {
        "fn": price_impact.kyles_lambda,
        "category": "base",
        "subcategory": "price_impact",
        "input": "trades",
        "description": "Kyle's Lambda",
        "default_params": {"window_trades": 100},
    },
    # ── Spread ──
    "trade_diff_spread": {
        "fn": spread.trade_diff_spread,
        "category": "base",
        "subcategory": "spread",
        "input": "trades",
        "description": "相邻成交价差滚动中位数",
        "default_params": {"window_trades": 100},
    },
    "roll_spread": {
        "fn": spread.roll_spread,
        "category": "base",
        "subcategory": "spread",
        "input": "trades",
        "description": "Roll (1984) spread 估计量",
        "default_params": {"window_trades": 200},
    },
    # ── Order Flow ──
    "trade_imbalance": {
        "fn": order_flow.trade_imbalance,
        "category": "base",
        "subcategory": "order_flow",
        "input": "trades",
        "description": "(BuyVol - SellVol) / TotalVol",
        "default_params": {"window_trades": 100},
    },
    "vpin": {
        "fn": order_flow.vpin,
        "category": "base",
        "subcategory": "order_flow",
        "input": "trades",
        "description": "VPIN 信息交易概率",
        "default_params": {"bucket_volume": 1000, "n_buckets": 50},
    },
    "flow_toxicity": {
        "fn": order_flow.flow_toxicity,
        "category": "base",
        "subcategory": "order_flow",
        "input": "trades",
        "description": "订单流毒性",
        "default_params": {"window_trades": 200},
    },
    # ── Volume Profile (需要 aggTrades) ──
    "taker_vol_corr": {
        "fn": volume_profile.taker_volume_corr,
        "category": "base",
        "subcategory": "volume_profile",
        "input": "trades",
        "description": "Buy/Sell taker volume 相关系数",
        "default_params": {"resample_freq": "5s", "window": 60},
        "requires_agg_trades": True,
    },
    "taker_vol_autocorr_buy": {
        "fn": lambda trades, **kw: volume_profile.taker_volume_autocorr(trades, side="buy", **kw),
        "category": "base",
        "subcategory": "volume_profile",
        "input": "trades",
        "description": "Buy taker volume 自相关",
        "default_params": {"resample_freq": "5s", "window": 60},
        "requires_agg_trades": True,
    },
    "taker_vol_autocorr_sell": {
        "fn": lambda trades, **kw: volume_profile.taker_volume_autocorr(trades, side="sell", **kw),
        "category": "base",
        "subcategory": "volume_profile",
        "input": "trades",
        "description": "Sell taker volume 自相关",
        "default_params": {"resample_freq": "5s", "window": 60},
        "requires_agg_trades": True,
    },
    "taker_vol_skewness": {
        "fn": volume_profile.taker_volume_skewness,
        "category": "base",
        "subcategory": "volume_profile",
        "input": "trades",
        "description": "Taker volume 偏度",
        "default_params": {"resample_freq": "5s", "window": 120},
        "requires_agg_trades": True,
    },
    "buy_sell_vol_ratio": {
        "fn": volume_profile.buy_sell_volume_ratio,
        "category": "base",
        "subcategory": "volume_profile",
        "input": "trades",
        "description": "log(BuyVol/SellVol)",
        "default_params": {"resample_freq": "5s", "window": 60},
    },
    "large_trade_ratio": {
        "fn": volume_profile.large_trade_ratio,
        "category": "base",
        "subcategory": "volume_profile",
        "input": "trades",
        "description": "大单成交占比",
        "default_params": {"resample_freq": "5s", "window": 60},
        "requires_agg_trades": True,
    },
    # ── Taker Flow (需要 taker_orders) ──
    "avg_levels_swept": {
        "fn": taker_flow.avg_levels_swept,
        "category": "base",
        "subcategory": "taker_flow",
        "input": "taker_orders",
        "description": "滚动平均 levels_swept（吃穿档数）",
        "default_params": {"resample_freq": "5s", "window": 60},
    },
    "large_taker_ratio": {
        "fn": taker_flow.large_taker_ratio,
        "category": "base",
        "subcategory": "taker_flow",
        "input": "taker_orders",
        "description": "激进 taker 单占比 (levels>=3)",
        "default_params": {"resample_freq": "5s", "window": 60, "levels_threshold": 3},
    },
    "taker_imbalance": {
        "fn": taker_flow.taker_imbalance,
        "category": "base",
        "subcategory": "taker_flow",
        "input": "taker_orders",
        "description": "Taker order 买卖不平衡",
        "default_params": {"resample_freq": "5s", "window": 60},
    },
    "sweep_autocorr": {
        "fn": taker_flow.sweep_depth_autocorr,
        "category": "base",
        "subcategory": "taker_flow",
        "input": "taker_orders",
        "description": "levels_swept 自相关",
        "default_params": {"resample_freq": "5s", "window": 60},
    },
    "impact_efficiency": {
        "fn": taker_flow.impact_efficiency,
        "category": "base",
        "subcategory": "taker_flow",
        "input": "taker_orders",
        "description": "全局 impact 效率 (Σimpact / Σvolume)",
        "default_params": {"resample_freq": "10s", "window": 30},
    },
    "taker_arrival_rate": {
        "fn": taker_flow.taker_arrival_rate,
        "category": "base",
        "subcategory": "taker_flow",
        "input": "taker_orders",
        "description": "Taker order 到达率",
        "default_params": {"resample_freq": "5s", "window": 60},
    },
    "taker_size_skewness": {
        "fn": taker_flow.taker_size_skewness,
        "category": "base",
        "subcategory": "taker_flow",
        "input": "taker_orders",
        "description": "Taker order size 偏度",
        "default_params": {"resample_freq": "10s", "window": 60},
    },
    "iceberg_score": {
        "fn": taker_flow.iceberg_score,
        "category": "base",
        "subcategory": "taker_flow",
        "input": "taker_orders",
        "description": "冰山单检测分数",
        "default_params": {"resample_freq": "30s", "window": 20, "fills_threshold": 5.0},
    },
    # ── Orderbook Lifecycle (需要 orderbook) ──
    "depth_change_rate": {
        "fn": orderbook_lifecycle.depth_change_rate,
        "category": "base",
        "subcategory": "orderbook_lifecycle",
        "input": "book",
        "description": "Bid/Ask 深度变化率",
        "default_params": {"levels": 5, "window": 10},
    },
    # refill_frequency: REMOVED - 虚假指标（top-N 聚合深度在 levels 被吃后自动恢复）
    # cancel_rate: REMOVED - 无法区分撤单与被成交
    "depth_resilience": {
        "fn": orderbook_lifecycle.depth_resilience,
        "category": "base",
        "subcategory": "orderbook_lifecycle",
        "input": "book",
        "description": "深度冲击恢复速度",
        "default_params": {"levels": 5, "shock_window": 3, "recovery_window": 10},
    },
    "level_thickness_profile": {
        "fn": orderbook_lifecycle.level_thickness_profile,
        "category": "base",
        "subcategory": "orderbook_lifecycle",
        "input": "book",
        "description": "各档位厚度占比（集中度）",
        "default_params": {"levels": 10},
    },
    # ── Orderbook Pressure ──
    "vwap_pressure": {
        "fn": orderbook_pressure.vwap_pressure,
        "category": "base",
        "subcategory": "orderbook_pressure",
        "input": "book",
        "description": "Orderbook VWAP 偏离 mid",
        "default_params": {"levels": 10},
    },
    "depth_imbalance": {
        "fn": orderbook_pressure.depth_imbalance,
        "category": "base",
        "subcategory": "orderbook_pressure",
        "input": "book",
        "description": "Bid/Ask 深度不平衡",
        "default_params": {"levels": 10},
    },
    "weighted_depth_slope": {
        "fn": orderbook_pressure.weighted_depth_slope,
        "category": "base",
        "subcategory": "orderbook_pressure",
        "input": "book",
        "description": "深度分布斜率",
        "default_params": {"levels": 10},
    },
    # ── Returns & Momentum (需要 OHLCV) ──
    "returns_momentum": {
        "fn": returns_momentum.compute_returns,
        "category": "base",
        "subcategory": "returns_momentum",
        "input": "ohlcv",
        "description": "收益率、动量、波动率、偏度、峰度、z-score",
        "default_params": {"windows": [3, 5, 10, 20]},
    },
    # ── Bar Structure (需要 OHLCV) ──
    "bar_structure": {
        "fn": bar_structure.compute_bar_structure,
        "category": "base",
        "subcategory": "bar_structure",
        "input": "ohlcv",
        "description": "Bar 形态：振幅、实体比、影线、效率、成交密度",
        "default_params": {"windows": [3, 5, 10, 20]},
    },
    # ── Tick Statistics (需要 trades) ──
    "tick_statistics": {
        "fn": tick_statistics.compute_tick_stats,
        "category": "base",
        "subcategory": "tick_statistics",
        "input": "trades",
        "description": "Tick-level 秒内统计：价位数、大单、VWAP偏差、size不对称",
        "default_params": {"freq": "1s", "windows": [3, 5, 10, 20]},
    },
}


def compute_all_base_indicators(
    trades: pd.DataFrame,
    taker_orders: pd.DataFrame = None,
    book: pd.DataFrame = None,
    resample_freq: str = "1min",
    include_agg_only: bool = True,
) -> pd.DataFrame:
    """
    一键计算所有基础指标并 resample 到统一频率。

    Args:
        trades: aggTrades DataFrame
        taker_orders: taker order DataFrame (group_taker_orders output)
        book: orderbook DataFrame
        resample_freq: 输出频率（如 '1min', '5min'）
        include_agg_only: 是否包含需要 aggTrades 的指标
    """
    has_agg = "agg_trade_id" in trades.columns
    result = {}

    for name, info in REGISTRY.items():
        if info.get("requires_agg_trades") and not has_agg and not include_agg_only:
            continue

        input_type = info["input"]

        # Select the right data source
        if input_type == "trades":
            data = trades
        elif input_type == "taker_orders":
            if taker_orders is None or taker_orders.empty:
                continue
            data = taker_orders
        elif input_type == "book":
            if book is None or book.empty:
                continue
            data = book
        else:
            continue

        try:
            output = info["fn"](data, **info["default_params"])
            if isinstance(output, pd.Series):
                s = output[~output.index.duplicated(keep="last")]
                result[name] = s.resample(resample_freq).last()
            elif isinstance(output, pd.DataFrame):
                for col in output.columns:
                    s = output[col][~output[col].index.duplicated(keep="last")]
                    result[f"{name}_{col}"] = s.resample(resample_freq).last()
        except Exception as e:
            print(f"  WARN: {name} failed: {e}")

    df = pd.DataFrame(result)
    return df.dropna(how="all")
