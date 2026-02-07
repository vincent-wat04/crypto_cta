"""
FVG Orderbook 特征工程。

从 orderbook 快照中提取 FVG 事件前后的深度、spread、不平衡等特征。
需要 lake-api 提供的历史 orderbook 数据。
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────

def preindex_book(book: pd.DataFrame) -> pd.DataFrame:
    """
    为 orderbook 建立时间索引（一次性操作，后续 .loc 切片 O(1)）。
    """
    b = book.copy()
    b["_ts"] = pd.to_datetime(b["origin_time"])
    b = b.set_index("_ts").sort_index()
    b = b[~b.index.duplicated(keep="last")]
    return b


# ─────────────────────────────────────────────────────────
# 单个时间点的 orderbook 特征
# ─────────────────────────────────────────────────────────

def compute_orderbook_features_at_time(
    book_indexed: pd.DataFrame,
    target_time: pd.Timestamp,
    window_before_ms: int = 5000,
    window_after_ms: int = 5000,
    depth_levels: int = 10,
) -> Dict[str, float]:
    """
    为指定时间点计算 orderbook 特征。

    Args:
        book_indexed: 预索引的 orderbook（由 preindex_book 生成）
        target_time: FVG 事件时间
        window_before_ms: 事件前的观察窗口（毫秒）
        window_after_ms: 事件后的观察窗口（毫秒）
        depth_levels: 使用的 orderbook 档位数
    """
    pre_start = target_time - pd.Timedelta(milliseconds=window_before_ms)
    post_end = target_time + pd.Timedelta(milliseconds=window_after_ms)

    try:
        pre_book = book_indexed.loc[pre_start:target_time]
        post_book = book_indexed.loc[target_time:post_end]
    except Exception:
        pre_book = pd.DataFrame()
        post_book = pd.DataFrame()

    features: Dict[str, float] = {}

    # ===== Pre-FVG =====
    if len(pre_book) > 0:
        last_pre = pre_book.iloc[-1]

        # Spread
        spread = float(last_pre["ask_0_price"] - last_pre["bid_0_price"])
        mid = (float(last_pre["ask_0_price"]) + float(last_pre["bid_0_price"])) / 2
        features["ob_pre_spread"] = spread
        features["ob_pre_spread_bps"] = spread / mid * 10000

        # Depth at 5 / 10 levels
        for n_lv, label in [(5, "5"), (min(depth_levels, 10), "10")]:
            bid_d = sum(float(last_pre.get(f"bid_{i}_size", 0)) for i in range(n_lv))
            ask_d = sum(float(last_pre.get(f"ask_{i}_size", 0)) for i in range(n_lv))
            features[f"ob_pre_bid_depth_{label}"] = bid_d
            features[f"ob_pre_ask_depth_{label}"] = ask_d
            features[f"ob_pre_depth_imbalance_{label}"] = (
                (bid_d - ask_d) / (bid_d + ask_d + 1e-10)
            )

        # Weighted depth (closer levels = higher weight)
        bid_w = sum(float(last_pre.get(f"bid_{i}_size", 0)) / (i + 1) for i in range(depth_levels))
        ask_w = sum(float(last_pre.get(f"ask_{i}_size", 0)) / (i + 1) for i in range(depth_levels))
        features["ob_pre_bid_weighted_depth"] = bid_w
        features["ob_pre_ask_weighted_depth"] = ask_w
        features["ob_pre_weighted_imbalance"] = (bid_w - ask_w) / (bid_w + ask_w + 1e-10)

        # Large orders (single level > 3x average)
        bid_sizes = [float(last_pre.get(f"bid_{i}_size", 0)) for i in range(depth_levels)]
        ask_sizes = [float(last_pre.get(f"ask_{i}_size", 0)) for i in range(depth_levels)]
        avg_size = (np.mean(bid_sizes) + np.mean(ask_sizes)) / 2
        features["ob_pre_large_bid_levels"] = sum(1 for s in bid_sizes if s > 3 * avg_size)
        features["ob_pre_large_ask_levels"] = sum(1 for s in ask_sizes if s > 3 * avg_size)

        # Spread dynamics in window
        if len(pre_book) > 5:
            spreads = pre_book["ask_0_price"].astype(float) - pre_book["bid_0_price"].astype(float)
            features["ob_pre_spread_std"] = spreads.std()
            features["ob_pre_spread_max"] = spreads.max()
            features["ob_pre_spread_trend"] = spreads.iloc[-1] - spreads.iloc[0]
        else:
            features["ob_pre_spread_std"] = 0
            features["ob_pre_spread_max"] = spread
            features["ob_pre_spread_trend"] = 0

        # Depth change rate
        if len(pre_book) > 5:
            d_start = sum(float(pre_book.iloc[0].get(f"bid_{i}_size", 0)) for i in range(5))
            d_end = sum(float(pre_book.iloc[-1].get(f"bid_{i}_size", 0)) for i in range(5))
            features["ob_pre_depth_change_rate"] = (d_end - d_start) / (d_start + 1e-10)
        else:
            features["ob_pre_depth_change_rate"] = 0
    else:
        for key in [
            "ob_pre_spread", "ob_pre_spread_bps",
            "ob_pre_bid_depth_5", "ob_pre_ask_depth_5", "ob_pre_depth_imbalance_5",
            "ob_pre_bid_depth_10", "ob_pre_ask_depth_10", "ob_pre_depth_imbalance_10",
            "ob_pre_bid_weighted_depth", "ob_pre_ask_weighted_depth",
            "ob_pre_weighted_imbalance",
            "ob_pre_large_bid_levels", "ob_pre_large_ask_levels",
            "ob_pre_spread_std", "ob_pre_spread_max", "ob_pre_spread_trend",
            "ob_pre_depth_change_rate",
        ]:
            features[key] = 0

    # ===== Post-FVG =====
    if len(post_book) > 0:
        first_post = post_book.iloc[0]

        post_spread = float(first_post["ask_0_price"] - first_post["bid_0_price"])
        features["ob_post_spread"] = post_spread

        post_bid_5 = sum(float(first_post.get(f"bid_{i}_size", 0)) for i in range(5))
        post_ask_5 = sum(float(first_post.get(f"ask_{i}_size", 0)) for i in range(5))
        features["ob_post_bid_depth_5"] = post_bid_5
        features["ob_post_ask_depth_5"] = post_ask_5
        features["ob_post_depth_imbalance_5"] = (
            (post_bid_5 - post_ask_5) / (post_bid_5 + post_ask_5 + 1e-10)
        )

        features["ob_spread_change"] = post_spread - features.get("ob_pre_spread", 0)
        features["ob_bid_depth_change"] = post_bid_5 - features.get("ob_pre_bid_depth_5", 0)
        features["ob_ask_depth_change"] = post_ask_5 - features.get("ob_pre_ask_depth_5", 0)

        if len(post_book) > 3:
            depths = [
                sum(float(post_book.iloc[j].get(f"bid_{i}_size", 0)) for i in range(5))
                for j in range(len(post_book))
            ]
            depth_diffs = np.diff(depths)
            features["ob_post_cancel_rate"] = np.mean(depth_diffs < 0) if len(depth_diffs) > 0 else 0
            features["ob_post_depth_volatility"] = np.std(depths) if len(depths) > 1 else 0
        else:
            features["ob_post_cancel_rate"] = 0
            features["ob_post_depth_volatility"] = 0
    else:
        for key in [
            "ob_post_spread", "ob_post_bid_depth_5", "ob_post_ask_depth_5",
            "ob_post_depth_imbalance_5",
            "ob_spread_change", "ob_bid_depth_change", "ob_ask_depth_change",
            "ob_post_cancel_rate", "ob_post_depth_volatility",
        ]:
            features[key] = 0

    return features


# ─────────────────────────────────────────────────────────
# 批量添加 orderbook 特征
# ─────────────────────────────────────────────────────────

def add_orderbook_features(
    feature_df: pd.DataFrame,
    book: pd.DataFrame,
    window_before_ms: int = 5000,
    window_after_ms: int = 5000,
) -> pd.DataFrame:
    """
    为 FVG 特征矩阵批量添加 orderbook 特征。

    Args:
        feature_df: 由 build_fvg_feature_matrix 生成的特征矩阵
        book: 原始 orderbook 数据（含 origin_time 列）
        window_before_ms / window_after_ms: 观察窗口

    Returns:
        增强后的 DataFrame（原始列 + ob_* 列）
    """
    print("[OB Features] Pre-indexing orderbook...")
    book_indexed = preindex_book(book)
    print(f"  Indexed {len(book_indexed)} snapshots")

    print(f"[OB Features] Computing for {len(feature_df)} FVGs...")
    ob_features_list = []

    for i, row in feature_df.iterrows():
        ts = pd.to_datetime(row["timestamp"])
        ob_feat = compute_orderbook_features_at_time(
            book_indexed, ts,
            window_before_ms=window_before_ms,
            window_after_ms=window_after_ms,
        )
        ob_features_list.append(ob_feat)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(feature_df)}...")

    ob_df = pd.DataFrame(ob_features_list)
    result = pd.concat([feature_df.reset_index(drop=True), ob_df], axis=1)
    print(f"  Added {len(ob_df.columns)} orderbook features")
    return result


def get_ob_feature_columns() -> List[str]:
    """返回所有 orderbook 特征列名。"""
    return [
        "ob_pre_spread", "ob_pre_spread_bps",
        "ob_pre_bid_depth_5", "ob_pre_ask_depth_5", "ob_pre_depth_imbalance_5",
        "ob_pre_bid_depth_10", "ob_pre_ask_depth_10", "ob_pre_depth_imbalance_10",
        "ob_pre_bid_weighted_depth", "ob_pre_ask_weighted_depth", "ob_pre_weighted_imbalance",
        "ob_pre_large_bid_levels", "ob_pre_large_ask_levels",
        "ob_pre_spread_std", "ob_pre_spread_max", "ob_pre_spread_trend",
        "ob_pre_depth_change_rate",
        "ob_post_spread", "ob_post_bid_depth_5", "ob_post_ask_depth_5",
        "ob_post_depth_imbalance_5",
        "ob_spread_change", "ob_bid_depth_change", "ob_ask_depth_change",
        "ob_post_cancel_rate", "ob_post_depth_volatility",
    ]
