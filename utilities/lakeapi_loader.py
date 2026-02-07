"""
Lake-API 数据加载器：历史 orderbook snapshots + trades。

用途：补充 Binance API 不提供的历史 orderbook 深度数据。
免费 sample：BTC-USDT, 2022-10-01 ~ 2022-10-03。
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

import pandas as pd


def load_lakeapi_orderbook(
    days: int = 3,
    symbol: str = "BTC-USDT",
    exchange: str = "BINANCE",
    resample_freq: str = "1s",
    depth_levels: int = 10,
) -> pd.DataFrame:
    """
    从 lake-api 加载历史 orderbook（降采样到 resample_freq）。

    Returns:
        DataFrame with origin_time, bid_0_price, bid_0_size, ..., ask_9_size
    """
    import lakeapi
    lakeapi.use_sample_data(anonymous_access=True)

    start = datetime(2022, 10, 1)
    end = datetime(2022, 10, min(1 + days, 4))

    needed_cols = ["origin_time"]
    for side in ["bid", "ask"]:
        for i in range(depth_levels):
            needed_cols.extend([f"{side}_{i}_price", f"{side}_{i}_size"])

    print(f"[LakeAPI] Loading orderbook ({days}d, {symbol})...")
    book_raw = lakeapi.load_data(
        table="book", start=start, end=end,
        symbols=[symbol], exchanges=[exchange],
        drop_partition_cols=True,
    )
    available = [c for c in needed_cols if c in book_raw.columns]
    book_raw = book_raw[available]

    if resample_freq:
        book_raw["_ts"] = pd.to_datetime(book_raw["origin_time"])
        book_raw["_sec"] = book_raw["_ts"].dt.floor(resample_freq)
        book = book_raw.groupby("_sec").last().reset_index(drop=True)
    else:
        book = book_raw

    print(f"  {len(book)} snapshots")
    return book


def load_lakeapi_trades(
    days: int = 3,
    symbol: str = "BTC-USDT",
    exchange: str = "BINANCE",
) -> pd.DataFrame:
    """
    从 lake-api 加载历史 trades（逐笔成交，无 taker 聚合信息）。

    注意：lake-api trades 不含 first/last trade ID，无法还原 taker order。
    如需 taker 聚合信息，请使用 binance_loader.load_agg_trades。
    """
    import lakeapi
    lakeapi.use_sample_data(anonymous_access=True)

    start = datetime(2022, 10, 1)
    end = datetime(2022, 10, min(1 + days, 4))

    print(f"[LakeAPI] Loading trades ({days}d, {symbol})...")
    raw = lakeapi.load_data(
        table="trades", start=start, end=end,
        symbols=[symbol], exchanges=[exchange],
        drop_partition_cols=True,
    )
    print(f"  {len(raw)} trades")

    return pd.DataFrame({
        "timestamp": pd.to_datetime(raw["origin_time"]),
        "price": raw["price"].values.astype(float),
        "amount": raw["quantity"].values.astype(float),
        "side": raw["side"].values,
        "cost": raw["price"].values.astype(float) * raw["quantity"].values.astype(float),
    })
