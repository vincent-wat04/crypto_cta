"""
Binance 数据加载器（aggTrades with taker aggregation info）。

aggTrades 端点返回：
  - a: Aggregate tradeId
  - p: Price
  - q: Quantity
  - f: First tradeId（同一 taker order 的第一笔撮合）
  - l: Last tradeId（同一 taker order 的最后一笔撮合）
  - T: Timestamp
  - m: Is buyer maker? (True = seller taker, False = buyer taker)

相比 lake-api trades，aggTrades 保留了 taker 聚合信息 (f, l)，
允许我们还原同一主动单吃了多少档流动性。
"""
from __future__ import annotations

import time
import logging
from datetime import datetime, date, timedelta
from typing import Optional

import pandas as pd

from .paths import DataPaths

logger = logging.getLogger(__name__)

try:
    import ccxt
except ImportError:
    ccxt = None


_exchange_cache = {}


def _get_exchange(api_key: str = "", api_secret: str = ""):
    if ccxt is None:
        raise ImportError("pip install ccxt")
    key = (api_key, api_secret)
    if key not in _exchange_cache:
        ex = ccxt.binance({
            "apiKey": api_key or None,
            "secret": api_secret or None,
            "enableRateLimit": True,
        })
        ex.load_markets()
        _exchange_cache[key] = ex
    return _exchange_cache[key]


# ─────────────────────────────────────────────────────────
# aggTrades（核心：保留 taker 聚合信息）
# ─────────────────────────────────────────────────────────

def load_agg_trades(
    symbol: str = "SOL/USDT",
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    days: int = 3,
    api_key: str = "",
    api_secret: str = "",
) -> pd.DataFrame:
    """
    加载 Binance aggTrades，按天缓存。

    返回 DataFrame 列:
        timestamp, price, amount, side, cost,
        agg_trade_id, first_trade_id, last_trade_id, n_trades_in_agg

    n_trades_in_agg = last_trade_id - first_trade_id + 1
    表示这条 aggTrade 聚合了多少笔原始撮合。
    """
    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=days)

    all_dfs = []
    current = start_date

    while current < end_date:
        cache_path = DataPaths.cache_trades(symbol, current)

        if cache_path.exists():
            logger.info(f"[Cache hit] {symbol} trades {current}")
            df = pd.read_parquet(cache_path)
            all_dfs.append(df)
        else:
            logger.info(f"[Fetching] {symbol} aggTrades {current}...")
            df = _fetch_agg_trades_day(symbol, current, api_key, api_secret)
            if not df.empty:
                df.to_parquet(cache_path, index=False)
                all_dfs.append(df)

        current += timedelta(days=1)

    if not all_dfs:
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    return result.sort_values("timestamp").reset_index(drop=True)


def _fetch_agg_trades_day(
    symbol: str, dt: date, api_key: str = "", api_secret: str = "",
) -> pd.DataFrame:
    """
    获取单天 aggTrades，直接调 Binance REST API 保留 f/l 字段。

    策略：先用 startTime 拿第一批（获得起始 fromId），
    然后用 fromId 分页（比 startTime 快 10x+）。
    """
    exchange = _get_exchange(api_key, api_secret)
    market = exchange.market(symbol)
    binance_symbol = market["id"]  # e.g. "BTCUSDT"

    start_ms = int(datetime.combine(dt, datetime.min.time()).timestamp() * 1000)
    end_ms = start_ms + 86400_000

    all_records = []
    n_requests = 0

    # Step 1: 用 startTime 获取当天第一批，拿到起始 aggTradeId
    try:
        first_batch = exchange.publicGetAggTrades({
            "symbol": binance_symbol,
            "startTime": start_ms,
            "limit": 1000,
        })
        n_requests += 1
    except Exception as e:
        logger.warning("Error fetching first batch: %s", e)
        return pd.DataFrame()

    if not first_batch:
        return pd.DataFrame()

    # 解析第一批
    for r in first_batch:
        ts = int(r["T"])
        if ts >= end_ms:
            break
        all_records.append(_parse_agg_trade(r))

    last_id = int(first_batch[-1]["a"])

    # Step 2: 用 fromId 分页（Binance 对 fromId 查询效率远高于 startTime）
    while True:
        try:
            batch = exchange.publicGetAggTrades({
                "symbol": binance_symbol,
                "fromId": last_id + 1,
                "limit": 1000,
            })
            n_requests += 1
        except Exception as e:
            logger.warning("Error fetching aggTrades fromId=%s: %s", last_id + 1, e)
            time.sleep(1)
            continue

        if not batch:
            break

        hit_end = False
        for r in batch:
            ts = int(r["T"])
            if ts >= end_ms:
                hit_end = True
                break
            all_records.append(_parse_agg_trade(r))

        last_id = int(batch[-1]["a"])

        if hit_end or len(batch) < 1000:
            break

        # Rate limiting: ~10 req/s is safe for public API
        if n_requests % 10 == 0:
            time.sleep(0.5)
            if n_requests % 100 == 0:
                logger.info(
                    "  ... %d requests, %d records so far",
                    n_requests, len(all_records),
                )

    logger.info(
        "  Fetched %d aggTrades in %d requests for %s",
        len(all_records), n_requests, dt.isoformat(),
    )

    if not all_records:
        return pd.DataFrame()

    return pd.DataFrame(all_records)


def _parse_agg_trade(r: dict) -> dict:
    """Parse a single aggTrade response dict."""
    return {
        "timestamp": pd.Timestamp(int(r["T"]), unit="ms", tz="UTC"),
        "price": float(r["p"]),
        "amount": float(r["q"]),
        "side": "sell" if r["m"] else "buy",
        "cost": float(r["p"]) * float(r["q"]),
        "agg_trade_id": int(r["a"]),
        "first_trade_id": int(r["f"]),
        "last_trade_id": int(r["l"]),
        "n_trades_in_agg": int(r["l"]) - int(r["f"]) + 1,
    }


# ─────────────────────────────────────────────────────────
# Klines
# ─────────────────────────────────────────────────────────

def load_klines(
    symbol: str = "SOL/USDT",
    timeframe: str = "1m",
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    days: int = 7,
    api_key: str = "",
    api_secret: str = "",
) -> pd.DataFrame:
    """加载 Binance OHLCV klines，按天缓存。"""
    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=days)

    all_dfs = []
    current = start_date

    while current < end_date:
        cache_path = DataPaths.cache_klines(symbol, timeframe, current)

        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            all_dfs.append(df)
        else:
            logger.info(f"[Fetching] {symbol} klines {timeframe} {current}...")
            df = _fetch_klines_day(symbol, timeframe, current, api_key, api_secret)
            if not df.empty:
                df.to_parquet(cache_path, index=False)
                all_dfs.append(df)
        current += timedelta(days=1)

    if not all_dfs:
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    result = result.drop_duplicates(subset=["timestamp"]).set_index("timestamp").sort_index()
    return result


def _fetch_klines_day(
    symbol: str, timeframe: str, dt: date,
    api_key: str = "", api_secret: str = "",
) -> pd.DataFrame:
    exchange = _get_exchange(api_key, api_secret)
    start_ms = int(datetime.combine(dt, datetime.min.time()).timestamp() * 1000)
    end_ms = start_ms + 86400_000

    all_ohlcv = []
    since = start_ms
    while since < end_ms:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        if ohlcv[-1][0] >= end_ms:
            break
        time.sleep(0.1)

    if not all_ohlcv:
        return pd.DataFrame()

    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", tz="UTC")
    df = df[(df["timestamp"] >= pd.Timestamp(dt, tz="UTC"))
            & (df["timestamp"] < pd.Timestamp(dt + timedelta(days=1), tz="UTC"))]
    return df


# ─────────────────────────────────────────────────────────
# Resample
# ─────────────────────────────────────────────────────────

def resample_trades_to_ohlcv(trades: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """将 aggTrades resample 为 OHLCV（附带 buy/sell volume）。"""
    ts = trades.set_index("timestamp")
    ts = ts[~ts.index.duplicated(keep="last")]

    ohlcv = ts["price"].resample(freq).ohlc()
    ohlcv["volume"] = ts["amount"].resample(freq).sum()
    ohlcv["buy_volume"] = (
        ts.loc[ts["side"] == "buy", "amount"].resample(freq).sum()
        .reindex(ohlcv.index).fillna(0)
    )
    ohlcv["sell_volume"] = (
        ts.loc[ts["side"] == "sell", "amount"].resample(freq).sum()
        .reindex(ohlcv.index).fillna(0)
    )
    ohlcv["n_trades"] = ts["agg_trade_id"].resample(freq).count() if "agg_trade_id" in ts.columns else 0
    ohlcv["volume"] = ohlcv["volume"].fillna(0)
    return ohlcv.dropna(subset=["open"]).ffill().fillna(0)
