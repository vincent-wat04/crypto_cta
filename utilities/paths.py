"""
统一路径管理：按交易对 / 数据类型 / 日期分区存储。

目录结构：
  data/
    cache/{symbol}/trades/{date}.parquet
    cache/{symbol}/klines/{tf}_{date}.parquet
    cache/{symbol}/orderbook/{date}.parquet
    features/{symbol}/{indicator_type}/{date}.parquet
    models/{strategy}/{model_name}.pkl
    backtest_results/{strategy}/{run_id}.csv
"""
from __future__ import annotations

import os
from pathlib import Path
from datetime import date

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("MR_DATA_ROOT", str(ROOT / "data")))


class DataPaths:
    """集中管理所有数据路径。"""

    root = ROOT
    data = DATA

    # ── Cache ──
    @staticmethod
    def cache_trades(symbol: str, dt: date, perpetual: bool = False) -> Path:
        subdir = "trades_perp" if perpetual else "trades"
        p = DATA / "cache" / symbol.replace("/", "_") / subdir
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{dt.isoformat()}.parquet"

    @staticmethod
    def cache_klines(symbol: str, timeframe: str, dt: date) -> Path:
        p = DATA / "cache" / symbol.replace("/", "_") / "klines"
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{timeframe}_{dt.isoformat()}.parquet"

    @staticmethod
    def cache_orderbook(symbol: str, dt: date) -> Path:
        p = DATA / "cache" / symbol.replace("/", "_") / "orderbook"
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{dt.isoformat()}.parquet"

    # ── Features ──
    @staticmethod
    def features(symbol: str, indicator_type: str, dt: date) -> Path:
        p = DATA / "features" / symbol.replace("/", "_") / indicator_type
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{dt.isoformat()}.parquet"

    @staticmethod
    def features_dir(symbol: str, indicator_type: str) -> Path:
        p = DATA / "features" / symbol.replace("/", "_") / indicator_type
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ── Models ──
    @staticmethod
    def model(strategy: str, name: str) -> Path:
        p = DATA / "models" / strategy
        p.mkdir(parents=True, exist_ok=True)
        return p / name

    # ── Backtest results ──
    @staticmethod
    def backtest_result(strategy: str, run_id: str) -> Path:
        p = DATA / "backtest_results" / strategy
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{run_id}.csv"

    @staticmethod
    def backtest_dir(strategy: str) -> Path:
        p = DATA / "backtest_results" / strategy
        p.mkdir(parents=True, exist_ok=True)
        return p
