"""
实时数据采集器：通过 WebSocket 采集订单簿和逐笔成交。

功能：
1. 实时采集 orderbook 快照
2. 实时采集逐笔成交
3. 保存到本地文件供后续回测
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Any

import pandas as pd

try:
    import websockets
except ImportError:
    websockets = None


class BinanceDataCollector:
    """Binance WebSocket 数据采集器。"""
    
    WS_BASE = "wss://stream.binance.com:9443/ws"
    
    def __init__(
        self,
        symbol: str = "solusdt",
        save_dir: Optional[Path] = None,
    ):
        self.symbol = symbol.lower()
        self.save_dir = save_dir or Path("data/realtime")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.trades = []
        self.orderbook_snapshots = []
        self.running = False
        
    async def collect_trades(
        self,
        duration_seconds: int = 3600,
        callback: Optional[Callable[[dict], Any]] = None,
    ):
        """
        采集逐笔成交。
        
        Args:
            duration_seconds: 采集时长（秒）
            callback: 每笔成交的回调函数
        """
        if websockets is None:
            raise ImportError("Install websockets: pip install websockets")
        
        stream = f"{self.symbol}@aggTrade"
        uri = f"{self.WS_BASE}/{stream}"
        
        start_time = time.time()
        self.running = True
        
        print(f"[DataCollector] Starting trade collection for {self.symbol}")
        print(f"[DataCollector] Duration: {duration_seconds}s")
        
        try:
            async with websockets.connect(uri) as ws:
                while self.running and (time.time() - start_time) < duration_seconds:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=30)
                        data = json.loads(msg)
                        
                        trade = {
                            "timestamp": pd.Timestamp.utcfromtimestamp(data["T"] / 1000),
                            "price": float(data["p"]),
                            "amount": float(data["q"]),
                            "side": "sell" if data["m"] else "buy",  # m=True: maker is buyer => taker is seller
                            "trade_id": data["a"],
                        }
                        self.trades.append(trade)
                        
                        if callback:
                            callback(trade)
                            
                    except asyncio.TimeoutError:
                        continue
                        
        except Exception as e:
            print(f"[DataCollector] Error: {e}")
        finally:
            self._save_trades()
            print(f"[DataCollector] Collected {len(self.trades)} trades")
    
    async def collect_orderbook(
        self,
        interval_ms: int = 1000,
        duration_seconds: int = 3600,
        depth: int = 20,
        callback: Optional[Callable[[dict], Any]] = None,
    ):
        """
        采集订单簿快照。
        
        Args:
            interval_ms: 采集间隔（毫秒）
            duration_seconds: 采集时长（秒）
            depth: 订单簿深度
            callback: 每次快照的回调函数
        """
        if websockets is None:
            raise ImportError("Install websockets: pip install websockets")
        
        # 使用 depth stream
        stream = f"{self.symbol}@depth{depth}@{interval_ms}ms"
        uri = f"{self.WS_BASE}/{stream}"
        
        start_time = time.time()
        self.running = True
        
        print(f"[DataCollector] Starting orderbook collection for {self.symbol}")
        
        try:
            async with websockets.connect(uri) as ws:
                while self.running and (time.time() - start_time) < duration_seconds:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=30)
                        data = json.loads(msg)
                        
                        snapshot = {
                            "timestamp": datetime.utcnow(),
                            "bids": [(float(p), float(q)) for p, q in data.get("bids", [])],
                            "asks": [(float(p), float(q)) for p, q in data.get("asks", [])],
                        }
                        self.orderbook_snapshots.append(snapshot)
                        
                        if callback:
                            callback(snapshot)
                            
                    except asyncio.TimeoutError:
                        continue
                        
        except Exception as e:
            print(f"[DataCollector] Error: {e}")
        finally:
            self._save_orderbook()
            print(f"[DataCollector] Collected {len(self.orderbook_snapshots)} snapshots")
    
    async def collect_all(
        self,
        duration_seconds: int = 3600,
        orderbook_interval_ms: int = 1000,
    ):
        """同时采集 trades 和 orderbook。"""
        await asyncio.gather(
            self.collect_trades(duration_seconds),
            self.collect_orderbook(
                interval_ms=orderbook_interval_ms,
                duration_seconds=duration_seconds,
            ),
        )
    
    def stop(self):
        """停止采集。"""
        self.running = False
    
    def _save_trades(self):
        """保存逐笔成交到文件。"""
        if not self.trades:
            return
        
        df = pd.DataFrame(self.trades)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = self.save_dir / f"trades_{self.symbol}_{timestamp}.parquet"
        df.to_parquet(filename, index=False)
        print(f"[DataCollector] Saved trades to {filename}")
    
    def _save_orderbook(self):
        """保存订单簿快照到文件。"""
        if not self.orderbook_snapshots:
            return
        
        # 转换为 DataFrame 友好格式
        records = []
        for snap in self.orderbook_snapshots:
            record = {
                "timestamp": snap["timestamp"],
                "best_bid": snap["bids"][0][0] if snap["bids"] else None,
                "best_ask": snap["asks"][0][0] if snap["asks"] else None,
                "bid_depth_5": sum(q for _, q in snap["bids"][:5]),
                "ask_depth_5": sum(q for _, q in snap["asks"][:5]),
                "bid_depth_10": sum(q for _, q in snap["bids"][:10]),
                "ask_depth_10": sum(q for _, q in snap["asks"][:10]),
                "spread": (snap["asks"][0][0] - snap["bids"][0][0]) if snap["bids"] and snap["asks"] else None,
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = self.save_dir / f"orderbook_{self.symbol}_{timestamp}.parquet"
        df.to_parquet(filename, index=False)
        print(f"[DataCollector] Saved orderbook to {filename}")


def load_collected_trades(
    data_dir: Path,
    symbol: str = "solusdt",
    days: int = 5,
) -> pd.DataFrame:
    """加载已采集的逐笔成交数据。"""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return pd.DataFrame()
    
    pattern = f"trades_{symbol.lower()}_*.parquet"
    files = sorted(data_dir.glob(pattern), reverse=True)
    
    if not files:
        return pd.DataFrame()
    
    # 合并文件
    dfs = []
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=days)
    
    for f in files:
        try:
            df = pd.read_parquet(f)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df[df["timestamp"] >= cutoff]
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True).sort_values("timestamp")


def load_collected_orderbook(
    data_dir: Path,
    symbol: str = "solusdt",
    days: int = 5,
) -> pd.DataFrame:
    """加载已采集的订单簿数据。"""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return pd.DataFrame()
    
    pattern = f"orderbook_{symbol.lower()}_*.parquet"
    files = sorted(data_dir.glob(pattern), reverse=True)
    
    if not files:
        return pd.DataFrame()
    
    dfs = []
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=days)
    
    for f in files:
        try:
            df = pd.read_parquet(f)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df[df["timestamp"] >= cutoff]
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True).sort_values("timestamp")
