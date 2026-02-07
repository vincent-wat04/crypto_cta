"""
数据加载工具集。

- binance_loader: Binance REST API (aggTrades with taker info, klines, orderbook)
- lakeapi_loader: Lake-API historical orderbook snapshots
- paths: 统一路径管理（cache / features / models / backtest_results）
"""
from .paths import DataPaths
