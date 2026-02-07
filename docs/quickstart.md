# 快速开始

## 项目结构

```
mean_reversion_cta_pm/
├── indicators/           # 高频指标库
│   ├── base/             # 基础指标
│   │   ├── price_impact.py       # 价格冲击
│   │   ├── spread.py             # Spread 估计
│   │   ├── order_flow.py         # 订单流
│   │   ├── volume_profile.py     # 成交量结构（需 aggTrades）
│   │   ├── orderbook_pressure.py # Orderbook 压力
│   │   ├── taker_flow.py         # ★ Taker Order 预处理 + 指标
│   │   └── orderbook_lifecycle.py# ★ 挂单生命周期指标
│   ├── composite/        # 复合指标
│   ├── regime/           # 市场状态识别
│   └── registry.py       # 统一注册表
├── utilities/            # 数据加载工具
│   ├── binance_loader.py # Binance aggTrades/klines
│   ├── lakeapi_loader.py # Lake-API orderbook
│   └── paths.py          # 统一路径管理
├── backtest/             # 回测框架
│   ├── labeling.py       # Triple Barrier Method
│   ├── engine.py         # 通用回测引擎
│   └── metrics.py        # 统计指标
├── cta/                  # 策略模块
│   ├── fvg/              # FVG 策略 (V1/V3)
│   │   ├── detector.py         # FVG 三根K线检测
│   │   ├── features.py         # V1 特征工程
│   │   ├── features_v3.py      # ★ V3 特征工程 (indicators/ + 可配时间窗口)
│   │   ├── model.py            # 聚类 + GBM 模型
│   │   └── orderbook_features.py
│   ├── microstructure/   # 微观结构反转 (委托 indicators/)
│   └── backtest.py       # FVG 专用回测
├── notebooks/            # 研究 Notebook
│   ├── 01_data_loading.ipynb
│   └── 02_indicator_research.ipynb
├── data/                 # 数据目录（自动分区）
│   ├── cache/{symbol}/trades|klines|orderbook/{date}.parquet
│   ├── features/{symbol}/{indicator}/{date}.parquet
│   ├── models/{strategy}/{model}.pkl
│   └── backtest_results/{strategy}/
├── scripts/              # 命令行入口
│   ├── run_fvg_v3.py            # ★ FVG V3 完整 Pipeline
│   ├── run_fvg_strategy.py      # FVG V1
│   ├── run_fvg_with_orderbook.py
│   ├── run_microstructure_backtest.py
│   └── run_bot.py
├── docs/
└── src/pm/               # Polymarket API
```

## 环境设置

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## 数据获取

### Binance aggTrades（推荐）

```python
from utilities.binance_loader import load_agg_trades, resample_trades_to_ohlcv

# 加载 3 天 SOL/USDT aggTrades（自动缓存）
trades = load_agg_trades(symbol='SOL/USDT', days=3)

# Resample 为 OHLCV
ohlcv_1m = resample_trades_to_ohlcv(trades, freq='1min')
```

### Taker Order 预处理

```python
from indicators.base.taker_flow import group_taker_orders

# 将 aggTrades 分组为 taker orders
taker_orders = group_taker_orders(trades)
print(f"{len(taker_orders):,} taker orders from {len(trades):,} aggTrades")
print(f"Avg levels swept: {taker_orders['levels_swept'].mean():.2f}")
print(f"Multi-level (>=2): {(taker_orders['levels_swept'] >= 2).mean()*100:.1f}%")
```

### Lake-API Orderbook

```python
from utilities.lakeapi_loader import load_lakeapi_orderbook

book = load_lakeapi_orderbook(days=3, symbol='BTC-USDT')
```

## 指标计算

```python
from indicators.base import (
    tick_price_impact,
    trade_imbalance,
    taker_volume_corr,
)
from indicators.base.taker_flow import (
    group_taker_orders,
    avg_levels_swept,
    impact_efficiency,
    taker_imbalance,
)
from indicators.base.orderbook_lifecycle import (
    depth_change_rate,
    refill_frequency,
    cancel_rate,
)

# 基础指标
impact = tick_price_impact(trades, window_ms=5000)
imbalance = trade_imbalance(trades, window_trades=100)

# Taker order 级别指标
taker_orders = group_taker_orders(trades)
levels = avg_levels_swept(taker_orders, resample_freq='5s', window=60)
eff = impact_efficiency(taker_orders, resample_freq='10s', window=30)

# Orderbook 生命周期指标 (需要 book 数据)
dcr = depth_change_rate(book, levels=5, window=10)
rfr = refill_frequency(book, levels=5, window=20)
```

## 一键计算所有指标

```python
from indicators.registry import compute_all_base_indicators

# 传入所有可用数据源
all_indicators = compute_all_base_indicators(
    trades=trades,
    taker_orders=taker_orders,
    book=book,  # 可选
    resample_freq='1min',
)
print(f"Computed {all_indicators.shape[1]} indicators")
```

## 单指标研究

```python
from backtest import triple_barrier_labels, backtest_single_indicator

# Triple Barrier 标签
labels = triple_barrier_labels(prices, upper_pct=0.5, lower_pct=0.3, max_bars=60)

# 回测
result = backtest_single_indicator(
    indicator=imbalance,
    prices=prices,
    long_threshold=0.3,
    short_threshold=-0.3,
    hold_bars=30,
    stop_loss_pct=0.3,
    take_profit_pct=0.5,
)
print(f"Sharpe: {result['metrics']['sharpe']}")
```

## 策略运行

```bash
# ★ FVG V3 策略（完整 Pipeline：taker 分组 + 全面指标 + TB标签 + GBM）
.venv/bin/python scripts/run_fvg_v3.py --symbol SOL/USDT --days 3

# FVG V3 + Grid Search 参数优化
.venv/bin/python scripts/run_fvg_v3.py --days 3 --grid-search

# FVG V3 + Orderbook 数据
.venv/bin/python scripts/run_fvg_v3.py --days 3 --use-orderbook

# FVG V1
.venv/bin/python scripts/run_fvg_strategy.py --days 3

# 微观结构反转
.venv/bin/python scripts/run_microstructure_backtest.py --days 5
```

## 数据缓存规则

所有数据按以下规则自动缓存：

- `data/cache/{SOL_USDT}/trades/{2026-02-01}.parquet`
- `data/cache/{SOL_USDT}/klines/{1m_2026-02-01}.parquet`
- `data/features/{SOL_USDT}/{base_indicators}/{2026-02-01}.parquet`

重新拉取相同日期不会重复下载。

## Binance aggTrades 数据说明

| 字段 | 含义 |
|------|------|
| `a` (agg_trade_id) | 聚合成交 ID |
| `f` (first_trade_id) | 同一 taker order 在同一价格的第一笔撮合 ID |
| `l` (last_trade_id) | 同一 taker order 在同一价格的最后一笔撮合 ID |
| `m` (is_buyer_maker) | True = taker 是卖方, False = taker 是买方 |

**Taker order 还原**：连续 aggTrades 满足 `f[i] = l[i-1] + 1` 且 `m[i] = m[i-1]` 时，属于同一 taker order（一笔市价单扫过多档）。
