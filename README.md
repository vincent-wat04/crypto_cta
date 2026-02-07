# Mean Reversion CTA + Polymarket

高频微观结构反转策略 + FVG (Fair Value Gap) 策略，目标接入 Polymarket 交易。

## 项目结构

```
mean_reversion_cta_pm/
├── indicators/                   # 高频指标库（核心）
│   ├── base/                     # 基础高频指标
│   │   ├── price_impact.py       # 价格冲击 (tick/VW/Kyle's Lambda)
│   │   ├── spread.py             # Spread 估计 (trade_diff/Roll/effective)
│   │   ├── order_flow.py         # 订单流 (imbalance/VPIN/toxicity)
│   │   ├── volume_profile.py     # 成交量结构 (taker corr/autocorr/skewness)
│   │   └── orderbook_pressure.py # Orderbook 压力 (VWAP/depth imbalance/slope)
│   ├── composite/                # 复合指标 (FVG/microstructure reversal)
│   ├── regime/                   # 市场状态识别
│   │   ├── volatility_regime.py  # 波动率 (realized/Parkinson/GARCH/regime)
│   │   ├── trend_strength.py     # 趋势 (ADX/efficiency ratio/Hurst)
│   │   └── liquidity_regime.py   # 流动性 (Amihud/turnover/regime)
│   └── registry.py               # 指标注册表 (统一发现与调用)
│
├── utilities/                    # 数据加载工具
│   ├── binance_loader.py         # Binance aggTrades（含 taker 聚合）/ klines
│   ├── lakeapi_loader.py         # Lake-API orderbook / trades
│   └── paths.py                  # 统一路径管理（data/ 分区）
│
├── backtest/                     # 通用回测框架
│   ├── labeling.py               # Triple Barrier Method 标签
│   ├── engine.py                 # 单指标回测 + 复合策略回测
│   └── metrics.py                # 统计指标 (Sharpe/Sortino/PF/MDD)
│
├── cta/                          # 策略模块
│   ├── fvg/                      # FVG 策略
│   │   ├── detector.py           # FVG 检测
│   │   ├── features.py           # 逐笔成交特征
│   │   ├── orderbook_features.py # Orderbook 特征
│   │   └── model.py              # 聚类 + GBM 分类
│   ├── microstructure/           # 微观结构反转
│   │   ├── price_impact.py       # Price Impact
│   │   ├── spread_estimator.py   # Spread 估计
│   │   ├── order_flow.py         # 订单流
│   │   ├── liquidity_detector.py # 流动性冲击 + MM 补单
│   │   └── data_collector.py     # WebSocket 实时采集
│   ├── backtest.py               # FVG 专用回测
│   └── metrics/accuracy.py       # 信号准确性评估
│
├── notebooks/                    # 研究 Notebook
│   ├── 01_data_loading.ipynb     # 数据加载演示
│   └── 02_indicator_research.ipynb  # 单指标研究 + Triple Barrier
│
├── data/                         # 数据目录（自动分区）
│   ├── cache/{symbol}/           # 原始数据缓存
│   │   ├── trades/{date}.parquet
│   │   ├── klines/{tf}_{date}.parquet
│   │   └── orderbook/{date}.parquet
│   ├── features/{symbol}/{type}/ # 计算好的特征
│   ├── models/{strategy}/        # 训练好的模型
│   └── backtest_results/{strat}/ # 回测结果
│
├── core/                         # 基础设施
│   ├── config.py                 # 配置管理 (YAML)
│   └── types.py                  # 数据类型
│
├── engine/                       # 实时交易引擎
│   ├── realtime.py               # WebSocket + 增量计算
│   └── executor.py               # Polymarket CLOB 执行
│
├── src/pm/                       # Polymarket API (legacy)
│
├── scripts/                      # 入口脚本
│   ├── run_fvg_strategy.py       # FVG 基础版
│   ├── run_fvg_with_orderbook.py # FVG V2 + Orderbook
│   ├── run_microstructure_backtest.py  # 微观结构反转
│   └── run_bot.py                # 实时 Bot
│
├── docs/                         # 文档
│   ├── indicators.md             # 指标库文档
│   ├── backtest_engine.md        # 回测引擎文档
│   └── quickstart.md             # 快速开始
│
├── requirements.txt
└── .gitignore
```

## 快速开始

```bash
# 安装
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Jupyter 研究
cd notebooks && jupyter notebook

# FVG 回测（Binance aggTrades）
.venv/bin/python scripts/run_fvg_strategy.py --days 3 --symbol SOL/USDT

# FVG V2 + Orderbook（lake-api BTC 数据）
.venv/bin/python scripts/run_fvg_with_orderbook.py

# FVG V2（Binance aggTrades，无 orderbook）
.venv/bin/python scripts/run_fvg_with_orderbook.py --use-binance --symbol SOL/USDT --days 3

# 微观结构反转
.venv/bin/python scripts/run_microstructure_backtest.py --days 5 --optimize

# 实时 Bot (dry-run)
.venv/bin/python scripts/run_bot.py --dry-run
```

## 数据源

| 来源 | 内容 | Taker 聚合 | 限制 |
|---|---|---|---|
| Binance aggTrades | 聚合成交（含 first/last trade ID） | **有** | API 限速 |
| Binance klines | OHLCV | N/A | API 限速 |
| Lake-API orderbook | 历史深度 (10Hz) | N/A | 免费仅 BTC-USDT 3 天 |
| Lake-API trades | 逐笔成交 | **无** | 同上 |

## 核心概念

### 指标层级
- **基础指标** (`indicators/base/`): 从原始 trades/orderbook 计算的单一指标
- **复合指标** (`cta/fvg/`, `cta/microstructure/`): 由多个基础指标组合，通过模型或规则生成
- **Regime 指标** (`indicators/regime/`): 识别市场状态，用于策略参数动态调整

### Triple Barrier Method
趋势预测标签系统 — 不看终点涨跌，而是看价格路径：
- 触及上轨 → 强劲上涨 (label=1)
- 触及下轨 → 强劲下跌 (label=-1)
- 时间屏障到期 → 震荡 (label=0)

### aggTrades vs Trades
Binance aggTrades 保留 `first_trade_id` / `last_trade_id`，允许还原同一 taker order 吃了多少档流动性。这对 volume_profile 类指标至关重要。
