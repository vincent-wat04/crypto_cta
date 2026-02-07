# 高频指标库文档

## 架构

```
indicators/
├── base/                       # 基础高频指标
│   ├── price_impact.py         # 价格冲击
│   ├── spread.py               # Spread 估计
│   ├── order_flow.py           # 订单流
│   ├── volume_profile.py       # 成交量结构（需 aggTrades taker 信息）
│   ├── orderbook_pressure.py   # Orderbook 压力（需 orderbook 数据）
│   ├── taker_flow.py           # ★ Taker Order 预处理 + 指标
│   └── orderbook_lifecycle.py  # ★ 挂单生命周期指标
├── composite/                  # 复合指标（FVG、微观结构反转）
├── regime/                     # 市场状态识别
│   ├── volatility_regime.py
│   ├── trend_strength.py
│   └── liquidity_regime.py
└── registry.py                 # 统一注册表
```

---

## 基础指标 (indicators/base/)

### 1. Price Impact（价格冲击）

| 指标 | 函数 | 原理 | 用途 |
|------|------|------|------|
| Tick Price Impact | `tick_price_impact(trades, window_ms=5000)` | `|ΔPrice| / ΣVolume` 的滚动窗口计算 | 高 impact → 流动性被消耗 → 可能反转 |
| Volume-Weighted Impact | `volume_weighted_impact(trades, window_ms=10000)` | `Σ(|ΔP_i| * V_i) / ΣV_i` | 成交量加权的冲击度，更稳定 |
| Kyle's Lambda | `kyles_lambda(trades, window_trades=100)` | `Cov(ΔP, signed_flow) / Var(signed_flow)` | 衡量价格对信息交易的敏感度 |

**机制**：大量市价单成交时，价格单方向剧烈波动，impact 飙升。这通常意味着流动性正在被消耗。

### 2. Spread 估计

| 指标 | 函数 | 原理 | 用途 |
|------|------|------|------|
| Trade Diff Spread | `trade_diff_spread(trades, window_trades=100)` | 相邻成交价差的滚动中位数 | 代理 bid-ask spread |
| Roll Spread | `roll_spread(trades, window_trades=200)` | `2 * sqrt(max(0, -Cov(ΔP_t, ΔP_{t-1})))` | Roll (1984) 经典估计量 |
| Effective Spread | `effective_spread(trades, window_ms=5000)` | 同窗口内买入最低价与卖出最高价差 | 实际交易成本 |
| Spread Percentile | `spread_percentile(spread_series, lookback=500)` | Spread 的历史分位数 (0~100) | 判断 spread 是否异常扩大 |

**机制**：Spread 扩大 → 做市商退出 → 流动性降低 → 反转前兆。

### 3. Order Flow（订单流）

| 指标 | 函数 | 原理 | 用途 |
|------|------|------|------|
| Trade Imbalance | `trade_imbalance(trades, window_trades=100)` | `(BuyVol - SellVol) / TotalVol` | 方向性压力 [-1, 1] |
| VPIN | `vpin(trades, bucket_volume=1000, n_buckets=50)` | 成交量分桶的不平衡滚动均值 | 信息交易概率 |
| Aggressive Flow | `aggressive_flow_events(trades, window_ms=5000)` | 检测短时间内大量单侧成交 | 事件型信号 |
| Flow Toxicity | `flow_toxicity(trades, window_trades=200)` | `|ΔPrice_window| / mean(spread_proxy)` | Taker 对价格的破坏力 |

**机制**：极端不平衡 → 单侧流动性被消耗 → 反转信号。

### 4. Volume Profile（成交量结构）

> **需要 Binance aggTrades 数据**（`first_trade_id`, `last_trade_id` 字段）

| 指标 | 函数 | 原理 | 用途 |
|------|------|------|------|
| Taker Vol Corr | `taker_volume_corr(trades, resample_freq='1s', window=60)` | Buy/Sell volume 按 taker 聚合后的 rolling corr | corr>0 → 买卖同步放大 → 流动性真空 |
| Taker Vol Autocorr | `taker_volume_autocorr(trades, ..., side='buy')` | 单侧 taker volume 的自相关 | 高 → 同侧连续扫货 → 动量 |
| Taker Vol Skewness | `taker_volume_skewness(trades, resample_freq='1s', window=120)` | Taker trade volume 分布偏度 | 高 → 超大 taker 单 → 信息交易者 |
| Buy/Sell Ratio | `buy_sell_volume_ratio(trades, ...)` | `log(BuyVol / SellVol)` | 方向性不平衡 |
| Large Trade Ratio | `large_trade_ratio(trades, ...)` | 大单成交占总成交量比例 | 机构/信息交易活跃度 |

### 5. ★ Taker Flow（Taker Order 级别指标）

> **核心创新**：将连续 aggTrades 分组为完整的 taker orders，还原市价单消耗 orderbook 的路径。

#### 预处理：`group_taker_orders(trades)`

分组规则：连续 aggTrades 满足 `first_trade_id[i] == last_trade_id[i-1] + 1` 且 `side[i] == side[i-1]` 时属于同一 taker order。

每个 taker order 的字段：

| 字段 | 含义 |
|------|------|
| `levels_swept` | 扫过的价格档位数（★ 核心：>=3 档 = 激进 taker） |
| `total_volume` | 总成交量 |
| `total_fills` | 总撮合笔数 |
| `price_impact` | `|last_price - first_price|` |
| `price_impact_pct` | Price impact 百分比 |
| `impact_per_volume` | `price_impact / total_volume` |
| `impact_per_level` | `price_impact / levels_swept` |
| `volume_taper` | 最后一档量 / 第一档量（衰减率） |
| `duration_ms` | 持续时间 |
| `fills_per_level_mean` | 每档平均 fill 数 |
| `fills_per_level_std` | 每档 fill 数标准差 |

#### 时序指标

| 指标 | 函数 | 原理 | 用途 |
|------|------|------|------|
| Avg Levels Swept | `avg_levels_swept(taker_orders, ...)` | 滚动平均吃穿档数 | 突升 → 流动性被扫空 |
| Large Taker Ratio | `large_taker_ratio(taker_orders, levels_threshold=3)` | 激进 taker 单（>=3档）占比 | 高 → 信息交易活跃 |
| Taker Imbalance | `taker_imbalance(taker_orders, ...)` | Taker order 层面买卖不平衡 | 每 taker order 只算一次，更纯净 |
| Sweep Autocorr | `sweep_depth_autocorr(taker_orders, ...)` | levels_swept 自相关 | 高 → 连续扫盘 → 趋势加速 |
| Impact Efficiency | `impact_efficiency(taker_orders, ...)` | `Σimpact / Σvolume`（Kyle's Lambda taker版） | 高 → 流动性薄 → 反转有利 |
| Taker Arrival Rate | `taker_arrival_rate(taker_orders, ...)` | 每时间桶的 taker order 数 | 活跃度衡量 |
| Taker Size Skewness | `taker_size_skewness(taker_orders, ...)` | Taker order size 偏度 | 高 → 少量超大单 |
| Iceberg Score | `iceberg_score(taker_orders, ...)` | 冰山单检测：同价格高 fill count 频率 | 高 → 大型隐藏挂单 |

### 6. ★ Orderbook Lifecycle（挂单生命周期）

> **需要 orderbook snapshot 序列**

| 指标 | 函数 | 原理 | 用途 |
|------|------|------|------|
| Depth Change Rate | `depth_change_rate(book, levels=5, window=10)` | bid/ask 深度变化率 | 正=补单，负=消耗/撤单 |
| Refill Frequency | `refill_frequency(book, levels=5, window=20)` | 深度下降后恢复的频率 | 高→做市商积极→支撑 |
| Cancel Rate | `cancel_rate(book, levels=5, window=20)` | depth下降但mid不变的频率 | 高→虚假流动性 |
| Depth Resilience | `depth_resilience(book, levels=5)` | 深度冲击后的恢复速度 | 高韧性→反转信号 |
| Level Thickness Profile | `level_thickness_profile(book, levels=10)` | Top3档占总深度比例 | 高集中→易被冲击 |

### 7. Orderbook Pressure

> **需要 orderbook 数据**

| 指标 | 函数 | 原理 | 用途 |
|------|------|------|------|
| VWAP Pressure | `vwap_pressure(book, levels=10)` | `(ask_vwap_N - mid) / mid` | ask_vwap 远离 mid → 卖方稀薄 |
| Depth Imbalance | `depth_imbalance(book, levels=10)` | `(bid_depth - ask_depth) / total` | 正 → 买方更厚 → 支撑 |
| Depth Slope | `weighted_depth_slope(book, levels=10)` | 按价格距离加权的深度分布斜率 | 陡 → 近档集中 → 易被冲击 |

---

## Regime 指标 (indicators/regime/)

### Volatility Regime

| 指标 | 函数 | 用途 |
|------|------|------|
| Realized Vol | `realized_volatility(prices, window=60)` | 对数收益率标准差 |
| Parkinson Vol | `parkinson_volatility(ohlcv, window=60)` | 高低价波动率（更高效） |
| GARCH Vol | `garch_like_vol(prices, alpha, beta, omega)` | 简化 GARCH(1,1) 递推 |
| Vol Regime | `volatility_regime(vol, lookback=500)` | "low" / "medium" / "high" |

### Trend Strength

| 指标 | 函数 | 用途 |
|------|------|------|
| ADX | `adx_indicator(ohlcv, period=14)` | 趋势强度 + DI+/DI- |
| Efficiency Ratio | `efficiency_ratio(prices, window=20)` | Kaufman 效率比 [0,1] |
| Hurst Exponent | `hurst_exponent(prices, max_lag=100)` | >0.5 趋势, <0.5 均值回归 |
| Trend Regime | `trend_regime(adx, hurst)` | "trending" / "ranging" / "transitional" |

### Liquidity Regime

| 指标 | 函数 | 用途 |
|------|------|------|
| Amihud | `amihud_illiquidity(ohlcv, window=20)` | `mean(|r| / V)` 非流动性 |
| Turnover | `turnover_ratio(ohlcv, window=20)` | 成交量/均值 |
| Liq Regime | `liquidity_regime(amihud, lookback=500)` | "liquid" / "normal" / "illiquid" |

---

## 复合指标 (cta/)

FVG 和微观结构反转是由基础指标通过模型/规则组合得到的复合指标：

- **FVG V3** (`cta/fvg/features_v3.py`): 使用 indicators/ 全面指标 + Triple Barrier 标签 + 可配置独立时间窗口
- **FVG V1** (`cta/fvg/features.py`): 原始手动特征提取（向后兼容）
- **Microstructure Reversal** (`cta/microstructure/`): Impact + Spread + Imbalance 多条件复合

---

## 数据要求

| 指标类别 | 最低数据 | 推荐数据 |
|----------|----------|----------|
| Price Impact / Spread / Order Flow | 逐笔 trades（任何来源） | Binance aggTrades（含 taker 信息）|
| Volume Profile | **Binance aggTrades** | 必须有 agg_trade_id / first_trade_id |
| **Taker Flow** | **Binance aggTrades** | 必须有 first_trade_id / last_trade_id |
| Orderbook Pressure / Lifecycle | **Orderbook snapshots** | Lake-API 或实时 WebSocket |
| Regime | 1min+ OHLCV | 5min+ OHLCV，至少 500 bars 回溯 |

---

## 特征时间窗口配置 (FVG V3)

FVG V3 支持每个指标类别使用独立的时间窗口：

```python
DEFAULT_FEATURE_WINDOWS = {
    "price_impact":     (10000, 5000),    # pre 10s, post 5s
    "spread":           (10000, 5000),
    "imbalance":        (30000, 10000),
    "taker_flow":       (60000, 30000),   # Taker 需更长窗口
    "volume":           (30000, 10000),
    "regime":           (300000, None),   # 仅看 pre，5min 回溯
}
```

这些窗口可通过 grid search 优化。
