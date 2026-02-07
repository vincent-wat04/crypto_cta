# 回测引擎文档

## 架构

```
backtest/
├── __init__.py
├── labeling.py    # Triple Barrier Method 标签
├── engine.py      # 通用回测引擎
└── metrics.py     # 统计指标
```

## Triple Barrier Method（核心标签系统）

### 原理

传统标签（如"未来 N 根 K 线涨跌"）与趋势的实际持续时长脱节。
Triple Barrier Method 使用路径依赖的标签：

```
         ┌── 上轨 (+2%) ── 触及 → label = 1 (强劲上涨)
         │
entry ──┤── 时间屏障 (60 bars) ── 到达 → label = 0 (震荡)
         │
         └── 下轨 (-1.5%) ── 触及 → label = -1 (强劲下跌)
```

### 使用方法

```python
from backtest.labeling import triple_barrier_labels, multi_horizon_labels

# 基础用法
labels = triple_barrier_labels(
    prices,              # pd.Series of close prices
    upper_pct=2.0,       # 上轨 +2%
    lower_pct=1.5,       # 下轨 -1.5%
    max_bars=60,         # 时间屏障 60 bars
)

# 返回 DataFrame:
#   label: 1 / -1 / 0
#   barrier_hit: "upper" / "lower" / "time"
#   bars_to_hit: 触发屏障所需 bar 数
#   return_at_hit: 触发时收益率 (%)
#   max_favorable: 最大有利偏移 (%)
#   max_adverse: 最大不利偏移 (%)

# 多时间跨度
multi = multi_horizon_labels(
    prices,
    horizons=[10, 30, 60, 120],  # 多个时间屏障
    upper_pct=0.5,
    lower_pct=0.3,
)
```

### 波动率自适应轨道

```python
labels = triple_barrier_labels(
    prices,
    volatility_adjusted=True,   # 启用波动率调整
    vol_lookback=100,           # 波动率回溯
    vol_multiplier_up=2.0,      # 上轨 = vol * 2.0
    vol_multiplier_down=1.5,    # 下轨 = vol * 1.5
    max_bars=60,
)
```

---

## 单指标回测

对单个指标进行标准化信号回测。

```python
from backtest.engine import backtest_single_indicator

result = backtest_single_indicator(
    indicator=indicator_series,      # 指标值 Series
    prices=close_prices,             # 价格 Series
    
    # 信号生成
    long_threshold=0.3,              # > 0.3 做多
    short_threshold=-0.3,            # < -0.3 做空
    # 或自定义信号函数:
    # signal_fn=lambda x: 1 if x > 0.5 else (-1 if x < -0.5 else 0),
    
    # 交易参数
    hold_bars=30,                    # 最大持仓
    stop_loss_pct=1.0,               # 止损 1%
    take_profit_pct=2.0,             # 止盈 2%
    cooldown_bars=5,                 # 冷却期
    
    # Triple Barrier 评估
    use_triple_barrier=True,
    tb_upper_pct=2.0,
    tb_lower_pct=1.5,
    tb_max_bars=60,
)

# 返回
result["metrics"]       # Dict: sharpe, win_rate, profit_factor, ...
result["results_df"]    # DataFrame: 每笔交易明细
result["label_analysis"] # Dict: Triple Barrier 标签分析
```

---

## 复合策略回测

用于 FVG + ML 模型等复合策略。

```python
from backtest.engine import backtest_composite

result = backtest_composite(
    feature_df=features,         # 特征矩阵
    model=trained_model,         # 训练好的模型 (需要 .predict 方法)
    ohlcv=ohlcv_data,           # OHLCV 数据
    hold_bars=10,
    min_proba=0.6,               # 最小预测概率
    stop_loss_pct=0.2,
    take_profit_pct=0.4,
    cooldown_bars=3,
)
```

---

## 统计指标

```python
from backtest.metrics import compute_backtest_metrics

metrics = compute_backtest_metrics(results_df)
# {
#   "n_signals": 150,
#   "n_wins": 90,
#   "win_rate": 60.0,
#   "total_return": 12.5,
#   "avg_return": 0.083,
#   "avg_win": 0.15,
#   "avg_loss": -0.05,
#   "profit_factor": 2.7,
#   "sharpe": 2.1,
#   "sortino": 3.5,
#   "max_drawdown": -2.3,
#   "median_return": 0.05,
#   "skewness": 0.8,
# }
```

---

## 数据存储

回测结果自动保存到 `data/backtest_results/{strategy}/` 目录：

```python
from utilities.paths import DataPaths

# 保存回测结果
path = DataPaths.backtest_result("fvg_v2", "run_20260203")
results_df.to_csv(path)

# 保存模型
model_path = DataPaths.model("fvg_v2", "best_model.pkl")
```

---

## 与 cta/backtest.py 的关系

`cta/backtest.py` 是旧的 FVG 专用回测引擎，保留用于向后兼容。
新的 `backtest/engine.py` 是通用引擎，支持单指标和复合策略。

建议：
- 新研究使用 `backtest/engine.py`
- FVG 策略可继续使用 `cta/backtest.py`（内部调用逻辑一致）
