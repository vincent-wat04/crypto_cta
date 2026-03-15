# Usage Guide

## 目录

1. [模型训练 → 实时模拟交易](#1-模型训练--实时模拟交易)
2. [因子回测框架](#2-因子回测框架)

---

## 1. 模型训练 → 实时模拟交易

整体流程: **训练模型** → **保存 artifacts** → **启动 RealtimeSimulator** → **输出 Metrics**

### Step 1: 训练模型

```bash
# 使用 perpetual FAPI 真实数据训练 GBM 模型
python scripts/train_sol_v5_perpetual.py \
    --symbol SOL/USDT \
    --maker-fill 0.7 \
    --maker-bps -2.0 \
    --taker-bps 4.0 \
    --top-features 50 \
    --threshold 0.0
```

训练完成后, 模型 artifacts 保存至:

```
data/backtest_results/sol_v5_perpetual/
├── model_artifacts.pkl    # model, scaler, selected_features, config
├── trades.csv             # 最佳场景的回测 trade log
└── results.json           # 所有场景的 metrics + feature importance
```

`model_artifacts.pkl` 内部结构:

```python
{
    "model":              HistGradientBoostingRegressor,  # sklearn 模型
    "scaler":             StandardScaler,                 # 特征标准化器
    "selected_features":  ["feat_1", "feat_2", ...],      # 50 个选中特征名
    "feature_importance": {"feat_1": 0.12, ...},          # 特征重要性排名
    "config": {
        "symbol":           "SOL/USDT",
        "interval":         300,           # 5min = 300s
        "feature_windows":  [3, 5, 10, 20],
        "threshold":        0.0,
        "train_dates":      ["2026-01-22", ...],
        "test_dates":       ["2026-02-04", ...],
        "maker_fee_bps":    -2.0,
        "taker_fee_bps":    4.0,
    },
}
```

### Step 2: 启动实时模拟器

```python
import asyncio
from trading.realtime_simulator import RealtimeSimulator, SimulatorConfig
from trading.position_controller import FixedPositionController, SignalAdjustedController

# ── 配置 ──
config = SimulatorConfig(
    symbol="SOL/USDT",
    interval_sec=300,                    # 每 5min 产生一次信号
    ws_url="wss://fstream.binance.com/ws",  # FAPI 真实 WebSocket
    model_path="data/backtest_results/sol_v5_perpetual/model_artifacts.pkl",
    maker_fee_bps=-2.0,
    taker_fee_bps=4.0,
    maker_fill_rate=0.7,                 # 假设 70% 订单 maker 成交
    trade_buffer_size=100_000,
    warmup_bars=60,                      # 至少积累 60 个 1s bar 后才出信号
)

# ── 仓位控制 (二选一) ──

# 方案 A: 固定仓位 — 信号超过阈值就满仓
controller = FixedPositionController(
    max_position=1.0,
    threshold=0.5,     # |signal| > 0.5 才入场, 避免噪音
)

# 方案 B: 信号调整仓位 — 按信号强度缩放
controller = SignalAdjustedController(
    max_position=1.0,
    threshold=0.5,     # |signal| > 0.5 的门槛不变
    scale_factor=5.0,  # position = min(1, |signal| / 5)
    min_position=0.1,  # 最小仓位 10%
)

# ── 信号回调 (可选, 用于实时监控) ──
def on_signal(info: dict):
    print(f"[{info['time']}] signal={info['signal']:.3f}  "
          f"pos={info['position']}  cum_pnl={info['cum_pnl']:.1f} bps")

# ── 启动 ──
simulator = RealtimeSimulator(
    config=config,
    position_controller=controller,
    on_signal=on_signal,
)

asyncio.run(simulator.run())
```

模拟器运行逻辑:

```
WebSocket (fstream.binance.com)
  └→ aggTrade stream: 每笔逐笔成交
       └→ trade_buffer (deque, 最近 100K 笔)
            └→ 每 5min 边界触发:
                 1. resample trades → 1s bars
                 2. compute_all_features_v5() → 特征矩阵
                 3. model.predict(scaler.transform(X)) → signal
                 4. PositionController.compute_target(signal) → PositionTarget
                 5. execute_position_change() → PnL 计算 (含 maker/taker 成本)
                 6. on_signal() 回调 → 日志/监控
```

### Step 3: 获取模拟结果 & Metrics

在模拟器运行一段时间后 (例如 Ctrl+C 停止):

```python
# 停止
simulator.stop()

# 获取完整结果
results = simulator.get_results()
```

`results` 结构:

```python
{
    "metrics": {
        "total_return_pct":       12.34,     # 总收益率 (%)
        "annualized_return_pct":  156.78,    # 年化收益率 (%)
        "max_drawdown_pct":       -2.34,     # 最大回撤 (%)
        "volatility_ann_pct":     45.67,     # 年化波动率 (%)
        "sharpe_ratio":           2.66,      # 夏普比率
        "sortino_ratio":          3.45,      # Sortino 比率
        "calmar_ratio":           67.0,      # Calmar 比率
        "win_rate":               58.5,      # 胜率 (%)
        "profit_factor":          1.35,      # 盈亏比
        "n_trades":               120,       # 交易次数
        "n_periods":              576,       # 总周期数
        "net_pnl_bps":            1234.0,    # 净 PnL (bps)
        "gross_pnl_bps":          2000.0,    # 毛盈利 (bps)
        "loss_pnl_bps":           -766.0,    # 毛亏损 (bps)
    },
    "trade_log": [
        {
            "time":     "2026-02-03T10:05:00+00:00",
            "signal":   1.23,
            "old_pos":  0,
            "new_pos":  1,
            "price":    23.456,
            "pnl_bps":  0.0,
            "cost_bps": 2.8,       # 混合成本: 0.7*(-2) + 0.3*4 = -0.2 bps × 2
            "net_pnl":  -2.8,
            "cum_pnl":  -2.8,
        },
        ...
    ],
    "n_trades":       120,
    "n_signals":      576,
    "cumulative_pnl": 1234.0,
}
```

输出格式化报告:

```python
from trading.metrics import format_metrics_report

report = format_metrics_report(results["metrics"])
print(report)
```

输出:

```
==================================================
Backtest Metrics
==================================================
  Net PnL (bps):             1234.00
  Total Return:                12.34%
  Annualized Return:          156.78%
  Max Drawdown:                -2.34%
  Volatility (ann):            45.67%
  Sharpe Ratio:                 2.66
  Sortino Ratio:                3.45
  Calmar Ratio:                67.00
  Win Rate:                    58.50%
  Profit Factor:                1.35
  N Trades:                      120
==================================================
```

保存 trade log 到 CSV:

```python
import pandas as pd

trade_df = pd.DataFrame(results["trade_log"])
trade_df.to_csv("simulation_trades.csv", index=False)
```

### 完整脚本示例

将上述步骤整合为一个可直接运行的脚本:

```python
#!/usr/bin/env python3
"""启动实时模拟交易, 运行 N 小时后输出报告."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trading.realtime_simulator import RealtimeSimulator, SimulatorConfig
from trading.position_controller import FixedPositionController
from trading.metrics import format_metrics_report

MODEL_PATH = "data/backtest_results/sol_v5_perpetual/model_artifacts.pkl"
RUN_HOURS = 24

async def main():
    config = SimulatorConfig(
        symbol="SOL/USDT",
        interval_sec=300,
        model_path=MODEL_PATH,
        maker_fee_bps=-2.0,
        taker_fee_bps=4.0,
        maker_fill_rate=0.7,
    )
    controller = FixedPositionController(max_position=1.0, threshold=0.5)

    def on_signal(info):
        print(f"signal={info['signal']:+.3f}  pos={info['position']:+d}  "
              f"cum_pnl={info['cum_pnl']:+.0f} bps")

    sim = RealtimeSimulator(config, controller, on_signal)

    # 运行 N 小时后自动停止
    async def stop_after():
        await asyncio.sleep(RUN_HOURS * 3600)
        sim.stop()

    await asyncio.gather(sim.run(), stop_after())

    # 输出结果
    results = sim.get_results()
    print(format_metrics_report(results["metrics"]))

    import pandas as pd
    pd.DataFrame(results["trade_log"]).to_csv("sim_trades.csv", index=False)
    print(f"Trade log saved: sim_trades.csv ({results['n_trades']} trades)")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 2. 因子回测框架

### 架构概览

```
因子回测流程:

aggTrades (raw)
  │
  ▼
compute_trade_primitives()          ← data_schema.py
  │ 输出 ~30 个原子 primitives:
  │   close, volume, return_1, trade_imbalance,
  │   price_impact, signed_volume, small_vol_pi_ratio, ...
  │
  ▼
Expression Engine                    ← expressions.py
  │ 将字符串表达式解析为 AST 并执行:
  │   "ts_rank(delta(price_impact, 5), 60)"
  │   "ts_reg_beta(return_1, signed_volume, 120)"
  │
  ▼
Factor Evaluator                     ← evaluator.py
  │ 计算每个因子的:
  │   - IC (Spearman rank corr with future returns)
  │   - Rolling IC → IC stability (mean, std, IR, positive%)
  │   - IC half-life (自相关衰减)
  │   - Decile analysis (10桶 × 未来收益均值)
  │   - Factor turnover
  │
  ▼
Scanner                              ← scanner.py
  │ 批量生成 & 评估:
  │   1st-order: operator(primitive, lookback)
  │   2nd-order: op1(op2(primitive, d2), d1)
  │   Regression: ts_reg_beta(y, x, d) (Kyle's lambda)
  │   Correlation: ts_corr(x, y, d)
  │
  ▼
Results (sorted by |IC_5min|)
```

### 运行因子回测

#### 基础用法

```bash
# 快速测试 (7天数据, 仅 1st-order, 500 表达式上限)
python scripts/run_factor_scan.py \
    --symbol SOL/USDT \
    --days 7 \
    --max-order 1 \
    --max-expressions 500

# 完整扫描 (365天, 1st + 2nd order, 5000 表达式)
python scripts/run_factor_scan.py \
    --symbol SOL/USDT \
    --days 365 \
    --max-order 2 \
    --max-expressions 5000

# 包含 orderbook 数据
python scripts/run_factor_scan.py \
    --symbol SOL/USDT \
    --days 365 \
    --orderbook-path data/orderbook/ \
    --max-expressions 10000

# 调低 IC 阈值 (保留更多因子)
python scripts/run_factor_scan.py \
    --symbol SOL/USDT \
    --days 365 \
    --min-ic 0.005

# 修改 bar 频率 (默认 1s)
python scripts/run_factor_scan.py \
    --symbol SOL/USDT \
    --days 30 \
    --freq 5s
```

#### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--symbol` | SOL/USDT | 交易对 |
| `--days` | 365 | 回溯天数 |
| `--freq` | 1s | Bar 频率 (1s, 5s, 1min) |
| `--max-order` | 2 | 最大 operator 嵌套层数 (1 或 2) |
| `--max-expressions` | 5000 | 最大评估表达式数量 |
| `--min-ic` | 0.01 | IC 绝对值低于此阈值的因子跳过完整评估 |
| `--ic-window` | 100 | Rolling IC 的窗口大小 |
| `--orderbook-path` | None | Orderbook parquet 数据路径 |
| `--output-dir` | auto | 输出目录 (默认 data/factor_scan_results/{symbol}/) |
| `--log-level` | INFO | 日志级别 |

#### 输出文件

```
data/factor_scan_results/SOL_USDT/
├── factor_scan_results.parquet   # 完整结果 (用于后续分析)
├── factor_scan_results.csv       # CSV 格式 (可读性)
├── top_factors_summary.txt       # Top 50 因子详细报告
└── scan_config.json              # 扫描配置记录
```

#### 结果字段说明

每行一个因子, 包含以下字段:

| 字段 | 说明 |
|------|------|
| `expression` | 因子表达式, 如 `ts_rank(delta(price_impact, 5), 60)` |
| `order` | Operator 嵌套层数 (1 或 2) |
| `fields` | 用到的 primitives |
| `ic_1min` | 对未来 1min return 的 IC |
| `ic_5min` | 对未来 5min return 的 IC |
| `ic_mean_5min` | Rolling IC 均值 |
| `ic_std_5min` | Rolling IC 标准差 |
| `ic_ir_5min` | IC Information Ratio (mean/std) |
| `ic_positive_pct_5min` | IC 为正的时间占比 (%) |
| `ic_half_life_5min` | IC 自相关半衰期 (越长越稳定) |
| `decile_spread_5min` | Top 桶 vs Bottom 桶收益差 (bps) |
| `decile_monotonicity_5min` | 桶序与收益的 Spearman 相关 (越接近 ±1 越单调) |
| `turnover` | 因子排名的日均变化幅度 (越低 = 越稳定 = 交易成本越低) |

### Operators 完整列表

#### 一元 Operators (Unary): `op(x, d)`

| Operator | 功能 |
|----------|------|
| `delay(x, d)` | x 的 d 期前的值 |
| `delta(x, d)` | x[t] - x[t-d] |
| `ts_mean(x, d)` | d 期滚动均值 |
| `ts_stddev(x, d)` | d 期滚动标准差 |
| `ts_sum(x, d)` | d 期滚动求和 |
| `ts_product(x, d)` | d 期滚动乘积 |
| `ts_min(x, d)` | d 期滚动最小值 |
| `ts_max(x, d)` | d 期滚动最大值 |
| `ts_argmin(x, d)` | 最小值发生的位置 |
| `ts_argmax(x, d)` | 最大值发生的位置 |
| `ts_rank(x, d)` | d 期内百分位排名 (0~1) |
| `decay_linear(x, d)` | 线性衰减加权均值 |
| `decay_exp(x, d, f)` | 指数衰减加权均值 |
| `ts_skew(x, d)` | d 期滚动偏度 |
| `ts_kurt(x, d)` | d 期滚动峰度 |
| `ts_median(x, d)` | d 期滚动中位数 |
| `ts_entropy(x, d)` | d 期滚动 Shannon 熵 |
| `ts_zscore(x, d)` | 滚动 z-score |
| `ts_returns(x, d)` | d 期百分比变化 |
| `ts_acceleration(x, d)` | 二阶差分: delta(delta(x,d),d) |
| `ts_momentum(x, d)` | x[t]/x[t-d] - 1 |
| `ts_range(x, d)` | d 期极差: max - min |
| `ts_cv(x, d)` | 变异系数: std/|mean| |
| `signedpower(x, a)` | 带符号幂: sign(x)·|x|^a |
| `log1p_safe(x)` | log(1+|x|)·sign(x) |
| `abs_val(x)` | 绝对值 |

#### 二元 Operators (Binary): `op(x, y, d)`

| Operator | 功能 |
|----------|------|
| `ts_corr(x, y, d)` | d 期滚动 Pearson 相关 |
| `ts_cov(x, y, d)` | d 期滚动协方差 |
| `ts_reg_beta(y, x, d)` | 滚动 OLS 斜率 (Kyle's λ) |
| `ts_reg_alpha(y, x, d)` | 滚动 OLS 截距 |
| `ts_reg_residual(y, x, d)` | 滚动 OLS 当期残差 |
| `ts_reg_r2(y, x, d)` | 滚动 OLS R² |

### Primitives 完整列表

#### Trade Primitives (来自 aggTrades)

| Primitive | 说明 |
|-----------|------|
| `close, open, high, low` | OHLC 价格 |
| `volume` | 成交量 |
| `return_1` | 1-bar 收益 (bps) |
| `log_return` | 对数收益 |
| `price_range` | (H-L)/C (bps) |
| `buy_volume, sell_volume` | 买/卖成交量 |
| `signed_volume` | 净成交量 (buy - sell), **Kyle's lambda 回归 x 变量** |
| `trade_imbalance` | (buy-sell)/(buy+sell) |
| `abs_imbalance` | |trade_imbalance| |
| `buy_volume_pct` | 买方占比 |
| `n_trades, n_agg_trades` | 成交笔数 |
| `avg_trade_size` | 平均成交量 |
| `fills_per_agg` | 每 aggTrade 的成交笔数 |
| `price_impact` | |return|/volume, **PI 基础量** |
| `signed_price_impact` | return/volume (带方向) |
| `small_vol_pi_ratio` | 小单 PI / 大单 PI, **流动性脆弱指标** |
| `small_vol_signed_volume` | 小成交量 bar 的净成交量 |
| `small_vol_return` | 小成交量 bar 的收益 |
| `small_vol_volume` | 小成交量 bar 的成交量 |
| `vwap` | 成交量加权均价 |
| `vwap_deviation` | (VWAP-close)/close (bps) |
| `volume_at_high` | close>(H+L)/2 时为 1 |
| `bar_efficiency` | |close-open|/(H-L), 方向性确认 |
| `body_direction` | sign(close-open) |
| `upper_wick_pct, lower_wick_pct` | 上/下影线占比 |
| `volume_per_range` | volume/price_range |

#### Orderbook Primitives (来自 LakeAPI)

| Primitive | 说明 |
|-----------|------|
| `mid_price` | (best_bid + best_ask) / 2 |
| `spread, spread_bps` | 买卖价差 |
| `bid_depth_N, ask_depth_N` | N 档深度 (N=1,3,5,10) |
| `total_depth_5` | 5 档总深度 |
| `depth_imbalance` | (bid_depth-ask_depth)/total |
| `book_skew` | 首档 size 偏斜 |
| `top_bid_size, top_ask_size` | 首档挂单量 |
| `top_concentration_bid/ask` | 首档占比 |

### 因子表达式示例

```python
# ── Kyle's Lambda (全量交易) ──
# ΔP = λ·V + c, 其中 λ = price impact per unit order flow
"ts_reg_beta(return_1, signed_volume, 60)"    # 60s 窗口
"ts_reg_beta(return_1, signed_volume, 300)"   # 5min 窗口

# ── Kyle's Lambda (小单) ──
# 小单引发异常 PI → 订单簿薄弱
"ts_reg_beta(small_vol_return, small_vol_signed_volume, 120)"

# ── Lambda 变化率: 信息交易加速 ──
"delta(ts_reg_beta(return_1, signed_volume, 60), 120)"
"ts_zscore(ts_reg_beta(return_1, signed_volume, 60), 300)"

# ── PI 加速度: MM 减薄订单簿 ──
"ts_acceleration(price_impact, 10)"
"ts_zscore(ts_acceleration(price_impact, 5), 60)"

# ── 毒性反馈环 ──
"ts_corr(price_impact, abs_imbalance, 120)"

# ── 成交量分布熵: 信息 vs 噪音 ──
"ts_entropy(volume, 300)"
"delta(ts_entropy(volume, 120), 60)"

# ── Spread-Volume 反馈 ──
"ts_corr(delta(price_range, 1), volume, 60)"

# ── 传统动量 ──
"ts_rank(delta(return_1, 5), 60)"
"decay_linear(delta(return_1, 10), 300)"
```

### 在 Python 中直接使用

```python
from factors.data_schema import compute_trade_primitives
from factors.expressions import FactorExpression
from factors.evaluator import evaluate_factor, compute_decile_returns

# 1. 准备数据
primitives = compute_trade_primitives(trades_df, freq="1s")

# 2. 计算前瞻收益
cum_ret = primitives["return_1"].cumsum()
returns_1min = cum_ret.shift(-60) - cum_ret
returns_5min = cum_ret.shift(-300) - cum_ret

# 3. 评估单个因子
expr = FactorExpression("ts_reg_beta(return_1, signed_volume, 120)")
factor_values = expr.evaluate(primitives)

metrics = evaluate_factor(factor_values, returns_1min, returns_5min, ic_window=100)
print(f"IC_1min: {metrics['ic_1min']:.4f}")
print(f"IC_5min: {metrics['ic_5min']:.4f}")
print(f"IC_IR_5min: {metrics['ic_ir_5min']:.3f}")
print(f"Decile spread 5min: {metrics['decile_spread_5min']:.2f} bps")

# 4. Decile 分析
deciles = compute_decile_returns(factor_values, returns_5min, n_buckets=10)
print(deciles)
#   bucket  count  mean_return  std_return  t_stat
# 0      1  10000        -2.34        5.12   -4.57  ← 最低分位
# ...
# 9     10  10000         3.12        4.89    6.38  ← 最高分位

# 5. 批量扫描
from factors.scanner import run_factor_scan

results = run_factor_scan(
    trades=trades_df,
    freq="1s",
    max_order=2,
    max_expressions=5000,
)
print(results[["expression", "ic_5min", "decile_spread_5min", "turnover"]].head(20))
```

### Orderbook 数据格式

如果有 orderbook 数据, 需要以下列格式 (parquet):

```
timestamp         | bid_1_price | bid_1_size | ask_1_price | ask_1_size | ... | bid_10_price | bid_10_size | ask_10_price | ask_10_size
datetime64[ns,UTC]| float64     | float64    | float64     | float64    | ... | float64      | float64     | float64      | float64
```

或者预计算格式:

```
timestamp | mid_price | spread | bid_depth_5 | ask_depth_5
```

传入方式:

```bash
python scripts/run_factor_scan.py --orderbook-path data/orderbook/
```

`data/orderbook/` 下放 parquet 文件 (按天分文件或一个大文件均可).

### 服务器运行建议

```bash
# 后台运行, 日志输出到文件
nohup python scripts/run_factor_scan.py \
    --symbol SOL/USDT \
    --days 365 \
    --max-order 2 \
    --max-expressions 10000 \
    --log-level INFO \
    > logs/factor_scan.log 2>&1 &

# 监控进度
tail -f logs/factor_scan.log
```

内存估算 (1s bars):
- 7 天 ≈ 600K bars × 30 cols ≈ 150 MB
- 30 天 ≈ 2.6M bars ≈ 650 MB
- 365 天 ≈ 31M bars ≈ 8 GB (建议 16GB+ 内存, 或用 `--freq 5s` 降采样)
