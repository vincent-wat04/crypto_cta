# indicators/ → WorldQuant Brain 框架映射

逐一检查 `indicators/` 中所有指标, 确认能否用 `factors/` 的 **primitives + operators** 表达式构造出来。

状态说明:
- OK = 已有 primitive 和 operator 可直接构造
- NEW_PRIM = 需要新增 primitive (已在本文档末尾补充)
- GAP = 框架缺少对应能力, 需要扩展

---

## 1. indicators/base/returns_momentum.py

| 原始指标 | WQ 表达式 | 状态 |
|----------|-----------|------|
| `return_1` | primitive `return_1` | OK |
| `log_return` | primitive `log_return` | OK |
| `return_sum_w` | `ts_sum(return_1, w)` | OK |
| `return_std_w` | `ts_stddev(return_1, w)` | OK |
| `return_skew_w` | `ts_skew(return_1, w)` | OK |
| `return_kurt_w` | `ts_kurt(return_1, w)` | OK |
| `return_zscore_w` | `ts_zscore(return_1, w)` | OK |

## 2. indicators/base/bar_structure.py

| 原始指标 | WQ 表达式 | 状态 |
|----------|-----------|------|
| `bar_range_bps` | primitive `price_range` | OK |
| `body_ratio` | primitive `bar_efficiency` (等价) | OK |
| `bar_direction` | primitive `body_direction` | OK |
| `upper_wick_pct` | primitive `upper_wick_pct` | OK |
| `lower_wick_pct` | primitive `lower_wick_pct` | OK |
| `bar_efficiency` | primitive `bar_efficiency` | OK |
| `volume_per_range` | primitive `volume_per_range` | OK |
| `volume_imbalance` | primitive `trade_imbalance` | OK |
| `direction_consistency_w` | `ts_mean(body_direction, w)` | OK |
| `volume_ma_w` | `ts_mean(volume, w)` | OK |
| `volume_ratio_w` | 需要除法 → 见下方 **GAP-1** | NEW_PRIM |
| `signed_flow_w` | `ts_sum(signed_volume, w)` | OK |
| `signed_flow_norm_w` | 需要除法 → 见下方 **GAP-1** | NEW_PRIM |

## 3. indicators/base/order_flow.py

| 原始指标 | WQ 表达式 | 状态 |
|----------|-----------|------|
| `trade_imbalance` | primitive `trade_imbalance`; `ts_mean(trade_imbalance, w)` | OK |
| `vpin` | 等量分桶 + BVC → 见下方 **GAP-2** | GAP |
| `aggressive_flow_events` | 事件检测 (threshold), 非时序指标 → 不适合表达式框架 | N/A |
| `flow_toxicity` | `delta(close, w) / ts_mean(price_range, w)` (近似) | OK |

## 4. indicators/base/tick_statistics.py

| 原始指标 | WQ 表达式 | 状态 |
|----------|-----------|------|
| `tick_n_prices` | 需要 bar 内 distinct price count → 见 **NEW_PRIM-1** | NEW_PRIM |
| `tick_max_trade` | 需要 bar 内 max single trade size → 见 **NEW_PRIM-2** | NEW_PRIM |
| `tick_avg_trade` | primitive `avg_trade_size` | OK |
| `tick_size_cv` | 需要 bar 内 trade size cv → 见 **NEW_PRIM-3** | NEW_PRIM |
| `tick_vwap_dev` | primitive `vwap_deviation` | OK |
| `tick_buy_avg_size` | 需要 bar 内 buy avg size → 见 **NEW_PRIM-4** | NEW_PRIM |
| `tick_sell_avg_size` | 需要 bar 内 sell avg size → 见 **NEW_PRIM-4** | NEW_PRIM |
| `tick_size_imbalance` | 可从 buy/sell avg size 推导 → 见 **NEW_PRIM-4** | NEW_PRIM |
| `tick_large_pct` | 需要 bar 内 large trade ratio → 见 **NEW_PRIM-5** | NEW_PRIM |
| `tick_range_per_trade` | `price_range / n_trades` → 见 **GAP-1** | NEW_PRIM |
| `tick_n_agg` | primitive `n_agg_trades` | OK |
| `tick_*_ma_w` (rolling mean) | `ts_mean(tick_*, w)` 对任意 primitive 生效 | OK |

## 5. indicators/base/price_impact.py

| 原始指标 | WQ 表达式 | 状态 |
|----------|-----------|------|
| `tick_price_impact` | primitive `price_impact` = \|return_1\| / volume | OK |
| `volume_weighted_impact` | 近似: `ts_mean(price_impact, w)` (PI 已经是 per-bar 量) | OK |
| `kyles_lambda` | `ts_reg_beta(return_1, signed_volume, w)` | OK |

## 6. indicators/base/spread.py

| 原始指标 | WQ 表达式 | 状态 |
|----------|-----------|------|
| `trade_diff_spread` | 需要 tick-level 价差中位数 → 见 **NEW_PRIM-6** | NEW_PRIM |
| `roll_spread` | 需要 Roll estimator → 见 **NEW_PRIM-7** | NEW_PRIM |
| `effective_spread` | 需要 bar 内 buy/sell 最高最低价 → 见 **NEW_PRIM-8** | NEW_PRIM |
| `spread_percentile` | `ts_rank(roll_spread, w)` (一旦有 primitive) | OK |

## 7. indicators/base/volume_profile.py

| 原始指标 | WQ 表达式 | 状态 |
|----------|-----------|------|
| `taker_vol_corr` | `ts_corr(buy_volume, sell_volume, w)` | OK |
| `taker_vol_autocorr` | 需要 ts_autocorr → 见 **GAP-3** | GAP |
| `taker_vol_skewness` | `ts_skew(volume, w)` | OK |
| `buy_sell_vol_ratio` | 需要 log 除法 → 见 **NEW_PRIM-9** | NEW_PRIM |
| `large_trade_ratio` | primitive `small_vol_pi_ratio` (反向概念); 或 **NEW_PRIM-5** | NEW_PRIM |

## 8. indicators/base/trade_microstructure.py

| 原始指标 | WQ 表达式 | 状态 |
|----------|-----------|------|
| `avg_fill_size` | primitive `fills_per_agg` 的倒数变体; primitive `avg_trade_size` | OK |
| `level_vol_concentration` | 需要 bar 内 volume HHI → 见 **NEW_PRIM-10** | NEW_PRIM |
| `level_top1_vol_share` | 需要 bar 内 top-level volume share → 见 **NEW_PRIM-10** | NEW_PRIM |
| `buy_autocorr` | `ts_corr(buy_volume, delay(buy_volume, 1), w)` → 见 **GAP-3** | GAP |
| `sell_autocorr` | `ts_corr(sell_volume, delay(sell_volume, 1), w)` → 见 **GAP-3** | GAP |
| `cross_corr` | `ts_corr(buy_volume, sell_volume, w)` | OK |
| `side_run_length` | 需要连续同侧长度 → 见 **NEW_PRIM-11** | NEW_PRIM |
| `n_agg_trades` | primitive `n_agg_trades` | OK |
| `trade_rate` | `n_agg_trades` / bar_seconds → 见 **NEW_PRIM-12** | NEW_PRIM |
| `vwap_close_dev` | primitive `vwap_deviation` | OK |

## 9. indicators/base/taker_flow.py

| 原始指标 | WQ 表达式 | 状态 |
|----------|-----------|------|
| `avg_levels_swept` | 需要 taker order 分组 → 见 **GAP-4** | GAP |
| `large_taker_ratio` | 需要 taker order 分组 → 见 **GAP-4** | GAP |
| `taker_imbalance` | primitive `trade_imbalance` (bar-level 近似) | OK |
| `sweep_depth_autocorr` | 需要 taker order 分组 → 见 **GAP-4** | GAP |
| `impact_efficiency` | primitive `price_impact`; `ts_mean(price_impact, w)` | OK |
| `taker_arrival_rate` | primitive `n_agg_trades` (近似) | OK |
| `taker_size_skewness` | `ts_skew(volume, w)` (bar-level 近似) | OK |
| `iceberg_score` | 需要 fills_per_level 统计 → 见 **GAP-4** | GAP |

## 10. indicators/base/orderbook_pressure.py

| 原始指标 | WQ 表达式 | 状态 |
|----------|-----------|------|
| `vwap_pressure` | 需要 OB VWAP 偏离 → 见 **NEW_PRIM-13** | NEW_PRIM |
| `depth_imbalance` | OB primitive `depth_imbalance` | OK |
| `weighted_depth_slope` | 需要档位回归 → 见 **NEW_PRIM-14** | NEW_PRIM |

## 11. indicators/base/orderbook_lifecycle.py

| 原始指标 | WQ 表达式 | 状态 |
|----------|-----------|------|
| `depth_change_rate` | `ts_mean(ts_returns(total_depth_5, 1), w)` | OK |
| `depth_change_std` | `ts_stddev(ts_returns(total_depth_5, 1), w)` | OK |
| `depth_change_mom` | `delta(total_depth_5, 1) - ts_mean(delta(total_depth_5, 1), w)` | OK |
| `depth_change_asymmetry` | 需要分侧深度 → primitive `bid_depth_5`, `ask_depth_5` 已有; `ts_mean(ts_returns(bid_depth_5, 1), w) - ts_mean(ts_returns(ask_depth_5, 1), w)` | OK |
| `depth_resilience` | 复杂条件逻辑 (shock detection + recovery) → 见 **GAP-5** | GAP |
| `level_thickness_profile` | OB primitives `top_concentration_bid`, `top_concentration_ask` | OK |

## 12. indicators/regime/volatility_regime.py

| 原始指标 | WQ 表达式 | 状态 |
|----------|-----------|------|
| `realized_volatility` | `ts_stddev(log_return, w)` | OK |
| `parkinson_volatility` | 需要 Parkinson 公式 → 见 **NEW_PRIM-15** | NEW_PRIM |
| `garch_like_vol` | 递归状态 → 见 **GAP-6** | GAP |
| `volatility_regime` | 分类 label → 不适合表达式 (用 `ts_rank` 做连续替代) | N/A |

## 13. indicators/regime/liquidity_regime.py

| 原始指标 | WQ 表达式 | 状态 |
|----------|-----------|------|
| `amihud_illiquidity` | primitive `price_impact` = \|return\|/volume → `ts_mean(price_impact, w)` | OK |
| `turnover_ratio` | 见 **GAP-1** (ratio primitive) | NEW_PRIM |
| `liquidity_regime` | 分类 label → 用 `ts_rank(ts_mean(price_impact, w1), w2)` 做连续替代 | N/A |

## 14. indicators/regime/trend_strength.py

| 原始指标 | WQ 表达式 | 状态 |
|----------|-----------|------|
| `adx` | 需要 True Range + DM smoothing → 见 **NEW_PRIM-16** | NEW_PRIM |
| `efficiency_ratio` | primitive `bar_efficiency` (单 bar); 多 bar: 见 **NEW_PRIM-17** | NEW_PRIM |
| `hurst_exponent` | R/S 分析需要嵌套循环 → 见 **GAP-7** | GAP |
| `trend_regime` | 分类 label → 不适合表达式 | N/A |

---

## GAP 汇总 & 解决方案

### GAP-1: 表达式不支持 primitive 之间的四则运算

现状: 表达式引擎只支持 `operator(primitive, d)` 和 `operator(series1, series2, d)`, 不支持 `primitive_a / primitive_b` 这种 inline 运算。

**解决方案**: 将常用比率作为 primitive 在 `data_schema.py` 中预计算。这符合 "primitives are atomic per-bar quantities" 的设计, 比率本身就是原子量。需新增:
- `volume_ratio_20`: volume / ts_mean(volume, 20)
- `range_per_trade`: price_range / n_trades
- `buy_sell_log_ratio`: log(buy_volume / sell_volume)

### GAP-2: VPIN (Volume-Synchronized Probability of Informed Trading)

VPIN 需要等量分桶 (volume-clock resampling) + BVC (Bulk Volume Classification), 是完全不同的时钟框架, 无法用 time-bar 的 TS operators 表达。

**解决方案**: 作为特殊 primitive 预计算。在 `data_schema.py` 的 `compute_trade_primitives` 中增加 `vpin` 列。计算完后可正常应用 TS operators: `ts_rank(vpin, d)`, `delta(vpin, d)` 等。

### GAP-3: ts_autocorr (自相关)

`ts_corr(x, delay(x, lag), d)` 理论上能表达自相关, 但当前表达式引擎不支持 `delay` 作为 `ts_corr` 的内嵌参数。

**解决方案**: 新增 `ts_autocorr(x, d, lag)` 作为独立 operator。

### GAP-4: Taker Order 分组指标

`levels_swept`, `large_taker_ratio`, `iceberg_score` 需要将 aggTrades 按 `first_trade_id == last_trade_id[prev] + 1` 分组为 taker orders, 然后在 taker order 粒度上计算。这是 aggTrade 级别的预处理, 无法用 bar-level TS operators 表达。

**解决方案**: 在 `data_schema.py` 中增加 `compute_taker_order_primitives(trades, freq)`, 将 taker order 统计聚合到 bar 级别, 输出 primitives:
- `taker_avg_levels_swept`: bar 内平均吃穿档数
- `taker_large_ratio`: 大型 taker 占比
- `taker_iceberg_score`: 冰山单比率
- `taker_volume_taper`: 量递减比率

之后这些 primitive 可正常接入表达式: `ts_zscore(taker_avg_levels_swept, d)` 等。

### GAP-5: Depth Resilience (深度韧性)

需要条件逻辑: 检测 shock → 计算 recovery ratio。包含 if/else 判断, 不是纯 TS operator。

**解决方案**: 作为 OB primitive 预计算: `compute_orderbook_primitives` 中增加 `bid_resilience`, `ask_resilience`。

### GAP-6: GARCH-like Volatility

递归更新: σ²[t] = ω + α·r²[t-1] + β·σ²[t-1], 是有状态的递推, 无法用无状态 rolling window 精确复制。

**解决方案**: 作为特殊 primitive 预计算。或用近似: `decay_exp(signedpower(return_1, 2), d, 0.94)` ≈ EWMA variance (RiskMetrics 方法), 这是表达式可构造的。

### GAP-7: Hurst Exponent

R/S 分析需要多尺度嵌套循环 + log-log 回归, 计算复杂度高, 无法用单个 TS operator 表达。

**解决方案**: 作为特殊 primitive 预计算。或用近似替代:
- `ts_autocorr(return_1, d, 1)` 反映持续性
- 方差比: `ts_stddev(return_1, d) / ts_stddev(return_1, d/2)` 的比值 → 随机游走 = √2, 趋势 > √2, 均值回归 < √2

---

## 需要新增的 Primitives

### 已标记的 NEW_PRIM, 按优先级排列:

| ID | Primitive | 来源 | 计算方式 |
|----|-----------|------|----------|
| **NEW_PRIM-1** | `tick_n_prices` | tick_stats | bar 内 distinct price count |
| **NEW_PRIM-2** | `tick_max_trade` | tick_stats | bar 内 max(amount) |
| **NEW_PRIM-3** | `tick_size_cv` | tick_stats | bar 内 std(amount)/mean(amount) |
| **NEW_PRIM-4** | `buy_avg_size`, `sell_avg_size`, `size_imbalance` | tick_stats | bar 内 buy/sell 各自 mean(amount) |
| **NEW_PRIM-5** | `large_trade_pct` | tick_stats/volume_profile | bar 内 amount > 2×mean 的占比 |
| **NEW_PRIM-6** | `tick_spread_median` | spread | bar 内 \|Δprice\| 的中位数 |
| **NEW_PRIM-7** | `roll_spread` | spread | 2×√(-Cov(Δp[t], Δp[t-1])) bar 内 |
| **NEW_PRIM-8** | `effective_spread` | spread | bar 内 buy_min_price - sell_max_price |
| **NEW_PRIM-9** | `buy_sell_log_ratio` | volume_profile | log(buy_volume / sell_volume) |
| **NEW_PRIM-10** | `vol_concentration_hhi` | trade_microstructure | bar 内 volume 的 HHI (price-bucket) |
| **NEW_PRIM-11** | `side_run_length` | trade_microstructure | bar 内平均连续同侧交易长度 |
| **NEW_PRIM-12** | `trade_rate` | trade_microstructure | n_agg_trades / bar_seconds |
| **NEW_PRIM-13** | `ob_ask_pressure`, `ob_bid_pressure` | orderbook_pressure | (ask_vwap - mid) / mid |
| **NEW_PRIM-14** | `ob_ask_depth_slope`, `ob_bid_depth_slope` | orderbook_pressure | depth vs distance 回归斜率 |
| **NEW_PRIM-15** | `parkinson_vol` | volatility_regime | √(ln(H/L)² / 4ln2) |
| **NEW_PRIM-16** | `adx`, `di_plus`, `di_minus` | trend_strength | Wilder's ADX |
| **NEW_PRIM-17** | `efficiency_ratio_w` | trend_strength | \|P[t]-P[t-w]\| / Σ\|ΔP\| |

### 需要新增的 Operators

| ID | Operator | 签名 | 说明 |
|----|----------|------|------|
| **GAP-3** | `ts_autocorr` | `ts_autocorr(x, d, lag)` | 滚动自相关: corr(x[t-d:t], x[t-d-lag:t-lag]) |

---

## 完全不可构造的指标 (N/A)

以下指标本质上不适合 WQ 表达式框架:

| 指标 | 原因 | 替代方案 |
|------|------|----------|
| `aggressive_flow_events` | 事件检测 (binary), 非连续时序 | 用 `ts_zscore(abs_imbalance, d)` 做连续替代 |
| `volatility_regime` | 分类标签 | 用 `ts_rank(ts_stddev(return_1, d1), d2)` 做连续替代 |
| `liquidity_regime` | 分类标签 | 用 `ts_rank(ts_mean(price_impact, d1), d2)` 做连续替代 |
| `trend_regime` | 分类标签 | 用 `ts_rank(efficiency_ratio_w, d)` 做连续替代 |

这些分类标签不需要在表达式框架中复现 — 连续版本 (rank/zscore) 在回归模型中更有用。

---

## 覆盖率统计

| 类别 | 总指标数 | OK | NEW_PRIM | GAP | N/A |
|------|---------|-----|----------|-----|-----|
| returns_momentum | 7 | 7 | 0 | 0 | 0 |
| bar_structure | 12 | 10 | 2 | 0 | 0 |
| order_flow | 4 | 2 | 0 | 1 | 1 |
| tick_statistics | 11 | 3 | 7 | 0 | 0 (rolling = OK) |
| price_impact | 3 | 3 | 0 | 0 | 0 |
| spread | 4 | 1 | 3 | 0 | 0 |
| volume_profile | 5 | 2 | 2 | 1 | 0 |
| trade_microstructure | 10 | 4 | 4 | 2 | 0 |
| taker_flow | 8 | 3 | 0 | 4 | 0 (bar 近似 OK) |
| orderbook_pressure | 3 | 1 | 2 | 0 | 0 |
| orderbook_lifecycle | 6 | 5 | 0 | 1 | 0 |
| volatility_regime | 4 | 1 | 1 | 1 | 1 |
| liquidity_regime | 3 | 1 | 1 | 0 | 1 |
| trend_strength | 4 | 0 | 2 | 1 | 1 |
| **合计** | **84** | **43 (51%)** | **24 (29%)** | **10 (12%)** | **4 (5%)** |

补充 NEW_PRIM 后覆盖率: **(43+24)/80 = 84%** (排除 N/A)
补充 GAP operators/primitives 后: **77/80 = 96%**
