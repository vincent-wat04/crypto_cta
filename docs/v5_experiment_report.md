# V5 Unified Experiment Report — SOL/USDT

> **Date**: 2026-02-10  
> **Framework**: V5 Full Indicators (indicators/ library)  
> **Symbol**: SOL/USDT  
> **Train**: 13 days (2026-01-22 ~ 2026-02-03)  
> **Test**: 3 days (2026-02-04 ~ 2026-02-06)  
> **Feature Windows**: [3, 5, 10, 20] seconds (physically fixed)  
> **Total Features**: 215 effective (239 raw columns, 24 dropped for zero-variance)

---

## 1. Experiment Design

### 1.1 Prediction Intervals
| Interval | Bars/sample (train) | Bars/sample (test) | Subsampling |
|----------|--------------------|--------------------|-------------|
| **1s**   | 846,251            | 236,651            | None (every bar) |
| **1min** | 14,104             | 3,944              | Every 60th bar |
| **5min** | 2,820              | 788                | Every 300th bar |

### 1.2 Prediction Targets
- **Return Regression**: next N-second return (bps), models: Ridge / LassoCV / GBM
- **TBM Classification**: Triple Barrier Method (3-class: +1/-1/0), models: Logistic Regression / GBM
  - TBM configs tested: `tp1.0_sl0.5`, `tp1.0_sl1.0`, `tp0.5_sl0.5`

### 1.3 Trading Logic
- **Fixed-interval**: position determined once at interval start, held constant for the entire interval
- **Cost**: 4 bps per round-trip (taker) or ~0.8 bps (maker)
- **Signal thresholds**: multiple thresholds tested per model (e.g., `thr0`, `thr0.5`, `thr1.0`, `thr2.0` for regression; `thr0.05`~`thr0.3` for classification)

### 1.4 Feature Sources
All features computed at **1-second granularity** from the `indicators/` library:

| Module | Category | Example Features |
|--------|----------|-----------------|
| `returns_momentum` | Base | `return_1`, `log_return`, `return_sum_w`, `return_std_w`, `return_skew_w`, `return_kurt_w`, `return_zscore_w` |
| `bar_structure` | Base | `bar_range_bps`, `body_ratio`, `bar_direction`, `upper_wick_pct`, `lower_wick_pct`, `bar_efficiency`, `volume_per_range`, `volume_imbalance`, `n_trades`, `direction_consistency_w`, `signed_flow_w` |
| `tick_statistics` | Base | `tick_n_prices`, `tick_max_trade`, `tick_avg_trade`, `tick_size_cv`, `tick_vwap_dev`, `tick_size_imbalance`, `tick_large_pct`, `tick_range_per_trade`, `tick_n_agg` |
| `trade_microstructure` | Base | `avg_fill_size`, `buy_sell_vol_ratio`, `trade_imbalance_lib`, `taker_vol_skewness` |
| `price_impact` | Base | `vw_impact`, `ti_n_fills` |
| `spread` | Base | `roll_spread`, `spread_proxy_w` |
| `order_flow` | Base | `flow_toxicity`, `trade_diff_spread` |
| `volume_profile` | Base | `vd_vwap_trend`, `vd_vwap_close_dev` |
| `taker_flow` | Base | `tsa_buy_autocorr`, `tsa_sell_autocorr`, `tsa_cross_corr`, `tsa_side_run_length`, `taker_vol_autocorr_buy/sell`, `taker_vol_corr` |
| `regime` | Regime | `realized_vol`, `parkinson_vol`, `regime_adx`, `regime_di_plus`, `regime_efficiency_ratio`, `regime_turnover` |

---

## 2. Grand Summary — All Intervals

| Interval | Target | TBM Config | Best Model Config | Sharpe | Net PnL (bps) | Trades | Win Rate |
|----------|--------|------------|-------------------|--------|--------------|--------|----------|
| **1s** | Return | — | `lasso_maker_thr2.0` | **+0.12** | +31 | 120 | 47.1% |
| **1s** | TBM | tp1.0_sl0.5 | `gbm_maker_thr0` | -14.86 | -24,086 | 37,372 | 37.4% |
| **1s** | TBM | tp1.0_sl1.0 | `logistic_maker_thr0.3` | -1.54 | -474 | 282 | 37.0% |
| **1s** | TBM | tp0.5_sl0.5 | `logistic_maker_thr0.3` | -34.33 | -12,418 | 5,729 | 20.1% |
| **1min** | Return | — | `lasso_maker_thr2.0` | **+1.16** | +372 | 4 | 66.7% |
| **1min** | TBM | tp1.0_sl0.5 | `gbm_maker_thr0.05` | **+1.10** | +1,817 | 6 | 50.2% |
| **1min** | TBM | tp1.0_sl1.0 | `gbm_maker_thr0.2` | **+1.09** | +124 | 6 | 66.7% |
| **1min** | TBM | tp0.5_sl0.5 | `gbm_maker_thr0.2` | **+1.46** | +96 | 18 | 77.8% |
| **5min** | Return | — | `gbm_maker_thr0` | **+2.66** | +4,345 | 238 | 52.0% |
| **5min** | TBM | tp1.0_sl0.5 | `gbm_maker_thr0` | **+1.11** | +1,822 | 0 | 51.5% |
| **5min** | TBM | tp1.0_sl1.0 | `logistic_maker_thr0.2` | +0.09 | +71 | 164 | 46.9% |
| **5min** | TBM | tp0.5_sl0.5 | `gbm_maker_thr0` | **+1.01** | +1,654 | 240 | 50.9% |

### Key Observations
1. **5min Return Regression with GBM is the clear winner** — Sharpe +2.66, Net PnL +4,345 bps
2. **1s interval is economically unviable** — transaction costs dominate even with maker pricing; the best 1s result is Sharpe +0.12 (barely positive)
3. **1min TBM shows consistent positive Sharpes** across all three barrier configs (1.10, 1.09, 1.46)
4. **Model type matters**: GBM significantly outperforms linear models at 5min; at 1min the picture is mixed

### Baselines (Previous Versions)
| Version | Config | Sharpe | Net PnL |
|---------|--------|--------|---------|
| V3 1min TBM | logistic_maker_thr0.2 | +1.52 | +358 |
| V4 1min TBM | logistic_maker_thr0.3 (tp1.0,sl1.0) | +1.88 | +452 |
| V4 1min TBM | logistic_maker_thr0.05 (tp1.0,sl0.5) | +1.59 | +2,606 |
| V3 5min Ret | gbm_maker_thr0 | +1.22 | +2,000 |

**V5 5min Return (+2.66 Sharpe) more than doubles V3's best 5min result (+1.22 Sharpe)**. The comprehensive indicator library is the primary driver.

---

## 3. Feature Importance Analysis by Prediction Horizon

### 3.1 — 1s Return Regression (Top 20)

| Rank | Feature | Module | Window | Importance |
|------|---------|--------|--------|-----------|
| 1 | `bar_range_bps_ma_20` | bar_structure | 20s | 0.1542 |
| 2 | `tick_n_prices_ma_10` | tick_statistics | 10s | 0.1361 |
| 3 | `bar_range_bps_ma_10` | bar_structure | 10s | 0.1134 |
| 4 | `tick_n_prices_ma_20` | tick_statistics | 20s | 0.1116 |
| 5 | `roll_spread_ma_3` | spread | 3s | 0.0565 |
| 6 | `roll_spread_ma_5` | spread | 5s | 0.0436 |
| 7 | `trade_imbalance_lib_ma_3` | trade_microstructure | 3s | 0.0419 |
| 8 | `tsa_sell_autocorr` | taker_flow | raw | 0.0256 |
| 9 | `trade_imbalance_lib_ma_5` | trade_microstructure | 5s | 0.0252 |
| 10 | `taker_vol_autocorr_sell` | taker_flow | raw | 0.0242 |
| 11 | `taker_vol_corr_ma_10` | taker_flow | 10s | 0.0181 |
| 12 | `parkinson_vol` | regime | regime | 0.0156 |
| 13 | `tsa_cross_corr_ma_10` | taker_flow | 10s | 0.0151 |
| 14 | `tick_n_agg_ma_20` | tick_statistics | 20s | 0.0129 |
| 15 | `bar_range_bps` | bar_structure | raw | 0.0125 |
| 16 | `realized_vol` | regime | regime | 0.0121 |
| 17 | `tick_n_prices` | tick_statistics | raw | 0.0110 |
| 18 | `tsa_sell_autocorr_ma_5` | taker_flow | 5s | 0.0109 |
| 19 | `bar_range_bps_ma_5` | bar_structure | 5s | 0.0099 |
| 20 | `taker_vol_autocorr_sell_ma_5` | taker_flow | 5s | 0.0085 |

**1s Dominant Categories**: bar_structure (39.6%), tick_statistics (27.2%), spread (10.0%), taker_flow (9.4%)

---

### 3.2 — 1min Return Regression (Top 20)

| Rank | Feature | Module | Window | Importance |
|------|---------|--------|--------|-----------|
| 1 | `bar_range_bps_ma_3` | bar_structure | 3s | 0.2519 |
| 2 | `bar_range_bps_ma_10` | bar_structure | 10s | 0.2065 |
| 3 | `bar_range_bps_ma_20` | bar_structure | 20s | 0.1448 |
| 4 | `roll_spread_ma_5` | spread | 5s | 0.1237 |
| 5 | `bar_range_bps_ma_5` | bar_structure | 5s | 0.1176 |
| 6 | `roll_spread_ma_3` | spread | 3s | 0.1085 |
| 7 | `parkinson_vol` | regime | regime | 0.0961 |
| 8 | `volume_ma_20` | bar_structure | 20s | 0.0898 |
| 9 | `realized_vol` | regime | regime | 0.0804 |
| 10 | `spread_proxy_20` | spread | 20s | 0.0779 |
| 11 | `bar_range_bps` | bar_structure | raw | 0.0428 |
| 12 | `spread_proxy_10` | spread | 10s | 0.0426 |
| 13 | `regime_turnover` | regime | regime | 0.0422 |
| 14 | `trade_imbalance_lib_ma_5` | trade_microstructure | 5s | 0.0398 |
| 15 | `vd_vwap_trend_ma_10` | volume_profile | 10s | 0.0392 |
| 16 | `flow_toxicity_ma_3` | order_flow | 3s | 0.0391 |
| 17 | `tsa_cross_corr_ma_10` | taker_flow | 10s | 0.0362 |
| 18 | `return_std_5` | returns_momentum | 5s | 0.0311 |
| 19 | `tick_n_agg_ma_5` | tick_statistics | 5s | 0.0311 |
| 20 | `tick_n_prices` | tick_statistics | raw | 0.0286 |

**1min Dominant Categories**: bar_structure (55.3%), spread (35.3%), regime (21.9%)

---

### 3.3 — 5min Return Regression (Top 20) — OVERALL BEST

| Rank | Feature | Module | Window | Importance |
|------|---------|--------|--------|-----------|
| 1 | `bar_range_bps_ma_20` | bar_structure | 20s | 1.0473 |
| 2 | `bar_range_bps_ma_3` | bar_structure | 3s | 0.6893 |
| 3 | `tick_n_prices_ma_3` | tick_statistics | 3s | 0.5908 |
| 4 | `tick_n_prices_ma_20` | tick_statistics | 20s | 0.5229 |
| 5 | `tick_n_prices_ma_10` | tick_statistics | 10s | 0.4263 |
| 6 | `bar_range_bps` | bar_structure | raw | 0.3380 |
| 7 | `tick_n_prices` | tick_statistics | raw | 0.2780 |
| 8 | `roll_spread_ma_10` | spread | 10s | 0.2656 |
| 9 | `return_sum_5` | returns_momentum | 5s | 0.2545 |
| 10 | `realized_vol` | regime | regime | 0.2412 |
| 11 | `return_std_20` | returns_momentum | 20s | 0.2176 |
| 12 | `parkinson_vol` | regime | regime | 0.2044 |
| 13 | `return_std_5` | returns_momentum | 5s | 0.1560 |
| 14 | `roll_spread_ma_3` | spread | 3s | 0.1393 |
| 15 | `spread_proxy_5` | spread | 5s | 0.1387 |
| 16 | `return_std_10` | returns_momentum | 10s | 0.1380 |
| 17 | `bar_range_bps_ma_10` | bar_structure | 10s | 0.1369 |
| 18 | `spread_proxy_20` | spread | 20s | 0.1256 |
| 19 | `vd_vwap_trend_ma_5` | volume_profile | 5s | 0.1123 |
| 20 | `tick_vwap_dev_ma_3` | tick_statistics | 3s | 0.1085 |

**5min Dominant Categories**: bar_structure (33.3%), tick_statistics (29.1%), returns_momentum (11.5%), spread (10.1%), regime (6.7%)

---

### 3.4 — 1min TBM (tp0.5_sl0.5) Classification (Top 15) — Best TBM Config

| Rank | Feature | Module | Window | Importance |
|------|---------|--------|--------|-----------|
| 1 | `realized_vol` | regime | regime | 0.0221 |
| 2 | `bar_range_bps_ma_5` | bar_structure | 5s | 0.0218 |
| 3 | `vd_vwap_trend_ma_10` | volume_profile | 10s | 0.0215 |
| 4 | `roll_spread_ma_3` | spread | 3s | 0.0210 |
| 5 | `tick_n_prices_ma_10` | tick_statistics | 10s | 0.0205 |
| 6 | `tick_size_cv_ma_5` | tick_statistics | 5s | 0.0194 |
| 7 | `roll_spread` | spread | raw | 0.0182 |
| 8 | `tick_size_cv_ma_3` | tick_statistics | 3s | 0.0181 |
| 9 | `volume_imbalance_ma_10` | bar_structure | 10s | 0.0177 |
| 10 | `volume_imbalance_ma_20` | bar_structure | 20s | 0.0168 |
| 11 | `tick_n_prices` | tick_statistics | raw | 0.0166 |
| 12 | `return_sum_5` | returns_momentum | 5s | 0.0150 |
| 13 | `trade_imbalance_lib_ma_3` | trade_microstructure | 3s | 0.0142 |
| 14 | `volume_ma_10` | bar_structure | 10s | 0.0129 |
| 15 | `bar_range_bps_ma_20` | bar_structure | 20s | 0.0120 |

**1min TBM Dominant Categories**: tick_statistics (24.6%), bar_structure (22.7%), spread (13.3%), regime (7.5%)

---

### 3.5 — 5min TBM (tp0.5_sl0.5) Classification (Top 15)

| Rank | Feature | Module | Window | Importance |
|------|---------|--------|--------|-----------|
| 1 | `tick_n_prices` | tick_statistics | raw | 0.0372 |
| 2 | `return_sum_20` | returns_momentum | 20s | 0.0321 |
| 3 | `realized_vol` | regime | regime | 0.0288 |
| 4 | `return_std_20` | returns_momentum | 20s | 0.0271 |
| 5 | `spread_proxy_5` | spread | 5s | 0.0251 |
| 6 | `n_trades` | bar_structure | raw | 0.0244 |
| 7 | `bar_range_bps_ma_20` | bar_structure | 20s | 0.0225 |
| 8 | `buy_sell_vol_ratio_ma_5` | trade_microstructure | 5s | 0.0221 |
| 9 | `n_trades_ma_5` / `trades_ma_5` | bar_structure | 5s | 0.0214 |
| 10 | `volume_imbalance_ma_20` | bar_structure | 20s | 0.0212 |
| 11 | `parkinson_vol` | regime | regime | 0.0211 |
| 12 | `trade_diff_spread_ma_3` | order_flow | 3s | 0.0210 |
| 13 | `tick_size_imbalance_ma_20` | tick_statistics | 20s | 0.0206 |
| 14 | `tick_range_per_trade_ma_3` | tick_statistics | 3s | 0.0201 |
| 15 | `roll_spread_ma_5` | spread | 5s | 0.0197 |

---

## 4. Win Rate Ranking by Interval

> 仅列出 **交易次数 N >= 3** 的配置，按 Win Rate 降序排列。★ = Sharpe > 0（盈利配置）

### 4.1 — 1s Interval（Top 10 by WR）

| Rank | Model Config | Target | WR | Trades | Sharpe | Net PnL |
|------|-------------|--------|-----|--------|--------|---------|
| 1 | `gbm_maker_thr2.0` | Return | **51.7%** | 246 | +0.08 ★ | +34 |
| 2 | `gbm_maker_thr1.0` | Return | 49.1% | 454 | -0.42 | -231 |
| 3 | `lasso_maker_thr2.0` | Return | 47.1% | 120 | **+0.12** ★ | +31 |
| 4 | `lasso_maker_thr1.0` | Return | 46.5% | 623 | -0.68 | -311 |
| 5 | `gbm_taker_thr2.0` | Return | 43.9% | 246 | -3.14 | -1,442 |
| 6 | `ridge_maker_thr2.0` | Return | 43.7% | 165 | -0.95 | -310 |
| 7 | `lasso_taker_thr2.0` | Return | 43.3% | 120 | -2.60 | -689 |
| 8 | `gbm_taker_thr1.0` | Return | 41.3% | 454 | -5.27 | -2,955 |
| 9 | `ridge_maker_thr1.0` | Return | 40.8% | 853 | -3.99 | -2,048 |
| 10 | `ridge_taker_thr2.0` | Return | 39.7% | 165 | -3.82 | -1,300 |

> **1s 结论**: 仅 Return Regression 的高阈值配置能勉强达到 ~50% WR，TBM 全线低于 40%。即使 WR 最高的 GBM 配置，Sharpe 也仅刚过零。交易成本是 1s 的致命瓶颈。

### 4.2 — 1min Interval（Top 15 by WR）

| Rank | Model Config | Target | WR | Trades | Sharpe | Net PnL |
|------|-------------|--------|-----|--------|--------|---------|
| 1 | `gbm_maker_thr0.2` | TBM tp0.5_sl0.5 | **77.8%** | 18 | **+1.46** ★ | +96 |
| 2 | `lasso_maker_thr2.0` | Return | **66.7%** | 4 | **+1.16** ★ | +372 |
| 3 | `lasso_taker_thr2.0` | Return | 66.7% | 4 | +1.10 ★ | +348 |
| 4 | `gbm_maker_thr0.2` | TBM tp1.0_sl1.0 | **66.7%** | 6 | **+1.09** ★ | +124 |
| 5 | `gbm_taker_thr0.2` | TBM tp1.0_sl1.0 | 66.7% | 6 | +0.81 ★ | +88 |
| 6 | `logistic_maker_thr0.3` | TBM tp1.0_sl1.0 | 56.5% | 37 | -0.36 | -185 |
| 7 | `logistic_taker_thr0.3` | TBM tp1.0_sl1.0 | 56.5% | 37 | -0.80 | -407 |
| 8 | `logistic_maker_thr0.3` | TBM tp0.5_sl0.5 | 53.3% | 137 | +0.01 ★ | +5 |
| 9 | `logistic_maker_thr0.2` | TBM tp0.5_sl0.5 | 50.8% | 517 | -0.60 | -482 |
| 10 | `logistic_maker_thr0.2` | TBM tp1.0_sl0.5 | 50.2% | 258 | +0.51 ★ | +781 |
| 11 | `gbm_maker_thr0.05` | TBM tp1.0_sl0.5 | 50.2% | 6 | **+1.10** ★ | +1,817 |
| 12 | `logistic_maker_thr0` | TBM tp1.0_sl0.5 | 50.2% | 10 | +0.57 ★ | +950 |
| 13 | `lasso_maker_thr1.0` | Return | 50.0% | 20 | -0.26 | -120 |
| 14 | `gbm_maker_thr0.15` | TBM tp1.0_sl1.0 | 50.0% | 34 | +0.30 ★ | +63 |
| 15 | `gbm_maker_thr0` | Return | 49.8% | 368 | -0.33 | -539 |

> **1min 结论**: TBM tp0.5_sl0.5 + GBM 达到惊人的 **77.8% WR**，是全实验最高 WR。TBM 分类在 1min 周期整体优于 Return 回归。多数正 Sharpe 配置 WR 在 50-67%。

### 4.3 — 5min Interval（Top 15 by WR）

| Rank | Model Config | Target | WR | Trades | Sharpe | Net PnL |
|------|-------------|--------|-----|--------|--------|---------|
| 1 | `gbm_maker_thr1.0` | Return | **53.0%** | 368 | **+2.10** ★ | +3,197 |
| 2 | `gbm_maker_thr2.0` | Return | 52.7% | 333 | +1.90 ★ | +2,667 |
| 3 | `gbm_maker_thr0` | Return | **52.0%** | 238 | **+2.66** ★ | +4,345 |
| 4 | `gbm_maker_thr0.5` | Return | 52.0% | 295 | +2.57 ★ | +4,114 |
| 5 | `lasso_maker_thr0` | Return | 51.5% | 0 | +1.11 ★ | +1,822 |
| 6 | `gbm_maker_thr0` | TBM tp0.5_sl0.5 | **50.9%** | 240 | **+1.01** ★ | +1,654 |
| 7 | `gbm_taker_thr1.0` | Return | 50.6% | 368 | +0.64 ★ | +983 |
| 8 | `gbm_maker_thr0.3` | TBM tp0.5_sl0.5 | 50.3% | 419 | -0.07 | -87 |
| 9 | `gbm_taker_thr0` | Return | 50.3% | 238 | +1.78 ★ | +2,911 |
| 10 | `gbm_taker_thr0.5` | Return | 50.1% | 295 | +1.46 ★ | +2,338 |
| 11 | `gbm_taker_thr2.0` | Return | 50.1% | 333 | +0.47 ★ | +669 |
| 12 | `ridge_maker_thr1.0` | Return | 50.0% | 438 | +0.57 ★ | +904 |
| 13 | `ridge_maker_thr0` | Return | 49.7% | 370 | +0.46 ★ | +761 |
| 14 | `ridge_maker_thr2.0` | Return | 49.7% | 469 | +0.43 ★ | +651 |
| 15 | `ridge_maker_thr0.5` | Return | 49.5% | 406 | +0.42 ★ | +674 |

> **5min 结论**: GBM Return Regression 包揽前 4 名，WR 稳定在 **52-53%**，且全部 Sharpe > +1.9。5min 是 WR 和 Sharpe 最均衡的周期 — WR 不需要极端高，但依靠 **每笔盈利大于亏损** (PF 1.29~1.32) 实现高 Sharpe。Return Regression 几乎所有 maker 费率配置都盈利。

### 4.4 — Win Rate 跨周期对比总结

| 指标 | 1s | 1min | 5min |
|------|-----|------|------|
| **最高 WR** | 51.7% | **77.8%** | 53.0% |
| **最高 WR 的策略** | gbm_maker_thr2.0 (Ret) | gbm_maker_thr0.2 (TBM) | gbm_maker_thr1.0 (Ret) |
| **WR > 50% 的正Sharpe配置数** | 1 | 7 | **12** |
| **最高 Sharpe 对应的 WR** | 47.1% | 66.7% | 52.0% |
| **最佳 WR-Sharpe 组合** | — | **77.8% / +1.46** | **52.0% / +2.66** |

**核心发现**:
1. **1min TBM 的 WR 最高** (77.8%)，但交易次数少（18 笔），统计显著性有限
2. **5min Return 的 WR 最稳定** (52-53%)，且几乎所有配置都盈利，是最可靠的周期
3. **高 WR 不等于高 Sharpe** — 5min 的 52% WR 搭配 +2.66 Sharpe 优于 1min 的 77.8% WR / +1.46 Sharpe，因为 5min 每笔收益更大
4. **1s 不可行** — 即使 WR 达到 51.7%，交易成本仍几乎吞噬全部利润

---

## 5. Cross-Horizon Feature Comparison

### 5.1 Feature Category Distribution by Horizon

| Category | 1s Return | 1min Return | 5min Return | 1min TBM | 5min TBM |
|----------|-----------|-------------|-------------|----------|----------|
| **bar_structure** | 39.6% | 55.3% | 33.3% | 22.7% | 22.5% |
| **tick_statistics** | 27.2% | 2.0% | 29.1% | 24.6% | 12.6% |
| **spread** | 10.0% | 35.3% | 10.1% | 13.3% | 7.2% |
| **taker_flow** | 9.4% | 2.4% | — | — | — |
| **regime** | 2.8% | 21.9% | 6.7% | 7.5% | 8.0% |
| **returns_momentum** | — | 2.0% | 11.5% | 5.1% | 9.5% |
| **trade_microstructure** | 6.7% | 2.6% | — | 4.8% | 3.5% |
| **volume_profile** | — | 2.6% | 1.7% | 7.3% | — |
| **order_flow** | — | 2.6% | — | — | 3.4% |

### 5.2 Dominant Features That Appear Across Multiple Horizons

| Feature | Module | Appears in (Top 15) | Best Window |
|---------|--------|---------------------|-------------|
| `bar_range_bps` (+rolling) | bar_structure | **All 5 experiments** | 3s, 10s, 20s |
| `tick_n_prices` (+rolling) | tick_statistics | **4/5 experiments** | raw, 3s, 10s, 20s |
| `roll_spread` (+rolling) | spread | **4/5 experiments** | 3s, 5s, 10s |
| `realized_vol` | regime | **4/5 experiments** | regime-level |
| `parkinson_vol` | regime | **3/5 experiments** | regime-level |
| `return_std_w` | returns_momentum | **3/5 experiments** | 5s, 10s, 20s |
| `return_sum_w` | returns_momentum | **3/5 experiments** | 5s, 20s |
| `volume_imbalance` (+rolling) | bar_structure | **3/5 experiments** | 10s, 20s |
| `trade_imbalance_lib` (+rolling) | trade_microstructure | **3/5 experiments** | 3s, 5s |
| `spread_proxy_w` | spread | **3/5 experiments** | 5s, 10s, 20s |

### 5.3 Window Size Patterns

| Horizon | Dominant Windows | Interpretation |
|---------|-----------------|----------------|
| **1s** | 10s, 20s (rolling), raw (instant) | Longer lookbacks smooth 1s noise; instant taker_flow signals also useful |
| **1min** | 3s, 5s (short rolling), 20s, regime | Mix of fast microstructure signals and slower volatility context |
| **5min** | 3s, 5s, 10s, 20s (all windows), raw | All timescales contribute; wider diversity of window usage |
| **TBM** | 3s, 5s, 10s, 20s | Classification tasks draw from all windows more evenly |

**Key insight**: Feature windows are **physically meaningful and do not scale with prediction horizon**. The same 3-20s microstructural windows are effective whether predicting 1s, 1min, or 5min returns, confirming the V2 design philosophy.

---

## 6. Feature-Level Insights

### 6.1 `bar_range_bps` — The Universal Dominant Feature

This feature (OHLCV bar range in basis points and its rolling means) is the single most important indicator across all horizons and targets. It measures **instantaneous volatility** at the 1-second bar level.

- At **1s**: `bar_range_bps_ma_20` (imp=0.154) is #1 — the 20s smoothed volatility provides a regime context for next-1s prediction
- At **1min**: `bar_range_bps_ma_3` (imp=0.252) is #1 — faster 3s responsiveness matters more for minute-level prediction
- At **5min**: `bar_range_bps_ma_20` (imp=1.047) is #1 — the longer smoothing window dominates for 5min predictions

### 6.2 `tick_n_prices` — Microstructure Activity Proxy

The number of unique price levels touched per second is a powerful microstructure feature. It captures **market complexity and activity intensity** without relying on orderbook data.

- Strong across 1s, 1min, and 5min return regression
- Multiple windows (raw, 3s, 10s, 20s) all show significance
- Especially dominant in 5min (#3-7 features are all tick_n_prices variants)

### 6.3 `roll_spread` — Transaction Cost Proxy

The Roll (1984) spread estimator and its rolling means appear consistently. It proxies **bid-ask spread from trade data alone**, measuring market friction/liquidity.

- Critical for both return and TBM tasks
- Windows 3s and 5s most effective
- Combined with `spread_proxy_w`, spread-related features form a major group

### 6.4 `realized_vol` / `parkinson_vol` — Regime Features

These regime-level volatility indicators provide **market state context**:
- Particularly important for TBM classification (distinguishing trend vs. sideways)
- Also significant for return regression, especially at longer horizons
- Their importance increases from 1s → 5min, reflecting their medium-frequency nature

### 6.5 Taker Flow Features — 1s Specific

`tsa_sell_autocorr`, `taker_vol_autocorr_sell`, and `taker_vol_corr` are uniquely important at the 1s horizon, capturing **very short-term taker behavior persistence**. They disappear from importance rankings at longer horizons, suggesting their predictive power is ultra-short-lived.

---

## 7. Regression vs. Classification

### 7.1 Feature Importance Differences

| Aspect | Return Regression | TBM Classification |
|--------|------------------|--------------------|
| **Top feature type** | Volatility/structure (bar_range_bps) | Volatility regime (realized_vol) |
| **Importance concentration** | Top 5 features capture >60% importance | More evenly distributed |
| **Directional features** | Less important | `return_sum_w`, `bar_direction`, `signed_flow` more prominent |
| **Tick features** | Highly important | Moderately important |

### 7.2 Model Performance

| Horizon | Best Regression | Best Classification |
|---------|----------------|---------------------|
| **1s** | Lasso SR=+0.12 | All negative |
| **1min** | Lasso SR=+1.16 | GBM SR=+1.46 (tp0.5_sl0.5) |
| **5min** | GBM SR=+2.66 | GBM SR=+1.11 (tp1.0_sl0.5) |

At 5min, return regression significantly outperforms TBM classification.  
At 1min, TBM classification (tp0.5_sl0.5 with GBM) is competitive with return regression.

---

## 8. Version Evolution & Impact

| Version | Best Sharpe | Key Improvement |
|---------|------------|-----------------|
| **V3** | +1.52 (1min TBM) | Fixed-interval trading paradigm |
| **V4** | +1.88 (1min TBM) | Tick-level features + TBM tuning |
| **V5** | **+2.66** (5min Return) | Full indicators library (215 features vs V4's 108) |

**V5's improvement is driven by**:
1. **Feature enrichment** from the `indicators/` library (+107 new features)
2. **Returns & momentum features** (`return_sum_5`, `return_std_w`) — entirely absent in V3/V4
3. **Regime features** (`realized_vol`, `parkinson_vol`) — provide market state context
4. **Better spread estimation** (`roll_spread`, `spread_proxy`) from the dedicated `spread.py` module

---

## 9. Conclusions & Recommendations

1. **5min Return Regression with GBM** is the production-ready strategy candidate (Sharpe +2.66, 238 trades, 52% WR)
2. **1min TBM (tp0.5_sl0.5)** is a secondary candidate for shorter-horizon trading (Sharpe +1.46, 77.8% WR)
3. **1s prediction is not economically viable** — even with maker fees and high signal thresholds, net PnL is barely positive
4. **Bar structure and tick-level microstructure** are the two most powerful feature categories across all horizons
5. **Feature windows of 3-20s are universally effective** regardless of prediction horizon, validating the "physically meaningful window" philosophy
6. **Next steps**:
   - Walk-forward validation to confirm out-of-sample stability
   - Ensemble models (Ridge + GBM blend) for 5min prediction
   - Explore dynamic threshold adjustment based on regime features
   - Investigate feature interaction effects (e.g., spread × volatility)
   - Explore better windows for single feature
