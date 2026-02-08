#!/usr/bin/env python3
"""
收益率自相关分析：BTC vs SOL（低内存版本）
- 逐天加载 → resample → 释放原始 trades
- 多频率（1s ~ 1h）下的 lag-1~50 ACF
- 日级稳定性检验（每日 lag-1 ACF 热力图）
- 简单 contrarian PnL 模拟
"""
import sys, os, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import date, timedelta
from utilities.binance_loader import load_agg_trades, resample_trades_to_ohlcv

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (16, 6)
plt.rcParams['font.size'] = 11

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'notebooks')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Config ──
btc_dates = [
    '2022-09-05','2022-09-06','2022-09-07','2022-09-08',
    '2022-10-01','2022-10-02','2022-10-03','2022-10-04','2022-10-05',
    '2022-10-06','2022-10-07','2022-10-08','2022-10-09','2022-10-10',
    '2022-10-11','2022-10-13','2022-10-14','2022-10-15','2022-10-16',
]
sol_dates = [
    '2026-01-28','2026-01-29','2026-01-30','2026-01-31',
    '2026-02-01','2026-02-02','2026-02-03','2026-02-04',
    '2026-02-05','2026-02-06',
]
freqs = ['1s', '5s', '10s', '30s', '1min', '5min', '15min', '30min', '1h']
check_freqs = ['1s', '5s', '10s', '30s', '1min', '5min']

# ── 1. 逐天 resample，只保留 OHLCV 聚合数据 ──
print('=' * 60)
print('STEP 1: Loading & Resampling (day-by-day, memory efficient)')
print('=' * 60)

def load_and_resample(symbol, dates, freqs):
    """逐天加载 trades → resample 为 OHLCV → 释放 trades 内存。"""
    ohlcv_by_freq = {f: [] for f in freqs}
    for d in dates:
        dt = date.fromisoformat(d)
        trades = load_agg_trades(symbol, dt, dt + timedelta(days=1))
        if trades.empty:
            print(f"  {d}: [empty]")
            continue
        n = len(trades)
        for f in freqs:
            bars = resample_trades_to_ohlcv(trades, f)
            ohlcv_by_freq[f].append(bars)
        del trades
        gc.collect()
        print(f"  {d}: {n:>10,} trades → resampled")

    result = {}
    for f in freqs:
        if ohlcv_by_freq[f]:
            result[f] = pd.concat(ohlcv_by_freq[f]).sort_index()
            result[f] = result[f][~result[f].index.duplicated(keep='last')]
        else:
            result[f] = pd.DataFrame()
    return result

print('\nBTC/USDT:')
btc_ohlcv = load_and_resample('BTC/USDT', btc_dates, freqs)
gc.collect()

print('\nSOL/USDT:')
sol_ohlcv = load_and_resample('SOL/USDT', sol_dates, freqs)
gc.collect()

print('\nBar counts:')
for f in freqs:
    print(f'{f:>5}: BTC {len(btc_ohlcv[f]):>7,} bars  |  SOL {len(sol_ohlcv[f]):>7,} bars')

# ── 2. Compute ACF ──
print('\n' + '=' * 60)
print('STEP 2: Computing Return Autocorrelation')
print('=' * 60)

def compute_autocorr(ohlcv_dict, max_lag=50):
    results = {}
    for freq, ohlcv in ohlcv_dict.items():
        if ohlcv.empty:
            results[freq] = {'acf': [0]*max_lag, 'lags': list(range(1,max_lag+1)), 'n': 0, 'ret_std': 0, 'ret_mean': 0}
            continue
        close = ohlcv['close'].astype(float)
        ret = np.log(close / close.shift(1)).dropna()
        # 替换 inf
        ret = ret.replace([np.inf, -np.inf], np.nan).dropna()
        if hasattr(ret.index, 'date'):
            daily_mean = ret.groupby(ret.index.date).transform('mean')
            ret_demean = ret - daily_mean
        else:
            ret_demean = ret - ret.mean()
        actual_max_lag = min(max_lag, len(ret_demean) // 10)
        if actual_max_lag < 1:
            actual_max_lag = 1
        acf = [ret_demean.autocorr(lag=k) for k in range(1, actual_max_lag + 1)]
        results[freq] = {
            'acf': acf,
            'lags': list(range(1, actual_max_lag + 1)),
            'n': len(ret_demean),
            'ret_std': float(ret.std()),
            'ret_mean': float(ret.mean()),
        }
        print(f'  {freq:>5}: N={len(ret_demean):>8,}  lag-1 ACF = {acf[0]:+.5f}')
    return results

print('\nBTC:')
btc_acf = compute_autocorr(btc_ohlcv, max_lag=50)
print('\nSOL:')
sol_acf = compute_autocorr(sol_ohlcv, max_lag=50)

print('\nReturn stats (log return):')
print(f'{"Freq":>5} | {"BTC mean":>10} {"BTC std":>10} {"BTC N":>8} | {"SOL mean":>10} {"SOL std":>10} {"SOL N":>8}')
print('-' * 80)
for f in freqs:
    b = btc_acf[f]
    s = sol_acf[f]
    print(f'{f:>5} | {b["ret_mean"]:>10.6f} {b["ret_std"]:>10.6f} {b["n"]:>8,} | {s["ret_mean"]:>10.6f} {s["ret_std"]:>10.6f} {s["n"]:>8,}')

# ── 3. Plot Lag-1 ACF by Frequency ──
print('\n' + '=' * 60)
print('STEP 3: Lag-1 Autocorrelation Plot')
print('=' * 60)

fig, ax = plt.subplots(figsize=(14, 6))
btc_lag1 = [btc_acf[f]['acf'][0] if btc_acf[f]['acf'] else 0 for f in freqs]
sol_lag1 = [sol_acf[f]['acf'][0] if sol_acf[f]['acf'] else 0 for f in freqs]

x = np.arange(len(freqs))
w = 0.35
ax.bar(x - w/2, btc_lag1, w, label='BTC/USDT', color='#FF9800', alpha=0.85)
ax.bar(x + w/2, sol_lag1, w, label='SOL/USDT', color='#2196F3', alpha=0.85)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(freqs, fontsize=12)
ax.set_ylabel('Lag-1 Autocorrelation', fontsize=13)
ax.set_xlabel('Bar Frequency', fontsize=13)
ax.set_title('Return Lag-1 Autocorrelation by Frequency\n(Negative = Mean Reversion, Positive = Momentum)', fontsize=14)
ax.legend(fontsize=12)

for f_idx, f in enumerate(freqs):
    n_b = btc_acf[f]['n']
    n_s = sol_acf[f]['n']
    if n_b > 0:
        ci_b = 2 / np.sqrt(n_b)
        if abs(btc_lag1[f_idx]) > ci_b:
            ax.annotate('*', (f_idx - w/2, btc_lag1[f_idx]),
                         ha='center', va='bottom' if btc_lag1[f_idx] > 0 else 'top',
                         fontsize=16, fontweight='bold', color='red')
    if n_s > 0:
        ci_s = 2 / np.sqrt(n_s)
        if abs(sol_lag1[f_idx]) > ci_s:
            ax.annotate('*', (f_idx + w/2, sol_lag1[f_idx]),
                         ha='center', va='bottom' if sol_lag1[f_idx] > 0 else 'top',
                         fontsize=16, fontweight='bold', color='blue')

ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'lag1_autocorr_by_freq.png'), dpi=150)
plt.close()
print('Saved: lag1_autocorr_by_freq.png')

print('\nLag-1 Autocorrelation (* = significant at 95% CI):')
print(f'{"Freq":>5} | {"BTC":>8} {"sig?":>5} | {"SOL":>8} {"sig?":>5}')
print('-' * 42)
for i, f in enumerate(freqs):
    n_b = btc_acf[f]['n']
    n_s = sol_acf[f]['n']
    ci_b = 2 / np.sqrt(n_b) if n_b > 0 else 999
    ci_s = 2 / np.sqrt(n_s) if n_s > 0 else 999
    sig_b = '***' if abs(btc_lag1[i]) > ci_b else ''
    sig_s = '***' if abs(sol_lag1[i]) > ci_s else ''
    print(f'{f:>5} | {btc_lag1[i]:>+8.4f} {sig_b:>5} | {sol_lag1[i]:>+8.4f} {sig_s:>5}')

# ── 4. Full ACF Plots ──
print('\n' + '=' * 60)
print('STEP 4: Full ACF Plots (Lag 1~30)')
print('=' * 60)

key_freqs = ['1s', '5s', '10s', '30s', '1min', '5min']
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Return ACF: BTC (orange) vs SOL (blue)', fontsize=15, y=1.02)

for idx, f in enumerate(key_freqs):
    ax = axes[idx // 3, idx % 3]
    b_acf = btc_acf[f]['acf']
    s_acf = sol_acf[f]['acf']
    b_lags = btc_acf[f]['lags']
    s_lags = sol_acf[f]['lags']
    n_b = btc_acf[f]['n']
    n_s = sol_acf[f]['n']

    plot_len = min(30, len(b_acf), len(s_acf))
    ax.bar(np.array(b_lags[:plot_len]) - 0.2, b_acf[:plot_len], width=0.4, color='#FF9800', alpha=0.7, label='BTC')
    ax.bar(np.array(s_lags[:plot_len]) + 0.2, s_acf[:plot_len], width=0.4, color='#2196F3', alpha=0.7, label='SOL')
    ax.axhline(y=0, color='black', linewidth=0.5)
    if n_b > 0:
        ci_b = 2 / np.sqrt(n_b)
        ax.axhline(y=ci_b, color='#FF9800', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.axhline(y=-ci_b, color='#FF9800', linewidth=0.8, linestyle='--', alpha=0.5)
    if n_s > 0:
        ci_s = 2 / np.sqrt(n_s)
        ax.axhline(y=ci_s, color='#2196F3', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.axhline(y=-ci_s, color='#2196F3', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_title(f'{f} bars', fontsize=13, fontweight='bold')
    ax.set_xlabel('Lag')
    ax.set_ylabel('ACF')
    if idx == 0:
        ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'acf_by_freq.png'), dpi=150)
plt.close()
print('Saved: acf_by_freq.png')

# ── 5. Daily Lag-1 ACF Heatmap ──
print('\n' + '=' * 60)
print('STEP 5: Daily Lag-1 ACF Heatmap (Stability)')
print('=' * 60)

def daily_lag1_acf(symbol_name, dates, freqs_subset):
    """逐天加载并计算 lag-1 ACF，内存友好。"""
    results = []
    for d in dates:
        dt = date.fromisoformat(d)
        t = load_agg_trades(symbol_name, dt, dt + timedelta(days=1))
        if t.empty:
            continue
        for f in freqs_subset:
            ohlcv = resample_trades_to_ohlcv(t, f)
            close = ohlcv['close'].astype(float)
            ret = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
            ret = ret - ret.mean()
            if len(ret) > 20:
                acf1 = ret.autocorr(lag=1)
                results.append({'date': d, 'freq': f, 'acf1': acf1, 'n': len(ret)})
        del t
        gc.collect()
        print(f'  {d}: done')
    return pd.DataFrame(results)

print('Computing daily lag-1 ACF for BTC...')
btc_daily = daily_lag1_acf('BTC/USDT', btc_dates, check_freqs)
print('Computing daily lag-1 ACF for SOL...')
sol_daily = daily_lag1_acf('SOL/USDT', sol_dates, check_freqs)

btc_pivot = btc_daily.pivot(index='date', columns='freq', values='acf1')[check_freqs]
sol_pivot = sol_daily.pivot(index='date', columns='freq', values='acf1')[check_freqs]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, max(8, len(btc_dates) * 0.4)))

im1 = ax1.imshow(btc_pivot.values, cmap='RdBu_r', aspect='auto', vmin=-0.15, vmax=0.15)
ax1.set_xticks(range(len(check_freqs)))
ax1.set_xticklabels(check_freqs)
ax1.set_yticks(range(len(btc_pivot.index)))
ax1.set_yticklabels(btc_pivot.index, fontsize=8)
ax1.set_title('BTC/USDT - Daily Lag-1 ACF (19 days)', fontsize=13)
for i in range(btc_pivot.shape[0]):
    for j in range(btc_pivot.shape[1]):
        v = btc_pivot.values[i, j]
        if not np.isnan(v):
            ax1.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=7)

im2 = ax2.imshow(sol_pivot.values, cmap='RdBu_r', aspect='auto', vmin=-0.15, vmax=0.15)
ax2.set_xticks(range(len(check_freqs)))
ax2.set_xticklabels(check_freqs)
ax2.set_yticks(range(len(sol_pivot.index)))
ax2.set_yticklabels(sol_pivot.index, fontsize=8)
ax2.set_title('SOL/USDT - Daily Lag-1 ACF (10 days)', fontsize=13)
for i in range(sol_pivot.shape[0]):
    for j in range(sol_pivot.shape[1]):
        v = sol_pivot.values[i, j]
        if not np.isnan(v):
            ax2.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=7)

fig.colorbar(im1, ax=[ax1, ax2], label='Lag-1 ACF', shrink=0.8)
plt.suptitle('Daily Lag-1 Autocorrelation Stability\n(Red=Momentum, Blue=Mean Reversion)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'daily_acf_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved: daily_acf_heatmap.png')

# ── 6. Summary Statistics ──
print('\n' + '=' * 60)
print('STEP 6: Summary Statistics (t-test on daily ACF)')
print('=' * 60)

print(f'\n{"Freq":>5} | {"BTC mean":>9} {"BTC std":>9} {"BTC t":>7} {"":>3} | {"SOL mean":>9} {"SOL std":>9} {"SOL t":>7} {"":>3}')
print('-' * 78)
for f in check_freqs:
    btc_vals = btc_daily[btc_daily['freq'] == f]['acf1'].dropna()
    sol_vals = sol_daily[sol_daily['freq'] == f]['acf1'].dropna()

    b_mean, b_std = btc_vals.mean(), btc_vals.std()
    b_t = b_mean / (b_std / np.sqrt(len(btc_vals)) + 1e-10) if len(btc_vals) > 1 else 0

    s_mean, s_std = sol_vals.mean(), sol_vals.std()
    s_t = s_mean / (s_std / np.sqrt(len(sol_vals)) + 1e-10) if len(sol_vals) > 1 else 0

    b_sig = '***' if abs(b_t) > 2.5 else '**' if abs(b_t) > 2 else '*' if abs(b_t) > 1.5 else ''
    s_sig = '***' if abs(s_t) > 2.5 else '**' if abs(s_t) > 2 else '*' if abs(s_t) > 1.5 else ''

    print(f'{f:>5} | {b_mean:>+9.4f} {b_std:>9.4f} {b_t:>+7.2f} {b_sig:>3} | {s_mean:>+9.4f} {s_std:>9.4f} {s_t:>+7.2f} {s_sig:>3}')

print('\n  * p<0.10  ** p<0.05  *** p<0.01')
print('  Negative = Mean Reversion, Positive = Momentum')

# ── 7. ACF Decay ──
print('\n' + '=' * 60)
print('STEP 7: ACF Decay Plot')
print('=' * 60)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, (name, acf_dict) in zip(axes, [('BTC/USDT', btc_acf), ('SOL/USDT', sol_acf)]):
    for f in check_freqs:
        acf = acf_dict[f]['acf']
        lags = acf_dict[f]['lags']
        plot_n = min(30, len(acf))
        ax.plot(lags[:plot_n], acf[:plot_n], marker='.', markersize=4, label=f, alpha=0.8)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Lag', fontsize=12)
    ax.set_ylabel('ACF', fontsize=12)
    ax.set_title(f'{name} - ACF Decay by Frequency', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'acf_decay.png'), dpi=150)
plt.close()
print('Saved: acf_decay.png')

# ── 8. Contrarian PnL ──
print('\n' + '=' * 60)
print('STEP 8: Naive Contrarian Strategy PnL')
print('=' * 60)

def simple_contrarian_pnl(ohlcv, cost_bps=4):
    close = ohlcv['close'].astype(float)
    ret = close.pct_change().dropna() * 10000  # bps
    signal = -np.sign(ret.shift(1))
    gross_pnl = signal * ret
    net_pnl = gross_pnl - cost_bps * 2
    return pd.DataFrame({
        'gross_pnl': gross_pnl, 'net_pnl': net_pnl,
        'gross_cum': gross_pnl.cumsum(), 'net_cum': net_pnl.cumsum(),
    }).dropna()

test_freqs = ['1s', '5s', '10s', '30s', '1min', '5min']
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Naive Contrarian Strategy\nGross (dashed) vs Net of 8bps RT cost (solid)', fontsize=14, y=1.02)

for idx, f in enumerate(test_freqs):
    ax = axes[idx // 3, idx % 3]
    bp = simple_contrarian_pnl(btc_ohlcv[f], cost_bps=4)
    sp = simple_contrarian_pnl(sol_ohlcv[f], cost_bps=4)
    ax.plot(bp['gross_cum'].values, '--', color='#FF9800', alpha=0.5, label='BTC gross')
    ax.plot(bp['net_cum'].values, '-', color='#FF9800', alpha=0.9, label='BTC net')
    ax.plot(sp['gross_cum'].values, '--', color='#2196F3', alpha=0.5, label='SOL gross')
    ax.plot(sp['net_cum'].values, '-', color='#2196F3', alpha=0.9, label='SOL net')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_title(f'{f} bars', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cum PnL (bps)')
    if idx == 0:
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'contrarian_pnl.png'), dpi=150)
plt.close()
print('Saved: contrarian_pnl.png')

print(f'\nContrarian Strategy Summary (net of 8bps RT cost):')
print(f'{"Freq":>5} | {"BTC gross":>10} {"BTC net":>10} {"BTC SR":>8} | {"SOL gross":>10} {"SOL net":>10} {"SOL SR":>8}')
print('-' * 75)
for f in test_freqs:
    bp = simple_contrarian_pnl(btc_ohlcv[f], cost_bps=4)
    sp = simple_contrarian_pnl(sol_ohlcv[f], cost_bps=4)
    b_sr = bp['net_pnl'].mean() / (bp['net_pnl'].std() + 1e-10) * np.sqrt(len(bp))
    s_sr = sp['net_pnl'].mean() / (sp['net_pnl'].std() + 1e-10) * np.sqrt(len(sp))
    print(f'{f:>5} | {bp["gross_cum"].iloc[-1]:>+10.1f} {bp["net_cum"].iloc[-1]:>+10.1f} {b_sr:>+8.2f} | {sp["gross_cum"].iloc[-1]:>+10.1f} {sp["net_cum"].iloc[-1]:>+10.1f} {s_sr:>+8.2f}')

# ── 9. Conclusion ──
print('\n' + '=' * 60)
print('STEP 9: FINDINGS & RECOMMENDATION')
print('=' * 60)
print()
print('Lag-1 ACF Pattern:')
for f in check_freqs:
    b = btc_acf[f]['acf'][0] if btc_acf[f]['acf'] else 0
    s = sol_acf[f]['acf'][0] if sol_acf[f]['acf'] else 0
    b_type = 'MOMENTUM' if b > 0.01 else 'REVERSION' if b < -0.01 else 'RANDOM'
    s_type = 'MOMENTUM' if s > 0.01 else 'REVERSION' if s < -0.01 else 'RANDOM'
    print(f'   {f:>5}: BTC={b:+.4f} ({b_type:>9})  SOL={s:+.4f} ({s_type:>9})')

print()
print('Recommendation:')
print('  - Frequencies with strong negative ACF → mean reversion target')
print('  - Frequencies with strong positive ACF → momentum target')
print('  - Choose the frequency where |ACF| is largest AND stable across days')
print()
print('All plots saved to notebooks/ directory.')
