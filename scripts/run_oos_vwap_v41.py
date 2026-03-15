#!/usr/bin/env python3
"""
Out-of-sample backtest for the notebook strategy vwap_dist_sum_v41.

Parameters from best in-sample result:
    fw1=8, ema_w1=10, fw2=10, ema_w2=10
    fill_rate=0.7

Replicates the notebook's logic inside the rule_backtest framework so the
OOS result is comparable to the full-backtest pipeline.

Uses:
  - merged trades + tpb=6000 bars (matching signal scan setup)
  - data from MR_DATA_ROOT (external drive)
  - date filter for OOS window (Mar 5 onwards)
"""
from __future__ import annotations

import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import timezone

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rule_backtest.data_loader import (
    load_cached_trades, merge_aggtrades, build_bars,
    estimate_avg_bar_seconds, _norm_imbalance,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# Strategy replication (from notebook vwap_dist_sum_v41)
# ═══════════════════════════════════════════════════════════

def calculate_delta_threshold(
    delta_series: pd.Series,
    window: int = 100,
    method: str = "mad",
    k: float = 1.0,
    vol_adjusted: bool = False,
    vol_series: pd.Series = None,
    local_window: int = 10,
) -> pd.Series:
    if method == "mad":
        rolling_median = delta_series.rolling(window).median()
        abs_dev = (delta_series - rolling_median).abs()
        rolling_mad = abs_dev.rolling(window).median()
        threshold = rolling_mad * k
    elif method == "std":
        rolling_std = delta_series.rolling(window).std()
        threshold = rolling_std * k
    else:  # quantile
        threshold = delta_series.rolling(window).quantile(0.75)

    if vol_adjusted and vol_series is not None:
        local_vol = vol_series.rolling(local_window).mean()
        avg_vol = local_vol.rolling(window).mean()
        threshold = threshold * np.sqrt(local_vol / (avg_vol + 1e-10))

    return threshold.ffill().bfill()


def compute_vwap_v41_factor(
    bars: pd.DataFrame,
    fw1: int = 8,
    ema_w1: int = 10,
    fw2: int = 10,
    ema_w2: int = 10,
) -> pd.DataFrame:
    """
    Compute all factor columns needed for the v41 strategy.
    Returns bars with additional columns appended.
    """
    b = bars.copy()

    # ── Factor 1: short window ──
    b[f"buy_vds_fw{fw1}"] = b["buy_vwap_dist_sum"].rolling(fw1).mean()
    b[f"sell_vds_fw{fw1}"] = b["sell_vwap_dist_sum"].rolling(fw1).mean()
    b[f"factor1_raw"] = _norm_imbalance(b[f"buy_vds_fw{fw1}"], b[f"sell_vds_fw{fw1}"])
    b["factor1"] = b["factor1_raw"].ewm(span=ema_w1, min_periods=1).mean()
    b["delta1"] = b["factor1"].diff()

    # ── Factor 2: long window ──
    b[f"buy_vds_fw{fw2}"] = b["buy_vwap_dist_sum"].rolling(fw2).mean()
    b[f"sell_vds_fw{fw2}"] = b["sell_vwap_dist_sum"].rolling(fw2).mean()
    b[f"factor2_raw"] = _norm_imbalance(b[f"buy_vds_fw{fw2}"], b[f"sell_vds_fw{fw2}"])
    b["factor2"] = b["factor2_raw"].ewm(span=ema_w2, min_periods=1).mean()
    b["delta2"] = b["factor2"].diff()

    # ── Adaptive thresholds ──
    b["thr1"] = calculate_delta_threshold(
        b["delta1"], window=100, method="mad", k=1.5,
        vol_adjusted=True, vol_series=b["volume"], local_window=10,
    )
    b["thr2"] = calculate_delta_threshold(
        b["delta2"], window=100, method="mad", k=1.0,
        vol_adjusted=True, vol_series=b["volume"], local_window=10,
    )

    # ── Sign series with forward-fill of 0 ──
    b["sign1"] = np.sign(b["delta1"]).replace(0, np.nan).ffill()
    b["sign2"] = np.sign(b["delta2"]).replace(0, np.nan).ffill()

    # ── Entry conditions ──
    b["reversal_neg"] = (b["sign2"].shift(1) == 1) & (b["sign2"] == -1)
    b["reversal_pos"] = (b["sign1"].shift(1) == -1) & (b["sign1"] == 1)
    b["filt1"] = b["delta1"].abs() > b["thr1"]
    b["filt2"] = b["delta2"].abs() > b["thr2"]

    b["short_entry"] = b["reversal_neg"] & b["filt2"]
    b["long_entry"] = b["reversal_pos"] & b["filt1"]

    return b


def run_backtest_v41(
    bars: pd.DataFrame,
    fw1: int = 8, ema_w1: int = 10,
    fw2: int = 10, ema_w2: int = 10,
    fill_rate: float = 0.7,
    maker_fee_bps: float = 2.0,
    taker_fee_bps: float = 5.0,
    label: str = "full",
) -> dict:
    fee_rate = (maker_fee_bps * fill_rate + taker_fee_bps * (1 - fill_rate)) * 1e-4

    b = compute_vwap_v41_factor(bars, fw1, ema_w1, fw2, ema_w2)

    # ── Position logic (hold-through) ──
    pos = np.full(len(b), np.nan)
    pos[b["short_entry"].values] = -1.0
    pos[b["long_entry"].values] = 1.0
    b["raw_pos"] = pos
    b["position"] = b["raw_pos"].ffill().fillna(0.0)

    # ── Bar returns and strategy returns ──
    b["bar_ret"] = (b["close"] - b["open"]) / b["open"]
    b["strat_ret"] = b["position"].shift(1) * b["bar_ret"]

    # ── Transaction costs ──
    b["pos_change"] = b["position"].diff().fillna(0)
    b["strat_ret"] -= fee_rate * b["pos_change"].abs()

    # ── Cumulative ──
    b["cum_ret"] = (1 + b["strat_ret"].fillna(0)).cumprod()
    b["bm_cum_ret"] = (1 + b["bar_ret"].fillna(0)).cumprod()

    # ── Metrics ──
    strat = b["strat_ret"].dropna()
    n = len(strat)
    n_bars_hist = len(b)

    # Approximate bar duration
    avg_sec = estimate_avg_bar_seconds(bars)
    periods_per_year = 365 * 24 * 3600 / avg_sec if avg_sec > 0 else 8760

    cum = b["cum_ret"].iloc[-1]
    total_ret_pct = (cum - 1) * 100
    ann_ret = (cum ** (periods_per_year / max(n, 1)) - 1) * 100 if cum > 0 else -100

    ret_std = strat.std()
    mean_ret = strat.mean()
    sharpe = mean_ret / (ret_std + 1e-12) * np.sqrt(periods_per_year)

    down = strat[strat < 0]
    down_std = down.std() if len(down) > 1 else 1e-12
    sortino = mean_ret / (down_std + 1e-12) * np.sqrt(periods_per_year)

    peak = b["cum_ret"].cummax()
    dd = (b["cum_ret"] - peak) / peak
    max_dd = dd.min() * 100

    n_signals = (b["pos_change"].abs() > 0).sum()
    n_long = (b["position"] > 0).sum()
    n_short = (b["position"] < 0).sum()
    long_ret = b.loc[b["position"] > 0, "strat_ret"].sum() * 10000
    short_ret = b.loc[b["position"] < 0, "strat_ret"].sum() * 10000
    total_cost = (b["pos_change"].abs() * fee_rate * 10000).sum()
    win_rate = (strat > 0).sum() / max(n, 1) * 100

    calmar = ann_ret / (abs(max_dd) + 1e-12)

    return {
        "label": label,
        "n_bars": n_bars_hist,
        "bar_duration_sec": round(avg_sec),
        "n_signals": int(n_signals),
        "n_long_bars": int(n_long),
        "n_short_bars": int(n_short),
        "total_return_pct": round(total_ret_pct, 4),
        "annualized_return_pct": round(ann_ret, 4),
        "max_drawdown_pct": round(max_dd, 4),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "calmar_ratio": round(calmar, 4),
        "win_rate_pct": round(win_rate, 2),
        "long_pnl_bps": round(long_ret, 2),
        "short_pnl_bps": round(short_ret, 2),
        "total_cost_bps": round(total_cost, 2),
        "net_pnl_bps": round(strat.sum() * 10000, 2),
    }, b


def print_metrics(m: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  {m['label']}")
    print(f"{'='*60}")
    print(f"  Bars:               {m['n_bars']} (~{m['bar_duration_sec']}s avg)")
    print(f"  Signals (trades):   {m['n_signals']}")
    print(f"  Long / Short bars:  {m['n_long_bars']} / {m['n_short_bars']}")
    print(f"  Net PnL (bps):      {m['net_pnl_bps']:.2f}")
    print(f"  Long PnL (bps):     {m['long_pnl_bps']:.2f}")
    print(f"  Short PnL (bps):    {m['short_pnl_bps']:.2f}")
    print(f"  Cost (bps):         {m['total_cost_bps']:.2f}")
    print(f"  Total Return:       {m['total_return_pct']:.4f}%")
    print(f"  Ann. Return:        {m['annualized_return_pct']:.4f}%")
    print(f"  Max Drawdown:       {m['max_drawdown_pct']:.4f}%")
    print(f"  Sharpe:             {m['sharpe_ratio']:.4f}")
    print(f"  Sortino:            {m['sortino_ratio']:.4f}")
    print(f"  Calmar:             {m['calmar_ratio']:.4f}")
    print(f"  Win Rate:           {m['win_rate_pct']:.2f}%")
    print(f"{'='*60}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="SOL/USDC")
    p.add_argument("--tpb", type=int, default=6000, help="Trades per bar")
    p.add_argument("--fw1", type=int, default=8)
    p.add_argument("--ema_w1", type=int, default=10)
    p.add_argument("--fw2", type=int, default=10)
    p.add_argument("--ema_w2", type=int, default=10)
    p.add_argument("--fill_rate", type=float, default=0.7)
    p.add_argument("--oos_start", default="2026-03-05",
                   help="OOS start date (YYYY-MM-DD); IS = everything before")
    p.add_argument("--output_dir", default=None)
    args = p.parse_args()

    print(f"\n{'='*60}")
    print(f"OOS Backtest: vwap_dist_sum_v41")
    print(f"  fw1={args.fw1} ema1={args.ema_w1} fw2={args.fw2} ema2={args.ema_w2}")
    print(f"  tpb={args.tpb}  fill_rate={args.fill_rate}")
    print(f"  OOS start: {args.oos_start}")
    print(f"{'='*60}\n")

    # Load all trades
    raw = load_cached_trades(args.symbol)
    logger.info("Loaded %d raw trades", len(raw))
    merged = merge_aggtrades(raw)
    logger.info("Merged → %d taker orders", len(merged))

    bars = build_bars(merged, "merged", "trade_count", trades_per_bar=args.tpb)
    logger.info("Built %d trade-count bars (tpb=%d)", len(bars), args.tpb)

    # Verify required columns exist
    required = ["buy_vwap_dist_sum", "sell_vwap_dist_sum", "volume", "open", "high", "low", "close"]
    missing = [c for c in required if c not in bars.columns]
    if missing:
        raise RuntimeError(f"Missing bar columns: {missing}. Ensure merged+trade_count bars are used.")

    # Split IS / OOS — normalize tz-awareness to match bar index
    oos_start_raw = pd.Timestamp(args.oos_start, tz="UTC")
    if bars.index.tz is None:
        oos_start = oos_start_raw.tz_localize(None)
    else:
        oos_start = oos_start_raw
    is_bars = bars[bars.index < oos_start]
    oos_bars = bars[bars.index >= oos_start]

    logger.info("IS bars: %d | OOS bars: %d", len(is_bars), len(oos_bars))

    kw = dict(fw1=args.fw1, ema_w1=args.ema_w1, fw2=args.fw2, ema_w2=args.ema_w2,
              fill_rate=args.fill_rate)

    # ── IS backtest (sanity check) ──
    if len(is_bars) >= 200:
        is_m, is_b = run_backtest_v41(is_bars, **kw, label="IN-SAMPLE (Feb 3 → Mar 4)")
        print_metrics(is_m)
    else:
        logger.warning("Too few IS bars: %d", len(is_bars))
        is_m = None

    # ── OOS backtest ──
    # Include IS bars as context so MAD thresholds are properly initialised,
    # then evaluate only the OOS slice.
    if len(oos_bars) < 20:
        print(f"\n[!] Only {len(oos_bars)} OOS bars available — data may still be fetching.")
        print("    Re-run after fetch completes.")
        return

    context_bars = is_bars.tail(150)   # 150 IS bars for threshold warmup
    full_oos_context = pd.concat([context_bars, oos_bars])
    _, full_b = run_backtest_v41(full_oos_context, **kw, label="_context_")
    # Slice out only the OOS rows from the computed DataFrame
    oos_only = full_b.iloc[len(context_bars):]

    # Re-compute metrics on the OOS-only slice
    fee_rate = (args.fill_rate * 2.0 + (1 - args.fill_rate) * 5.0) * 1e-4
    avg_sec_oos = estimate_avg_bar_seconds(oos_bars)
    periods_per_year_oos = 365 * 24 * 3600 / avg_sec_oos if avg_sec_oos > 0 else 8760

    oos_strat = oos_only["strat_ret"].dropna()
    cum = oos_only["cum_ret"].iloc[-1] if len(oos_only) else 1.0
    n = max(len(oos_strat), 1)
    mean_r = oos_strat.mean()
    std_r = oos_strat.std()
    sharpe = mean_r / (std_r + 1e-12) * np.sqrt(periods_per_year_oos)
    down_r = oos_strat[oos_strat < 0].std()
    sortino = mean_r / (down_r + 1e-12) * np.sqrt(periods_per_year_oos)
    ann_ret = ((cum ** (periods_per_year_oos / n)) - 1) * 100 if cum > 0 else -100
    peak = oos_only["cum_ret"].cummax()
    dd = (oos_only["cum_ret"] - peak) / peak
    max_dd = dd.min() * 100
    n_sig = int((oos_only["pos_change"].abs() > 0).sum())
    oos_m = {
        "label": f"OUT-OF-SAMPLE ({args.oos_start} → today)",
        "n_bars": len(oos_only),
        "bar_duration_sec": round(avg_sec_oos),
        "n_signals": n_sig,
        "n_long_bars": int((oos_only["position"] > 0).sum()),
        "n_short_bars": int((oos_only["position"] < 0).sum()),
        "total_return_pct": round((cum - 1) * 100, 4),
        "annualized_return_pct": round(ann_ret, 4),
        "max_drawdown_pct": round(max_dd, 4),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "calmar_ratio": round(ann_ret / (abs(max_dd) + 1e-12), 4),
        "win_rate_pct": round((oos_strat > 0).sum() / n * 100, 2),
        "long_pnl_bps": round(oos_only.loc[oos_only["position"] > 0, "strat_ret"].sum() * 10000, 2),
        "short_pnl_bps": round(oos_only.loc[oos_only["position"] < 0, "strat_ret"].sum() * 10000, 2),
        "total_cost_bps": round((oos_only["pos_change"].abs() * fee_rate * 10000).sum(), 2),
        "net_pnl_bps": round(oos_strat.sum() * 10000, 2),
    }
    oos_b = oos_only
    print_metrics(oos_m)

    # ── IS vs OOS comparison ──
    if is_m:
        print(f"\n{'='*60}")
        print("IS vs OOS comparison")
        print(f"{'='*60}")
        for key in ["sharpe_ratio", "total_return_pct", "max_drawdown_pct",
                    "win_rate_pct", "net_pnl_bps", "total_cost_bps"]:
            iv = is_m.get(key, 0)
            ov = oos_m.get(key, 0)
            ratio = ov / iv if abs(iv) > 1e-6 else float("nan")
            flag = "✓" if abs(ratio - 1) < 0.5 else "⚠"
            print(f"  {flag} {key:30s}: IS={iv:>10.3f}  OOS={ov:>10.3f}  ratio={ratio:.2f}")

    # ── Save ──
    out_root = Path(args.output_dir) if args.output_dir else (
        Path(os.environ.get("MR_DATA_ROOT", str(ROOT / "data")))
        / "oos_results" / args.symbol.replace("/", "_")
    )
    out_root.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame([m for m in [is_m, oos_m] if m])
    results.to_csv(out_root / "vwap_v41_oos.csv", index=False)
    logger.info("Saved → %s", out_root / "vwap_v41_oos.csv")

    # Save OOS trades detail
    oos_b_out = oos_b[["open", "close", "volume", "factor1", "factor2",
                         "position", "strat_ret", "cum_ret"]].copy()
    oos_b_out.to_csv(out_root / "vwap_v41_oos_bars.csv")
    logger.info("OOS bars → %s", out_root / "vwap_v41_oos_bars.csv")

    print(f"\nResults saved to: {out_root}")


if __name__ == "__main__":
    main()
