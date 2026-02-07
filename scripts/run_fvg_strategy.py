#!/usr/bin/env python3
"""
FVG 策略（基础版）：使用 Binance aggTrades 数据。

运行：
    .venv/bin/python scripts/run_fvg_strategy.py --days 1 --symbol SOL/USDT
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from utilities.binance_loader import load_agg_trades, resample_trades_to_ohlcv
from utilities.paths import DataPaths
from cta.fvg.detector import detect_fvg, FVGType, FVGStatus
from cta.fvg.features import build_fvg_feature_matrix, get_feature_columns
from cta.fvg.model import FVGModel
from cta.backtest import backtest_fvg


def analyze_fvg_microstructure(
    fvgs: List, trades: pd.DataFrame, window_ms: int = 60000,
) -> Dict[str, Any]:
    """分析 FVG 事件前后的高频指标时序变化。"""
    filled = [f for f in fvgs if f.status == FVGStatus.FILLED]
    unfilled = [f for f in fvgs if f.status in (FVGStatus.OPEN, FVGStatus.EXPIRED)]

    def _agg(fvg_list, label):
        if not fvg_list:
            return {}
        imbs, vols, imps = [], [], []
        for fvg in fvg_list:
            ts = fvg.timestamp
            w = trades[(trades["timestamp"] >= ts - pd.Timedelta(milliseconds=window_ms))
                       & (trades["timestamp"] < ts)]
            if w.empty:
                continue
            bv = w.loc[w["side"] == "buy", "amount"].sum()
            sv = w.loc[w["side"] == "sell", "amount"].sum()
            tv = bv + sv
            if tv > 0:
                imbs.append((bv - sv) / tv)
                vols.append(tv)
            p = w["price"]
            if len(p) > 1:
                imps.append(abs(p.iloc[-1] - p.iloc[0]) / (p.iloc[0] + 1e-10) * 100 / (tv + 1e-10))
        return {
            f"{label}_count": len(fvg_list),
            f"{label}_avg_imbalance": np.mean(imbs) if imbs else 0,
            f"{label}_avg_volume": np.mean(vols) if vols else 0,
            f"{label}_avg_impact": np.mean(imps) if imps else 0,
            f"{label}_avg_gap_pct": np.mean([f.gap_size_pct for f in fvg_list]),
        }

    analysis = {
        "total": len(fvgs),
        "bullish": sum(1 for f in fvgs if f.fvg_type == FVGType.BULLISH),
        "bearish": sum(1 for f in fvgs if f.fvg_type == FVGType.BEARISH),
        "filled": len(filled),
        "fill_rate": round(len(filled) / max(len(fvgs), 1) * 100, 2),
    }
    analysis.update(_agg(filled, "filled"))
    analysis.update(_agg(unfilled, "unfilled"))
    return analysis


def main():
    parser = argparse.ArgumentParser(description="FVG Strategy (Binance aggTrades)")
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--symbol", default="SOL/USDT")
    parser.add_argument("--timeframe", default="1min")
    parser.add_argument("--min-gap", type=float, default=0.02)
    args = parser.parse_args()

    print("=" * 70)
    print("FVG STRATEGY (Binance aggTrades)")
    print("=" * 70)
    print(f"Symbol: {args.symbol}, Days: {args.days}")

    # 1. 加载 aggTrades
    print("\n[1] Loading Binance aggTrades...")
    trades = load_agg_trades(symbol=args.symbol, days=args.days)
    if trades.empty:
        print("ERROR: No trades!"); return 1
    print(f"  {len(trades):,} aggTrades")
    if "n_trades_in_agg" in trades.columns:
        print(f"  Taker agg info: mean={trades['n_trades_in_agg'].mean():.1f} trades/agg")

    # 2. Resample
    print("\n[2] Resampling to OHLCV...")
    timeframes = ["1min", "3min", "5min"]
    ohlcv_dict = {}
    for tf in timeframes:
        ohlcv_dict[tf] = resample_trades_to_ohlcv(trades, freq=tf)
        print(f"  {tf}: {len(ohlcv_dict[tf])} bars")

    # 3. 检测
    print("\n[3] Detecting FVGs...")
    for tf, ohlcv in ohlcv_dict.items():
        fvgs = detect_fvg(ohlcv, min_gap_pct=args.min_gap)
        filled = sum(1 for f in fvgs if f.status == FVGStatus.FILLED)
        print(f"  {tf}: {len(fvgs)} FVGs ({filled} filled)")

    ohlcv = ohlcv_dict[args.timeframe]
    fvgs = detect_fvg(ohlcv, min_gap_pct=args.min_gap)
    print(f"\n  Using {args.timeframe}, gap>{args.min_gap}%: {len(fvgs)} FVGs")

    # 4. 分析
    print("\n[4] Microstructure analysis...")
    analysis = analyze_fvg_microstructure(fvgs, trades)
    print(f"  Fill rate: {analysis['fill_rate']}%")

    # 5. 特征 + 训练
    print("\n[5] Building features & training...")
    feat_df = build_fvg_feature_matrix(fvgs, trades, ohlcv)
    if len(feat_df) < 20:
        print(f"  Too few FVGs ({len(feat_df)})"); return 1
    print(f"  Features: {feat_df.shape}")

    model = FVGModel(n_clusters=4)
    report = model.fit(feat_df, target_col="label_filled")
    if "error" in report:
        print(f"  Error: {report['error']}"); return 1

    print(f"  Train accuracy: {report['train_accuracy']}%, CV: {report['cv_accuracy']}%")

    # 6. 回测
    print("\n[6] Backtesting...")
    split_idx = int(len(feat_df) * 0.7)
    test_df = feat_df.iloc[split_idx:]
    if len(test_df) >= 5:
        model_oos = FVGModel(n_clusters=4)
        model_oos.fit(feat_df.iloc[:split_idx], target_col="label_filled")
        model = model_oos

    bt = backtest_fvg(test_df, model, ohlcv, hold_bars=10, min_proba=0.6)
    print(f"\n  Signals: {bt['n_signals']}, Win rate: {bt['win_rate']}%")
    print(f"  Return: {bt['total_return']}%, Sharpe: {bt.get('sharpe', 0)}")

    # 保存结果
    save_path = DataPaths.backtest_result("fvg_basic", "latest")
    if "results_df" in bt and not bt["results_df"].empty:
        bt["results_df"].to_csv(save_path, index=False)
        print(f"\n  Saved: {save_path}")

    return 0


if __name__ == "__main__":
    exit(main() or 0)
