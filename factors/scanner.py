"""
Factor combination scanner and backtest runner.

Generates 1st-order and 2nd-order factor expressions, evaluates them,
and ranks by IC / decile spread / stability.

1st-order: operator(primitive, lookback)
2nd-order: operator1(operator2(primitive, lookback2), lookback1)

All factors — including microstructure-themed ones — are expressed as
primitive + operator combinations. There are no hardcoded indicator
functions; everything flows through the expression engine.

Microstructure-motivated 2nd-order templates:
  - ts_acceleration(price_impact, d):         PI acceleration (MM book thinning)
  - ts_zscore(ts_acceleration(price_impact, d1), d2): normalized PI accel
  - delta(small_vol_pi_ratio, d):             change in small-order fragility
  - ts_corr(delta(price_range, 1), volume, d): spread-volume feedback
  - ts_entropy(volume, d):                    volume distribution regime
  - ts_corr(abs_imbalance, price_impact, d):  toxicity-PI feedback loop
  - delta(ts_mean(price_impact, d1), d2) - delta(ts_mean(volume, d1), d2):
      depth thinning (PI rises while volume drops) — via scanner composite

General 2nd-order templates (WorldQuant Brain-inspired):
  - ts_rank(delta(x, d1), d2):       rank of momentum
  - decay_linear(delta(x, d1), d2):  decayed momentum
  - delta(ts_mean(x, d1), d2):       mean-reversion signal
  - ts_stddev(delta(x, d1), d2):     volatility of changes
  - ts_rank(ts_stddev(x, d1), d2):   regime volatility rank
  - ts_zscore(delta(x, d1), d2):     normalized momentum
  - ts_entropy(delta(x, d1), d2):    entropy of changes
  - ts_skew(delta(x, d1), d2):       asymmetry of momentum
  - signedpower(ts_rank(x, d), a):   nonlinear rank emphasis
  - ts_rank(ts_corr(x, y, d1), d2):  rank of correlation
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .expressions import FactorExpression
from .evaluator import evaluate_factor

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
# Lookback periods (in bars; 1-bar = 1s default)
# ══════════════════════════════════════════════════════════

LOOKBACKS_SHORT = [3, 5, 10, 20, 30]
LOOKBACKS_MEDIUM = [60, 120, 180, 300]
LOOKBACKS_LONG = [600, 900, 1800]

ALL_LOOKBACKS = LOOKBACKS_SHORT + LOOKBACKS_MEDIUM + LOOKBACKS_LONG


# ══════════════════════════════════════════════════════════
# 1st-order expressions: operator(primitive, lookback)
# ══════════════════════════════════════════════════════════

FIRST_ORDER_OPERATORS = [
    "ts_mean", "ts_stddev", "ts_rank", "ts_sum",
    "ts_min", "ts_max", "ts_argmin", "ts_argmax",
    "decay_linear", "ts_skew", "ts_kurt", "ts_median",
    "ts_entropy", "ts_zscore", "ts_cv",
    "delta", "ts_returns", "ts_momentum", "ts_range",
    "ts_acceleration", "ts_autocorr",
]


def generate_first_order_expressions(
    primitives: List[str],
    lookbacks: Optional[List[int]] = None,
    operators: Optional[List[str]] = None,
) -> List[str]:
    """Generate all 1st-order factor expressions."""
    if lookbacks is None:
        lookbacks = ALL_LOOKBACKS
    if operators is None:
        operators = FIRST_ORDER_OPERATORS

    expressions = []
    for op in operators:
        for prim in primitives:
            for d in lookbacks:
                expressions.append(f"{op}({prim}, {d})")
    return expressions


# ══════════════════════════════════════════════════════════
# 2nd-order expressions: op1(op2(x, d2), d1)
# ══════════════════════════════════════════════════════════

# General templates that apply to ALL primitives
GENERAL_SECOND_ORDER_TEMPLATES = [
    "ts_rank(delta({prim}, {d1}), {d2})",
    "decay_linear(delta({prim}, {d1}), {d2})",
    "delta(ts_mean({prim}, {d1}), {d2})",
    "ts_stddev(delta({prim}, {d1}), {d2})",
    "ts_rank(ts_stddev({prim}, {d1}), {d2})",
    "ts_zscore(delta({prim}, {d1}), {d2})",
    "ts_entropy(delta({prim}, {d1}), {d2})",
    "ts_skew(delta({prim}, {d1}), {d2})",
    "ts_rank(ts_mean({prim}, {d1}), {d2})",
    "decay_linear(ts_zscore({prim}, {d1}), {d2})",
    "ts_zscore(ts_rank({prim}, {d1}), {d2})",
]

# Microstructure-specific templates: only applied to relevant primitives
# These encode the economic intuitions from the liquidity provider model
MICROSTRUCTURE_TEMPLATES = {
    # PI acceleration: book thinning → subsequent trades hit thin book → PI rises
    "price_impact": [
        "ts_acceleration(price_impact, {d1})",
        "ts_zscore(ts_acceleration(price_impact, {d1}), {d2})",
        "ts_rank(ts_acceleration(price_impact, {d1}), {d2})",
        "ts_skew(price_impact, {d2})",
        "ts_kurt(price_impact, {d2})",
        "decay_linear(ts_acceleration(price_impact, {d1}), {d2})",
    ],
    # Small-vol PI: small orders cause outsized impact → fragile state
    "small_vol_pi_ratio": [
        "ts_zscore(small_vol_pi_ratio, {d2})",
        "delta(small_vol_pi_ratio, {d1})",
        "ts_rank(delta(small_vol_pi_ratio, {d1}), {d2})",
        "ts_acceleration(small_vol_pi_ratio, {d1})",
    ],
    # Toxicity feedback: imbalance drives PI
    "abs_imbalance": [
        "ts_zscore(abs_imbalance, {d2})",
        "decay_linear(abs_imbalance, {d2})",
    ],
    # Spread-volume feedback: spread widens with volume → toxic flow
    "price_range": [
        "ts_acceleration(price_range, {d1})",
        "ts_zscore(ts_acceleration(price_range, {d1}), {d2})",
    ],
    # Volume distribution: entropy reveals informed vs noise regimes
    "volume": [
        "ts_entropy(volume, {d2})",
        "delta(ts_entropy(volume, {d1}), {d2})",
        "ts_rank(ts_entropy(volume, {d1}), {d2})",
    ],
    # VWAP deviation: where volume concentrates vs close
    "vwap_deviation": [
        "ts_skew(vwap_deviation, {d2})",
        "ts_rank(vwap_deviation, {d2})",
        "decay_linear(vwap_deviation, {d2})",
    ],
    # Bar efficiency + direction
    "bar_efficiency": [
        "ts_mean(bar_efficiency, {d2})",
        "ts_zscore(bar_efficiency, {d2})",
    ],
}

# ── Regression pairs: ts_reg_beta/alpha/residual/r2(y, x, d) ──
# Kyle's lambda and variants — ΔP = λ·V + c
REGRESSION_PAIRS = [
    # All-trade Kyle's lambda: return ~ signed_volume
    ("return_1", "signed_volume"),
    # All-trade Kyle's lambda: return ~ volume (unsigned)
    ("return_1", "volume"),
    # Directional: return ~ trade_imbalance
    ("return_1", "trade_imbalance"),
    # Small-volume Kyle's lambda: isolate fragile-state information
    ("small_vol_return", "small_vol_signed_volume"),
    ("small_vol_return", "small_vol_volume"),
    # Price impact ~ volume (PI slope = liquidity depth proxy)
    ("price_impact", "volume"),
    # Price range ~ volume (spread response to flow)
    ("price_range", "volume"),
    ("price_range", "signed_volume"),
]

REGRESSION_OUTPUTS = ["ts_reg_beta", "ts_reg_alpha", "ts_reg_residual", "ts_reg_r2"]

# Cross-field correlation pairs: ts_corr(x, y, d)
CORRELATION_PAIRS = [
    ("return_1", "volume"),
    ("return_1", "trade_imbalance"),
    ("volume", "trade_imbalance"),
    ("return_1", "avg_trade_size"),
    ("price_range", "volume"),
    # Microstructure-motivated
    ("price_impact", "volume"),               # PI-volume: thinning detection
    ("price_impact", "abs_imbalance"),         # toxicity feedback
    ("price_impact", "trade_imbalance"),       # directional toxicity
    ("abs_imbalance", "price_range"),          # imbalance-spread feedback
    ("small_vol_pi_ratio", "price_impact"),    # fragility × impact
    ("vwap_deviation", "trade_imbalance"),     # volume asymmetry × direction
    ("bar_efficiency", "volume"),              # conviction × activity
    ("signed_price_impact", "trade_imbalance"),# directional PI × imbalance
]

# Orderbook cross-field pairs (when orderbook data available)
ORDERBOOK_CORRELATION_PAIRS = [
    ("depth_imbalance", "return_1"),
    ("depth_imbalance", "price_impact"),
    ("spread_bps", "volume"),
    ("spread_bps", "price_impact"),
    ("book_skew", "return_1"),
    ("book_skew", "trade_imbalance"),
    ("total_depth_5", "price_impact"),
]


def generate_second_order_expressions(
    primitives: List[str],
    lookbacks_inner: Optional[List[int]] = None,
    lookbacks_outer: Optional[List[int]] = None,
) -> List[str]:
    """
    Generate economically meaningful 2nd-order factor expressions.

    Includes:
    1. General templates × all primitives
    2. Microstructure-specific templates × relevant primitives
    3. Cross-field correlations (+ ranked correlations)
    4. Nonlinear rank emphasis
    """
    if lookbacks_inner is None:
        lookbacks_inner = LOOKBACKS_SHORT + LOOKBACKS_MEDIUM
    if lookbacks_outer is None:
        lookbacks_outer = LOOKBACKS_MEDIUM + LOOKBACKS_LONG

    expressions: List[str] = []

    # ── 1. General templates ──
    for prim in primitives:
        for d1 in lookbacks_inner:
            for d2 in lookbacks_outer:
                if d2 <= d1:
                    continue
                for tmpl in GENERAL_SECOND_ORDER_TEMPLATES:
                    expressions.append(tmpl.format(prim=prim, d1=d1, d2=d2))

    # ── 2. Microstructure templates ──
    for prim_key, templates in MICROSTRUCTURE_TEMPLATES.items():
        if prim_key not in primitives:
            continue
        for tmpl in templates:
            needs_d1 = "{d1}" in tmpl
            needs_d2 = "{d2}" in tmpl
            if needs_d1 and needs_d2:
                for d1 in lookbacks_inner:
                    for d2 in lookbacks_outer:
                        if d2 <= d1:
                            continue
                        expressions.append(tmpl.format(d1=d1, d2=d2))
            elif needs_d1:
                for d1 in lookbacks_inner:
                    expressions.append(tmpl.format(d1=d1))
            elif needs_d2:
                for d2 in lookbacks_outer:
                    expressions.append(tmpl.format(d2=d2))

    # ── 3. Cross-field correlations ──
    all_corr_pairs = CORRELATION_PAIRS.copy()
    for f1, f2 in all_corr_pairs:
        if f1 not in primitives or f2 not in primitives:
            continue
        for d in lookbacks_outer:
            expressions.append(f"ts_corr({f1}, {f2}, {d})")
            for d2 in lookbacks_outer:
                if d2 > d:
                    expressions.append(f"ts_rank(ts_corr({f1}, {f2}, {d}), {d2})")

    # ── 4. Regression (Kyle's lambda and variants) ──
    for y_field, x_field in REGRESSION_PAIRS:
        if y_field not in primitives or x_field not in primitives:
            continue
        for d in ALL_LOOKBACKS:
            for reg_op in REGRESSION_OUTPUTS:
                expressions.append(f"{reg_op}({y_field}, {x_field}, {d})")

        # 2nd-order: apply TS operators to regression outputs
        for d1 in lookbacks_inner:
            for d2 in lookbacks_outer:
                if d2 <= d1:
                    continue
                # Rank/zscore/delta of Kyle's lambda over time
                expressions.append(f"ts_rank(ts_reg_beta({y_field}, {x_field}, {d1}), {d2})")
                expressions.append(f"ts_zscore(ts_reg_beta({y_field}, {x_field}, {d1}), {d2})")
                expressions.append(f"delta(ts_reg_beta({y_field}, {x_field}, {d1}), {d2})")
                expressions.append(f"ts_acceleration(ts_reg_beta({y_field}, {x_field}, {d1}), {d2})")
                # Entropy/skew of residuals — regime detection
                expressions.append(f"ts_entropy(ts_reg_residual({y_field}, {x_field}, {d1}), {d2})")
                expressions.append(f"ts_skew(ts_reg_residual({y_field}, {x_field}, {d1}), {d2})")

    # ── 5. Nonlinear rank emphasis ──
    for prim in primitives:
        for d in lookbacks_outer:
            expressions.append(f"signedpower(ts_rank({prim}, {d}), 2)")

    return expressions


def generate_orderbook_expressions(
    ob_primitives: List[str],
    trade_primitives: List[str],
    lookbacks_inner: Optional[List[int]] = None,
    lookbacks_outer: Optional[List[int]] = None,
) -> List[str]:
    """
    Generate expressions that involve orderbook primitives.
    Applied to: pure OB 1st-order, OB 2nd-order, and OB×trade cross-correlations.
    """
    if lookbacks_inner is None:
        lookbacks_inner = LOOKBACKS_SHORT + LOOKBACKS_MEDIUM
    if lookbacks_outer is None:
        lookbacks_outer = LOOKBACKS_MEDIUM + LOOKBACKS_LONG

    expressions: List[str] = []

    # 1st-order on OB primitives
    for op in FIRST_ORDER_OPERATORS:
        for prim in ob_primitives:
            for d in ALL_LOOKBACKS:
                expressions.append(f"{op}({prim}, {d})")

    # 2nd-order on OB primitives
    for prim in ob_primitives:
        for d1 in lookbacks_inner:
            for d2 in lookbacks_outer:
                if d2 <= d1:
                    continue
                for tmpl in GENERAL_SECOND_ORDER_TEMPLATES:
                    expressions.append(tmpl.format(prim=prim, d1=d1, d2=d2))

    # Cross-correlations between OB and trade primitives
    for f1, f2 in ORDERBOOK_CORRELATION_PAIRS:
        all_available = ob_primitives + trade_primitives
        if f1 not in all_available or f2 not in all_available:
            continue
        for d in lookbacks_outer:
            expressions.append(f"ts_corr({f1}, {f2}, {d})")
            for d2 in lookbacks_outer:
                if d2 > d:
                    expressions.append(f"ts_rank(ts_corr({f1}, {f2}, {d}), {d2})")

    # Regression with OB data: return ~ spread, return ~ depth_imbalance
    ob_regression_pairs = [
        ("return_1", "spread_bps"),
        ("return_1", "depth_imbalance"),
        ("price_impact", "spread_bps"),
        ("price_impact", "total_depth_5"),
    ]
    for y_field, x_field in ob_regression_pairs:
        all_available = ob_primitives + trade_primitives
        if y_field not in all_available or x_field not in all_available:
            continue
        for d in ALL_LOOKBACKS:
            for reg_op in REGRESSION_OUTPUTS:
                expressions.append(f"{reg_op}({y_field}, {x_field}, {d})")

    return expressions


# ══════════════════════════════════════════════════════════
# Scanner
# ══════════════════════════════════════════════════════════

def scan_factors(
    data: pd.DataFrame,
    returns_1min: pd.Series,
    returns_5min: pd.Series,
    expressions: Optional[List[str]] = None,
    primitives: Optional[List[str]] = None,
    max_order: int = 2,
    ic_window: int = 100,
    min_ic_abs: float = 0.01,
    max_expressions: int = 5000,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Scan factor expressions and evaluate them.

    data: DataFrame with primitive columns (from compute_trade_primitives)
    returns_1min: 1-minute forward returns (bps)
    returns_5min: 5-minute forward returns (bps)
    expressions: list of expression strings (auto-generated if None)
    max_order: 1 or 2 (controls expression generation)

    Returns DataFrame with one row per expression, sorted by |IC_5min|.
    """
    if primitives is None:
        from .data_schema import TRADE_PRIMITIVES
        primitives = [p for p in TRADE_PRIMITIVES if p in data.columns]

    if expressions is None:
        expressions = []
        expressions.extend(generate_first_order_expressions(primitives))
        if max_order >= 2:
            expressions.extend(generate_second_order_expressions(primitives))

    # Deduplicate
    expressions = list(dict.fromkeys(expressions))

    if len(expressions) > max_expressions:
        logger.info("Limiting to %d expressions (from %d)", max_expressions, len(expressions))
        expressions = expressions[:max_expressions]

    logger.info("Scanning %d factor expressions...", len(expressions))

    results = []
    t0 = time.time()
    n_errors = 0

    for i, expr_str in enumerate(expressions):
        if verbose and (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            logger.info("  %d/%d (%.1fs, %d errors)", i + 1, len(expressions), elapsed, n_errors)

        try:
            expr = FactorExpression(expr_str)
            factor_values = expr.evaluate(data)

            from .evaluator import compute_factor_ic
            quick_ic = compute_factor_ic(factor_values, returns_5min)
            if np.isnan(quick_ic) or abs(quick_ic) < min_ic_abs:
                continue

            metrics = evaluate_factor(factor_values, returns_1min, returns_5min, ic_window)
            metrics["expression"] = expr_str
            metrics["order"] = expr.order
            metrics["fields"] = ",".join(expr.fields_used)
            results.append(metrics)

        except Exception as e:
            n_errors += 1
            if n_errors <= 10:
                logger.debug("Error evaluating '%s': %s", expr_str, e)

    elapsed = time.time() - t0
    logger.info("Scan complete: %d/%d factors passed (%.1fs, %d errors)",
                len(results), len(expressions), elapsed, n_errors)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values("ic_5min", key=abs, ascending=False).reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════
# Convenience: run full pipeline
# ══════════════════════════════════════════════════════════

def run_factor_scan(
    trades: pd.DataFrame,
    freq: str = "1s",
    max_order: int = 2,
    max_expressions: int = 5000,
    orderbook: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    End-to-end factor scan: raw trades → primitives → forward returns → scan.

    trades: raw aggTrades DataFrame
    freq: bar frequency for primitives
    orderbook: optional orderbook DataFrame (adds OB-based expressions)
    """
    from .data_schema import compute_trade_primitives, compute_orderbook_primitives
    from .data_schema import TRADE_PRIMITIVES, ORDERBOOK_PRIMITIVES

    logger.info("Computing trade primitives at %s...", freq)
    primitives = compute_trade_primitives(trades, freq)

    if len(primitives) < 500:
        logger.warning("Too few bars (%d), skipping scan", len(primitives))
        return pd.DataFrame()

    # Forward returns
    cum_ret = primitives["return_1"].cumsum()
    returns_1min = cum_ret.shift(-60) - cum_ret    # 60 bars at 1s = 1min
    returns_5min = cum_ret.shift(-300) - cum_ret   # 300 bars at 1s = 5min

    # Build expressions
    avail_prims = [p for p in TRADE_PRIMITIVES if p in primitives.columns]
    expressions = generate_first_order_expressions(avail_prims)
    if max_order >= 2:
        expressions.extend(generate_second_order_expressions(avail_prims))

    # If orderbook data is available, merge and add OB expressions
    data = primitives
    if orderbook is not None and len(orderbook) > 0:
        logger.info("Computing orderbook primitives...")
        ob_prims = compute_orderbook_primitives(orderbook)
        ob_prims = ob_prims.reindex(primitives.index, method="ffill")
        data = pd.concat([primitives, ob_prims], axis=1)

        avail_ob = [p for p in ORDERBOOK_PRIMITIVES if p in ob_prims.columns]
        expressions.extend(generate_orderbook_expressions(avail_ob, avail_prims))

    logger.info("Scanning factors (max_order=%d, max_expr=%d, total_generated=%d)...",
                max_order, max_expressions, len(expressions))

    return scan_factors(
        data=data,
        returns_1min=returns_1min,
        returns_5min=returns_5min,
        expressions=expressions,
        primitives=avail_prims,
        max_order=max_order,
        max_expressions=max_expressions,
    )
