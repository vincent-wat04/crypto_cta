"""
Time series operators (WorldQuant Brain-style).

All operators are pure functions: pd.Series → pd.Series.
Convention: first argument is always the data series, `d` is the lookback period.

Categories:
  - Lag/Diff: delay, delta
  - Statistics: ts_mean, ts_stddev, ts_sum, ts_product, ts_corr, ts_cov
  - Extrema: ts_min, ts_max, ts_argmin, ts_argmax
  - Rank: ts_rank
  - Decay: decay_linear, decay_exp
  - Higher moments: ts_skew, ts_kurt, ts_median, ts_entropy
  - Nonlinear: signedpower, log1p_safe, abs_val
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ══════════════════════════════════════════════════════════
# Lag / Diff
# ══════════════════════════════════════════════════════════

def delay(x: pd.Series, d: int) -> pd.Series:
    """x shifted by d periods (x[t-d])."""
    return x.shift(d)


def delta(x: pd.Series, d: int) -> pd.Series:
    """x[t] - x[t-d]."""
    return x - x.shift(d)


# ══════════════════════════════════════════════════════════
# Rolling Statistics
# ══════════════════════════════════════════════════════════

def ts_mean(x: pd.Series, d: int) -> pd.Series:
    """Rolling mean over past d periods."""
    return x.rolling(d, min_periods=max(1, d // 2)).mean()


def ts_stddev(x: pd.Series, d: int) -> pd.Series:
    """Rolling standard deviation over past d periods."""
    return x.rolling(d, min_periods=max(2, d // 2)).std()


def ts_sum(x: pd.Series, d: int) -> pd.Series:
    """Rolling sum over past d periods."""
    return x.rolling(d, min_periods=1).sum()


def ts_product(x: pd.Series, d: int) -> pd.Series:
    """Rolling product over past d periods. Uses log-sum-exp for stability."""
    log_x = np.log(x.clip(lower=1e-20))
    log_sum = log_x.rolling(d, min_periods=1).sum()
    return np.exp(log_sum)


def ts_corr(x: pd.Series, y: pd.Series, d: int) -> pd.Series:
    """Rolling Pearson correlation between x and y over d periods."""
    return x.rolling(d, min_periods=max(3, d // 2)).corr(y)


def ts_cov(x: pd.Series, y: pd.Series, d: int) -> pd.Series:
    """Rolling covariance between x and y over d periods."""
    return x.rolling(d, min_periods=max(3, d // 2)).cov(y)


# ══════════════════════════════════════════════════════════
# Extrema
# ══════════════════════════════════════════════════════════

def ts_min(x: pd.Series, d: int) -> pd.Series:
    """Rolling minimum over past d periods."""
    return x.rolling(d, min_periods=1).min()


def ts_max(x: pd.Series, d: int) -> pd.Series:
    """Rolling maximum over past d periods."""
    return x.rolling(d, min_periods=1).max()


def ts_argmin(x: pd.Series, d: int) -> pd.Series:
    """Position of minimum within rolling window (0=most recent, d-1=oldest)."""
    return d - 1 - x.rolling(d, min_periods=1).apply(np.argmin, raw=True)


def ts_argmax(x: pd.Series, d: int) -> pd.Series:
    """Position of maximum within rolling window (0=most recent, d-1=oldest)."""
    return d - 1 - x.rolling(d, min_periods=1).apply(np.argmax, raw=True)


# ══════════════════════════════════════════════════════════
# Rank
# ══════════════════════════════════════════════════════════

def ts_rank(x: pd.Series, d: int) -> pd.Series:
    """Rolling percentile rank within window (0 to 1)."""
    def _rank_pct(arr):
        if len(arr) < 2:
            return 0.5
        rank = sp_stats.rankdata(arr)[-1]
        return rank / len(arr)
    return x.rolling(d, min_periods=max(2, d // 2)).apply(_rank_pct, raw=True)


# ══════════════════════════════════════════════════════════
# Decay-Weighted Averages
# ══════════════════════════════════════════════════════════

def decay_linear(x: pd.Series, d: int) -> pd.Series:
    """Linear decay-weighted average: weight[i] = d-i (most recent gets weight 1)."""
    weights = np.arange(1, d + 1, dtype=float)
    weights /= weights.sum()

    def _wma(arr):
        n = len(arr)
        w = weights[-n:]
        w = w / w.sum()
        return np.dot(arr, w)

    return x.rolling(d, min_periods=max(1, d // 2)).apply(_wma, raw=True)


def decay_exp(x: pd.Series, d: int, factor: float = 0.5) -> pd.Series:
    """Exponential decay-weighted average: weight[i] = factor^(d-1-i)."""
    weights = np.array([factor ** i for i in range(d - 1, -1, -1)], dtype=float)
    weights /= weights.sum()

    def _ewma(arr):
        n = len(arr)
        w = weights[-n:]
        w = w / w.sum()
        return np.dot(arr, w)

    return x.rolling(d, min_periods=max(1, d // 2)).apply(_ewma, raw=True)


# ══════════════════════════════════════════════════════════
# Higher Moments
# ══════════════════════════════════════════════════════════

def ts_skew(x: pd.Series, d: int) -> pd.Series:
    """Rolling skewness over d periods."""
    return x.rolling(d, min_periods=max(3, d // 2)).skew()


def ts_kurt(x: pd.Series, d: int) -> pd.Series:
    """Rolling excess kurtosis over d periods."""
    return x.rolling(d, min_periods=max(4, d // 2)).kurt()


def ts_median(x: pd.Series, d: int) -> pd.Series:
    """Rolling median over d periods."""
    return x.rolling(d, min_periods=1).median()


def ts_entropy(x: pd.Series, d: int, n_bins: int = 10) -> pd.Series:
    """
    Rolling Shannon entropy of x's distribution over d periods.
    Discretizes into n_bins and computes -sum(p * log(p)).
    Higher entropy = more uniform; lower = concentrated.
    """
    def _entropy(arr):
        arr = arr[~np.isnan(arr)]
        if len(arr) < 3:
            return 0.0
        counts, _ = np.histogram(arr, bins=n_bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs))
    return x.rolling(d, min_periods=max(3, d // 2)).apply(_entropy, raw=True)


# ══════════════════════════════════════════════════════════
# Nonlinear
# ══════════════════════════════════════════════════════════

def signedpower(x: pd.Series, a: float) -> pd.Series:
    """Signed power: sign(x) * |x|^a. Preserves sign."""
    return np.sign(x) * np.abs(x) ** a


def log1p_safe(x: pd.Series) -> pd.Series:
    """log(1 + |x|) * sign(x). Safe for negative values."""
    return np.sign(x) * np.log1p(np.abs(x))


def abs_val(x: pd.Series) -> pd.Series:
    """Absolute value."""
    return np.abs(x)


# ══════════════════════════════════════════════════════════
# Additional useful operators
# ══════════════════════════════════════════════════════════

def ts_zscore(x: pd.Series, d: int) -> pd.Series:
    """Rolling z-score: (x - mean) / std."""
    m = ts_mean(x, d)
    s = ts_stddev(x, d)
    return (x - m) / (s + 1e-12)


def ts_returns(x: pd.Series, d: int = 1) -> pd.Series:
    """Percentage return over d periods: (x[t] - x[t-d]) / x[t-d]."""
    return x.pct_change(d)


def ts_acceleration(x: pd.Series, d: int) -> pd.Series:
    """Second derivative: delta(delta(x, d), d)."""
    return delta(delta(x, d), d)


def ts_momentum(x: pd.Series, d: int) -> pd.Series:
    """Momentum: x[t] / x[t-d] - 1."""
    return x / x.shift(d) - 1


def ts_range(x: pd.Series, d: int) -> pd.Series:
    """Rolling range: max - min over d periods."""
    return ts_max(x, d) - ts_min(x, d)


def ts_cv(x: pd.Series, d: int) -> pd.Series:
    """Coefficient of variation: std / |mean|."""
    m = ts_mean(x, d)
    s = ts_stddev(x, d)
    return s / (np.abs(m) + 1e-12)


def ts_autocorr(x: pd.Series, d: int, lag: int = 1) -> pd.Series:
    """Rolling autocorrelation: corr(x[t], x[t-lag]) over d periods."""
    return x.rolling(d, min_periods=max(3, d // 2)).corr(x.shift(lag))


# ══════════════════════════════════════════════════════════
# Regression (OLS)
# ══════════════════════════════════════════════════════════
# Key use case: Kyle's lambda = ts_reg_beta(return_1, signed_volume, d)
#   ΔP = λ·V + c  →  λ measures price impact per unit of signed order flow
# Also useful for small-volume trades: ts_reg_beta(small_vol_return, small_vol_volume, d)

def _rolling_ols(y: pd.Series, x: pd.Series, d: int, output: str) -> pd.Series:
    """
    Rolling OLS regression: y = beta * x + alpha + residual.

    output: "beta" | "alpha" | "residual" | "r2"
    Returns the requested component as a Series.
    """
    min_obs = max(3, d // 2)
    result = pd.Series(np.nan, index=y.index, dtype=float)

    y_vals = y.values.astype(float)
    x_vals = x.values.astype(float)
    n = len(y_vals)

    for i in range(d - 1, n):
        start = i - d + 1
        yy = y_vals[start:i + 1]
        xx = x_vals[start:i + 1]
        valid = ~(np.isnan(yy) | np.isnan(xx))
        if valid.sum() < min_obs:
            continue
        yy = yy[valid]
        xx = xx[valid]

        x_mean = xx.mean()
        y_mean = yy.mean()
        x_dev = xx - x_mean
        ss_xx = np.dot(x_dev, x_dev)
        if ss_xx < 1e-20:
            continue
        ss_xy = np.dot(x_dev, yy - y_mean)

        beta = ss_xy / ss_xx
        alpha = y_mean - beta * x_mean

        if output == "beta":
            result.iloc[i] = beta
        elif output == "alpha":
            result.iloc[i] = alpha
        elif output == "residual":
            y_hat = beta * x_vals[i] + alpha
            result.iloc[i] = y_vals[i] - y_hat
        elif output == "r2":
            y_hat = beta * xx + alpha
            ss_res = np.sum((yy - y_hat) ** 2)
            ss_tot = np.sum((yy - y_mean) ** 2)
            result.iloc[i] = 1 - ss_res / (ss_tot + 1e-20)

    return result


def ts_reg_beta(y: pd.Series, x: pd.Series, d: int) -> pd.Series:
    """Rolling OLS slope: y = beta*x + alpha. Returns beta (e.g. Kyle's lambda)."""
    return _rolling_ols(y, x, d, "beta")


def ts_reg_alpha(y: pd.Series, x: pd.Series, d: int) -> pd.Series:
    """Rolling OLS intercept: y = beta*x + alpha. Returns alpha."""
    return _rolling_ols(y, x, d, "alpha")


def ts_reg_residual(y: pd.Series, x: pd.Series, d: int) -> pd.Series:
    """Rolling OLS residual at current bar: y[t] - (beta*x[t] + alpha)."""
    return _rolling_ols(y, x, d, "residual")


def ts_reg_r2(y: pd.Series, x: pd.Series, d: int) -> pd.Series:
    """Rolling OLS R-squared: goodness of fit."""
    return _rolling_ols(y, x, d, "r2")


# ══════════════════════════════════════════════════════════
# Operator Registry
# ══════════════════════════════════════════════════════════

UNARY_OPERATORS = {
    "delay": delay,
    "delta": delta,
    "ts_mean": ts_mean,
    "ts_stddev": ts_stddev,
    "ts_sum": ts_sum,
    "ts_product": ts_product,
    "ts_min": ts_min,
    "ts_max": ts_max,
    "ts_argmin": ts_argmin,
    "ts_argmax": ts_argmax,
    "ts_rank": ts_rank,
    "decay_linear": decay_linear,
    "decay_exp": decay_exp,
    "ts_skew": ts_skew,
    "ts_kurt": ts_kurt,
    "ts_median": ts_median,
    "ts_entropy": ts_entropy,
    "ts_zscore": ts_zscore,
    "ts_returns": ts_returns,
    "ts_acceleration": ts_acceleration,
    "ts_momentum": ts_momentum,
    "ts_range": ts_range,
    "ts_cv": ts_cv,
    "ts_autocorr": ts_autocorr,
    "signedpower": signedpower,
    "log1p_safe": log1p_safe,
    "abs_val": abs_val,
}

BINARY_OPERATORS = {
    "ts_corr": ts_corr,
    "ts_cov": ts_cov,
    "ts_reg_beta": ts_reg_beta,
    "ts_reg_alpha": ts_reg_alpha,
    "ts_reg_residual": ts_reg_residual,
    "ts_reg_r2": ts_reg_r2,
}
