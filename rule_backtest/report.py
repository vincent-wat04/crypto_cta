"""
Backtest report:
  - 5-panel static PNG  (panels 1-5: cum return, drawdown, portfolio, long/short, turnover)
  - Interactive HTML     (panel 6 replacement: price + entries/exits + position strip, scrollable)
  - Metrics text file
  - Trades CSV
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def generate_report(
    trades: pd.DataFrame,
    metrics: Dict[str, Any],
    params: Dict[str, Any],
    initial_capital: float = 100_000.0,
    output_dir: Optional[Path] = None,
    show: bool = False,
) -> Path | None:
    if trades.empty:
        print("[Report] No trades to report.")
        return None

    df = trades.copy()

    # Derived series
    returns_pct = df["net_pnl_bps"] / 10000.0
    cum_return = (1 + returns_pct).cumprod()
    cum_return_pct = (cum_return - 1) * 100
    peak = cum_return.cummax()
    drawdown_pct = (cum_return - peak) / peak * 100
    portfolio_value = initial_capital * cum_return

    long_ret = df["net_pnl_bps"].where(df["position"] > 1e-8, 0) / 10000.0
    short_ret = df["net_pnl_bps"].where(df["position"] < -1e-8, 0) / 10000.0
    long_cum_pct = ((1 + long_ret).cumprod() - 1) * 100
    short_cum_pct = ((1 + short_ret).cumprod() - 1) * 100

    turnover_pct = df["position"].diff().abs() * 100
    turnover_smooth = turnover_pct.rolling(30, min_periods=1).mean()
    pos_count = (df["position"].abs() > 1e-8).astype(int)

    # ── Static PNG (5 panels) ──
    title_str = _build_title(params)
    fig = plt.figure(figsize=(16, 20), facecolor="white")
    fig.suptitle(title_str, fontsize=14, fontweight="bold", y=0.99)

    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3,
                          left=0.08, right=0.95, top=0.96, bottom=0.04)

    ax1 = fig.add_subplot(gs[0, :])
    _plot_cumulative_return(ax1, cum_return_pct)

    ax2 = fig.add_subplot(gs[1, 0])
    _plot_drawdown(ax2, drawdown_pct, metrics)

    ax3 = fig.add_subplot(gs[1, 1])
    _plot_portfolio_value(ax3, portfolio_value)

    ax4 = fig.add_subplot(gs[2, 0])
    _plot_long_short(ax4, long_cum_pct, short_cum_pct)

    ax5 = fig.add_subplot(gs[2, 1])
    _plot_turnover_positions(ax5, turnover_smooth, pos_count)

    # Save
    saved_path = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        png_path = output_dir / "backtest_report.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        saved_path = png_path
        print(f"[Report] Chart saved to {png_path}")

        txt_path = output_dir / "backtest_metrics.txt"
        txt_path.write_text(_format_metrics(metrics, params), encoding="utf-8")
        print(f"[Report] Metrics saved to {txt_path}")

        csv_path = output_dir / "trades.csv"
        df.to_csv(csv_path)
        print(f"[Report] Trades CSV saved to {csv_path}")

        # Interactive HTML for panel 6
        html_path = _generate_interactive_html(df, metrics, params, output_dir)
        if html_path:
            print(f"[Report] Interactive chart saved to {html_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return saved_path


# ═══════════════════════════════════════════════════════════
# Interactive HTML (panel 6 replacement)
# ═══════════════════════════════════════════════════════════

def _pick_ma_windows(n_bars: int) -> tuple[int, int]:
    """Choose SMA / EMA window lengths that scale with data size."""
    if n_bars >= 2000:
        return 60, 20
    if n_bars >= 500:
        return 30, 10
    if n_bars >= 150:
        return 15, 5
    return 8, 3

def _generate_interactive_html(
    df: pd.DataFrame, metrics: Dict, params: Dict, output_dir: Path,
) -> Path | None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("[Report] plotly not installed — skipping interactive chart.")
        return None

    close = df["close"]
    pos = df["position"]
    factor = df["factor"] if "factor" in df.columns else None

    # Use bar index for x-axis (works for both time and trade-count bars)
    x = df.index

    # Entry/exit masks
    pos_prev = pos.shift(1).fillna(0)
    long_entry = (pos > 1e-8) & ((pos_prev.abs() < 1e-8) | (pos_prev < -1e-8))
    short_entry = (pos < -1e-8) & ((pos_prev.abs() < 1e-8) | (pos_prev > 1e-8))
    exit_mask = (pos.abs() < 1e-8) & (pos_prev.abs() > 1e-8)

    n_rows = 3 if factor is not None else 2
    row_heights = [0.6, 0.15, 0.25] if n_rows == 3 else [0.7, 0.3]
    subplot_titles = ["Price & Trades", "Position"]
    if factor is not None:
        subplot_titles.append("Factor")

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    # Row 1: Price line
    fig.add_trace(go.Scatter(
        x=x, y=close.values, mode="lines", name="Price",
        line=dict(color="#1f77b4", width=1.2),
    ), row=1, col=1)

    # Long entries
    le_idx = df.index[long_entry]
    fig.add_trace(go.Scatter(
        x=le_idx, y=close.reindex(le_idx).values,
        mode="markers", name="Long Entry",
        marker=dict(symbol="triangle-up", size=8, color="#2ecc71",
                    line=dict(width=0.5, color="black")),
    ), row=1, col=1)

    # Short entries
    se_idx = df.index[short_entry]
    fig.add_trace(go.Scatter(
        x=se_idx, y=close.reindex(se_idx).values,
        mode="markers", name="Short Entry",
        marker=dict(symbol="triangle-down", size=8, color="#e74c3c",
                    line=dict(width=0.5, color="black")),
    ), row=1, col=1)

    # Exits
    ex_idx = df.index[exit_mask]
    fig.add_trace(go.Scatter(
        x=ex_idx, y=close.reindex(ex_idx).values,
        mode="markers", name="Exit",
        marker=dict(symbol="circle", size=5, color="#f39c12",
                    line=dict(width=0.3, color="black")),
    ), row=1, col=1)

    # Row 2: Position as colored bars
    pos_colors = ["#2ecc71" if v > 1e-8 else "#e74c3c" if v < -1e-8 else "#d5d5d5"
                  for v in pos.values]
    fig.add_trace(go.Bar(
        x=x, y=pos.values, name="Position",
        marker_color=pos_colors, showlegend=False,
    ), row=2, col=1)

    # Row 3: Factor with SMA / EMA overlays
    if factor is not None:
        fig.add_trace(go.Scatter(
            x=x, y=factor.values, mode="lines", name="Factor",
            line=dict(color="#9b59b6", width=0.8),
        ), row=n_rows, col=1)

        sma_w, ema_w = _pick_ma_windows(len(factor))
        sma = factor.rolling(sma_w, min_periods=1).mean()
        ema = factor.ewm(span=ema_w, min_periods=1).mean()

        fig.add_trace(go.Scatter(
            x=x, y=sma.values, mode="lines",
            name=f"SMA({sma_w})",
            line=dict(color="#e67e22", width=1.0, dash="dot"),
        ), row=n_rows, col=1)
        fig.add_trace(go.Scatter(
            x=x, y=ema.values, mode="lines",
            name=f"EMA({ema_w})",
            line=dict(color="#2ecc71", width=1.0, dash="dash"),
        ), row=n_rows, col=1)

    # Layout: scrollable via rangeslider
    avg_min = metrics.get("avg_hold_minutes", 0)
    n_rt = metrics.get("n_round_trips", 0)
    fill_rate = params.get("maker_fill_rate", "?")
    factor_name = params.get("factor", "")

    fig.update_layout(
        title=dict(
            text=(f"{factor_name} | Avg Hold: {avg_min:.1f} min | "
                  f"{n_rt} round-trips | Fill Rate: {fill_rate}"),
            font=dict(size=14),
        ),
        height=800,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=80, b=40),
    )

    # Enable x-axis range slider on the bottom subplot for scrolling
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.05), row=n_rows, col=1)
    fig.update_xaxes(showticklabels=True, row=1, col=1)

    html_path = output_dir / "price_position_interactive.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    return html_path


# ═══════════════════════════════════════════════════════════
# Static plot helpers (panels 1-5)
# ═══════════════════════════════════════════════════════════

def _plot_cumulative_return(ax: plt.Axes, cum_pct: pd.Series) -> None:
    ax.set_title("Portfolio Cumulative Return Over Time", fontsize=12, fontweight="bold")
    ax.fill_between(cum_pct.index, 0, cum_pct.values, alpha=0.3, color="deepskyblue")
    ax.plot(cum_pct.index, cum_pct.values, color="deepskyblue", linewidth=1.2)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    min_idx = cum_pct.idxmin()
    min_val = cum_pct.min()
    ax.annotate(f"Min: {min_val:.2f}%",
                xy=(min_idx, min_val), fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))
    ax.set_ylabel("Cumulative Return (%)")
    _format_date_axis(ax)


def _plot_drawdown(ax: plt.Axes, dd_pct: pd.Series, metrics: Dict) -> None:
    ax.set_title("Portfolio Drawdown", fontsize=12, fontweight="bold")
    ax.fill_between(dd_pct.index, 0, dd_pct.values, alpha=0.4, color="mediumvioletred")
    ax.plot(dd_pct.index, dd_pct.values, color="mediumvioletred", linewidth=0.8)
    max_dd = metrics.get("max_drawdown_pct", 0)
    ax.set_ylabel("Drawdown (%)")
    ax.legend([f"Max DD: {max_dd:.2f}%"], loc="lower left", fontsize=8)
    _format_date_axis(ax)


def _plot_portfolio_value(ax: plt.Axes, value: pd.Series) -> None:
    ax.set_title("Portfolio Value", fontsize=12, fontweight="bold")
    ax.plot(value.index, value.values, color="darkorange", linewidth=1.2)
    ax.set_ylabel("Value (USD)")
    _format_date_axis(ax)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))


def _plot_long_short(ax: plt.Axes, long_pct: pd.Series, short_pct: pd.Series) -> None:
    ax.set_title("Long vs Short Cumulative Returns", fontsize=12, fontweight="bold")
    ax.plot(long_pct.index, long_pct.values, color="mediumturquoise", linewidth=1.0, label="Long")
    ax.plot(short_pct.index, short_pct.values, color="mediumvioletred", linewidth=1.0, label="Short")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend(loc="upper left", fontsize=8)
    _format_date_axis(ax)


def _plot_turnover_positions(ax: plt.Axes, turnover: pd.Series, pos_count: pd.Series) -> None:
    ax.set_title("Turnover & Position Count", fontsize=12, fontweight="bold")
    color1, color2 = "mediumturquoise", "deeppink"

    n = len(turnover)
    if n > 2000:
        t_rs = turnover.resample("1h").mean()
        p_rs = pos_count.resample("1h").mean()
    elif n > 500:
        t_rs = turnover.resample("15min").mean()
        p_rs = pos_count.resample("15min").mean()
    else:
        t_rs, p_rs = turnover, pos_count

    bw = (t_rs.index[-1] - t_rs.index[0]) / len(t_rs) * 0.8
    ax.bar(t_rs.index, t_rs.values, width=bw, color=color1, alpha=0.6, label="Turnover (%)")
    ax.set_ylabel("Turnover (%)", color=color1)
    ax.tick_params(axis="y", labelcolor=color1)

    ax2 = ax.twinx()
    ax2.plot(p_rs.index, p_rs.values, color=color2, linewidth=1.0, label="Positions")
    ax2.set_ylabel("Positions", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    _format_date_axis(ax)


def _format_date_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.grid(True, alpha=0.3)


def _build_title(params: Dict) -> str:
    factor = params.get("factor", "unknown")
    signal = params.get("signal", "")
    fill = params.get("maker_fill_rate", "?")
    bar_sec = params.get("bar_seconds", 60)
    bar_label = f"{bar_sec // 60}min" if bar_sec >= 60 else f"{bar_sec}s"
    bar_mode = params.get("bar_mode", "time")
    return (f"Rule Strategy: {factor} | Signal={signal} | "
            f"{bar_label} ({bar_mode}) | Fill={fill}")


def _format_metrics(metrics: Dict, params: Dict) -> str:
    lines = [
        "=" * 60,
        "BACKTEST REPORT",
        "=" * 60,
        "",
        "── Parameters ──",
    ]
    for k, v in sorted(params.items()):
        lines.append(f"  {k:30s}: {v}")
    lines += [
        "",
        "── Performance ──",
        f"  {'Net PnL (bps)':30s}: {metrics.get('net_pnl_bps', 0):>12.2f}",
        f"  {'Total Return':30s}: {metrics.get('total_return_pct', 0):>12.2f}%",
        f"  {'Annualized Return':30s}: {metrics.get('annualized_return_pct', 0):>12.2f}%",
        f"  {'Max Drawdown':30s}: {metrics.get('max_drawdown_pct', 0):>12.2f}%",
        f"  {'Volatility (ann)':30s}: {metrics.get('volatility_ann_pct', 0):>12.2f}%",
        f"  {'Sharpe':30s}: {metrics.get('sharpe_ratio', 0):>12.4f}",
        f"  {'Sortino':30s}: {metrics.get('sortino_ratio', 0):>12.4f}",
        f"  {'Calmar':30s}: {metrics.get('calmar_ratio', 0):>12.4f}",
        f"  {'Win Rate':30s}: {metrics.get('win_rate_pct', 0):>12.2f}%",
        f"  {'Profit Factor':30s}: {metrics.get('profit_factor', 0):>12.4f}",
        "",
        "── Activity ──",
        f"  {'Total Bars':30s}: {metrics.get('n_bars', 0):>12}",
        f"  {'Position Changes':30s}: {metrics.get('n_position_changes', 0):>12}",
        f"  {'Long Bars':30s}: {metrics.get('long_bars', 0):>12}",
        f"  {'Short Bars':30s}: {metrics.get('short_bars', 0):>12}",
        f"  {'Flat Bars':30s}: {metrics.get('flat_bars', 0):>12}",
        f"  {'Long PnL (bps)':30s}: {metrics.get('long_pnl_bps', 0):>12.2f}",
        f"  {'Short PnL (bps)':30s}: {metrics.get('short_pnl_bps', 0):>12.2f}",
        f"  {'Total Cost (bps)':30s}: {metrics.get('total_cost_bps', 0):>12.2f}",
        f"  {'Avg Turnover':30s}: {metrics.get('avg_turnover_pct', 0):>12.2f}%",
        "",
        "── Holding Period ──",
        f"  {'Round-trips':30s}: {metrics.get('n_round_trips', 0):>12}",
        f"  {'Avg Hold (bars)':30s}: {metrics.get('avg_hold_bars', 0):>12.1f}",
        f"  {'Avg Hold (min)':30s}: {metrics.get('avg_hold_minutes', 0):>12.1f}",
        f"  {'Median Hold (bars)':30s}: {metrics.get('median_hold_bars', 0):>12}",
        f"  {'Max Hold (bars)':30s}: {metrics.get('max_hold_bars', 0):>12}",
        f"  {'Min Hold (bars)':30s}: {metrics.get('min_hold_bars', 0):>12}",
        "=" * 60,
    ]
    return "\n".join(lines)
