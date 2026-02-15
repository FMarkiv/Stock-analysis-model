"""
Charting and visualisation utilities using Plotly.

Provides interactive charts for equity model analysis including:
revenue bridges, margin trends, DCF waterfalls, sensitivity heatmaps,
Monte Carlo distributions, scenario comparisons, and football field charts.

Each public chart function returns a :class:`plotly.graph_objects.Figure`
instance that can be rendered in a notebook, saved to HTML, or embedded
in a larger report via :func:`generate_full_report`.

Graceful degradation
--------------------
All chart functions are designed to degrade gracefully when underlying
data is missing or incomplete:

* ``revenue_bridge`` falls back to a year-over-year waterfall when segment
  data is unavailable.
* ``margin_trends`` skips the forecast shading region when no forecast
  periods exist (e.g. the model was never run).
* ``football_field`` tolerates zero or invalid DCF / multiples valuations
  and renders only the methods that produced valid results.
* ``generate_full_report`` wraps every sub-chart in a ``try``/``except``
  so that a single failing visualisation does not abort the whole report.
"""

import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from db.schema import get_connection  # noqa: E402
from model.statements import FinancialModel, _sort_periods, _fy_year  # noqa: E402
from model.dcf import DCFValuation  # noqa: E402
from model.scenarios import run_scenarios, monte_carlo  # noqa: E402

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_DEFAULT_DB = os.path.join(_PROJECT_ROOT, "data", "equity.duckdb")

COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "positive": "#2ca02c",
    "negative": "#d62728",
    "neutral": "#7f7f7f",
    "forecast_bg": "rgba(200, 200, 200, 0.15)",
    "bull": "#2ca02c",
    "base": "#1f77b4",
    "bear": "#d62728",
}

TEMPLATE = "plotly_white"


def _fmt_b(val):
    """Format a numeric value as a dollar amount in billions.

    Parameters
    ----------
    val : float or None
        Value in raw units (e.g. 1_500_000_000).

    Returns
    -------
    str
        Formatted string such as ``"$1.5B"``, or ``""`` if *val* is
        ``None`` or ``NaN``.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    return f"${val / 1e9:.1f}B"


def _resolve_db(db_path):
    """Return *db_path* if provided, otherwise fall back to the project default."""
    return db_path or _DEFAULT_DB


# ---------------------------------------------------------------------------
# 1. revenue_bridge
# ---------------------------------------------------------------------------


def revenue_bridge(ticker, db_path=None, scenario="base"):
    """Build a waterfall chart showing revenue progression.

    The chart walks from the last actual fiscal year through forecast
    years.  When segment-level revenue data is available the waterfall
    decomposes each year-over-year change by segment; otherwise it shows
    simple year-over-year deltas.

    Parameters
    ----------
    ticker : str
        Equity ticker symbol (case-insensitive).
    db_path : str or None
        Path to the DuckDB database.  Defaults to the project-level
        ``data/equity.duckdb``.
    scenario : str
        Forecast scenario name (``"base"``, ``"bull"``, or ``"bear"``).

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive waterfall chart.

    Notes
    -----
    If segment data is empty the function still produces a valid chart
    using total-revenue year-over-year changes only.
    """
    ticker = ticker.upper()
    db = _resolve_db(db_path)

    model = FinancialModel(ticker, db)
    model.compute_historical_metrics()
    model.forecast(years=5, scenario=scenario)

    inc = model.get_statement("income", include_forecast=True, scenario=scenario)
    hist_inc = model.historical.get("income", pd.DataFrame())

    if inc.empty or "total_revenue" not in inc.index:
        fig = go.Figure()
        fig.update_layout(title=f"{ticker}: No revenue data available")
        return fig

    all_periods = _sort_periods(list(inc.columns))
    hist_periods = _sort_periods(list(hist_inc.columns)) if not hist_inc.empty else []
    fc_periods = [p for p in all_periods if p not in hist_periods]

    last_actual = hist_periods[-1] if hist_periods else None
    display_periods = ([last_actual] if last_actual else []) + fc_periods

    revenues = [float(inc.at["total_revenue", p]) for p in display_periods]

    # Check for segment data
    has_segments = False
    seg_all = pd.DataFrame()
    segment_names = []
    try:
        con = get_connection(db)
        try:
            seg_df = con.execute(
                "SELECT period, segment_name, revenue, is_forecast, forecast_scenario "
                "FROM segments WHERE ticker = ? ORDER BY period, segment_name",
                [ticker],
            ).fetchdf()
        finally:
            con.close()

        has_segments = not seg_df.empty

        if has_segments:
            seg_actual = seg_df[~seg_df["is_forecast"]]
            seg_forecast = seg_df[
                seg_df["is_forecast"] & (seg_df["forecast_scenario"] == scenario)
            ]
            seg_all = pd.concat([seg_actual, seg_forecast])
            segment_names = sorted(seg_all["segment_name"].unique())

            # If we got a dataframe back but it contains no rows relevant to
            # our display periods treat it as "no segments".
            relevant = seg_all[seg_all["period"].isin(display_periods)]
            if relevant.empty:
                has_segments = False
                logger.info(
                    "%s: segment table exists but has no rows for the "
                    "display periods; falling back to YoY waterfall.",
                    ticker,
                )
    except Exception:
        has_segments = False
        logger.info(
            "%s: unable to query segment data; falling back to "
            "year-over-year revenue waterfall.",
            ticker,
            exc_info=True,
        )

    if has_segments:
        x_labels, y_values, measures, text_vals = [], [], [], []

        # Starting bar: last actual total
        x_labels.append(display_periods[0])
        y_values.append(revenues[0])
        measures.append("absolute")
        text_vals.append(_fmt_b(revenues[0]))

        for i in range(1, len(display_periods)):
            period = display_periods[i]
            prev_period = display_periods[i - 1]

            curr_segs = seg_all[seg_all["period"] == period]
            prev_segs = seg_all[seg_all["period"] == prev_period]

            if not curr_segs.empty and not prev_segs.empty:
                for seg in segment_names:
                    curr_rev = curr_segs.loc[
                        curr_segs["segment_name"] == seg, "revenue"
                    ]
                    prev_rev = prev_segs.loc[
                        prev_segs["segment_name"] == seg, "revenue"
                    ]
                    curr_val = float(curr_rev.iloc[0]) if len(curr_rev) > 0 else 0
                    prev_val = float(prev_rev.iloc[0]) if len(prev_rev) > 0 else 0
                    delta = curr_val - prev_val

                    x_labels.append(f"{period}<br>{seg}")
                    y_values.append(delta)
                    measures.append("relative")
                    text_vals.append(_fmt_b(delta))
            else:
                delta = revenues[i] - revenues[i - 1]
                x_labels.append(f"{period}<br>Change")
                y_values.append(delta)
                measures.append("relative")
                text_vals.append(_fmt_b(delta))

            # Subtotal for the year
            x_labels.append(f"{period} Total")
            y_values.append(revenues[i])
            measures.append("total")
            text_vals.append(_fmt_b(revenues[i]))

        fig = go.Figure(
            go.Waterfall(
                x=x_labels,
                y=y_values,
                measure=measures,
                text=text_vals,
                textposition="outside",
                connector={"line": {"color": COLORS["neutral"], "width": 1}},
                increasing={"marker": {"color": COLORS["positive"]}},
                decreasing={"marker": {"color": COLORS["negative"]}},
                totals={"marker": {"color": COLORS["primary"]}},
            )
        )
    else:
        # Simple waterfall: year-over-year changes
        x_labels, y_values, measures, text_vals = [], [], [], []

        x_labels.append(f"{display_periods[0]} (Actual)")
        y_values.append(revenues[0])
        measures.append("absolute")
        text_vals.append(_fmt_b(revenues[0]))

        for i in range(1, len(display_periods)):
            delta = revenues[i] - revenues[i - 1]
            pct = delta / revenues[i - 1] if revenues[i - 1] != 0 else 0
            x_labels.append(display_periods[i])
            y_values.append(delta)
            measures.append("relative")
            text_vals.append(f"{_fmt_b(delta)} ({pct:+.1%})")

        x_labels.append(f"{display_periods[-1]} Total")
        y_values.append(revenues[-1])
        measures.append("total")
        text_vals.append(_fmt_b(revenues[-1]))

        fig = go.Figure(
            go.Waterfall(
                x=x_labels,
                y=y_values,
                measure=measures,
                text=text_vals,
                textposition="outside",
                connector={"line": {"color": COLORS["neutral"], "width": 1}},
                increasing={"marker": {"color": COLORS["positive"]}},
                decreasing={"marker": {"color": COLORS["negative"]}},
                totals={"marker": {"color": COLORS["primary"]}},
            )
        )

    fig.update_layout(
        title=f"{ticker} — Revenue Bridge ({scenario.title()} Case)",
        yaxis_title="Revenue (USD)",
        template=TEMPLATE,
        showlegend=False,
        yaxis=dict(tickformat="$.3s"),
        height=500,
    )
    return fig


# ---------------------------------------------------------------------------
# 2. margin_trends
# ---------------------------------------------------------------------------


def margin_trends(ticker, db_path=None, scenario="base"):
    """Line chart of gross, operating, net, and FCF margins.

    Historical and forecast periods are plotted together.  When forecast
    data is present the forecast region is highlighted with a shaded
    background; if no forecast periods exist (e.g. the model was never
    run) the chart is rendered without shading.

    Parameters
    ----------
    ticker : str
        Equity ticker symbol (case-insensitive).
    db_path : str or None
        Path to the DuckDB database.
    scenario : str
        Forecast scenario name.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive line chart.
    """
    ticker = ticker.upper()
    db = _resolve_db(db_path)

    model = FinancialModel(ticker, db)
    model.compute_historical_metrics()
    model.forecast(years=5, scenario=scenario)

    inc = model.get_statement("income", include_forecast=True, scenario=scenario)
    cf = model.get_statement("cashflow", include_forecast=True, scenario=scenario)
    hist_inc = model.historical.get("income", pd.DataFrame())

    if inc.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{ticker}: No data available for margin analysis")
        return fig

    all_periods = _sort_periods(list(inc.columns))
    hist_periods = (
        _sort_periods(list(hist_inc.columns)) if not hist_inc.empty else []
    )

    def _val(df, item, period):
        if df.empty or item not in df.index or period not in df.columns:
            return None
        v = df.at[item, period]
        return float(v) if pd.notna(v) else None

    margin_series = {
        "Gross Margin": [],
        "Operating Margin": [],
        "Net Margin": [],
        "FCF Margin": [],
    }

    for p in all_periods:
        rev = _val(inc, "total_revenue", p)
        gp = _val(inc, "gross_profit", p)
        op = _val(inc, "operating_income", p)
        ni = _val(inc, "net_income", p)
        fcf = _val(cf, "free_cash_flow", p)

        if rev and rev != 0:
            margin_series["Gross Margin"].append(gp / rev if gp is not None else None)
            margin_series["Operating Margin"].append(
                op / rev if op is not None else None
            )
            margin_series["Net Margin"].append(ni / rev if ni is not None else None)
            margin_series["FCF Margin"].append(fcf / rev if fcf is not None else None)
        else:
            for k in margin_series:
                margin_series[k].append(None)

    colors_map = {
        "Gross Margin": "#1f77b4",
        "Operating Margin": "#ff7f0e",
        "Net Margin": "#2ca02c",
        "FCF Margin": "#9467bd",
    }

    fig = go.Figure()
    for name, values in margin_series.items():
        fig.add_trace(
            go.Scatter(
                x=all_periods,
                y=values,
                mode="lines+markers",
                name=name,
                line=dict(color=colors_map[name], width=2),
                hovertemplate="%{x}: %{y:.1%}<extra>" + name + "</extra>",
            )
        )

    # Shade forecast region -- only when forecast data actually exists.
    fc_periods = [p for p in all_periods if p not in hist_periods]
    if hist_periods and fc_periods:
        fig.add_vrect(
            x0=hist_periods[-1],
            x1=all_periods[-1],
            fillcolor=COLORS["forecast_bg"],
            layer="below",
            line_width=0,
            annotation_text="Forecast",
            annotation_position="top right",
        )
    elif not hist_periods:
        logger.info(
            "%s: no historical periods detected; skipping forecast shading.",
            ticker,
        )
    elif not fc_periods:
        logger.info(
            "%s: no forecast periods detected (model may not have been run); "
            "skipping forecast shading.",
            ticker,
        )

    fig.update_layout(
        title=f"{ticker} — Margin Trends ({scenario.title()} Case)",
        yaxis_title="Margin",
        yaxis=dict(tickformat=".0%"),
        template=TEMPLATE,
        hovermode="x unified",
        height=500,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# 3. three_statement_summary
# ---------------------------------------------------------------------------


def three_statement_summary(ticker, db_path=None, scenario="base"):
    """Dashboard-style layout with four panels.

    The panels display Revenue + growth, EPS, FCF, and Net Debt / EBITDA
    across all historical and forecast periods.

    Parameters
    ----------
    ticker : str
        Equity ticker symbol (case-insensitive).
    db_path : str or None
        Path to the DuckDB database.
    scenario : str
        Forecast scenario name.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive multi-panel figure.
    """
    ticker = ticker.upper()
    db = _resolve_db(db_path)

    model = FinancialModel(ticker, db)
    model.compute_historical_metrics()
    model.forecast(years=5, scenario=scenario)

    inc = model.get_statement("income", include_forecast=True, scenario=scenario)
    cf = model.get_statement("cashflow", include_forecast=True, scenario=scenario)
    bal = model.get_statement("balance", include_forecast=True, scenario=scenario)
    hist_inc = model.historical.get("income", pd.DataFrame())

    if inc.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{ticker}: No data available")
        return fig

    all_periods = _sort_periods(list(inc.columns))
    hist_periods = (
        _sort_periods(list(hist_inc.columns)) if not hist_inc.empty else []
    )
    fc_periods = [p for p in all_periods if p not in hist_periods]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Revenue & Revenue Growth",
            "Diluted EPS",
            "Free Cash Flow",
            "Net Debt / EBITDA",
        ),
        specs=[[{"secondary_y": True}, {}], [{}, {}]],
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )

    def _get_vals(df, item):
        out = []
        for p in all_periods:
            if not df.empty and item in df.index and p in df.columns:
                v = df.at[item, p]
                out.append(float(v) if pd.notna(v) else None)
            else:
                out.append(None)
        return out

    bar_colors = [
        COLORS["primary"] if p in hist_periods else COLORS["secondary"]
        for p in all_periods
    ]

    # --- Panel 1: Revenue + Growth ---
    revenues = _get_vals(inc, "total_revenue")
    fig.add_trace(
        go.Bar(
            x=all_periods,
            y=revenues,
            name="Revenue",
            marker_color=bar_colors,
            showlegend=False,
            hovertemplate="%{x}: $%{y:.3s}<extra>Revenue</extra>",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    growth = [None]
    for i in range(1, len(revenues)):
        if (
            revenues[i] is not None
            and revenues[i - 1] is not None
            and revenues[i - 1] != 0
        ):
            growth.append((revenues[i] - revenues[i - 1]) / revenues[i - 1])
        else:
            growth.append(None)

    fig.add_trace(
        go.Scatter(
            x=all_periods,
            y=growth,
            name="YoY Growth",
            mode="lines+markers",
            line=dict(color=COLORS["negative"], width=2),
            showlegend=False,
            hovertemplate="%{x}: %{y:.1%}<extra>YoY Growth</extra>",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    fig.update_yaxes(
        title_text="Revenue", tickformat="$.3s", row=1, col=1, secondary_y=False
    )
    fig.update_yaxes(
        title_text="Growth", tickformat=".0%", row=1, col=1, secondary_y=True
    )

    # --- Panel 2: Diluted EPS ---
    eps = _get_vals(inc, "diluted_eps")
    fig.add_trace(
        go.Bar(
            x=all_periods,
            y=eps,
            name="Diluted EPS",
            marker_color=bar_colors,
            showlegend=False,
            hovertemplate="%{x}: $%{y:.2f}<extra>Diluted EPS</extra>",
        ),
        row=1,
        col=2,
    )
    fig.update_yaxes(title_text="EPS ($)", tickprefix="$", row=1, col=2)

    # --- Panel 3: Free Cash Flow ---
    fcf = _get_vals(cf, "free_cash_flow")
    fig.add_trace(
        go.Bar(
            x=all_periods,
            y=fcf,
            name="Free Cash Flow",
            marker_color=bar_colors,
            showlegend=False,
            hovertemplate="%{x}: $%{y:.3s}<extra>FCF</extra>",
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="FCF", tickformat="$.3s", row=2, col=1)

    # --- Panel 4: Net Debt / EBITDA ---
    ebitda_vals = _get_vals(inc, "ebitda")
    lt_debt = _get_vals(bal, "long_term_debt")
    st_debt = _get_vals(bal, "short_term_debt")
    cash_vals = _get_vals(bal, "cash_and_equivalents")

    nd_ebitda = []
    for i in range(len(all_periods)):
        ltd = lt_debt[i] or 0
        std = st_debt[i] or 0
        c = cash_vals[i] or 0
        eb = ebitda_vals[i]
        if eb and eb != 0:
            nd_ebitda.append((ltd + std - c) / eb)
        else:
            nd_ebitda.append(None)

    fig.add_trace(
        go.Scatter(
            x=all_periods,
            y=nd_ebitda,
            name="Net Debt/EBITDA",
            mode="lines+markers",
            line=dict(color=COLORS["primary"], width=2),
            fill="tozeroy",
            fillcolor="rgba(31, 119, 180, 0.1)",
            showlegend=False,
            hovertemplate="%{x}: %{y:.2f}x<extra>Net Debt/EBITDA</extra>",
        ),
        row=2,
        col=2,
    )
    fig.update_yaxes(title_text="Net Debt / EBITDA (x)", row=2, col=2)

    # Shade forecast region on all panels
    if hist_periods and fc_periods:
        for row in [1, 2]:
            for col in [1, 2]:
                fig.add_vrect(
                    x0=hist_periods[-1],
                    x1=all_periods[-1],
                    fillcolor=COLORS["forecast_bg"],
                    layer="below",
                    line_width=0,
                    row=row,
                    col=col,
                )

    fig.update_layout(
        title=f"{ticker} — Three-Statement Summary ({scenario.title()} Case)",
        template=TEMPLATE,
        height=700,
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# 4. dcf_waterfall
# ---------------------------------------------------------------------------


def dcf_waterfall(ticker, db_path=None, scenario="base"):
    """Waterfall chart of the DCF bridge to equity value.

    Steps: PV of FCFs -> terminal value -> enterprise value -> minus
    debt -> plus cash -> equity value, with an annotation showing the
    implied share price and upside/downside vs the current market price.

    Parameters
    ----------
    ticker : str
        Equity ticker symbol (case-insensitive).
    db_path : str or None
        Path to the DuckDB database.
    scenario : str
        Forecast scenario name.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive waterfall chart.
    """
    ticker = ticker.upper()
    db = _resolve_db(db_path)

    dcf = DCFValuation(ticker, db)
    dcf.compute_wacc()
    result = dcf.dcf_valuation(scenario=scenario)

    pv_fcfs = result["pv_fcfs"]
    pv_tv = (result["pv_terminal_perpetuity"] + result["pv_terminal_exit_multiple"]) / 2
    ev = result["enterprise_value"]
    total_debt = result["net_debt"] + result["cash"]
    cash = result["cash"]
    equity_value = result["equity_value"]
    implied_price = result["implied_price"]

    x_labels = [
        "PV of FCFs",
        "PV of Terminal Value",
        "Enterprise Value",
        "Less: Debt",
        "Plus: Cash",
        "Equity Value",
    ]
    y_values = [pv_fcfs, pv_tv, ev, -total_debt, cash, equity_value]
    measures = ["absolute", "relative", "total", "relative", "relative", "total"]
    text = [
        _fmt_b(pv_fcfs),
        _fmt_b(pv_tv),
        _fmt_b(ev),
        f"-{_fmt_b(total_debt)}",
        f"+{_fmt_b(cash)}",
        _fmt_b(equity_value),
    ]

    fig = go.Figure(
        go.Waterfall(
            x=x_labels,
            y=y_values,
            measure=measures,
            text=text,
            textposition="outside",
            connector={"line": {"color": COLORS["neutral"], "width": 1}},
            increasing={"marker": {"color": COLORS["positive"]}},
            decreasing={"marker": {"color": COLORS["negative"]}},
            totals={"marker": {"color": COLORS["primary"]}},
        )
    )

    upside_color = COLORS["positive"] if result["upside_downside"] >= 0 else COLORS["negative"]
    fig.add_annotation(
        x=0.98,
        y=0.95,
        xref="paper",
        yref="paper",
        text=(
            f"<b>Implied Price: ${implied_price:.2f}</b><br>"
            f"Current Price: ${result['current_price']:.2f}<br>"
            f"<span style='color:{upside_color}'>Upside/Downside: "
            f"{result['upside_downside']:+.1%}</span><br>"
            f"WACC: {result['wacc']:.2%} | TGR: {result['terminal_growth']:.2%}<br>"
            f"Shares: {result['diluted_shares'] / 1e9:.1f}B"
        ),
        showarrow=False,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=COLORS["primary"],
        borderwidth=1,
        font=dict(size=11),
        align="right",
    )

    fig.update_layout(
        title=f"{ticker} — DCF Valuation Waterfall ({scenario.title()} Case)",
        yaxis_title="Value (USD)",
        yaxis=dict(tickformat="$.3s"),
        template=TEMPLATE,
        showlegend=False,
        height=500,
    )
    return fig


# ---------------------------------------------------------------------------
# 5. sensitivity_heatmap
# ---------------------------------------------------------------------------


def sensitivity_heatmap(ticker, db_path=None):
    """Heatmap of the 2-D sensitivity table (terminal growth vs WACC).

    Cells are coloured green when the implied share price exceeds the
    current market price (undervalued) and red when it falls below
    (overvalued).

    Parameters
    ----------
    ticker : str
        Equity ticker symbol (case-insensitive).
    db_path : str or None
        Path to the DuckDB database.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive heatmap.
    """
    ticker = ticker.upper()
    db = _resolve_db(db_path)

    dcf = DCFValuation(ticker, db)
    dcf.compute_wacc()
    current_price = dcf._get_current_price()

    sens = dcf.sensitivity_table(
        variable1="terminal_growth",
        range1=(-0.010, 0.015, 0.005),
        variable2="wacc",
        range2=(-0.020, 0.020, 0.005),
    )

    row_labels = [f"{v:.1%}" for v in sens.index]
    col_labels = [f"{v:.1%}" for v in sens.columns]
    z_values = sens.values
    text_vals = [[f"${v:.0f}" for v in row] for row in z_values]

    fig = go.Figure(
        go.Heatmap(
            z=z_values,
            x=col_labels,
            y=row_labels,
            text=text_vals,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorscale=[
                [0, "#d62728"],
                [0.5, "#ffffbf"],
                [1, "#2ca02c"],
            ],
            zmid=current_price,
            colorbar=dict(title="Implied<br>Price ($)", tickprefix="$"),
            hovertemplate=(
                "Terminal Growth: %{y}<br>"
                "WACC: %{x}<br>"
                "Implied Price: %{text}<br>"
                f"Current Price: ${current_price:.2f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=f"{ticker} — Sensitivity Analysis (Current: ${current_price:.2f})",
        xaxis_title="WACC",
        yaxis_title="Terminal Growth Rate",
        template=TEMPLATE,
        height=500,
        width=750,
    )
    return fig


# ---------------------------------------------------------------------------
# 6. monte_carlo_distribution
# ---------------------------------------------------------------------------


def monte_carlo_distribution(ticker, db_path=None):
    """Histogram of Monte Carlo simulated fair values.

    Vertical lines mark the current market price and key percentiles
    (P10, P25, P50, P75, P90).  A summary annotation shows the mean,
    standard deviation, and the 10th-to-90th percentile range.

    Parameters
    ----------
    ticker : str
        Equity ticker symbol (case-insensitive).
    db_path : str or None
        Path to the DuckDB database.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive histogram.
    """
    ticker = ticker.upper()
    db = _resolve_db(db_path)

    mc = monte_carlo(ticker, db, iterations=10_000)
    values = mc["all_values"]
    pcts = mc["percentiles"]

    dcf_obj = DCFValuation(ticker, db)
    current_price = dcf_obj._get_current_price()

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=values,
            nbinsx=80,
            marker_color=COLORS["primary"],
            opacity=0.7,
            name="Simulated Fair Values",
            hovertemplate="Price: $%{x:.0f}<br>Count: %{y}<extra></extra>",
        )
    )

    # Current price line
    fig.add_vline(
        x=current_price,
        line=dict(color=COLORS["negative"], width=2.5, dash="dash"),
        annotation_text=f"Current: ${current_price:.0f}",
        annotation_position="top",
    )

    # Percentile lines
    pct_config = {
        "p10": {"color": "#d62728", "dash": "dot", "label": "P10"},
        "p25": {"color": "#ff7f0e", "dash": "dot", "label": "P25"},
        "p50": {"color": "#2ca02c", "dash": "solid", "label": "P50"},
        "p75": {"color": "#ff7f0e", "dash": "dot", "label": "P75"},
        "p90": {"color": "#d62728", "dash": "dot", "label": "P90"},
    }

    for key, cfg in pct_config.items():
        pct_val = pcts[key]
        fig.add_vline(
            x=pct_val,
            line=dict(color=cfg["color"], width=1.5, dash=cfg["dash"]),
            annotation_text=f"{cfg['label']}: ${pct_val:.0f}",
            annotation_position=(
                "top right" if key in ("p75", "p90") else "top left"
            ),
        )

    fig.add_annotation(
        x=0.98,
        y=0.92,
        xref="paper",
        yref="paper",
        text=(
            f"<b>Monte Carlo ({mc['iterations']:,} runs)</b><br>"
            f"Mean: ${mc['mean']:.2f}<br>"
            f"Std Dev: ${mc['std']:.2f}<br>"
            f"Median: ${pcts['p50']:.2f}<br>"
            f"10th–90th: ${pcts['p10']:.0f} – ${pcts['p90']:.0f}"
        ),
        showarrow=False,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=COLORS["primary"],
        borderwidth=1,
        font=dict(size=10),
        align="left",
    )

    fig.update_layout(
        title=f"{ticker} — Monte Carlo Fair Value Distribution",
        xaxis_title="Implied Share Price ($)",
        yaxis_title="Frequency",
        xaxis=dict(tickprefix="$"),
        template=TEMPLATE,
        showlegend=False,
        height=500,
    )
    return fig


# ---------------------------------------------------------------------------
# 7. scenario_comparison
# ---------------------------------------------------------------------------


def scenario_comparison(ticker, db_path=None):
    """Grouped bar chart comparing bull / base / bear scenarios.

    Key metrics shown are revenue, EPS, FCF, and implied share price
    for the first forecast year.

    Parameters
    ----------
    ticker : str
        Equity ticker symbol (case-insensitive).
    db_path : str or None
        Path to the DuckDB database.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive grouped-bar chart.
    """
    ticker = ticker.upper()
    db = _resolve_db(db_path)

    scenarios_df = run_scenarios(ticker, db)

    # Determine the first forecast year
    model = FinancialModel(ticker, db)
    model.compute_historical_metrics()
    hist_inc = model.historical.get("income", pd.DataFrame())
    hist_periods = (
        _sort_periods(list(hist_inc.columns)) if not hist_inc.empty else []
    )
    last_year = _fy_year(hist_periods[-1]) if hist_periods else 2024
    y1 = f"FY{last_year + 1}"

    scenario_names = ["bull", "base", "bear"]
    bar_colors = [COLORS["bull"], COLORS["base"], COLORS["bear"]]

    metrics = {
        "Revenue ($B)": [],
        "Diluted EPS ($)": [],
        "FCF ($B)": [],
        "Implied Price ($)": [],
    }

    con = get_connection(db)
    try:
        for sc in scenario_names:
            for item, key in [
                ("total_revenue", "Revenue ($B)"),
                ("diluted_eps", "Diluted EPS ($)"),
                ("free_cash_flow", "FCF ($B)"),
            ]:
                row = con.execute(
                    "SELECT value FROM financials "
                    "WHERE ticker=? AND period=? AND line_item=? "
                    "AND is_forecast=true AND forecast_scenario=?",
                    [ticker, y1, item, sc],
                ).fetchone()
                val = row[0] if row else 0
                if key in ("Revenue ($B)", "FCF ($B)"):
                    val = val / 1e9
                metrics[key].append(val)

            metrics["Implied Price ($)"].append(
                scenarios_df.at[sc, "implied_price"]
                if sc in scenarios_df.index
                else 0
            )
    finally:
        con.close()

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=list(metrics.keys()),
        vertical_spacing=0.18,
        horizontal_spacing=0.12,
    )

    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    tick_fmts = ["$.0f", "$.2f", "$.0f", "$.0f"]

    for idx, (metric_name, values) in enumerate(metrics.items()):
        row, col = positions[idx]
        # Format text based on metric
        if "EPS" in metric_name:
            txt = [f"${v:.2f}" for v in values]
        elif "B)" in metric_name:
            txt = [f"${v:.1f}B" for v in values]
        else:
            txt = [f"${v:.0f}" for v in values]

        fig.add_trace(
            go.Bar(
                x=["Bull", "Base", "Bear"],
                y=values,
                marker_color=bar_colors,
                text=txt,
                textposition="outside",
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title=f"{ticker} — Scenario Comparison (Year 1: {y1})",
        template=TEMPLATE,
        height=600,
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# 8. football_field
# ---------------------------------------------------------------------------


def football_field(ticker, db_path=None):
    """Horizontal bar chart showing valuation range from each method.

    Methods included (when valid data is available):

    * **DCF** -- bear-to-bull implied price range.
    * **Forward P/E** -- point estimate with a +/-15% band.
    * **EV/EBITDA** -- point estimate with a +/-15% band.
    * **P/FCF** -- point estimate with a +/-15% band.
    * **Monte Carlo** -- 10th-to-90th percentile range.

    A vertical dashed line marks the current market price.

    If DCF or any multiples valuation returns zero, negative, ``NaN``,
    or otherwise invalid values, that method is silently omitted and
    the remaining methods are still rendered.

    Parameters
    ----------
    ticker : str
        Equity ticker symbol (case-insensitive).
    db_path : str or None
        Path to the DuckDB database.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive horizontal bar chart.
    """
    ticker = ticker.upper()
    db = _resolve_db(db_path)

    scenarios_df = run_scenarios(ticker, db)

    dcf_obj = DCFValuation(ticker, db)
    dcf_obj.compute_wacc()
    current_price = dcf_obj._get_current_price()

    # Attempt multiples valuation; tolerate failure.
    multiples = None
    try:
        multiples = dcf_obj.multiples_valuation()
    except Exception:
        logger.warning(
            "%s: multiples valuation failed; skipping multiples methods "
            "in football field.",
            ticker,
            exc_info=True,
        )

    # Attempt Monte Carlo; tolerate failure.
    mc = None
    try:
        mc = monte_carlo(ticker, db, iterations=10_000)
    except Exception:
        logger.warning(
            "%s: Monte Carlo simulation failed; skipping Monte Carlo "
            "range in football field.",
            ticker,
            exc_info=True,
        )

    methods, low_vals, mid_vals, high_vals = [], [], [], []

    def _is_valid(value):
        """Return True if *value* is a finite positive number."""
        if value is None:
            return False
        try:
            v = float(value)
        except (TypeError, ValueError):
            return False
        return np.isfinite(v) and v > 0

    # DCF (bear -> bull)
    if "bear" in scenarios_df.index and "bull" in scenarios_df.index:
        bear_price = scenarios_df.at["bear", "implied_price"]
        base_price = scenarios_df.at["base", "implied_price"]
        bull_price = scenarios_df.at["bull", "implied_price"]
        if _is_valid(bear_price) and _is_valid(base_price) and _is_valid(bull_price):
            methods.append("DCF")
            low_vals.append(bear_price)
            mid_vals.append(base_price)
            high_vals.append(bull_price)
        else:
            logger.warning(
                "%s: DCF scenario prices contain invalid values "
                "(bear=%.2f, base=%.2f, bull=%.2f); skipping DCF in "
                "football field.",
                ticker,
                float(bear_price) if bear_price is not None else 0.0,
                float(base_price) if base_price is not None else 0.0,
                float(bull_price) if bull_price is not None else 0.0,
            )

    # Multiples-based methods (+/-15% band around point estimate)
    for label, key in [
        ("Forward P/E", "forward_pe"),
        ("EV/EBITDA", "ev_ebitda"),
        ("P/FCF", "price_fcf"),
    ]:
        if multiples and key in multiples:
            p = multiples[key].get("implied_price", 0)
            if _is_valid(p):
                methods.append(label)
                low_vals.append(p * 0.85)
                mid_vals.append(p)
                high_vals.append(p * 1.15)
            else:
                logger.info(
                    "%s: %s implied price is invalid (%.4f); "
                    "omitting from football field.",
                    ticker,
                    label,
                    float(p) if p is not None else 0.0,
                )

    # Monte Carlo 10th-90th
    if mc is not None:
        mc_p10 = mc["percentiles"]["p10"]
        mc_p50 = mc["percentiles"]["p50"]
        mc_p90 = mc["percentiles"]["p90"]
        if _is_valid(mc_p10) and _is_valid(mc_p50) and _is_valid(mc_p90):
            methods.append("Monte Carlo\n(P10–P90)")
            low_vals.append(mc_p10)
            mid_vals.append(mc_p50)
            high_vals.append(mc_p90)
        else:
            logger.warning(
                "%s: Monte Carlo percentiles contain invalid values; "
                "skipping Monte Carlo in football field.",
                ticker,
            )

    # If no methods produced valid ranges, return an informative empty chart.
    if not methods:
        logger.error(
            "%s: no valuation methods produced valid ranges for the "
            "football field chart.",
            ticker,
        )
        fig = go.Figure()
        fig.update_layout(
            title=f"{ticker} — Football Field Valuation Summary "
                  f"(no valid valuation data)",
            template=TEMPLATE,
            height=350,
        )
        return fig

    fig = go.Figure()

    for i, method in enumerate(methods):
        # Range bar
        fig.add_trace(
            go.Bar(
                y=[method],
                x=[high_vals[i] - low_vals[i]],
                base=[low_vals[i]],
                orientation="h",
                marker_color="rgba(31, 119, 180, 0.3)",
                showlegend=False,
                hovertemplate=(
                    f"<b>{method}</b><br>"
                    f"Low: ${low_vals[i]:.0f}<br>"
                    f"Mid: ${mid_vals[i]:.0f}<br>"
                    f"High: ${high_vals[i]:.0f}"
                    "<extra></extra>"
                ),
            )
        )
        # Mid-point diamond
        fig.add_trace(
            go.Scatter(
                x=[mid_vals[i]],
                y=[method],
                mode="markers",
                marker=dict(
                    color=COLORS["primary"], size=12, symbol="diamond"
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Current price line
    fig.add_vline(
        x=current_price,
        line=dict(color=COLORS["negative"], width=2.5, dash="dash"),
        annotation_text=f"Current: ${current_price:.0f}",
        annotation_position="top",
    )

    fig.update_layout(
        title=f"{ticker} — Football Field Valuation Summary",
        xaxis_title="Share Price ($)",
        xaxis=dict(tickprefix="$"),
        template=TEMPLATE,
        height=max(350, 80 * len(methods) + 120),
        showlegend=False,
        yaxis=dict(autorange="reversed"),
    )
    return fig


# ---------------------------------------------------------------------------
# 9. generate_full_report
# ---------------------------------------------------------------------------


def generate_full_report(ticker, db_path=None):
    """Run all charts and save them as a single interactive HTML report.

    The report opens with a summary section (current price, base-case
    fair value, upside/downside, key assumptions, WACC, top
    sensitivities) followed by all eight interactive Plotly charts.

    Each sub-chart is generated inside its own ``try``/``except`` block
    so that a failure in any single visualisation does not prevent the
    remaining charts from being rendered.  Failures are logged at the
    ``ERROR`` level.

    Parameters
    ----------
    ticker : str
        Equity ticker symbol (case-insensitive).
    db_path : str or None
        Path to the DuckDB database.

    Returns
    -------
    str
        Absolute path to the saved HTML file.
    """
    ticker = ticker.upper()
    db = _resolve_db(db_path)

    logger.info("Generating full report for %s ...", ticker)
    print(f"Generating full report for {ticker}...")

    # ---- Compute model data for the summary section ----
    print("  Computing DCF valuation...")
    dcf = DCFValuation(ticker, db)
    wacc_info = dcf.compute_wacc()
    base_dcf = dcf.dcf_valuation(scenario="base")

    print("  Running scenarios...")
    scenarios_df = run_scenarios(ticker, db)

    print("  Running Monte Carlo simulation...")
    mc = monte_carlo(ticker, db, iterations=10_000)

    print("  Building sensitivity table...")
    sens = dcf.sensitivity_table(
        variable1="terminal_growth",
        range1=(-0.010, 0.015, 0.005),
        variable2="wacc",
        range2=(-0.020, 0.020, 0.005),
    )

    current_price = base_dcf["current_price"]
    fair_value = base_dcf["implied_price"]
    upside = base_dcf["upside_downside"]
    wacc = wacc_info["wacc"]
    assumptions = dcf.model._load_or_generate_assumptions("base")

    sens_range = float(sens.values.max() - sens.values.min())

    con = get_connection(db)
    try:
        company_row = con.execute(
            "SELECT name FROM company WHERE ticker = ?", [ticker]
        ).fetchone()
    finally:
        con.close()
    company_name = company_row[0] if company_row else ticker

    upside_color = "#2ca02c" if upside > 0 else "#d62728"

    # ---- Build scenario rows for the summary table ----
    scenario_rows = ""
    for sc in ["bull", "base", "bear"]:
        if sc in scenarios_df.index:
            r = scenarios_df.loc[sc]
            sc_color = "#2ca02c" if r["upside_downside"] > 0 else "#d62728"
            scenario_rows += (
                f'<tr>'
                f'<td style="padding:8px;border:1px solid #dee2e6;font-weight:bold;">'
                f'{sc.title()}</td>'
                f'<td style="padding:8px;text-align:right;border:1px solid #dee2e6;">'
                f'${r["implied_price"]:.2f}</td>'
                f'<td style="padding:8px;text-align:right;border:1px solid #dee2e6;'
                f'color:{sc_color};">{r["upside_downside"]:+.1%}</td>'
                f'<td style="padding:8px;text-align:right;border:1px solid #dee2e6;">'
                f'{r["wacc"]:.2%}</td>'
                f'</tr>\n'
            )

    summary_html = f"""
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
                max-width:950px;margin:0 auto;padding:20px;">
      <h1 style="border-bottom:3px solid #1f77b4;padding-bottom:10px;">
        {company_name} ({ticker}) — Equity Research Report
      </h1>
      <p style="color:#666;font-size:12px;">
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
      </p>

      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:15px;margin:20px 0;">
        <div style="background:#f8f9fa;padding:15px;border-radius:8px;
                    border-left:4px solid #1f77b4;">
          <div style="font-size:12px;color:#666;">Current Price</div>
          <div style="font-size:24px;font-weight:bold;">${current_price:.2f}</div>
        </div>
        <div style="background:#f8f9fa;padding:15px;border-radius:8px;
                    border-left:4px solid {upside_color};">
          <div style="font-size:12px;color:#666;">Base Case Fair Value</div>
          <div style="font-size:24px;font-weight:bold;">${fair_value:.2f}</div>
        </div>
        <div style="background:#f8f9fa;padding:15px;border-radius:8px;
                    border-left:4px solid {upside_color};">
          <div style="font-size:12px;color:#666;">Upside / Downside</div>
          <div style="font-size:24px;font-weight:bold;color:{upside_color};">
            {upside:+.1%}
          </div>
        </div>
      </div>

      <h3>Key Assumptions</h3>
      <table style="border-collapse:collapse;width:100%;margin-bottom:20px;">
        <tr style="background:#f1f3f5;">
          <th style="padding:8px;text-align:left;border:1px solid #dee2e6;">Parameter</th>
          <th style="padding:8px;text-align:right;border:1px solid #dee2e6;">Value</th>
        </tr>
        <tr>
          <td style="padding:8px;border:1px solid #dee2e6;">WACC</td>
          <td style="padding:8px;text-align:right;border:1px solid #dee2e6;">{wacc:.2%}</td>
        </tr>
        <tr>
          <td style="padding:8px;border:1px solid #dee2e6;">Terminal Growth Rate</td>
          <td style="padding:8px;text-align:right;border:1px solid #dee2e6;">
            {base_dcf['terminal_growth']:.2%}</td>
        </tr>
        <tr>
          <td style="padding:8px;border:1px solid #dee2e6;">Revenue Growth (Base)</td>
          <td style="padding:8px;text-align:right;border:1px solid #dee2e6;">
            {assumptions.get('revenue_growth', 0):.2%}</td>
        </tr>
        <tr>
          <td style="padding:8px;border:1px solid #dee2e6;">Operating Margin</td>
          <td style="padding:8px;text-align:right;border:1px solid #dee2e6;">
            {assumptions.get('operating_margin', 0):.2%}</td>
        </tr>
        <tr>
          <td style="padding:8px;border:1px solid #dee2e6;">Exit EBITDA Multiple</td>
          <td style="padding:8px;text-align:right;border:1px solid #dee2e6;">
            {base_dcf['exit_multiple']:.1f}x</td>
        </tr>
        <tr>
          <td style="padding:8px;border:1px solid #dee2e6;">Beta</td>
          <td style="padding:8px;text-align:right;border:1px solid #dee2e6;">
            {wacc_info['beta']:.2f}</td>
        </tr>
      </table>

      <h3>Scenario Summary</h3>
      <table style="border-collapse:collapse;width:100%;margin-bottom:20px;">
        <tr style="background:#f1f3f5;">
          <th style="padding:8px;text-align:left;border:1px solid #dee2e6;">Scenario</th>
          <th style="padding:8px;text-align:right;border:1px solid #dee2e6;">Implied Price</th>
          <th style="padding:8px;text-align:right;border:1px solid #dee2e6;">Upside/Downside</th>
          <th style="padding:8px;text-align:right;border:1px solid #dee2e6;">WACC</th>
        </tr>
        {scenario_rows}
      </table>

      <h3>Top 3 Sensitivities</h3>
      <ol style="margin-bottom:20px;">
        <li><b>WACC:</b> Full sensitivity grid spans ${sens_range:.0f} in implied price</li>
        <li><b>Terminal Growth Rate:</b> Ranged from
            {base_dcf['terminal_growth'] - 0.01:.1%} to
            {base_dcf['terminal_growth'] + 0.015:.1%}</li>
        <li><b>Monte Carlo (P10–P90):</b>
            ${mc['percentiles']['p10']:.0f} – ${mc['percentiles']['p90']:.0f}</li>
      </ol>

      <hr style="border:1px solid #dee2e6;margin:30px 0;">
    </div>
    """

    # ---- Generate all charts ----
    chart_specs = [
        ("Valuation Summary — Football Field", football_field),
        ("DCF Valuation Bridge", dcf_waterfall),
        ("Three-Statement Summary", three_statement_summary),
        ("Revenue Bridge", revenue_bridge),
        ("Margin Trends", margin_trends),
        ("Sensitivity Analysis", sensitivity_heatmap),
        ("Monte Carlo Simulation", monte_carlo_distribution),
        ("Scenario Comparison", scenario_comparison),
    ]

    chart_divs = []
    for i, (title, fn) in enumerate(chart_specs, 1):
        print(f"  Generating chart {i}/{len(chart_specs)}: {title}...")
        try:
            fig = fn(ticker, db)
            div_html = fig.to_html(full_html=False, include_plotlyjs=False)
            chart_divs.append(
                f'<div style="max-width:960px;margin:20px auto;padding:0 20px;">\n'
                f'  <h2 style="font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\','
                f"Roboto,sans-serif;color:#333;"
                f'border-bottom:2px solid #e9ecef;padding-bottom:8px;">{title}</h2>\n'
                f"  {div_html}\n"
                f"</div>\n"
            )
        except Exception:
            logger.error(
                "Chart '%s' for %s failed; skipping.",
                title,
                ticker,
                exc_info=True,
            )
            chart_divs.append(
                f'<div style="max-width:960px;margin:20px auto;padding:0 20px;">\n'
                f'  <h2 style="font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\','
                f"Roboto,sans-serif;color:#999;"
                f'border-bottom:2px solid #e9ecef;padding-bottom:8px;">'
                f"{title} (unavailable)</h2>\n"
                f'  <p style="color:#999;">This chart could not be generated. '
                f"Check the log for details.</p>\n"
                f"</div>\n"
            )

    full_html = (
        "<!DOCTYPE html>\n<html>\n<head>\n"
        '  <meta charset="utf-8">\n'
        f"  <title>{ticker} Equity Research Report</title>\n"
        '  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>\n'
        "</head>\n<body style='margin:0;padding:0;background:#fff;'>\n"
        f"{summary_html}\n"
        + "\n".join(chart_divs)
        + "\n<div style='text-align:center;padding:30px;color:#999;font-size:11px;"
        "font-family:-apple-system,BlinkMacSystemFont,sans-serif;'>"
        f"Generated by Equity Model &middot; {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        "</div>\n</body>\n</html>"
    )

    output_dir = os.path.join(_PROJECT_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{ticker}_report.html")

    with open(output_path, "w") as f:
        f.write(full_html)

    print(f"\nReport saved to: {output_path}")
    logger.info("Report saved to: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate equity research charts")
    parser.add_argument("ticker", nargs="?", default="AAPL", help="Ticker symbol")
    parser.add_argument("--db", default=None, help="Path to DuckDB file")
    args = parser.parse_args()

    generate_full_report(args.ticker, args.db)
