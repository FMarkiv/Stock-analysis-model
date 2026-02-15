"""
Scenario management: run bull / base / bear DCF valuations and
Monte Carlo simulation.
"""

import os
import sys

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from db.schema import get_connection  # noqa: E402
from model.dcf import DCFValuation, _load_config  # noqa: E402
from model.statements import FinancialModel, _sort_periods, _fy_year  # noqa: E402


# ---------------------------------------------------------------------------
# 1. run_scenarios
# ---------------------------------------------------------------------------


def run_scenarios(ticker: str, db_path: str | None = None) -> pd.DataFrame:
    """Run bull, base, and bear DCF valuations.

    **Scenario definitions (relative to auto-generated base):**

    * **Bull** — revenue growth +2 pp, operating margin +1 pp,
      WACC -1 pp.
    * **Base** — auto-generated assumptions (from FinancialModel).
    * **Bear** — revenue growth -3 pp, operating margin -1.5 pp,
      WACC +1.5 pp.

    Each scenario's assumptions and forecasts are stored in the
    ``assumptions`` and ``financials`` tables so they can be queried
    later.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    db_path : str, optional
        Path to DuckDB file.

    Returns
    -------
    pd.DataFrame
        Columns: ``scenario``, ``implied_price``, ``upside_downside``,
        ``revenue_growth``, ``operating_margin``, ``wacc``.
    """
    ticker = ticker.upper()
    db = db_path or os.path.join(_PROJECT_ROOT, "data", "equity.duckdb")

    # Build the DCF model (auto-generates base forecast + assumptions)
    dcf = DCFValuation(ticker, db)
    wacc_info = dcf.compute_wacc()
    base_wacc = wacc_info["wacc"]

    # Read base assumptions that were auto-generated
    model = dcf.model
    base_assumptions = model._load_or_generate_assumptions("base")
    base_rev_growth = base_assumptions.get("revenue_growth", 0.05)
    base_op_margin = base_assumptions.get("operating_margin", 0.20)

    # --- Define scenario adjustments ---
    adjustments = {
        "bull": {
            "revenue_growth": base_rev_growth + 0.02,
            "operating_margin": base_op_margin + 0.01,
            "wacc_delta": -0.01,
        },
        "base": {
            "revenue_growth": base_rev_growth,
            "operating_margin": base_op_margin,
            "wacc_delta": 0.0,
        },
        "bear": {
            "revenue_growth": base_rev_growth - 0.03,
            "operating_margin": base_op_margin - 0.015,
            "wacc_delta": 0.015,
        },
    }

    rows: list[dict] = []

    for scenario_name, adj in adjustments.items():
        # Write scenario assumptions to DB
        if scenario_name != "base":
            # Copy all base assumptions, then override
            scenario_assumptions = dict(base_assumptions)
            scenario_assumptions["revenue_growth"] = adj["revenue_growth"]
            scenario_assumptions["operating_margin"] = adj["operating_margin"]
            model._save_assumptions(scenario_name, scenario_assumptions)

            # Generate forecast with these assumptions
            dcf._forecasts[scenario_name] = model.forecast(
                years=5, scenario=scenario_name,
            )

        # Run DCF with possibly adjusted WACC
        base_dcf_result = dcf.dcf_valuation(scenario=scenario_name)

        # If WACC adjustment, re-discount with shifted WACC
        if adj["wacc_delta"] != 0:
            adjusted_wacc = base_wacc + adj["wacc_delta"]
            implied_price = dcf._quick_dcf(
                base_dcf=base_dcf_result,
                wacc=adjusted_wacc,
            )
            current_price = base_dcf_result["current_price"]
            upside = (
                (implied_price / current_price - 1)
                if current_price > 0
                else 0.0
            )
        else:
            implied_price = base_dcf_result["implied_price"]
            upside = base_dcf_result["upside_downside"]
            adjusted_wacc = base_wacc

        rows.append({
            "scenario": scenario_name,
            "implied_price": round(implied_price, 2),
            "upside_downside": round(upside, 4),
            "revenue_growth": round(adj["revenue_growth"], 4),
            "operating_margin": round(adj["operating_margin"], 4),
            "wacc": round(adjusted_wacc, 4),
        })

    return pd.DataFrame(rows).set_index("scenario")


# ---------------------------------------------------------------------------
# 2. monte_carlo
# ---------------------------------------------------------------------------


def monte_carlo(
    ticker: str,
    db_path: str | None = None,
    iterations: int = 10_000,
) -> dict:
    """Monte Carlo simulation of fair value.

    Samples key assumptions from statistical distributions centred on
    the base case and runs a lightweight DCF for each draw.

    **Distributions used:**

    * Revenue growth — Normal(base, 2 pp std).
    * Operating margin — Triangular(base - 3 pp, base, base + 2 pp).
    * Terminal growth — Normal(base, 0.5 pp std), clipped to [0 %, 4 %].
    * WACC — Normal(base, 1 pp std), clipped to [4 %, 20 %].

    Parameters
    ----------
    ticker : str
        Stock ticker.
    db_path : str, optional
        DuckDB path.
    iterations : int
        Number of simulation runs (default 10 000).

    Returns
    -------
    dict
        ``percentiles`` — dict mapping percentile labels to fair-value
        estimates (10th, 25th, 50th, 75th, 90th).
        ``all_values`` — numpy array of all simulated fair values
        (suitable for histogram plotting).
        ``mean``, ``std`` — summary statistics.
    """
    ticker = ticker.upper()
    db = db_path or os.path.join(_PROJECT_ROOT, "data", "equity.duckdb")

    dcf = DCFValuation(ticker, db)
    wacc_info = dcf.compute_wacc()
    base_dcf = dcf.dcf_valuation(scenario="base")

    base_assumptions = dcf.model._load_or_generate_assumptions("base")
    base_rev_growth = base_assumptions.get("revenue_growth", 0.05)
    base_op_margin = base_assumptions.get("operating_margin", 0.20)
    base_wacc = wacc_info["wacc"]
    base_tg = dcf.model_params.get("terminal_growth_rate", 0.025)
    tax_rate = wacc_info["tax_rate"]

    # Retrieve last historical values for simplified forecast
    inc = dcf.model.historical.get("income", pd.DataFrame())
    cf = dcf.model.historical.get("cashflow", pd.DataFrame())
    bal = dcf.model.historical.get("balance", pd.DataFrame())

    last_revenue = 0.0
    da_pct = base_assumptions.get("da_pct_revenue", 0.03)
    capex_pct = base_assumptions.get("capex_pct_revenue", 0.05)

    if not inc.empty and "total_revenue" in inc.index:
        rev_s = inc.loc["total_revenue"].dropna()
        if len(rev_s) > 0:
            last_revenue = float(rev_s.iloc[-1])

    bs = dcf._get_balance_sheet_items()
    net_debt = bs["net_debt"]
    shares = bs["diluted_shares"]

    # Pre-compute the base EBITDA for exit-multiple TV
    forecast = dcf._forecasts.get("base")
    base_ebitda_terminal = 0.0
    exit_multiple = dcf.model_params.get("exit_ebitda_multiple", 12.0)
    if forecast is not None and "ebitda" in forecast.index:
        last_period = _sort_periods(list(forecast.columns))[-1]
        base_ebitda_terminal = float(forecast.at["ebitda", last_period])

    # Long-run GDP growth for revenue-growth decay target
    gdp_growth = 0.025
    n_years = 5

    rng = np.random.default_rng(42)

    # --- Sample all parameters at once (vectorised) ---
    rev_growth_samples = rng.normal(base_rev_growth, 0.02, iterations)
    op_margin_samples = rng.triangular(
        max(base_op_margin - 0.03, 0.01),
        base_op_margin,
        base_op_margin + 0.02,
        iterations,
    )
    tg_samples = np.clip(rng.normal(base_tg, 0.005, iterations), 0.0, 0.04)
    wacc_samples = np.clip(rng.normal(base_wacc, 0.01, iterations), 0.04, 0.20)

    fair_values = np.empty(iterations)

    for i in range(iterations):
        rev_g = rev_growth_samples[i]
        op_m = op_margin_samples[i]
        tg = tg_samples[i]
        wacc = wacc_samples[i]

        # Simplified 5-year FCF projection
        pv_fcfs = 0.0
        revenue = last_revenue
        final_fcf = 0.0
        ebitda_terminal = 0.0

        for yr in range(1, n_years + 1):
            # Revenue growth decays toward GDP growth
            decay = max(0, 1 - (yr - 1) / n_years)
            g = rev_g * decay + gdp_growth * (1 - decay)
            revenue *= (1 + g)

            ebit = revenue * op_m
            da = revenue * da_pct
            capex = revenue * capex_pct
            # Simplified: assume WC change ≈ 0 on average
            ufcf = ebit * (1 - tax_rate) + da - capex

            ebitda_terminal = ebit + da
            final_fcf = ufcf
            pv_fcfs += ufcf / (1 + wacc) ** yr

        # Terminal value (average of both methods)
        if wacc > tg:
            tv_perp = final_fcf * (1 + tg) / (wacc - tg)
        else:
            tv_perp = final_fcf * 25
        pv_tv_perp = tv_perp / (1 + wacc) ** n_years

        tv_exit = ebitda_terminal * exit_multiple
        pv_tv_exit = tv_exit / (1 + wacc) ** n_years

        ev = pv_fcfs + (pv_tv_perp + pv_tv_exit) / 2
        equity = ev - net_debt
        fair_values[i] = equity / shares if shares > 0 else 0.0

    # --- Compute percentiles ---
    pcts = {
        "p10": float(np.percentile(fair_values, 10)),
        "p25": float(np.percentile(fair_values, 25)),
        "p50": float(np.percentile(fair_values, 50)),
        "p75": float(np.percentile(fair_values, 75)),
        "p90": float(np.percentile(fair_values, 90)),
    }

    return {
        "percentiles": pcts,
        "mean": float(np.mean(fair_values)),
        "std": float(np.std(fair_values)),
        "all_values": fair_values,
        "iterations": iterations,
    }
