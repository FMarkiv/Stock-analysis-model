"""
Scenario management: run bull / base / bear DCF valuations and
Monte Carlo simulation.

This module provides two primary entry points:

* :func:`run_scenarios` — deterministic bull / base / bear analysis
  that shifts key assumptions by fixed amounts and returns a
  comparison table of implied equity prices.

* :func:`monte_carlo` — stochastic simulation that samples assumptions
  from statistical distributions centred on the base case and produces
  a probability-weighted range of fair-value estimates.

Both functions persist their intermediate results (assumptions,
forecasts) to the project DuckDB so downstream notebooks / dashboards
can query them directly.
"""

import logging
import os
import sys

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from db.schema import get_connection  # noqa: E402
from model.dcf import DCFValuation, _load_config  # noqa: E402
from model.statements import FinancialModel, _sort_periods, _fy_year  # noqa: E402

# ---------------------------------------------------------------------------
# Scenario offset constants
# ---------------------------------------------------------------------------
# These constants define how much each scenario shifts the base-case
# assumptions.  They are expressed in *percentage-point* terms (i.e. 0.02
# means "+2 pp").  Changing them here will affect every call to
# ``run_scenarios``.

# Bull scenario: optimistic revenue growth, higher margins, lower WACC
BULL_REVENUE_GROWTH_OFFSET = 0.02      # +2 pp revenue growth
BULL_OPERATING_MARGIN_OFFSET = 0.01    # +1 pp operating margin
BULL_WACC_OFFSET = -0.01              # -1 pp WACC (lower discount rate)

# Base scenario: no adjustments (use auto-generated assumptions as-is)
BASE_REVENUE_GROWTH_OFFSET = 0.0
BASE_OPERATING_MARGIN_OFFSET = 0.0
BASE_WACC_OFFSET = 0.0

# Bear scenario: lower growth, compressed margins, higher WACC
BEAR_REVENUE_GROWTH_OFFSET = -0.03    # -3 pp revenue growth
BEAR_OPERATING_MARGIN_OFFSET = -0.015  # -1.5 pp operating margin
BEAR_WACC_OFFSET = 0.015             # +1.5 pp WACC (higher discount rate)


# ---------------------------------------------------------------------------
# 1. run_scenarios
# ---------------------------------------------------------------------------


def run_scenarios(ticker: str, db_path: str | None = None) -> pd.DataFrame:
    """Run bull, base, and bear DCF valuations and return a comparison table.

    For each of the three scenarios the function:

    1. Adjusts the base-case assumptions by the predefined offsets
       (see module-level constants ``BULL_*``, ``BASE_*``, ``BEAR_*``).
    2. Writes the adjusted assumptions to the database and generates a
       new forecast.
    3. Runs (or re-discounts) a DCF valuation with the scenario's WACC.
    4. Collects the implied share price and upside/downside into a row.

    **Scenario definitions (relative to auto-generated base):**

    * **Bull** -- revenue growth +2 pp, operating margin +1 pp,
      WACC -1 pp.
    * **Base** -- auto-generated assumptions (from FinancialModel).
    * **Bear** -- revenue growth -3 pp, operating margin -1.5 pp,
      WACC +1.5 pp.

    Each scenario's assumptions and forecasts are stored in the
    ``assumptions`` and ``financials`` tables so they can be queried
    later.

    Error handling
    ~~~~~~~~~~~~~~
    Each scenario is wrapped in its own ``try / except``.  If bull or
    bear fails (e.g. negative earnings break the DCF), a warning is
    logged and the remaining scenarios still execute.  When
    ``dcf_valuation`` returns ``dcf_valid == False`` (e.g. all projected
    FCFs are negative), the row is still included but the
    ``dcf_valid`` column is set to ``False`` so callers can filter or
    flag it.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    db_path : str, optional
        Path to DuckDB file.  Defaults to ``<project>/data/equity.duckdb``.

    Returns
    -------
    pd.DataFrame
        Columns: ``scenario``, ``implied_price``, ``upside_downside``,
        ``revenue_growth``, ``operating_margin``, ``wacc``,
        ``dcf_valid``.
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

    # --- Define scenario adjustments using named constants ---
    adjustments = {
        "bull": {
            "revenue_growth": base_rev_growth + BULL_REVENUE_GROWTH_OFFSET,
            "operating_margin": base_op_margin + BULL_OPERATING_MARGIN_OFFSET,
            "wacc_delta": BULL_WACC_OFFSET,
        },
        "base": {
            "revenue_growth": base_rev_growth + BASE_REVENUE_GROWTH_OFFSET,
            "operating_margin": base_op_margin + BASE_OPERATING_MARGIN_OFFSET,
            "wacc_delta": BASE_WACC_OFFSET,
        },
        "bear": {
            "revenue_growth": base_rev_growth + BEAR_REVENUE_GROWTH_OFFSET,
            "operating_margin": base_op_margin + BEAR_OPERATING_MARGIN_OFFSET,
            "wacc_delta": BEAR_WACC_OFFSET,
        },
    }

    rows: list[dict] = []

    for scenario_name, adj in adjustments.items():
        try:
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

            # Check whether the DCF result is flagged as invalid
            dcf_valid = base_dcf_result.get("dcf_valid", True)
            if not dcf_valid:
                logger.warning(
                    "%s scenario for %s: DCF flagged as invalid (%s). "
                    "Row included but marked dcf_valid=False.",
                    scenario_name,
                    ticker,
                    base_dcf_result.get("dcf_warning", "unknown reason"),
                )

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
                "dcf_valid": dcf_valid,
            })

        except Exception:
            logger.warning(
                "Scenario '%s' for %s failed; skipping.",
                scenario_name,
                ticker,
                exc_info=True,
            )

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
    the base case and runs a lightweight DCF for each draw, producing a
    probability-weighted distribution of implied share prices.

    **Distributions used:**

    * Revenue growth -- Normal(base, 2 pp std).
    * Operating margin -- Triangular(base - 3 pp, base, base + 2 pp).
    * Terminal growth -- Normal(base, 0.5 pp std), clipped to [0 %, 4 %].
    * WACC -- Normal(base, 1 pp std), clipped to [4 %, 20 %].

    Negative / zero prices
    ~~~~~~~~~~~~~~~~~~~~~~
    Some iterations may produce negative or zero implied prices (e.g.
    when a loss-making company's projected FCFs are deeply negative).
    These iterations are **excluded** from the summary statistics
    (mean, std, percentiles) but are still present in the raw
    ``all_values`` array.  The return dict includes
    ``negative_iterations`` reporting how many draws produced a
    non-positive price.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    db_path : str, optional
        DuckDB path.  Defaults to ``<project>/data/equity.duckdb``.
    iterations : int
        Number of simulation runs (default 10 000).

    Returns
    -------
    dict
        ``percentiles`` -- dict mapping percentile labels to fair-value
        estimates (10th, 25th, 50th, 75th, 90th).
        ``all_values`` -- numpy array of *all* simulated fair values
        (including negatives; suitable for histogram plotting).
        ``positive_values`` -- numpy array containing only the
        strictly-positive simulated fair values used for statistics.
        ``mean``, ``std`` -- summary statistics (computed on positive
        values only).
        ``iterations`` -- total number of iterations requested.
        ``negative_iterations`` -- count of iterations that produced a
        zero or negative implied price.
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

    # --- Filter out negative / zero prices for statistics ---
    positive_mask = fair_values > 0
    positive_values = fair_values[positive_mask]
    negative_count = int(np.sum(~positive_mask))

    if negative_count > 0:
        logger.warning(
            "Monte Carlo for %s: %d of %d iterations (%.1f%%) produced "
            "zero or negative implied prices; these are excluded from "
            "summary statistics.",
            ticker,
            negative_count,
            iterations,
            100.0 * negative_count / iterations,
        )

    # --- Compute percentiles on positive values only ---
    if len(positive_values) > 0:
        pcts = {
            "p10": float(np.percentile(positive_values, 10)),
            "p25": float(np.percentile(positive_values, 25)),
            "p50": float(np.percentile(positive_values, 50)),
            "p75": float(np.percentile(positive_values, 75)),
            "p90": float(np.percentile(positive_values, 90)),
        }
        mean_val = float(np.mean(positive_values))
        std_val = float(np.std(positive_values))
    else:
        logger.warning(
            "Monte Carlo for %s: all %d iterations produced zero or "
            "negative prices. Statistics will be zero.",
            ticker,
            iterations,
        )
        pcts = {"p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0}
        mean_val = 0.0
        std_val = 0.0

    return {
        "percentiles": pcts,
        "mean": mean_val,
        "std": std_val,
        "all_values": fair_values,
        "positive_values": positive_values,
        "iterations": iterations,
        "negative_iterations": negative_count,
    }
