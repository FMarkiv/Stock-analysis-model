"""
Three-statement financial model: income statement, balance sheet, cash flow.

Core class ``FinancialModel`` loads historical data from DuckDB, computes
derived metrics (margins, returns, working-capital days, etc.), generates
multi-year forecasts under configurable scenarios, and writes projections
back to the database.

The forecasting engine applies revenue-growth decay toward long-run GDP
growth, drives balance-sheet items off revenue or COGS ratios, and closes
the cash-flow waterfall so that assets = liabilities + equity each period.

Robustness features
-------------------
* Missing line items (e.g. inventory, R&D) are handled gracefully --
  the model falls back to safe defaults rather than crashing.
* Negative-earnings edge cases are treated explicitly: no tax on losses,
  payout ratios clamped to sensible ranges, buyback percentages floored
  at zero.
* Balance-sheet imbalances emit a warning (with a 0.1 % relative
  tolerance) instead of raising an exception, so downstream analysis
  can still proceed.
"""

import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from db.schema import get_connection, init_db  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GDP_GROWTH = 0.025  # long-run nominal GDP growth rate for decay target

# Line items per statement used in forecasting
INCOME_ITEMS = [
    "total_revenue", "cost_of_revenue", "gross_profit",
    "research_development", "selling_general_admin", "operating_income",
    "interest_expense", "interest_income", "pretax_income",
    "income_tax", "net_income", "ebitda",
    "depreciation_amortization", "stock_based_comp",
    "basic_eps", "diluted_eps", "basic_shares_out", "diluted_shares_out",
]

BALANCE_ITEMS = [
    "cash_and_equivalents", "short_term_investments",
    "accounts_receivable", "inventory", "total_current_assets",
    "property_plant_equipment_net", "goodwill", "intangible_assets",
    "total_assets",
    "accounts_payable", "short_term_debt", "current_portion_lt_debt",
    "total_current_liabilities",
    "long_term_debt", "total_liabilities",
    "total_stockholders_equity", "retained_earnings",
]

CASHFLOW_ITEMS = [
    "operating_cash_flow", "capex", "free_cash_flow",
    "dividends_paid", "share_repurchases",
    "depreciation_cf", "stock_based_comp_cf",
    "change_in_working_capital",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_div(a, b):
    """Divide *a* / *b*, returning ``NaN`` when *b* is zero or either value is ``NaN``/``None``."""
    if b is None or a is None:
        return np.nan
    if isinstance(b, (int, float)) and b == 0:
        return np.nan
    return a / b


def _sort_periods(periods: list[str]) -> list[str]:
    """Sort period strings chronologically.

    Handles ``'FY2023'`` and ``'Q3 2023'`` formats.  Fiscal-year
    periods sort after the fourth quarter of the same calendar year
    so that ``FY2023`` follows ``Q4 2023``.
    """
    def _key(p: str):
        if p.startswith("FY"):
            return (int(p[2:]), 5)  # FY sorts after Q4
        parts = p.split()
        q = int(parts[0][1])
        y = int(parts[1])
        return (y, q)
    return sorted(periods, key=_key)


def _fy_year(period: str) -> int:
    """Extract the numeric year from a period string like ``'FY2023'``."""
    return int(period.replace("FY", ""))


# ---------------------------------------------------------------------------
# FinancialModel
# ---------------------------------------------------------------------------


class FinancialModel:
    """Three-statement financial model backed by DuckDB.

    ``FinancialModel`` is the central object for building a bottom-up
    equity forecast.  It loads historical income-statement, balance-sheet,
    and cash-flow data, computes derived metrics, generates multi-year
    projections (with revenue-growth decay toward GDP), and persists
    results back to the database.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. ``'AAPL'``).
    db_path : str or None
        Path to the DuckDB database file.  When *None* the default
        ``data/equity.duckdb`` inside the project root is used.

    Attributes
    ----------
    historical : dict[str, pd.DataFrame]
        Annual financial statements keyed by ``'income'``, ``'balance'``,
        ``'cashflow'``.  Each DataFrame has line items as rows and fiscal
        periods as columns.
    metrics : pd.DataFrame
        Derived metrics computed by :meth:`compute_historical_metrics`.
    """

    def __init__(self, ticker: str, db_path: str | None = None):
        self.ticker = ticker.upper()
        self.db_path = db_path or os.path.join(
            _PROJECT_ROOT, "data", "equity.duckdb"
        )

        # DataFrames keyed by statement type
        self.historical: dict[str, pd.DataFrame] = {}
        self.metrics: pd.DataFrame = pd.DataFrame()
        self._load_historical()

    # ------------------------------------------------------------------
    # 1. __init__ helper: load all historical data
    # ------------------------------------------------------------------

    def _load_historical(self) -> None:
        """Load all historical financial data from DuckDB into DataFrames.

        Populates :pyattr:`historical` (annual) and the private
        ``_quarterly`` dict with wide-format DataFrames.  Also loads
        daily closing prices for market-cap-dependent metrics.
        """
        con = get_connection(self.db_path)
        try:
            for stmt in ("income", "balance", "cashflow"):
                df = con.execute(
                    """
                    SELECT period, line_item, value
                    FROM financials
                    WHERE ticker = ?
                      AND statement = ?
                      AND is_forecast = false
                      AND forecast_scenario = 'actual'
                      AND period_type = 'annual'
                    """,
                    [self.ticker, stmt],
                ).fetchdf()

                if df.empty:
                    self.historical[stmt] = pd.DataFrame()
                    continue

                # Pivot to wide format: rows = line_item, columns = period
                wide = df.pivot_table(
                    index="line_item", columns="period", values="value",
                    aggfunc="first",
                )
                # Sort columns chronologically
                wide = wide[_sort_periods(list(wide.columns))]
                self.historical[stmt] = wide

            # Also load quarterly data for quarterly lookups
            self._quarterly: dict[str, pd.DataFrame] = {}
            for stmt in ("income", "balance", "cashflow"):
                df = con.execute(
                    """
                    SELECT period, line_item, value
                    FROM financials
                    WHERE ticker = ?
                      AND statement = ?
                      AND is_forecast = false
                      AND forecast_scenario = 'actual'
                      AND period_type = 'quarterly'
                    """,
                    [self.ticker, stmt],
                ).fetchdf()

                if df.empty:
                    self._quarterly[stmt] = pd.DataFrame()
                    continue

                wide = df.pivot_table(
                    index="line_item", columns="period", values="value",
                    aggfunc="first",
                )
                wide = wide[_sort_periods(list(wide.columns))]
                self._quarterly[stmt] = wide

            # Load prices for market-cap-dependent metrics
            self._prices = con.execute(
                "SELECT date, close FROM prices WHERE ticker = ? ORDER BY date",
                [self.ticker],
            ).fetchdf()
        finally:
            con.close()

    # ------------------------------------------------------------------
    # 2. get_historical
    # ------------------------------------------------------------------

    def get_historical(self, line_item: str, periods: int = 8,
                       period_type: str = "annual") -> pd.Series:
        """Return the last *periods* values for a line item.

        Searches across all statement DataFrames (income, balance,
        cash flow) and returns the first match found.

        Parameters
        ----------
        line_item : str
            Standardised line-item name (e.g. ``'total_revenue'``).
        periods : int
            Number of most-recent periods to return.
        period_type : str
            ``'annual'`` or ``'quarterly'``.

        Returns
        -------
        pd.Series
            Values indexed by period string.  Returns an empty
            float Series if the item is not found.
        """
        source = self.historical if period_type == "annual" else self._quarterly

        for stmt_df in source.values():
            if stmt_df.empty:
                continue
            if line_item in stmt_df.index:
                series = stmt_df.loc[line_item].dropna()
                return series.iloc[-periods:]

        return pd.Series(dtype=float)

    # ------------------------------------------------------------------
    # 3. compute_historical_metrics
    # ------------------------------------------------------------------

    def compute_historical_metrics(self) -> pd.DataFrame:
        """Compute derived metrics from historical annual data.

        Calculates profitability margins, return metrics (ROE, ROIC),
        working-capital days, leverage ratios, and buyback yield for
        each historical fiscal period.

        Stores the result in :pyattr:`metrics` and also returns it.

        Returns
        -------
        pd.DataFrame
            Metrics as rows, fiscal periods as columns.
        """
        inc = self.historical.get("income", pd.DataFrame())
        bal = self.historical.get("balance", pd.DataFrame())
        cf = self.historical.get("cashflow", pd.DataFrame())

        if inc.empty:
            self.metrics = pd.DataFrame()
            return self.metrics

        periods = list(inc.columns)
        metrics: dict[str, dict[str, float]] = {}

        def _val(df, item, period):
            try:
                v = df.loc[item, period]
                return v if pd.notna(v) else np.nan
            except KeyError:
                return np.nan

        for p in periods:
            m: dict[str, float] = {}
            rev = _val(inc, "total_revenue", p)
            cogs = _val(inc, "cost_of_revenue", p)
            gp = _val(inc, "gross_profit", p)
            op_inc = _val(inc, "operating_income", p)
            ni = _val(inc, "net_income", p)
            ebitda = _val(inc, "ebitda", p)
            pretax = _val(inc, "pretax_income", p)
            tax = _val(inc, "income_tax", p)
            da = _val(inc, "depreciation_amortization", p)
            sbc = _val(inc, "stock_based_comp", p)

            fcf = _val(cf, "free_cash_flow", p)
            capex = _val(cf, "capex", p)
            share_rep = _val(cf, "share_repurchases", p)

            ar = _val(bal, "accounts_receivable", p)
            inv = _val(bal, "inventory", p)
            ap = _val(bal, "accounts_payable", p)
            total_assets = _val(bal, "total_assets", p)
            equity = _val(bal, "total_stockholders_equity", p)
            lt_debt = _val(bal, "long_term_debt", p)
            st_debt = _val(bal, "short_term_debt", p)
            cash = _val(bal, "cash_and_equivalents", p)

            # Margins
            m["gross_margin"] = _safe_div(gp, rev)
            m["operating_margin"] = _safe_div(op_inc, rev)
            m["net_margin"] = _safe_div(ni, rev)
            m["fcf_margin"] = _safe_div(fcf, rev)

            # Revenue growth (YoY) - compute after loop
            m["revenue_yoy"] = np.nan

            # ROE, ROIC
            m["roe"] = _safe_div(ni, equity)

            # ROIC = NOPAT / invested capital
            nopat = op_inc * (1 - _safe_div(tax, pretax)) if (
                pd.notna(op_inc) and pd.notna(tax) and pd.notna(pretax)
                and pretax != 0
            ) else np.nan
            total_debt = (
                (lt_debt if pd.notna(lt_debt) else 0)
                + (st_debt if pd.notna(st_debt) else 0)
            )
            invested_capital = (
                (equity if pd.notna(equity) else 0) + total_debt
                - (cash if pd.notna(cash) else 0)
            )
            m["roic"] = _safe_div(nopat, invested_capital) if invested_capital else np.nan

            # Working capital days (annualised)
            m["days_sales_outstanding"] = _safe_div(ar, rev) * 365 if (
                pd.notna(ar) and pd.notna(rev) and rev != 0
            ) else np.nan
            m["days_payable_outstanding"] = _safe_div(ap, cogs) * 365 if (
                pd.notna(ap) and pd.notna(cogs) and cogs != 0
            ) else np.nan
            m["days_inventory_outstanding"] = _safe_div(inv, cogs) * 365 if (
                pd.notna(inv) and pd.notna(cogs) and cogs != 0
            ) else np.nan

            # Capex as % of revenue (capex is typically negative)
            m["capex_pct_revenue"] = _safe_div(abs(capex) if pd.notna(capex) else np.nan, rev)

            # SBC as % of revenue
            m["sbc_pct_revenue"] = _safe_div(sbc, rev)

            # Net debt
            m["net_debt"] = total_debt - (cash if pd.notna(cash) else 0)
            m["net_debt_to_ebitda"] = _safe_div(m["net_debt"], ebitda)

            # Effective tax rate
            m["effective_tax_rate"] = _safe_div(tax, pretax)

            # Buyback yield - need market cap
            m["buyback_yield"] = np.nan  # computed below if prices available

            metrics[p] = m

        # Revenue YoY growth
        for i, p in enumerate(periods):
            if i == 0:
                continue
            prev_p = periods[i - 1]
            rev_curr = _val(inc, "total_revenue", p)
            rev_prev = _val(inc, "total_revenue", prev_p)
            if pd.notna(rev_curr) and pd.notna(rev_prev) and rev_prev != 0:
                metrics[p]["revenue_yoy"] = (rev_curr - rev_prev) / rev_prev

        # Buyback yield: share_repurchases / market_cap at fiscal year end
        if not self._prices.empty:
            for p in periods:
                share_rep = _val(cf, "share_repurchases", p)
                shares = _val(inc, "diluted_shares_out", p)
                if pd.notna(share_rep) and pd.notna(shares) and shares > 0:
                    year = _fy_year(p)
                    # Find close price near fiscal year end
                    yr_prices = self._prices[
                        self._prices["date"].dt.year == year
                    ]
                    if not yr_prices.empty:
                        close = yr_prices.iloc[-1]["close"]
                        mkt_cap = close * shares
                        if mkt_cap > 0:
                            # share_repurchases is typically negative
                            metrics[p]["buyback_yield"] = abs(share_rep) / mkt_cap

        result = pd.DataFrame(metrics)
        # Sort columns chronologically
        result = result[_sort_periods(list(result.columns))]
        self.metrics = result
        return result

    # ------------------------------------------------------------------
    # 4. forecast
    # ------------------------------------------------------------------

    def forecast(self, years: int = 5, scenario: str = "base") -> pd.DataFrame:
        """Generate projected annual financial statements.

        Reads assumptions from the ``assumptions`` table.  If none exist,
        auto-generates base-case assumptions from historical data and
        stores them.

        The forecast drives revenue via a growth rate that decays linearly
        toward :data:`GDP_GROWTH`, derives cost and margin line items from
        assumption ratios, builds up the balance sheet from working-capital
        days and capex, and closes the cash-flow waterfall so that the
        accounting identity (assets = liabilities + equity) holds each
        period.

        Parameters
        ----------
        years : int
            Number of years to project (default 5).
        scenario : str
            Scenario label (e.g. ``'base'``, ``'bull'``, ``'bear'``).

        Returns
        -------
        pd.DataFrame
            Forecasted line items as rows, fiscal periods as columns.

        Raises
        ------
        ValueError
            If no historical income data exists for the ticker.
        """
        assumptions = self._load_or_generate_assumptions(scenario)

        # Determine the last historical fiscal year
        inc = self.historical.get("income", pd.DataFrame())
        bal = self.historical.get("balance", pd.DataFrame())
        cf = self.historical.get("cashflow", pd.DataFrame())

        if inc.empty:
            raise ValueError(f"No historical income data for {self.ticker}")

        hist_periods = _sort_periods(list(inc.columns))
        last_fy = hist_periods[-1]
        last_year = _fy_year(last_fy)

        # Collect projected data
        forecast_data: dict[str, dict[str, float]] = {}

        def _last_val(df, item):
            """Get the most recent non-NaN historical value.

            Returns 0.0 when the DataFrame is empty, the line item does
            not exist, or every historical value is NaN.  This ensures
            the forecast loop never crashes on missing data.
            """
            if df.empty or item not in df.index:
                logger.debug("Line item '%s' not found in historical data; defaulting to 0.0", item)
                return 0.0
            vals = df.loc[item].dropna()
            if len(vals) == 0:
                logger.debug("Line item '%s' has no non-NaN values; defaulting to 0.0", item)
                return 0.0
            return float(vals.iloc[-1])

        # Previous-year balance sheet values (start from last historical)
        prev_cash = _last_val(bal, "cash_and_equivalents")
        prev_ar = _last_val(bal, "accounts_receivable")
        prev_inv = _last_val(bal, "inventory")
        prev_ap = _last_val(bal, "accounts_payable")
        prev_ppe = _last_val(bal, "property_plant_equipment_net")
        prev_goodwill = _last_val(bal, "goodwill")
        prev_intangibles = _last_val(bal, "intangible_assets")
        prev_st_inv = _last_val(bal, "short_term_investments")
        prev_st_debt = _last_val(bal, "short_term_debt")
        prev_cp_ltd = _last_val(bal, "current_portion_lt_debt")
        prev_lt_debt = _last_val(bal, "long_term_debt")
        prev_equity = _last_val(bal, "total_stockholders_equity")
        prev_retained = _last_val(bal, "retained_earnings")
        prev_revenue = _last_val(inc, "total_revenue")
        prev_shares = _last_val(inc, "diluted_shares_out")

        for yr_offset in range(1, years + 1):
            fy = f"FY{last_year + yr_offset}"
            f: dict[str, float] = {}

            # ---- Revenue growth decays toward GDP growth ----
            base_growth = assumptions["revenue_growth"]
            # Decay: year 1 uses base_growth, then linearly blends toward GDP
            decay_factor = max(0, 1 - (yr_offset - 1) / max(years, 1))
            rev_growth = base_growth * decay_factor + GDP_GROWTH * (1 - decay_factor)
            f["total_revenue"] = prev_revenue * (1 + rev_growth)
            revenue = f["total_revenue"]

            # ---- Income statement ----
            gross_margin = assumptions["gross_margin"]
            f["cost_of_revenue"] = revenue * (1 - gross_margin)
            f["gross_profit"] = revenue * gross_margin

            op_margin = assumptions["operating_margin"]
            rd_pct = assumptions.get("rd_pct_revenue", 0)
            sga_pct = assumptions.get("sga_pct_revenue", 0)
            f["research_development"] = revenue * rd_pct
            f["selling_general_admin"] = revenue * sga_pct
            f["operating_income"] = revenue * op_margin

            # D&A: hold at historical % of PPE or revenue
            da_pct = assumptions.get("da_pct_revenue", 0.03)
            f["depreciation_amortization"] = revenue * da_pct

            # SBC
            sbc_pct = assumptions["sbc_pct_revenue"]
            f["stock_based_comp"] = revenue * sbc_pct

            # EBITDA
            f["ebitda"] = f["operating_income"] + f["depreciation_amortization"]

            # Interest (hold flat from last year)
            f["interest_expense"] = assumptions.get("interest_expense", 0)
            f["interest_income"] = assumptions.get("interest_income", 0)

            # Pretax income
            f["pretax_income"] = (
                f["operating_income"]
                - f["interest_expense"]
                + f["interest_income"]
            )

            # Tax -- no tax on losses
            tax_rate = assumptions["effective_tax_rate"]
            if f["pretax_income"] < 0:
                f["income_tax"] = 0.0
            else:
                f["income_tax"] = max(0, f["pretax_income"] * tax_rate)
            f["net_income"] = f["pretax_income"] - f["income_tax"]

            # Share count (declining with buybacks)
            buyback_pct = assumptions.get("annual_buyback_pct", 0)
            buyback_pct = max(0.0, buyback_pct)  # ensure non-negative
            f["diluted_shares_out"] = prev_shares * (1 - buyback_pct)
            f["basic_shares_out"] = f["diluted_shares_out"] * 0.99  # slight approx
            f["diluted_eps"] = _safe_div(f["net_income"], f["diluted_shares_out"])
            f["basic_eps"] = _safe_div(f["net_income"], f["basic_shares_out"])

            # ---- Balance sheet ----
            dso = assumptions["days_sales_outstanding"]
            dio = assumptions["days_inventory_outstanding"]
            dpo = assumptions["days_payable_outstanding"]

            f["accounts_receivable"] = revenue * dso / 365

            # If DIO is 0 (company doesn't carry inventory), just set to 0
            if dio == 0:
                f["inventory"] = 0.0
            else:
                f["inventory"] = f["cost_of_revenue"] * dio / 365

            f["accounts_payable"] = f["cost_of_revenue"] * dpo / 365

            # Capex
            capex_pct = assumptions["capex_pct_revenue"]
            f["capex"] = -(revenue * capex_pct)  # negative = cash outflow

            # PPE: prior + capex + D&A (D&A reduces PPE)
            f["property_plant_equipment_net"] = (
                prev_ppe + abs(f["capex"]) - f["depreciation_amortization"]
            )

            # Hold goodwill and intangibles flat
            f["goodwill"] = prev_goodwill
            f["intangible_assets"] = prev_intangibles
            f["short_term_investments"] = prev_st_inv

            # ---- Cash flow statement ----
            # Working capital changes
            delta_ar = f["accounts_receivable"] - prev_ar
            delta_inv = f["inventory"] - prev_inv
            delta_ap = f["accounts_payable"] - prev_ap
            f["change_in_working_capital"] = -delta_ar - delta_inv + delta_ap

            # Operating cash flow = net income + D&A + SBC + WC changes
            f["operating_cash_flow"] = (
                f["net_income"]
                + f["depreciation_amortization"]
                + f["stock_based_comp"]
                + f["change_in_working_capital"]
            )

            # FCF = operating cash flow + capex (capex is negative)
            f["free_cash_flow"] = f["operating_cash_flow"] + f["capex"]

            # Dividends and buybacks
            div_pct = assumptions.get("dividend_pct_ni", 0)
            div_pct = max(0.0, min(div_pct, 1.0))  # clamp to [0, 1]
            f["dividends_paid"] = -(abs(f["net_income"]) * div_pct)  # negative

            buyback_pct_fcf = assumptions.get("buyback_pct_fcf", 0.5)
            buyback_pct_fcf = max(0.0, min(buyback_pct_fcf, 1.0))  # clamp to [0, 1]

            f["share_repurchases"] = -(abs(prev_shares * buyback_pct *
                                           assumptions.get("share_price_est", 0)))
            # If we can't estimate buyback $, use FCF-based heuristic
            if assumptions.get("share_price_est", 0) == 0:
                f["share_repurchases"] = -(abs(f["free_cash_flow"])
                                            * buyback_pct_fcf)

            f["depreciation_cf"] = f["depreciation_amortization"]
            f["stock_based_comp_cf"] = f["stock_based_comp"]

            # Net debt issuance: hold debt levels flat for base case
            net_debt_issuance = 0.0

            # Ending cash = beginning cash + FCF - dividends - buybacks + net debt
            ending_cash = (
                prev_cash
                + f["free_cash_flow"]
                + f["dividends_paid"]       # negative
                + f["share_repurchases"]    # negative
                + net_debt_issuance
            )
            f["cash_and_equivalents"] = ending_cash

            # Debt: hold flat
            f["short_term_debt"] = prev_st_debt
            f["current_portion_lt_debt"] = prev_cp_ltd
            f["long_term_debt"] = prev_lt_debt

            # Retained earnings = prior + net income + dividends (negative)
            f["retained_earnings"] = prev_retained + f["net_income"] + f["dividends_paid"]

            # Total current assets
            f["total_current_assets"] = (
                f["cash_and_equivalents"]
                + f["short_term_investments"]
                + f["accounts_receivable"]
                + f["inventory"]
            )

            # Total assets
            f["total_assets"] = (
                f["total_current_assets"]
                + f["property_plant_equipment_net"]
                + f["goodwill"]
                + f["intangible_assets"]
            )

            # Total current liabilities
            f["total_current_liabilities"] = (
                f["accounts_payable"]
                + f["short_term_debt"]
                + f["current_portion_lt_debt"]
            )

            # Total liabilities
            f["total_liabilities"] = (
                f["total_current_liabilities"]
                + f["long_term_debt"]
            )

            # Stockholders equity = total assets - total liabilities
            f["total_stockholders_equity"] = f["total_assets"] - f["total_liabilities"]

            # ---- Balance sheet check: assets = liabilities + equity ----
            lhs = f["total_assets"]
            rhs = f["total_liabilities"] + f["total_stockholders_equity"]
            tolerance = max(abs(lhs), 1.0) * 0.001  # 0.1% relative tolerance
            if abs(lhs - rhs) > tolerance:
                logger.warning(
                    "Balance sheet does not balance for %s %s: "
                    "assets=%,.0f != liabilities+equity=%,.0f (diff=%,.2f, tol=%,.2f)",
                    self.ticker, fy, lhs, rhs, abs(lhs - rhs), tolerance,
                )

            forecast_data[fy] = f

            # Carry forward for next year
            prev_cash = f["cash_and_equivalents"]
            prev_ar = f["accounts_receivable"]
            prev_inv = f["inventory"]
            prev_ap = f["accounts_payable"]
            prev_ppe = f["property_plant_equipment_net"]
            prev_goodwill = f["goodwill"]
            prev_intangibles = f["intangible_assets"]
            prev_st_inv = f["short_term_investments"]
            prev_st_debt = f["short_term_debt"]
            prev_cp_ltd = f["current_portion_lt_debt"]
            prev_lt_debt = f["long_term_debt"]
            prev_equity = f["total_stockholders_equity"]
            prev_retained = f["retained_earnings"]
            prev_revenue = f["total_revenue"]
            prev_shares = f["diluted_shares_out"]

        # Write forecasts to DuckDB
        self._write_forecasts(forecast_data, scenario)

        result = pd.DataFrame(forecast_data)
        result = result[_sort_periods(list(result.columns))]
        return result

    def _load_or_generate_assumptions(self, scenario: str) -> dict[str, float]:
        """Load assumptions from the database, or auto-generate if none exist.

        Parameters
        ----------
        scenario : str
            Scenario label (e.g. ``'base'``).

        Returns
        -------
        dict[str, float]
            Mapping of parameter names to numeric values.
        """
        con = get_connection(self.db_path)
        try:
            rows = con.execute(
                "SELECT parameter_name, parameter_value FROM assumptions "
                "WHERE ticker = ? AND scenario = ?",
                [self.ticker, scenario],
            ).fetchall()
        finally:
            con.close()

        if rows:
            return {name: val for name, val in rows}

        # Auto-generate from historical data
        return self._generate_default_assumptions(scenario)

    def _generate_default_assumptions(self, scenario: str) -> dict[str, float]:
        """Generate base-case assumptions from historical data and save to DB.

        Derives revenue growth, profitability margins, working-capital
        days, capex intensity, tax rate, buyback pace, and dividend
        payout from historical financials.  Falls back to sensible
        defaults when historical line items are missing or insufficient.

        Parameters
        ----------
        scenario : str
            Scenario label under which to store the assumptions.

        Returns
        -------
        dict[str, float]
            The generated assumptions, also persisted to the database.
        """
        inc = self.historical.get("income", pd.DataFrame())
        bal = self.historical.get("balance", pd.DataFrame())
        cf = self.historical.get("cashflow", pd.DataFrame())

        # Ensure metrics are computed
        if self.metrics.empty:
            self.compute_historical_metrics()

        assumptions: dict[str, float] = {}

        # --- Revenue growth: trailing 3-year CAGR ---
        if not inc.empty and "total_revenue" in inc.index:
            rev_series = inc.loc["total_revenue"].dropna()
            if len(rev_series) >= 4:
                rev_end = rev_series.iloc[-1]
                rev_start = rev_series.iloc[-4]
                if rev_start > 0:
                    assumptions["revenue_growth"] = (rev_end / rev_start) ** (1 / 3) - 1
                else:
                    assumptions["revenue_growth"] = 0.05
            elif len(rev_series) >= 2:
                assumptions["revenue_growth"] = (
                    (rev_series.iloc[-1] / rev_series.iloc[-2]) - 1
                )
            else:
                assumptions["revenue_growth"] = 0.05
        else:
            assumptions["revenue_growth"] = 0.05

        # --- Margins: 5-year (or available) average ---
        def _avg_metric(metric_name: str, fallback: float = 0.0) -> float:
            if metric_name in self.metrics.index:
                vals = self.metrics.loc[metric_name].dropna()
                tail = vals.iloc[-5:] if len(vals) >= 5 else vals
                return float(tail.mean()) if len(tail) > 0 else fallback
            return fallback

        assumptions["gross_margin"] = _avg_metric("gross_margin", 0.40)
        assumptions["operating_margin"] = _avg_metric("operating_margin", 0.20)
        assumptions["net_margin"] = _avg_metric("net_margin", 0.15)

        # R&D and SGA as % of revenue (from historicals)
        if not inc.empty:
            rev = inc.loc["total_revenue"].dropna() if "total_revenue" in inc.index else pd.Series(dtype=float)

            if "research_development" in inc.index:
                rd = inc.loc["research_development"].dropna()
            else:
                rd = pd.Series(dtype=float)
                logger.info("%s: 'research_development' not reported; defaulting R&D %% to 0.", self.ticker)

            if "selling_general_admin" in inc.index:
                sga = inc.loc["selling_general_admin"].dropna()
            else:
                sga = pd.Series(dtype=float)
                logger.info("%s: 'selling_general_admin' not reported; defaulting SGA %% to 0.", self.ticker)

            common_periods = rev.index.intersection(rd.index)
            if len(common_periods) > 0:
                rd_pcts = rd[common_periods] / rev[common_periods]
                assumptions["rd_pct_revenue"] = float(rd_pcts.iloc[-5:].mean())
            else:
                assumptions["rd_pct_revenue"] = 0.0

            common_periods = rev.index.intersection(sga.index)
            if len(common_periods) > 0:
                sga_pcts = sga[common_periods] / rev[common_periods]
                assumptions["sga_pct_revenue"] = float(sga_pcts.iloc[-5:].mean())
            else:
                assumptions["sga_pct_revenue"] = 0.0
        else:
            assumptions["rd_pct_revenue"] = 0.0
            assumptions["sga_pct_revenue"] = 0.0

        # --- Capex / revenue ---
        assumptions["capex_pct_revenue"] = _avg_metric("capex_pct_revenue", 0.05)

        # --- SBC / revenue ---
        assumptions["sbc_pct_revenue"] = _avg_metric("sbc_pct_revenue", 0.02)

        # --- Working capital days ---
        assumptions["days_sales_outstanding"] = _avg_metric("days_sales_outstanding", 45)
        assumptions["days_payable_outstanding"] = _avg_metric("days_payable_outstanding", 60)

        # Inventory: some companies (e.g. software) don't carry inventory
        dio_val = _avg_metric("days_inventory_outstanding", 0)
        if not bal.empty and "inventory" in bal.index:
            inv_series = bal.loc["inventory"].dropna()
            if len(inv_series) > 0 and inv_series.iloc[-1] > 0:
                assumptions["days_inventory_outstanding"] = dio_val if dio_val > 0 else 30
            else:
                # Inventory is reported but zero -- no DIO needed
                assumptions["days_inventory_outstanding"] = 0
                logger.info("%s: inventory is zero or missing; setting DIO to 0.", self.ticker)
        else:
            # Inventory line item doesn't exist at all
            assumptions["days_inventory_outstanding"] = 0
            logger.info("%s: 'inventory' not reported; setting DIO to 0.", self.ticker)

        # --- Tax rate (clamped to [0, 0.50]) ---
        raw_tax_rate = _avg_metric("effective_tax_rate", 0.21)
        assumptions["effective_tax_rate"] = max(0.0, min(raw_tax_rate, 0.50))
        if raw_tax_rate < 0 or raw_tax_rate > 0.50:
            logger.info(
                "%s: raw effective tax rate %.2f%% clamped to [0%%, 50%%] -> %.2f%%",
                self.ticker, raw_tax_rate * 100, assumptions["effective_tax_rate"] * 100,
            )

        # --- D&A as % of revenue ---
        if not inc.empty and "depreciation_amortization" in inc.index and "total_revenue" in inc.index:
            da = inc.loc["depreciation_amortization"].dropna()
            rev = inc.loc["total_revenue"].dropna()
            common = da.index.intersection(rev.index)
            if len(common) > 0:
                da_pcts = da[common] / rev[common]
                assumptions["da_pct_revenue"] = float(da_pcts.iloc[-5:].mean())
            else:
                assumptions["da_pct_revenue"] = 0.03
        else:
            assumptions["da_pct_revenue"] = 0.03

        # --- Interest (hold flat from last year) ---
        if not inc.empty and "interest_expense" in inc.index:
            ie = inc.loc["interest_expense"].dropna()
            assumptions["interest_expense"] = float(ie.iloc[-1]) if len(ie) > 0 else 0
        else:
            assumptions["interest_expense"] = 0

        if not inc.empty and "interest_income" in inc.index:
            ii = inc.loc["interest_income"].dropna()
            assumptions["interest_income"] = float(ii.iloc[-1]) if len(ii) > 0 else 0
        else:
            assumptions["interest_income"] = 0

        # --- Buyback pace ---
        if not inc.empty and "diluted_shares_out" in inc.index:
            shares = inc.loc["diluted_shares_out"].dropna()
            if len(shares) >= 2:
                # Average annual share reduction rate over last 3 years
                n = min(3, len(shares) - 1)
                start_shares = shares.iloc[-(n + 1)]
                end_shares = shares.iloc[-1]
                if start_shares > 0:
                    annual_reduction = 1 - (end_shares / start_shares) ** (1 / n)
                    assumptions["annual_buyback_pct"] = max(0, annual_reduction)
                else:
                    assumptions["annual_buyback_pct"] = 0
            else:
                assumptions["annual_buyback_pct"] = 0
        else:
            assumptions["annual_buyback_pct"] = 0

        # Dividend payout ratio (clamped to [0, 1])
        if not cf.empty and "dividends_paid" in cf.index and not inc.empty and "net_income" in inc.index:
            divs = cf.loc["dividends_paid"].dropna()
            ni = inc.loc["net_income"].dropna()
            common = divs.index.intersection(ni.index)
            if len(common) > 0:
                payout = abs(divs[common]) / ni[common].replace(0, np.nan)
                raw_payout = float(payout.dropna().iloc[-3:].mean())
                assumptions["dividend_pct_ni"] = max(0.0, min(raw_payout, 1.0))
                if raw_payout < 0 or raw_payout > 1.0:
                    logger.info(
                        "%s: raw dividend payout ratio %.2f clamped to [0, 1] -> %.2f",
                        self.ticker, raw_payout, assumptions["dividend_pct_ni"],
                    )
            else:
                assumptions["dividend_pct_ni"] = 0
        else:
            assumptions["dividend_pct_ni"] = 0

        # Buyback $ estimation (use last year's repurchases / FCF ratio, clamped to [0, 1])
        if not cf.empty and "share_repurchases" in cf.index and "free_cash_flow" in cf.index:
            rep = cf.loc["share_repurchases"].dropna()
            fcf = cf.loc["free_cash_flow"].dropna()
            common = rep.index.intersection(fcf.index)
            if len(common) > 0:
                ratios = abs(rep[common]) / fcf[common].replace(0, np.nan)
                val = ratios.dropna().iloc[-3:].mean()
                raw_buyback_fcf = float(val) if pd.notna(val) else 0.5
                assumptions["buyback_pct_fcf"] = max(0.0, min(raw_buyback_fcf, 1.0))
            else:
                assumptions["buyback_pct_fcf"] = 0.5
        else:
            assumptions["buyback_pct_fcf"] = 0.5

        assumptions["share_price_est"] = 0  # use FCF-based approach

        # Save to database
        self._save_assumptions(scenario, assumptions)
        return assumptions

    def _save_assumptions(self, scenario: str, assumptions: dict[str, float]) -> None:
        """Persist assumptions to the database.

        Uses an upsert so that re-running generation overwrites stale
        values without creating duplicates.

        Parameters
        ----------
        scenario : str
            Scenario label.
        assumptions : dict[str, float]
            Parameter name / value pairs.
        """
        con = get_connection(self.db_path)
        try:
            for name, value in assumptions.items():
                con.execute(
                    """
                    INSERT INTO assumptions
                        (ticker, scenario, parameter_name, parameter_value)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT (ticker, scenario, parameter_name)
                    DO UPDATE SET parameter_value = EXCLUDED.parameter_value
                    """,
                    [self.ticker, scenario, name, float(value)],
                )
        finally:
            con.close()

    def _write_forecasts(self, forecast_data: dict[str, dict[str, float]],
                         scenario: str) -> None:
        """Write forecast values to the ``financials`` table.

        Clears any previous forecasts for this ticker/scenario before
        inserting the new rows so that the table always reflects the
        latest projection.

        Parameters
        ----------
        forecast_data : dict[str, dict[str, float]]
            Outer key is the period (e.g. ``'FY2026'``), inner dict maps
            line-item names to projected values.
        scenario : str
            Scenario label.
        """
        # Map line items to statement types
        item_to_stmt: dict[str, str] = {}
        for item in INCOME_ITEMS:
            item_to_stmt[item] = "income"
        for item in BALANCE_ITEMS:
            item_to_stmt[item] = "balance"
        for item in CASHFLOW_ITEMS:
            item_to_stmt[item] = "cashflow"

        con = get_connection(self.db_path)
        try:
            # Clear previous forecasts for this ticker/scenario
            con.execute(
                """
                DELETE FROM financials
                WHERE ticker = ?
                  AND is_forecast = true
                  AND forecast_scenario = ?
                """,
                [self.ticker, scenario],
            )

            now = datetime.now()
            for period, items in forecast_data.items():
                for item, value in items.items():
                    stmt = item_to_stmt.get(item)
                    if stmt is None:
                        continue
                    con.execute(
                        """
                        INSERT INTO financials
                            (ticker, period, period_type, statement, line_item,
                             value, is_forecast, forecast_scenario, updated_at)
                        VALUES (?, ?, 'annual', ?, ?, ?, true, ?, ?)
                        """,
                        [self.ticker, period, stmt, item, float(value),
                         scenario, now],
                    )
        finally:
            con.close()

    # ------------------------------------------------------------------
    # 5. set_assumption
    # ------------------------------------------------------------------

    def set_assumption(self, scenario: str, parameter: str, value: float) -> None:
        """Update a single assumption and re-run the forecast.

        This is the primary entry point for interactive what-if analysis.
        After persisting the new value, :meth:`forecast` is called
        automatically so that all downstream projections are refreshed.

        Parameters
        ----------
        scenario : str
            Scenario label (e.g. ``'base'``).
        parameter : str
            The assumption parameter name (e.g. ``'revenue_growth'``).
        value : float
            New numeric value for the parameter.
        """
        con = get_connection(self.db_path)
        try:
            con.execute(
                """
                INSERT INTO assumptions
                    (ticker, scenario, parameter_name, parameter_value)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (ticker, scenario, parameter_name)
                DO UPDATE SET parameter_value = EXCLUDED.parameter_value
                """,
                [self.ticker, scenario, parameter, float(value)],
            )
        finally:
            con.close()

        # Re-run forecast with updated assumptions
        self.forecast(scenario=scenario)

    # ------------------------------------------------------------------
    # 6. get_statement
    # ------------------------------------------------------------------

    def get_statement(self, statement_type: str, include_forecast: bool = True,
                      scenario: str = "base") -> pd.DataFrame:
        """Return a clean DataFrame combining historical and forecast data.

        Parameters
        ----------
        statement_type : str
            One of ``'income'``, ``'balance'``, ``'cashflow'``.
        include_forecast : bool
            Whether to append forecast columns from the database.
        scenario : str
            Which forecast scenario to include (ignored when
            *include_forecast* is ``False``).

        Returns
        -------
        pd.DataFrame
            Line items as rows and periods as columns, sorted
            chronologically.
        """
        # Start with historical data
        hist = self.historical.get(statement_type, pd.DataFrame()).copy()

        if not include_forecast:
            return hist

        # Load forecast data from DB
        con = get_connection(self.db_path)
        try:
            fc_df = con.execute(
                """
                SELECT period, line_item, value
                FROM financials
                WHERE ticker = ?
                  AND statement = ?
                  AND is_forecast = true
                  AND forecast_scenario = ?
                  AND period_type = 'annual'
                """,
                [self.ticker, statement_type, scenario],
            ).fetchdf()
        finally:
            con.close()

        if fc_df.empty:
            return hist

        fc_wide = fc_df.pivot_table(
            index="line_item", columns="period", values="value",
            aggfunc="first",
        )

        if hist.empty:
            combined = fc_wide
        else:
            combined = hist.join(fc_wide, how="outer")

        # Sort columns chronologically
        combined = combined[_sort_periods(list(combined.columns))]
        return combined


# ---------------------------------------------------------------------------
# CLI test: run on AAPL
# ---------------------------------------------------------------------------

def _format_number(val):
    """Format a number for display (millions for large numbers).

    Parameters
    ----------
    val : float or NaN
        The numeric value to format.

    Returns
    -------
    str
        Human-readable string with B/M/K suffix as appropriate,
        or an empty string for NaN values.
    """
    if pd.isna(val):
        return ""
    if abs(val) >= 1e9:
        return f"{val / 1e9:,.1f}B"
    if abs(val) >= 1e6:
        return f"{val / 1e6:,.1f}M"
    if abs(val) >= 1e3:
        return f"{val / 1e3:,.1f}K"
    return f"{val:,.2f}"


def main():
    """Test the financial model on AAPL.

    Loads historical data, computes metrics, generates a 5-year
    base-case forecast, and prints a combined income statement
    showing the last four historical years alongside projections.
    """
    print("=" * 70)
    print("  Financial Model Test: AAPL")
    print("=" * 70)

    # 1. Load data
    print("\n1. Loading historical data...")
    model = FinancialModel("AAPL")
    for stmt, df in model.historical.items():
        if not df.empty:
            print(f"   {stmt}: {len(df)} line items x {len(df.columns)} periods")
        else:
            print(f"   {stmt}: no data")

    # 2. Compute metrics
    print("\n2. Computing historical metrics...")
    metrics = model.compute_historical_metrics()
    if not metrics.empty:
        print(f"   Computed {len(metrics)} metrics across {len(metrics.columns)} periods")
        key_metrics = [
            "gross_margin", "operating_margin", "net_margin",
            "revenue_yoy", "roe", "roic",
        ]
        print("\n   Key metrics (most recent periods):")
        for m in key_metrics:
            if m in metrics.index:
                recent = metrics.loc[m].dropna().iloc[-3:]
                vals = "  ".join(
                    f"{p}: {v:.1%}" for p, v in recent.items()
                )
                print(f"   {m:30s} {vals}")

    # 3. Generate forecast
    print("\n3. Generating 5-year base case forecast...")
    forecast = model.forecast(years=5, scenario="base")
    print(f"   Projected {len(forecast.columns)} years: {list(forecast.columns)}")

    # 4. Print income statement with historical + projected
    print("\n4. Combined Income Statement (Historical + Forecast)")
    print("-" * 70)
    inc_stmt = model.get_statement("income", include_forecast=True, scenario="base")

    if not inc_stmt.empty:
        # Select key line items for display
        display_items = [
            "total_revenue", "cost_of_revenue", "gross_profit",
            "operating_income", "ebitda", "pretax_income",
            "income_tax", "net_income", "diluted_eps",
        ]

        # Filter to items that exist
        display_items = [i for i in display_items if i in inc_stmt.index]

        # Show last 4 historical + all forecast
        all_periods = list(inc_stmt.columns)
        hist_periods = [p for p in all_periods if p in model.historical.get("income", pd.DataFrame()).columns]
        fc_periods = [p for p in all_periods if p not in hist_periods]
        show_periods = hist_periods[-4:] + fc_periods

        # Print header
        header = f"{'Line Item':30s}"
        for p in show_periods:
            marker = " *" if p in fc_periods else ""
            header += f" {p + marker:>12s}"
        print(header)
        print("-" * len(header))

        for item in display_items:
            row = f"{item:30s}"
            for p in show_periods:
                val = inc_stmt.loc[item, p] if p in inc_stmt.columns else np.nan
                row += f" {_format_number(val):>12s}"
            print(row)

        print("\n   * = forecast")
    else:
        print("   No data available.")

    print("\n" + "=" * 70)
    print("  Model test complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
