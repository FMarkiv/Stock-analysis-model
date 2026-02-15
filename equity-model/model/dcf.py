"""
Discounted cash-flow valuation engine.

Provides ``DCFValuation`` for WACC calculation, DCF valuation,
sensitivity analysis, and relative-multiples valuation.

This module implements a full equity valuation workflow:

1. **WACC calculation** via CAPM beta regression and balance-sheet data.
2. **DCF valuation** using projected unlevered free cash flows, with both
   perpetuity-growth and exit-multiple terminal value approaches.
3. **Sensitivity analysis** producing a 2-D table of implied share prices
   across varying WACC / terminal-growth / exit-multiple assumptions.
4. **Relative-multiples valuation** using trailing Forward P/E, EV/EBITDA,
   and Price/FCF multiples applied to Year-1 forecast estimates.

The engine is designed to degrade gracefully when data is missing or
earnings are negative, returning partial results with warning flags
rather than raising exceptions.
"""

import logging
import os
import sys

import numpy as np
import pandas as pd
import yaml

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from db.schema import get_connection, init_db  # noqa: E402
from model.statements import FinancialModel, _sort_periods, _safe_div, _fy_year  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load ``config.yaml`` from the project root.

    Returns
    -------
    dict
        Parsed configuration dictionary, or an empty dict if the file
        is not found.
    """
    config_path = os.path.join(_PROJECT_ROOT, "config.yaml")
    try:
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


# ---------------------------------------------------------------------------
# DCFValuation
# ---------------------------------------------------------------------------


class DCFValuation:
    """Discounted Cash Flow valuation model.

    Orchestrates WACC computation, multi-year DCF projection, sensitivity
    analysis, and relative-multiples valuation for a single equity ticker.

    The model pulls historical financial statements from a DuckDB database,
    builds a 3-statement forecast via :class:`model.statements.FinancialModel`,
    and then values the equity using both intrinsic (DCF) and relative
    (multiples) approaches.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. ``'AAPL'``).
    db_path : str, optional
        Path to the DuckDB database file.  Falls back to the default
        ``data/equity.duckdb`` location.
    """

    def __init__(self, ticker: str, db_path: str | None = None):
        self.ticker = ticker.upper()
        self.db_path = db_path or os.path.join(
            _PROJECT_ROOT, "data", "equity.duckdb",
        )

        # Load project configuration
        self.config = _load_config()
        self.model_params = self.config.get("model_parameters", {})

        # Build the 3-statement financial model and compute metrics
        self.model = FinancialModel(self.ticker, self.db_path)
        self.model.compute_historical_metrics()

        # Run the base-case forecast (others generated on demand)
        self._forecasts: dict[str, pd.DataFrame] = {}
        self._forecasts["base"] = self.model.forecast(years=5, scenario="base")

        # WACC is computed lazily and cached
        self._wacc_result: dict | None = None

    # ------------------------------------------------------------------
    # 2. compute_wacc
    # ------------------------------------------------------------------

    def compute_wacc(self) -> dict:
        """Calculate Weighted Average Cost of Capital.

        Uses CAPM for cost of equity (risk-free rate + beta * ERP) and
        the ratio of interest expense to total debt for cost of debt.
        Capital-structure weights are derived from market capitalisation
        (preferred) or book equity as a fallback.

        Returns
        -------
        dict
            Keys: ``wacc``, ``cost_of_equity``, ``cost_of_debt``,
            ``cost_of_debt_pretax``, ``beta``, ``risk_free_rate``,
            ``equity_risk_premium``, ``tax_rate``, ``debt_to_equity``,
            ``weight_equity``, ``weight_debt``, ``total_debt``,
            ``equity_book_value``, ``market_cap``.
        """
        # --- Risk-free rate (config default, optionally from FRED) ---
        risk_free = self.model_params.get("risk_free_rate", 0.043)

        fred_key = (
            self.config
            .get("api_keys", {})
            .get("fred", {})
            .get("api_key", "")
        )
        if fred_key and fred_key != "YOUR_FRED_API_KEY":
            try:
                import requests
                resp = requests.get(
                    "https://api.stlouisfed.org/fred/series/observations",
                    params={
                        "series_id": "DGS10",
                        "api_key": fred_key,
                        "file_type": "json",
                        "sort_order": "desc",
                        "limit": 1,
                    },
                    timeout=10,
                )
                if resp.status_code == 200:
                    obs = resp.json().get("observations", [])
                    if obs and obs[0]["value"] != ".":
                        risk_free = float(obs[0]["value"]) / 100.0
            except Exception:
                pass  # keep config default

        # --- Equity risk premium ---
        erp = self.model_params.get("equity_risk_premium", 0.055)

        # --- Beta from price regression vs SPY ---
        beta = self._compute_beta()

        # --- Cost of equity (CAPM) ---
        cost_of_equity = risk_free + beta * erp

        # --- Balance-sheet aggregates ---
        bal = self.model.historical.get("balance", pd.DataFrame())
        inc = self.model.historical.get("income", pd.DataFrame())

        interest_expense = 0.0
        total_debt = 0.0
        equity_book = 0.0
        cash = 0.0

        if not inc.empty and "interest_expense" in inc.index:
            ie = inc.loc["interest_expense"].dropna()
            if len(ie) > 0:
                interest_expense = abs(float(ie.iloc[-1]))

        if not bal.empty:
            for item in ("long_term_debt", "short_term_debt"):
                if item in bal.index:
                    s = bal.loc[item].dropna()
                    if len(s) > 0:
                        total_debt += float(s.iloc[-1])
            if "total_stockholders_equity" in bal.index:
                eq = bal.loc["total_stockholders_equity"].dropna()
                if len(eq) > 0:
                    equity_book = float(eq.iloc[-1])
            if "cash_and_equivalents" in bal.index:
                c = bal.loc["cash_and_equivalents"].dropna()
                if len(c) > 0:
                    cash = float(c.iloc[-1])

        # --- Cost of debt ---
        cost_of_debt_pretax = (
            interest_expense / total_debt if total_debt > 0 else 0.04
        )

        # Effective tax rate (clamp to [0, 0.50])
        tax_rate = 0.21
        if (
            not self.model.metrics.empty
            and "effective_tax_rate" in self.model.metrics.index
        ):
            etr = self.model.metrics.loc["effective_tax_rate"].dropna()
            if len(etr) > 0:
                tax_rate = max(0.0, min(float(etr.iloc[-1]), 0.50))

        cost_of_debt = cost_of_debt_pretax * (1 - tax_rate)

        # --- Capital-structure weights ---
        market_cap = self._get_market_cap()
        if market_cap and market_cap > 0:
            total_capital = market_cap + total_debt
            weight_equity = market_cap / total_capital if total_capital > 0 else 1.0
        else:
            equity_for_w = max(equity_book, 1.0)
            total_capital = equity_for_w + total_debt
            weight_equity = equity_for_w / total_capital if total_capital > 0 else 1.0

        weight_debt = 1.0 - weight_equity
        d_e = total_debt / max(equity_book, 1.0) if equity_book > 0 else total_debt

        wacc = weight_equity * cost_of_equity + weight_debt * cost_of_debt

        result = {
            "wacc": wacc,
            "cost_of_equity": cost_of_equity,
            "cost_of_debt": cost_of_debt,
            "cost_of_debt_pretax": cost_of_debt_pretax,
            "beta": beta,
            "risk_free_rate": risk_free,
            "equity_risk_premium": erp,
            "tax_rate": tax_rate,
            "debt_to_equity": d_e,
            "weight_equity": weight_equity,
            "weight_debt": weight_debt,
            "total_debt": total_debt,
            "equity_book_value": equity_book,
            "market_cap": market_cap,
        }
        self._wacc_result = result
        return result

    # ------------------------------------------------------------------
    # Beta helper
    # ------------------------------------------------------------------

    def _compute_beta(self, years: int = 2) -> float:
        """Regression beta of weekly returns vs SPY over *years* years.

        Computes the slope coefficient from a simple OLS regression of the
        ticker's weekly returns on SPY's weekly returns, using Friday-close
        resampled prices.

        Falls back to 1.0 if fewer than 20 weekly observations of
        overlapping data are available for either the ticker or SPY, and
        logs a warning.

        Parameters
        ----------
        years : int
            Number of years of historical price data to use.

        Returns
        -------
        float
            Estimated beta coefficient.  Returns ``1.0`` as a safe default
            when insufficient data is available.
        """
        con = get_connection(self.db_path)
        try:
            ticker_prices = con.execute(
                "SELECT date, close FROM prices WHERE ticker = ? ORDER BY date",
                [self.ticker],
            ).fetchdf()

            spy_prices = con.execute(
                "SELECT date, close FROM prices WHERE ticker = 'SPY' ORDER BY date",
            ).fetchdf()
        finally:
            con.close()

        if ticker_prices.empty or spy_prices.empty:
            logger.warning(
                "%s: No price data for %s or SPY; defaulting beta to 1.0",
                self.ticker,
                self.ticker if ticker_prices.empty else "SPY",
            )
            return 1.0

        # Convert to weekly (Friday close)
        for df in (ticker_prices, spy_prices):
            df["date"] = pd.to_datetime(df["date"])

        tk = ticker_prices.set_index("date")["close"].resample("W-FRI").last().dropna()
        sp = spy_prices.set_index("date")["close"].resample("W-FRI").last().dropna()

        # Keep only last N years from the ticker's most-recent date
        if len(tk) > 0:
            cutoff = tk.index[-1] - pd.DateOffset(years=years)
            tk = tk[tk.index >= cutoff]
            sp = sp[sp.index >= cutoff]

        common = tk.index.intersection(sp.index)
        if len(common) < 20:
            logger.warning(
                "%s: Only %d overlapping weekly observations (need 20); "
                "defaulting beta to 1.0",
                self.ticker,
                len(common),
            )
            return 1.0

        tk_ret = tk[common].pct_change().dropna()
        sp_ret = sp[common].pct_change().dropna()
        common_ret = tk_ret.index.intersection(sp_ret.index)
        if len(common_ret) < 20:
            logger.warning(
                "%s: Only %d overlapping return observations (need 20); "
                "defaulting beta to 1.0",
                self.ticker,
                len(common_ret),
            )
            return 1.0

        x = sp_ret[common_ret].values
        y = tk_ret[common_ret].values
        cov = np.cov(x, y)
        if cov[0, 0] == 0:
            logger.warning(
                "%s: SPY variance is zero; defaulting beta to 1.0",
                self.ticker,
            )
            return 1.0

        return float(cov[0, 1] / cov[0, 0])

    # ------------------------------------------------------------------
    # Market helpers
    # ------------------------------------------------------------------

    def _get_market_cap(self) -> float | None:
        """Most-recent price multiplied by diluted shares outstanding.

        Returns
        -------
        float or None
            Market capitalisation in the same currency unit as the
            financial statements, or ``None`` if price or share-count
            data is unavailable.
        """
        con = get_connection(self.db_path)
        try:
            row = con.execute(
                "SELECT close FROM prices "
                "WHERE ticker = ? ORDER BY date DESC LIMIT 1",
                [self.ticker],
            ).fetchone()
        finally:
            con.close()

        if row is None:
            return None

        price = row[0]
        inc = self.model.historical.get("income", pd.DataFrame())
        if not inc.empty and "diluted_shares_out" in inc.index:
            shares = inc.loc["diluted_shares_out"].dropna()
            if len(shares) > 0:
                return price * float(shares.iloc[-1])
        return None

    def _get_current_price(self) -> float:
        """Return the most-recent closing price from the database.

        If no price data exists for the ticker, returns ``0.0`` and logs
        a warning instead of raising an exception.

        Returns
        -------
        float
            Most-recent closing price, or ``0.0`` if unavailable.
        """
        con = get_connection(self.db_path)
        try:
            row = con.execute(
                "SELECT close FROM prices "
                "WHERE ticker = ? ORDER BY date DESC LIMIT 1",
                [self.ticker],
            ).fetchone()
        finally:
            con.close()

        if row is None:
            logger.warning(
                "%s: No price data found; returning 0.0 as current price",
                self.ticker,
            )
            return 0.0
        return row[0]

    def _get_balance_sheet_items(self) -> dict:
        """Extract latest total_debt, cash, net_debt, and diluted_shares.

        Missing balance-sheet line items (e.g. ``short_term_debt`` for
        companies that have none) are defaulted to ``0.0`` rather than
        causing a ``KeyError``.

        Returns
        -------
        dict
            Keys: ``total_debt``, ``cash``, ``net_debt``,
            ``diluted_shares``.
        """
        bal = self.model.historical.get("balance", pd.DataFrame())
        inc = self.model.historical.get("income", pd.DataFrame())

        total_debt = 0.0
        cash = 0.0
        if not bal.empty:
            for item in ("long_term_debt", "short_term_debt"):
                if item in bal.index:
                    s = bal.loc[item].dropna()
                    if len(s) > 0:
                        val = s.iloc[-1]
                        total_debt += float(val) if pd.notna(val) else 0.0
                # If the item is missing from the index, it simply
                # contributes 0 -- no error raised.
            if "cash_and_equivalents" in bal.index:
                c = bal.loc["cash_and_equivalents"].dropna()
                if len(c) > 0:
                    val = c.iloc[-1]
                    cash = float(val) if pd.notna(val) else 0.0

        shares = 15_000_000_000.0
        if not inc.empty and "diluted_shares_out" in inc.index:
            s = inc.loc["diluted_shares_out"].dropna()
            if len(s) > 0:
                shares = float(s.iloc[-1])

        return {
            "total_debt": total_debt,
            "cash": cash,
            "net_debt": total_debt - cash,
            "diluted_shares": shares,
        }

    # ------------------------------------------------------------------
    # 3. dcf_valuation
    # ------------------------------------------------------------------

    def dcf_valuation(self, scenario: str = "base") -> dict:
        """Run a standard DCF valuation.

        Projects unlevered free cash flows (UFCF) from the 3-statement
        model, discounts them at WACC, and computes terminal value using
        both the perpetuity-growth and exit-multiple methods.  The final
        enterprise value is the average of the two TV approaches.

        **Robustness features:**

        * If all projected FCFs are negative the result is flagged with
          ``dcf_valid = False`` and a warning message; ``implied_price``
          is set to ``0``.
        * If ``WACC <= terminal_growth``, the WACC is automatically
          increased by 1 percentage point (or terminal growth reduced)
          to prevent division by zero in the perpetuity formula.
        * If equity value is negative, ``implied_price`` is set to ``0``
          with a warning flag.

        Parameters
        ----------
        scenario : str
            ``'base'``, ``'bull'``, or ``'bear'``.

        Returns
        -------
        dict
            ``implied_price``, ``current_price``, ``upside_downside``,
            ``enterprise_value``, ``equity_value``, ``dcf_valid``,
            optionally ``dcf_warning``, plus detailed intermediate
            results.
        """
        # Ensure WACC is available
        if self._wacc_result is None:
            self.compute_wacc()

        wacc = self._wacc_result["wacc"]
        tax_rate = self._wacc_result["tax_rate"]
        terminal_growth = self.model_params.get("terminal_growth_rate", 0.025)

        # --- Guard: WACC must exceed terminal growth for perpetuity ---
        warnings_list: list[str] = []
        if wacc <= terminal_growth:
            original_wacc = wacc
            original_tg = terminal_growth
            # Try increasing WACC by 1pp first
            wacc = wacc + 0.01
            if wacc <= terminal_growth:
                # Also reduce terminal growth so the spread is at least 1pp
                terminal_growth = wacc - 0.01
            logger.warning(
                "%s: WACC (%.4f) <= terminal_growth (%.4f); adjusted to "
                "WACC=%.4f, terminal_growth=%.4f",
                self.ticker,
                original_wacc,
                original_tg,
                wacc,
                terminal_growth,
            )
            warnings_list.append(
                f"WACC ({original_wacc:.4f}) <= terminal_growth "
                f"({original_tg:.4f}); auto-adjusted to WACC={wacc:.4f}, "
                f"terminal_growth={terminal_growth:.4f}"
            )

        # Ensure forecast exists for the requested scenario
        if scenario not in self._forecasts:
            self._forecasts[scenario] = self.model.forecast(
                years=5, scenario=scenario,
            )
        forecast = self._forecasts[scenario]
        forecast_periods = _sort_periods(list(forecast.columns))

        # ---- Project unlevered FCF ----
        # UFCF = EBIT*(1-t) + D&A - CapEx - delta-WC
        # In our storage: capex is negative, change_in_working_capital
        # encodes the cash-flow impact (negative = WC increase = cash used).
        # So: UFCF = EBIT*(1-t) + D&A + capex_stored + delta-wc_stored
        projected_fcfs: dict[str, float] = {}
        for period in forecast_periods:
            ebit = float(forecast.at["operating_income", period]) if "operating_income" in forecast.index else 0.0
            da = float(forecast.at["depreciation_amortization", period]) if "depreciation_amortization" in forecast.index else 0.0
            capex = float(forecast.at["capex", period]) if "capex" in forecast.index else 0.0
            dwc = float(forecast.at["change_in_working_capital", period]) if "change_in_working_capital" in forecast.index else 0.0

            ufcf = ebit * (1 - tax_rate) + da + capex + dwc
            projected_fcfs[period] = ufcf

        # --- Check if all projected FCFs are negative ---
        dcf_valid = True
        all_fcfs_negative = all(v < 0 for v in projected_fcfs.values()) if projected_fcfs else True
        if all_fcfs_negative:
            dcf_valid = False
            neg_warning = "All projected FCFs are negative; DCF valuation unreliable"
            warnings_list.append(neg_warning)
            logger.warning("%s: %s", self.ticker, neg_warning)

        # ---- Discount projected FCFs ----
        pv_fcfs = 0.0
        fcf_values: list[float] = []
        for i, period in enumerate(forecast_periods):
            pv_fcfs += projected_fcfs[period] / (1 + wacc) ** (i + 1)
            fcf_values.append(projected_fcfs[period])

        n_years = len(forecast_periods)
        final_fcf = fcf_values[-1] if fcf_values else 0.0

        # ---- Terminal value -- perpetuity growth ----
        if wacc > terminal_growth:
            tv_perpetuity = final_fcf * (1 + terminal_growth) / (wacc - terminal_growth)
        else:
            tv_perpetuity = final_fcf * 25  # safety cap
        pv_tv_perpetuity = tv_perpetuity / (1 + wacc) ** n_years

        # ---- Terminal value -- exit multiple ----
        exit_multiple = self.model_params.get("exit_ebitda_multiple", 12.0)
        terminal_ebitda = (
            float(forecast.at["ebitda", forecast_periods[-1]])
            if "ebitda" in forecast.index
            else 0.0
        )
        tv_exit_multiple = terminal_ebitda * exit_multiple
        pv_tv_exit_multiple = tv_exit_multiple / (1 + wacc) ** n_years

        # ---- Enterprise value (average of both TV methods) ----
        ev_perpetuity = pv_fcfs + pv_tv_perpetuity
        ev_exit = pv_fcfs + pv_tv_exit_multiple
        enterprise_value = (ev_perpetuity + ev_exit) / 2

        # ---- Bridge to equity value ----
        bs = self._get_balance_sheet_items()
        equity_value = enterprise_value - bs["net_debt"]

        # --- Handle negative equity value ---
        if equity_value < 0:
            implied_price = 0.0
            eq_warning = (
                f"Equity value is negative ({equity_value:,.0f}); "
                "implied price set to 0"
            )
            warnings_list.append(eq_warning)
            logger.warning("%s: %s", self.ticker, eq_warning)
        elif all_fcfs_negative:
            # All FCFs negative: set price to 0 but still return results
            implied_price = 0.0
        else:
            implied_price = equity_value / bs["diluted_shares"] if bs["diluted_shares"] > 0 else 0.0

        current_price = self._get_current_price()

        upside_downside = (
            (implied_price / current_price - 1) if current_price > 0 else 0.0
        )

        result = {
            "implied_price": implied_price,
            "current_price": current_price,
            "upside_downside": upside_downside,
            "enterprise_value": enterprise_value,
            "equity_value": equity_value,
            "pv_fcfs": pv_fcfs,
            "terminal_value_perpetuity": tv_perpetuity,
            "terminal_value_exit_multiple": tv_exit_multiple,
            "pv_terminal_perpetuity": pv_tv_perpetuity,
            "pv_terminal_exit_multiple": pv_tv_exit_multiple,
            "ev_perpetuity_method": ev_perpetuity,
            "ev_exit_multiple_method": ev_exit,
            "wacc": wacc,
            "terminal_growth": terminal_growth,
            "exit_multiple": exit_multiple,
            "net_debt": bs["net_debt"],
            "cash": bs["cash"],
            "diluted_shares": bs["diluted_shares"],
            "projected_fcfs": projected_fcfs,
            "dcf_valid": dcf_valid,
        }

        if warnings_list:
            result["dcf_warning"] = "; ".join(warnings_list)

        return result

    # ------------------------------------------------------------------
    # 4. sensitivity_table
    # ------------------------------------------------------------------

    def sensitivity_table(
        self,
        variable1: str = "terminal_growth",
        range1: tuple = (-0.01, 0.04, 0.005),
        variable2: str = "wacc",
        range2: tuple = (-0.02, 0.02, 0.005),
    ) -> pd.DataFrame:
        """2-D sensitivity table of implied share price.

        Produces a matrix of implied share prices by varying two model
        parameters around their base-case values.  Each cell is computed
        via :meth:`_quick_dcf`, which re-discounts the base-case FCFs
        under the perturbed parameters.

        Parameters
        ----------
        variable1 : str
            Row variable (``'terminal_growth'``, ``'wacc'``,
            ``'exit_multiple'``).
        range1 : tuple
            ``(start_offset, end_offset, step)`` relative to the base
            value.
        variable2 : str
            Column variable.
        range2 : tuple
            ``(start_offset, end_offset, step)`` relative to the base
            value.

        Returns
        -------
        pd.DataFrame
            Rows = *variable1* values, columns = *variable2* values,
            cells = implied share price.
        """
        if self._wacc_result is None:
            self.compute_wacc()

        base_dcf = self.dcf_valuation(scenario="base")

        base_values = {
            "terminal_growth": base_dcf["terminal_growth"],
            "wacc": base_dcf["wacc"],
            "exit_multiple": base_dcf["exit_multiple"],
        }

        def _axis(var, rng):
            base = base_values.get(var, 0)
            return np.arange(
                base + rng[0],
                base + rng[1] + rng[2] / 2,
                rng[2],
            )

        vals1 = _axis(variable1, range1)
        vals2 = _axis(variable2, range2)

        rows: dict[float, dict[float, float]] = {}
        for v1 in vals1:
            row: dict[float, float] = {}
            for v2 in vals2:
                price = self._quick_dcf(
                    base_dcf=base_dcf,
                    **{variable1: v1, variable2: v2},
                )
                row[round(v2, 4)] = round(price, 2)
            rows[round(v1, 4)] = row

        df = pd.DataFrame(rows).T
        df.index.name = variable1
        df.columns.name = variable2
        return df

    def _quick_dcf(self, base_dcf: dict, **overrides) -> float:
        """Fast DCF re-discount with overridden parameters.

        Re-uses projected FCFs from the base run and only recalculates
        the discounting / terminal-value step, making it suitable for
        sensitivity and Monte Carlo loops.

        Parameters
        ----------
        base_dcf : dict
            Result dictionary from a prior :meth:`dcf_valuation` call.
        **overrides
            Parameter overrides.  Supported keys: ``wacc``,
            ``terminal_growth``, ``exit_multiple``.

        Returns
        -------
        float
            Implied share price under the overridden assumptions.
        """
        wacc = overrides.get("wacc", base_dcf["wacc"])
        tg = overrides.get("terminal_growth", base_dcf["terminal_growth"])
        exit_mult = overrides.get("exit_multiple", base_dcf["exit_multiple"])

        fcfs = base_dcf["projected_fcfs"]
        periods = _sort_periods(list(fcfs.keys()))

        pv_fcfs = 0.0
        fcf_list: list[float] = []
        for i, period in enumerate(periods):
            pv_fcfs += fcfs[period] / (1 + wacc) ** (i + 1)
            fcf_list.append(fcfs[period])

        final_fcf = fcf_list[-1] if fcf_list else 0.0
        n = len(periods)

        # Perpetuity TV
        if wacc > tg:
            tv_perp = final_fcf * (1 + tg) / (wacc - tg)
        else:
            tv_perp = final_fcf * 25
        pv_tv_perp = tv_perp / (1 + wacc) ** n

        # Exit-multiple TV (use terminal EBITDA from forecast)
        ebitda_for_tv = 0.0
        forecast = self._forecasts.get("base")
        if forecast is not None and "ebitda" in forecast.index:
            last_period = _sort_periods(list(forecast.columns))[-1]
            ebitda_for_tv = float(forecast.at["ebitda", last_period])

        tv_exit = ebitda_for_tv * exit_mult
        pv_tv_exit = tv_exit / (1 + wacc) ** n

        ev = pv_fcfs + (pv_tv_perp + pv_tv_exit) / 2
        equity = ev - base_dcf["net_debt"]
        shares = base_dcf["diluted_shares"]
        return equity / shares if shares > 0 else 0.0

    # ------------------------------------------------------------------
    # 5. multiples_valuation
    # ------------------------------------------------------------------

    def multiples_valuation(self) -> dict:
        """Implied value via relative-valuation multiples.

        Calculates Forward P/E, EV/EBITDA, and Price/FCF using the
        trailing 5-year average multiples for this stock applied to
        forecast Year-1 estimates.

        **Robustness features:**

        * If forward EPS is zero or negative, the P/E method is skipped.
        * If forecast EBITDA is zero or negative, the EV/EBITDA method
          is skipped.
        * If forecast FCF is zero or negative, the Price/FCF method is
          skipped.
        * Whatever methods are valid are returned.  An empty ``methods``
          dict is possible if all inputs are non-positive.

        Returns
        -------
        dict
            ``methods`` sub-dict containing whichever of ``forward_pe``,
            ``ev_ebitda``, ``price_fcf`` are valid (each a dict with
            ``hist_avg_multiple``, implied inputs, and
            ``implied_price``), plus ``current_price`` and
            ``multiples_warning`` if any methods were skipped.
        """
        inc = self.model.historical.get("income", pd.DataFrame())
        cf = self.model.historical.get("cashflow", pd.DataFrame())
        bal = self.model.historical.get("balance", pd.DataFrame())

        if inc.empty:
            logger.warning("%s: No income data; cannot compute multiples", self.ticker)
            return {}

        hist_periods = _sort_periods(list(inc.columns))

        # --- Trailing historical multiples ---
        hist_pe: list[float] = []
        hist_ev_ebitda: list[float] = []
        hist_p_fcf: list[float] = []

        con = get_connection(self.db_path)
        try:
            for period in hist_periods:
                year = _fy_year(period)
                row = con.execute(
                    "SELECT close FROM prices "
                    "WHERE ticker = ? AND EXTRACT(year FROM date) = ? "
                    "ORDER BY date DESC LIMIT 1",
                    [self.ticker, year],
                ).fetchone()
                if row is None:
                    continue

                price = row[0]

                # P/E
                if "diluted_eps" in inc.index:
                    eps = inc.at["diluted_eps", period]
                    if pd.notna(eps) and eps > 0:
                        hist_pe.append(price / eps)

                # EV/EBITDA
                if "ebitda" in inc.index and "diluted_shares_out" in inc.index:
                    ebitda = inc.at["ebitda", period]
                    shares = inc.at["diluted_shares_out", period]
                    if pd.notna(ebitda) and ebitda > 0 and pd.notna(shares) and shares > 0:
                        mkt_cap = price * shares
                        nd = 0.0
                        if not bal.empty and period in bal.columns:
                            for debt_item in ("long_term_debt", "short_term_debt"):
                                if debt_item in bal.index:
                                    v = bal.at[debt_item, period]
                                    nd += v if pd.notna(v) else 0.0
                            if "cash_and_equivalents" in bal.index:
                                v = bal.at["cash_and_equivalents", period]
                                nd -= v if pd.notna(v) else 0.0
                        ev = mkt_cap + nd
                        hist_ev_ebitda.append(ev / ebitda)

                # Price / FCF
                if (
                    not cf.empty
                    and "free_cash_flow" in cf.index
                    and period in cf.columns
                    and "diluted_shares_out" in inc.index
                ):
                    fcf = cf.at["free_cash_flow", period]
                    shares = inc.at["diluted_shares_out", period]
                    if pd.notna(fcf) and fcf > 0 and pd.notna(shares) and shares > 0:
                        hist_p_fcf.append(price / (fcf / shares))
        finally:
            con.close()

        avg_pe = float(np.mean(hist_pe[-5:])) if hist_pe else 20.0
        avg_ev_ebitda = float(np.mean(hist_ev_ebitda[-5:])) if hist_ev_ebitda else 12.0
        avg_p_fcf = float(np.mean(hist_p_fcf[-5:])) if hist_p_fcf else 15.0

        # --- Forecast Year-1 values ---
        forecast = self._forecasts.get("base")
        if forecast is None or forecast.empty:
            logger.warning(
                "%s: No base forecast available for multiples valuation",
                self.ticker,
            )
            return {}

        y1 = _sort_periods(list(forecast.columns))[0]

        forecast_eps = (
            float(forecast.at["diluted_eps", y1])
            if "diluted_eps" in forecast.index
            else 0.0
        )
        forecast_ebitda = (
            float(forecast.at["ebitda", y1])
            if "ebitda" in forecast.index
            else 0.0
        )
        forecast_fcf = (
            float(forecast.at["free_cash_flow", y1])
            if "free_cash_flow" in forecast.index
            else 0.0
        )

        bs = self._get_balance_sheet_items()
        current_price = self._get_current_price()

        # --- Build result with only valid methods ---
        methods: dict[str, dict] = {}
        skipped: list[str] = []

        # Forward P/E -- skip if EPS <= 0
        if forecast_eps > 0:
            implied_pe = avg_pe * forecast_eps
            methods["forward_pe"] = {
                "hist_avg_multiple": round(avg_pe, 1),
                "forecast_eps": round(forecast_eps, 2),
                "implied_price": round(implied_pe, 2),
            }
        else:
            skipped.append(
                f"Forward P/E skipped (EPS={forecast_eps:.2f})"
            )
            logger.info(
                "%s: Forward P/E skipped because forecast EPS (%.2f) is "
                "zero or negative",
                self.ticker,
                forecast_eps,
            )

        # EV/EBITDA -- skip if EBITDA <= 0
        if forecast_ebitda > 0:
            implied_ev = avg_ev_ebitda * forecast_ebitda
            implied_ev_ebitda_price = (
                (implied_ev - bs["net_debt"]) / bs["diluted_shares"]
                if bs["diluted_shares"] > 0
                else 0.0
            )
            methods["ev_ebitda"] = {
                "hist_avg_multiple": round(avg_ev_ebitda, 1),
                "forecast_ebitda": forecast_ebitda,
                "implied_price": round(implied_ev_ebitda_price, 2),
            }
        else:
            skipped.append(
                f"EV/EBITDA skipped (EBITDA={forecast_ebitda:,.0f})"
            )
            logger.info(
                "%s: EV/EBITDA skipped because forecast EBITDA (%.0f) is "
                "zero or negative",
                self.ticker,
                forecast_ebitda,
            )

        # Price/FCF -- skip if FCF <= 0
        if forecast_fcf > 0:
            fcf_per_share = (
                forecast_fcf / bs["diluted_shares"]
                if bs["diluted_shares"] > 0
                else 0.0
            )
            implied_p_fcf = avg_p_fcf * fcf_per_share
            methods["price_fcf"] = {
                "hist_avg_multiple": round(avg_p_fcf, 1),
                "forecast_fcf_per_share": round(fcf_per_share, 2),
                "implied_price": round(implied_p_fcf, 2),
            }
        else:
            skipped.append(
                f"Price/FCF skipped (FCF={forecast_fcf:,.0f})"
            )
            logger.info(
                "%s: Price/FCF skipped because forecast FCF (%.0f) is "
                "zero or negative",
                self.ticker,
                forecast_fcf,
            )

        result: dict = {
            "forward_pe": methods.get("forward_pe", {
                "hist_avg_multiple": round(avg_pe, 1),
                "forecast_eps": round(forecast_eps, 2),
                "implied_price": 0.0,
                "skipped": True,
            }),
            "ev_ebitda": methods.get("ev_ebitda", {
                "hist_avg_multiple": round(avg_ev_ebitda, 1),
                "forecast_ebitda": forecast_ebitda,
                "implied_price": 0.0,
                "skipped": True,
            }),
            "price_fcf": methods.get("price_fcf", {
                "hist_avg_multiple": round(avg_p_fcf, 1),
                "forecast_fcf_per_share": round(
                    forecast_fcf / bs["diluted_shares"]
                    if bs["diluted_shares"] > 0
                    else 0.0,
                    2,
                ),
                "implied_price": 0.0,
                "skipped": True,
            }),
            "current_price": current_price,
        }

        if skipped:
            result["multiples_warning"] = "; ".join(skipped)

        return result
