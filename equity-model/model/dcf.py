"""
Discounted cash-flow valuation engine.

Provides ``DCFValuation`` for WACC calculation, DCF valuation,
sensitivity analysis, and relative-multiples valuation.
"""

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load config.yaml from the project root."""
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

        Falls back to 1.0 if fewer than 20 weekly observations are
        available for either the ticker or SPY.
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
            return 1.0

        tk_ret = tk[common].pct_change().dropna()
        sp_ret = sp[common].pct_change().dropna()
        common_ret = tk_ret.index.intersection(sp_ret.index)
        if len(common_ret) < 20:
            return 1.0

        x = sp_ret[common_ret].values
        y = tk_ret[common_ret].values
        cov = np.cov(x, y)
        if cov[0, 0] == 0:
            return 1.0

        return float(cov[0, 1] / cov[0, 0])

    # ------------------------------------------------------------------
    # Market helpers
    # ------------------------------------------------------------------

    def _get_market_cap(self) -> float | None:
        """Most-recent price * diluted shares outstanding."""
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
        """Return the most-recent closing price from the DB."""
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
            raise ValueError(f"No price data found for {self.ticker}")
        return row[0]

    def _get_balance_sheet_items(self) -> dict:
        """Extract latest total_debt, cash, net_debt, diluted_shares."""
        bal = self.model.historical.get("balance", pd.DataFrame())
        inc = self.model.historical.get("income", pd.DataFrame())

        total_debt = 0.0
        cash = 0.0
        if not bal.empty:
            for item in ("long_term_debt", "short_term_debt"):
                if item in bal.index:
                    s = bal.loc[item].dropna()
                    if len(s) > 0:
                        total_debt += float(s.iloc[-1])
            if "cash_and_equivalents" in bal.index:
                c = bal.loc["cash_and_equivalents"].dropna()
                if len(c) > 0:
                    cash = float(c.iloc[-1])

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

        Parameters
        ----------
        scenario : str
            ``'base'``, ``'bull'``, or ``'bear'``.

        Returns
        -------
        dict
            ``implied_price``, ``current_price``, ``upside_downside``,
            ``enterprise_value``, ``equity_value``, plus detailed
            intermediate results.
        """
        # Ensure WACC is available
        if self._wacc_result is None:
            self.compute_wacc()

        wacc = self._wacc_result["wacc"]
        tax_rate = self._wacc_result["tax_rate"]
        terminal_growth = self.model_params.get("terminal_growth_rate", 0.025)

        # Ensure forecast exists for the requested scenario
        if scenario not in self._forecasts:
            self._forecasts[scenario] = self.model.forecast(
                years=5, scenario=scenario,
            )
        forecast = self._forecasts[scenario]
        forecast_periods = _sort_periods(list(forecast.columns))

        # ---- Project unlevered FCF ----
        # UFCF = EBIT*(1-t) + D&A - CapEx - ΔWC
        # In our storage: capex is negative, change_in_working_capital
        # encodes the cash-flow impact (negative = WC increase = cash used).
        # So: UFCF = EBIT*(1-t) + D&A + capex_stored + Δwc_stored
        projected_fcfs: dict[str, float] = {}
        for period in forecast_periods:
            ebit = float(forecast.at["operating_income", period]) if "operating_income" in forecast.index else 0.0
            da = float(forecast.at["depreciation_amortization", period]) if "depreciation_amortization" in forecast.index else 0.0
            capex = float(forecast.at["capex", period]) if "capex" in forecast.index else 0.0
            dwc = float(forecast.at["change_in_working_capital", period]) if "change_in_working_capital" in forecast.index else 0.0

            ufcf = ebit * (1 - tax_rate) + da + capex + dwc
            projected_fcfs[period] = ufcf

        # ---- Discount projected FCFs ----
        pv_fcfs = 0.0
        fcf_values: list[float] = []
        for i, period in enumerate(forecast_periods):
            pv_fcfs += projected_fcfs[period] / (1 + wacc) ** (i + 1)
            fcf_values.append(projected_fcfs[period])

        n_years = len(forecast_periods)
        final_fcf = fcf_values[-1] if fcf_values else 0.0

        # ---- Terminal value – perpetuity growth ----
        if wacc > terminal_growth:
            tv_perpetuity = final_fcf * (1 + terminal_growth) / (wacc - terminal_growth)
        else:
            tv_perpetuity = final_fcf * 25  # safety cap
        pv_tv_perpetuity = tv_perpetuity / (1 + wacc) ** n_years

        # ---- Terminal value – exit multiple ----
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
        implied_price = equity_value / bs["diluted_shares"] if bs["diluted_shares"] > 0 else 0.0

        try:
            current_price = self._get_current_price()
        except ValueError:
            current_price = 0.0

        upside_downside = (
            (implied_price / current_price - 1) if current_price > 0 else 0.0
        )

        return {
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
        }

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

        Returns
        -------
        dict
            ``forward_pe``, ``ev_ebitda``, ``price_fcf`` (each a dict
            with ``hist_avg_multiple``, implied inputs, and
            ``implied_price``), plus ``current_price``.
        """
        inc = self.model.historical.get("income", pd.DataFrame())
        cf = self.model.historical.get("cashflow", pd.DataFrame())
        bal = self.model.historical.get("balance", pd.DataFrame())

        if inc.empty:
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

        implied_pe = avg_pe * forecast_eps if forecast_eps > 0 else 0.0

        implied_ev = avg_ev_ebitda * forecast_ebitda
        implied_ev_ebitda_price = (
            (implied_ev - bs["net_debt"]) / bs["diluted_shares"]
            if bs["diluted_shares"] > 0
            else 0.0
        )

        fcf_per_share = forecast_fcf / bs["diluted_shares"] if bs["diluted_shares"] > 0 else 0.0
        implied_p_fcf = avg_p_fcf * fcf_per_share

        try:
            current_price = self._get_current_price()
        except ValueError:
            current_price = 0.0

        return {
            "forward_pe": {
                "hist_avg_multiple": round(avg_pe, 1),
                "forecast_eps": round(forecast_eps, 2),
                "implied_price": round(implied_pe, 2),
            },
            "ev_ebitda": {
                "hist_avg_multiple": round(avg_ev_ebitda, 1),
                "forecast_ebitda": forecast_ebitda,
                "implied_price": round(implied_ev_ebitda_price, 2),
            },
            "price_fcf": {
                "hist_avg_multiple": round(avg_p_fcf, 1),
                "forecast_fcf_per_share": round(fcf_per_share, 2),
                "implied_price": round(implied_p_fcf, 2),
            },
            "current_price": current_price,
        }
