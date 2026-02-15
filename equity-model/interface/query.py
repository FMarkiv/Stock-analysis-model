"""
Natural language query interface for the equity model.

Provides an interactive CLI that translates plain-English queries into
the appropriate model function calls.  Supports data queries, assumption
changes, valuation queries, and report generation.

Usage (standalone)::

    python -m interface.query AAPL
"""

import os
import re
import sys
import traceback

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from db.schema import get_connection, init_db
from data.ingest import ingest_all
from model.statements import FinancialModel
from model.dcf import DCFValuation
from model.scenarios import run_scenarios, monte_carlo
from output.charts import generate_full_report, football_field

# ---------------------------------------------------------------------------
# Console
# ---------------------------------------------------------------------------

console = Console()

# ---------------------------------------------------------------------------
# Default DB path
# ---------------------------------------------------------------------------

_DEFAULT_DB = os.path.join(_PROJECT_ROOT, "data", "equity.duckdb")


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def run_full_pipeline(ticker: str, db_path: str | None = None) -> dict:
    """Run the full pipeline: ingest -> model -> forecast -> DCF -> report.

    Returns a dict with the initialised objects so the interactive loop
    can reuse them without re-creating everything.
    """
    db = db_path or _DEFAULT_DB
    ticker = ticker.upper()

    # 1. Initialise database
    console.print("\n[bold blue]Step 1/6[/] Initialising database...")
    con = init_db(db)
    con.close()

    # 2. Ingest data
    console.print("[bold blue]Step 2/6[/] Ingesting data...")
    try:
        ingest_all(ticker)
    except Exception as exc:
        console.print(f"  [yellow]Warning during ingest:[/] {exc}")

    # 3. Build financial model & compute metrics
    console.print("[bold blue]Step 3/6[/] Building financial model...")
    model = FinancialModel(ticker, db)
    model.compute_historical_metrics()

    # 4. Generate forecasts for all scenarios
    console.print("[bold blue]Step 4/6[/] Generating forecasts (base/bull/bear)...")
    model.forecast(years=5, scenario="base")
    try:
        scenarios_df = run_scenarios(ticker, db)
    except Exception as exc:
        console.print(f"  [yellow]Warning during scenarios:[/] {exc}")
        scenarios_df = None

    # 5. Run DCF valuation
    console.print("[bold blue]Step 5/6[/] Running DCF valuation...")
    dcf = DCFValuation(ticker, db)
    dcf.compute_wacc()
    base_dcf = dcf.dcf_valuation(scenario="base")

    # 6. Generate report
    console.print("[bold blue]Step 6/6[/] Generating report...")
    try:
        report_path = generate_full_report(ticker, db)
        console.print(f"  Report saved: [green]{report_path}[/]")
    except Exception as exc:
        console.print(f"  [yellow]Warning during report:[/] {exc}")
        report_path = None

    # Summary
    console.print()
    _print_valuation_summary(ticker, base_dcf, dcf)

    return {
        "ticker": ticker,
        "db_path": db,
        "model": model,
        "dcf": dcf,
        "base_dcf": base_dcf,
        "scenarios_df": scenarios_df,
        "report_path": report_path,
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _fmt_number(val, pct: bool = False) -> str:
    """Format a number for display."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "-"
    if pct:
        return f"{val:.2%}"
    if abs(val) >= 1e9:
        return f"${val / 1e9:,.1f}B"
    if abs(val) >= 1e6:
        return f"${val / 1e6:,.1f}M"
    if abs(val) >= 1e3:
        return f"${val / 1e3:,.1f}K"
    return f"${val:,.2f}"


def _print_valuation_summary(ticker: str, base_dcf: dict, dcf: DCFValuation) -> None:
    """Print a concise valuation summary panel."""
    current = base_dcf.get("current_price", 0)
    implied = base_dcf.get("implied_price", 0)
    upside = base_dcf.get("upside_downside", 0)
    wacc = dcf._wacc_result.get("wacc", 0) if dcf._wacc_result else 0

    color = "green" if upside > 0 else "red"

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold")
    table.add_column(justify="right")
    table.add_row("Current Price", f"${current:,.2f}")
    table.add_row("Fair Value (Base)", f"${implied:,.2f}")
    table.add_row("Upside/Downside", f"[{color}]{upside:+.1%}[/]")
    table.add_row("WACC", f"{wacc:.2%}")

    console.print(Panel(table, title=f"[bold]{ticker} Valuation Summary[/]",
                        border_style="blue"))


def _display_dataframe(df: pd.DataFrame, title: str = "") -> None:
    """Render a pandas DataFrame as a rich Table."""
    if df.empty:
        console.print("[yellow]No data available.[/]")
        return

    table = Table(title=title, show_lines=True)
    table.add_column("", style="bold")  # index / row label

    for col in df.columns:
        table.add_column(str(col), justify="right")

    for idx, row in df.iterrows():
        cells = []
        for val in row:
            if pd.isna(val):
                cells.append("-")
            elif isinstance(val, float):
                if abs(val) < 1 and abs(val) > 0:
                    cells.append(f"{val:.2%}")
                elif abs(val) >= 1e6:
                    cells.append(f"{val / 1e6:,.1f}M")
                else:
                    cells.append(f"{val:,.2f}")
            else:
                cells.append(str(val))
        table.add_row(str(idx), *cells)

    console.print(table)


def _display_dict(data: dict, title: str = "") -> None:
    """Render a flat dict as a two-column rich Table."""
    table = Table(title=title, show_lines=True)
    table.add_column("Key", style="bold")
    table.add_column("Value", justify="right")

    for k, v in data.items():
        if isinstance(v, dict):
            continue  # skip nested
        if isinstance(v, float):
            if abs(v) < 1 and v != 0:
                table.add_row(str(k), f"{v:.4f}")
            elif abs(v) >= 1e6:
                table.add_row(str(k), f"{v / 1e6:,.1f}M")
            else:
                table.add_row(str(k), f"{v:,.2f}")
        else:
            table.add_row(str(k), str(v))

    console.print(table)


# ---------------------------------------------------------------------------
# Query parser - simple keyword matching
# ---------------------------------------------------------------------------

# Patterns for extracting numbers (e.g. "8%", "0.08", "75%", "12")
_NUMBER_RE = re.compile(r"(\d+\.?\d*)\s*%?")


def _extract_number(text: str) -> float | None:
    """Extract the first number from text, converting % to decimal."""
    m = _NUMBER_RE.search(text)
    if not m:
        return None
    val = float(m.group(1))
    # If the text has a % sign right after the number, treat as percentage
    after = text[m.end(1):m.end(1) + 1]
    if after == "%" or val > 1 and val <= 100:
        val /= 100.0
    return val


def _extract_periods(text: str) -> int:
    """Extract a number of periods from text like 'last 8 quarters'."""
    m = re.search(r"(\d+)\s*(quarter|period|year)", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return 8  # default


def _match_line_item(text: str) -> str | None:
    """Try to match a line item name from natural language."""
    # Map common phrases to DB line_item names
    aliases = {
        "revenue": "total_revenue",
        "total revenue": "total_revenue",
        "sales": "total_revenue",
        "cogs": "cost_of_revenue",
        "cost of revenue": "cost_of_revenue",
        "cost of goods": "cost_of_revenue",
        "gross profit": "gross_profit",
        "gross margin": "gross_margin",
        "operating income": "operating_income",
        "operating profit": "operating_income",
        "op income": "operating_income",
        "ebitda": "ebitda",
        "net income": "net_income",
        "earnings": "net_income",
        "eps": "diluted_eps",
        "diluted eps": "diluted_eps",
        "basic eps": "basic_eps",
        "shares": "diluted_shares_out",
        "share count": "diluted_shares_out",
        "shares outstanding": "diluted_shares_out",
        "r&d": "research_development",
        "research": "research_development",
        "sga": "selling_general_admin",
        "sg&a": "selling_general_admin",
        "selling": "selling_general_admin",
        "interest expense": "interest_expense",
        "interest income": "interest_income",
        "tax": "income_tax",
        "income tax": "income_tax",
        "pretax": "pretax_income",
        "pretax income": "pretax_income",
        "depreciation": "depreciation_amortization",
        "d&a": "depreciation_amortization",
        "amortization": "depreciation_amortization",
        "sbc": "stock_based_comp",
        "stock based comp": "stock_based_comp",
        "stock compensation": "stock_based_comp",
        "cash": "cash_and_equivalents",
        "cash and equivalents": "cash_and_equivalents",
        "receivables": "accounts_receivable",
        "accounts receivable": "accounts_receivable",
        "ar": "accounts_receivable",
        "inventory": "inventory",
        "payables": "accounts_payable",
        "accounts payable": "accounts_payable",
        "ap": "accounts_payable",
        "current assets": "total_current_assets",
        "total assets": "total_assets",
        "ppe": "property_plant_equipment_net",
        "property plant": "property_plant_equipment_net",
        "goodwill": "goodwill",
        "intangibles": "intangible_assets",
        "current liabilities": "total_current_liabilities",
        "total liabilities": "total_liabilities",
        "equity": "total_stockholders_equity",
        "stockholders equity": "total_stockholders_equity",
        "book value": "total_stockholders_equity",
        "retained earnings": "retained_earnings",
        "long term debt": "long_term_debt",
        "lt debt": "long_term_debt",
        "short term debt": "short_term_debt",
        "st debt": "short_term_debt",
        "operating cash flow": "operating_cash_flow",
        "ocf": "operating_cash_flow",
        "capex": "capex",
        "capital expenditure": "capex",
        "free cash flow": "free_cash_flow",
        "fcf": "free_cash_flow",
        "dividends": "dividends_paid",
        "buybacks": "share_repurchases",
        "share repurchases": "share_repurchases",
        "repurchases": "share_repurchases",
        "working capital": "change_in_working_capital",
    }

    text_lower = text.lower()
    # Try longest match first
    for phrase in sorted(aliases.keys(), key=len, reverse=True):
        if phrase in text_lower:
            return aliases[phrase]
    return None


def _match_assumption_param(text: str) -> str | None:
    """Try to match an assumption parameter name from text."""
    aliases = {
        "revenue growth": "revenue_growth",
        "rev growth": "revenue_growth",
        "growth rate": "revenue_growth",
        "growth": "revenue_growth",
        "gross margin": "gross_margin",
        "operating margin": "operating_margin",
        "op margin": "operating_margin",
        "net margin": "net_margin",
        "services margin": "operating_margin",
        "capex": "capex_pct_revenue",
        "capex %": "capex_pct_revenue",
        "capex pct": "capex_pct_revenue",
        "tax rate": "effective_tax_rate",
        "effective tax": "effective_tax_rate",
        "sbc": "sbc_pct_revenue",
        "stock comp": "sbc_pct_revenue",
        "r&d": "rd_pct_revenue",
        "research": "rd_pct_revenue",
        "sga": "sga_pct_revenue",
        "sg&a": "sga_pct_revenue",
        "dso": "days_sales_outstanding",
        "days sales": "days_sales_outstanding",
        "dpo": "days_payable_outstanding",
        "days payable": "days_payable_outstanding",
        "dio": "days_inventory_outstanding",
        "days inventory": "days_inventory_outstanding",
        "buyback": "annual_buyback_pct",
        "dividend": "dividend_pct_ni",
        "payout": "dividend_pct_ni",
    }

    text_lower = text.lower()
    for phrase in sorted(aliases.keys(), key=len, reverse=True):
        if phrase in text_lower:
            return aliases[phrase]
    return None


# ---------------------------------------------------------------------------
# Query categories
# ---------------------------------------------------------------------------

def _classify_query(text: str) -> str:
    """Classify a query into a category using keyword matching."""
    t = text.lower().strip()

    # Exit commands
    if t in ("quit", "exit", "q", "bye"):
        return "exit"

    # Help
    if t in ("help", "?", "commands"):
        return "help"

    # Assumption changes: "set|change|adjust" + parameter + value
    if re.search(r"\b(set|change|adjust|update|make)\b", t):
        return "assumption_change"

    # Reset to defaults
    if re.search(r"\breset\b.*\bdefault", t):
        return "reset_defaults"

    # What-if / scenario analysis
    if re.search(r"\b(what if|what-if|sensitivity|impact|stress)\b", t):
        return "what_if"

    # Monte Carlo
    if re.search(r"\b(monte\s*carlo|simulation|simulate)\b", t):
        return "monte_carlo"

    # Compare scenarios
    if re.search(r"\b(compare|comparison)\b.*\bscenario", t):
        return "compare_scenarios"
    if re.search(r"\bscenario.*\b(compare|comparison|summary)\b", t):
        return "compare_scenarios"

    # Fair value / DCF / valuation
    if re.search(r"\b(fair\s*value|dcf|valuation|intrinsic|worth|target\s*price|implied)\b", t):
        return "valuation"

    # WACC
    if re.search(r"\bwacc\b", t):
        return "wacc"

    # Multiples
    if re.search(r"\b(multiples?|p/?e\b|ev/?ebitda|comps|relative)", t):
        return "multiples"

    # Report generation
    if re.search(r"\b(generate|create|build|run)\b.*\breport\b", t):
        return "generate_report"
    if re.search(r"\breport\b", t):
        return "generate_report"

    # Football field chart
    if re.search(r"\b(football\s*field|valuation\s*range)\b", t):
        return "football_field"

    # Show chart/plot
    if re.search(r"\b(chart|plot|graph|visual)\b", t):
        return "chart"

    # Segment breakdown
    if re.search(r"\bsegment\b", t):
        return "segment"

    # Assumptions display (check before generic data_query)
    if re.search(r"\bassumptions?\b", t):
        return "show_assumptions"

    # Metrics display (only match "metrics"/"ratios" as standalone,
    # not "margin" which could be a specific line-item query)
    if re.search(r"\b(metrics|ratios)\b", t):
        return "show_metrics"

    # Income / balance / cashflow statement
    if re.search(r"\b(income\s*statement|p&l|profit\s*(and|&)\s*loss)\b", t):
        return "income_statement"
    if re.search(r"\b(balance\s*sheet)\b", t):
        return "balance_sheet"
    if re.search(r"\b(cash\s*flow|cashflow)\b", t) and not re.search(r"\bfree\b", t):
        return "cashflow_statement"

    # Data queries: "show|display|what is|get" + line_item
    if re.search(r"\b(show|display|get|what\s*is|what\'s|current|latest|last|history|historical|trend)\b", t):
        return "data_query"

    # Fallback: if we can match a line item, treat as data query
    if _match_line_item(t):
        return "data_query"

    return "unknown"


# ---------------------------------------------------------------------------
# Query handlers
# ---------------------------------------------------------------------------


def _handle_help() -> None:
    """Print available commands."""
    help_text = """
[bold]Data Queries:[/]
  show revenue for last 8 quarters
  what is the current gross margin
  show segment breakdown
  show income statement / balance sheet / cash flow

[bold]Assumptions:[/]
  show assumptions
  set revenue growth to 8% for base case
  reset to defaults

[bold]What-If / Scenarios:[/]
  what if services margin expands to 75%
  compare scenarios
  run monte carlo

[bold]Valuation:[/]
  what is fair value
  show sensitivity to WACC and terminal growth
  show multiples / comps
  show wacc

[bold]Output:[/]
  generate report
  show football field

[bold]Other:[/]
  help          Show this message
  quit / exit   Leave interactive mode
"""
    console.print(Panel(help_text.strip(), title="[bold]Available Commands[/]",
                        border_style="cyan"))


def _handle_data_query(text: str, ctx: dict) -> None:
    """Handle data lookup queries."""
    model: FinancialModel = ctx["model"]
    ticker = ctx["ticker"]

    line_item = _match_line_item(text)
    if not line_item:
        console.print("[yellow]Could not identify a line item. Try 'help' for examples.[/]")
        return

    # Determine quarterly vs annual
    is_quarterly = bool(re.search(r"\bquarter", text.lower()))
    period_type = "quarterly" if is_quarterly else "annual"
    periods = _extract_periods(text)

    series = model.get_historical(line_item, periods=periods, period_type=period_type)

    if series.empty:
        console.print(f"[yellow]No data found for '{line_item}' ({period_type}).[/]")
        return

    # Check if this is a margin/rate metric (values between -1 and 1)
    is_pct = line_item in (
        "gross_margin", "operating_margin", "net_margin", "fcf_margin",
        "revenue_yoy", "roe", "roic",
    )

    table = Table(title=f"{ticker} — {line_item} ({period_type})")
    table.add_column("Period", style="bold")
    table.add_column("Value", justify="right")

    for period, val in series.items():
        if pd.isna(val):
            table.add_row(str(period), "-")
        elif is_pct:
            table.add_row(str(period), f"{val:.2%}")
        elif abs(val) >= 1e6:
            table.add_row(str(period), f"{val / 1e6:,.1f}M")
        else:
            table.add_row(str(period), f"{val:,.2f}")

    console.print(table)


def _handle_statement(statement_type: str, ctx: dict) -> None:
    """Show a full financial statement."""
    model: FinancialModel = ctx["model"]
    df = model.get_statement(statement_type, include_forecast=True, scenario="base")
    if df.empty:
        console.print(f"[yellow]No {statement_type} data available.[/]")
        return

    # Show last 4 historical + forecast
    all_cols = list(df.columns)
    hist_cols = [c for c in all_cols if not c.startswith("FY") or
                 c in model.historical.get(statement_type, pd.DataFrame()).columns]
    fc_cols = [c for c in all_cols if c not in hist_cols]
    show_cols = hist_cols[-4:] + fc_cols

    display_df = df[show_cols] if show_cols else df
    _display_dataframe(display_df, title=f"{ctx['ticker']} — {statement_type.title()} Statement")


def _handle_show_metrics(ctx: dict) -> None:
    """Show computed historical metrics."""
    model: FinancialModel = ctx["model"]
    if model.metrics.empty:
        model.compute_historical_metrics()
    if model.metrics.empty:
        console.print("[yellow]No metrics available.[/]")
        return
    _display_dataframe(model.metrics, title=f"{ctx['ticker']} — Historical Metrics")


def _handle_show_assumptions(ctx: dict) -> None:
    """Show current model assumptions."""
    model: FinancialModel = ctx["model"]
    db = ctx["db_path"]
    con = get_connection(db)
    try:
        rows = con.execute(
            "SELECT scenario, parameter_name, parameter_value "
            "FROM assumptions WHERE ticker = ? ORDER BY scenario, parameter_name",
            [ctx["ticker"]],
        ).fetchall()
    finally:
        con.close()

    if not rows:
        console.print("[yellow]No assumptions stored.[/]")
        return

    table = Table(title=f"{ctx['ticker']} — Model Assumptions", show_lines=True)
    table.add_column("Scenario", style="bold")
    table.add_column("Parameter")
    table.add_column("Value", justify="right")

    for scenario, param, val in rows:
        if abs(val) < 1 and val != 0:
            table.add_row(scenario, param, f"{val:.4f}")
        else:
            table.add_row(scenario, param, f"{val:,.2f}")

    console.print(table)


def _handle_assumption_change(text: str, ctx: dict) -> None:
    """Parse and apply an assumption change."""
    model: FinancialModel = ctx["model"]
    param = _match_assumption_param(text)
    if not param:
        console.print("[yellow]Could not identify the assumption to change. "
                      "Try: set revenue growth to 8%[/]")
        return

    value = _extract_number(text)
    if value is None:
        console.print("[yellow]Could not find a numeric value. "
                      "Try: set revenue growth to 8%[/]")
        return

    # Determine scenario (default to base)
    scenario = "base"
    if "bull" in text.lower():
        scenario = "bull"
    elif "bear" in text.lower():
        scenario = "bear"

    console.print(f"Setting [bold]{param}[/] = {value:.4f} for [bold]{scenario}[/] case...")
    model.set_assumption(scenario, param, value)

    # Re-run DCF
    console.print("Re-running DCF valuation...")
    dcf = DCFValuation(ctx["ticker"], ctx["db_path"])
    dcf.compute_wacc()
    new_dcf = dcf.dcf_valuation(scenario=scenario)

    ctx["dcf"] = dcf
    ctx["base_dcf"] = new_dcf

    _print_valuation_summary(ctx["ticker"], new_dcf, dcf)


def _handle_reset_defaults(ctx: dict) -> None:
    """Reset assumptions to auto-generated defaults."""
    model: FinancialModel = ctx["model"]
    db = ctx["db_path"]
    ticker = ctx["ticker"]

    # Delete existing assumptions
    con = get_connection(db)
    try:
        con.execute("DELETE FROM assumptions WHERE ticker = ?", [ticker])
    finally:
        con.close()

    # Re-generate
    console.print("Regenerating default assumptions...")
    model._generate_default_assumptions("base")
    model.forecast(years=5, scenario="base")

    # Re-run DCF
    dcf = DCFValuation(ticker, db)
    dcf.compute_wacc()
    new_dcf = dcf.dcf_valuation(scenario="base")

    ctx["dcf"] = dcf
    ctx["base_dcf"] = new_dcf

    console.print("[green]Assumptions reset to defaults.[/]")
    _print_valuation_summary(ticker, new_dcf, dcf)


def _handle_what_if(text: str, ctx: dict) -> None:
    """Handle what-if scenario queries."""
    # Try to extract an assumption change
    param = _match_assumption_param(text)
    value = _extract_number(text)

    if param and value is not None:
        # Save old value, apply change, show delta
        model: FinancialModel = ctx["model"]
        old_assumptions = model._load_or_generate_assumptions("base")
        old_val = old_assumptions.get(param, 0)

        console.print(f"[bold]What-if:[/] {param} changes from {old_val:.4f} to {value:.4f}")

        model.set_assumption("base", param, value)

        dcf = DCFValuation(ctx["ticker"], ctx["db_path"])
        dcf.compute_wacc()
        new_dcf = dcf.dcf_valuation(scenario="base")
        old_price = ctx["base_dcf"].get("implied_price", 0)
        new_price = new_dcf.get("implied_price", 0)
        delta = new_price - old_price

        color = "green" if delta > 0 else "red"
        console.print(f"  Fair value: ${old_price:,.2f} -> ${new_price:,.2f} "
                      f"([{color}]{delta:+,.2f}[/])")

        ctx["dcf"] = dcf
        ctx["base_dcf"] = new_dcf
    else:
        # Show sensitivity table
        console.print("Running sensitivity analysis (WACC vs Terminal Growth)...")
        dcf: DCFValuation = ctx["dcf"]
        sens = dcf.sensitivity_table()
        _display_dataframe(sens, title="Sensitivity: Terminal Growth (rows) vs WACC (cols)")


def _handle_valuation(ctx: dict) -> None:
    """Show DCF valuation result."""
    dcf: DCFValuation = ctx["dcf"]
    result = ctx["base_dcf"]

    table = Table(title=f"{ctx['ticker']} — DCF Valuation", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    display_keys = [
        ("Implied Price", "implied_price", "${:,.2f}"),
        ("Current Price", "current_price", "${:,.2f}"),
        ("Upside/Downside", "upside_downside", "{:+.1%}"),
        ("Enterprise Value", "enterprise_value", "${:,.0f}"),
        ("Equity Value", "equity_value", "${:,.0f}"),
        ("PV of FCFs", "pv_fcfs", "${:,.0f}"),
        ("Terminal Value (Perpetuity)", "terminal_value_perpetuity", "${:,.0f}"),
        ("Terminal Value (Exit Multiple)", "terminal_value_exit_multiple", "${:,.0f}"),
        ("WACC", "wacc", "{:.2%}"),
        ("Terminal Growth", "terminal_growth", "{:.2%}"),
        ("Exit Multiple", "exit_multiple", "{:.1f}x"),
        ("Net Debt", "net_debt", "${:,.0f}"),
        ("Diluted Shares", "diluted_shares", "{:,.0f}"),
    ]

    for label, key, fmt in display_keys:
        val = result.get(key)
        if val is not None:
            table.add_row(label, fmt.format(val))

    console.print(table)


def _handle_wacc(ctx: dict) -> None:
    """Show WACC breakdown."""
    dcf: DCFValuation = ctx["dcf"]
    wacc_info = dcf._wacc_result
    if not wacc_info:
        wacc_info = dcf.compute_wacc()

    table = Table(title=f"{ctx['ticker']} — WACC Breakdown", show_lines=True)
    table.add_column("Component", style="bold")
    table.add_column("Value", justify="right")

    display = [
        ("WACC", "wacc", "{:.2%}"),
        ("Cost of Equity", "cost_of_equity", "{:.2%}"),
        ("Cost of Debt (after-tax)", "cost_of_debt", "{:.2%}"),
        ("Cost of Debt (pre-tax)", "cost_of_debt_pretax", "{:.2%}"),
        ("Beta", "beta", "{:.2f}"),
        ("Risk-Free Rate", "risk_free_rate", "{:.2%}"),
        ("Equity Risk Premium", "equity_risk_premium", "{:.2%}"),
        ("Tax Rate", "tax_rate", "{:.2%}"),
        ("Weight Equity", "weight_equity", "{:.1%}"),
        ("Weight Debt", "weight_debt", "{:.1%}"),
        ("Debt/Equity", "debt_to_equity", "{:.2f}"),
        ("Total Debt", "total_debt", "${:,.0f}"),
        ("Market Cap", "market_cap", "${:,.0f}"),
    ]

    for label, key, fmt in display:
        val = wacc_info.get(key)
        if val is not None:
            table.add_row(label, fmt.format(val))

    console.print(table)


def _handle_multiples(ctx: dict) -> None:
    """Show multiples-based valuation."""
    dcf: DCFValuation = ctx["dcf"]
    result = dcf.multiples_valuation()

    if not result:
        console.print("[yellow]Multiples valuation not available.[/]")
        return

    table = Table(title=f"{ctx['ticker']} — Relative Valuation", show_lines=True)
    table.add_column("Method", style="bold")
    table.add_column("Hist Avg Multiple", justify="right")
    table.add_column("Implied Price", justify="right")

    for method_key, label in [("forward_pe", "Forward P/E"),
                               ("ev_ebitda", "EV/EBITDA"),
                               ("price_fcf", "Price/FCF")]:
        m = result.get(method_key, {})
        if m:
            table.add_row(label,
                          f"{m.get('hist_avg_multiple', 0):.1f}x",
                          f"${m.get('implied_price', 0):,.2f}")

    table.add_row("", "", "")
    table.add_row("Current Price", "", f"${result.get('current_price', 0):,.2f}")
    console.print(table)


def _handle_compare_scenarios(ctx: dict) -> None:
    """Show bull/base/bear scenario comparison."""
    ticker = ctx["ticker"]
    db = ctx["db_path"]

    console.print("Running scenario analysis...")
    try:
        scenarios_df = run_scenarios(ticker, db)
    except Exception as exc:
        console.print(f"[red]Error running scenarios:[/] {exc}")
        return

    ctx["scenarios_df"] = scenarios_df
    _display_dataframe(scenarios_df, title=f"{ticker} — Scenario Comparison")


def _handle_monte_carlo(ctx: dict) -> None:
    """Run Monte Carlo simulation."""
    ticker = ctx["ticker"]
    db = ctx["db_path"]

    console.print("Running Monte Carlo simulation (10,000 iterations)...")
    try:
        mc = monte_carlo(ticker, db, iterations=10_000)
    except Exception as exc:
        console.print(f"[red]Error running Monte Carlo:[/] {exc}")
        return

    table = Table(title=f"{ticker} — Monte Carlo Results", show_lines=True)
    table.add_column("Percentile", style="bold")
    table.add_column("Fair Value", justify="right")

    pcts = mc["percentiles"]
    for label, key in [("10th", "p10"), ("25th", "p25"),
                       ("50th (Median)", "p50"),
                       ("75th", "p75"), ("90th", "p90")]:
        table.add_row(label, f"${pcts[key]:,.2f}")

    table.add_row("", "")
    table.add_row("Mean", f"${mc['mean']:,.2f}")
    table.add_row("Std Dev", f"${mc['std']:,.2f}")
    table.add_row("Iterations", f"{mc['iterations']:,}")

    console.print(table)


def _handle_segment(ctx: dict) -> None:
    """Show segment breakdown from the database."""
    ticker = ctx["ticker"]
    db = ctx["db_path"]

    con = get_connection(db)
    try:
        rows = con.execute(
            "SELECT period, segment_name, revenue, operating_income "
            "FROM segments WHERE ticker = ? "
            "ORDER BY period DESC, revenue DESC",
            [ticker],
        ).fetchall()
    finally:
        con.close()

    if not rows:
        console.print("[yellow]No segment data available.[/]")
        return

    table = Table(title=f"{ticker} — Segment Breakdown", show_lines=True)
    table.add_column("Period", style="bold")
    table.add_column("Segment")
    table.add_column("Revenue", justify="right")
    table.add_column("Op Income", justify="right")

    for period, name, rev, oi in rows:
        rev_str = f"{rev / 1e6:,.1f}M" if rev and rev >= 1e6 else (f"{rev:,.0f}" if rev else "-")
        oi_str = f"{oi / 1e6:,.1f}M" if oi and oi >= 1e6 else (f"{oi:,.0f}" if oi else "-")
        table.add_row(str(period), str(name), rev_str, oi_str)

    console.print(table)


def _handle_generate_report(ctx: dict) -> None:
    """Generate the full HTML report."""
    ticker = ctx["ticker"]
    db = ctx["db_path"]

    console.print("Generating full report...")
    try:
        path = generate_full_report(ticker, db)
        console.print(f"[green]Report saved:[/] {path}")
        ctx["report_path"] = path
    except Exception as exc:
        console.print(f"[red]Error generating report:[/] {exc}")


def _handle_football_field(ctx: dict) -> None:
    """Generate and save the football field chart."""
    ticker = ctx["ticker"]
    db = ctx["db_path"]

    console.print("Generating football field chart...")
    try:
        fig = football_field(ticker, db)
        output_dir = os.path.join(_PROJECT_ROOT, "output")
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{ticker}_football_field.html")
        fig.write_html(path)
        console.print(f"[green]Chart saved:[/] {path}")
    except Exception as exc:
        console.print(f"[red]Error generating chart:[/] {exc}")


def _handle_chart(text: str, ctx: dict) -> None:
    """Handle generic chart requests by generating the full report."""
    console.print("Generating full report with all charts...")
    _handle_generate_report(ctx)


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------


def interactive_loop(ctx: dict) -> None:
    """Run the interactive query loop."""
    ticker = ctx["ticker"]

    console.print(f"\n[bold green]Interactive mode for {ticker}[/]")
    console.print("Type a question or command. Type [bold]help[/] for options, "
                  "[bold]quit[/] to exit.\n")

    while True:
        try:
            query = console.input(f"[bold blue]{ticker}>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye!")
            break

        if not query:
            continue

        category = _classify_query(query)

        try:
            if category == "exit":
                console.print("Goodbye!")
                break
            elif category == "help":
                _handle_help()
            elif category == "data_query":
                _handle_data_query(query, ctx)
            elif category == "income_statement":
                _handle_statement("income", ctx)
            elif category == "balance_sheet":
                _handle_statement("balance", ctx)
            elif category == "cashflow_statement":
                _handle_statement("cashflow", ctx)
            elif category == "show_metrics":
                _handle_show_metrics(ctx)
            elif category == "show_assumptions":
                _handle_show_assumptions(ctx)
            elif category == "assumption_change":
                _handle_assumption_change(query, ctx)
            elif category == "reset_defaults":
                _handle_reset_defaults(ctx)
            elif category == "what_if":
                _handle_what_if(query, ctx)
            elif category == "valuation":
                _handle_valuation(ctx)
            elif category == "wacc":
                _handle_wacc(ctx)
            elif category == "multiples":
                _handle_multiples(ctx)
            elif category == "compare_scenarios":
                _handle_compare_scenarios(ctx)
            elif category == "monte_carlo":
                _handle_monte_carlo(ctx)
            elif category == "segment":
                _handle_segment(ctx)
            elif category == "generate_report":
                _handle_generate_report(ctx)
            elif category == "football_field":
                _handle_football_field(ctx)
            elif category == "chart":
                _handle_chart(query, ctx)
            else:
                console.print("[yellow]I didn't understand that. Type 'help' for "
                              "available commands.[/]")
        except Exception as exc:
            console.print(f"[red]Error:[/] {exc}")
            console.print(f"[dim]{traceback.format_exc()}[/]")


# ---------------------------------------------------------------------------
# CLI entry point: python -m interface.query AAPL
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for the interactive query interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Interactive equity model query interface",
    )
    parser.add_argument("ticker", help="Stock ticker symbol (e.g. AAPL)")
    parser.add_argument("--db", default=None, help="Path to DuckDB database")
    parser.add_argument(
        "--skip-pipeline", action="store_true",
        help="Skip the full pipeline and jump straight to interactive mode",
    )
    args = parser.parse_args()

    ticker = args.ticker.upper()
    db = args.db or _DEFAULT_DB

    if args.skip_pipeline:
        # Just load existing data
        console.print(f"[bold]Loading existing data for {ticker}...[/]")
        init_db(db).close()
        model = FinancialModel(ticker, db)
        model.compute_historical_metrics()
        dcf = DCFValuation(ticker, db)
        dcf.compute_wacc()
        base_dcf = dcf.dcf_valuation(scenario="base")

        ctx = {
            "ticker": ticker,
            "db_path": db,
            "model": model,
            "dcf": dcf,
            "base_dcf": base_dcf,
            "scenarios_df": None,
            "report_path": None,
        }
    else:
        ctx = run_full_pipeline(ticker, db)

    interactive_loop(ctx)


if __name__ == "__main__":
    main()
