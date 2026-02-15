"""
Entry point for the equity model.

Orchestrates the full analysis pipeline: data ingestion from Yahoo
Finance and SEC EDGAR, three-statement financial modelling, DCF and
multiples-based valuation, scenario analysis, and interactive
querying.

Usage
-----
::

    python main.py AAPL                   Full pipeline: ingest -> model -> report -> interactive
    python main.py AAPL --ingest-only     Only ingest data (no model/report)
    python main.py AAPL --report-only     Only generate report (assumes data exists)
    python main.py AAPL --interactive     Skip pipeline, jump to interactive mode
    python main.py init                   Initialise (or re-initialise) the database
    python main.py reset                  Drop and recreate all tables
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta

import yaml

from db.schema import get_connection, init_db, reset_schema
from data.ingest import ingest_all
from model.statements import FinancialModel
from model.dcf import DCFValuation
from model.scenarios import run_scenarios
from output.charts import generate_full_report
from interface.query import run_full_pipeline, interactive_loop, console

logger = logging.getLogger(__name__)

# Data older than this threshold triggers a refresh prompt.
DATA_FRESHNESS_THRESHOLD = timedelta(hours=24)


def load_config(path: str = "config.yaml") -> dict:
    """Load ``config.yaml`` and return the parsed dict (empty dict on error)."""
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def check_data_freshness(ticker: str, db_path: str) -> bool:
    """Check whether data for *ticker* was ingested within the last 24 hours.

    Returns ``True`` if the data is fresh (or freshness cannot be determined),
    ``False`` if the data is stale and the user should be prompted to refresh.
    """
    try:
        con = get_connection(db_path)
        try:
            row = con.execute(
                "SELECT last_ingested FROM company WHERE ticker = ?",
                [ticker],
            ).fetchone()
        finally:
            con.close()
    except Exception:
        # Table/column may not exist yet — treat as fresh (will be ingested)
        return True

    if row is None or row[0] is None:
        return True  # never ingested — pipeline will handle it

    last_ingested = row[0]
    # DuckDB may return a datetime or a string depending on driver version
    if isinstance(last_ingested, str):
        last_ingested = datetime.fromisoformat(last_ingested)

    age = datetime.now() - last_ingested
    if age > DATA_FRESHNESS_THRESHOLD:
        console.print(
            f"\n[yellow]Data for {ticker} was last ingested "
            f"{age.total_seconds() / 3600:.1f} hours ago.[/]"
        )
        try:
            answer = console.input(
                "[bold]Refresh data? [Y/n]:[/] "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        return answer in ("y", "yes", "")

    return True  # fresh


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate pipeline step."""
    parser = argparse.ArgumentParser(
        description="Equity research model — ingest, analyse, and query stocks",
    )
    parser.add_argument(
        "ticker",
        nargs="?",
        help="Stock ticker (e.g. AAPL) or command (init, reset)",
    )
    parser.add_argument(
        "--ingest-only", action="store_true",
        help="Only ingest data, do not run model or report",
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Only generate report (data must already exist)",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Skip pipeline, jump straight to interactive mode (data must exist)",
    )
    parser.add_argument(
        "--db", default=None,
        help="Path to DuckDB database file",
    )

    args = parser.parse_args()

    config = load_config()
    db_path = args.db or config.get("database", {}).get("path", "data/equity.duckdb")

    # ── Legacy commands: init / reset ──────────────────────────────────
    if args.ticker in ("init", "reset"):
        if args.ticker == "init":
            con = init_db(db_path)
            tables = con.execute("SHOW TABLES").fetchall()
            print("Database initialised. Tables:", [t[0] for t in tables])
            con.close()
        else:
            con = reset_schema(db_path)
            tables = con.execute("SHOW TABLES").fetchall()
            print("Schema reset. Tables:", [t[0] for t in tables])
            con.close()
        return

    # ── Ticker is required for everything else ─────────────────────────
    if not args.ticker:
        parser.print_help()
        sys.exit(1)

    ticker = args.ticker.upper()

    # Ensure DB is initialised
    con = init_db(db_path)
    con.close()

    # ── --ingest-only ──────────────────────────────────────────────────
    if args.ingest_only:
        console.print(f"[bold]Ingesting data for {ticker}...[/]")
        ingest_all(ticker)
        console.print("[green]Ingest complete.[/]")
        return

    # ── --report-only ──────────────────────────────────────────────────
    if args.report_only:
        console.print(f"[bold]Generating report for {ticker}...[/]")
        try:
            path = generate_full_report(ticker, db_path)
            console.print(f"[green]Report saved:[/] {path}")
        except Exception as exc:
            console.print(f"[red]Error generating report:[/] {exc}")
            console.print("Run without --report-only first to ingest data and build the model.")
            sys.exit(1)
        return

    # ── --interactive (skip pipeline) ──────────────────────────────────
    if args.interactive:
        # Check data freshness before loading
        should_refresh = not check_data_freshness(ticker, db_path)
        if should_refresh:
            console.print(f"[bold]Refreshing data for {ticker}...[/]")
            ingest_all(ticker)

        console.print(f"[bold]Loading existing data for {ticker}...[/]")
        try:
            model = FinancialModel(ticker, db_path)
            model.compute_historical_metrics()
            dcf = DCFValuation(ticker, db_path)
            dcf.compute_wacc()
            base_dcf = dcf.dcf_valuation(scenario="base")
        except Exception as exc:
            console.print(f"[red]Error loading data:[/] {exc}")
            console.print("Run without --interactive first to ingest data.")
            sys.exit(1)

        ctx = {
            "ticker": ticker,
            "db_path": db_path,
            "model": model,
            "dcf": dcf,
            "base_dcf": base_dcf,
            "scenarios_df": None,
            "report_path": None,
        }
        interactive_loop(ctx)
        return

    # ── Default: full pipeline -> interactive ──────────────────────────
    ctx = run_full_pipeline(ticker, db_path)
    interactive_loop(ctx)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s [%(name)s] %(message)s",
    )
    main()
