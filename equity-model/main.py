"""
Entry point for the equity model.

Usage:
    python main.py AAPL                   Full pipeline: ingest -> model -> report -> interactive
    python main.py AAPL --ingest-only     Only ingest data (no model/report)
    python main.py AAPL --report-only     Only generate report (assumes data exists)
    python main.py AAPL --interactive     Skip pipeline, jump to interactive mode
    python main.py init                   Initialise (or re-initialise) the database
    python main.py reset                  Drop and recreate all tables
"""

import argparse
import sys

import yaml

from db.schema import init_db, reset_schema
from data.ingest import ingest_all
from model.statements import FinancialModel
from model.dcf import DCFValuation
from model.scenarios import run_scenarios
from output.charts import generate_full_report
from interface.query import run_full_pipeline, interactive_loop, console


def load_config(path: str = "config.yaml") -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}


def main() -> None:
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
        except (ValueError, Exception) as exc:
            console.print(f"[red]Error generating report:[/] {exc}")
            console.print("Run without --report-only first to ingest data and build the model.")
            sys.exit(1)
        return

    # ── --interactive (skip pipeline) ──────────────────────────────────
    if args.interactive:
        console.print(f"[bold]Loading existing data for {ticker}...[/]")
        try:
            model = FinancialModel(ticker, db_path)
            model.compute_historical_metrics()
            dcf = DCFValuation(ticker, db_path)
            dcf.compute_wacc()
            base_dcf = dcf.dcf_valuation(scenario="base")
        except (ValueError, Exception) as exc:
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
    main()
