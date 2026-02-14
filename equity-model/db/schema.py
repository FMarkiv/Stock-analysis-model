"""
Database schema definition and management for the equity model.

Uses DuckDB with a single file at data/equity.duckdb.
All financial data is stored in long/narrow format for flexibility.
"""

import os
import duckdb

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "equity.duckdb")


def get_connection(db_path: str = DEFAULT_DB_PATH) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection, creating the data directory if needed."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    return duckdb.connect(db_path)


def create_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Create all tables if they do not already exist."""

    con.execute("""
        CREATE TABLE IF NOT EXISTS company (
            ticker       VARCHAR NOT NULL PRIMARY KEY,
            name         VARCHAR,
            sector       VARCHAR,
            currency     VARCHAR DEFAULT 'USD',
            fiscal_year_end VARCHAR          -- e.g. 'December', 'June'
        );
    """)

    # Long/narrow format: one row per (ticker, period, statement, line_item).
    # is_forecast + forecast_scenario distinguish actuals from projections.
    # forecast_scenario uses 'actual' (not NULL) so the UNIQUE constraint works
    # — SQL treats NULLs as distinct, which would break duplicate detection.
    con.execute("""
        CREATE TABLE IF NOT EXISTS financials (
            ticker             VARCHAR NOT NULL,
            period             VARCHAR NOT NULL,   -- 'Q1 2020', 'FY2020', etc.
            period_type        VARCHAR NOT NULL,   -- 'quarterly' | 'annual'
            statement          VARCHAR NOT NULL,   -- 'income' | 'balance' | 'cashflow'
            line_item          VARCHAR NOT NULL,
            value              DOUBLE,
            unit               VARCHAR DEFAULT 'USD',
            is_forecast        BOOLEAN NOT NULL DEFAULT false,
            forecast_scenario  VARCHAR NOT NULL DEFAULT 'actual',  -- 'actual' | 'base' | 'bull' | 'bear'
            updated_at         TIMESTAMP DEFAULT current_timestamp,

            UNIQUE (ticker, period, statement, line_item, is_forecast, forecast_scenario)
        );
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS segments (
            ticker             VARCHAR NOT NULL,
            period             VARCHAR NOT NULL,
            segment_name       VARCHAR NOT NULL,
            revenue            DOUBLE,
            operating_income   DOUBLE,
            is_forecast        BOOLEAN NOT NULL DEFAULT false,
            forecast_scenario  VARCHAR NOT NULL DEFAULT 'actual',

            UNIQUE (ticker, period, segment_name, is_forecast, forecast_scenario)
        );
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS assumptions (
            ticker           VARCHAR NOT NULL,
            scenario         VARCHAR NOT NULL,   -- 'base' | 'bull' | 'bear'
            parameter_name   VARCHAR NOT NULL,
            parameter_value  DOUBLE,
            description      VARCHAR,

            UNIQUE (ticker, scenario, parameter_name)
        );
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            ticker  VARCHAR NOT NULL,
            date    DATE NOT NULL,
            open    DOUBLE,
            high    DOUBLE,
            low     DOUBLE,
            close   DOUBLE,
            volume  BIGINT,

            UNIQUE (ticker, date)
        );
    """)


def drop_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Drop all project tables."""
    for table in ("prices", "assumptions", "segments", "financials", "company"):
        con.execute(f"DROP TABLE IF EXISTS {table};")


def reset_schema(db_path: str = DEFAULT_DB_PATH) -> duckdb.DuckDBPyConnection:
    """Drop and recreate all tables. Returns the open connection."""
    con = get_connection(db_path)
    drop_tables(con)
    create_tables(con)
    return con


def init_db(db_path: str = DEFAULT_DB_PATH) -> duckdb.DuckDBPyConnection:
    """Initialise the database (create tables if missing). Returns the open connection."""
    con = get_connection(db_path)
    create_tables(con)
    return con


if __name__ == "__main__":
    con = init_db()
    tables = con.execute("SHOW TABLES").fetchall()
    print("Tables created:", [t[0] for t in tables])
    con.close()
