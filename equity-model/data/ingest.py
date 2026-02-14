"""
Data ingestion from external sources (SEC EDGAR, Yahoo Finance, FRED).
"""


def fetch_financials(ticker: str) -> None:
    """Pull financial statements for a ticker and store in DuckDB."""
    raise NotImplementedError


def fetch_prices(ticker: str, start: str = "2015-01-01") -> None:
    """Download historical price data via yfinance and store in DuckDB."""
    raise NotImplementedError


def fetch_macro(series_id: str = "DGS10") -> None:
    """Pull macro series from FRED and store locally."""
    raise NotImplementedError
