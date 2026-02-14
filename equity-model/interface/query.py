"""
Query interface for retrieving and filtering data from the equity database.
"""


def get_financials(ticker: str, statement: str = None, period: str = None) -> list[dict]:
    """Query financials with optional filters on statement type and period."""
    raise NotImplementedError


def get_prices(ticker: str, start: str = None, end: str = None) -> list[dict]:
    """Query historical prices for a ticker within an optional date range."""
    raise NotImplementedError


def search_line_items(keyword: str) -> list[str]:
    """Search for line-item names containing a keyword."""
    raise NotImplementedError
