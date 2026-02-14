"""
Financial statement construction and projection logic.
"""


def build_income_statement(ticker: str, period: str) -> dict:
    """Assemble an income statement from the long-format financials table."""
    raise NotImplementedError


def project_income_statement(ticker: str, scenario: str, periods: int = 5) -> list[dict]:
    """Project future income statements under a given scenario."""
    raise NotImplementedError
