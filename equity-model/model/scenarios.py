"""
Scenario management: store, load, and compare base / bull / bear assumptions.
"""


def load_assumptions(ticker: str, scenario: str) -> dict:
    """Load scenario assumptions from the database."""
    raise NotImplementedError


def save_assumptions(ticker: str, scenario: str, params: dict) -> None:
    """Persist scenario assumptions to the database."""
    raise NotImplementedError


def compare_scenarios(ticker: str) -> dict:
    """Run DCF under all scenarios and return a comparison summary."""
    raise NotImplementedError
