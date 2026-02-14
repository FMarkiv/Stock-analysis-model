"""
Discounted cash-flow valuation engine.
"""


def compute_wacc(ticker: str, risk_free: float, erp: float, tax_rate: float) -> float:
    """Estimate weighted-average cost of capital."""
    raise NotImplementedError


def run_dcf(ticker: str, scenario: str, forecast_years: int = 5) -> dict:
    """Run a full DCF and return enterprise value, equity value, and per-share value."""
    raise NotImplementedError
