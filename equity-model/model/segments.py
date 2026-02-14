"""
Segment-level analysis and projection.
"""


def get_segment_breakdown(ticker: str, period: str) -> list[dict]:
    """Return segment revenue and margin breakdown for a period."""
    raise NotImplementedError


def project_segments(ticker: str, scenario: str, periods: int = 5) -> list[dict]:
    """Project segment-level financials under a given scenario."""
    raise NotImplementedError
