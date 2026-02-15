"""
Segment-level analysis and projection (stub).

This module provides placeholder functions for segment-level analysis.
The actual segment ingestion is handled by ``data/ingest.py``
(``ingest_segments_edgar``), which stores segment revenue data in the
``segments`` table.  When no segment data is available, the pipeline
operates on a consolidated basis using total revenue as a single
segment.

Future work could implement segment-level margin modelling and
independent growth projections per business line.
"""


def get_segment_breakdown(ticker: str, period: str) -> list[dict]:
    """Return segment revenue and margin breakdown for a period.

    Not yet implemented.  Use the ``segments`` table directly for
    current segment data.
    """
    raise NotImplementedError


def project_segments(ticker: str, scenario: str, periods: int = 5) -> list[dict]:
    """Project segment-level financials under a given scenario.

    Not yet implemented.  The current model projects at the
    consolidated level and applies segment splits from config or
    EDGAR for presentation purposes only.
    """
    raise NotImplementedError
