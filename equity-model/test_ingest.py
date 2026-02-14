"""
Integration test for data/ingest.py using mock yfinance data.

Verifies that all ingestion functions correctly transform data and
store it in DuckDB with proper upsert behaviour.
"""

import os
import sys
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db.schema import reset_schema

# ---------------------------------------------------------------------------
# Mock data factories
# ---------------------------------------------------------------------------

def _make_price_history():
    """Create a realistic price DataFrame matching yfinance output."""
    dates = pd.date_range("2015-01-02", "2024-12-31", freq="B")
    np.random.seed(42)
    n = len(dates)
    close = 100 + np.cumsum(np.random.randn(n) * 1.5)
    return pd.DataFrame(
        {
            "Open": close + np.random.randn(n) * 0.5,
            "High": close + abs(np.random.randn(n)) * 1.0,
            "Low": close - abs(np.random.randn(n)) * 1.0,
            "Close": close,
            "Volume": np.random.randint(50_000_000, 200_000_000, size=n),
            "Dividends": 0.0,
            "Stock Splits": 0.0,
        },
        index=pd.DatetimeIndex(dates, name="Date").tz_localize("America/New_York"),
    )


def _make_income_stmt(annual=True):
    if annual:
        dates = [pd.Timestamp(f"{y}-09-30") for y in [2024, 2023, 2022, 2021]]
    else:
        dates = [
            pd.Timestamp("2024-06-30"),
            pd.Timestamp("2024-03-31"),
            pd.Timestamp("2023-12-31"),
            pd.Timestamp("2023-09-30"),
        ]
    items = {
        "Total Revenue":         [391_035e6, 383_285e6, 394_328e6, 365_817e6],
        "Cost Of Revenue":       [214_137e6, 214_553e6, 223_546e6, 212_981e6],
        "Gross Profit":          [176_898e6, 168_732e6, 170_782e6, 152_836e6],
        "Research And Development": [29_915e6, 29_926e6, 26_251e6, 21_914e6],
        "Selling General And Administration": [25_094e6, 24_932e6, 25_094e6, 21_973e6],
        "Operating Income":      [121_889e6, 114_301e6, 119_437e6, 108_949e6],
        "Interest Expense":      [3_933e6, 3_933e6, 2_931e6, 2_645e6],
        "Interest Income":       [3_999e6, 3_750e6, 2_825e6, 2_843e6],
        "Pretax Income":         [123_485e6, 113_736e6, 119_103e6, 109_207e6],
        "Tax Provision":         [29_749e6, 16_741e6, 19_300e6, 14_527e6],
        "Net Income":            [93_736e6, 96_995e6, 99_803e6, 94_680e6],
        "Basic EPS":             [6.14, 6.16, 6.15, 5.67],
        "Diluted EPS":           [6.11, 6.13, 6.11, 5.61],
        "Basic Average Shares":  [15_287e6, 15_744e6, 16_216e6, 16_702e6],
        "Diluted Average Shares": [15_408e6, 15_813e6, 16_326e6, 16_865e6],
        "EBITDA":                [134_661e6, 125_820e6, 130_541e6, 120_233e6],
        "Reconciled Depreciation": [11_519e6, 11_519e6, 11_104e6, 11_284e6],
        "Stock Based Compensation": [11_688e6, 10_833e6, 9_038e6, 7_906e6],
    }
    return pd.DataFrame(items, index=dates).T


def _make_balance_sheet(annual=True):
    if annual:
        dates = [pd.Timestamp(f"{y}-09-30") for y in [2024, 2023, 2022, 2021]]
    else:
        dates = [
            pd.Timestamp("2024-06-30"),
            pd.Timestamp("2024-03-31"),
            pd.Timestamp("2023-12-31"),
            pd.Timestamp("2023-09-30"),
        ]
    items = {
        "Cash And Cash Equivalents": [29_943e6, 29_965e6, 23_646e6, 34_940e6],
        "Other Short Term Investments": [35_228e6, 31_590e6, 24_658e6, 27_699e6],
        "Accounts Receivable":    [66_243e6, 60_985e6, 60_932e6, 51_506e6],
        "Inventory":              [7_286e6, 6_331e6, 4_946e6, 6_580e6],
        "Current Assets":         [152_987e6, 143_566e6, 135_405e6, 134_836e6],
        "Net PPE":                [44_856e6, 43_715e6, 42_117e6, 39_440e6],
        "Goodwill":               [0.0, 0.0, 0.0, 0.0],
        "Other Intangible Assets": [0.0, 0.0, 0.0, 0.0],
        "Total Assets":           [364_980e6, 352_583e6, 352_755e6, 351_002e6],
        "Accounts Payable":       [68_960e6, 62_611e6, 64_115e6, 54_763e6],
        "Current Debt":           [10_912e6, 15_807e6, 11_128e6, 9_613e6],
        "Current Debt And Capital Lease Obligation": [12_560e6, 18_140e6, 21_110e6, 16_168e6],
        "Current Liabilities":    [176_392e6, 145_308e6, 153_982e6, 125_481e6],
        "Long Term Debt":         [96_302e6, 95_281e6, 98_959e6, 109_106e6],
        "Total Liabilities Net Minority Interest": [308_030e6, 290_437e6, 302_083e6, 287_912e6],
        "Stockholders Equity":    [56_950e6, 62_146e6, 50_672e6, 63_090e6],
        "Retained Earnings":      [-214e6, -214e6, -3_068e6, 5_562e6],
    }
    return pd.DataFrame(items, index=dates).T


def _make_cashflow(annual=True):
    if annual:
        dates = [pd.Timestamp(f"{y}-09-30") for y in [2024, 2023, 2022, 2021]]
    else:
        dates = [
            pd.Timestamp("2024-06-30"),
            pd.Timestamp("2024-03-31"),
            pd.Timestamp("2023-12-31"),
            pd.Timestamp("2023-09-30"),
        ]
    items = {
        "Operating Cash Flow":    [118_254e6, 110_543e6, 122_151e6, 104_038e6],
        "Capital Expenditure":    [-9_959e6, -10_959e6, -10_708e6, -11_085e6],
        "Free Cash Flow":         [108_295e6, 99_584e6, 111_443e6, 92_953e6],
        "Common Stock Dividend Paid": [-15_234e6, -15_025e6, -14_841e6, -14_467e6],
        "Repurchase Of Capital Stock": [-94_949e6, -77_550e6, -89_402e6, -85_500e6],
        "Depreciation And Amortization": [11_519e6, 11_519e6, 11_104e6, 11_284e6],
        "Stock Based Compensation": [11_688e6, 10_833e6, 9_038e6, 7_906e6],
        "Change In Working Capital": [-6_577e6, -6_577e6, 1_200e6, -4_911e6],
    }
    return pd.DataFrame(items, index=dates).T


def _mock_ticker():
    """Build a fully mocked yfinance.Ticker object."""
    ticker = MagicMock()
    ticker.history.return_value = _make_price_history()
    type(ticker).income_stmt = PropertyMock(return_value=_make_income_stmt(annual=True))
    type(ticker).quarterly_income_stmt = PropertyMock(return_value=_make_income_stmt(annual=False))
    type(ticker).balance_sheet = PropertyMock(return_value=_make_balance_sheet(annual=True))
    type(ticker).quarterly_balance_sheet = PropertyMock(return_value=_make_balance_sheet(annual=False))
    type(ticker).cashflow = PropertyMock(return_value=_make_cashflow(annual=True))
    type(ticker).quarterly_cashflow = PropertyMock(return_value=_make_cashflow(annual=False))
    type(ticker).info = PropertyMock(return_value={
        "shortName": "Apple Inc.",
        "sector": "Technology",
        "fiscalYearEnd": "September",
    })
    return ticker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "test_equity.duckdb")


class _NonClosingConnection:
    """Wrapper around a DuckDB connection that makes close() a no-op."""

    def __init__(self, real_con):
        object.__setattr__(self, "_real_con", real_con)

    def close(self):
        pass  # no-op so ingestion functions don't kill our test connection

    def real_close(self):
        self._real_con.close()

    def __getattr__(self, name):
        return getattr(self._real_con, name)


def _setup():
    """Reset the test database and return a wrapped connection."""
    con = reset_schema(DB_PATH)
    return _NonClosingConnection(con)


def _teardown(con=None):
    """Close connection and remove the test database file."""
    if con is not None:
        try:
            con.real_close()
        except Exception:
            pass
    for ext in ("", ".wal"):
        path = DB_PATH + ext
        if os.path.exists(path):
            os.remove(path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@patch("data.ingest.init_db")
@patch("data.ingest.yf")
def test_ingest_price_data(mock_yf, mock_init_db):
    from data.ingest import ingest_price_data

    con = _setup()
    mock_init_db.return_value = con
    mock_yf.Ticker.return_value = _mock_ticker()

    result = ingest_price_data("AAPL", years=10)

    assert result["ticker"] == "AAPL"
    assert result["rows"] > 2000

    count = con.execute("SELECT COUNT(*) FROM prices WHERE ticker = 'AAPL'").fetchone()[0]
    assert count == result["rows"], f"Expected {result['rows']} rows, got {count}"

    sample = con.execute(
        "SELECT * FROM prices WHERE ticker = 'AAPL' ORDER BY date LIMIT 1"
    ).fetchone()
    assert sample[0] == "AAPL"
    assert sample[1] is not None
    assert sample[6] > 0

    print(f"  PASS: ingest_price_data loaded {result['rows']} rows ({result['start']} to {result['end']})")
    _teardown(con)


@patch("data.ingest.init_db")
@patch("data.ingest.yf")
def test_ingest_price_data_upsert(mock_yf, mock_init_db):
    """Verify re-running ingestion replaces data rather than duplicating."""
    from data.ingest import ingest_price_data

    con = _setup()
    mock_init_db.return_value = con
    mock_yf.Ticker.return_value = _mock_ticker()

    ingest_price_data("AAPL")
    result2 = ingest_price_data("AAPL")

    count = con.execute("SELECT COUNT(*) FROM prices WHERE ticker = 'AAPL'").fetchone()[0]
    assert count == result2["rows"], f"Upsert duplicated data: {count} != {result2['rows']}"

    print(f"  PASS: upsert check - {count} rows (no duplicates)")
    _teardown(con)


@patch("data.ingest.init_db")
@patch("data.ingest.yf")
def test_ingest_financials(mock_yf, mock_init_db):
    from data.ingest import ingest_financials_from_yfinance

    con = _setup()
    mock_init_db.return_value = con
    mock_yf.Ticker.return_value = _mock_ticker()

    result = ingest_financials_from_yfinance("AAPL")

    assert result["ticker"] == "AAPL"
    assert result["total_items_loaded"] > 0
    assert result["periods"] > 0

    # Check income items
    income_rows = con.execute(
        "SELECT DISTINCT line_item FROM financials "
        "WHERE ticker = 'AAPL' AND statement = 'income' AND period_type = 'annual'"
    ).fetchall()
    income_items = {r[0] for r in income_rows}

    expected_income = {
        "total_revenue", "cost_of_revenue", "gross_profit", "operating_income",
        "net_income", "basic_eps", "diluted_eps", "ebitda",
    }
    assert expected_income.issubset(income_items), (
        f"Missing income items: {expected_income - income_items}"
    )

    # Check balance sheet items
    balance_rows = con.execute(
        "SELECT DISTINCT line_item FROM financials "
        "WHERE ticker = 'AAPL' AND statement = 'balance' AND period_type = 'annual'"
    ).fetchall()
    balance_items = {r[0] for r in balance_rows}

    expected_balance = {
        "cash_and_equivalents", "total_assets", "total_liabilities",
        "total_stockholders_equity",
    }
    assert expected_balance.issubset(balance_items), (
        f"Missing balance items: {expected_balance - balance_items}"
    )

    # Check cashflow items
    cf_rows = con.execute(
        "SELECT DISTINCT line_item FROM financials "
        "WHERE ticker = 'AAPL' AND statement = 'cashflow' AND period_type = 'annual'"
    ).fetchall()
    cf_items = {r[0] for r in cf_rows}

    expected_cf = {"operating_cash_flow", "capex", "free_cash_flow"}
    assert expected_cf.issubset(cf_items), (
        f"Missing cashflow items: {expected_cf - cf_items}"
    )

    # Verify both annual and quarterly data exist
    annual_count = con.execute(
        "SELECT COUNT(*) FROM financials WHERE ticker = 'AAPL' AND period_type = 'annual'"
    ).fetchone()[0]
    quarterly_count = con.execute(
        "SELECT COUNT(*) FROM financials WHERE ticker = 'AAPL' AND period_type = 'quarterly'"
    ).fetchone()[0]

    assert annual_count > 0, "No annual data"
    assert quarterly_count > 0, "No quarterly data"

    # Verify period format
    periods = con.execute(
        "SELECT DISTINCT period FROM financials WHERE ticker = 'AAPL' ORDER BY period"
    ).fetchall()
    period_strs = [r[0] for r in periods]
    annual_periods = [p for p in period_strs if p.startswith("FY")]
    quarterly_periods = [p for p in period_strs if p.startswith("Q")]
    assert len(annual_periods) > 0, "No FY periods found"
    assert len(quarterly_periods) > 0, "No Q periods found"

    print(
        f"  PASS: ingest_financials loaded {result['total_items_loaded']} items "
        f"across {result['periods']} periods"
    )
    print(f"         Annual: {annual_count} items, Quarterly: {quarterly_count} items")
    print(f"         Periods: {period_strs}")
    if result["missing_items"]:
        print(f"         Missing items: {result['missing_items']}")
    _teardown(con)


@patch("data.ingest.init_db")
@patch("data.ingest.yf")
def test_ingest_segments(mock_yf, mock_init_db):
    from data.ingest import ingest_segments

    con = _setup()
    mock_init_db.return_value = con
    mock_yf.Ticker.return_value = _mock_ticker()

    result = ingest_segments("AAPL")

    assert result["ticker"] == "AAPL"
    assert result["segments_loaded"] == 4  # 4 annual periods

    rows = con.execute(
        "SELECT period, revenue FROM segments WHERE ticker = 'AAPL' ORDER BY period"
    ).fetchall()
    assert len(rows) == 4
    for period, revenue in rows:
        assert period.startswith("FY")
        assert revenue > 0

    print(f"  PASS: ingest_segments loaded {result['segments_loaded']} entries")
    for period, revenue in rows:
        print(f"         {period}: ${revenue/1e9:.1f}B")
    _teardown(con)


@patch("data.ingest.init_db")
@patch("data.ingest.yf")
def test_ingest_all(mock_yf, mock_init_db):
    from data.ingest import ingest_all

    con = _setup()
    mock_init_db.return_value = con
    mock_yf.Ticker.return_value = _mock_ticker()

    ingest_all("AAPL")

    # Verify all tables have data
    for table in ("prices", "financials", "segments", "company"):
        count = con.execute(f"SELECT COUNT(*) FROM {table} WHERE ticker = 'AAPL'").fetchone()[0]
        assert count > 0, f"Table {table} has no AAPL data"

    print(f"  PASS: ingest_all completed successfully")
    _teardown(con)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    tests = [
        ("ingest_price_data", test_ingest_price_data),
        ("ingest_price_data (upsert)", test_ingest_price_data_upsert),
        ("ingest_financials_from_yfinance", test_ingest_financials),
        ("ingest_segments", test_ingest_segments),
        ("ingest_all", test_ingest_all),
    ]

    print("=" * 60)
    print("  Running data ingestion tests")
    print("=" * 60)

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"\nTest: {name}")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")

    sys.exit(0 if failed == 0 else 1)
