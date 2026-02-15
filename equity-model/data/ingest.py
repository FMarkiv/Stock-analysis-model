"""
Data ingestion from external sources (Yahoo Finance, SEC EDGAR).

Pulls price, financial-statement, and segment data into DuckDB.
Designed to be resilient: if any single data source fails the
pipeline logs a warning and continues with whatever is available.

Functions
---------
ingest_price_data              -- daily OHLCV from yfinance  -> prices table
ingest_financials_from_yfinance -- quarterly & annual statements -> financials table
ingest_segments_edgar          -- segment revenue from SEC EDGAR XBRL / config fallback -> segments table
ingest_all                     -- master driver that runs everything and prints a summary
"""

import logging
import os
import sys
from datetime import datetime

import pandas as pd
import requests
import yaml
import yfinance as yf

# Allow running as ``python data/ingest.py`` from the equity-model directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from db.schema import get_connection, init_db  # noqa: E402
from data.mappings import (  # noqa: E402
    YFINANCE_INCOME_MAP,
    YFINANCE_BALANCE_MAP,
    YFINANCE_CASHFLOW_MAP,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _date_to_period(dt: pd.Timestamp, period_type: str) -> str:
    """Convert a pandas Timestamp to a period string.

    Annual  -> ``'FY2023'``
    Quarterly -> ``'Q3 2023'``  (calendar quarter based on period-end month)
    """
    if period_type == "annual":
        return f"FY{dt.year}"
    quarter = (dt.month - 1) // 3 + 1
    return f"Q{quarter} {dt.year}"


def _strip_tz(series: pd.Series) -> pd.Series:
    """Remove timezone info from a datetime Series (if present)."""
    if hasattr(series.dt, "tz") and series.dt.tz is not None:
        return series.dt.tz_convert(None)
    return series


# ---------------------------------------------------------------------------
# 1. Price data
# ---------------------------------------------------------------------------


def ingest_price_data(ticker: str, years: int = 10) -> dict:
    """Pull daily price/volume data from yfinance and upsert into the
    ``prices`` table.

    Uses a delete-then-insert strategy for the given ticker which is
    equivalent to a full upsert and much faster than row-by-row.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    years : int
        Number of years of history to fetch (default 10).

    Returns
    -------
    dict
        ``{'ticker', 'rows', 'start', 'end'}``
    """
    logger.info("Fetching price data for %s (%d years)", ticker, years)

    t = yf.Ticker(ticker)
    df = t.history(period=f"{years}y")

    if df.empty:
        logger.warning("No price data returned for %s", ticker)
        return {"ticker": ticker, "rows": 0}

    # Prepare DataFrame to match the prices table schema
    df = df.reset_index()
    df["Date"] = _strip_tz(df["Date"])

    price_df = pd.DataFrame({
        "ticker": ticker,
        "date": df["Date"],
        "open": df["Open"],
        "high": df["High"],
        "low": df["Low"],
        "close": df["Close"],
        "volume": df["Volume"].fillna(0).astype("int64"),
    })

    con = init_db()
    try:
        con.execute("DELETE FROM prices WHERE ticker = ?", [ticker])
        con.register("_price_buf", price_df)
        con.execute("INSERT INTO prices SELECT * FROM _price_buf")
        con.unregister("_price_buf")
    finally:
        con.close()

    result = {
        "ticker": ticker,
        "rows": len(price_df),
        "start": str(price_df["date"].min().date()),
        "end": str(price_df["date"].max().date()),
    }
    logger.info(
        "Loaded %d price rows for %s (%s to %s)",
        result["rows"], ticker, result["start"], result["end"],
    )
    return result


# ---------------------------------------------------------------------------
# 2. Financial statements
# ---------------------------------------------------------------------------


def _ingest_statement(
    con,
    ticker: str,
    stmt_df: pd.DataFrame,
    statement: str,
    period_type: str,
    mapping: dict[str, str],
) -> dict:
    """Ingest one financial-statement DataFrame into the ``financials`` table.

    Handles missing line items gracefully -- items that are absent from the
    source DataFrame are simply skipped and logged as warnings.

    Returns ``{'loaded': int, 'skipped': list[str], 'periods': set[str]}``.
    """
    loaded = 0
    skipped: list[str] = []
    periods: set[str] = set()

    if stmt_df is None or stmt_df.empty:
        logger.warning("No %s %s data for %s", period_type, statement, ticker)
        return {"loaded": 0, "skipped": [], "periods": set()}

    # Track which standardised names we've already seen per period to avoid
    # inserting duplicates when multiple yfinance aliases map to the same name.
    seen: set[tuple[str, str]] = set()

    for date_col in stmt_df.columns:
        period = _date_to_period(date_col, period_type)

        for yf_name, std_name in mapping.items():
            if yf_name not in stmt_df.index:
                continue

            # Skip duplicate mappings for the same (period, std_name)
            key = (period, std_name)
            if key in seen:
                continue

            value = stmt_df.loc[yf_name, date_col]
            if pd.isna(value):
                continue

            now = datetime.now()
            con.execute(
                """
                INSERT INTO financials
                    (ticker, period, period_type, statement, line_item, value,
                     is_forecast, forecast_scenario, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, false, 'actual', ?)
                ON CONFLICT (ticker, period, statement, line_item, is_forecast, forecast_scenario)
                DO UPDATE SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at
                """,
                [ticker, period, period_type, statement, std_name, float(value), now],
            )
            loaded += 1
            periods.add(period)
            seen.add(key)

    # Determine which standardised items we never found across any period
    expected = set(mapping.values())
    found = {std for (_, std) in seen}
    skipped = sorted(expected - found)

    if skipped:
        logger.warning(
            "Missing %s %s items for %s: %s",
            period_type, statement, ticker, ", ".join(skipped),
        )

    return {"loaded": loaded, "skipped": skipped, "periods": periods}


def ingest_financials_from_yfinance(ticker: str) -> dict:
    """Pull quarterly and annual income-statement, balance-sheet, and
    cash-flow data from yfinance, map to standardised names, and upsert
    into the ``financials`` table.

    Each statement type is wrapped in its own try/except so that a failure
    in one (e.g. missing quarterly data for some companies) does not block
    the rest.

    Returns a summary dict.
    """
    logger.info("Fetching financials for %s", ticker)

    t = yf.Ticker(ticker)

    # (attribute_name, statement_key, period_type, mapping_dict)
    statements_to_fetch = [
        ("income_stmt", "income", "annual", YFINANCE_INCOME_MAP),
        ("quarterly_income_stmt", "income", "quarterly", YFINANCE_INCOME_MAP),
        ("balance_sheet", "balance", "annual", YFINANCE_BALANCE_MAP),
        ("quarterly_balance_sheet", "balance", "quarterly", YFINANCE_BALANCE_MAP),
        ("cashflow", "cashflow", "annual", YFINANCE_CASHFLOW_MAP),
        ("quarterly_cashflow", "cashflow", "quarterly", YFINANCE_CASHFLOW_MAP),
    ]

    con = init_db()
    try:
        total_loaded = 0
        all_skipped: list[str] = []
        all_periods: set[str] = set()

        for attr_name, statement, period_type, mapping in statements_to_fetch:
            try:
                stmt_df = getattr(t, attr_name, None)
                result = _ingest_statement(
                    con, ticker, stmt_df, statement, period_type, mapping,
                )
                total_loaded += result["loaded"]
                for s in result["skipped"]:
                    if s not in all_skipped:
                        all_skipped.append(s)
                all_periods.update(result["periods"])
            except Exception as e:
                logger.warning(
                    "Error ingesting %s %s for %s: %s (continuing with other statements)",
                    period_type, statement, ticker, e,
                )
    finally:
        con.close()

    summary = {
        "ticker": ticker,
        "total_items_loaded": total_loaded,
        "periods": len(all_periods),
        "period_list": sorted(all_periods),
        "missing_items": all_skipped,
    }
    logger.info(
        "Loaded %d financial data points across %d periods for %s",
        total_loaded, len(all_periods), ticker,
    )
    return summary


# ---------------------------------------------------------------------------
# 3. Segments -- SEC EDGAR XBRL with config-based fallback
# ---------------------------------------------------------------------------

_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.yaml")


def _load_config() -> dict:
    """Load and return the project ``config.yaml`` as a dict."""
    try:
        with open(_CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning("config.yaml not found at %s", _CONFIG_PATH)
        return {}


def _resolve_cik(ticker: str, user_agent: str) -> str | None:
    """Resolve a stock ticker to a zero-padded SEC CIK string.

    Uses the SEC company tickers JSON endpoint.  Returns ``None`` on
    any network failure or if the ticker is not found.
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {"User-Agent": user_agent}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("Failed to fetch SEC company tickers: %s", e)
        return None

    ticker_upper = ticker.upper()
    for entry in data.values():
        if entry.get("ticker", "").upper() == ticker_upper:
            return str(entry["cik_str"]).zfill(10)

    logger.warning("Ticker %s not found in SEC company tickers", ticker)
    return None


def _fetch_edgar_segments(
    ticker: str, cik: str, user_agent: str,
) -> list[dict] | None:
    """Fetch segment revenue data from SEC EDGAR XBRL company facts.

    Looks for revenue concepts that carry a segment dimension
    (``srt:ProductOrServiceAxis``, ``us-gaap:StatementBusinessSegmentsAxis``,
    or ``srt:ConsolidationItemsAxis``).

    Returns a list of dicts ``{period, segment_name, revenue}`` or
    ``None`` if segment data cannot be extracted.
    """
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    headers = {"User-Agent": user_agent}
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        facts = resp.json()
    except Exception as e:
        logger.warning("Failed to fetch EDGAR company facts for %s: %s", ticker, e)
        return None

    us_gaap = facts.get("facts", {}).get("us-gaap", {})

    # Revenue concepts to search, in priority order
    revenue_concepts = [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ]

    segments: list[dict] = []

    for concept in revenue_concepts:
        concept_data = us_gaap.get(concept)
        if concept_data is None:
            continue

        units = concept_data.get("units", {})
        usd_facts = units.get("USD", [])

        for fact in usd_facts:
            # Segment-dimensioned facts include a "dimensions" key that
            # distinguishes them from the consolidated total.
            dim = fact.get("dimensions", {})
            seg_member = dim.get(
                "srt:ProductOrServiceAxis",
                dim.get(
                    "us-gaap:StatementBusinessSegmentsAxis",
                    dim.get("srt:ConsolidationItemsAxis"),
                ),
            )
            if not seg_member:
                continue

            # Only use annual 10-K filings to match our period format
            if fact.get("form") != "10-K":
                continue

            fy = fact.get("fy")
            val = fact.get("val")
            if fy is None or val is None:
                continue

            # fp == "FY" means full fiscal year (not a quarterly fragment)
            if fact.get("fp") != "FY":
                continue

            # Clean up the segment member name
            # e.g. "aapl:IPhoneMember" -> "IPhone"
            seg_name = seg_member.split(":")[-1]
            seg_name = seg_name.replace("Member", "").replace("Segment", "")

            segments.append({
                "period": f"FY{fy}",
                "segment_name": seg_name,
                "revenue": float(val),
            })

        if segments:
            break  # Found segment data with this concept; stop searching

    if not segments:
        logger.info("No segment-dimensioned revenue found in EDGAR for %s", ticker)
        return None

    logger.info(
        "Found %d segment-period entries from EDGAR for %s", len(segments), ticker,
    )
    return segments


def _apply_segment_overrides(
    ticker: str, con, config: dict,
) -> int:
    """Apply manual segment percentage overrides from ``config.yaml``.

    Reads ``total_revenue`` from the ``financials`` table for each annual
    period and splits it according to the configured percentages.

    Returns the number of segment rows inserted.
    """
    overrides = config.get("segment_overrides", {}).get(ticker.upper())
    if not overrides:
        logger.info("No segment overrides in config for %s", ticker)
        return 0

    # Get total revenue for each annual period
    rows = con.execute(
        """
        SELECT period, value FROM financials
        WHERE ticker = ? AND line_item = 'total_revenue'
          AND period_type = 'annual' AND is_forecast = false
          AND forecast_scenario = 'actual'
        ORDER BY period
        """,
        [ticker],
    ).fetchall()

    if not rows:
        logger.warning("No annual total_revenue in financials for %s; cannot apply overrides", ticker)
        return 0

    loaded = 0
    for period, total_revenue in rows:
        for seg in overrides:
            seg_name = seg["name"]
            seg_revenue = total_revenue * seg["revenue_pct"]

            con.execute(
                """
                INSERT INTO segments
                    (ticker, period, segment_name, revenue, is_forecast, forecast_scenario)
                VALUES (?, ?, ?, ?, false, 'actual')
                ON CONFLICT (ticker, period, segment_name, is_forecast, forecast_scenario)
                DO UPDATE SET revenue = EXCLUDED.revenue
                """,
                [ticker, period, seg_name, seg_revenue],
            )
            loaded += 1

    logger.info("Applied %d segment override entries for %s", loaded, ticker)
    return loaded


def ingest_segments_edgar(ticker: str) -> dict:
    """Ingest segment-level revenue data into the ``segments`` table.

    Three-tier fallback strategy:

    1. **SEC EDGAR XBRL** -- fetch segment-dimensioned revenue from the
       company facts API.
    2. **Config overrides** -- manual percentage splits defined in
       ``config.yaml`` under ``segment_overrides``.
    3. **Total fallback** -- store total revenue as a single ``'Total'``
       segment so downstream code always has something to work with.
    """
    logger.info("Fetching segment data for %s", ticker)
    config = _load_config()
    user_agent = (
        config.get("api_keys", {}).get("sec_edgar", {}).get("user_agent", "")
    )

    source = "none"
    loaded = 0

    # --- Strategy 1: SEC EDGAR XBRL -----------------------------------------
    edgar_segments = None
    if user_agent:
        cik = _resolve_cik(ticker, user_agent)
        if cik:
            edgar_segments = _fetch_edgar_segments(ticker, cik, user_agent)

    con = init_db()
    try:
        if edgar_segments:
            source = "edgar"
            for seg in edgar_segments:
                con.execute(
                    """
                    INSERT INTO segments
                        (ticker, period, segment_name, revenue,
                         is_forecast, forecast_scenario)
                    VALUES (?, ?, ?, ?, false, 'actual')
                    ON CONFLICT (ticker, period, segment_name,
                                 is_forecast, forecast_scenario)
                    DO UPDATE SET revenue = EXCLUDED.revenue
                    """,
                    [ticker, seg["period"], seg["segment_name"], seg["revenue"]],
                )
                loaded += 1
            logger.info(
                "Loaded %d EDGAR segment entries for %s", loaded, ticker,
            )
        else:
            # --- Strategy 2: Config overrides --------------------------------
            loaded = _apply_segment_overrides(ticker, con, config)
            if loaded > 0:
                source = "config_override"
            else:
                # --- Strategy 3: Total revenue fallback ----------------------
                source = "total_fallback"
                t = yf.Ticker(ticker)
                income = getattr(t, "income_stmt", None)
                if income is not None and not income.empty:
                    # Try multiple yfinance revenue field names for robustness
                    revenue_row_name = None
                    for candidate in ("Total Revenue", "Operating Revenue", "Revenue"):
                        if candidate in income.index:
                            revenue_row_name = candidate
                            break

                    if revenue_row_name is None:
                        logger.warning(
                            "No recognisable revenue row in income statement for %s",
                            ticker,
                        )
                    else:
                        for date_col in income.columns:
                            period = _date_to_period(date_col, "annual")
                            revenue = income.loc[revenue_row_name, date_col]
                            if pd.isna(revenue):
                                continue
                            con.execute(
                                """
                                INSERT INTO segments
                                    (ticker, period, segment_name, revenue,
                                     is_forecast, forecast_scenario)
                                VALUES (?, ?, 'Total', ?, false, 'actual')
                                ON CONFLICT (ticker, period, segment_name,
                                             is_forecast, forecast_scenario)
                                DO UPDATE SET revenue = EXCLUDED.revenue
                                """,
                                [ticker, period, float(revenue)],
                            )
                            loaded += 1

        logger.info(
            "Segment ingestion for %s complete: %d entries (source: %s)",
            ticker, loaded, source,
        )
        return {
            "ticker": ticker,
            "segments_loaded": loaded,
            "source": source,
        }
    finally:
        con.close()


# ---------------------------------------------------------------------------
# 4. Master driver
# ---------------------------------------------------------------------------


def ingest_all(ticker: str) -> None:
    """Run all ingestion steps for *ticker* and print a human-readable summary.

    Each step is wrapped in its own try/except so that a failure in one
    data source does not prevent the others from running.  After all steps
    complete, the ``company.last_ingested`` timestamp is updated.
    """
    print(f"\n{'=' * 60}")
    print(f"  Ingesting data for {ticker}")
    print(f"{'=' * 60}\n")

    # --- Ensure a company record exists -----------------------------------
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        con = init_db()
        try:
            con.execute(
                """
                INSERT INTO company (ticker, name, sector, fiscal_year_end, last_ingested)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (ticker) DO UPDATE SET
                    name = EXCLUDED.name,
                    sector = EXCLUDED.sector,
                    fiscal_year_end = EXCLUDED.fiscal_year_end,
                    last_ingested = EXCLUDED.last_ingested
                """,
                [
                    ticker,
                    info.get("shortName", ticker),
                    info.get("sector"),
                    info.get("fiscalYearEnd"),
                    datetime.now(),
                ],
            )
        finally:
            con.close()
    except Exception as e:
        logger.warning("Could not fetch company info for %s: %s", ticker, e)

    # --- 1. Price data ----------------------------------------------------
    print("1. Price data...")
    try:
        price_result = ingest_price_data(ticker)
        if price_result["rows"] > 0:
            print(
                f"   Loaded {price_result['rows']} daily price records "
                f"({price_result['start']} to {price_result['end']})"
            )
        else:
            print("   No price data available")
    except Exception as e:
        print(f"   [WARNING] Price data failed: {e}")
        logger.exception("Price ingestion failed for %s", ticker)
        price_result = {"rows": 0}

    # --- 2. Financial statements ------------------------------------------
    print("2. Financial statements...")
    try:
        fin_result = ingest_financials_from_yfinance(ticker)
        print(
            f"   Loaded {fin_result['total_items_loaded']} data points "
            f"across {fin_result['periods']} periods"
        )
        if fin_result["missing_items"]:
            print(f"   Missing items: {', '.join(fin_result['missing_items'])}")
    except Exception as e:
        print(f"   [WARNING] Financial data failed: {e}")
        logger.exception("Financial ingestion failed for %s", ticker)
        fin_result = {"total_items_loaded": 0, "periods": 0, "missing_items": []}

    # --- 3. Segments (EDGAR + config fallback) -----------------------------
    print("3. Segments...")
    try:
        seg_result = ingest_segments_edgar(ticker)
        source = seg_result.get("source", "unknown")
        print(
            f"   Loaded {seg_result['segments_loaded']} segment entries "
            f"(source: {source})"
        )
    except Exception as e:
        print(f"   [WARNING] Segment data failed: {e}")
        logger.exception("Segment ingestion failed for %s", ticker)
        seg_result = {"segments_loaded": 0}

    # --- Update last_ingested timestamp -----------------------------------
    try:
        con = init_db()
        try:
            con.execute(
                "UPDATE company SET last_ingested = ? WHERE ticker = ?",
                [datetime.now(), ticker],
            )
        finally:
            con.close()
    except Exception:
        pass  # non-critical

    # --- Summary ----------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  Summary for {ticker}")
    print(f"{'=' * 60}")
    print(f"  Price records:    {price_result.get('rows', 0)}")
    print(f"  Financial items:  {fin_result.get('total_items_loaded', 0)}")
    print(f"  Periods covered:  {fin_result.get('periods', 0)}")
    print(f"  Segment entries:  {seg_result.get('segments_loaded', 0)}")
    if fin_result.get("missing_items"):
        print(f"  Missing items:    {len(fin_result['missing_items'])}")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Legacy wrappers (keep original function signatures working)
# ---------------------------------------------------------------------------


def fetch_financials(ticker: str) -> None:
    """Pull financial statements for a ticker and store in DuckDB."""
    ingest_financials_from_yfinance(ticker)


def fetch_prices(ticker: str, start: str = "2015-01-01") -> None:
    """Download historical price data via yfinance and store in DuckDB."""
    ingest_price_data(ticker)


def fetch_macro(series_id: str = "DGS10") -> None:
    """Pull macro series from FRED and store locally."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    ticker_arg = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    ingest_all(ticker_arg)
