# Equity Model

A transparent, code-as-model equity research platform that replaces
spreadsheet-based financial modelling with a reproducible Python pipeline.
Ingests data from Yahoo Finance and SEC EDGAR, builds three-statement
financial models, runs DCF and multiples-based valuations, and provides an
interactive CLI for what-if analysis.

## Setup

### 1. Install dependencies

```bash
cd equity-model
pip install -r requirements.txt
```

Dependencies: `duckdb`, `yfinance`, `requests`, `pyyaml`, `pandas`, `numpy`,
`scipy`, `plotly`, `rich`.

### 2. Configure API keys

Copy and edit `config.yaml`:

```yaml
api_keys:
  sec_edgar:
    # SEC EDGAR requires a User-Agent with your name and email.
    # Without this, segment data requests will be blocked.
    user_agent: "Jane Doe jane@example.com"
  fred:
    # Optional: used to fetch live 10-year Treasury yield for WACC.
    # If missing, the model uses the risk_free_rate from model_parameters.
    api_key: "YOUR_FRED_API_KEY"
```

### 3. Initialise the database (optional)

```bash
python main.py init
```

This creates `data/equity.duckdb` with all required tables.  The database is
also auto-created on first run.

## Quick Start

```bash
# Full pipeline: ingest data, build model, run valuation, open interactive mode
python main.py AAPL

# Ingest-only (useful for refreshing data without re-running the model)
python main.py AAPL --ingest-only

# Generate HTML report only (data must already exist)
python main.py AAPL --report-only

# Jump to interactive mode (skips ingestion, assumes data exists)
python main.py AAPL --interactive

# Reset the database (drop all tables and recreate)
python main.py reset
```

## Project Structure

```
equity-model/
├── main.py                 Entry point — CLI dispatcher
├── config.yaml             API keys, model parameters, segment overrides
├── requirements.txt        Python dependencies
│
├── data/
│   ├── ingest.py           Data ingestion (yfinance, SEC EDGAR)
│   └── mappings.py         Field-name mappings (yfinance → internal names)
│
├── db/
│   └── schema.py           DuckDB schema definition and migrations
│
├── model/
│   ├── statements.py       Three-statement financial model & forecasting
│   ├── dcf.py              DCF valuation (WACC, terminal value, sensitivity)
│   ├── scenarios.py        Bull/base/bear scenarios & Monte Carlo simulation
│   └── segments.py         Segment-level analysis (stub)
│
├── output/
│   └── charts.py           Plotly visualisations and HTML report generation
│
├── interface/
│   ├── query.py            Interactive natural-language query interface
│   └── __main__.py         CLI entry for `python -m interface.query`
│
├── seed_aapl.py            Test data seeder (synthetic AAPL financials)
├── test_ingest.py          Unit tests for data ingestion
└── test_dcf_scenarios.py   End-to-end DCF and scenario tests
```

## Module Descriptions

### `data/ingest.py` — Data Ingestion

Pulls data from three sources with a fallback strategy:

1. **Price data** — 10 years of daily OHLCV from Yahoo Finance.
2. **Financial statements** — annual and quarterly income statements, balance
   sheets, and cash-flow statements from yfinance.  Field names are
   standardised via `data/mappings.py`.
3. **Segment data** — three-tier fallback:
   - SEC EDGAR XBRL segment-dimensioned revenue (preferred)
   - Manual percentage splits from `config.yaml` (fallback)
   - Total revenue as a single "Total" segment (last resort)

Each step is wrapped in try/except — if one source fails, the pipeline
continues with whatever data is available.

### `db/schema.py` — Database Schema

Five tables in DuckDB:

| Table        | Purpose                                        |
|--------------|------------------------------------------------|
| `company`    | Ticker metadata + `last_ingested` timestamp    |
| `financials` | Long-format financial data (income/balance/CF) |
| `segments`   | Segment-level revenue breakdown                |
| `assumptions`| Model assumptions per (ticker, scenario)       |
| `prices`     | Daily OHLCV price history                      |

### `model/statements.py` — Three-Statement Model

- Loads historical data from DuckDB and pivots into wide-format DataFrames.
- Computes 15+ derived metrics (margins, returns, leverage, working capital).
- Generates 5-year forecasts based on auto-derived or user-specified
  assumptions.
- Revenue growth decays toward long-run GDP growth (2.5% nominal) over the
  forecast horizon.
- Balance sheet self-balances; imbalances within 0.1% tolerance are logged
  as warnings.
- Handles companies with negative earnings (no tax on losses, capped payout
  ratios).

### `model/dcf.py` — DCF Valuation

- **WACC**: CAPM-based cost of equity (regression beta vs SPY) + after-tax
  cost of debt, weighted by market cap vs total debt.
- **Unlevered FCF**: EBIT × (1 − t) + D&A + CapEx + ΔWC.
- **Terminal value**: average of perpetuity-growth and exit-multiple methods.
  Default exit multiple: 12×. Default terminal growth: 2.5%.
- **Sensitivity table**: two-dimensional grid showing implied share price
  across WACC and terminal growth combinations.
- **Multiples valuation**: forward P/E, EV/EBITDA, and Price/FCF using
  5-year historical averages.  Methods with negative forward values are
  automatically skipped.
- If DCF produces a negative equity value, the result is flagged with
  `dcf_valid=False` and multiples-based valuation is still available.

### `model/scenarios.py` — Scenario Analysis

- Runs bull/base/bear scenarios with adjustable offsets to revenue growth,
  operating margin, and WACC.
- Monte Carlo simulation (default 10,000 iterations) with randomised
  assumptions drawn from normal and triangular distributions.

### `output/charts.py` — Visualisations

Generates interactive Plotly charts:

- Revenue bridge / waterfall (with segment breakdown if available)
- Margin trends (gross, operating, net, FCF)
- Three-statement summary dashboard
- DCF waterfall
- Sensitivity heatmap
- Monte Carlo histogram
- Scenario comparison bars
- Football field valuation range chart
- Full HTML report combining all charts

### `interface/query.py` — Interactive Mode

Natural-language CLI with keyword matching.  Supports:

- Data queries: `show revenue for last 8 quarters`
- Statements: `show income statement`, `show balance sheet`
- Assumption changes: `set revenue growth to 8%`
- Valuation: `what is fair value`, `show multiples`
- Analysis: `compare scenarios`, `run monte carlo`
- Reports: `generate report`, `show football field`

## Example Interactive Queries

```
AAPL> show revenue for last 8 quarters
AAPL> what is fair value
AAPL> show wacc
AAPL> set revenue growth to 10%
AAPL> what if operating margin expands to 35%
AAPL> compare scenarios
AAPL> run monte carlo
AAPL> show multiples
AAPL> show income statement
AAPL> show assumptions
AAPL> generate report
AAPL> show football field
AAPL> help
AAPL> quit
```

## Data Freshness

The model tracks when data was last ingested via the `company.last_ingested`
timestamp.  If data is older than 24 hours, the CLI prompts the user to
refresh before running the pipeline.

## Known Limitations

1. **yfinance data quality** — yfinance is an unofficial Yahoo Finance API.
   Data may be delayed, incomplete, or occasionally incorrect.  Some
   companies report line items differently (e.g. "Operating Revenue" vs
   "Total Revenue"), and the model handles common aliases but may miss
   edge cases.

2. **Segment data availability** — SEC EDGAR XBRL segment data is only
   available for US-listed companies and not all companies report segment
   breakdowns in a parseable format.  The model falls back to config
   overrides or consolidated totals when EDGAR data is unavailable.

3. **No quarterly forecasting** — forecasts are generated on an annual
   basis only.  Quarterly seasonality is not modelled.

4. **Single-currency** — the model assumes all values are in USD.
   International companies reporting in other currencies require manual
   conversion.

5. **Simplified WACC** — uses a single-factor CAPM with historical beta.
   Does not account for size premiums, country risk, or industry-specific
   adjustments.

6. **No consensus estimates** — the model derives forecasts from historical
   trends, not analyst consensus.  Forecasts will diverge from market
   expectations for companies undergoing structural changes.

7. **Terminal value sensitivity** — as with any DCF model, the valuation is
   highly sensitive to terminal growth rate and exit multiple assumptions.
   Always review the sensitivity table.

8. **Loss-making companies** — the model handles negative earnings (no tax
   on losses, skipped P/E multiples) but the DCF may produce unreliable
   results.  The model flags this with `dcf_valid=False` and falls back
   to multiples where possible.

## Running Tests

```bash
cd equity-model

# Unit tests for data ingestion (uses mocked yfinance/EDGAR)
python test_ingest.py

# End-to-end DCF and scenario tests (uses seeded AAPL data)
python test_dcf_scenarios.py
```
