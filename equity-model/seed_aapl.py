"""
Seed DuckDB with representative AAPL historical financial data for testing.
Used when yfinance API access is unavailable.
"""

import os
import sys
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from db.schema import init_db

# ---------------------------------------------------------------------------
# AAPL historical data (FY2021-FY2024, approximate values in USD)
# Source: Apple 10-K filings (rounded for testing purposes)
# ---------------------------------------------------------------------------

COMPANY = ("AAPL", "Apple Inc.", "Technology", "USD", "September")

# Income statement data (annual, values in raw USD)
INCOME_DATA = {
    "FY2021": {
        "total_revenue":             365_817_000_000,
        "cost_of_revenue":           212_981_000_000,
        "gross_profit":              152_836_000_000,
        "research_development":       21_914_000_000,
        "selling_general_admin":      21_973_000_000,
        "operating_income":          108_949_000_000,
        "interest_expense":            2_645_000_000,
        "interest_income":             2_843_000_000,
        "pretax_income":             109_207_000_000,
        "income_tax":                 14_527_000_000,
        "net_income":                 94_680_000_000,
        "ebitda":                    120_233_000_000,
        "depreciation_amortization":  11_284_000_000,
        "stock_based_comp":            7_906_000_000,
        "basic_eps":                          5.67,
        "diluted_eps":                        5.61,
        "basic_shares_out":       16_701_272_000,
        "diluted_shares_out":     16_864_919_000,
    },
    "FY2022": {
        "total_revenue":             394_328_000_000,
        "cost_of_revenue":           223_546_000_000,
        "gross_profit":              170_782_000_000,
        "research_development":       26_251_000_000,
        "selling_general_admin":      25_094_000_000,
        "operating_income":          119_437_000_000,
        "interest_expense":            2_931_000_000,
        "interest_income":             2_825_000_000,
        "pretax_income":             119_103_000_000,
        "income_tax":                 19_300_000_000,
        "net_income":                 99_803_000_000,
        "ebitda":                    130_541_000_000,
        "depreciation_amortization":  11_104_000_000,
        "stock_based_comp":            9_038_000_000,
        "basic_eps":                          6.15,
        "diluted_eps":                        6.11,
        "basic_shares_out":       16_215_963_000,
        "diluted_shares_out":     16_325_819_000,
    },
    "FY2023": {
        "total_revenue":             383_285_000_000,
        "cost_of_revenue":           214_137_000_000,
        "gross_profit":              169_148_000_000,
        "research_development":       29_915_000_000,
        "selling_general_admin":      24_932_000_000,
        "operating_income":          114_301_000_000,
        "interest_expense":            3_933_000_000,
        "interest_income":             3_750_000_000,
        "pretax_income":             113_736_000_000,
        "income_tax":                 16_741_000_000,
        "net_income":                 96_995_000_000,
        "ebitda":                    125_820_000_000,
        "depreciation_amortization":  11_519_000_000,
        "stock_based_comp":           10_833_000_000,
        "basic_eps":                          6.16,
        "diluted_eps":                        6.13,
        "basic_shares_out":       15_744_231_000,
        "diluted_shares_out":     15_812_547_000,
    },
    "FY2024": {
        "total_revenue":             391_035_000_000,
        "cost_of_revenue":           210_352_000_000,
        "gross_profit":              180_683_000_000,
        "research_development":       31_370_000_000,
        "selling_general_admin":      26_146_000_000,
        "operating_income":          123_167_000_000,
        "interest_expense":            3_002_000_000,
        "interest_income":             3_567_000_000,
        "pretax_income":             123_485_000_000,
        "income_tax":                 29_749_000_000,
        "net_income":                 93_736_000_000,
        "ebitda":                    134_661_000_000,
        "depreciation_amortization":  11_494_000_000,
        "stock_based_comp":           11_688_000_000,
        "basic_eps":                          6.11,
        "diluted_eps":                        6.08,
        "basic_shares_out":       15_343_783_000,
        "diluted_shares_out":     15_408_095_000,
    },
}

# Balance sheet data (annual)
BALANCE_DATA = {
    "FY2021": {
        "cash_and_equivalents":       34_940_000_000,
        "short_term_investments":     27_699_000_000,
        "accounts_receivable":        26_278_000_000,
        "inventory":                   6_580_000_000,
        "total_current_assets":      134_836_000_000,
        "property_plant_equipment_net": 39_440_000_000,
        "goodwill":                        0,
        "intangible_assets":               0,
        "total_assets":              351_002_000_000,
        "accounts_payable":           54_763_000_000,
        "short_term_debt":             9_613_000_000,
        "current_portion_lt_debt":     9_613_000_000,
        "total_current_liabilities": 125_481_000_000,
        "long_term_debt":            109_106_000_000,
        "total_liabilities":         287_912_000_000,
        "total_stockholders_equity":  63_090_000_000,
        "retained_earnings":           5_562_000_000,
    },
    "FY2022": {
        "cash_and_equivalents":       23_646_000_000,
        "short_term_investments":     24_658_000_000,
        "accounts_receivable":        28_184_000_000,
        "inventory":                   4_946_000_000,
        "total_current_assets":      135_405_000_000,
        "property_plant_equipment_net": 42_117_000_000,
        "goodwill":                        0,
        "intangible_assets":               0,
        "total_assets":              352_755_000_000,
        "accounts_payable":           64_115_000_000,
        "short_term_debt":            11_128_000_000,
        "current_portion_lt_debt":    11_128_000_000,
        "total_current_liabilities": 153_982_000_000,
        "long_term_debt":             98_959_000_000,
        "total_liabilities":         302_083_000_000,
        "total_stockholders_equity":  50_672_000_000,
        "retained_earnings":          -3_068_000_000,
    },
    "FY2023": {
        "cash_and_equivalents":       29_965_000_000,
        "short_term_investments":     31_590_000_000,
        "accounts_receivable":        29_508_000_000,
        "inventory":                   6_331_000_000,
        "total_current_assets":      143_566_000_000,
        "property_plant_equipment_net": 43_715_000_000,
        "goodwill":                        0,
        "intangible_assets":               0,
        "total_assets":              352_583_000_000,
        "accounts_payable":           62_611_000_000,
        "short_term_debt":             5_985_000_000,
        "current_portion_lt_debt":     9_822_000_000,
        "total_current_liabilities": 145_308_000_000,
        "long_term_debt":             95_281_000_000,
        "total_liabilities":         290_437_000_000,
        "total_stockholders_equity":  62_146_000_000,
        "retained_earnings":             -214_000_000,
    },
    "FY2024": {
        "cash_and_equivalents":       29_943_000_000,
        "short_term_investments":     35_228_000_000,
        "accounts_receivable":        32_833_000_000,
        "inventory":                   7_286_000_000,
        "total_current_assets":      152_987_000_000,
        "property_plant_equipment_net": 44_856_000_000,
        "goodwill":                        0,
        "intangible_assets":               0,
        "total_assets":              364_980_000_000,
        "accounts_payable":           68_960_000_000,
        "short_term_debt":            10_912_000_000,
        "current_portion_lt_debt":    10_912_000_000,
        "total_current_liabilities": 176_392_000_000,
        "long_term_debt":             96_822_000_000,
        "total_liabilities":         308_030_000_000,
        "total_stockholders_equity":  56_950_000_000,
        "retained_earnings":          -1_264_000_000,
    },
}

# Cash flow statement data (annual)
CASHFLOW_DATA = {
    "FY2021": {
        "operating_cash_flow":       104_038_000_000,
        "capex":                     -11_085_000_000,
        "free_cash_flow":             92_953_000_000,
        "dividends_paid":            -14_467_000_000,
        "share_repurchases":         -85_500_000_000,
        "depreciation_cf":            11_284_000_000,
        "stock_based_comp_cf":         7_906_000_000,
        "change_in_working_capital":  -4_911_000_000,
    },
    "FY2022": {
        "operating_cash_flow":       122_151_000_000,
        "capex":                     -10_708_000_000,
        "free_cash_flow":            111_443_000_000,
        "dividends_paid":            -14_841_000_000,
        "share_repurchases":         -89_402_000_000,
        "depreciation_cf":            11_104_000_000,
        "stock_based_comp_cf":         9_038_000_000,
        "change_in_working_capital":   1_200_000_000,
    },
    "FY2023": {
        "operating_cash_flow":       110_543_000_000,
        "capex":                     -10_959_000_000,
        "free_cash_flow":             99_584_000_000,
        "dividends_paid":            -15_025_000_000,
        "share_repurchases":         -77_550_000_000,
        "depreciation_cf":            11_519_000_000,
        "stock_based_comp_cf":        10_833_000_000,
        "change_in_working_capital":  -6_577_000_000,
    },
    "FY2024": {
        "operating_cash_flow":       118_254_000_000,
        "capex":                      -9_959_000_000,
        "free_cash_flow":            108_295_000_000,
        "dividends_paid":            -15_234_000_000,
        "share_repurchases":         -94_949_000_000,
        "depreciation_cf":            11_494_000_000,
        "stock_based_comp_cf":        11_688_000_000,
        "change_in_working_capital":   3_651_000_000,
    },
}

# Representative daily prices (last trading day of each FY, plus recent)
PRICE_DATA = [
    ("AAPL", "2021-09-30", 141.50, 142.20, 139.80, 141.11, 89_000_000),
    ("AAPL", "2022-09-30", 138.20, 140.30, 137.50, 138.20, 95_000_000),
    ("AAPL", "2023-09-29", 171.50, 172.40, 170.10, 171.21, 51_000_000),
    ("AAPL", "2024-09-30", 233.00, 234.50, 231.20, 233.28, 56_000_000),
    ("AAPL", "2025-01-31", 236.00, 237.80, 234.20, 236.12, 62_000_000),
]


def seed():
    """Insert AAPL test data into DuckDB."""
    con = init_db()
    try:
        now = datetime.now()

        # Company
        con.execute(
            """
            INSERT INTO company (ticker, name, sector, currency, fiscal_year_end, last_ingested)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (ticker) DO UPDATE SET
                name = EXCLUDED.name,
                sector = EXCLUDED.sector,
                fiscal_year_end = EXCLUDED.fiscal_year_end,
                last_ingested = EXCLUDED.last_ingested
            """,
            list(COMPANY) + [now],
        )

        # Income statement
        for period, items in INCOME_DATA.items():
            for line_item, value in items.items():
                con.execute(
                    """
                    INSERT INTO financials
                        (ticker, period, period_type, statement, line_item,
                         value, is_forecast, forecast_scenario, updated_at)
                    VALUES (?, ?, 'annual', 'income', ?, ?, false, 'actual', ?)
                    ON CONFLICT (ticker, period, statement, line_item,
                                 is_forecast, forecast_scenario)
                    DO UPDATE SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at
                    """,
                    ["AAPL", period, line_item, float(value), now],
                )

        # Balance sheet
        for period, items in BALANCE_DATA.items():
            for line_item, value in items.items():
                con.execute(
                    """
                    INSERT INTO financials
                        (ticker, period, period_type, statement, line_item,
                         value, is_forecast, forecast_scenario, updated_at)
                    VALUES (?, ?, 'annual', 'balance', ?, ?, false, 'actual', ?)
                    ON CONFLICT (ticker, period, statement, line_item,
                                 is_forecast, forecast_scenario)
                    DO UPDATE SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at
                    """,
                    ["AAPL", period, line_item, float(value), now],
                )

        # Cash flow
        for period, items in CASHFLOW_DATA.items():
            for line_item, value in items.items():
                con.execute(
                    """
                    INSERT INTO financials
                        (ticker, period, period_type, statement, line_item,
                         value, is_forecast, forecast_scenario, updated_at)
                    VALUES (?, ?, 'annual', 'cashflow', ?, ?, false, 'actual', ?)
                    ON CONFLICT (ticker, period, statement, line_item,
                                 is_forecast, forecast_scenario)
                    DO UPDATE SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at
                    """,
                    ["AAPL", period, line_item, float(value), now],
                )

        # Prices
        for row in PRICE_DATA:
            con.execute(
                """
                INSERT INTO prices (ticker, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (ticker, date)
                DO UPDATE SET close = EXCLUDED.close
                """,
                list(row),
            )

        # Verify counts
        fin_count = con.execute(
            "SELECT COUNT(*) FROM financials WHERE ticker = 'AAPL'"
        ).fetchone()[0]
        price_count = con.execute(
            "SELECT COUNT(*) FROM prices WHERE ticker = 'AAPL'"
        ).fetchone()[0]

        print(f"Seeded AAPL: {fin_count} financial rows, {price_count} price rows")

    finally:
        con.close()


if __name__ == "__main__":
    seed()
