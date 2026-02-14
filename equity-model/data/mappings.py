"""
Mappings between external data source field names and internal line-item names.
"""

# SEC EDGAR XBRL tags -> internal line-item names
XBRL_TO_LINE_ITEM: dict[str, str] = {
    "Revenues": "revenue",
    "CostOfGoodsAndServicesSold": "cost_of_revenue",
    "GrossProfit": "gross_profit",
    "OperatingIncomeLoss": "operating_income",
    "NetIncomeLoss": "net_income",
}

STATEMENT_MAP: dict[str, str] = {
    "income": "income",
    "balance": "balance",
    "cashflow": "cashflow",
}

# ---------------------------------------------------------------------------
# yfinance field name -> standardised line-item name
# Organised by financial statement type.
# Multiple yfinance names may map to the same internal name (aliases).
# ---------------------------------------------------------------------------

YFINANCE_INCOME_MAP: dict[str, str] = {
    # Revenue & costs
    "Total Revenue": "total_revenue",
    "Cost Of Revenue": "cost_of_revenue",
    "Gross Profit": "gross_profit",
    # Operating expenses
    "Research And Development": "research_development",
    "Research Development": "research_development",
    "Selling General And Administration": "selling_general_admin",
    "Selling General And Administrative": "selling_general_admin",
    # Operating income
    "Operating Income": "operating_income",
    # Interest
    "Interest Expense": "interest_expense",
    "Interest Expense Non Operating": "interest_expense",
    "Interest Income": "interest_income",
    "Interest Income Non Operating": "interest_income",
    # Pre-tax & tax
    "Pretax Income": "pretax_income",
    "Tax Provision": "income_tax",
    "Income Tax Expense": "income_tax",
    # Net income
    "Net Income": "net_income",
    "Net Income Common Stockholders": "net_income",
    # Per-share
    "Basic EPS": "basic_eps",
    "Diluted EPS": "diluted_eps",
    "Basic Average Shares": "basic_shares_out",
    "Diluted Average Shares": "diluted_shares_out",
    # EBITDA & non-cash
    "EBITDA": "ebitda",
    "Normalized EBITDA": "ebitda",
    "Reconciled Depreciation": "depreciation_amortization",
    "Depreciation And Amortization In Income Statement": "depreciation_amortization",
    # Stock-based compensation (sometimes on income statement)
    "Stock Based Compensation": "stock_based_comp",
}

YFINANCE_BALANCE_MAP: dict[str, str] = {
    # Current assets
    "Cash And Cash Equivalents": "cash_and_equivalents",
    "Cash Cash Equivalents And Short Term Investments": "cash_and_equivalents",
    "Other Short Term Investments": "short_term_investments",
    "Accounts Receivable": "accounts_receivable",
    "Receivables": "accounts_receivable",
    "Inventory": "inventory",
    "Current Assets": "total_current_assets",
    # Non-current assets
    "Net PPE": "property_plant_equipment_net",
    "Gross PPE": "property_plant_equipment_net",
    "Goodwill": "goodwill",
    "Goodwill And Other Intangible Assets": "goodwill",
    "Other Intangible Assets": "intangible_assets",
    "Total Assets": "total_assets",
    # Current liabilities
    "Accounts Payable": "accounts_payable",
    "Current Debt": "short_term_debt",
    "Current Debt And Capital Lease Obligation": "current_portion_lt_debt",
    "Current Liabilities": "total_current_liabilities",
    # Non-current liabilities
    "Long Term Debt": "long_term_debt",
    "Long Term Debt And Capital Lease Obligation": "long_term_debt",
    "Total Liabilities Net Minority Interest": "total_liabilities",
    # Equity
    "Stockholders Equity": "total_stockholders_equity",
    "Total Equity Gross Minority Interest": "total_stockholders_equity",
    "Retained Earnings": "retained_earnings",
}

YFINANCE_CASHFLOW_MAP: dict[str, str] = {
    "Operating Cash Flow": "operating_cash_flow",
    "Capital Expenditure": "capex",
    "Free Cash Flow": "free_cash_flow",
    # Dividends & buybacks
    "Common Stock Dividend Paid": "dividends_paid",
    "Cash Dividends Paid": "dividends_paid",
    "Repurchase Of Capital Stock": "share_repurchases",
    "Common Stock Payments": "share_repurchases",
    # Non-cash items
    "Depreciation And Amortization": "depreciation_cf",
    "Depreciation Amortization Depletion": "depreciation_cf",
    "Stock Based Compensation": "stock_based_comp_cf",
    # Working capital
    "Change In Working Capital": "change_in_working_capital",
    "Changes In Working Capital": "change_in_working_capital",
}
