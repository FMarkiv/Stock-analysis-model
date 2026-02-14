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
