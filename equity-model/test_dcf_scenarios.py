"""
End-to-end test: run DCF valuation, sensitivity analysis, scenario
comparison, and Monte Carlo simulation on AAPL using seeded data.

Usage:
    python test_dcf_scenarios.py
"""

import os
import sys

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from db.schema import reset_schema, get_connection
from seed_aapl import seed as seed_aapl


# ---------------------------------------------------------------------------
# Seed synthetic weekly prices for AAPL + SPY (needed for beta calculation)
# ---------------------------------------------------------------------------

def _seed_weekly_prices(db_path: str) -> None:
    """Generate ~2.5 years of synthetic weekly prices for AAPL and SPY.

    Uses geometric Brownian motion with realistic parameters and a
    correlation of ~0.80 between the two series.
    """
    rng = np.random.default_rng(123)

    weeks = 130  # ~2.5 years
    start = pd.Timestamp("2022-07-01")
    dates = pd.bdate_range(start, periods=weeks, freq="W-FRI")

    # Correlated weekly returns
    mu_aapl, sigma_aapl = 0.002, 0.035  # ~10% annual, ~18% vol
    mu_spy, sigma_spy = 0.0015, 0.022   # ~8% annual, ~11% vol
    rho = 0.80

    z1 = rng.standard_normal(weeks)
    z2 = rng.standard_normal(weeks)
    z_aapl = z1
    z_spy = rho * z1 + np.sqrt(1 - rho**2) * z2

    ret_aapl = mu_aapl + sigma_aapl * z_aapl
    ret_spy = mu_spy + sigma_spy * z_spy

    price_aapl = 145.0 * np.cumprod(1 + ret_aapl)
    price_spy = 390.0 * np.cumprod(1 + ret_spy)

    con = get_connection(db_path)
    try:
        for i, dt in enumerate(dates):
            d = dt.strftime("%Y-%m-%d")
            p_a = float(price_aapl[i])
            p_s = float(price_spy[i])

            con.execute(
                """
                INSERT INTO prices (ticker, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (ticker, date) DO UPDATE SET close = EXCLUDED.close
                """,
                ["AAPL", d, p_a * 0.998, p_a * 1.005, p_a * 0.994, p_a, 50_000_000],
            )
            con.execute(
                """
                INSERT INTO prices (ticker, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (ticker, date) DO UPDATE SET close = EXCLUDED.close
                """,
                ["SPY", d, p_s * 0.998, p_s * 1.005, p_s * 0.994, p_s, 80_000_000],
            )

        count_a = con.execute(
            "SELECT COUNT(*) FROM prices WHERE ticker = 'AAPL'"
        ).fetchone()[0]
        count_s = con.execute(
            "SELECT COUNT(*) FROM prices WHERE ticker = 'SPY'"
        ).fetchone()[0]
        print(f"   Seeded {count_a} AAPL + {count_s} SPY weekly price rows")
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_usd(val: float) -> str:
    if abs(val) >= 1e12:
        return f"${val / 1e12:,.2f}T"
    if abs(val) >= 1e9:
        return f"${val / 1e9:,.1f}B"
    if abs(val) >= 1e6:
        return f"${val / 1e6:,.1f}M"
    return f"${val:,.2f}"


def _fmt_pct(val: float) -> str:
    return f"{val:+.1%}" if val != 0 else "0.0%"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    db_path = os.path.join(_PROJECT_ROOT, "data", "equity.duckdb")

    print("=" * 72)
    print("  DCF & Scenario Analysis — AAPL End-to-End Test")
    print("=" * 72)

    # 0. Reset and seed
    print("\n0. Preparing database...")
    reset_schema(db_path)
    seed_aapl()
    _seed_weekly_prices(db_path)

    # Delayed imports so the DB is ready before FinancialModel.__init__
    from model.dcf import DCFValuation
    from model.scenarios import run_scenarios, monte_carlo

    # ------------------------------------------------------------------
    # 1. DCF Valuation
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  1. DCF VALUATION")
    print("=" * 72)

    dcf = DCFValuation("AAPL", db_path)

    # 1a. WACC
    print("\n--- WACC Components ---")
    wacc = dcf.compute_wacc()
    wacc_items = [
        ("Risk-free rate",       f"{wacc['risk_free_rate']:.2%}"),
        ("Equity risk premium",  f"{wacc['equity_risk_premium']:.2%}"),
        ("Beta",                 f"{wacc['beta']:.2f}"),
        ("Cost of equity",       f"{wacc['cost_of_equity']:.2%}"),
        ("Cost of debt (pre-tax)", f"{wacc['cost_of_debt_pretax']:.2%}"),
        ("Cost of debt (after-tax)", f"{wacc['cost_of_debt']:.2%}"),
        ("Tax rate",             f"{wacc['tax_rate']:.1%}"),
        ("Debt / Equity",        f"{wacc['debt_to_equity']:.2f}x"),
        ("Weight equity",        f"{wacc['weight_equity']:.1%}"),
        ("Weight debt",          f"{wacc['weight_debt']:.1%}"),
        ("WACC",                 f"{wacc['wacc']:.2%}"),
    ]
    for label, val in wacc_items:
        print(f"  {label:30s} {val}")

    # 1b. DCF result
    print("\n--- Base-Case DCF ---")
    result = dcf.dcf_valuation(scenario="base")
    dcf_items = [
        ("PV of projected FCFs",       _fmt_usd(result["pv_fcfs"])),
        ("Terminal value (perpetuity)", _fmt_usd(result["terminal_value_perpetuity"])),
        ("Terminal value (exit mult.)", _fmt_usd(result["terminal_value_exit_multiple"])),
        ("PV of TV (perpetuity)",       _fmt_usd(result["pv_terminal_perpetuity"])),
        ("PV of TV (exit mult.)",       _fmt_usd(result["pv_terminal_exit_multiple"])),
        ("EV (perpetuity method)",      _fmt_usd(result["ev_perpetuity_method"])),
        ("EV (exit multiple method)",   _fmt_usd(result["ev_exit_multiple_method"])),
        ("Enterprise value (avg)",      _fmt_usd(result["enterprise_value"])),
        ("Net debt",                    _fmt_usd(result["net_debt"])),
        ("Equity value",                _fmt_usd(result["equity_value"])),
        ("Diluted shares",              f"{result['diluted_shares'] / 1e9:.2f}B"),
        ("Implied share price",         f"${result['implied_price']:,.2f}"),
        ("Current price",               f"${result['current_price']:,.2f}"),
        ("Upside / downside",           _fmt_pct(result["upside_downside"])),
    ]
    for label, val in dcf_items:
        print(f"  {label:30s} {val}")

    # Projected FCFs
    print("\n  Projected Unlevered FCFs:")
    for period, fcf in result["projected_fcfs"].items():
        print(f"    {period}: {_fmt_usd(fcf)}")

    # ------------------------------------------------------------------
    # 2. Sensitivity Table
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  2. SENSITIVITY TABLE (Terminal Growth vs WACC)")
    print("=" * 72)

    sens = dcf.sensitivity_table(
        variable1="terminal_growth",
        range1=(-0.01, 0.02, 0.005),
        variable2="wacc",
        range2=(-0.02, 0.02, 0.005),
    )
    # Format for display
    print(f"\n  Base WACC = {wacc['wacc']:.2%}, "
          f"Base terminal growth = {result['terminal_growth']:.2%}\n")
    # Header row
    tg_wacc_label = "TG \\ WACC"
    header = f"{tg_wacc_label:>10s}"
    for col in sens.columns:
        header += f" {col:>8.2%}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for idx in sens.index:
        row_str = f"  {idx:>8.2%}"
        for col in sens.columns:
            val = sens.at[idx, col]
            row_str += f" ${val:>7.0f}"
        print(row_str)

    # ------------------------------------------------------------------
    # 3. Multiples Valuation
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  3. MULTIPLES VALUATION")
    print("=" * 72)

    multiples = dcf.multiples_valuation()
    if multiples:
        print(f"\n  Current price: ${multiples['current_price']:,.2f}\n")
        for method_key, label in [
            ("forward_pe", "Forward P/E"),
            ("ev_ebitda", "EV/EBITDA"),
            ("price_fcf", "Price/FCF"),
        ]:
            m = multiples[method_key]
            print(f"  {label}:")
            print(f"    Hist avg multiple:  {m['hist_avg_multiple']:.1f}x")
            print(f"    Implied price:      ${m['implied_price']:,.2f}")
    else:
        print("  (No multiples data available)")

    # ------------------------------------------------------------------
    # 4. Scenario Comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  4. SCENARIO COMPARISON (Bull / Base / Bear)")
    print("=" * 72)

    scenarios_df = run_scenarios("AAPL", db_path)
    print()
    # Format the DataFrame for display
    display = scenarios_df.copy()
    display["implied_price"] = display["implied_price"].map(lambda x: f"${x:,.2f}")
    display["upside_downside"] = display["upside_downside"].map(lambda x: f"{x:+.1%}")
    display["revenue_growth"] = display["revenue_growth"].map(lambda x: f"{x:.1%}")
    display["operating_margin"] = display["operating_margin"].map(lambda x: f"{x:.1%}")
    display["wacc"] = display["wacc"].map(lambda x: f"{x:.2%}")
    print(display.to_string())

    # ------------------------------------------------------------------
    # 5. Monte Carlo
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  5. MONTE CARLO SIMULATION (10,000 iterations)")
    print("=" * 72)

    mc = monte_carlo("AAPL", db_path, iterations=10_000)
    print(f"\n  Iterations:  {mc['iterations']:,}")
    print(f"  Mean:        ${mc['mean']:,.2f}")
    print(f"  Std Dev:     ${mc['std']:,.2f}")
    print("\n  Percentile Distribution:")
    for label, val in mc["percentiles"].items():
        print(f"    {label:>4s}:  ${val:,.2f}")

    # Simple text histogram
    values = mc["all_values"]
    p5 = np.percentile(values, 5)
    p95 = np.percentile(values, 95)
    bins = np.linspace(p5, p95, 21)
    counts, edges = np.histogram(values, bins=bins)
    max_count = max(counts)
    bar_width = 40

    print(f"\n  Fair Value Distribution (5th-95th percentile):")
    print(f"  {'Range':>22s}  {'Count':>6s}  Bar")
    print("  " + "-" * 60)
    for i, count in enumerate(counts):
        lo = edges[i]
        hi = edges[i + 1]
        bar_len = int(count / max_count * bar_width) if max_count > 0 else 0
        bar = "#" * bar_len
        print(f"  ${lo:>8,.0f}-${hi:>8,.0f}  {count:>6d}  {bar}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  All tests complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
