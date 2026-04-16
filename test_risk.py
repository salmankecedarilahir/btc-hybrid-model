"""
test_risk.py — Validasi Phase 6 Risk Management Results.
"""

import sys
from pathlib import Path

import pandas as pd

INPUT_PATH = Path(__file__).parent / "data" / "btc_risk_managed_results.csv"
DIV = "═" * 58
SEP = "─" * 58
INITIAL_EQUITY = 10_000.0


def load() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        print(f"[ERROR] File tidak ditemukan: {INPUT_PATH}")
        print("        Jalankan risk_engine.py terlebih dahulu.")
        sys.exit(1)
    df = pd.read_csv(INPUT_PATH, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def print_summary(df: pd.DataFrame) -> None:
    final_equity = df["equity"].iloc[-1]
    total_return = (final_equity - INITIAL_EQUITY) / INITIAL_EQUITY * 100
    max_dd       = df["drawdown"].min() * 100

    active = df[df["risk_adjusted_return"] != 0]["risk_adjusted_return"]
    sharpe = 0.0
    if len(active) > 1 and active.std() > 0:
        import numpy as np
        sharpe = (active.mean() / active.std()) * (2190 ** 0.5)

    kill_rows = df[df.get("kill_switch_active", pd.Series(False, index=df.index)) == True]
    kill_info = kill_rows["timestamp"].iloc[0].strftime("%Y-%m-%d") if not kill_rows.empty else "Not triggered"

    print(f"\n{DIV}")
    print("  RISK MANAGEMENT SUMMARY")
    print(DIV)
    print(f"  {'Initial Equity':<30}: ${INITIAL_EQUITY:>10,.2f}")
    print(f"  {'Final Equity':<30}: ${final_equity:>10,.2f}")
    print(SEP)
    print(f"  {'Total Return':<30}: {total_return:>+10.2f}%")
    print(f"  {'Max Drawdown':<30}: {max_dd:>10.2f}%")
    print(f"  {'Risk-Adjusted Sharpe':<30}: {sharpe:>10.3f}")
    print(SEP)
    print(f"  {'Kill Switch':<30}: {kill_info:>20}")
    print(DIV)


def print_top_equity(df: pd.DataFrame, n: int = 5) -> None:
    print(f"\n{SEP}")
    print(f"  Top {n} Highest Equity Points")
    print(SEP)

    cols = ["timestamp", "close", "signal", "equity",
            "drawdown", "risk_adjusted_return"]
    cols = [c for c in cols if c in df.columns]

    top = df.nlargest(n, "equity")[cols].copy()
    top["timestamp"] = top["timestamp"].dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M")
    if "close" in top.columns:
        top["close"]  = top["close"].map("${:,.2f}".format)
    top["equity"]     = top["equity"].map("${:,.2f}".format)
    top["drawdown"]   = top["drawdown"].map("{:.2%}".format)
    if "risk_adjusted_return" in top.columns:
        top["risk_adjusted_return"] = top["risk_adjusted_return"].map("{:+.4f}".format)

    print(top.to_string(index=False))


def print_worst_drawdowns(df: pd.DataFrame, n: int = 5) -> None:
    print(f"\n{SEP}")
    print(f"  Top {n} Deepest Drawdown Points")
    print(SEP)

    cols = ["timestamp", "close", "signal", "equity",
            "drawdown", "running_max_equity"]
    cols = [c for c in cols if c in df.columns]

    worst = df.nsmallest(n, "drawdown")[cols].copy()
    worst["timestamp"] = worst["timestamp"].dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M")
    if "close" in worst.columns:
        worst["close"]             = worst["close"].map("${:,.2f}".format)
    worst["equity"]                = worst["equity"].map("${:,.2f}".format)
    worst["drawdown"]              = worst["drawdown"].map("{:.2%}".format)
    if "running_max_equity" in worst.columns:
        worst["running_max_equity"] = worst["running_max_equity"].map("${:,.2f}".format)

    print(worst.to_string(index=False))


def main() -> None:
    print(DIV)
    print("  BTC Hybrid Model — Phase 6: Risk Management Validation")
    print(DIV)

    df = load()
    print(f"\n  Loaded : {len(df):,} baris")
    print(f"  Range  : {df['timestamp'].iloc[0].strftime('%Y-%m-%d')} "
          f"→ {df['timestamp'].iloc[-1].strftime('%Y-%m-%d')}")

    print_summary(df)
    print_top_equity(df)
    print_worst_drawdowns(df)

    print(f"\n{DIV}")
    print("  Validation complete.")
    print(DIV)


if __name__ == "__main__":
    main()
