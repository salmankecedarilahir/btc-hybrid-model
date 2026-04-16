"""
test_hybrid.py — Validasi btc_trading_signals.csv (Phase 4 output).
"""

import sys
from pathlib import Path

import pandas as pd

INPUT_PATH = Path(__file__).parent / "data" / "btc_trading_signals.csv"
DIV = "═" * 62
SEP = "─" * 62


def load() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        print(f"[ERROR] File tidak ditemukan: {INPUT_PATH}")
        print("        Jalankan hybrid_engine.py terlebih dahulu.")
        sys.exit(1)
    df = pd.read_csv(INPUT_PATH, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def print_signal_summary(df: pd.DataFrame) -> None:
    print(f"\n{SEP}")
    print("  Signal Summary")
    print(SEP)

    total = len(df)
    for label, emoji in [("LONG", "▲"), ("SHORT", "▼"), ("NONE", "─")]:
        n   = (df["signal"] == label).sum()
        pct = 100 * n / total
        print(f"  {emoji} {label:<6} : {n:>5} candles  ({pct:5.2f}%)")
    print(f"\n  Total candles : {total:,}")


def print_strength_distribution(df: pd.DataFrame) -> None:
    print(f"\n{SEP}")
    print("  Signal Strength Distribution")
    print(SEP)

    total = len(df)
    for label in ["STRONG", "NORMAL", "WEAK", "NONE"]:
        n   = (df["signal_strength"] == label).sum()
        pct = 100 * n / total
        bar = "█" * int(pct / 3)
        print(f"  {label:<7} : {n:>5} candles  ({pct:5.1f}%)  {bar}")


def print_top_signals(df: pd.DataFrame, n: int = 5) -> None:
    print(f"\n{SEP}")
    print(f"  Top {n} Candles by Hybrid Score")
    print(SEP)

    cols = ["timestamp", "close", "regime", "signal",
            "signal_strength", "trend_score", "derivatives_score", "hybrid_score"]
    cols = [c for c in cols if c in df.columns]

    top = df[df["signal"] != "NONE"].nlargest(n, "hybrid_score")[cols].copy()

    if top.empty:
        print("  Tidak ada signal LONG/SHORT yang ditemukan.")
        return

    top["timestamp"] = top["timestamp"].dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M")
    if "close" in top.columns:
        top["close"] = top["close"].map("${:,.2f}".format)

    print(top.to_string(index=False))


def print_recent_signals(df: pd.DataFrame, n: int = 5) -> None:
    print(f"\n{SEP}")
    print(f"  Last {n} Candles (terbaru)")
    print(SEP)

    cols = ["timestamp", "close", "regime", "signal",
            "signal_strength", "hybrid_score", "atr_percentile"]
    cols = [c for c in cols if c in df.columns]

    tail = df.tail(n)[cols].copy()
    tail["timestamp"] = tail["timestamp"].dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M")
    if "close" in tail.columns:
        tail["close"] = tail["close"].map("${:,.2f}".format)
    if "atr_percentile" in tail.columns:
        tail["atr_percentile"] = tail["atr_percentile"].map(
            lambda x: f"{x:.1f}" if pd.notna(x) else "NaN"
        )

    print(tail.to_string(index=False))


def main() -> None:
    print(DIV)
    print("  BTC Hybrid Model — Phase 4: Signal Validation")
    print(DIV)

    df = load()
    print(f"\n  Loaded : {len(df):,} baris")
    print(f"  Range  : {df['timestamp'].iloc[0].strftime('%Y-%m-%d')} "
          f"→ {df['timestamp'].iloc[-1].strftime('%Y-%m-%d')}")

    print_signal_summary(df)
    print_strength_distribution(df)
    print_top_signals(df)
    print_recent_signals(df)

    print(f"\n{DIV}")
    print("  Validation complete.")
    print(DIV)


if __name__ == "__main__":
    main()
