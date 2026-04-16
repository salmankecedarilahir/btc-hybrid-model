"""
test_derivatives.py — Validasi btc_full_hybrid_dataset.csv
"""

import sys
from pathlib import Path

import pandas as pd

INPUT_PATH = Path(__file__).parent / "data" / "btc_full_hybrid_dataset.csv"
DIV = "═" * 60
SEP = "─" * 60


def load() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        print(f"[ERROR] File tidak ditemukan: {INPUT_PATH}")
        print("        Jalankan derivatives_engine.py terlebih dahulu.")
        sys.exit(1)
    df = pd.read_csv(INPUT_PATH, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def print_score_distribution(df: pd.DataFrame) -> None:
    print(f"\n{SEP}")
    print("  Derivatives Score Distribution")
    print(SEP)

    counts = df["derivatives_score"].value_counts().sort_index()
    total  = len(df)

    for score, count in counts.items():
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  Score {int(score)}  :  {count:>5} candles  ({pct:5.1f}%)  {bar}")

    print(f"\n  Total candles : {total:,}")
    mean_score = df["derivatives_score"].mean()
    print(f"  Mean score    : {mean_score:.3f}")


def print_squeeze_count(df: pd.DataFrame) -> None:
    print(f"\n{SEP}")
    print("  Potential Squeeze Summary")
    print(SEP)

    if "potential_squeeze" not in df.columns:
        print("  [WARNING] Kolom potential_squeeze tidak ditemukan.")
        return

    total    = len(df)
    squeeze  = (df["potential_squeeze"] == 1).sum()
    pct      = squeeze / total * 100

    print(f"  Total candles dengan potential_squeeze = 1 : {squeeze:,}  ({pct:.1f}%)")

    if "funding_rate" in df.columns and squeeze > 0:
        sq_df = df[df["potential_squeeze"] == 1]
        oi_col = "oi_zscore" if "oi_zscore" in sq_df.columns else "oi_change" if "oi_change" in sq_df.columns else None
        if oi_col:
            long_sq  = ((sq_df["funding_rate"] > 0.0005) & (sq_df[oi_col] > 0)).sum()
            short_sq = ((sq_df["funding_rate"] < -0.0005) & (sq_df[oi_col] > 0)).sum()
            print(f"    Long  squeeze signal  : {long_sq:,}")
            print(f"    Short squeeze signal  : {short_sq:,}")
        else:
            print("    oi_zscore / oi_change column not found — skipping squeeze breakdown")


def print_top_score_bars(df: pd.DataFrame, n: int = 5) -> None:
    print(f"\n{SEP}")
    print(f"  Top {n} Candles by Derivatives Score")
    print(SEP)

    cols = ["timestamp", "close", "regime", "funding_rate",
            "oi_change", "funding_extreme", "oi_spike",
            "potential_squeeze", "derivatives_score"]
    cols = [c for c in cols if c in df.columns]

    top = df.nlargest(n, "derivatives_score")[cols].copy()
    top["timestamp"] = top["timestamp"].dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M")

    if "close" in top.columns:
        top["close"] = top["close"].map("${:,.2f}".format)
    if "funding_rate" in top.columns:
        top["funding_rate"] = top["funding_rate"].map("{:.6f}".format)
    if "oi_change" in top.columns:
        top["oi_change"] = top["oi_change"].map("{:+.4f}".format)

    print(top.to_string(index=False))


def main() -> None:
    print(DIV)
    print("  BTC Hybrid Model — Derivatives Validation")
    print(DIV)

    df = load()
    print(f"\n  Loaded: {len(df):,} baris | "
          f"{df['timestamp'].iloc[0].strftime('%Y-%m-%d')} → "
          f"{df['timestamp'].iloc[-1].strftime('%Y-%m-%d')}")

    print_score_distribution(df)
    print_squeeze_count(df)
    print_top_score_bars(df)

    print(f"\n{DIV}")
    print("  Validation complete.")
    print(DIV)


if __name__ == "__main__":
    main()
