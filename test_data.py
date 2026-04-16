"""
test_data.py — Validate all three CSVs and print a comprehensive report.

Run directly:
    python test_data.py
"""

import sys
import logging

import pandas as pd

from config import OHLCV_PATH, FUNDING_PATH, OI_PATH

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

DIVIDER   = "═" * 65
SEPARATOR = "─" * 65


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load(path, name: str) -> pd.DataFrame:
    if not path.exists():
        log.error("File not found: %s", path)
        log.error("Run data_fetcher.py first.")
        sys.exit(1)
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def _print_section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def _report(df: pd.DataFrame, name: str, value_col: str = None) -> None:
    _print_section(f"Dataset: {name}")

    print(f"  Rows        : {len(df):,}")
    print(f"  Columns     : {list(df.columns)}")
    print(f"  Date range  : {df['timestamp'].iloc[0]}  →  {df['timestamp'].iloc[-1]}")

    total_days = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).days
    print(f"  Span (days) : {total_days}")

    # NaN audit
    nan_counts = df.isna().sum()
    nan_cols   = nan_counts[nan_counts > 0]
    if len(nan_cols):
        print(f"  [WARN] NaN found:\n{nan_cols}")
    else:
        print(f"  ✓ No NaN values")

    # Head
    print("\n  HEAD (first 3 rows):")
    print(df.head(3).to_string(index=False))

    # Tail
    print("\n  TAIL (last 3 rows):")
    print(df.tail(3).to_string(index=False))


def _candle_consistency_check(ohlcv: pd.DataFrame) -> None:
    _print_section("4H Candle Consistency Check")

    expected_per_day = 6   # 24h / 4h = 6

    ohlcv = ohlcv.copy()
    ohlcv["date"] = ohlcv["timestamp"].dt.date
    candles_per_day = ohlcv.groupby("date").size()

    total_days   = len(candles_per_day)
    perfect_days = (candles_per_day == expected_per_day).sum()
    short_days   = (candles_per_day < expected_per_day).sum()
    extra_days   = (candles_per_day > expected_per_day).sum()

    print(f"  Expected candles per day : {expected_per_day}")
    print(f"  Total days examined      : {total_days}")
    print(f"  ✓ Days with exactly 6    : {perfect_days} ({100*perfect_days/total_days:.1f}%)")
    print(f"  [WARN] Days with < 6 candles  : {short_days}")
    print(f"  [WARN] Days with > 6 candles  : {extra_days}")

    if short_days > 0:
        bad = candles_per_day[candles_per_day < expected_per_day]
        print(f"\n  Incomplete days (first 10):")
        print(bad.head(10).to_string())

    if short_days == 0 and extra_days == 0:
        print(f"\n  ✓ All {total_days} days have exactly 6 candles — data is consistent.")


def _cross_dataset_check(
    ohlcv:   pd.DataFrame,
    funding: pd.DataFrame,
    oi:      pd.DataFrame,
) -> None:
    _print_section("Cross-Dataset Overlap Check")

    # Find common date window
    start = max(
        ohlcv["timestamp"].min(),
        funding["timestamp"].min(),
        oi["timestamp"].min(),
    )
    end = min(
        ohlcv["timestamp"].max(),
        funding["timestamp"].max(),
        oi["timestamp"].max(),
    )

    print(f"  Common window : {start}  →  {end}")
    span = (end - start).days
    print(f"  Span (days)   : {span}")

    ohlcv_in   = ohlcv[(ohlcv["timestamp"] >= start)   & (ohlcv["timestamp"] <= end)]
    funding_in = funding[(funding["timestamp"] >= start) & (funding["timestamp"] <= end)]
    oi_in      = oi[(oi["timestamp"] >= start)           & (oi["timestamp"] <= end)]

    print(f"  OHLCV rows in window   : {len(ohlcv_in):,}")
    print(f"  Funding rows in window : {len(funding_in):,}")
    print(f"  OI rows in window      : {len(oi_in):,}")

    ohlcv_ts   = set(ohlcv_in["timestamp"])
    funding_ts = set(funding_in["timestamp"])
    oi_ts      = set(oi_in["timestamp"])

    gap_fund = len(ohlcv_ts - funding_ts)
    gap_oi   = len(ohlcv_ts - oi_ts)

    if gap_fund == 0:
        print("  ✓ Funding timestamps fully cover OHLCV window.")
    else:
        print(f"  [WARN] {gap_fund} OHLCV timestamps missing in Funding.")

    if gap_oi == 0:
        print("  ✓ OI timestamps fully cover OHLCV window.")
    else:
        print(f"  [WARN] {gap_oi} OHLCV timestamps missing in OI.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(DIVIDER)
    print("  BTC Hybrid Model — Phase 1: Data Validation")
    print(DIVIDER)

    ohlcv   = _load(OHLCV_PATH,   "OHLCV")
    funding = _load(FUNDING_PATH, "Funding")
    oi      = _load(OI_PATH,      "OI")

    _report(ohlcv,   "OHLCV (btc_4h_ohlcv.csv)",   value_col="close")
    _report(funding, "Funding Rate (funding_4h.csv)", value_col="funding_rate")
    _report(oi,      "Open Interest (oi_4h.csv)",    value_col="open_interest")

    _candle_consistency_check(ohlcv)
    _cross_dataset_check(ohlcv, funding, oi)

    print(f"\n{DIVIDER}")
    print("  Validation complete.")
    print(DIVIDER)


if __name__ == "__main__":
    main()
