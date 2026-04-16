"""
data_cleaner.py — Validate and clean the three data sources.

Checks performed
────────────────
1. Missing 4H candles in OHLCV (gaps in the expected 4H grid)
2. Forward-fill missing Funding Rate / OI values at 4H anchors
3. Timestamp alignment — all three datasets share the same 4H index
4. Duplicate check on every dataset
5. NaN audit after merging

Run directly:
    python data_cleaner.py
"""

import logging
import sys

import pandas as pd

from config import OHLCV_PATH, FUNDING_PATH, OI_PATH

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load(path, name: str, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            log.error("Missing file: %s — run data_fetcher.py first.", path)
            sys.exit(1)
        else:
            log.warning("WARNING: %s tidak ditemukan — %s dilewati, pipeline lanjut.", path, name)
            return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    log.info("Loaded %-12s → %d rows  [%s → %s]",
             name,
             len(df),
             df["timestamp"].iloc[0].strftime("%Y-%m-%d"),
             df["timestamp"].iloc[-1].strftime("%Y-%m-%d"))
    return df


# ── Check 1 — Missing 4H candles ─────────────────────────────────────────────

def check_missing_candles(ohlcv: pd.DataFrame) -> pd.DatetimeIndex:
    """
    Build the expected 4H UTC grid and find gaps in the OHLCV data.
    Returns a DatetimeIndex of missing timestamps.
    """
    start = ohlcv["timestamp"].min()
    end   = ohlcv["timestamp"].max()

    # Snap start to a clean 4H boundary
    start_snapped = start.floor("4h")
    expected = pd.date_range(start=start_snapped, end=end, freq="4h", tz="UTC")
    actual   = pd.DatetimeIndex(ohlcv["timestamp"])
    missing  = expected.difference(actual)

    if len(missing) == 0:
        log.info("✓ No missing 4H candles detected.")
    else:
        log.warning("[WARN] %d missing 4H candle(s) detected:", len(missing))
        for ts in missing[:20]:                          # show first 20 only
            log.warning("    %s", ts)
        if len(missing) > 20:
            log.warning("    … and %d more.", len(missing) - 20)

    return missing


# ── Check 2 — Duplicate timestamps ───────────────────────────────────────────

def check_duplicates(df: pd.DataFrame, name: str) -> pd.DataFrame:
    dups = df[df["timestamp"].duplicated()]
    if len(dups):
        log.warning("[WARN] %s has %d duplicate timestamps — dropping extras.", name, len(dups))
        df = df.drop_duplicates("timestamp")
    else:
        log.info("✓ No duplicates in %s.", name)
    return df


# ── Check 3 — Align all datasets to OHLCV 4H grid ───────────────────────────

def align_to_ohlcv(
    ohlcv:   pd.DataFrame,
    funding: pd.DataFrame,
    oi:      pd.DataFrame,
) -> pd.DataFrame:
    """
    Reindex funding and OI to the exact 4H timestamps present in OHLCV,
    forward-filling any gaps.  Returns a single merged DataFrame.
    Jika funding atau OI kosong (file tidak ditemukan), kolom akan diisi 0.
    """
    base = ohlcv.set_index("timestamp").sort_index()

    # ── Funding ──────────────────────────────────────────────────────────────
    if funding.empty:
        log.warning("[WARN] Funding data tidak tersedia — kolom funding_rate diisi 0.")
        fund_idx = pd.DataFrame({"funding_rate": 0.0}, index=base.index)
    else:
        fund_idx = funding.set_index("timestamp").sort_index()
        fund_idx = fund_idx.reindex(base.index).ffill()
        missing_fund = fund_idx["funding_rate"].isna().sum()
        if missing_fund:
            log.warning("[WARN] Funding: %d NaN after forward-fill (backfilling remaining).", missing_fund)
            fund_idx = fund_idx.bfill().fillna(0)
        else:
            log.info("✓ Funding aligned — no NaN after forward-fill.")

    # ── Open Interest ─────────────────────────────────────────────────────────
    if oi.empty:
        log.warning("[WARN] OI data tidak tersedia — kolom open_interest diisi 0.")
        oi_idx = pd.DataFrame({"open_interest": 0.0}, index=base.index)
    else:
        oi_idx = oi.set_index("timestamp").sort_index()
        oi_idx = oi_idx.reindex(base.index).ffill()
        missing_oi = oi_idx["open_interest"].isna().sum()
        if missing_oi:
            log.warning("[WARN] OI: %d NaN after forward-fill (backfilling remaining).", missing_oi)
            oi_idx = oi_idx.bfill().fillna(0)
        else:
            log.info("✓ OI aligned — no NaN after forward-fill.")

    # ── Merge ─────────────────────────────────────────────────────────────────
    merged = base.join(fund_idx, how="left").join(oi_idx, how="left")
    merged = merged.reset_index()

    total_nan = merged.isna().sum().sum()
    if total_nan:
        log.warning("[WARN] Merged dataset still has %d NaN value(s):", total_nan)
        log.warning("\n%s", merged.isna().sum()[merged.isna().sum() > 0])
    else:
        log.info("✓ Merged dataset is fully clean — 0 NaN values.")

    return merged


# ── Check 4 — Timestamp alignment report ─────────────────────────────────────

def report_alignment(
    ohlcv:   pd.DataFrame,
    funding: pd.DataFrame,
    oi:      pd.DataFrame,
) -> None:
    ohlcv_ts   = set(ohlcv["timestamp"])
    funding_ts = set(funding["timestamp"]) if not funding.empty else set()
    oi_ts      = set(oi["timestamp"])      if not oi.empty      else set()

    in_ohlcv_not_funding = ohlcv_ts - funding_ts
    in_ohlcv_not_oi      = ohlcv_ts - oi_ts

    log.info("Alignment report:")
    log.info("  OHLCV timestamps   : %d", len(ohlcv_ts))
    log.info("  Funding timestamps : %d  %s", len(funding_ts),
             "(file tidak ditemukan)" if not funding_ts else "")
    log.info("  OI timestamps      : %d  %s", len(oi_ts),
             "(file tidak ditemukan)" if not oi_ts else "")
    log.info("  OHLCV timestamps missing in Funding : %d", len(in_ohlcv_not_funding))
    log.info("  OHLCV timestamps missing in OI      : %d", len(in_ohlcv_not_oi))


# ── Main ──────────────────────────────────────────────────────────────────────

def run_cleaning() -> pd.DataFrame:
    log.info("═" * 60)
    log.info("BTC Hybrid Model — Phase 1: Data Cleaning")
    log.info("═" * 60)

    ohlcv   = _load(OHLCV_PATH,   "OHLCV",         required=True)
    funding = _load(FUNDING_PATH, "funding_4h",    required=False)
    oi      = _load(OI_PATH,      "OpenInterest",  required=False)

    log.info("─" * 60)
    log.info("[1] Checking for missing 4H candles …")
    check_missing_candles(ohlcv)

    log.info("─" * 60)
    log.info("[2] Checking for duplicates …")
    ohlcv   = check_duplicates(ohlcv,   "OHLCV")
    if not funding.empty:
        funding = check_duplicates(funding, "Funding")
    if not oi.empty:
        oi      = check_duplicates(oi,      "OI")

    log.info("─" * 60)
    log.info("[3] Timestamp alignment report …")
    report_alignment(ohlcv, funding, oi)

    log.info("─" * 60)
    log.info("[4] Aligning datasets to OHLCV 4H grid …")
    merged = align_to_ohlcv(ohlcv, funding, oi)

    log.info("═" * 60)
    log.info("Cleaning complete — merged shape: %s", merged.shape)
    log.info("Columns: %s", list(merged.columns))
    log.info("═" * 60)

    return merged


if __name__ == "__main__":
    run_cleaning()
