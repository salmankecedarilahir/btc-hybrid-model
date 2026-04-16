"""
data_fetcher.py — Fetch BTC-USD OHLCV dari Yahoo Finance.

Strategi Opsi C:
  - 2017-01-01 → 2024-03-01 : interval 1D → resample ke 4H (forward fill)
  - 2024-03-01 → sekarang   : interval 1H → resample ke 4H (akurat)
  - Gabungkan kedua dataset, deduplicate, sort ascending

Output: data/btc_4h_raw.csv
"""

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).parent
OUTPUT_PATH = BASE_DIR / "data" / "btc_4h_raw.csv"

TICKER          = "BTC-USD"
START_HISTORICAL = "2017-01-01"
CUTOVER_DATE    = "2024-03-01"   # batas antara 1D dan 1H


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase kolom dan rename datetime → timestamp."""
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    for col in ["datetime", "date", "index"]:
        if col in df.columns:
            return df.rename(columns={col: "timestamp"})
    if df.columns[0] != "timestamp":
        return df.rename(columns={df.columns[0]: "timestamp"})
    return df


def _keep_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    return df[[c for c in cols if c in df.columns]].copy()


# ── Fetch 1D (2017 → cutover) ─────────────────────────────────────────────────

def fetch_daily(ticker: str = TICKER,
                start: str  = START_HISTORICAL,
                end: str    = CUTOVER_DATE) -> pd.DataFrame:
    """Fetch 1D OHLCV dari Yahoo Finance — history panjang sejak 2017."""
    log.info("Fetching 1D  : %s → %s …", start, end)
    try:
        raw = yf.Ticker(ticker).history(start=start, end=end, interval="1d")
        if raw.empty:
            raise ValueError("Data 1D kosong.")
        raw = _normalize_columns(raw.reset_index())
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
        raw = _keep_ohlcv(raw).sort_values("timestamp").reset_index(drop=True)
        log.info("  1D fetched : %d baris | %s → %s",
                 len(raw),
                 raw["timestamp"].iloc[0].strftime("%Y-%m-%d"),
                 raw["timestamp"].iloc[-1].strftime("%Y-%m-%d"))
        return raw
    except Exception as e:
        log.error("Gagal fetch 1D: %s", e)
        return pd.DataFrame()


# ── Fetch 1H (cutover → sekarang) ────────────────────────────────────────────

def fetch_hourly(ticker: str = TICKER) -> pd.DataFrame:
    """Fetch 1H OHLCV dari Yahoo Finance — max 730 hari terakhir."""
    log.info("Fetching 1H  : 730 hari terakhir …")
    try:
        raw = yf.Ticker(ticker).history(period="730d", interval="1h")
        if raw.empty:
            raise ValueError("Data 1H kosong.")
        raw = _normalize_columns(raw.reset_index())
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
        raw = _keep_ohlcv(raw).sort_values("timestamp").reset_index(drop=True)
        log.info("  1H fetched : %d baris | %s → %s",
                 len(raw),
                 raw["timestamp"].iloc[0].strftime("%Y-%m-%d"),
                 raw["timestamp"].iloc[-1].strftime("%Y-%m-%d"))
        return raw
    except Exception as e:
        log.error("Gagal fetch 1H: %s", e)
        return pd.DataFrame()


# ── Resample ──────────────────────────────────────────────────────────────────

def resample_to_4h(df: pd.DataFrame, source: str = "") -> pd.DataFrame:
    """Resample OHLCV ke 4H."""
    if df.empty:
        return df
    df = df.set_index("timestamp")
    out = df.resample("4h").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna(how="all").reset_index()
    log.info("  → 4H %-6s: %d candles | %s → %s",
             source,
             len(out),
             out["timestamp"].iloc[0].strftime("%Y-%m-%d"),
             out["timestamp"].iloc[-1].strftime("%Y-%m-%d"))
    return out


# ── Merge ─────────────────────────────────────────────────────────────────────

def merge_datasets(df_hist: pd.DataFrame, df_recent: pd.DataFrame) -> pd.DataFrame:
    """
    Gabungkan historical (1D→4H) dan recent (1H→4H).
    Data recent lebih akurat → prioritas untuk periode overlap.
    """
    if df_hist.empty and df_recent.empty:
        raise ValueError("Kedua dataset kosong.")
    if df_hist.empty:
        return df_recent
    if df_recent.empty:
        return df_hist

    cutover = pd.Timestamp(CUTOVER_DATE, tz="UTC")

    # Potong historical sebelum cutover, recent dari cutover
    hist_cut   = df_hist[df_hist["timestamp"] < cutover].copy()
    recent_cut = df_recent[df_recent["timestamp"] >= cutover].copy()

    combined = pd.concat([hist_cut, recent_cut], ignore_index=True)
    combined = combined.drop_duplicates("timestamp")
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    log.info("Merged total : %d candles | %s → %s",
             len(combined),
             combined["timestamp"].iloc[0].strftime("%Y-%m-%d"),
             combined["timestamp"].iloc[-1].strftime("%Y-%m-%d"))
    return combined


# ── Clean ─────────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").ffill()
    df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0.0)
    df = df.dropna(subset=["open", "close"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ── Save ──────────────────────────────────────────────────────────────────────

def save_data(df: pd.DataFrame, path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("Tersimpan → %s  (%d baris, %d kolom)", path, len(df), len(df.columns))
    log.info("Kolom: %s", list(df.columns))


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> pd.DataFrame:
    log.info("═" * 60)
    log.info("Data Fetcher — BTC-USD 4H (Opsi C: 2017 + 2 tahun 1H)")
    log.info("  Historical : 1D  %s → %s", START_HISTORICAL, CUTOVER_DATE)
    log.info("  Recent     : 1H  %s → sekarang", CUTOVER_DATE)
    log.info("═" * 60)

    # Fetch
    daily  = fetch_daily()
    hourly = fetch_hourly()

    # Resample ke 4H
    daily_4h  = resample_to_4h(daily,  "1D")
    hourly_4h = resample_to_4h(hourly, "1H")

    # Gabungkan
    df = merge_datasets(daily_4h, hourly_4h)
    df = clean_data(df)

    log.info("═" * 60)
    log.info("SUMMARY")
    log.info("  Total candles : %d", len(df))
    log.info("  Date range    : %s → %s",
             df["timestamp"].iloc[0].strftime("%Y-%m-%d"),
             df["timestamp"].iloc[-1].strftime("%Y-%m-%d"))
    log.info("  Span          : %.0f hari",
             (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).days)
    log.info("═" * 60)

    save_data(df)
    return df


if __name__ == "__main__":
    run()
