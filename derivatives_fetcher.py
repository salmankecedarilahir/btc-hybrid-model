"""
derivatives_fetcher.py — Fetch Binance Futures BTCUSDT funding rate history.
Start : 2019-01-01
Output: data/btc_derivatives_raw.csv
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).parent
OUTPUT_PATH = BASE_DIR / "data" / "btc_derivatives_raw.csv"

BINANCE_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
SYMBOL      = "BTCUSDT"
START_DATE  = "2019-01-01"
LIMIT       = 1000
PAUSE_SEC   = 0.3   # jeda antar request agar tidak rate-limit


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_ms(dt_str: str) -> int:
    """Convert date string 'YYYY-MM-DD' ke millisecond UTC timestamp."""
    dt = datetime.strptime(dt_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _fmt(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_funding_history(
    symbol: str     = SYMBOL,
    start_date: str = START_DATE,
    limit: int      = LIMIT,
) -> pd.DataFrame:
    """
    Paginate Binance /fapi/v1/fundingRate dari start_date hingga sekarang.
    Binance funding rate setiap 8 jam → ~3 records/hari.
    """
    start_ms = _to_ms(start_date)
    end_ms   = _now_ms()
    all_rows = []
    page     = 0

    log.info("Fetching Binance funding rate: %s | %s → now …", symbol, start_date)

    while start_ms < end_ms:
        params = {
            "symbol":    symbol,
            "startTime": start_ms,
            "endTime":   end_ms,
            "limit":     limit,
        }

        try:
            resp = requests.get(BINANCE_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            log.error("Request gagal (page %d): %s", page, e)
            break

        if not data:
            log.info("  Tidak ada data lagi — pagination selesai.")
            break

        all_rows.extend(data)
        page += 1

        last_ts = data[-1]["fundingTime"]
        log.info("  Page %d: %d records | sampai %s | total: %d",
                 page, len(data), _fmt(last_ts), len(all_rows))

        # Jika dapat kurang dari limit → sudah habis
        if len(data) < limit:
            break

        # Lanjut dari setelah record terakhir
        start_ms = last_ts + 1
        time.sleep(PAUSE_SEC)

    if not all_rows:
        log.warning("Tidak ada data funding rate yang berhasil diambil.")
        return pd.DataFrame(columns=["timestamp", "funding_rate"])

    df = pd.DataFrame(all_rows)
    log.info("Total raw records: %d", len(df))
    return df


# ── Process ───────────────────────────────────────────────────────────────────

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Bersihkan dan standardisasi kolom."""
    if df.empty:
        return df

    # Rename kolom
    df = df.rename(columns={
        "fundingTime": "timestamp",
        "fundingRate": "funding_rate",
    })

    # Convert timestamp
    df["timestamp"]    = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")

    # Kolom final
    keep = ["timestamp", "funding_rate"]
    if "markPrice" in df.columns:
        df = df.rename(columns={"markPrice": "mark_price"})
        df["mark_price"] = pd.to_numeric(df["mark_price"], errors="coerce")
        keep.append("mark_price")

    df = df[keep].copy()
    df = df.drop_duplicates("timestamp")
    df = df.sort_values("timestamp").reset_index(drop=True)

    log.info("Processed: %d baris | %s → %s",
             len(df),
             df["timestamp"].iloc[0].strftime("%Y-%m-%d"),
             df["timestamp"].iloc[-1].strftime("%Y-%m-%d"))
    return df


# ── Save ──────────────────────────────────────────────────────────────────────

def save_data(df: pd.DataFrame, path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("Tersimpan → %s  (%d baris, %d kolom)", path, len(df), len(df.columns))
    log.info("Kolom: %s", list(df.columns))


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> pd.DataFrame:
    log.info("═" * 55)
    log.info("Derivatives Fetcher — Binance Futures Funding Rate")
    log.info("  Symbol    : %s", SYMBOL)
    log.info("  Start     : %s", START_DATE)
    log.info("  Endpoint  : %s", BINANCE_URL)
    log.info("═" * 55)

    raw = fetch_funding_history()
    df  = process_data(raw)

    if df.empty:
        log.error("Dataset kosong — tidak disimpan.")
        return df

    save_data(df)
    log.info("═" * 55)
    return df


if __name__ == "__main__":
    run()
