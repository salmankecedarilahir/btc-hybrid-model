"""
regime_engine.py — Market Regime Classification Engine.

Logic Regime (priority order):
  1. ATR percentile < 30  → SIDEWAYS  (volatilitas rendah, abaikan trend)
  2. ema20 > ema50 AND close > ema50  → UP      (uptrend confirmed)
  3. ema20 < ema50 AND close < ema50  → DOWN    (downtrend confirmed)
  4. else                             → NEUTRAL  (transisi / tidak jelas)

Trend Score:
  UP      → +2
  DOWN    → -2
  SIDEWAYS / NEUTRAL → 0

Tidak ada lookahead bias:
  - EMA dihitung dengan ewm causal (hanya lihat past data)
  - ATR dihitung dengan Wilder smoothing (hanya lihat past data)
  - ATR percentile dihitung dengan rolling window (hanya lihat past data)
  - shift() hanya dipakai untuk prev_close di True Range (benar secara definisi)

Jalankan: python regime_engine.py
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

from indicators import calculate_ema, calculate_atr, calculate_atr_percentile

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
INPUT_PATH = BASE_DIR / "data" / "btc_4h_ohlcv.csv"
OUTPUT_PATH = BASE_DIR / "data" / "btc_4h_with_regime.csv"

# ── Hyper-parameters ──────────────────────────────────────────────────────────
EMA_FAST          = 20
EMA_SLOW          = 50
ATR_PERIOD        = 14
ATR_PCT_WINDOW    = 100
SIDEWAYS_THRESHOLD = 30.0    # ATR percentile di bawah ini → SIDEWAYS


# ── Regime Classification ─────────────────────────────────────────────────────

def classify_regime(row: pd.Series) -> tuple[str, int]:
    """
    Klasifikasi regime untuk satu baris.

    Returns (regime_label, trend_score)
    """
    atr_pct = row["atr_percentile"]
    ema20   = row[f"ema_{EMA_FAST}"]
    ema50   = row[f"ema_{EMA_SLOW}"]
    close   = row["close"]

    # Jika salah satu indikator belum tersedia (warmup period)
    if pd.isna(atr_pct) or pd.isna(ema20) or pd.isna(ema50):
        return "NEUTRAL", 0

    # Priority 1: Low volatility → SIDEWAYS (trend signal tidak reliable)
    if atr_pct < SIDEWAYS_THRESHOLD:
        return "SIDEWAYS", 0

    # Priority 2: Uptrend
    if ema20 > ema50 and close > ema50:
        return "UP", 2

    # Priority 3: Downtrend
    if ema20 < ema50 and close < ema50:
        return "DOWN", -2

    # Priority 4: Transition / unclear
    return "NEUTRAL", 0


# ── Main Engine ───────────────────────────────────────────────────────────────

def run_regime_engine() -> pd.DataFrame:
    """
    Load OHLCV → hitung indikator → klasifikasi regime → simpan CSV.
    """
    log.info("═" * 60)
    log.info("BTC Hybrid Model — Phase 2: Market Regime Engine")
    log.info("═" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    if not INPUT_PATH.exists():
        log.error("File tidak ditemukan: %s", INPUT_PATH)
        log.error("Jalankan data_fetcher.py (Phase 1) terlebih dahulu.")
        raise FileNotFoundError(INPUT_PATH)

    df = pd.read_csv(INPUT_PATH, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    log.info("Data loaded: %d baris | %s → %s",
             len(df),
             df["timestamp"].iloc[0].strftime("%Y-%m-%d"),
             df["timestamp"].iloc[-1].strftime("%Y-%m-%d"))

    # ── Hitung indikator ──────────────────────────────────────────────────────
    log.info("Menghitung EMA%d …", EMA_FAST)
    df[f"ema_{EMA_FAST}"] = calculate_ema(df, EMA_FAST)

    log.info("Menghitung EMA%d …", EMA_SLOW)
    df[f"ema_{EMA_SLOW}"] = calculate_ema(df, EMA_SLOW)

    log.info("Menghitung ATR(%d) …", ATR_PERIOD)
    df[f"atr_{ATR_PERIOD}"] = calculate_atr(df, ATR_PERIOD)

    log.info("Menghitung ATR Percentile (window=%d) — no lookahead …", ATR_PCT_WINDOW)
    df["atr_percentile"] = calculate_atr_percentile(
        df[f"atr_{ATR_PERIOD}"], window=ATR_PCT_WINDOW
    )

    # ── Klasifikasi regime ────────────────────────────────────────────────────
    log.info("Mengklasifikasikan regime …")
    results = df.apply(classify_regime, axis=1)
    df["regime"]      = results.apply(lambda x: x[0])
    df["trend_score"] = results.apply(lambda x: x[1])

    # ── Warmup info ───────────────────────────────────────────────────────────
    warmup_rows = df["atr_percentile"].isna().sum()
    valid_rows  = len(df) - warmup_rows
    log.info(
        "Warmup period: %d candles (EMA%d + ATR%d + window%d) — %d candles tersedia untuk regime",
        warmup_rows, EMA_SLOW, ATR_PERIOD, ATR_PCT_WINDOW, valid_rows
    )

    # ── Distribusi regime ─────────────────────────────────────────────────────
    regime_counts = df["regime"].value_counts()
    log.info("Distribusi Regime:")
    for regime, count in regime_counts.items():
        pct = count / len(df) * 100
        log.info("  %-10s : %4d candles (%5.1f%%)", regime, count, pct)

    # ── Simpan ke CSV ─────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_PATH, index=False)
    log.info("Hasil tersimpan → %s  (%d baris, %d kolom)",
             OUTPUT_PATH, len(df), len(df.columns))
    log.info("Kolom: %s", list(df.columns))
    log.info("═" * 60)

    return df


if __name__ == "__main__":
    run_regime_engine()
