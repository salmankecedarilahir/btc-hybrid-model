"""
derivatives_engine.py — Phase 3: Statistical Derivatives Engine.
Input:  data/btc_4h_with_regime.csv
        data/btc_derivatives_raw.csv
Output: data/btc_full_hybrid_dataset.csv
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR      = Path(__file__).parent
REGIME_PATH   = BASE_DIR / "data" / "btc_4h_with_regime.csv"
DERIV_PATH    = BASE_DIR / "data" / "btc_derivatives_raw.csv"
FUNDING_PATH  = BASE_DIR / "data" / "funding_4h.csv"
OUTPUT_PATH   = BASE_DIR / "data" / "btc_full_hybrid_dataset.csv"

ROLLING_WIN   = 30   # window untuk rolling zscore
ZSCORE_THRESH = 2.0  # threshold zscore untuk extreme


# ── Load ──────────────────────────────────────────────────────────────────────

def _load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        log.warning("[%s] File tidak ditemukan: %s", label, path)
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    log.info("%-20s: %d baris | %s → %s",
             label,
             len(df),
             df["timestamp"].iloc[0].strftime("%Y-%m-%d"),
             df["timestamp"].iloc[-1].strftime("%Y-%m-%d"))
    return df


def load_all() -> tuple:
    regime  = _load_csv(REGIME_PATH,  "regime")
    deriv   = _load_csv(DERIV_PATH,   "derivatives_raw")
    funding = _load_csv(FUNDING_PATH, "funding_4h (Bitget)")
    return regime, deriv, funding


# ── Merge ─────────────────────────────────────────────────────────────────────

def merge_data(regime: pd.DataFrame,
               deriv: pd.DataFrame,
               funding: pd.DataFrame) -> pd.DataFrame:
    df = regime.copy()

    # Merge Binance derivatives (funding_rate, mark_price, open_interest)
    if not deriv.empty:
        cols = ["timestamp"] + [c for c in deriv.columns if c != "timestamp"]
        df = pd.merge_asof(
            df.sort_values("timestamp"),
            deriv[cols].sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta("8h"),
        )
        covered = df["funding_rate"].notna().sum()
        log.info("funding_rate  : %d / %d candles covered", covered, len(df))

    # Merge Bitget funding sebagai fallback jika Binance tidak cover
    if not funding.empty and "funding_rate" in funding.columns:
        fund_col = "funding_rate"
        funding_renamed = funding[["timestamp", fund_col]].rename(
            columns={fund_col: "funding_rate_bitget"}
        )
        df = pd.merge_asof(
            df.sort_values("timestamp"),
            funding_renamed.sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta("4h"),
        )
        # Gunakan Bitget jika Binance NaN
        if "funding_rate_bitget" in df.columns:
            df["funding_rate"] = df["funding_rate"].fillna(df["funding_rate_bitget"])
            df = df.drop(columns=["funding_rate_bitget"])

    log.info("Setelah merge: %d baris, %d kolom", len(df), len(df.columns))
    return df


# ── Z-Score Calculations ──────────────────────────────────────────────────────

def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score: (x - rolling_mean) / rolling_std."""
    mean = series.rolling(window, min_periods=max(1, window // 2)).mean()
    std  = series.rolling(window, min_periods=max(1, window // 2)).std()
    std  = std.replace(0, np.nan)
    return (series - mean) / std


def calc_derivatives_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Funding Rate Z-Score ──────────────────────────────────────────────────
    if "funding_rate" in df.columns:
        df["funding_rate"]   = pd.to_numeric(df["funding_rate"], errors="coerce")
        df["funding_zscore"] = rolling_zscore(df["funding_rate"], ROLLING_WIN)
    else:
        df["funding_rate"]   = np.nan
        df["funding_zscore"] = np.nan

    # ── Open Interest Z-Score ─────────────────────────────────────────────────
    oi_col = None
    for candidate in ["open_interest", "mark_price"]:
        if candidate in df.columns:
            oi_col = candidate
            break

    if oi_col:
        df[oi_col]      = pd.to_numeric(df[oi_col], errors="coerce")
        df["oi_zscore"] = rolling_zscore(df[oi_col], ROLLING_WIN)
    else:
        df["oi_zscore"] = np.nan

    # ── Signal Flags ──────────────────────────────────────────────────────────
    fz = df["funding_zscore"]
    oz = df["oi_zscore"]

    df["funding_extreme"] = (fz.abs() > ZSCORE_THRESH).astype(int)
    df["oi_spike"]        = (oz > ZSCORE_THRESH).astype(int)

    df["potential_squeeze"] = (
        ((fz >  ZSCORE_THRESH) & (oz > 1.0)) |
        ((fz < -ZSCORE_THRESH) & (oz > 1.0))
    ).astype(int)

    # ── Derivatives Score ─────────────────────────────────────────────────────
    df["derivatives_score"] = (
        df["funding_extreme"] * 1 +
        df["oi_spike"]        * 1 +
        df["potential_squeeze"] * 2
    ).fillna(0).astype(int)

    # NaN safety
    for col in ["funding_zscore", "oi_zscore", "funding_extreme",
                "oi_spike", "potential_squeeze"]:
        df[col] = df[col].fillna(0)

    log.info("derivatives_score distribution:")
    log.info("\n%s", df["derivatives_score"].value_counts().sort_index().to_string())
    log.info("funding_extreme  : %d candles (%.1f%%)",
             df["funding_extreme"].sum(), df["funding_extreme"].mean() * 100)
    log.info("oi_spike         : %d candles (%.1f%%)",
             df["oi_spike"].sum(), df["oi_spike"].mean() * 100)
    log.info("potential_squeeze: %d candles (%.1f%%)",
             df["potential_squeeze"].sum(), df["potential_squeeze"].mean() * 100)

    return df


# ── Save ──────────────────────────────────────────────────────────────────────

def save(df: pd.DataFrame) -> None:
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    log.info("Tersimpan → %s  (%d baris, %d kolom)", OUTPUT_PATH, len(df), len(df.columns))
    log.info("Kolom: %s", list(df.columns))


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> pd.DataFrame:
    log.info("═" * 55)
    log.info("Derivatives Engine — Statistical Z-Score")
    log.info("  Rolling window : %d candles", ROLLING_WIN)
    log.info("  Z-Score thresh : %.1f", ZSCORE_THRESH)
    log.info("═" * 55)

    regime, deriv, funding = load_all()

    if regime.empty:
        raise FileNotFoundError(f"Regime file tidak ditemukan: {REGIME_PATH}")

    df = merge_data(regime, deriv, funding)
    df = calc_derivatives_signals(df)
    save(df)

    log.info("═" * 55)
    return df


if __name__ == "__main__":
    run()
