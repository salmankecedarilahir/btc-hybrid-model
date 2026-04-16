"""
hybrid_engine.py — Phase 4: Hybrid Signal Engine (Pendekatan C).

SHORT logic — Hybrid Cycle + Derivatives:
  Genuine SHORT hanya valid saat:
    (A) Bear Market Cycle:
        close < EMA_200daily (EMA1200 pada 4H)
        AND EMA_50daily (EMA300 4H) < EMA_200daily (EMA1200 4H)  ← death cross
    OR
    (B) Derivatives Extreme:
        funding_zscore < -2 (funding sangat negatif)
        AND oi_spike = 1    (open interest spike = forced liquidation incoming)

  PLUS regime == "DOWN" tetap required sebagai konfirmasi trend pendek.

  Tujuan:
    - SHORT hanya saat 2018/2022 genuine bear market
    - Hindari short di koreksi bull market (2019, 2021-mid, 2024)
    - Sistem bisa berkelanjutan untuk konfirmasi candle real-time

Input:  data/btc_full_hybrid_dataset.csv
Output: data/btc_trading_signals.csv
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

BASE_DIR    = Path(__file__).parent
INPUT_PATH  = BASE_DIR / "data" / "btc_full_hybrid_dataset.csv"
OUTPUT_PATH = BASE_DIR / "data" / "btc_trading_signals.csv"

# ── Volatility regime ─────────────────────────────────────────────────────────
ATR_LOW_MAX  = 30.0
ATR_HIGH_MIN = 70.0

# ── LONG thresholds (aggressive) ─────────────────────────────────────────────
LONG_THRESHOLD = {
    "HIGH": 2,
    "MID":  2,
    "LOW":  3,
}

# ── Bear Market Filter (4H approximations) ────────────────────────────────────
# 1 hari = 6 candle 4H
# EMA200 weekly ≈ EMA200 daily ≈ EMA(1200) pada 4H
# EMA50  weekly ≈ EMA50  daily ≈ EMA(300)  pada 4H
EMA_FAST_PERIOD = 300    # ≈ EMA50 daily
EMA_SLOW_PERIOD = 1200   # ≈ EMA200 daily

# ── Derivatives extreme thresholds ───────────────────────────────────────────
FUNDING_ZSCORE_SHORT = -2.0   # funding sangat negatif → squeeze incoming
OI_SPIKE_SHORT       = 1      # oi_spike = 1 (dari derivatives_engine)


# ── Load ──────────────────────────────────────────────────────────────────────

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {path}\n"
            "Jalankan derivatives_engine.py terlebih dahulu."
        )
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    log.info("Loaded : %d baris | %s → %s",
             len(df),
             df["timestamp"].iloc[0].strftime("%Y-%m-%d"),
             df["timestamp"].iloc[-1].strftime("%Y-%m-%d"))
    return df


# ── Hybrid Score ──────────────────────────────────────────────────────────────

def calc_hybrid_score(df: pd.DataFrame) -> pd.DataFrame:
    df["trend_score"]       = pd.to_numeric(df.get("trend_score", 0),       errors="coerce").fillna(0)
    df["derivatives_score"] = pd.to_numeric(df.get("derivatives_score", 0), errors="coerce").fillna(0)
    df["hybrid_score"]      = df["trend_score"] + df["derivatives_score"]
    log.info("hybrid_score — min:%.0f  max:%.0f  mean:%.3f",
             df["hybrid_score"].min(), df["hybrid_score"].max(), df["hybrid_score"].mean())
    return df


# ── Volatility Regime ─────────────────────────────────────────────────────────

def add_volatility_regime(df: pd.DataFrame) -> pd.DataFrame:
    df["atr_percentile"] = pd.to_numeric(df.get("atr_percentile", 50), errors="coerce").fillna(50)

    def _classify(x: float) -> str:
        if x < ATR_LOW_MAX:   return "LOW"
        if x > ATR_HIGH_MIN:  return "HIGH"
        return "MID"

    df["volatility_regime"] = df["atr_percentile"].apply(_classify)
    dist = df["volatility_regime"].value_counts()
    for vr in ["HIGH", "MID", "LOW"]:
        cnt = dist.get(vr, 0)
        log.info("vol_regime %-4s: %d candles (%.1f%%)", vr, cnt, cnt / len(df) * 100)
    return df


# ── Bear Market Filter ────────────────────────────────────────────────────────

def add_bear_market_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bear market konfirmasi via EMA death cross (no lookahead).

    bear_market = True jika:
      close < EMA_slow (close di bawah EMA200 daily)
      AND ema_fast < ema_slow (death cross: EMA50 < EMA200)

    Menggunakan 4H candles:
      EMA_fast = EMA(300)  ≈ EMA50  daily
      EMA_slow = EMA(1200) ≈ EMA200 daily

    Semua EMA dihitung dengan ewm(adjust=False) = no lookahead.
    """
    close = df["close"].astype(float)

    ema_fast = close.ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
    ema_slow = close.ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()

    df["ema_fast"] = ema_fast   # ≈ EMA50 daily
    df["ema_slow"] = ema_slow   # ≈ EMA200 daily

    # Death cross: ema_fast < ema_slow (trend turun jangka panjang)
    death_cross = ema_fast < ema_slow
    # Close di bawah EMA200 daily (bear territory)
    below_slow  = close < ema_slow

    df["bear_market"] = (death_cross & below_slow).astype(int)

    bear_count = df["bear_market"].sum()
    log.info("Bear market bars: %d (%.1f%%)", bear_count, bear_count / len(df) * 100)
    return df


# ── Derivatives Extreme ───────────────────────────────────────────────────────

def add_derivatives_extreme(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derivatives extreme SHORT condition:
      funding_zscore < -2 AND oi_spike = 1

    Menandakan: funding sangat negatif (short pain incoming)
    + open interest spike → potensi forced liquidation
    """
    # Kolom ini dihasilkan oleh derivatives_engine.py
    funding_z = pd.to_numeric(df.get("funding_zscore", 0),  errors="coerce").fillna(0)
    oi_spike  = pd.to_numeric(df.get("oi_spike",        0),  errors="coerce").fillna(0)

    df["deriv_extreme_short"] = (
        (funding_z <= FUNDING_ZSCORE_SHORT) & (oi_spike >= OI_SPIKE_SHORT)
    ).astype(int)

    de_count = df["deriv_extreme_short"].sum()
    log.info("Derivatives extreme SHORT bars: %d (%.1f%%)",
             de_count, de_count / len(df) * 100)
    return df


# ── Signal Generation (Vectorized, no lookahead) ──────────────────────────────

def add_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    LONG: regime=UP AND hybrid_score >= LONG_THRESHOLD[vol_regime]

    SHORT: regime=DOWN AND trend_score <= -1
           AND (bear_market == 1 OR deriv_extreme_short == 1)

    Dapat digunakan untuk konfirmasi candle real-time karena:
    - Semua indikator (EMA, ATR, derivatives zscore) bersifat causal
    - Tidak ada data masa depan yang digunakan
    """
    regime      = df["regime"]
    vol_reg     = df["volatility_regime"]
    score       = df["hybrid_score"]
    trend       = df["trend_score"]
    bear        = df["bear_market"]
    deriv_ext   = df["deriv_extreme_short"]

    long_thresh = vol_reg.map(LONG_THRESHOLD).fillna(3)

    # LONG: regime UP + hybrid score cukup
    long_cond  = (regime == "UP") & (score >= long_thresh)

    # SHORT: regime DOWN + trend negatif + (bear cycle ATAU derivatives extreme)
    short_base = (regime == "DOWN") & (trend <= -1)
    short_cond = short_base & ((bear == 1) | (deriv_ext == 1))

    df["signal"] = "NONE"
    df.loc[long_cond,  "signal"] = "LONG"
    df.loc[short_cond, "signal"] = "SHORT"

    return df


# ── Signal Strength ───────────────────────────────────────────────────────────

def add_signal_strength(df: pd.DataFrame) -> pd.DataFrame:
    signal      = df["signal"]
    score       = df["hybrid_score"]
    vol_reg     = df["volatility_regime"]
    bear        = df["bear_market"]
    deriv_ext   = df["deriv_extreme_short"]
    long_thresh = vol_reg.map(LONG_THRESHOLD).fillna(3)
    abs_score   = score.abs()

    strength = pd.Series("NONE", index=df.index)

    # LONG strength dari hybrid_score
    lm = signal == "LONG"
    strength[lm & (abs_score >= long_thresh + 2)] = "STRONG"
    strength[lm & (abs_score >= long_thresh + 1) & (abs_score < long_thresh + 2)] = "NORMAL"
    strength[lm & (abs_score < long_thresh + 1)]  = "WEAK"

    # SHORT strength:
    #   STRONG = bear_market + deriv_extreme (double confirmation)
    #   NORMAL = salah satu saja
    sm = signal == "SHORT"
    both = (bear == 1) & (deriv_ext == 1)
    strength[sm &  both] = "STRONG"
    strength[sm & ~both] = "NORMAL"

    df["signal_strength"] = strength
    return df


# ── Print Summary ─────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    div = "═" * 58
    sep = "─" * 58
    n   = len(df)

    long_c  = (df["signal"] == "LONG").sum()
    short_c = (df["signal"] == "SHORT").sum()
    none_c  = (df["signal"] == "NONE").sum()
    active  = long_c + short_c

    strong = (df["signal_strength"] == "STRONG").sum()
    normal = (df["signal_strength"] == "NORMAL").sum()
    weak   = (df["signal_strength"] == "WEAK").sum()

    bear_n   = df["bear_market"].sum()
    deriv_n  = df["deriv_extreme_short"].sum()

    print(f"\n{div}")
    print("  HYBRID SIGNAL SUMMARY (Pendekatan C — Cycle + Derivatives)")
    print(div)
    print(f"  {'Total candles':<28}: {n:,}")
    print(f"  {'Active signals':<28}: {active:,}  ({active/n*100:.1f}%)")
    print(sep)
    print(f"  {'LONG':<28}: {long_c:,} ({long_c/n*100:.1f}%)")
    print(f"  {'SHORT':<28}: {short_c:,} ({short_c/n*100:.1f}%)")
    print(f"  {'NONE':<28}: {none_c:,} ({none_c/n*100:.1f}%)")
    print(sep)
    print(f"  {'Bear market bars':<28}: {bear_n:,} ({bear_n/n*100:.1f}%)")
    print(f"  {'Derivatives extreme bars':<28}: {deriv_n:,} ({deriv_n/n*100:.1f}%)")
    print(sep)
    print(f"  {'STRONG':<28}: {strong:,} ({strong/n*100:.1f}%)")
    print(f"  {'NORMAL':<28}: {normal:,} ({normal/n*100:.1f}%)")
    print(f"  {'WEAK':<28}: {weak:,} ({weak/n*100:.1f}%)")
    print(div)
    print(f"\n  Volatility Regime × Signal:")
    print(sep)
    ct = pd.crosstab(df["volatility_regime"], df["signal"])
    print(ct.to_string())
    print()


# ── Save ──────────────────────────────────────────────────────────────────────

def save(df: pd.DataFrame) -> None:
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    log.info("Tersimpan → %s  (%d baris, %d kolom)", OUTPUT_PATH, len(df), len(df.columns))


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> pd.DataFrame:
    log.info("═" * 58)
    log.info("Hybrid Engine — Pendekatan C (Cycle + Derivatives)")
    log.info("  LONG  : regime=UP AND hybrid_score >= %d/%d/%d (HIGH/MID/LOW)",
             LONG_THRESHOLD["HIGH"], LONG_THRESHOLD["MID"], LONG_THRESHOLD["LOW"])
    log.info("  SHORT : regime=DOWN AND trend<=-1")
    log.info("          AND (bear_market OR deriv_extreme_short)")
    log.info("  Bear  : close < EMA%d AND EMA%d < EMA%d (death cross)",
             EMA_SLOW_PERIOD, EMA_FAST_PERIOD, EMA_SLOW_PERIOD)
    log.info("  Deriv : funding_zscore <= %.1f AND oi_spike >= %d",
             FUNDING_ZSCORE_SHORT, OI_SPIKE_SHORT)
    log.info("═" * 58)

    df = load_data(INPUT_PATH)
    df = calc_hybrid_score(df)
    df = add_volatility_regime(df)
    df = add_bear_market_filter(df)
    df = add_derivatives_extreme(df)
    df = add_signals(df)
    df = add_signal_strength(df)

    print_summary(df)
    save(df)

    log.info("═" * 58)
    return df


if __name__ == "__main__":
    run()
