"""
ai_dataset_builder.py — BTC Hybrid Model: AI-Ready Dataset Builder
===================================================================

BAGIAN 9 — DATASET SIAP AI

Mempersiapkan dataset bersih untuk training AI model dari:
  1. Market features (OHLCV + technical indicators)
  2. Trade signals (LONG / SHORT / NONE)
  3. Trade outcomes (return, win/loss, holding period)
  4. Market regime (UP / DOWN / SIDEWAYS / NEUTRAL)
  5. Risk state (tier0 / tier1 / tier2 paused)

Output:
  data/ai_training_dataset.csv   — dataset lengkap
  data/ai_feature_summary.csv    — statistik tiap feature

Cara pakai:
    python ai_dataset_builder.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

BASE         = Path(__file__).parent / "data"
BARS_PER_YEAR = 2190


# ════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════

def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tambah market features untuk AI training.
    Semua features harus NO LOOKAHEAD (dihitung dari data yang sudah tersedia).
    """
    out = df.copy()

    # ── Returns ─────────────────────────────────────────────────────
    out["ret_1bar"]  = out["close"].pct_change(1)
    out["ret_4bar"]  = out["close"].pct_change(4)
    out["ret_12bar"] = out["close"].pct_change(12)
    out["ret_24bar"] = out["close"].pct_change(24)

    # ── Volatility (realized vol) ───────────────────────────────────
    log_ret = np.log(out["close"] / out["close"].shift(1))
    out["vol_24bar"]  = log_ret.rolling(24).std()   * np.sqrt(BARS_PER_YEAR)
    out["vol_72bar"]  = log_ret.rolling(72).std()   * np.sqrt(BARS_PER_YEAR)
    out["vol_126bar"] = log_ret.rolling(126).std()  * np.sqrt(BARS_PER_YEAR)

    # Vol ratio: short/long vol (regime change detector)
    out["vol_ratio"]  = out["vol_24bar"] / (out["vol_72bar"] + 1e-8)

    # ── Trend indicators ────────────────────────────────────────────
    if "ema_20" not in out.columns:
        out["ema_20"]  = out["close"].ewm(span=20,  adjust=False).mean()
    if "ema_50" not in out.columns:
        out["ema_50"]  = out["close"].ewm(span=50,  adjust=False).mean()
    out["ema_200"] = out["close"].ewm(span=200, adjust=False).mean()

    # Price relative to EMA (normalized)
    out["price_vs_ema20"]  = (out["close"] - out["ema_20"])  / (out["ema_20"]  + 1e-8)
    out["price_vs_ema50"]  = (out["close"] - out["ema_50"])  / (out["ema_50"]  + 1e-8)
    out["price_vs_ema200"] = (out["close"] - out["ema_200"]) / (out["ema_200"] + 1e-8)
    out["ema20_vs_ema50"]  = (out["ema_20"] - out["ema_50"]) / (out["ema_50"]  + 1e-8)

    # ── Momentum ────────────────────────────────────────────────────
    out["momentum_24"] = out["close"].pct_change(24)
    out["momentum_72"] = out["close"].pct_change(72)

    # ── ATR (Average True Range) ────────────────────────────────────
    if "atr_14" not in out.columns and all(c in out.columns for c in ["high", "low"]):
        tr = pd.concat([
            out["high"] - out["low"],
            (out["high"] - out["close"].shift(1)).abs(),
            (out["low"]  - out["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        out["atr_14"] = tr.rolling(14).mean()
        out["atr_pct"] = out["atr_14"] / (out["close"] + 1e-8)

    # ── Volume features (jika tersedia) ─────────────────────────────
    if "volume" in out.columns:
        out["volume_ma24"] = out["volume"].rolling(24).mean()
        out["volume_ratio"] = out["volume"] / (out["volume_ma24"] + 1e-8)

    return out


def add_regime_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode market regime dari kolom 'regime' ke numerik.
    Regime dari market_regime_engine: UP / DOWN / SIDEWAYS / NEUTRAL
    """
    out = df.copy()

    if "regime" in out.columns:
        regime_map = {"UP": 1, "DOWN": -1, "SIDEWAYS": 0, "NEUTRAL": 0}
        out["regime_encoded"] = out["regime"].map(regime_map).fillna(0)

        # One-hot encoding juga (lebih baik untuk AI)
        out["regime_up"]       = (out["regime"] == "UP").astype(int)
        out["regime_down"]     = (out["regime"] == "DOWN").astype(int)
        out["regime_sideways"] = (out["regime"] == "SIDEWAYS").astype(int)

    if "trend_score" in out.columns:
        # Normalize trend score ke [-1, 1]
        ts = out["trend_score"]
        out["trend_score_norm"] = ts / (ts.abs().max() + 1e-8)

    return out


def add_signal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode signal dan buat target variable untuk AI.
    """
    out = df.copy()

    # Signal encoding
    if "signal" in out.columns:
        signal_map = {"LONG": 1, "SHORT": -1, "NONE": 0}
        out["signal_encoded"] = out["signal"].map(signal_map).fillna(0)

    # Position encoding (forward-filled dari signal)
    if "position" in out.columns:
        out["position_norm"] = out["position"].astype(float)

    # ── Target variable untuk supervised learning ────────────────────
    # Target: return N bar ke depan (future return)
    # Ini adalah LABEL yang ingin diprediksi AI
    # PENTING: hanya dipakai saat training, BUKAN saat inference live
    #
    # FIX D3: market_return mungkin tidak ada di signals_df
    # Fallback: hitung dari close price pct_change (lebih robust)
    if "market_return" in out.columns:
        mr = out["market_return"]
    elif "close" in out.columns:
        mr = out["close"].pct_change()
    else:
        mr = None

    if mr is not None:
        out["target_ret_1bar"]  = mr.shift(-1)                           # 1 bar ke depan
        out["target_ret_4bar"]  = out["close"].pct_change(4).shift(-4)  if "close" in out.columns else mr.rolling(4).sum().shift(-4)
        out["target_ret_12bar"] = out["close"].pct_change(12).shift(-12) if "close" in out.columns else mr.rolling(12).sum().shift(-12)
        out["target_direction_1bar"] = np.sign(out["target_ret_1bar"])
        out["target_profitable_trade"] = (
            out["signal_encoded"] * out["target_ret_1bar"] > 0
        ).astype(int) if "signal_encoded" in out.columns else np.nan

    return out


def add_risk_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tambah risk state features dari kill switch.
    """
    out = df.copy()

    if "kill_switch_active" in out.columns:
        out["risk_state"] = out["kill_switch_active"].astype(int)
        # Berapa bar sudah dalam keadaan kill switch
        ks = out["kill_switch_active"].astype(int)
        out["ks_duration"] = ks.groupby((ks != ks.shift()).cumsum()).cumcount()

    return out


# ════════════════════════════════════════════════════════════════════
#  DATASET BUILDER
# ════════════════════════════════════════════════════════════════════

def build_ai_dataset(signals_df: pd.DataFrame,
                     risk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge dan build final AI-ready dataset.

    Steps:
      1. Merge signals dengan risk-managed results
      2. Tambah market features
      3. Tambah regime encoding
      4. Tambah signal features + target
      5. Tambah risk state
      6. Clean NaN (dari rolling windows warmup)
      7. Normalize features
    """
    log.info("Building AI dataset ...")

    # ── Step 1: Merge ────────────────────────────────────────────────
    # Key: timestamp
    df = pd.merge(
        signals_df,
        risk_df[["timestamp", "equity", "drawdown", "equity_return",
                  "leverage_used", "kill_switch_active", "shadow_equity",
                  "running_max_equity"]],
        on="timestamp", how="left", suffixes=("", "_risk")
    )

    # ── Step 2-5: Add features ───────────────────────────────────────
    df = add_market_features(df)
    df = add_regime_encoding(df)
    df = add_signal_features(df)
    df = add_risk_state(df)

    # ── Step 6: Drop warmup period (rolling windows butuh min 200 bars) ──
    n_before = len(df)
    df = df.dropna(subset=["ema_200", "vol_126bar"])
    n_dropped = n_before - len(df)
    log.info("Dropped %d warmup bars (%.1f%%)", n_dropped, n_dropped/n_before*100)

    # ── Step 7: Report ───────────────────────────────────────────────
    log.info("Final dataset: %d bars | %d features",
             len(df), len(df.columns))

    return df.reset_index(drop=True)


def generate_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Statistik tiap feature untuk quality check.
    Cek: NaN%, range, distribusi — sebelum training AI.
    """
    numeric = df.select_dtypes(include=[np.number])
    summary = []
    for col in numeric.columns:
        s = numeric[col]
        summary.append({
            "feature":   col,
            "dtype":     str(s.dtype),
            "n_valid":   s.notna().sum(),
            "nan_pct":   round(s.isna().mean() * 100, 2),
            "mean":      round(float(s.mean()), 6),
            "std":       round(float(s.std()),  6),
            "min":       round(float(s.min()),  6),
            "p25":       round(float(s.quantile(0.25)), 6),
            "median":    round(float(s.median()), 6),
            "p75":       round(float(s.quantile(0.75)), 6),
            "max":       round(float(s.max()),  6),
            "skew":      round(float(s.skew()), 4),
        })
    return pd.DataFrame(summary)


def check_dataset_readiness(df: pd.DataFrame) -> bool:
    """
    Verifikasi dataset siap untuk AI training.
    """
    DIV = "═" * 65
    print(f"\n{DIV}")
    print("  AI DATASET READINESS CHECK")
    print(DIV)

    issues = []

    # Cek jumlah data
    if len(df) < 5000:
        issues.append(f"Dataset terlalu kecil: {len(df)} bars (min 5000)")
    else:
        print(f"  [OK] Dataset size: {len(df):,} bars")

    # Cek target variable
    target_cols = [c for c in df.columns if c.startswith("target_")]
    if not target_cols:
        issues.append("Tidak ada target variable (target_ret_* / target_direction_*)")
    else:
        print(f"  [OK] Target variables: {target_cols}")

    # Cek NaN di key features
    key_features = ["signal_encoded", "regime_encoded", "vol_24bar",
                    "momentum_24", "price_vs_ema20"]
    for f in key_features:
        if f in df.columns:
            nan_pct = df[f].isna().mean() * 100
            if nan_pct > 5:
                issues.append(f"Feature {f} punya {nan_pct:.1f}% NaN")
            else:
                print(f"  [OK] Feature {f}: {nan_pct:.1f}% NaN")

    # Cek class balance (signal distribution)
    if "signal_encoded" in df.columns:
        dist = df["signal_encoded"].value_counts(normalize=True) * 100
        print(f"\n  Signal distribution:")
        for val, pct in dist.items():
            label = {1: "LONG", -1: "SHORT", 0: "NONE"}.get(val, str(val))
            print(f"    {label:>6}: {pct:.1f}%")
        if dist.get(0, 0) > 80:
            issues.append("NONE sangat dominan (>80%) — class imbalance untuk AI")

    # Cek regime distribution
    if "regime" in df.columns:
        rdist = df["regime"].value_counts(normalize=True) * 100
        print(f"\n  Regime distribution:")
        for r, p in rdist.items():
            print(f"    {r:>10}: {p:.1f}%")

    print(f"\n  Issues found: {len(issues)}")
    for issue in issues:
        print(f"  ❌ {issue}")

    ready = len(issues) == 0
    if ready:
        print(f"\n  [OK] Dataset siap untuk AI training")
    else:
        print(f"\n  [WARN]️  Selesaikan {len(issues)} issue sebelum training")

    print(DIV)
    return ready


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    signals_path = BASE / "btc_trading_signals.csv"
    risk_path    = BASE / "btc_risk_managed_results.csv"

    for p in [signals_path, risk_path]:
        if not p.exists():
            print(f"❌ File tidak ditemukan: {p}")
            raise SystemExit(1)

    signals_df = pd.read_csv(signals_path, parse_dates=["timestamp"])
    risk_df    = pd.read_csv(risk_path,    parse_dates=["timestamp"])

    for df in [signals_df, risk_df]:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Build
    ai_df = build_ai_dataset(signals_df, risk_df)

    # Save
    out_path = BASE / "ai_training_dataset.csv"
    ai_df.to_csv(out_path, index=False)
    log.info("✓ AI dataset saved: %s (%d rows, %d cols)", out_path, len(ai_df), len(ai_df.columns))

    # Feature summary
    summary = generate_feature_summary(ai_df)
    summary_path = BASE / "ai_feature_summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info("✓ Feature summary saved: %s", summary_path)

    # Readiness check
    check_dataset_readiness(ai_df)

    # Print top features
    print(f"\n  Feature list ({len(ai_df.columns)} total):")
    for col in sorted(ai_df.columns):
        print(f"    {col}")
