"""
audit_short_signal.py — Audit: Mengapa SHORT bars = 0?
Input:  data/btc_trading_signals.csv
        data/btc_full_hybrid_dataset.csv
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR      = Path(__file__).parent
SIGNALS_PATH  = BASE_DIR / "data" / "btc_trading_signals.csv"
HYBRID_PATH   = BASE_DIR / "data" / "btc_full_hybrid_dataset.csv"

SCORE_THRESHOLD = {"HIGH": 3, "MID": 4, "LOW": 5}


# ── Load ──────────────────────────────────────────────────────────────────────

def load(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"[{label}] File tidak ditemukan: {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    log.info("[%s] Loaded %d baris | %s → %s",
             label, len(df),
             df["timestamp"].iloc[0].strftime("%Y-%m-%d"),
             df["timestamp"].iloc[-1].strftime("%Y-%m-%d"))
    return df


# ── Utils ─────────────────────────────────────────────────────────────────────

def div(char="═", n=60): return char * n
def sep(char="─", n=60): return char * n


# ── 1. Regime Distribution ────────────────────────────────────────────────────

def audit_regime_distribution(df: pd.DataFrame) -> None:
    print(f"\n{div()}")
    print("  AUDIT 1 — Distribusi Regime")
    print(div())
    vc  = df["regime"].value_counts()
    tot = len(df)
    for regime, cnt in vc.items():
        print(f"  {regime:<15}: {cnt:>8,}  ({cnt/tot*100:>5.1f}%)")
    print(sep())
    print(f"  {'TOTAL':<15}: {tot:>8,}")


# ── 2. hybrid_score saat DOWN ─────────────────────────────────────────────────

def audit_hybrid_score_when_down(df: pd.DataFrame) -> None:
    print(f"\n{div()}")
    print("  AUDIT 2 — Distribusi hybrid_score saat regime == DOWN")
    print(div())

    down = df[df["regime"] == "DOWN"]
    if down.empty:
        print("  [WARN] Tidak ada bar dengan regime == DOWN!")
        return

    print(f"  Total DOWN bars: {len(down):,}")
    print(sep())

    score_dist = down["hybrid_score"].value_counts().sort_index()
    print(f"  {'Score':<10} {'Count':>8} {'Pct':>8}")
    print(f"  {'─'*10} {'─'*8} {'─'*8}")
    for score, cnt in score_dist.items():
        print(f"  {score:<10} {cnt:>8,}  ({cnt/len(down)*100:>5.1f}%)")

    print(sep())
    print(f"  hybrid_score stats:")
    print(f"    min  = {down['hybrid_score'].min():.0f}")
    print(f"    max  = {down['hybrid_score'].max():.0f}")
    print(f"    mean = {down['hybrid_score'].mean():.3f}")
    print(f"    std  = {down['hybrid_score'].std():.3f}")


# ── 3. DOWN bars yang memenuhi threshold ──────────────────────────────────────

def audit_down_threshold(df: pd.DataFrame) -> None:
    print(f"\n{div()}")
    print("  AUDIT 3 — DOWN bars memenuhi threshold per volatility_regime")
    print(div())

    down = df[df["regime"] == "DOWN"].copy()
    if down.empty:
        print("  [WARN] Tidak ada bar dengan regime == DOWN!")
        return

    if "volatility_regime" not in down.columns:
        if "atr_percentile" in down.columns:
            def _vr(x):
                if x < 30:   return "LOW"
                elif x > 70: return "HIGH"
                else:         return "MID"
            down["volatility_regime"] = down["atr_percentile"].apply(_vr)
        else:
            down["volatility_regime"] = "MID"
            print("  [WARN] volatility_regime tidak ditemukan — diasumsikan MID")

    print(f"  {'Vol Regime':<12} {'DOWN bars':>10} {'Threshold':>10} "
          f"{'Meet thresh':>12} {'Pct meet':>10}")
    print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*12} {'─'*10}")

    for vr in ["HIGH", "MID", "LOW"]:
        sub    = down[down["volatility_regime"] == vr]
        thresh = SCORE_THRESHOLD[vr]
        meet   = (sub["hybrid_score"] >= thresh).sum()
        pct    = meet / len(sub) * 100 if len(sub) > 0 else 0.0
        print(f"  {vr:<12} {len(sub):>10,} {thresh:>10} {meet:>12,} {pct:>9.1f}%")

    print(sep())
    # Total
    total_meet = 0
    for vr, thresh in SCORE_THRESHOLD.items():
        sub  = down[down["volatility_regime"] == vr]
        total_meet += (sub["hybrid_score"] >= thresh).sum()
    print(f"  Total DOWN bars yang seharusnya SHORT: {total_meet:,}")


# ── 4. Crosstab regime × signal ───────────────────────────────────────────────

def audit_crosstab(df: pd.DataFrame) -> None:
    print(f"\n{div()}")
    print("  AUDIT 4 — Crosstab regime × signal")
    print(div())

    if "signal" not in df.columns:
        print("  [WARN] Kolom 'signal' tidak ditemukan di dataset ini.")
        return

    ct = pd.crosstab(df["regime"], df["signal"], margins=True, margins_name="TOTAL")
    print(ct.to_string())


# ── 5. Deep Dive: trend_score saat DOWN ───────────────────────────────────────

def audit_trend_score_down(df: pd.DataFrame) -> None:
    print(f"\n{div()}")
    print("  AUDIT 5 — trend_score saat regime == DOWN (root cause)")
    print(div())

    down = df[df["regime"] == "DOWN"]
    if down.empty:
        print("  [WARN] Tidak ada bar dengan regime == DOWN!")
        return

    for col in ["trend_score", "derivatives_score", "hybrid_score"]:
        if col in down.columns:
            s = down[col]
            print(f"  {col}:")
            print(f"    min={s.min():.0f}  max={s.max():.0f}  "
                  f"mean={s.mean():.3f}  std={s.std():.3f}")
            vc = s.value_counts().sort_index()
            for val, cnt in vc.items():
                bar = "█" * min(int(cnt / len(down) * 40), 40)
                print(f"    {val:>4.0f} : {cnt:>6,} ({cnt/len(down)*100:>5.1f}%)  {bar}")
        else:
            print(f"  [WARN] Kolom '{col}' tidak ditemukan.")
        print()


# ── 6. Signal generation trace ────────────────────────────────────────────────

def audit_signal_logic(df: pd.DataFrame) -> None:
    print(f"\n{div()}")
    print("  AUDIT 6 — Signal generation trace (10 sample DOWN bars)")
    print(div())

    down = df[df["regime"] == "DOWN"].copy()
    if down.empty:
        print("  [WARN] Tidak ada bar dengan regime == DOWN!")
        return

    if "volatility_regime" not in down.columns:
        if "atr_percentile" in down.columns:
            def _vr(x):
                if x < 30:   return "LOW"
                elif x > 70: return "HIGH"
                else:         return "MID"
            down["volatility_regime"] = down["atr_percentile"].apply(_vr)
        else:
            down["volatility_regime"] = "MID"

    cols = ["timestamp", "regime", "volatility_regime",
            "trend_score", "derivatives_score", "hybrid_score"]
    if "signal" in down.columns:
        cols.append("signal")

    avail = [c for c in cols if c in down.columns]
    sample = down[avail].tail(10)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(sample.to_string(index=False))

    print(sep())
    print("  Threshold per volatility_regime:")
    for vr, thresh in SCORE_THRESHOLD.items():
        print(f"    {vr:<6}: hybrid_score >= {thresh}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> None:
    print(f"\n{'═'*60}")
    print("  SHORT SIGNAL AUDIT — BTC Hybrid Model")
    print(f"{'═'*60}")

    # Load signals (has signal column)
    sig_df = load(SIGNALS_PATH, "signals")

    # Load hybrid dataset (has trend_score, derivatives_score)
    hyb_df = load(HYBRID_PATH,  "hybrid")

    # Merge signal, volatility_regime, hybrid_score dari signals file ke hybrid df
    for col in ["signal", "volatility_regime", "hybrid_score"]:
        if col in sig_df.columns and col not in hyb_df.columns:
            hyb_df = hyb_df.merge(
                sig_df[["timestamp", col]],
                on="timestamp", how="left"
            )

    # Fallback: hitung hybrid_score jika masih tidak ada
    if "hybrid_score" not in hyb_df.columns:
        t = pd.to_numeric(hyb_df.get("trend_score", pd.Series(0.0, index=hyb_df.index)), errors="coerce").fillna(0)
        d = pd.to_numeric(hyb_df.get("derivatives_score", pd.Series(0.0, index=hyb_df.index)), errors="coerce").fillna(0)
        hyb_df["hybrid_score"] = t + d
        log.info("hybrid_score dihitung dari trend_score + derivatives_score.")

    print()

    # Run all audits on hybrid dataset (has all score columns)
    audit_regime_distribution(hyb_df)
    audit_hybrid_score_when_down(hyb_df)
    audit_down_threshold(hyb_df)
    audit_crosstab(hyb_df)
    audit_trend_score_down(hyb_df)
    audit_signal_logic(hyb_df)

    print(f"\n{'═'*60}")
    print("  AUDIT SELESAI")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    run()
