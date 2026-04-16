"""
trade_distribution_analysis.py — Trade Return Distribution Analysis.
Input:  data/btc_risk_managed_results.csv
Output: data/trade_distribution_report.json
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).parent
INPUT_PATH  = BASE_DIR / "data" / "btc_risk_managed_results.csv"
OUTPUT_PATH = BASE_DIR / "data" / "trade_distribution_report.json"


# ── Load ──────────────────────────────────────────────────────────────────────

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    log.info("Loaded : %d baris | %s → %s",
             len(df),
             df["timestamp"].iloc[0].strftime("%Y-%m-%d"),
             df["timestamp"].iloc[-1].strftime("%Y-%m-%d"))
    return df


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyze_distribution(df: pd.DataFrame) -> dict:
    ret    = df["risk_adjusted_return"].fillna(0.0)
    trades = ret[ret != 0].reset_index(drop=True)

    log.info("Total active trade bars: %d", len(trades))

    mean     = float(trades.mean())
    median   = float(trades.median())
    std      = float(trades.std())
    skewness = float(scipy_stats.skew(trades))
    kurtosis = float(scipy_stats.kurtosis(trades))   # excess kurtosis

    pct_5    = float(np.percentile(trades, 5))
    pct_95   = float(np.percentile(trades, 95))

    # Best & worst 5 trades — gabungkan dengan timestamp dan signal
    trade_df = df[df["risk_adjusted_return"] != 0][
        ["timestamp", "close", "signal", "risk_adjusted_return"]
    ].copy()
    trade_df["timestamp"] = trade_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M UTC")

    def _to_records(sub: pd.DataFrame) -> list:
        rows = []
        for _, r in sub.iterrows():
            rows.append({
                "timestamp":            r["timestamp"],
                "close":                round(float(r["close"]), 2),
                "signal":               r["signal"],
                "risk_adjusted_return": round(float(r["risk_adjusted_return"]), 6),
            })
        return rows

    best5  = _to_records(trade_df.nlargest(5,  "risk_adjusted_return"))
    worst5 = _to_records(trade_df.nsmallest(5, "risk_adjusted_return"))

    return {
        "total_trade_bars": len(trades),
        "descriptive": {
            "mean":     round(mean,     6),
            "median":   round(median,   6),
            "std":      round(std,      6),
            "skewness": round(skewness, 4),
            "kurtosis": round(kurtosis, 4),
        },
        "percentiles": {
            "pct_5":  round(pct_5,  6),
            "pct_95": round(pct_95, 6),
        },
        "best_5_trades":  best5,
        "worst_5_trades": worst5,
    }


# ── Print ─────────────────────────────────────────────────────────────────────

def print_report(m: dict) -> None:
    div = "═" * 56
    sep = "─" * 56
    d   = m["descriptive"]
    p   = m["percentiles"]

    print(f"\n{div}")
    print("  TRADE RETURN DISTRIBUTION ANALYSIS")
    print(div)
    print(f"  {'Total Trade Bars':<30}: {m['total_trade_bars']:>10,}")
    print(sep)
    print(f"  {'Mean':<30}: {d['mean']:>+14.6f}")
    print(f"  {'Median':<30}: {d['median']:>+14.6f}")
    print(f"  {'Std Dev':<30}: {d['std']:>14.6f}")
    print(f"  {'Skewness':<30}: {d['skewness']:>14.4f}")
    print(f"  {'Kurtosis (excess)':<30}: {d['kurtosis']:>14.4f}")
    print(sep)
    print(f"  {'5th Percentile (VaR 95%)':<30}: {p['pct_5']:>+14.6f}")
    print(f"  {'95th Percentile':<30}: {p['pct_95']:>+14.6f}")

    print(f"\n{sep}")
    print("  TOP 5 BEST TRADES")
    print(sep)
    for i, t in enumerate(m["best_5_trades"], 1):
        print(f"  {i}. {t['timestamp']}  {t['signal']:<6}  "
              f"${t['close']:>10,.2f}  {t['risk_adjusted_return']:>+10.4f}")

    print(f"\n{sep}")
    print("  TOP 5 WORST TRADES")
    print(sep)
    for i, t in enumerate(m["worst_5_trades"], 1):
        print(f"  {i}. {t['timestamp']}  {t['signal']:<6}  "
              f"${t['close']:>10,.2f}  {t['risk_adjusted_return']:>+10.4f}")

    print(f"\n{div}")

    # Interpretasi skewness & kurtosis
    sk = d["skewness"]
    kt = d["kurtosis"]
    print(f"\n  Interpretasi:")
    sk_desc = "positif (lebih banyak win kecil, loss besar jarang)" if sk > 0 \
              else "negatif (lebih banyak loss kecil, win besar jarang)"
    kt_desc = "leptokurtic — fat tails (kejadian ekstrem lebih sering)" if kt > 0 \
              else "platykurtic — thin tails (distribusi lebih datar)"
    print(f"  Skewness  {sk:>+.4f} → {sk_desc}")
    print(f"  Kurtosis  {kt:>+.4f} → {kt_desc}")
    print(f"{div}\n")


# ── Save ──────────────────────────────────────────────────────────────────────

def save_json(m: dict, path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(m, f, indent=2)
    log.info("Tersimpan → %s", path)


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> dict:
    log.info("═" * 55)
    log.info("Trade Distribution Analysis — BTC Hybrid Model")
    log.info("═" * 55)

    df = load_data(INPUT_PATH)
    m  = analyze_distribution(df)

    print_report(m)
    save_json(m)

    log.info("═" * 55)
    return m


if __name__ == "__main__":
    run()
