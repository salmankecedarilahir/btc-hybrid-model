"""
predictive_power_test.py — Predictive Power Test.
Input:  data/btc_full_hybrid_dataset.csv
Output: data/predictive_power_report.json
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
INPUT_PATH  = BASE_DIR / "data" / "btc_trading_signals.csv"
OUTPUT_PATH = BASE_DIR / "data" / "predictive_power_report.json"


# ── Load ──────────────────────────────────────────────────────────────────────

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    required = ["hybrid_score", "derivatives_score", "close"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom tidak ditemukan: {missing}")

    log.info("Loaded : %d baris | %s → %s",
             len(df),
             df["timestamp"].iloc[0].strftime("%Y-%m-%d"),
             df["timestamp"].iloc[-1].strftime("%Y-%m-%d"))
    return df


# ── Feature Engineering ───────────────────────────────────────────────────────

def add_future_return(df: pd.DataFrame) -> pd.DataFrame:
    df["future_return_4h"] = df["close"].shift(-1) / df["close"] - 1
    # Drop last row (NaN future return)
    df = df.dropna(subset=["future_return_4h"]).reset_index(drop=True)
    log.info("future_return_4h — mean: %.4f%%  std: %.4f%%",
             df["future_return_4h"].mean() * 100,
             df["future_return_4h"].std()  * 100)
    return df


# ── Pearson Correlation ───────────────────────────────────────────────────────

def calc_pearson(x: pd.Series, y: pd.Series, label: str) -> dict:
    mask = x.notna() & y.notna()
    x_c  = x[mask]
    y_c  = y[mask]

    if len(x_c) < 10:
        log.warning("%s: tidak cukup data untuk korelasi.", label)
        return {"correlation": None, "p_value": None, "n": 0, "significant": False}

    r, p = scipy_stats.pearsonr(x_c, y_c)
    sig  = p < 0.05

    log.info("%-30s: r=%.4f  p=%.4f  n=%d  %s",
             label, r, p, len(x_c),
             "✓ SIGNIFICANT" if sig else "✗ not significant")

    return {
        "correlation": round(float(r), 6),
        "p_value":     round(float(p), 6),
        "n":           int(len(x_c)),
        "significant": bool(sig),
    }


# ── Grouped Mean ──────────────────────────────────────────────────────────────

def calc_grouped_mean(df: pd.DataFrame) -> dict:
    grouped = (
        df.groupby("hybrid_score")["future_return_4h"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped.columns = ["hybrid_score", "mean_return", "std_return", "count"]

    result = {}
    for _, row in grouped.iterrows():
        key = str(int(row["hybrid_score"]))
        result[key] = {
            "mean_return": round(float(row["mean_return"]) * 100, 6),
            "std_return":  round(float(row["std_return"])  * 100, 6),
            "count":       int(row["count"]),
        }

    log.info("Grouped mean by hybrid_score:")
    for score, vals in sorted(result.items(), key=lambda x: int(x[0])):
        log.info("  score %3s: mean=%.4f%%  n=%d",
                 score, vals["mean_return"], vals["count"])

    return result


# ── Print Report ──────────────────────────────────────────────────────────────

def print_report(m: dict) -> None:
    div = "═" * 58
    sep = "─" * 58

    print(f"\n{div}")
    print("  PREDICTIVE POWER TEST — BTC Hybrid Model")
    print(div)
    print(f"  {'Total Samples':<35}: {m['total_samples']:>10,}")
    print(sep)
    print("  Pearson Correlation vs future_return_4H")
    print(sep)

    for key, label in [
        ("hybrid_vs_future",      "hybrid_score"),
        ("derivatives_vs_future", "derivatives_score"),
    ]:
        c = m["correlations"][key]
        r = c["correlation"]
        p = c["p_value"]
        s = "✓ SIGNIFICANT" if c["significant"] else "✗ not significant"
        print(f"  {label:<35}: r={r:>+.4f}  p={p:.4f}  {s}")

    print(sep)
    print("  Mean Future Return 4H by hybrid_score")
    print(sep)
    print(f"  {'Score':<10} {'Mean Return':>14} {'Std':>14} {'Count':>8}")
    print(f"  {'─'*10} {'─'*14} {'─'*14} {'─'*8}")

    grouped = m["grouped_by_hybrid_score"]
    for score in sorted(grouped.keys(), key=int):
        v    = grouped[score]
        mean = v["mean_return"]
        std  = v["std_return"]
        cnt  = v["count"]
        bar  = "▲" if mean > 0 else "▼"
        print(f"  {score:<10} {mean:>+13.4f}% {std:>13.4f}% {cnt:>8,}  {bar}")

    print(f"\n{div}")
    print("  Interpretasi:")
    r_h = m["correlations"]["hybrid_vs_future"]["correlation"]
    r_d = m["correlations"]["derivatives_vs_future"]["correlation"]
    if r_h and abs(r_h) > 0.05:
        print(f"  hybrid_score memiliki predictive power (r={r_h:+.4f})")
    else:
        print(f"  hybrid_score memiliki predictive power lemah (r={r_h:+.4f})")
    if r_d and abs(r_d) > 0.05:
        print(f"  derivatives_score memiliki predictive power (r={r_d:+.4f})")
    else:
        print(f"  derivatives_score memiliki predictive power lemah (r={r_d:+.4f})")
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
    log.info("Predictive Power Test — BTC Hybrid Model")
    log.info("═" * 55)

    df  = load_data(INPUT_PATH)
    df  = add_future_return(df)

    fr  = df["future_return_4h"]
    hs  = pd.to_numeric(df["hybrid_score"],      errors="coerce")
    ds  = pd.to_numeric(df["derivatives_score"], errors="coerce")

    corr_hybrid = calc_pearson(hs, fr, "hybrid_score vs future_return")
    corr_deriv  = calc_pearson(ds, fr, "derivatives_score vs future_return")
    grouped     = calc_grouped_mean(df)

    m = {
        "total_samples": len(df),
        "correlations": {
            "hybrid_vs_future":      corr_hybrid,
            "derivatives_vs_future": corr_deriv,
        },
        "grouped_by_hybrid_score": grouped,
    }

    print_report(m)
    save_json(m)

    log.info("═" * 55)
    return m


if __name__ == "__main__":
    run()
