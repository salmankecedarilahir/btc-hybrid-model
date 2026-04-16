"""
regime_performance_analysis.py — Regime Performance Analysis.
Input:  data/btc_risk_managed_results.csv
Output: data/regime_performance_report.json
"""

import json
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
INPUT_PATH  = BASE_DIR / "data" / "btc_risk_managed_results.csv"
OUTPUT_PATH = BASE_DIR / "data" / "regime_performance_report.json"

BARS_PER_YEAR = 6 * 365


# ── Load ──────────────────────────────────────────────────────────────────────

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Fallback: volatility_regime dari atr_percentile jika tidak ada
    if "volatility_regime" not in df.columns:
        if "atr_percentile" in df.columns:
            def _vr(x):
                if x < 30:   return "LOW"
                elif x > 70: return "HIGH"
                else:         return "MID"
            df["volatility_regime"] = df["atr_percentile"].apply(_vr)
            log.info("volatility_regime dibuat dari atr_percentile.")
        else:
            df["volatility_regime"] = "UNKNOWN"
            log.warning("volatility_regime tidak ditemukan — set ke UNKNOWN.")

    required = ["regime", "volatility_regime", "risk_adjusted_return"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom tidak ditemukan: {missing}")

    log.info("Loaded : %d baris | %s → %s",
             len(df),
             df["timestamp"].iloc[0].strftime("%Y-%m-%d"),
             df["timestamp"].iloc[-1].strftime("%Y-%m-%d"))
    return df


# ── Per-Group Metrics ─────────────────────────────────────────────────────────

def calc_group_metrics(sub: pd.DataFrame) -> dict:
    ret    = sub["risk_adjusted_return"].fillna(0.0)
    trades = ret[ret != 0]

    total_return = float(ret.sum())
    trade_count  = int(len(trades))
    winrate      = float((trades > 0).sum() / len(trades)) if len(trades) > 0 else 0.0

    if len(trades) > 1 and trades.std() > 0:
        sharpe = float((trades.mean() / trades.std()) * np.sqrt(BARS_PER_YEAR))
    else:
        sharpe = 0.0

    return {
        "trade_count":  trade_count,
        "total_return": round(total_return, 4),
        "sharpe":       round(sharpe, 4),
        "winrate":      round(winrate * 100, 4),
    }


# ── Group Analysis ────────────────────────────────────────────────────────────

def analyze_by_regime(df: pd.DataFrame) -> dict:
    result = {}
    for regime, sub in df.groupby("regime"):
        result[str(regime)] = calc_group_metrics(sub)
    return result


def analyze_by_volatility(df: pd.DataFrame) -> dict:
    result = {}
    for vr, sub in df.groupby("volatility_regime"):
        result[str(vr)] = calc_group_metrics(sub)
    return result


def analyze_by_combined(df: pd.DataFrame) -> dict:
    result = {}
    for (regime, vr), sub in df.groupby(["regime", "volatility_regime"]):
        key = f"{regime}_{vr}"
        result[key] = calc_group_metrics(sub)
    return result


# ── Print Report ──────────────────────────────────────────────────────────────

def _print_table(data: dict, title: str) -> None:
    sep = "─" * 62
    print(f"\n  {title}")
    print(f"  {sep}")
    print(f"  {'Group':<22} {'Trades':>8} {'Return':>12} {'Sharpe':>10} {'WinRate':>10}")
    print(f"  {'─'*22} {'─'*8} {'─'*12} {'─'*10} {'─'*10}")
    for key, m in sorted(data.items()):
        print(f"  {key:<22} {m['trade_count']:>8,} "
              f"{m['total_return']:>+11.2f} "
              f"{m['sharpe']:>10.4f} "
              f"{m['winrate']:>9.2f}%")
    print(f"  {sep}")


def print_report(m: dict) -> None:
    div = "═" * 64
    print(f"\n{div}")
    print("  REGIME PERFORMANCE ANALYSIS — BTC Hybrid Model")
    print(div)
    _print_table(m["by_regime"],            "By Market Regime")
    _print_table(m["by_volatility_regime"], "By Volatility Regime")
    _print_table(m["by_combined"],          "By Regime × Volatility")
    print(f"\n{div}\n")


# ── Save ──────────────────────────────────────────────────────────────────────

def save_json(m: dict, path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(m, f, indent=2)
    log.info("Tersimpan → %s", path)


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> dict:
    log.info("═" * 55)
    log.info("Regime Performance Analysis — BTC Hybrid Model")
    log.info("═" * 55)

    df = load_data(INPUT_PATH)

    m = {
        "by_regime":            analyze_by_regime(df),
        "by_volatility_regime": analyze_by_volatility(df),
        "by_combined":          analyze_by_combined(df),
    }

    print_report(m)
    save_json(m)

    log.info("═" * 55)
    return m


if __name__ == "__main__":
    run()
