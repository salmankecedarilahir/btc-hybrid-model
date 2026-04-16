"""
walk_forward_engine.py — Walk Forward Validation (Fixed Date Windows).
Input:  data/btc_risk_managed_results.csv
Output: data/walk_forward_report.json
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
OUTPUT_PATH = BASE_DIR / "data" / "walk_forward_report.json"

BARS_PER_YEAR = 6 * 365

WINDOWS = [
    {
        "window":     1,
        "train_start": "2017-01-01",
        "train_end":   "2020-12-31",
        "test_start":  "2021-01-01",
        "test_end":    "2021-12-31",
    },
    {
        "window":     2,
        "train_start": "2018-01-01",
        "train_end":   "2021-12-31",
        "test_start":  "2022-01-01",
        "test_end":    "2022-12-31",
    },
    {
        "window":     3,
        "train_start": "2019-01-01",
        "train_end":   "2022-12-31",
        "test_start":  "2023-01-01",
        "test_end":    "2023-12-31",
    },
]


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


# ── Slice ─────────────────────────────────────────────────────────────────────

def slice_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end,   tz="UTC") + pd.Timedelta(days=1)
    return df[(df["timestamp"] >= s) & (df["timestamp"] < e)].copy()


# ── Metrics ───────────────────────────────────────────────────────────────────

def calc_metrics(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        log.warning("%s: slice kosong.", label)
        return {
            "bars":         0,
            "total_return": None,
            "sharpe":       None,
            "max_drawdown": None,
        }

    ret    = df["risk_adjusted_return"].fillna(0.0)
    trades = ret[ret != 0]

    # Total return dari equity
    eq           = df["equity"]
    total_return = float((eq.iloc[-1] - eq.iloc[0]) / eq.iloc[0]) if eq.iloc[0] != 0 else 0.0

    # Sharpe
    if len(trades) > 1 and trades.std() > 0:
        sharpe = float((trades.mean() / trades.std()) * np.sqrt(BARS_PER_YEAR))
    else:
        sharpe = 0.0

    # Max drawdown
    if "drawdown" in df.columns:
        max_dd = float(df["drawdown"].min())
    else:
        roll_max = eq.cummax()
        max_dd   = float(((eq - roll_max) / roll_max).min())

    log.info("  %-8s bars=%-6d  return=%+.2f%%  sharpe=%.3f  dd=%.2f%%",
             label,
             len(df),
             total_return * 100,
             sharpe,
             max_dd * 100)

    return {
        "bars":         len(df),
        "total_return": round(total_return * 100, 4),
        "sharpe":       round(sharpe, 4),
        "max_drawdown": round(max_dd * 100, 4),
    }


# ── Run Windows ───────────────────────────────────────────────────────────────

def run_windows(df: pd.DataFrame) -> list:
    results = []

    for w in WINDOWS:
        wn = w["window"]
        log.info("─" * 50)
        log.info("Window %d", wn)

        train_df = slice_window(df, w["train_start"], w["train_end"])
        test_df  = slice_window(df, w["test_start"],  w["test_end"])

        log.info("  Train: %s → %s", w["train_start"], w["train_end"])
        train_m  = calc_metrics(train_df, "TRAIN")

        log.info("  Test : %s → %s", w["test_start"], w["test_end"])
        test_m   = calc_metrics(test_df,  "TEST")

        results.append({
            "window":     wn,
            "train": {
                "start":        w["train_start"],
                "end":          w["train_end"],
                **train_m,
            },
            "test": {
                "start":        w["test_start"],
                "end":          w["test_end"],
                **test_m,
            },
        })

    return results


# ── Summary Stats ─────────────────────────────────────────────────────────────

def calc_summary(results: list) -> dict:
    test_sharpe  = [r["test"]["sharpe"]       for r in results if r["test"]["sharpe"]  is not None]
    test_return  = [r["test"]["total_return"] for r in results if r["test"]["total_return"] is not None]
    test_dd      = [r["test"]["max_drawdown"] for r in results if r["test"]["max_drawdown"] is not None]

    return {
        "avg_test_sharpe":       round(float(np.mean(test_sharpe)),  4) if test_sharpe else None,
        "best_test_sharpe":      round(float(np.max(test_sharpe)),   4) if test_sharpe else None,
        "worst_test_sharpe":     round(float(np.min(test_sharpe)),   4) if test_sharpe else None,
        "avg_test_return":       round(float(np.mean(test_return)),  4) if test_return else None,
        "best_test_return":      round(float(np.max(test_return)),   4) if test_return else None,
        "worst_test_return":     round(float(np.min(test_return)),   4) if test_return else None,
        "avg_test_max_drawdown": round(float(np.mean(test_dd)),      4) if test_dd    else None,
        "worst_test_drawdown":   round(float(np.min(test_dd)),       4) if test_dd    else None,
    }


# ── Print Report ──────────────────────────────────────────────────────────────

def print_report(results: list, summary: dict) -> None:
    div = "═" * 62
    sep = "─" * 62

    print(f"\n{div}")
    print("  WALK FORWARD VALIDATION REPORT")
    print(div)

    for r in results:
        tr = r["train"]
        te = r["test"]
        print(f"\n  Window {r['window']}")
        print(f"  {sep}")
        print(f"  {'':>6} {'Period':<24} {'Bars':>6} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8}")
        print(f"  {'─'*6} {'─'*24} {'─'*6} {'─'*10} {'─'*8} {'─'*8}")
        print(f"  {'TRAIN':<6} {tr['start']} → {tr['end']}  "
              f"{tr['bars']:>6,}  "
              f"{tr['total_return']:>+9.2f}%  "
              f"{tr['sharpe']:>7.3f}  "
              f"{tr['max_drawdown']:>7.2f}%")
        print(f"  {'TEST':<6} {te['start']} → {te['end']}  "
              f"{te['bars']:>6,}  "
              f"{te['total_return']:>+9.2f}%  "
              f"{te['sharpe']:>7.3f}  "
              f"{te['max_drawdown']:>7.2f}%")

    print(f"\n{div}")
    print("  SUMMARY (TEST WINDOWS)")
    print(sep)
    print(f"  {'Avg Test Sharpe':<30}: {summary['avg_test_sharpe']:>10.4f}")
    print(f"  {'Best Test Sharpe':<30}: {summary['best_test_sharpe']:>10.4f}")
    print(f"  {'Worst Test Sharpe':<30}: {summary['worst_test_sharpe']:>10.4f}")
    print(sep)
    print(f"  {'Avg Test Return':<30}: {summary['avg_test_return']:>+10.2f}%")
    print(f"  {'Best Test Return':<30}: {summary['best_test_return']:>+10.2f}%")
    print(f"  {'Worst Test Return':<30}: {summary['worst_test_return']:>+10.2f}%")
    print(sep)
    print(f"  {'Avg Test Max Drawdown':<30}: {summary['avg_test_max_drawdown']:>10.2f}%")
    print(f"  {'Worst Test Drawdown':<30}: {summary['worst_test_drawdown']:>10.2f}%")
    print(f"{div}\n")


# ── Save ──────────────────────────────────────────────────────────────────────

def save_json(data: dict, path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    log.info("Tersimpan → %s", path)


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> dict:
    log.info("═" * 55)
    log.info("Walk Forward Engine — Fixed Date Windows")
    log.info("  Windows : %d", len(WINDOWS))
    log.info("═" * 55)

    df      = load_data(INPUT_PATH)
    results = run_windows(df)
    summary = calc_summary(results)

    print_report(results, summary)

    output = {"windows": results, "summary": summary}
    save_json(output)

    log.info("═" * 55)
    return output


if __name__ == "__main__":
    run()
