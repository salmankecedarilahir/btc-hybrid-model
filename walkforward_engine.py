"""
walkforward_engine.py — Phase 7: Walk Forward Validation.
Input:  data/btc_risk_managed_results.csv
Output: data/walkforward_results.csv
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
INPUT_PATH    = BASE_DIR / "data" / "btc_risk_managed_results.csv"
OUTPUT_PATH   = BASE_DIR / "data" / "walkforward_results.csv"

TRAIN_RATIO   = 0.70
TEST_RATIO    = 0.30
STEP_RATIO    = 0.20
BARS_PER_YEAR = 6 * 365


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


# ── Metrics per window ────────────────────────────────────────────────────────

def calc_window_metrics(df_slice: pd.DataFrame) -> dict:
    ret = df_slice["risk_adjusted_return"].fillna(0.0)

    # Equity curve dari slice
    # equity curve tidak dipakai di metrics — dihapus untuk kompatibilitas pandas baru

    active = ret[ret != 0]
    if len(active) > 1 and active.std() > 0:
        sharpe = (active.mean() / active.std()) * np.sqrt(BARS_PER_YEAR)
    else:
        sharpe = 0.0

    eq = df_slice["equity"]
    total_return = (eq.iloc[-1] - eq.iloc[0]) / eq.iloc[0] if eq.iloc[0] != 0 else 0.0

    roll_max = eq.cummax()
    drawdown = ((eq - roll_max) / roll_max)
    max_dd   = drawdown.min()

    return {
        "sharpe":       sharpe,
        "total_return": total_return,
        "max_drawdown": max_dd,
    }


# ── Walk Forward ──────────────────────────────────────────────────────────────

def run_walkforward(df: pd.DataFrame) -> pd.DataFrame:
    n          = len(df)
    train_size = int(n * TRAIN_RATIO)
    test_size  = int(n * TEST_RATIO)
    step_size  = int(n * STEP_RATIO)

    results = []
    window  = 0
    start   = 0

    while start + train_size + test_size <= n:
        train_end  = start + train_size
        test_end   = train_end + test_size

        train_df   = df.iloc[start:train_end]
        test_df    = df.iloc[train_end:test_end]

        train_m    = calc_window_metrics(train_df)
        test_m     = calc_window_metrics(test_df)

        row = {
            "window":              window,
            "train_start":         train_df["timestamp"].iloc[0].strftime("%Y-%m-%d"),
            "train_end":           train_df["timestamp"].iloc[-1].strftime("%Y-%m-%d"),
            "test_start":          test_df["timestamp"].iloc[0].strftime("%Y-%m-%d"),
            "test_end":            test_df["timestamp"].iloc[-1].strftime("%Y-%m-%d"),
            "train_sharpe":        round(train_m["sharpe"], 4),
            "test_sharpe":         round(test_m["sharpe"], 4),
            "train_total_return":  round(train_m["total_return"] * 100, 4),
            "test_total_return":   round(test_m["total_return"] * 100, 4),
            "train_max_drawdown":  round(train_m["max_drawdown"] * 100, 4),
            "test_max_drawdown":   round(test_m["max_drawdown"] * 100, 4),
        }
        results.append(row)

        log.info(
            "Window %d | Train: %s→%s (Sharpe=%.3f) | Test: %s→%s (Sharpe=%.3f)",
            window,
            row["train_start"], row["train_end"], row["train_sharpe"],
            row["test_start"],  row["test_end"],  row["test_sharpe"],
        )

        window += 1
        start  += step_size

    return pd.DataFrame(results)


# ── Print Summary ─────────────────────────────────────────────────────────────

def print_summary(results: pd.DataFrame) -> None:
    div = "═" * 52
    sep = "─" * 52
    sharpe = results["test_sharpe"]
    ret    = results["test_total_return"]
    dd     = results["test_max_drawdown"]

    print(f"\n{div}")
    print("  WALK FORWARD VALIDATION SUMMARY")
    print(div)
    print(f"  {'Total Windows':<28}: {len(results)}")
    print(sep)
    print(f"  {'Avg Test Sharpe':<28}: {sharpe.mean():.4f}")
    print(f"  {'Best Test Sharpe':<28}: {sharpe.max():.4f}")
    print(f"  {'Worst Test Sharpe':<28}: {sharpe.min():.4f}")
    print(sep)
    print(f"  {'Avg Test Return':<28}: {ret.mean():+.2f}%")
    print(f"  {'Best Test Return':<28}: {ret.max():+.2f}%")
    print(f"  {'Worst Test Return':<28}: {ret.min():+.2f}%")
    print(sep)
    print(f"  {'Avg Test Max Drawdown':<28}: {dd.mean():.2f}%")
    print(f"  {'Worst Test Max Drawdown':<28}: {dd.min():.2f}%")
    print(div)
    print()
    print(results.to_string(index=False))
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> pd.DataFrame:
    log.info("═" * 55)
    log.info("Walk Forward Validation — Phase 7")
    log.info("  Train: %.0f%%  Test: %.0f%%  Step: %.0f%%",
             TRAIN_RATIO * 100, TEST_RATIO * 100, STEP_RATIO * 100)
    log.info("═" * 55)

    df      = load_data(INPUT_PATH)
    results = run_walkforward(df)

    print_summary(results)

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)
    log.info("Tersimpan → %s  (%d windows)", OUTPUT_PATH, len(results))

    return results


if __name__ == "__main__":
    run()
