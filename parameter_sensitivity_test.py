"""
parameter_sensitivity_test.py — Parameter Sensitivity Test.
Input:  data/btc_full_hybrid_dataset.csv
Output: data/parameter_sensitivity_report.json
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
INPUT_PATH  = BASE_DIR / "data" / "btc_full_hybrid_dataset.csv"
OUTPUT_PATH = BASE_DIR / "data" / "parameter_sensitivity_report.json"

BARS_PER_YEAR    = 6 * 365
THRESHOLDS       = list(range(3, 7))   # 3, 4, 5, 6
INITIAL_EQ       = 10_000.0


# ── Load ──────────────────────────────────────────────────────────────────────

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Pastikan hybrid_score ada — hitung jika perlu
    if "hybrid_score" not in df.columns:
        t = pd.to_numeric(df.get("trend_score", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0)
        d = pd.to_numeric(df.get("derivatives_score", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0)
        df["hybrid_score"] = t + d
        log.info("hybrid_score dihitung dari trend_score + derivatives_score.")

    required = ["hybrid_score", "regime", "close"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom tidak ditemukan: {missing}")

    log.info("Loaded : %d baris | %s → %s",
             len(df),
             df["timestamp"].iloc[0].strftime("%Y-%m-%d"),
             df["timestamp"].iloc[-1].strftime("%Y-%m-%d"))
    return df


# ── Signal Generation ─────────────────────────────────────────────────────────

def generate_signals(df: pd.DataFrame, threshold: int) -> pd.Series:
    regime = df["regime"]
    score  = pd.to_numeric(df["hybrid_score"], errors="coerce").fillna(0)

    signal = pd.Series("NONE", index=df.index)
    signal[(regime == "UP")   & (score >= threshold)] = "LONG"
    signal[(regime == "DOWN") & (score >= threshold)] = "SHORT"
    return signal


# ── Returns ───────────────────────────────────────────────────────────────────

def calc_simple_returns(df: pd.DataFrame, signal: pd.Series) -> pd.Series:
    """
    Simple return per bar:
      LONG  → next bar close / this bar close - 1
      SHORT → -(next bar close / this bar close - 1)
      NONE  → 0
    """
    pct_change = df["close"].pct_change().shift(-1).fillna(0)
    ret = pd.Series(0.0, index=df.index)
    ret[signal == "LONG"]  =  pct_change[signal == "LONG"]
    ret[signal == "SHORT"] = -pct_change[signal == "SHORT"]
    return ret


# ── Equity & Metrics ──────────────────────────────────────────────────────────

def build_equity(returns: pd.Series) -> pd.Series:
    eq    = [INITIAL_EQ]
    for r in returns:
        eq.append(max(eq[-1] * (1 + r), 0.01))
    return pd.Series(eq[1:], index=returns.index)


def calc_metrics(returns: pd.Series, signal: pd.Series) -> dict:
    trades = returns[returns != 0]

    # Total return dari equity
    equity       = build_equity(returns)
    total_return = float((equity.iloc[-1] - INITIAL_EQ) / INITIAL_EQ)

    # Sharpe
    if len(trades) > 1 and trades.std() > 0:
        sharpe = float((trades.mean() / trades.std()) * np.sqrt(BARS_PER_YEAR))
    else:
        sharpe = 0.0

    # Max drawdown
    roll_max = equity.cummax()
    max_dd   = float(((equity - roll_max) / roll_max).min())

    # Winrate & profit factor
    wins   = trades[trades > 0]
    losses = trades[trades < 0]
    winrate = float((trades > 0).sum() / len(trades)) if len(trades) > 0 else 0.0
    pf      = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else float("inf")

    n_long  = int((signal == "LONG").sum())
    n_short = int((signal == "SHORT").sum())
    n_none  = int((signal == "NONE").sum())

    return {
        "total_trades":  int(len(trades)),
        "n_long":        n_long,
        "n_short":       n_short,
        "n_none":        n_none,
        "total_return":  round(total_return * 100, 4),
        "final_equity":  round(float(equity.iloc[-1]), 2),
        "sharpe":        round(sharpe, 4),
        "max_drawdown":  round(max_dd * 100, 4),
        "winrate":       round(winrate * 100, 4),
        "profit_factor": round(pf, 4) if pf != float("inf") else None,
    }


# ── Run Sensitivity ───────────────────────────────────────────────────────────

def run_sensitivity(df: pd.DataFrame) -> list:
    results = []

    for thresh in THRESHOLDS:
        log.info("─" * 50)
        log.info("Threshold = %d", thresh)

        signal  = generate_signals(df, thresh)
        returns = calc_simple_returns(df, signal)
        metrics = calc_metrics(returns, signal)

        log.info("  trades=%d  return=%+.2f%%  sharpe=%.3f  dd=%.2f%%",
                 metrics["total_trades"],
                 metrics["total_return"],
                 metrics["sharpe"],
                 metrics["max_drawdown"])

        results.append({
            "threshold": thresh,
            **metrics,
        })

    return results


# ── Print Report ──────────────────────────────────────────────────────────────

def print_report(results: list) -> None:
    div = "═" * 72
    sep = "─" * 72

    print(f"\n{div}")
    print("  PARAMETER SENSITIVITY TEST — hybrid_score threshold")
    print(div)
    print(f"  {'Thresh':>6} {'Trades':>8} {'LONG':>6} {'SHORT':>6} "
          f"{'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>9}")
    print(f"  {'─'*6} {'─'*8} {'─'*6} {'─'*6} "
          f"{'─'*10} {'─'*8} {'─'*8} {'─'*9}")

    for r in results:
        print(f"  {r['threshold']:>6}  "
              f"{r['total_trades']:>7,}  "
              f"{r['n_long']:>5,}  "
              f"{r['n_short']:>5,}  "
              f"{r['total_return']:>+9.2f}%  "
              f"{r['sharpe']:>7.3f}  "
              f"{r['max_drawdown']:>7.2f}%  "
              f"{r['winrate']:>8.2f}%")

    print(sep)

    # Optimal threshold
    best = max(results, key=lambda x: x["sharpe"])
    print(f"\n  Optimal threshold (highest Sharpe): {best['threshold']}")
    print(f"  → Sharpe={best['sharpe']:.4f}  "
          f"Return={best['total_return']:+.2f}%  "
          f"MaxDD={best['max_drawdown']:.2f}%")
    print(f"\n{div}\n")


# ── Save ──────────────────────────────────────────────────────────────────────

def save_json(data: dict, path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    log.info("Tersimpan → %s", path)


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> dict:
    log.info("═" * 55)
    log.info("Parameter Sensitivity Test — BTC Hybrid Model")
    log.info("  Thresholds tested : %s", THRESHOLDS)
    log.info("═" * 55)

    df      = load_data(INPUT_PATH)
    results = run_sensitivity(df)

    print_report(results)

    best = max(results, key=lambda x: x["sharpe"])
    output = {
        "thresholds_tested": THRESHOLDS,
        "results":           results,
        "optimal_threshold": best["threshold"],
        "optimal_sharpe":    best["sharpe"],
    }

    save_json(output)
    log.info("═" * 55)
    return output


if __name__ == "__main__":
    run()
