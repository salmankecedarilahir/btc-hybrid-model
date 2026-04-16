"""
equity_stability_analysis.py — Equity Stability Analysis.
Input:  data/btc_risk_managed_results.csv
Output: data/equity_stability_report.json
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
OUTPUT_PATH = BASE_DIR / "data" / "equity_stability_report.json"

BARS_PER_YEAR = 6 * 365
WIN_50        = 50
WIN_100       = 100
WIN_VOL       = 50   # rolling volatility window


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


# ── Rolling Sharpe ────────────────────────────────────────────────────────────

def rolling_sharpe(returns: pd.Series, window: int) -> pd.Series:
    roll_mean = returns.rolling(window, min_periods=window // 2).mean()
    roll_std  = returns.rolling(window, min_periods=window // 2).std()
    roll_std  = roll_std.replace(0, np.nan)
    return (roll_mean / roll_std) * np.sqrt(BARS_PER_YEAR)


# ── Rolling Volatility ────────────────────────────────────────────────────────

def rolling_volatility(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window, min_periods=window // 2).std() * np.sqrt(BARS_PER_YEAR)


# ── Equity Slope Regression ───────────────────────────────────────────────────

def equity_slope_regression(equity: pd.Series) -> dict:
    """Linear regression of log equity over bar index."""
    log_eq = np.log(equity.clip(lower=0.01))
    x      = np.arange(len(log_eq))
    mask   = ~np.isnan(log_eq)

    if mask.sum() < 10:
        return {"slope": None, "r_squared": None, "p_value": None, "intercept": None}

    slope, intercept, r, p, se = scipy_stats.linregress(x[mask], log_eq[mask])

    log.info("Equity slope regression:")
    log.info("  slope     = %.6f (per bar)", slope)
    log.info("  r_squared = %.4f", r ** 2)
    log.info("  p_value   = %.6f", p)

    return {
        "slope":      round(float(slope),     6),
        "intercept":  round(float(intercept), 6),
        "r_squared":  round(float(r ** 2),    6),
        "p_value":    round(float(p),         6),
        "std_err":    round(float(se),        6),
        "annualized_drift_pct": round(float(slope * BARS_PER_YEAR * 100), 4),
    }


# ── Stability Stats ───────────────────────────────────────────────────────────

def stability_stats(series: pd.Series, label: str) -> dict:
    clean = series.dropna()
    if len(clean) == 0:
        return {}
    return {
        "mean":   round(float(clean.mean()),   4),
        "std":    round(float(clean.std()),    4),
        "min":    round(float(clean.min()),    4),
        "max":    round(float(clean.max()),    4),
        "pct_5":  round(float(np.percentile(clean, 5)),  4),
        "pct_95": round(float(np.percentile(clean, 95)), 4),
        "pct_positive": round(float((clean > 0).mean() * 100), 2),
    }


# ── Yearly Breakdown ──────────────────────────────────────────────────────────

def yearly_breakdown(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["year"] = df["timestamp"].dt.year
    result = {}
    for year, sub in df.groupby("year"):
        ret    = sub["risk_adjusted_return"].fillna(0)
        trades = ret[ret != 0]
        eq     = sub["equity"]

        total_ret = float((eq.iloc[-1] - eq.iloc[0]) / eq.iloc[0]) if eq.iloc[0] > 0 else 0.0
        sharpe    = float((trades.mean() / trades.std()) * np.sqrt(BARS_PER_YEAR)) \
                    if len(trades) > 1 and trades.std() > 0 else 0.0
        vol       = float(trades.std() * np.sqrt(BARS_PER_YEAR)) if len(trades) > 1 else 0.0

        roll_max  = eq.cummax()
        max_dd    = float(((eq - roll_max) / roll_max).min()) if len(eq) > 1 else 0.0

        result[str(year)] = {
            "bars":         len(sub),
            "total_return": round(total_ret * 100, 4),
            "sharpe":       round(sharpe, 4),
            "volatility":   round(vol * 100, 4),
            "max_drawdown": round(max_dd * 100, 4),
        }
        log.info("  %d: return=%+.2f%%  sharpe=%.3f  vol=%.2f%%  dd=%.2f%%",
                 year, total_ret*100, sharpe, vol*100, max_dd*100)

    return result


# ── Print Report ──────────────────────────────────────────────────────────────

def print_report(m: dict) -> None:
    div = "═" * 60
    sep = "─" * 60

    print(f"\n{div}")
    print("  EQUITY STABILITY ANALYSIS — BTC Hybrid Model")
    print(div)

    reg = m["equity_slope_regression"]
    print(f"\n  Equity Slope Regression (log-linear)")
    print(sep)
    print(f"  {'Slope (per bar)':<35}: {reg['slope']:>12.6f}")
    print(f"  {'Annualized Drift':<35}: {reg['annualized_drift_pct']:>+11.4f}%")
    print(f"  {'R-Squared':<35}: {reg['r_squared']:>12.4f}")
    print(f"  {'P-Value':<35}: {reg['p_value']:>12.6f}")

    for label, key in [("Rolling Sharpe (50)", "rolling_sharpe_50"),
                        ("Rolling Sharpe (100)", "rolling_sharpe_100"),
                        ("Rolling Volatility (50)", "rolling_volatility_50")]:
        s = m[key]
        if not s:
            continue
        print(f"\n  {label}")
        print(sep)
        print(f"  {'Mean':<35}: {s['mean']:>12.4f}")
        print(f"  {'Std':<35}: {s['std']:>12.4f}")
        print(f"  {'Min':<35}: {s['min']:>12.4f}")
        print(f"  {'Max':<35}: {s['max']:>12.4f}")
        print(f"  {'5th Percentile':<35}: {s['pct_5']:>12.4f}")
        print(f"  {'95th Percentile':<35}: {s['pct_95']:>12.4f}")
        print(f"  {'% Positive':<35}: {s['pct_positive']:>11.2f}%")

    print(f"\n  Yearly Breakdown")
    print(sep)
    print(f"  {'Year':<6} {'Bars':>6} {'Return':>10} {'Sharpe':>8} {'Vol':>8} {'MaxDD':>8}")
    print(f"  {'─'*6} {'─'*6} {'─'*10} {'─'*8} {'─'*8} {'─'*8}")
    for year, v in sorted(m["yearly_breakdown"].items()):
        print(f"  {year:<6} {v['bars']:>6,} "
              f"{v['total_return']:>+9.2f}%  "
              f"{v['sharpe']:>7.3f}  "
              f"{v['volatility']:>7.2f}%  "
              f"{v['max_drawdown']:>7.2f}%")
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
    log.info("Equity Stability Analysis — BTC Hybrid Model")
    log.info("  Rolling Sharpe windows : %d, %d", WIN_50, WIN_100)
    log.info("  Rolling Vol window     : %d", WIN_VOL)
    log.info("═" * 55)

    df  = load_data(INPUT_PATH)
    ret = df["risk_adjusted_return"].fillna(0.0)
    eq  = df["equity"]

    rs50  = rolling_sharpe(ret,    WIN_50)
    rs100 = rolling_sharpe(ret,    WIN_100)
    rvol  = rolling_volatility(ret, WIN_VOL)
    reg   = equity_slope_regression(eq)

    log.info("Yearly breakdown:")
    yearly = yearly_breakdown(df)

    m = {
        "equity_slope_regression": reg,
        "rolling_sharpe_50":       stability_stats(rs50,  "sharpe_50"),
        "rolling_sharpe_100":      stability_stats(rs100, "sharpe_100"),
        "rolling_volatility_50":   stability_stats(rvol,  "vol_50"),
        "yearly_breakdown":        yearly,
    }

    print_report(m)
    save_json(m)

    log.info("═" * 55)
    return m


if __name__ == "__main__":
    run()
