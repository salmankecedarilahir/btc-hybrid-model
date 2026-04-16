"""
performance_report.py — Comprehensive Performance Report.
Input:  data/btc_risk_managed_results.csv
Output: data/performance_summary.json

Fixes:
  - Sharpe & Sortino from equity_return (% return per bar, full series)
  - Trade metrics (profit_factor, winrate, avg_win, avg_loss, expectancy)
    from trade_return, filtered by signal != NONE
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

BASE_DIR     = Path(__file__).parent
INPUT_PATH   = BASE_DIR / "data" / "btc_risk_managed_results.csv"
OUTPUT_PATH  = BASE_DIR / "data" / "performance_summary.json"

BARS_PER_DAY  = 6
DAYS_PER_YEAR = 365
BARS_PER_YEAR = BARS_PER_DAY * DAYS_PER_YEAR   # 2190
INITIAL_EQ    = 10_000.0


# ── Load ──────────────────────────────────────────────────────────────────────

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {path}\n"
            "Jalankan risk_engine.py terlebih dahulu."
        )
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    required = ["timestamp", "equity", "equity_return", "signal", "drawdown"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Kolom tidak ditemukan: {missing}\n"
            "Pastikan risk_engine.py sudah diupdate dan dijalankan ulang."
        )

    log.info("Loaded : %d baris | %s → %s",
             len(df),
             df["timestamp"].iloc[0].strftime("%Y-%m-%d"),
             df["timestamp"].iloc[-1].strftime("%Y-%m-%d"))
    return df


# ── Return & Equity Metrics ───────────────────────────────────────────────────

def calc_total_return(df: pd.DataFrame) -> float:
    start = df["equity"].iloc[0]
    end   = df["equity"].iloc[-1]
    return (end - start) / start


def calc_cagr(df: pd.DataFrame) -> float:
    n_bars  = len(df)
    n_years = n_bars / BARS_PER_YEAR
    if n_years <= 0:
        return 0.0
    start = df["equity"].iloc[0]
    end   = df["equity"].iloc[-1]
    if start <= 0:
        return 0.0
    return (end / start) ** (1 / n_years) - 1


def calc_max_drawdown(df: pd.DataFrame) -> float:
    return float(df["drawdown"].min())


def calc_calmar(cagr: float, max_dd: float) -> float:
    if max_dd == 0:
        return 0.0
    return cagr / abs(max_dd)


# ── Sharpe & Sortino from equity_return (full series) ─────────────────────────

def calc_sharpe(df: pd.DataFrame) -> float:
    """
    Sharpe = mean(equity_return) / std(equity_return) * sqrt(BARS_PER_YEAR)
    Uses full equity_return series (no filtering by != 0).
    """
    eq_ret = df["equity_return"].fillna(0.0)
    std    = eq_ret.std()
    if std == 0:
        return 0.0
    return float((eq_ret.mean() / std) * np.sqrt(BARS_PER_YEAR))


def calc_sortino(df: pd.DataFrame) -> float:
    """
    Sortino = mean(equity_return) / downside_std * sqrt(BARS_PER_YEAR)
    Downside std = std of negative equity_return values.
    Uses full equity_return series.
    """
    eq_ret   = df["equity_return"].fillna(0.0)
    downside = eq_ret[eq_ret < 0]
    down_std = downside.std()
    if down_std == 0 or np.isnan(down_std):
        return 0.0
    return float((eq_ret.mean() / down_std) * np.sqrt(BARS_PER_YEAR))


# ── Trade Metrics from trade_return (signal bars only) ────────────────────────

def get_trade_returns(df: pd.DataFrame) -> pd.Series:
    """
    Trade-level percent return.
    Uses trade_return column, filtered to bars where signal != NONE.
    """
    trade_df = df[df["signal"] != "NONE"].copy()

    if "trade_return" not in trade_df.columns:
        log.warning("Kolom 'trade_return' tidak ditemukan — trade metrics akan 0.")
        return pd.Series(dtype=float)

    return pd.to_numeric(trade_df["trade_return"], errors="coerce").dropna()


def calc_profit_factor(trade_ret: pd.Series) -> float:
    """
    Profit Factor = gross_profit / gross_loss (per-trade basis).
    Konsisten dengan expectancy calculation.
    """
    wins   = trade_ret[trade_ret > 0]
    losses = trade_ret[trade_ret < 0]
    if len(losses) == 0:
        return float("inf")
    gross_profit = wins.sum()
    gross_loss   = abs(losses.sum())
    if gross_loss == 0:
        return float("inf")
    return float(gross_profit / gross_loss)


def calc_winrate(trade_ret: pd.Series) -> float:
    if len(trade_ret) == 0:
        return 0.0
    return float((trade_ret > 0).sum() / len(trade_ret))


def calc_avg_win(trade_ret: pd.Series) -> float:
    wins = trade_ret[trade_ret > 0]
    return float(wins.mean()) if len(wins) > 0 else 0.0


def calc_avg_loss(trade_ret: pd.Series) -> float:
    losses = trade_ret[trade_ret < 0]
    return float(losses.mean()) if len(losses) > 0 else 0.0


def calc_expectancy(winrate: float, avg_win: float, avg_loss: float) -> float:
    return (winrate * avg_win) + ((1 - winrate) * avg_loss)


# ── Aggregate ─────────────────────────────────────────────────────────────────

def calc_all_metrics(df: pd.DataFrame) -> dict:
    total_return  = calc_total_return(df)
    cagr          = calc_cagr(df)
    sharpe        = calc_sharpe(df)
    sortino       = calc_sortino(df)
    max_dd        = calc_max_drawdown(df)
    calmar        = calc_calmar(cagr, max_dd)

    trade_ret     = get_trade_returns(df)
    profit_factor = calc_profit_factor(trade_ret)
    winrate       = calc_winrate(trade_ret)
    avg_win       = calc_avg_win(trade_ret)
    avg_loss      = calc_avg_loss(trade_ret)
    expectancy    = calc_expectancy(winrate, avg_win, avg_loss)
    total_trades  = int(len(trade_ret))

    n_bars  = len(df)
    n_years = n_bars / BARS_PER_YEAR

    return {
        "period": {
            "start":      df["timestamp"].iloc[0].strftime("%Y-%m-%d"),
            "end":        df["timestamp"].iloc[-1].strftime("%Y-%m-%d"),
            "total_bars": n_bars,
            "years":      round(n_years, 2),
        },
        "equity": {
            "initial": INITIAL_EQ,
            "final":   round(df["equity"].iloc[-1], 2),
        },
        "returns": {
            "total_return": round(total_return * 100, 4),
            "cagr":         round(cagr * 100, 4),
        },
        "risk_adjusted": {
            "sharpe_ratio":  round(sharpe,  4),
            "sortino_ratio": round(sortino, 4),
            "calmar_ratio":  round(calmar,  4),
        },
        "drawdown": {
            "max_drawdown": round(max_dd * 100, 4),
        },
        "trade_stats": {
            "total_trades":  total_trades,
            "winrate":       round(winrate * 100, 4),
            "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else None,
            "avg_win":       round(avg_win,    6),
            "avg_loss":      round(avg_loss,   6),
            "expectancy":    round(expectancy, 6),
        },
    }


# ── Print Report ──────────────────────────────────────────────────────────────

def print_report(m: dict) -> None:
    div = "═" * 54
    sep = "─" * 54

    p  = m["period"]
    eq = m["equity"]
    r  = m["returns"]
    ra = m["risk_adjusted"]
    dd = m["drawdown"]
    ts = m["trade_stats"]

    print(f"\n{div}")
    print("  BTC HYBRID MODEL — PERFORMANCE REPORT")
    print(div)
    print(f"  {'Period':<30}: {p['start']} → {p['end']}")
    print(f"  {'Duration':<30}: {p['years']:.2f} years  ({p['total_bars']:,} bars)")
    print(sep)
    print(f"  {'Initial Equity':<30}: ${eq['initial']:>12,.2f}")
    print(f"  {'Final Equity':<30}: ${eq['final']:>12,.2f}")
    print(sep)
    print(f"  {'Total Return':<30}: {r['total_return']:>+12.2f}%")
    print(f"  {'CAGR':<30}: {r['cagr']:>+12.2f}%")
    print(sep)
    print(f"  {'Sharpe Ratio (ann.)':<30}: {ra['sharpe_ratio']:>12.4f}")
    print(f"  {'Sortino Ratio (ann.)':<30}: {ra['sortino_ratio']:>12.4f}")
    print(f"  {'Calmar Ratio':<30}: {ra['calmar_ratio']:>12.4f}")
    print(sep)
    print(f"  {'Max Drawdown':<30}: {dd['max_drawdown']:>12.2f}%")
    print(sep)
    pf_str = f"{ts['profit_factor']:.4f}" if ts["profit_factor"] is not None else "∞"
    print(f"  {'Total Trades':<30}: {ts['total_trades']:>12,}")
    print(f"  {'Win Rate':<30}: {ts['winrate']:>12.2f}%")
    print(f"  {'Profit Factor':<30}: {pf_str:>12}")
    print(f"  {'Avg Win':<30}: {ts['avg_win']:>+12.6f}")
    print(f"  {'Avg Loss':<30}: {ts['avg_loss']:>+12.6f}")
    print(f"  {'Expectancy':<30}: {ts['expectancy']:>+12.6f}")
    print(f"{div}\n")


# ── Save JSON ─────────────────────────────────────────────────────────────────

def save_json(m: dict, path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(m, f, indent=2)
    log.info("Tersimpan → %s", path)


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> dict:
    log.info("═" * 55)
    log.info("Performance Report — BTC Hybrid Model")
    log.info("  Sharpe/Sortino : from equity_return (full series)")
    log.info("  Trade metrics  : from trade_return (signal bars only)")
    log.info("  Annualization  : %d bars/year", BARS_PER_YEAR)
    log.info("═" * 55)

    df      = load_data(INPUT_PATH)
    metrics = calc_all_metrics(df)

    print_report(metrics)
    save_json(metrics)

    log.info("═" * 55)
    return metrics


if __name__ == "__main__":
    run()
