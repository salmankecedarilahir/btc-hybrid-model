"""
test_backtest.py — Validasi Phase 5 Backtest Results.
"""

import sys
from pathlib import Path

import pandas as pd

INPUT_PATH = Path(__file__).parent / "data" / "btc_backtest_results.csv"
DIV = "═" * 60
SEP = "─" * 60


def load() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        print(f"[ERROR] File tidak ditemukan: {INPUT_PATH}")
        print("        Jalankan backtest_engine.py terlebih dahulu.")
        sys.exit(1)
    df = pd.read_csv(INPUT_PATH, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def print_summary(df: pd.DataFrame) -> None:
    trades = df[df["signal"] != "NONE"].copy()
    if trades.empty:
        print("  Tidak ada trades ditemukan.")
        return

    total   = len(trades)
    wins    = (trades["trade_return"] > 0).sum()
    losses  = (trades["trade_return"] < 0).sum()
    winrate = wins / total * 100

    win_ret  = trades[trades["trade_return"] > 0]["trade_return"]
    loss_ret = trades[trades["trade_return"] < 0]["trade_return"]

    avg_win  = win_ret.mean() * 100  if len(win_ret)  > 0 else 0.0
    avg_loss = loss_ret.mean() * 100 if len(loss_ret) > 0 else 0.0

    pf = (win_ret.sum() / abs(loss_ret.sum())
          if len(loss_ret) > 0 and loss_ret.sum() != 0 else float("inf"))

    total_return = (df["equity_curve"].iloc[-1] - 1) * 100

    # Sharpe (annualized, 4H = 2190 bars/year)
    r    = trades["trade_return"]
    sharpe = (r.mean() / r.std() * (2190 ** 0.5)) if r.std() > 0 else 0.0

    # Max drawdown
    ec   = df["equity_curve"]
    dd   = ((ec - ec.cummax()) / ec.cummax()).min() * 100

    print(f"\n{DIV}")
    print("  BACKTEST RESULTS SUMMARY")
    print(DIV)
    print(f"  {'Total Trades':<28}: {total:,}")
    print(f"  {'  Wins':<28}: {wins:,}  ({winrate:.1f}%)")
    print(f"  {'  Losses':<28}: {losses:,}  ({100-winrate:.1f}%)")
    print(SEP)
    print(f"  {'Win Rate':<28}: {winrate:.2f}%")
    print(f"  {'Avg Win':<28}: {avg_win:+.4f}%")
    print(f"  {'Avg Loss':<28}: {avg_loss:+.4f}%")
    print(f"  {'Profit Factor':<28}: {pf:.3f}")
    print(SEP)
    print(f"  {'Total Return':<28}: {total_return:+.2f}%")
    print(f"  {'Sharpe Ratio (ann.)':<28}: {sharpe:.3f}")
    print(f"  {'Max Drawdown':<28}: {dd:.2f}%")
    print(DIV)


def print_best_trades(df: pd.DataFrame, n: int = 5) -> None:
    print(f"\n{SEP}")
    print(f"  Top {n} Best Trades")
    print(SEP)

    trades = df[df["signal"] != "NONE"].copy()
    if trades.empty:
        print("  Tidak ada trades.")
        return

    cols = ["timestamp", "close", "signal", "signal_strength",
            "hybrid_score", "trade_return"]
    cols = [c for c in cols if c in trades.columns]

    top = trades.nlargest(n, "trade_return")[cols].copy()
    top["timestamp"]    = top["timestamp"].dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M")
    top["close"]        = top["close"].map("${:,.2f}".format)
    top["trade_return"] = top["trade_return"].map("{:+.4f}".format)
    print(top.to_string(index=False))


def print_worst_trades(df: pd.DataFrame, n: int = 5) -> None:
    print(f"\n{SEP}")
    print(f"  Top {n} Worst Trades")
    print(SEP)

    trades = df[df["signal"] != "NONE"].copy()
    if trades.empty:
        print("  Tidak ada trades.")
        return

    cols = ["timestamp", "close", "signal", "signal_strength",
            "hybrid_score", "trade_return"]
    cols = [c for c in cols if c in trades.columns]

    worst = trades.nsmallest(n, "trade_return")[cols].copy()
    worst["timestamp"]    = worst["timestamp"].dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M")
    worst["close"]        = worst["close"].map("${:,.2f}".format)
    worst["trade_return"] = worst["trade_return"].map("{:+.4f}".format)
    print(worst.to_string(index=False))


def main() -> None:
    print(DIV)
    print("  BTC Hybrid Model — Phase 5: Backtest Validation")
    print(DIV)

    df = load()
    print(f"\n  Loaded : {len(df):,} baris")
    print(f"  Range  : {df['timestamp'].iloc[0].strftime('%Y-%m-%d')} "
          f"→ {df['timestamp'].iloc[-1].strftime('%Y-%m-%d')}")

    print_summary(df)
    print_best_trades(df)
    print_worst_trades(df)

    print(f"\n{DIV}")
    print("  Validation complete.")
    print(DIV)


if __name__ == "__main__":
    main()
