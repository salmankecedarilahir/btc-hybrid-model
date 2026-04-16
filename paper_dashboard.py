"""
paper_dashboard.py  —  BTC Hybrid Paper Trader: Dashboard Viewer.

Membaca:  data/paper_trading_log.csv
          data/paper_trading_state.json

Cara pakai:
  python paper_dashboard.py            # tampilkan summary sekali
  python paper_dashboard.py --watch    # auto-refresh tiap 60 detik
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ── Optional: rich untuk pretty terminal ──────────────────────────
try:
    from rich.console import Console
    from rich.table   import Table
    from rich.panel   import Panel
    from rich.columns import Columns
    from rich         import box as rbox
    RICH    = True
    console = Console()
except ImportError:
    RICH = False

# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
LOG_PATH   = DATA_DIR / "paper_trading_log.csv"
STATE_PATH = DATA_DIR / "paper_trading_state.json"

INITIAL_EQUITY = 100.0
BARS_PER_YEAR  = 2190   # 6 candles/hari × 365 hari


# ════════════════════════════════════════════════════════════════════
#  DATA LOADERS
# ════════════════════════════════════════════════════════════════════

def load_state() -> dict:
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    return {}


def load_log() -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(LOG_PATH, parse_dates=["timestamp"])
    except Exception:
        return pd.DataFrame()


# ════════════════════════════════════════════════════════════════════
#  COMPUTE METRICS
# ════════════════════════════════════════════════════════════════════

def compute_summary(df: pd.DataFrame, state: dict) -> dict:
    if len(df) == 0:
        return {}

    equity   = float(state.get("equity",   INITIAL_EQUITY))
    max_eq   = float(state.get("max_equity", INITIAL_EQUITY))
    total_tr = int(state.get("total_trades",   0))
    winning  = int(state.get("winning_trades", 0))

    pnl     = equity - INITIAL_EQUITY
    pnl_pct = pnl / INITIAL_EQUITY * 100
    dd      = (equity - max_eq) / max_eq * 100 if max_eq > 0 else 0.0
    n_yrs   = len(df) / BARS_PER_YEAR
    cagr    = ((equity / INITIAL_EQUITY) ** (1 / max(n_yrs, 0.001)) - 1) * 100

    er      = df["bar_return_pct"].fillna(0) / 100
    sharpe  = float((er.mean() / er.std()) * np.sqrt(BARS_PER_YEAR)) \
              if er.std() > 0 else 0.0
    neg_r   = er[er < 0]
    sortino = float((er.mean() / neg_r.std()) * np.sqrt(BARS_PER_YEAR)) \
              if len(neg_r) > 0 and neg_r.std() > 0 else 0.0
    calmar  = float(cagr / abs(dd)) if dd != 0 else 0.0
    win_rate= winning / max(total_tr, 1) * 100

    sig_counts = df["signal"].value_counts().to_dict()
    t1_bars    = int((df["kill_tier"] == 1).sum()) if "kill_tier" in df.columns else 0
    t2_bars    = int((df["kill_tier"] == 2).sum()) if "kill_tier" in df.columns else 0

    # V5 regime stats
    regime_bull_bars = int(df["regime_bull"].sum()) if "regime_bull" in df.columns else 0

    # 1D trend from state (live) or last log row
    trend_1d     = state.get("trend_1d", "NEUTRAL")
    entry_quality= state.get("entry_quality", "NO_DATA")

    # YoY breakdown
    df2 = df.copy()
    df2["year"] = pd.to_datetime(df2["timestamp"]).dt.year
    yoy = {}
    for yr, grp in df2.groupby("year"):
        if len(grp) < 2:
            continue
        s = float(grp["equity"].iloc[0])
        e = float(grp["equity"].iloc[-1])
        if s > 0:
            yoy[int(yr)] = (e - s) / s * 100

    recent_cols = ["timestamp", "signal", "position", "price", "equity", "bar_return_pct"]
    if "trend_1d" in df.columns:
        recent_cols.append("trend_1d")
    if "entry_quality" in df.columns:
        recent_cols.append("entry_quality")
    recent = df.tail(10)[recent_cols].copy()

    return dict(
        equity       = equity,
        pnl          = pnl,
        pnl_pct      = pnl_pct,
        dd           = dd,
        cagr         = cagr,
        sharpe       = sharpe,
        sortino      = sortino,
        calmar       = calmar,
        win_rate     = win_rate,
        total_tr     = total_tr,
        sig_long     = sig_counts.get("LONG",  0),
        sig_short    = sig_counts.get("SHORT", 0),
        sig_none     = sig_counts.get("NONE",  0),
        t1_bars      = t1_bars,
        t2_bars      = t2_bars,
        total_bars   = len(df),
        best_bar     = float(df["bar_return_pct"].max()),
        worst_bar    = float(df["bar_return_pct"].min()),
        recent       = recent,
        yoy          = yoy,
        position     = int(state.get("position",   0)),
        tier         = int(state.get("tier",        0)),
        last_ts      = str(state.get("last_bar_ts", "—")),
        started_at   = str(state.get("started_at",  "—")),
        # V5 additions
        trend_1d         = trend_1d,
        entry_quality    = entry_quality,
        regime_bull_bars = regime_bull_bars,
        bull_1d          = int(state.get("bull_1d", 0)),
    )


# ════════════════════════════════════════════════════════════════════
#  PRINT DASHBOARD
# ════════════════════════════════════════════════════════════════════

POS_LBL  = {1: "LONG  ▲", -1: "SHORT ▼",  0: "FLAT  —"}
TIER_LBL = {0: "NORMAL ✓", 1: "TIER1 [ALERT] half-size", 2: "TIER2 [WARN]  PAUSED"}


def _print_rich(m: dict, now: str) -> None:
    sign = "+" if m["pnl"] >= 0 else ""

    q_emoji = {"EXCELLENT":"[GREEN]","GOOD":"[YELLOW]","WAIT":"🔵","NO_ENTRY":"[RED]","NO_DATA":"⬜"}
    q_color = {"EXCELLENT":"green","GOOD":"yellow","WAIT":"blue","NO_ENTRY":"red","NO_DATA":"dim"}
    t1d_emoji= {"BULLISH":"[UP]","BEARISH":"[DOWN]","NEUTRAL":"➡️"}
    t1d_color= {"BULLISH":"green","BEARISH":"red","NEUTRAL":"white"}

    trend_1d  = m.get("trend_1d", "NEUTRAL")
    eq_raw    = m.get("entry_quality", "NO_DATA")
    bull_1d   = m.get("bull_1d", 0)
    reg_lbl   = "🔥 BULL BOOST (×1.3)" if bull_1d else "NORMAL (×1.0)"
    reg_color = "bold yellow" if bull_1d else "dim"

    # ── Performance table ────────────────────────────────────────
    t1 = Table(box=rbox.SIMPLE, show_header=False, padding=(0, 1))
    t1.add_column("k", style="dim")
    t1.add_column("v", justify="right")
    t1.add_row("Virtual Equity", f"[bold green]${m['equity']:.4f}[/]")
    clr = "green" if m["pnl"] >= 0 else "red"
    t1.add_row("P&L",
               f"[{clr}]{sign}${m['pnl']:.4f}  ({sign}{m['pnl_pct']:.2f}%)[/]")
    t1.add_row("Drawdown",
               f"[red]{m['dd']:.2f}%[/]" if m["dd"] < -10 else f"{m['dd']:.2f}%")
    t1.add_row("CAGR (ann.)",  f"{m['cagr']:+.2f}%")
    t1.add_row("Sharpe",       f"{m['sharpe']:.4f}")
    t1.add_row("Sortino",      f"{m['sortino']:.4f}")
    t1.add_row("Calmar",       f"[bold]{m['calmar']:.4f}[/]")

    # ── Status table ─────────────────────────────────────────────
    t2 = Table(box=rbox.SIMPLE, show_header=False, padding=(0, 1))
    t2.add_column("k", style="dim")
    t2.add_column("v", justify="right")
    t2.add_row("Position",      POS_LBL.get(m["position"], "—"))
    t2.add_row("Kill Switch",   TIER_LBL.get(m["tier"], "—"))
    t2.add_row("Win Rate",      f"{m['win_rate']:.1f}%")
    t2.add_row("Total Trades",  str(m["total_tr"]))
    t2.add_row("Total Bars",    str(m["total_bars"]))
    t2.add_row("LONG  signals", str(m["sig_long"]))
    t2.add_row("SHORT signals", str(m["sig_short"]))
    t2.add_row("Tier2 paused",  f"{m['t2_bars']} bars")
    # V5 additions
    t2.add_row("─────────────", "──────────────")
    t2.add_row("Regime (V5)",
               f"[{reg_color}]{reg_lbl}[/]")
    t2.add_row(f"{t1d_emoji.get(trend_1d,'')} 1D Trend",
               f"[{t1d_color.get(trend_1d,'white')}]{trend_1d}[/]")
    t2.add_row("Bull bars (V5)",f"{m.get('regime_bull_bars',0)}")
    t2.add_row("─────────────", "──────────────")
    t2.add_row("15m Entry",
               f"[{q_color.get(eq_raw,'dim')}]{q_emoji.get(eq_raw,'⬜')} {eq_raw}[/]")

    console.rule(f"[bold cyan]BTC Paper Trader — {now}[/]")
    console.print(Columns([
        Panel(t1, title="[green]Performance[/]", border_style="green"),
        Panel(t2, title="[blue]Status + V5[/]",  border_style="blue"),
    ]))

    # ── Recent bars ──────────────────────────────────────────────
    rec = Table(title="Recent 10 Bars", box=rbox.SIMPLE_HEAD)
    rec.add_column("Time",   width=17)
    rec.add_column("Signal", width=7)
    rec.add_column("Pos",    justify="right", width=4)
    rec.add_column("Price",  justify="right")
    rec.add_column("Equity", justify="right")
    rec.add_column("BarRet", justify="right")
    has_1d = "trend_1d" in m["recent"].columns
    if has_1d:
        rec.add_column("1D", width=8)
    for _, row in m["recent"].iterrows():
        br    = float(row.get("bar_return_pct", 0))
        color = "green" if br > 0 else ("red" if br < 0 else "")
        br_s  = f"[{color}]{br:+.3f}%[/]" if color else f"{br:+.3f}%"
        ts    = str(row["timestamp"])[:16] if pd.notna(row["timestamp"]) else "—"
        row_data = [ts,
                    str(row.get("signal", "")),
                    str(int(row.get("position", 0))),
                    f"${float(row.get('price', 0)):,.0f}",
                    f"${float(row.get('equity', 0)):.4f}",
                    br_s]
        if has_1d:
            td = str(row.get("trend_1d",""))
            tc = "green" if td=="BULLISH" else ("red" if td=="BEARISH" else "white")
            row_data.append(f"[{tc}]{td[:4]}[/]")
        rec.add_row(*row_data)
    console.print(rec)

    # ── YoY table ────────────────────────────────────────────────
    if m["yoy"]:
        yoy_t = Table(title="Year-by-Year Return", box=rbox.SIMPLE_HEAD)
        yoy_t.add_column("Year", width=6)
        yoy_t.add_column("Return", justify="right")
        yoy_t.add_column("Visual", width=24)
        for yr, ret in sorted(m["yoy"].items()):
            bar = ("█" * min(int(abs(ret) / 20), 24)) if ret >= 0 \
                  else ("▒" * min(int(abs(ret) / 20), 24))
            clr = "green" if ret >= 0 else "red"
            yoy_t.add_row(str(yr),
                          f"[{clr}]{ret:+.1f}%[/]",
                          f"[{clr}]{bar}[/]")
        console.print(yoy_t)

    console.rule()


def _print_plain(m: dict, now: str) -> None:
    sign = "+" if m["pnl"] >= 0 else ""
    div  = "=" * 64
    sep  = "-" * 64

    trend_1d  = m.get("trend_1d", "NEUTRAL")
    eq_raw    = m.get("entry_quality", "NO_DATA")
    bull_1d   = m.get("bull_1d", 0)
    q_emoji   = {"EXCELLENT":"[GREEN]","GOOD":"[YELLOW]","WAIT":"🔵","NO_ENTRY":"[RED]","NO_DATA":"⬜"}
    t1d_emoji = {"BULLISH":"[UP]","BEARISH":"[DOWN]","NEUTRAL":"➡️"}
    reg_lbl   = "BULL BOOST x1.3 🔥" if bull_1d else "NORMAL x1.0"

    print(f"\n{div}")
    print(f"  BTC PAPER TRADER DASHBOARD — {now}")
    print(div)
    print(f"  Virtual Equity   : ${m['equity']:.4f}")
    print(f"  P&L              : {sign}${m['pnl']:.4f}  ({sign}{m['pnl_pct']:.2f}%)")
    print(f"  Drawdown         : {m['dd']:.2f}%")
    print(f"  CAGR (ann.)      : {m['cagr']:+.2f}%")
    print(f"  Sharpe           : {m['sharpe']:.4f}")
    print(f"  Sortino          : {m['sortino']:.4f}")
    print(f"  Calmar           : {m['calmar']:.4f}   <- metric utama")
    print(sep)
    print(f"  Position         : {POS_LBL.get(m['position'], '—')}")
    print(f"  Kill Switch      : {TIER_LBL.get(m['tier'], '—')}")
    print(f"  Win Rate         : {m['win_rate']:.1f}%")
    print(f"  Total Trades     : {m['total_tr']}")
    print(f"  Total Bars       : {m['total_bars']}")
    print(f"  LONG  signals    : {m['sig_long']}")
    print(f"  SHORT signals    : {m['sig_short']}")
    print(f"  Tier2 bars       : {m['t2_bars']}")
    print(sep)
    print(f"  === V5 UPGRADE ===")
    print(f"  Regime (V5)      : {reg_lbl}")
    print(f"  1D Trend         : {t1d_emoji.get(trend_1d,'')} {trend_1d}")
    print(f"  Bull bars (V5)   : {m.get('regime_bull_bars',0)}")
    print(f"  15m Entry        : {q_emoji.get(eq_raw,'⬜')} {eq_raw}")
    print(sep)
    print(f"  Best bar         : {m['best_bar']:+.3f}%")
    print(f"  Worst bar        : {m['worst_bar']:+.3f}%")
    print(f"  Last bar at      : {m['last_ts']}")
    print(f"  Started at       : {m['started_at']}")

    print(f"\n  Recent 10 Bars:")
    has_1d = "trend_1d" in m["recent"].columns
    hdr = f"  {'Time':<18} {'Sig':<6} {'Pos':>3} {'Price':>10} {'Equity':>10} {'BarRet':>8}"
    if has_1d: hdr += f" {'1D':>8}"
    print(hdr)
    print(f"  {sep}")
    for _, row in m["recent"].iterrows():
        ts  = str(row["timestamp"])[:16] if pd.notna(row["timestamp"]) else "—"
        line= (f"  {ts:<18} {str(row.get('signal','')):<6} "
               f"{int(row.get('position',0)):>3} "
               f"${float(row.get('price',0)):>9,.0f} "
               f"${float(row.get('equity',0)):>9.4f} "
               f"{float(row.get('bar_return_pct',0)):>+7.3f}%")
        if has_1d:
            line += f" {str(row.get('trend_1d',''))[:4]:>8}"
        print(line)

    if m["yoy"]:
        print(f"\n  Year-by-Year Return:")
        print(f"  {'Year':<6} {'Return':>10}  Visual")
        print(f"  {'-'*42}")
        for yr, ret in sorted(m["yoy"].items()):
            bar = ("█" * min(int(abs(ret) / 20), 22)) if ret >= 0 \
                  else ("▒" * min(int(abs(ret) / 20), 22))
            print(f"  {yr:<6} {ret:>+9.1f}%  {bar}")

    print(div)


def print_dashboard(m: dict) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if RICH:
        _print_rich(m, now)
    else:
        _print_plain(m, now)


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BTC Paper Trader Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh:
  python paper_dashboard.py            # tampilkan sekali
  python paper_dashboard.py --watch    # auto-refresh tiap 60 detik
        """,
    )
    parser.add_argument("--watch", action="store_true",
                        help="Auto-refresh dashboard tiap 60 detik")
    args = parser.parse_args()

    def show():
        state = load_state()
        df    = load_log()
        if len(df) == 0:
            print("[WARN]  Belum ada data.")
            print("   Jalankan dulu: python paper_trader.py --backfill")
            return
        m = compute_summary(df, state)
        if m:
            print_dashboard(m)

    if args.watch:
        print("Dashboard watch mode — Ctrl+C untuk berhenti.")
        while True:
            show()
            time.sleep(60)
    else:
        show()


if __name__ == "__main__":
    main()
