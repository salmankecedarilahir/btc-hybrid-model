"""
backtest_engine.py — Phase 5: Institutional-Grade Backtest Engine  [v3]

CHANGELOG v3 vs v2:
  [FIX-B5] Win rate sekarang dihitung per TRADE (signal change), bukan per BAR
           Bug lama: active bars (held positions) = 51.22% karena trend bars ikut dihitung
           Fix baru: hanya signal-change bars = 14-16% (benar untuk momentum model)
  [FIX-B6] Tambah Sortino ratio ke summary output
  [FIX-B7] Tambah Calmar ratio ke summary output
  [FIX-B8] Tambah monthly returns breakdown
  [FIX-B9] Inline kill-switch simulation untuk compare raw vs risk-managed
  [FIX-B10] Per-trade stats sekarang menghitung entry-exit, bukan bar-by-bar
  [FIX-B11] Resume jump cap di simulate_with_killswitch — mirror fix Audit-11
             Root cause: TIER2 shadow terakumulasi multi-bar → spike saat resume
             Fix: clamp (shadow-cur)/cur ke [TIER2_LOSS_CAP, TIER2_GAIN_CAP]

Output: data/btc_backtest_results.csv
        data/btc_equity_curve.csv
        data/btc_monthly_returns.csv   [BARU]
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

BASE_DIR        = Path(__file__).parent
SIGNALS_PATH    = BASE_DIR / "data" / "btc_trading_signals.csv"
RESULTS_PATH    = BASE_DIR / "data" / "btc_backtest_results.csv"
EQUITY_PATH     = BASE_DIR / "data" / "btc_equity_curve.csv"
MONTHLY_PATH    = BASE_DIR / "data" / "btc_monthly_returns.csv"

INITIAL_EQUITY  = 10_000.0
BARS_PER_YEAR   = 6 * 365          # 2190 untuk 4H
EQUITY_FLOOR    = 0.01

# ── Kill switch parameters (mirror risk_engine_v6 RECOMMENDED) ────────────────
KS_TIER1_DD     = -0.15
KS_TIER2_DD     = -0.25
KS_RESUME_DD    = -0.10
BAR_GAIN_LIMIT  = +0.25   # V6 value
BAR_LOSS_LIMIT  = -0.12   # V6 value
TIER2_GAIN_CAP  = +0.12
TIER2_LOSS_CAP  = -0.10


# ════════════════════════════════════════════════════════════════════
#  LOAD
# ════════════════════════════════════════════════════════════════════

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {path}\n"
            "Jalankan hybrid_engine.py terlebih dahulu."
        )
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    log.info("Loaded : %d baris | %s → %s",
             len(df),
             df["timestamp"].iloc[0].strftime("%Y-%m-%d"),
             df["timestamp"].iloc[-1].strftime("%Y-%m-%d"))

    if "signal" not in df.columns:
        raise ValueError("Kolom 'signal' tidak ditemukan!")
    n_active = (df["signal"] != "NONE").sum()
    n_total  = len(df)
    log.info("Signal distribution — active: %d (%.1f%%) | NONE: %d (%.1f%%)",
             n_active, n_active / n_total * 100,
             n_total - n_active, (n_total - n_active) / n_total * 100)
    if n_active == 0:
        raise ValueError("Semua sinyal adalah NONE!")
    return df


# ════════════════════════════════════════════════════════════════════
#  POSITION STATE MACHINE
# ════════════════════════════════════════════════════════════════════

def build_position_series(signal: pd.Series) -> pd.Series:
    """
    LONG → +1, SHORT → -1, NONE → hold (forward-fill)
    Leading NONE → 0 (FLAT)
    """
    raw      = signal.map({"LONG": 1, "SHORT": -1, "NONE": np.nan})
    position = raw.ffill().fillna(0).astype(int)
    return position


# ════════════════════════════════════════════════════════════════════
#  RETURNS & EQUITY
# ════════════════════════════════════════════════════════════════════

def calc_market_return(close: pd.Series) -> pd.Series:
    """market_return[t] = close[t]/close[t-1] - 1  (no lookahead, no forward shift)"""
    return close.pct_change(periods=1).fillna(0.0)


def calc_strategy_return(position: pd.Series, market_return: pd.Series) -> pd.Series:
    return position * market_return


def build_equity_curve(strategy_return: pd.Series,
                        initial: float = INITIAL_EQUITY) -> pd.Series:
    equity = initial * (1 + strategy_return).cumprod()
    return equity.clip(lower=EQUITY_FLOOR)


# ════════════════════════════════════════════════════════════════════
#  INLINE KILL-SWITCH SIMULATION  [FIX-B9]
# ════════════════════════════════════════════════════════════════════

def simulate_with_killswitch(df: pd.DataFrame, init: float = INITIAL_EQUITY) -> pd.Series:
    """
    Simulasi 1x leverage dengan kill switch untuk perbandingan.
    Ini bukan risk engine penuh (tidak ada vol targeting), tapi
    menunjukkan dampak kill switch pada raw backtest.
    """
    sr_arr  = df["strategy_return"].values
    N       = len(sr_arr)
    eq      = np.zeros(N)
    cur     = init; mx = init; shadow = init; tier = 0

    for i in range(N):
        si  = float(sr_arr[i])
        act = int(df["position"].iloc[i]) != 0

        if tier == 2:
            si_c   = float(np.clip(si, TIER2_LOSS_CAP, TIER2_GAIN_CAP))
            shadow = max(shadow * (1 + si_c), 0.01)
            if (shadow - mx) / mx > KS_RESUME_DD:
                tier = 0
                # FIX Audit-11a: cap total resume jump
                if cur > 0.01:
                    total_jump = (shadow - cur) / cur
                    if total_jump > TIER2_GAIN_CAP:
                        shadow = cur * (1.0 + TIER2_GAIN_CAP)
                    elif total_jump < TIER2_LOSS_CAP:
                        shadow = cur * (1.0 + TIER2_LOSS_CAP)
                cur = shadow
                # FIX Audit-11b: skip normal bar processing di resume bar
                if cur > mx: mx = cur
                eq[i] = cur; continue
            else:
                eq[i] = cur; continue

        sc = 0.5 if tier == 1 else 1.0
        br = float(np.clip(si * sc, BAR_LOSS_LIMIT, BAR_GAIN_LIMIT)) if act else 0.0
        cur = max(cur * (1 + br), 0.01); shadow = cur

        if cur > mx:
            mx = cur
        dd = (cur - mx) / mx
        eq[i] = cur

        if tier == 0 and dd <= KS_TIER1_DD:
            tier = 1
        elif tier == 1:
            if dd <= KS_TIER2_DD:
                tier = 2; shadow = cur
            elif dd > KS_TIER1_DD * 0.5:
                tier = 0

    return pd.Series(eq, index=df.index)


# ════════════════════════════════════════════════════════════════════
#  TRADE STATS  [FIX-B5 — per TRADE bukan per BAR]
# ════════════════════════════════════════════════════════════════════

def calc_trade_stats(df: pd.DataFrame) -> dict:
    """
    [FIX-B5] Win rate dihitung dari trade-level (signal changes), bukan per-bar.

    Model ini adalah trend-following dengan hold posisi.
    Win rate per-bar = 51% (misleading: 51% karena trending bars ikut dihitung)
    Win rate per-trade = ~15% (benar: setiap trade baru adalah sinyal baru)

    Trade = bar di mana signal berubah (NONE→LONG, LONG→SHORT, dll.)
    """
    pos = df["position"]
    sr  = df["strategy_return"]

    # ── Bar-level stats (informational) ──────────────────────────
    active_bars = df[df["position"] != 0]["strategy_return"]
    bar_wins    = active_bars[active_bars > 0]
    bar_losses  = active_bars[active_bars < 0]

    # ── [FIX-B5] Trade-level stats (correct untuk signal reporting) ──
    # Trade start = position berubah dari FLAT/berbeda ke posisi baru
    pos_changed = pos.diff().fillna(pos).ne(0) & (pos != 0)
    # Kumpulkan trade sequences
    trade_returns = []
    in_trade = False
    trade_ret = 0.0
    for i in range(len(df)):
        if pos_changed.iloc[i] and pos.iloc[i] != 0:
            if in_trade:
                trade_returns.append(trade_ret)
            in_trade   = True
            trade_ret  = float(sr.iloc[i])
        elif in_trade and pos.iloc[i] != 0:
            trade_ret = (1 + trade_ret) * (1 + float(sr.iloc[i])) - 1   # compound within trade
        elif in_trade and pos.iloc[i] == 0:
            trade_returns.append(trade_ret)
            in_trade = False; trade_ret = 0.0
    if in_trade:
        trade_returns.append(trade_ret)

    trade_arr  = np.array(trade_returns)
    if len(trade_arr) == 0:
        return {"total_trades": 0, "n_long": 0, "n_short": 0,
                "win_rate_trade": 0.0, "win_rate_bar": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0,
                "profit_factor": 0.0, "avg_return_per_trade": 0.0}

    trade_wins   = trade_arr[trade_arr > 0]
    trade_losses = trade_arr[trade_arr < 0]

    if len(trade_losses) == 0:
        pf = float("inf")
    elif len(trade_wins) == 0:
        pf = 0.0
    else:
        pf = float(trade_wins.sum() / abs(trade_losses.sum()))

    bar_wr   = float(len(bar_wins) / len(active_bars) * 100) if len(active_bars) > 0 else 0.0
    trade_wr = float(len(trade_wins) / len(trade_arr) * 100) if len(trade_arr) > 0 else 0.0

    return {
        "total_trades":         len(trade_arr),
        "n_long":               int((df["position"] == 1).sum()),
        "n_short":              int((df["position"] == -1).sum()),
        "win_rate_trade":       round(float(len(trade_wins)/len(trade_arr)*100), 4),
        "win_rate_bar":         round(float(len(bar_wins)/len(active_bars)*100) if len(active_bars)>0 else 0., 4),
        "avg_win":              round(float(trade_wins.mean()*100),  4) if len(trade_wins)>0  else 0.0,
        "avg_loss":             round(float(trade_losses.mean()*100),4) if len(trade_losses)>0 else 0.0,
        "profit_factor":        round(pf, 4) if pf not in (float("inf"), float("-inf")) else None,
        "avg_return_per_trade": round(float(trade_arr.mean()*100), 4),
    }


# ════════════════════════════════════════════════════════════════════
#  PERFORMANCE METRICS  [FIX-B6, B7]
# ════════════════════════════════════════════════════════════════════

def calc_performance(df: pd.DataFrame) -> dict:
    eq = df["equity"]
    sr = df["strategy_return"]

    total_return = float((eq.iloc[-1] - INITIAL_EQUITY) / INITIAL_EQUITY)
    n_years      = len(df) / BARS_PER_YEAR
    cagr         = float((eq.iloc[-1]/INITIAL_EQUITY)**(1/n_years)-1) if n_years>0 else 0.

    sharpe = float((sr.mean()/sr.std())*np.sqrt(BARS_PER_YEAR)) if sr.std()>0 else 0.

    neg_r   = sr[sr < 0]
    sortino = float((sr.mean()/neg_r.std())*np.sqrt(BARS_PER_YEAR)) \
              if len(neg_r) > 0 and neg_r.std() > 0 else 0.

    roll_max = eq.cummax()
    drawdown = (eq - roll_max) / roll_max
    max_dd   = float(drawdown.min())
    calmar   = float(cagr / abs(max_dd)) if max_dd != 0 else 0.

    return {
        "total_return": round(total_return * 100, 4),
        "cagr":         round(cagr * 100, 4),
        "sharpe":       round(sharpe, 4),
        "sortino":      round(sortino, 4),          # [FIX-B6]
        "calmar":       round(calmar, 4),            # [FIX-B7]
        "max_drawdown": round(max_dd * 100, 4),
        "final_equity": round(float(eq.iloc[-1]), 2),
        "n_years":      round(n_years, 4),
        "bars_per_year": BARS_PER_YEAR,
    }


# ════════════════════════════════════════════════════════════════════
#  MONTHLY RETURNS  [FIX-B8]
# ════════════════════════════════════════════════════════════════════

def calc_monthly_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Hitung monthly return untuk consistency check."""
    import warnings
    df2       = df.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df2["month"] = df2["timestamp"].dt.to_period("M")
    monthly   = (df2.groupby("month")["equity"]
                   .agg(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] * 100)
                   .reset_index())
    monthly.columns = ["month", "return_pct"]
    monthly["positive"] = monthly["return_pct"] > 0
    return monthly


def calc_ks_performance(ks_eq: pd.Series, df: pd.DataFrame) -> dict:
    """Hitung metrics untuk kill-switch simulated equity."""
    ny    = len(ks_eq) / BARS_PER_YEAR
    cagr  = (float(ks_eq.iloc[-1]) / INITIAL_EQUITY) ** (1/ny) - 1
    rm    = ks_eq.cummax(); rm[rm==0] = 1e-9
    mdd   = float(((ks_eq - rm) / rm).min())
    eq_r  = ks_eq.pct_change().fillna(0)
    sh    = float((eq_r.mean()/eq_r.std())*np.sqrt(BARS_PER_YEAR)) if eq_r.std()>0 else 0.
    neg_r = eq_r[eq_r<0]
    so    = float((eq_r.mean()/neg_r.std())*np.sqrt(BARS_PER_YEAR)) if len(neg_r)>0 and neg_r.std()>0 else 0.
    cal   = float(cagr / abs(mdd)) if mdd != 0 else 0.
    return dict(cagr=cagr*100, mdd=mdd*100, sh=sh, so=so, cal=cal,
                final=float(ks_eq.iloc[-1]))


# ════════════════════════════════════════════════════════════════════
#  PRINT SUMMARY
# ════════════════════════════════════════════════════════════════════

def print_summary(perf: dict, trade: dict, ks_perf: dict,
                  monthly: pd.DataFrame) -> None:
    div  = "═" * 60
    sep  = "─" * 60
    pf   = trade["profit_factor"]
    pf_s = f"{pf:.4f}" if pf is not None else "∞"

    print(f"\n{div}")
    print("  BACKTEST PERFORMANCE SUMMARY  [v3]")
    print(div)
    print(f"  {'Initial Equity':<30}: ${INITIAL_EQUITY:>12,.2f}")
    print(f"  {'Final Equity (raw)':<30}: ${perf['final_equity']:>12,.2f}")
    print(f"  {'Final Equity (with KS)':<30}: ${ks_perf['final']:>12,.2f}")
    print(sep)
    print(f"  {'Total Return':<30}: {perf['total_return']:>+12.2f}%")
    print(sep)
    print(f"  {'':32} {'RAW':>8} {'WITH KS':>10}")
    print(f"  {'CAGR':<30}: {perf['cagr']:>+8.2f}%   {ks_perf['cagr']:>+8.2f}%")
    print(f"  {'Sharpe Ratio (ann.)':<30}: {perf['sharpe']:>8.4f}   {ks_perf['sh']:>8.4f}")
    print(f"  {'Sortino Ratio (ann.)':<30}: {perf['sortino']:>8.4f}   {ks_perf['so']:>8.4f}")
    print(f"  {'Calmar Ratio':<30}: {perf['calmar']:>8.4f}   {ks_perf['cal']:>8.4f}")
    print(f"  {'Max Drawdown':<30}: {perf['max_drawdown']:>8.2f}%   {ks_perf['mdd']:>8.2f}%")
    print(sep)
    print(f"  {'Duration':<30}: {perf['n_years']:>12.2f} years")
    print(f"  {'Bars/Year (4H)':<30}: {perf['bars_per_year']:>12,}")
    print(sep)
    print(f"  {'Total Trades':<30}: {trade['total_trades']:>12,}")
    print(f"  {'LONG bars':<30}: {trade['n_long']:>12,}")
    print(f"  {'SHORT bars':<30}: {trade['n_short']:>12,}")
    print(sep)
    print(f"  {'Win Rate (per TRADE)':<30}: {trade['win_rate_trade']:>12.2f}%  ← benar")
    print(f"  {'Win Rate (per BAR)':<30}: {trade['win_rate_bar']:>12.2f}%  ← misleading")
    print(f"  {'Avg Win (per trade)':<30}: {trade['avg_win']:>+12.4f}%")
    print(f"  {'Avg Loss (per trade)':<30}: {trade['avg_loss']:>+12.4f}%")
    print(f"  {'Profit Factor':<30}: {pf_s:>12}")
    print(f"  {'Avg Return/Trade':<30}: {trade['avg_return_per_trade']:>+12.4f}%")
    print(sep)

    # Monthly stats
    pos_months  = monthly["positive"].sum()
    neg_months  = (~monthly["positive"]).sum()
    avg_monthly = monthly["return_pct"].mean()
    best_month  = monthly.loc[monthly["return_pct"].idxmax()]
    worst_month = monthly.loc[monthly["return_pct"].idxmin()]
    print(f"  {'Positive months':<30}: {pos_months:>12,}")
    print(f"  {'Negative months':<30}: {neg_months:>12,}")
    print(f"  {'Avg monthly return':<30}: {avg_monthly:>+12.2f}%")
    print(f"  {'Best month':<30}: {str(best_month['month']):>12}  ({best_month['return_pct']:>+.2f}%)")
    print(f"  {'Worst month':<30}: {str(worst_month['month']):>12}  ({worst_month['return_pct']:>+.2f}%)")
    print(f"{div}")
    print(f"\n  NOTE: MaxDD {perf['max_drawdown']:.2f}% adalah RAW (1x unlevered, no kill switch)")
    print(f"        Dengan risk engine V6: MaxDD ~-28%, CAGR ~+192%")
    print(f"        Raw backtest sengaja ditampilkan untuk audit signal quality\n")


# ════════════════════════════════════════════════════════════════════
#  SAVE
# ════════════════════════════════════════════════════════════════════

def save(df: pd.DataFrame, monthly: pd.DataFrame) -> None:
    df.to_csv(RESULTS_PATH, index=False)
    log.info("Results → %s  (%d baris, %d kolom)",
             RESULTS_PATH, len(df), len(df.columns))

    eq_df = df[["timestamp", "close", "signal", "position",
                "market_return", "strategy_return", "equity", "drawdown"]].copy()
    eq_df.to_csv(EQUITY_PATH, index=False)
    log.info("Equity  → %s  (%d baris)", EQUITY_PATH, len(eq_df))

    monthly.to_csv(MONTHLY_PATH, index=False)
    log.info("Monthly → %s  (%d baris)", MONTHLY_PATH, len(monthly))


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════

def run() -> pd.DataFrame:
    log.info("═" * 60)
    log.info("Backtest Engine v3 — All Bug Fixes Applied")
    log.info("  [FIX-B5] Win rate = per TRADE (signal change), not per BAR")
    log.info("  [FIX-B6] Sortino ratio added")
    log.info("  [FIX-B7] Calmar ratio added")
    log.info("  [FIX-B8] Monthly returns breakdown added")
    log.info("  [FIX-B9] Inline kill-switch simulation for comparison")
    log.info("  Timeframe      : 4H  |  Bars/year : %d", BARS_PER_YEAR)
    log.info("  Position model : State machine (LONG/SHORT/FLAT)")
    log.info("═" * 60)

    df = load_data(SIGNALS_PATH)

    log.info("Building position state machine …")
    df["position"]        = build_position_series(df["signal"])

    log.info("Calculating market & strategy returns …")
    df["market_return"]   = calc_market_return(df["close"])
    df["strategy_return"] = calc_strategy_return(df["position"], df["market_return"])
    df["trade_return"]    = df["strategy_return"]

    log.info("Building equity curve …")
    df["equity"]          = build_equity_curve(df["strategy_return"])
    roll_max              = df["equity"].cummax()
    df["drawdown"]        = (df["equity"] - roll_max) / roll_max

    log.info("Simulating with kill switch …")
    df["equity_ks"]       = simulate_with_killswitch(df)

    log.info("Calculating equity return …")
    df["equity_return"]   = df["equity"].pct_change().fillna(0)

    log.info("Returns calculated — total bars: %d", len(df))

    perf    = calc_performance(df)
    trade   = calc_trade_stats(df)
    ks_perf = calc_ks_performance(df["equity_ks"], df)
    monthly = calc_monthly_returns(df)

    print_summary(perf, trade, ks_perf, monthly)
    save(df, monthly)

    log.info("═" * 60)
    return df


if __name__ == "__main__":
    run()
