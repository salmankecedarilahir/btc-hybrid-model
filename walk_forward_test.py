"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          walk_forward_test.py  —  BTC Hybrid Model V7                      ║
║          Walk Forward Analysis — Overfitting Detection                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  TUJUAN:                                                                     ║
║    Deteksi overfitting dengan membagi data menjadi In-Sample (IS) dan       ║
║    Out-of-Sample (OOS) windows secara rolling.                              ║
║                                                                              ║
║    Jika parameter model hanya perform bagus di IS tapi tidak di OOS,        ║
║    model tersebut overfit ke data historis.                                 ║
║                                                                              ║
║  DUA MODE WFA:                                                               ║
║    1. ANCHORED WFA:  training selalu mulai dari tanggal sama (expanding)    ║
║       → Cocok jika model cumulative (semakin banyak data = semakin baik)   ║
║    2. ROLLING WFA:   training window geser bersama test window              ║
║       → Lebih realistik untuk non-stationary market data                   ║
║    Keduanya dijalankan dan WFA Efficiency dihitung untuk masing-masing.     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  WFA EFFICIENCY RATIO:                                                       ║
║    WFA_Eff = mean(OOS CAGR) / mean(IS CAGR)                                ║
║    Interpretasi:                                                             ║
║      > 0.70 : EXCELLENT — parameter sangat robust                          ║
║      0.50-0.70: GOOD    — acceptable degradation                            ║
║      0.30-0.50: FAIR    — overfitting moderate, perlu improvement           ║
║      < 0.30 : POOR     — significant overfitting, jangan lanjut ke AI      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  OUTPUT:                                                                     ║
║    Terminal: per-window stats + aggregate + WFA efficiency                 ║
║    data/wfa_results.csv     — per-window detailed results                  ║
║    data/wfa_report.png      — 4-panel visualization                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

CARA PAKAI:
  python walk_forward_test.py                    # default 24M train / 6M test
  python walk_forward_test.py --train 18 --test 3  # custom windows
  python walk_forward_test.py --mode rolling     # rolling WFA only
  python walk_forward_test.py --mode anchored    # anchored WFA only
  python walk_forward_test.py --no-plot          # skip chart
  python walk_forward_test.py --detail           # verbose per-window output
"""

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
BACKTEST_PATH = BASE_DIR / "data" / "btc_backtest_results.csv"
WFA_PATH      = BASE_DIR / "data" / "wfa_results.csv"
CHART_PATH    = BASE_DIR / "data" / "wfa_report.png"

# ─── defaults ─────────────────────────────────────────────────────────────────
DEFAULT_TRAIN_MONTHS = 24
DEFAULT_TEST_MONTHS  = 6
DEFAULT_STEP_MONTHS  = 3
MIN_TEST_TRADES      = 3      # minimum trade events per OOS window untuk valid
BARS_PER_YEAR        = 2190


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path: Path) -> pd.DataFrame:
    """Load dan validasi backtest results."""
    if not path.exists():
        log.error("File tidak ditemukan: %s", path)
        log.error("Jalankan backtest_engine.py terlebih dahulu.")
        raise SystemExit(1)

    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    log.info("Data loaded: %d baris | %s → %s",
             len(df),
             df["timestamp"].iloc[0].strftime("%Y-%m-%d"),
             df["timestamp"].iloc[-1].strftime("%Y-%m-%d"))

    required = ["position", "strategy_return", "market_return", "signal", "close"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  WINDOW METRICS CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

def calc_window_metrics(df_window: pd.DataFrame, init: float = 10_000.0) -> dict:
    """
    Hitung performance metrics untuk satu window data.

    Returns:
        dict metrics atau None jika window tidak valid
    """
    if len(df_window) == 0:
        return None

    pos = df_window["position"].values
    sr  = df_window["strategy_return"].values
    sig = df_window["signal"].values

    # ── Equity curve ─────────────────────────────────────────
    eq   = init * np.cumprod(1.0 + sr)
    eq   = np.maximum(eq, 0.01)
    peak = np.maximum.accumulate(eq)
    peak = np.where(peak > 0, peak, 1e-9)
    dd   = (eq - peak) / peak

    # ── CAGR ─────────────────────────────────────────────────
    n_bars  = len(sr)
    n_years = n_bars / BARS_PER_YEAR
    final   = float(eq[-1])
    cagr    = (final / init) ** (1.0 / max(n_years, 0.01)) - 1.0

    # ── Sharpe ───────────────────────────────────────────────
    sr_s    = pd.Series(sr)
    sharpe  = float((sr_s.mean() / sr_s.std()) * np.sqrt(BARS_PER_YEAR)) \
              if sr_s.std() > 0 else 0.0

    # ── Sortino ──────────────────────────────────────────────
    neg_r   = sr_s[sr_s < 0]
    sortino = float((sr_s.mean() / neg_r.std()) * np.sqrt(BARS_PER_YEAR)) \
              if (len(neg_r) > 0 and neg_r.std() > 0) else 0.0

    # ── Max DD ────────────────────────────────────────────────
    max_dd  = float(dd.min())

    # ── Calmar ───────────────────────────────────────────────
    calmar  = float(cagr / abs(max_dd)) if max_dd != 0 else 0.0

    # ── Trade-level stats ────────────────────────────────────
    pos_s   = pd.Series(pos)
    sig_chg = (pos_s.diff().fillna(pos_s).ne(0)) & (pos_s != 0)
    trade_rets = sr[sig_chg.values]
    n_trades   = len(trade_rets)
    win_rate   = float((trade_rets > 0).mean() * 100) if n_trades > 0 else 0.0

    wins   = trade_rets[trade_rets > 0]
    losses = trade_rets[trade_rets < 0]
    pf     = float(wins.sum() / abs(losses.sum())) \
             if (len(losses) > 0 and losses.sum() != 0) else 999.0
    pf     = min(pf, 99.0)

    # ── Active bars ──────────────────────────────────────────
    active_bars  = int((pos != 0).sum())
    active_pct   = active_bars / max(n_bars, 1) * 100

    return {
        "n_bars":       n_bars,
        "n_years":      round(n_years, 3),
        "n_trades":     n_trades,
        "final_equity": round(final, 2),
        "cagr":         round(cagr * 100, 4),
        "sharpe":       round(sharpe, 4),
        "sortino":      round(sortino, 4),
        "calmar":       round(calmar, 4),
        "max_drawdown": round(max_dd * 100, 4),
        "win_rate":     round(win_rate, 2),
        "profit_factor": round(pf, 4),
        "active_pct":   round(active_pct, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  IS PERFORMANCE CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

def calc_is_performance(df_train: pd.DataFrame) -> dict:
    """
    Hitung In-Sample performance.
    IS = performance model pada data training (dengan parameter saat ini).
    """
    return calc_window_metrics(df_train) or {}


# ══════════════════════════════════════════════════════════════════════════════
#  WINDOW GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_windows(df: pd.DataFrame,
                     train_months: int,
                     test_months: int,
                     step_months: int,
                     mode: str = "rolling") -> list:
    """
    Generate list of (train_start, train_end, test_end) tuples.

    mode = "rolling"  : train window bergeser bersama test
    mode = "anchored" : train selalu mulai dari awal data (expanding window)
    """
    from dateutil.relativedelta import relativedelta

    t_min   = df["timestamp"].min()
    t_max   = df["timestamp"].max()
    windows = []
    train_start = t_min

    while True:
        train_end = train_start + relativedelta(months=train_months)
        test_end  = train_end  + relativedelta(months=test_months)

        if test_end > t_max:
            break

        # Check enough data in training window
        n_train = len(df[(df["timestamp"] >= train_start) & (df["timestamp"] < train_end)])
        n_test  = len(df[(df["timestamp"] >= train_end)  & (df["timestamp"] < test_end)])

        if n_train > 0 and n_test > 0:
            windows.append({
                "train_start": train_start,
                "train_end":   train_end,
                "test_end":    test_end,
            })

        if mode == "anchored":
            train_start = t_min
            train_start = train_start + relativedelta(months=step_months)
        else:
            # Rolling: seluruh window maju
            train_start = train_start + relativedelta(months=step_months)

    return windows


# ══════════════════════════════════════════════════════════════════════════════
#  SIMPLIFIED WINDOW GENERATOR (lebih reliable)
# ══════════════════════════════════════════════════════════════════════════════

def generate_windows_v2(df: pd.DataFrame,
                         train_months: int,
                         test_months: int,
                         step_months: int,
                         mode: str = "rolling") -> list:
    """
    Reliable window generator menggunakan year-month arithmetic.
    """
    ts = df["timestamp"]
    t_min  = ts.min()
    t_max  = ts.max()

    windows    = []
    step_num   = 0
    max_steps  = 200  # safety

    for _ in range(max_steps):
        from dateutil.relativedelta import relativedelta

        offset_months = step_num * step_months

        if mode == "rolling":
            train_start = t_min + relativedelta(months=offset_months)
        else:  # anchored
            train_start = t_min

        train_end = train_start + relativedelta(months=train_months)
        test_end  = train_end   + relativedelta(months=test_months)

        if test_end > t_max:
            break

        df_train = df[(ts >= train_start) & (ts < train_end)]
        df_test  = df[(ts >= train_end)   & (ts < test_end)]

        if len(df_train) > 50 and len(df_test) > 20:
            windows.append({
                "win_id":      len(windows) + 1,
                "mode":        mode,
                "train_start": train_start,
                "train_end":   train_end,
                "test_end":    test_end,
                "df_train":    df_train.copy(),
                "df_test":     df_test.copy(),
            })

        step_num += 1

        # Anchored: no need to increment forever if train_start fixed
        if mode == "anchored" and step_num > (
                int((t_max - t_min).days / 30 / step_months) + 1):
            break

    return windows


# ══════════════════════════════════════════════════════════════════════════════
#  RUN WFA
# ══════════════════════════════════════════════════════════════════════════════

def run_wfa(df: pd.DataFrame,
             train_months: int = DEFAULT_TRAIN_MONTHS,
             test_months: int  = DEFAULT_TEST_MONTHS,
             step_months: int  = DEFAULT_STEP_MONTHS,
             mode: str         = "rolling",
             detail: bool      = False) -> pd.DataFrame:
    """
    Jalankan Walk Forward Analysis.

    Returns:
        DataFrame dengan IS/OOS metrics per window
    """
    log.info("─" * 65)
    log.info("Walk Forward Analysis — %s mode", mode.upper())
    log.info("  Train: %d months | Test: %d months | Step: %d months",
             train_months, test_months, step_months)
    log.info("─" * 65)

    windows = generate_windows_v2(df, train_months, test_months, step_months, mode)

    if not windows:
        log.error("Tidak ada windows yang valid! Cek ukuran dataset.")
        return pd.DataFrame()

    log.info("Total windows: %d", len(windows))

    rows = []
    for w in windows:
        win_id = w["win_id"]
        df_tr  = w["df_train"]
        df_te  = w["df_test"]

        # IS metrics
        is_m = calc_window_metrics(df_tr)
        # OOS metrics
        oos_m = calc_window_metrics(df_te)

        if is_m is None or oos_m is None:
            continue

        # Check min trades in OOS
        if oos_m.get("n_trades", 0) < MIN_TEST_TRADES:
            log.warning("Window %d: OOS trades=%d < min=%d, skip",
                        win_id, oos_m.get("n_trades", 0), MIN_TEST_TRADES)

        row = {
            "win_id":          win_id,
            "mode":            mode,
            "train_start":     w["train_start"].strftime("%Y-%m-%d"),
            "train_end":       w["train_end"].strftime("%Y-%m-%d"),
            "test_end":        w["test_end"].strftime("%Y-%m-%d"),
            # IS
            "is_cagr":         is_m.get("cagr", 0),
            "is_sharpe":       is_m.get("sharpe", 0),
            "is_sortino":      is_m.get("sortino", 0),
            "is_max_dd":       is_m.get("max_drawdown", 0),
            "is_calmar":       is_m.get("calmar", 0),
            "is_pf":           is_m.get("profit_factor", 0),
            "is_win_rate":     is_m.get("win_rate", 0),
            "is_n_trades":     is_m.get("n_trades", 0),
            # OOS
            "oos_cagr":        oos_m.get("cagr", 0),
            "oos_sharpe":      oos_m.get("sharpe", 0),
            "oos_sortino":     oos_m.get("sortino", 0),
            "oos_max_dd":      oos_m.get("max_drawdown", 0),
            "oos_calmar":      oos_m.get("calmar", 0),
            "oos_pf":          oos_m.get("profit_factor", 0),
            "oos_win_rate":    oos_m.get("win_rate", 0),
            "oos_n_trades":    oos_m.get("n_trades", 0),
            "oos_n_bars":      oos_m.get("n_bars", 0),
        }
        # WFA efficiency ratio per window
        row["wfa_eff_cagr"]   = (row["oos_cagr"] / max(abs(row["is_cagr"]), 0.1)
                                  if row["is_cagr"] != 0 else 0.0)
        row["oos_positive"]   = row["oos_cagr"] > 0

        rows.append(row)

        if detail:
            icon = "✓" if row["oos_positive"] else "✗"
            log.info("[W%02d] %s Train:%s→%s | IS CAGR %+.1f%% Sharpe %.3f | "
                     "OOS CAGR %+.1f%% Sharpe %.3f Trades %d  %s",
                     win_id, mode.upper(),
                     row["train_start"], row["train_end"],
                     row["is_cagr"], row["is_sharpe"],
                     row["oos_cagr"], row["oos_sharpe"],
                     row["oos_n_trades"], icon)

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  AGGREGATE STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def calc_aggregate(results_df: pd.DataFrame) -> dict:
    """Hitung aggregate statistics dari semua OOS windows."""
    if len(results_df) == 0:
        return {}

    r = results_df

    agg = {
        "n_windows":         len(r),
        "n_windows_oos_pos": int(r["oos_positive"].sum()),
        "pct_windows_pos":   float(r["oos_positive"].mean() * 100),
        # OOS aggregate
        "oos_cagr_mean":     float(r["oos_cagr"].mean()),
        "oos_cagr_std":      float(r["oos_cagr"].std()),
        "oos_cagr_min":      float(r["oos_cagr"].min()),
        "oos_cagr_max":      float(r["oos_cagr"].max()),
        "oos_sharpe_mean":   float(r["oos_sharpe"].mean()),
        "oos_sharpe_std":    float(r["oos_sharpe"].std()),
        "oos_sortino_mean":  float(r["oos_sortino"].mean()),
        "oos_max_dd_mean":   float(r["oos_max_dd"].mean()),
        "oos_max_dd_worst":  float(r["oos_max_dd"].min()),
        "oos_pf_mean":       float(r["oos_pf"].mean()),
        "oos_win_rate_mean": float(r["oos_win_rate"].mean()),
        # IS aggregate
        "is_cagr_mean":      float(r["is_cagr"].mean()),
        "is_sharpe_mean":    float(r["is_sharpe"].mean()),
        # WFA efficiency
        "wfa_eff_mean":      float(r["wfa_eff_cagr"].mean()),
        "wfa_eff_std":       float(r["wfa_eff_cagr"].std()),
    }
    return agg


# ══════════════════════════════════════════════════════════════════════════════
#  WFA EFFICIENCY INTERPRETATION
# ══════════════════════════════════════════════════════════════════════════════

def interpret_wfa(agg: dict) -> list:
    """
    Evaluasi apakah model robust berdasarkan WFA results.

    Returns: list of (test, passed, detail)
    """
    tests = []

    def add(name, passed, detail):
        tests.append((name, passed, detail))

    e = agg.get("wfa_eff_mean", 0)
    if e >= 0.70:
        eff_label = "EXCELLENT"
    elif e >= 0.50:
        eff_label = "GOOD"
    elif e >= 0.30:
        eff_label = "FAIR"
    else:
        eff_label = "POOR — OVERFITTING DETECTED"

    add("WFA Efficiency > 0.50",
        e >= 0.50,
        f"Efficiency = {e:.3f}  ({eff_label})")

    add("% OOS Positive Windows > 55%",
        agg.get("pct_windows_pos", 0) > 55,
        f"{agg.get('pct_windows_pos',0):.1f}% windows positive  "
        f"({agg.get('n_windows_oos_pos',0)}/{agg.get('n_windows',0)})")

    add("Mean OOS CAGR > 20%",
        agg.get("oos_cagr_mean", 0) > 20,
        f"Mean OOS CAGR = {agg.get('oos_cagr_mean',0):+.2f}%")

    add("Mean OOS Sharpe > 0.7",
        agg.get("oos_sharpe_mean", 0) > 0.7,
        f"Mean OOS Sharpe = {agg.get('oos_sharpe_mean',0):.4f}")

    add("Mean OOS Max DD > -35%",
        agg.get("oos_max_dd_mean", 0) > -35,
        f"Mean OOS DD = {agg.get('oos_max_dd_mean',0):.2f}%")

    add("OOS CAGR std not too high (<50)",
        agg.get("oos_cagr_std", 999) < 50,
        f"CAGR std = {agg.get('oos_cagr_std',0):.2f}% (lower = more stable)")

    return tests


# ══════════════════════════════════════════════════════════════════════════════
#  PRINTING
# ══════════════════════════════════════════════════════════════════════════════

def print_wfa_table(results_df: pd.DataFrame, mode: str) -> None:
    """Print per-window table."""
    div = "═" * 90
    sep = "─" * 90
    print(f"\n{div}")
    print(f"  WALK FORWARD ANALYSIS — {mode.upper()}")
    print(div)
    print(f"  {'Win':>3} {'Train Start':>11} {'Train End':>11} {'Test End':>11} "
          f"{'IS CAGR':>9} {'IS Sh':>7} {'OOS CAGR':>10} {'OOS Sh':>7} {'OOS DD':>8} "
          f"{'OOS PF':>7} {'Trades':>7}  {''}  ")
    print(sep)
    for _, row in results_df.iterrows():
        icon = "✓" if row["oos_positive"] else "✗"
        print(f"  {int(row['win_id']):>3}  {row['train_start']:>11}  {row['train_end']:>11}  "
              f"{row['test_end']:>11} "
              f"{row['is_cagr']:>+8.1f}% {row['is_sharpe']:>7.3f} "
              f"{row['oos_cagr']:>+9.1f}% {row['oos_sharpe']:>7.3f} "
              f"{row['oos_max_dd']:>7.1f}% "
              f"{row['oos_pf']:>7.3f} {int(row['oos_n_trades']):>7}  {icon}")
    print(f"{div}\n")


def print_aggregate(agg: dict, tests: list, mode: str) -> None:
    div = "═" * 65
    sep = "─" * 65
    print(f"{div}")
    print(f"  AGGREGATE STATISTICS — {mode.upper()}")
    print(div)
    print(f"  Windows analyzed          : {agg.get('n_windows',0)}")
    print(f"  OOS profitable windows    : {agg.get('n_windows_oos_pos',0)}  "
          f"({agg.get('pct_windows_pos',0):.1f}%)")
    print(sep)
    print(f"  {'':34} {'IS':>10} {'OOS':>10}")
    print(f"  {'CAGR (mean)':34} {agg.get('is_cagr_mean',0):>+9.2f}% {agg.get('oos_cagr_mean',0):>+9.2f}%")
    print(f"  {'Sharpe (mean)':34} {agg.get('is_sharpe_mean',0):>10.4f} {agg.get('oos_sharpe_mean',0):>10.4f}")
    print(f"  {'Sortino (mean, OOS)':34} {'':>10} {agg.get('oos_sortino_mean',0):>10.4f}")
    print(f"  {'Max DD (mean, OOS)':34} {'':>10} {agg.get('oos_max_dd_mean',0):>9.2f}%")
    print(f"  {'Profit Factor (mean, OOS)':34} {'':>10} {agg.get('oos_pf_mean',0):>10.4f}")
    print(sep)

    eff = agg.get("wfa_eff_mean", 0)
    if eff >= 0.70:   eff_color = "EXCELLENT"
    elif eff >= 0.50: eff_color = "GOOD"
    elif eff >= 0.30: eff_color = "FAIR — needs work"
    else:             eff_color = "POOR — OVERFITTING"
    print(f"  WFA EFFICIENCY RATIO      : {eff:.4f}  [{eff_color}]")
    print(sep)
    print(f"  PASS / FAIL EVALUATION:")
    print(sep)
    all_pass = True
    for name, passed, detail in tests:
        icon = "✓" if passed else "✗"
        all_pass = all_pass and passed
        note = "" if passed else "  ← ACTION NEEDED"
        print(f"  {icon}  {name:<38}  {detail}{note}")
    print(sep)
    verdict = "✓ ROBUST — Aman lanjut ke Phase B" if all_pass else "[WARN] NEEDS WORK — Fix sebelum AI layer"
    print(f"  VERDICT: {verdict}")
    print(f"{div}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def save_charts(res_rolling: pd.DataFrame,
                res_anchored: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        BG    = "#0D1117"; PAN   = "#161B22"; BORDER = "#30363D"
        GREEN = "#3FB950"; BLUE  = "#58A6FF"; RED    = "#F85149"
        ORANGE= "#E3B341"; TEXT  = "#E6EDF3"; MUTED  = "#8B949E"

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.patch.set_facecolor(BG)
        fig.suptitle("Walk Forward Analysis — BTC Hybrid Model V7",
                     color=TEXT, fontsize=13, fontweight="bold")

        def style_ax(ax, title):
            ax.set_facecolor(PAN)
            ax.set_title(title, color=TEXT, fontsize=9, fontweight="bold")
            ax.tick_params(colors=MUTED, labelsize=7)
            for s in ["bottom","left"]:
                ax.spines[s].set_color(BORDER)
            for s in ["top","right"]:
                ax.spines[s].set_visible(False)

        for datasets, label in [(res_rolling, "Rolling"), (res_anchored, "Anchored")]:
            row = 0 if label == "Rolling" else 1

            # Panel left: IS vs OOS CAGR
            ax = axes[row][0]
            if len(datasets) > 0:
                x = datasets["win_id"].values
                ax.plot(x, datasets["is_cagr"].values,  "o-", color=BLUE,
                        lw=1.5, ms=5, label="IS CAGR", alpha=0.9)
                ax.plot(x, datasets["oos_cagr"].values, "s-", color=GREEN,
                        lw=1.5, ms=5, label="OOS CAGR", alpha=0.9)
                ax.axhline(0, color=RED, lw=1, linestyle="--")
                ax.fill_between(x, datasets["oos_cagr"].values, 0,
                                where=(datasets["oos_cagr"] > 0),
                                alpha=0.15, color=GREEN)
                ax.fill_between(x, datasets["oos_cagr"].values, 0,
                                where=(datasets["oos_cagr"] <= 0),
                                alpha=0.15, color=RED)
                ax.set_xlabel("Window ID", color=MUTED, fontsize=8)
                ax.set_ylabel("CAGR (%)", color=MUTED, fontsize=8)
                ax.legend(fontsize=7, facecolor=PAN, labelcolor=TEXT)
            style_ax(ax, f"{label} WFA — IS vs OOS CAGR per Window")

            # Panel right: OOS Sharpe + DD
            ax  = axes[row][1]
            ax2 = ax.twinx()
            if len(datasets) > 0:
                x = datasets["win_id"].values
                ax.bar(x, datasets["oos_sharpe"].values, color=BLUE, alpha=0.7, label="OOS Sharpe")
                ax2.plot(x, datasets["oos_max_dd"].values, "o-", color=RED,
                         lw=1.5, ms=5, label="OOS Max DD")
                ax.axhline(1.0, color=GREEN, lw=1, linestyle="--")
                ax.set_xlabel("Window ID", color=MUTED, fontsize=8)
                ax.set_ylabel("OOS Sharpe", color=MUTED, fontsize=8)
                ax2.set_ylabel("OOS Max DD (%)", color=MUTED, fontsize=8)
                ax.tick_params(colors=MUTED, labelsize=7)
                ax2.tick_params(colors=MUTED, labelsize=7)
                ax2.spines["right"].set_color(BORDER)
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2,
                          fontsize=7, facecolor=PAN, labelcolor=TEXT)
            style_ax(ax, f"{label} WFA — OOS Sharpe & Drawdown per Window")

        plt.tight_layout(pad=2)
        plt.savefig(CHART_PATH, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close()
        log.info("Chart saved → %s", CHART_PATH)

    except ImportError as e:
        log.warning("matplotlib tidak tersedia: %s", e)
    except Exception as e:
        log.warning("Chart generation gagal: %s", e)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(train_months: int = DEFAULT_TRAIN_MONTHS,
         test_months: int  = DEFAULT_TEST_MONTHS,
         step_months: int  = DEFAULT_STEP_MONTHS,
         mode: str         = "both",
         detail: bool      = False,
         no_plot: bool     = False) -> dict:

    df = load_data(BACKTEST_PATH)

    results_all = {}
    res_rolling  = pd.DataFrame()
    res_anchored = pd.DataFrame()

    # ── Rolling WFA ───────────────────────────────────────────
    if mode in ("rolling", "both"):
        res_rolling = run_wfa(df, train_months, test_months, step_months,
                               mode="rolling", detail=detail)
        if len(res_rolling) > 0:
            print_wfa_table(res_rolling, "rolling")
            agg_r = calc_aggregate(res_rolling)
            tests_r = interpret_wfa(agg_r)
            print_aggregate(agg_r, tests_r, "rolling")
            results_all["rolling"] = {"results": res_rolling, "agg": agg_r}

    # ── Anchored WFA ──────────────────────────────────────────
    if mode in ("anchored", "both"):
        res_anchored = run_wfa(df, train_months, test_months, step_months,
                                mode="anchored", detail=detail)
        if len(res_anchored) > 0:
            print_wfa_table(res_anchored, "anchored")
            agg_a = calc_aggregate(res_anchored)
            tests_a = interpret_wfa(agg_a)
            print_aggregate(agg_a, tests_a, "anchored")
            results_all["anchored"] = {"results": res_anchored, "agg": agg_a}

    # ── Save ──────────────────────────────────────────────────
    combined = pd.concat([res_rolling, res_anchored], ignore_index=True)
    if len(combined) > 0:
        combined.to_csv(WFA_PATH, index=False)
        log.info("WFA results saved → %s  (%d rows)", WFA_PATH, len(combined))

    # ── Charts ────────────────────────────────────────────────
    if not no_plot:
        save_charts(res_rolling, res_anchored)

    # ── Final verdict ─────────────────────────────────────────
    if results_all:
        print("═" * 65)
        print("  FINAL WFA VERDICT — SUMMARY")
        print("─" * 65)
        for m, r in results_all.items():
            agg = r["agg"]
            eff = agg.get("wfa_eff_mean", 0)
            pos = agg.get("pct_windows_pos", 0)
            print(f"  {m.upper():<12}  WFA Eff: {eff:.3f}  "
                  f"OOS+ windows: {pos:.0f}%  "
                  f"Avg OOS CAGR: {agg.get('oos_cagr_mean',0):+.1f}%  "
                  f"Avg OOS Sh: {agg.get('oos_sharpe_mean',0):.3f}")
        print("═" * 65 + "\n")

    return results_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walk Forward Analysis — BTC Hybrid Model")
    parser.add_argument("--train",    type=int, default=DEFAULT_TRAIN_MONTHS,
                        help=f"Training window months (default: {DEFAULT_TRAIN_MONTHS})")
    parser.add_argument("--test",     type=int, default=DEFAULT_TEST_MONTHS,
                        help=f"Test window months (default: {DEFAULT_TEST_MONTHS})")
    parser.add_argument("--step",     type=int, default=DEFAULT_STEP_MONTHS,
                        help=f"Step size months (default: {DEFAULT_STEP_MONTHS})")
    parser.add_argument("--mode",     choices=["rolling","anchored","both"], default="both",
                        help="WFA mode (default: both)")
    parser.add_argument("--detail",   action="store_true",
                        help="Tampilkan detail per-window di terminal")
    parser.add_argument("--no-plot",  action="store_true",
                        help="Skip chart generation")
    args = parser.parse_args()

    main(train_months=args.train, test_months=args.test,
         step_months=args.step, mode=args.mode,
         detail=args.detail, no_plot=args.no_plot)
