"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          monte_carlo_simulation.py  —  BTC Hybrid Model V7                 ║
║          Robustness Testing via Bootstrap Resampling                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  TUJUAN:                                                                     ║
║    Mensimulasikan 10,000 possible futures dari distribusi return historis.  ║
║    Menjawab pertanyaan kritis:                                               ║
║      ✦ Seberapa buruk worst-case drawdown yang mungkin terjadi?             ║
║      ✦ Apakah edge model konsisten atau hanya luck?                         ║
║      ✦ Berapa probability of ruin (DD > threshold)?                         ║
║      ✦ Seberapa stabil ekspektansi return?                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  DUA MODE SIMULASI:                                                          ║
║    1. BAR-LEVEL MC:  randomisasi urutan return per-4H-bar (20,000 obs)      ║
║       → Fokus pada distribusi bar returns yang dialami model                ║
║    2. TRADE-LEVEL MC: randomisasi urutan trade penuh (85 trades)            ║
║       → Fokus pada apakah sekuens trade bisa lebih buruk                   ║
║    Keduanya dijalankan dan dibandingkan.                                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  OUTPUT:                                                                     ║
║    Terminal: full statistics + PASS/FAIL verdict                            ║
║    data/mc_results_bar.csv      — bar-level simulation results              ║
║    data/mc_results_trade.csv    — trade-level simulation results            ║
║    data/mc_report.png           — 6-panel visualization                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

CARA PAKAI:
  python monte_carlo_simulation.py               # default 10,000 sims
  python monte_carlo_simulation.py -n 50000      # custom N simulasi
  python monte_carlo_simulation.py --no-plot     # skip chart
  python monte_carlo_simulation.py --bar-only    # hanya bar-level MC
  python monte_carlo_simulation.py --trade-only  # hanya trade-level MC
  python monte_carlo_simulation.py --ruin 0.40   # ruin threshold 40%
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
BACKTEST_PATH   = BASE_DIR / "data" / "btc_backtest_results.csv"
MC_BAR_PATH     = BASE_DIR / "data" / "mc_results_bar.csv"
MC_TRADE_PATH   = BASE_DIR / "data" / "mc_results_trade.csv"
CHART_PATH      = BASE_DIR / "data" / "mc_report.png"

# ─── defaults ─────────────────────────────────────────────────────────────────
DEFAULT_N_SIM       = 10_000
DEFAULT_INIT        = 10_000.0
DEFAULT_RUIN_THRESH = 0.50      # ruin = DD > -50%
BARS_PER_YEAR       = 2190


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & TRADE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def load_bar_returns(path: Path) -> np.ndarray:
    """
    Load bar-level strategy returns dari backtest results.
    Filter: hanya bars di mana position != 0 DAN return != 0.
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    active = df[(df["position"] != 0) & (df["strategy_return"] != 0)]["strategy_return"].values
    log.info("Bar returns loaded: %d bars (dari %d total, %d active)",
             len(active), len(df), (df["position"] != 0).sum())
    return active


def extract_trade_returns(path: Path) -> tuple:
    """
    Ekstrak trade-level returns (entry → exit).
    Trade = segment di mana position held dari signal change ke signal change berikutnya.

    Returns:
        trade_returns: array of cumulative returns per trade
        trade_meta: list of dicts with trade metadata
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    pos   = df["position"].values
    sr    = df["strategy_return"].values
    sig   = df["signal"].values
    ts    = df["timestamp"].values
    close = df["close"].values

    trades     = []
    in_trade   = False
    trade_ret  = 0.0
    t_entry    = None
    p_sig      = None

    for i in range(len(df)):
        if not in_trade:
            if pos[i] != 0:
                in_trade  = True
                trade_ret = float(sr[i])
                t_entry   = ts[i]
                p_sig     = sig[i]
        else:
            if pos[i] == 0:
                trades.append({"signal": p_sig, "ret": trade_ret,
                                "entry": t_entry, "exit": ts[i]})
                in_trade = False; trade_ret = 0.0
            elif pd.notna(sig[i]) and sig[i] != "NONE" and sig[i] != p_sig:
                trades.append({"signal": p_sig, "ret": trade_ret,
                                "entry": t_entry, "exit": ts[i]})
                in_trade = True; trade_ret = float(sr[i])
                t_entry = ts[i]; p_sig = sig[i]
            else:
                trade_ret = (1 + trade_ret) * (1 + float(sr[i])) - 1

    if in_trade:
        trades.append({"signal": p_sig, "ret": trade_ret,
                        "entry": t_entry, "exit": ts[-1]})

    trade_arr = np.array([t["ret"] for t in trades])
    log.info("Trades extracted: %d (Win rate: %.1f%%  PF: %.4f)",
             len(trade_arr),
             (trade_arr > 0).mean() * 100,
             trade_arr[trade_arr > 0].sum() / max(abs(trade_arr[trade_arr < 0].sum()), 1e-10))
    return trade_arr, trades


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE SIMULATION (vectorized)
# ══════════════════════════════════════════════════════════════════════════════

def run_single_sim(returns: np.ndarray, init: float, rng: np.random.Generator) -> dict:
    """
    Jalankan satu simulasi bootstrap:
      1. Resample returns dengan replacement (same length)
      2. Build equity curve
      3. Hitung metrik
    """
    # Bootstrap resample
    shuffled = rng.choice(returns, size=len(returns), replace=True)

    # Equity curve
    eq   = init * np.cumprod(1.0 + shuffled)
    peak = np.maximum.accumulate(eq)
    peak[peak == 0] = 1e-9
    dd   = (eq - peak) / peak

    # Metrics
    n_years = len(shuffled) / BARS_PER_YEAR
    final   = float(eq[-1])
    cagr    = (final / init) ** (1.0 / max(n_years, 0.01)) - 1.0
    max_dd  = float(dd.min())

    eq_ret = np.diff(eq, prepend=init) / np.where(np.concatenate([[init], eq[:-1]]) > 0, np.concatenate([[init], eq[:-1]]), 1.0)
    er     = eq_ret
    sharpe = float((er.mean() / er.std()) * np.sqrt(BARS_PER_YEAR)) if er.std() > 0 else 0.0

    neg_r  = er[er < 0]
    sortino = float((er.mean() / neg_r.std()) * np.sqrt(BARS_PER_YEAR)) \
              if (len(neg_r) > 0 and neg_r.std() > 0) else 0.0

    wins   = shuffled[shuffled > 0]
    losses = shuffled[shuffled < 0]
    pf     = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 else 999.0

    return {
        "final_equity": final,
        "cagr":         cagr * 100,
        "max_drawdown": max_dd * 100,
        "sharpe":       sharpe,
        "sortino":      sortino,
        "profit_factor": min(pf, 99.0),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MONTE CARLO ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def run_monte_carlo(returns: np.ndarray,
                    n_sim: int          = DEFAULT_N_SIM,
                    init: float         = DEFAULT_INIT,
                    ruin_thresh: float  = DEFAULT_RUIN_THRESH,
                    label: str          = "BAR-LEVEL",
                    seed: int           = 42) -> dict:
    """
    Jalankan N simulasi bootstrap dan kumpulkan distribusi metrics.

    Returns:
        results dict dengan statistik distribusi semua metrics
    """
    rng    = np.random.default_rng(seed)
    log.info("Menjalankan %s Monte Carlo: N=%d, init=$%.0f, ruin=%.0f%%",
             label, n_sim, init, ruin_thresh * 100)

    t0 = time.time()

    # ── Batch vectorized simulation ────────────────────────────
    # Build semua simulasi sekaligus untuk speed
    N   = len(returns)
    idx = rng.integers(0, N, size=(n_sim, N))    # shape: (n_sim, N)
    sims = returns[idx]                            # shape: (n_sim, N) — bootstrap

    # Equity curves (vectorized)
    log.info("Building %d equity curves (vectorized)...", n_sim)
    eq_matrix = init * np.cumprod(1.0 + sims, axis=1)    # shape: (n_sim, N)

    # Max drawdown per simulation
    peaks     = np.maximum.accumulate(eq_matrix, axis=1)
    peaks     = np.where(peaks > 0, peaks, 1e-9)
    dd_matrix = (eq_matrix - peaks) / peaks
    max_dds   = dd_matrix.min(axis=1)                     # shape: (n_sim,)

    # Final equity
    finals    = eq_matrix[:, -1]                          # shape: (n_sim,)

    # CAGR — clip finals dulu agar tidak overflow saat power calculation
    n_years   = N / BARS_PER_YEAR
    finals_c  = np.clip(finals, 0.01, init * 1e8)        # max 10 juta kali return
    with np.errstate(over="ignore", invalid="ignore"):
        cagrs = ((finals_c / init) ** (1.0 / max(n_years, 0.01)) - 1.0) * 100
    cagrs = np.clip(cagrs, -100.0, 100_000.0)            # clip display range

    # Sharpe (approximation dari bar returns)
    bar_means = sims.mean(axis=1)
    bar_stds  = sims.std(axis=1)
    sharpes   = np.where(bar_stds > 0,
                         (bar_means / bar_stds) * np.sqrt(BARS_PER_YEAR),
                         0.0)

    # Profit factor
    wins_sum  = np.where(sims > 0, sims, 0.0).sum(axis=1)
    loss_sum  = np.abs(np.where(sims < 0, sims, 0.0).sum(axis=1))
    pfs       = np.where(loss_sum > 0, wins_sum / loss_sum, 99.0)
    pfs       = np.clip(pfs, 0, 99)

    elapsed   = time.time() - t0
    log.info("Simulasi selesai dalam %.2f detik", elapsed)

    # ── Statistics ────────────────────────────────────────────
    def pct(arr, p): return float(np.percentile(arr, p))

    ruin_pct   = float((max_dds < -ruin_thresh).mean() * 100)
    pos_final  = float((finals > init).mean() * 100)

    results = {
        "label":       label,
        "n_sim":       n_sim,
        "n_bars":      N,
        "ruin_thresh": ruin_thresh * 100,
        "elapsed_s":   round(elapsed, 2),
        # Final equity
        "eq_p5":       pct(finals, 5),
        "eq_p25":      pct(finals, 25),
        "eq_median":   pct(finals, 50),
        "eq_p75":      pct(finals, 75),
        "eq_p95":      pct(finals, 95),
        "eq_pct_positive": pos_final,
        # Max drawdown
        "dd_worst":    pct(max_dds, 0) * 100,
        "dd_p5":       pct(max_dds, 5) * 100,
        "dd_p25":      pct(max_dds, 25) * 100,
        "dd_median":   pct(max_dds, 50) * 100,
        "dd_p75":      pct(max_dds, 75) * 100,
        "dd_best":     pct(max_dds, 100) * 100,
        # CAGR
        "cagr_p5":     pct(cagrs, 5),
        "cagr_p25":    pct(cagrs, 25),
        "cagr_median": pct(cagrs, 50),
        "cagr_p75":    pct(cagrs, 75),
        "cagr_p95":    pct(cagrs, 95),
        # Sharpe
        "sharpe_p5":   pct(sharpes, 5),
        "sharpe_median": pct(sharpes, 50),
        "sharpe_p95":  pct(sharpes, 95),
        # PF
        "pf_p5":       pct(pfs, 5),
        "pf_median":   pct(pfs, 50),
        "pf_p95":      pct(pfs, 95),
        # Risk of ruin
        "ruin_pct":    ruin_pct,
        # Raw arrays for plotting
        "_finals":     finals,
        "_max_dds":    max_dds * 100,
        "_cagrs":      cagrs,
        "_sharpes":    sharpes,
        "_pfs":        pfs,
    }
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  PASS/FAIL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_results(results: dict, init: float = DEFAULT_INIT) -> list:
    """
    Evaluasi hasil MC terhadap institutional criteria.

    Returns: list of (test_name, passed, details)
    """
    tests = []

    def add(name, passed, detail):
        tests.append((name, passed, detail))

    r = results

    add("Risk of Ruin < 5%",
        r["ruin_pct"] < 5.0,
        f"P(DD > {r['ruin_thresh']:.0f}%) = {r['ruin_pct']:.2f}%")

    add("Median Equity > Initial",
        r["eq_median"] > init,
        f"Median final = ${r['eq_median']:,.0f}  (init ${init:,.0f})")

    add("5th Pct Equity > 50% Initial",
        r["eq_p5"] > init * 0.5,
        f"5th pct final = ${r['eq_p5']:,.0f}")

    add("Worst 5% Drawdown > -40%",
        r["dd_p5"] > -40.0,
        f"5th pct max DD = {r['dd_p5']:.1f}%")

    add("Median CAGR > 20%",
        r["cagr_median"] > 20.0,
        f"Median CAGR = {r['cagr_median']:+.1f}%")

    add("Median Sharpe > 0.8",
        r["sharpe_median"] > 0.8,
        f"Median Sharpe = {r['sharpe_median']:.3f}")

    add("Median PF > 1.1",
        r["pf_median"] > 1.1,
        f"Median PF = {r['pf_median']:.3f}")

    add("% Simulations Profitable > 60%",
        r["eq_pct_positive"] > 60.0,
        f"{r['eq_pct_positive']:.1f}% sims profitable")

    return tests


# ══════════════════════════════════════════════════════════════════════════════
#  PRINTING
# ══════════════════════════════════════════════════════════════════════════════

def print_results(results: dict, tests: list, init: float = DEFAULT_INIT) -> None:
    r   = results
    div = "═" * 68
    sep = "─" * 68

    print(f"\n{div}")
    print(f"  MONTE CARLO SIMULATION — {r['label']}")
    print(f"  N = {r['n_sim']:,}  |  Bars per sim = {r['n_bars']:,}  |  Elapsed = {r['elapsed_s']}s")
    print(div)

    print(f"\n  {'FINAL EQUITY DISTRIBUTION':}")
    print(f"  {'':4} {'5th pct':>12} {'25th pct':>12} {'Median':>12} {'75th pct':>12} {'95th pct':>12}")
    print(f"  {'':4} {r['eq_p5']:>12,.0f} {r['eq_p25']:>12,.0f} {r['eq_median']:>12,.0f} {r['eq_p75']:>12,.0f} {r['eq_p95']:>12,.0f}")
    print(f"  Pct of sims profitable: {r['eq_pct_positive']:.1f}%")

    print(f"\n  {'MAX DRAWDOWN DISTRIBUTION'}")
    print(f"  {'':4} {'Worst':>10} {'5th pct':>10} {'25th pct':>10} {'Median':>10} {'Best':>10}")
    print(f"  {'':4} {r['dd_worst']:>10.2f}% {r['dd_p5']:>10.2f}% {r['dd_p25']:>10.2f}% {r['dd_median']:>10.2f}% {r['dd_best']:>10.2f}%")

    print(f"\n  {'CAGR DISTRIBUTION'}")
    print(f"  {'':4} {'5th pct':>10} {'25th pct':>10} {'Median':>10} {'75th pct':>10} {'95th pct':>10}")
    def _fmt_cagr(v):
        """Format CAGR, truncate if overflow."""
        if abs(v) > 999999: return f"+{'∞':>9}"
        return f"{v:>+10.1f}%"
    print(f"  {'':4} {_fmt_cagr(r['cagr_p5'])} {_fmt_cagr(r['cagr_p25'])} {_fmt_cagr(r['cagr_median'])} {_fmt_cagr(r['cagr_p75'])} {_fmt_cagr(r['cagr_p95'])}")

    print(f"\n  {'SHARPE / SORTINO / PF'}")
    print(f"  Sharpe  — p5: {r['sharpe_p5']:.3f}  median: {r['sharpe_median']:.3f}  p95: {r['sharpe_p95']:.3f}")
    print(f"  PF      — p5: {r['pf_p5']:.3f}  median: {r['pf_median']:.3f}  p95: {r['pf_p95']:.3f}")

    print(f"\n  {'RISK OF RUIN'}")
    print(f"  P(Max DD > {r['ruin_thresh']:.0f}%) = {r['ruin_pct']:.3f}%  {'✓ PASS' if r['ruin_pct'] < 5 else '✗ FAIL — CRITICAL!'}")

    print(f"\n{sep}")
    print(f"  PASS / FAIL EVALUATION")
    print(sep)
    all_pass = True
    for name, passed, detail in tests:
        icon = "✓" if passed else "✗"
        color_txt = "" if passed else " ← ACTION NEEDED"
        all_pass  = all_pass and passed
        print(f"  {icon}  {name:<38}  {detail}{color_txt}")

    print(sep)
    verdict = "✓ ALL PASS — Model statistik robust" if all_pass else "[WARN] SOME FAIL — Address sebelum AI layer"
    print(f"  VERDICT: {verdict}")
    print(f"{div}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def save_charts(res_bar: dict, res_trade: dict, init: float = DEFAULT_INIT) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        BG    = "#0D1117"; PAN   = "#161B22"; BORDER = "#30363D"
        GREEN = "#3FB950"; BLUE  = "#58A6FF"; RED    = "#F85149"
        ORANGE= "#E3B341"; TEXT  = "#E6EDF3"; MUTED  = "#8B949E"

        fig = plt.figure(figsize=(16, 12), facecolor=BG)
        gs  = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
        fig.suptitle("Monte Carlo Robustness Test — BTC Hybrid Model V7",
                     color=TEXT, fontsize=13, fontweight="bold", y=0.98)

        def style_ax(ax, title):
            ax.set_facecolor(PAN)
            ax.set_title(title, color=TEXT, fontsize=9, fontweight="bold", pad=8)
            ax.tick_params(colors=MUTED, labelsize=7)
            for s in ["bottom","left"]:
                ax.spines[s].set_color(BORDER)
            for s in ["top","right"]:
                ax.spines[s].set_visible(False)
            ax.yaxis.label.set_color(MUTED)
            ax.xaxis.label.set_color(MUTED)

        # ── Panel 1: Final equity distribution (bar-level) ───────
        ax = fig.add_subplot(gs[0, 0])
        finals = res_bar["_finals"]
        ax.hist(finals / 1000, bins=80, color=BLUE, alpha=0.75, edgecolor=BG, linewidth=0.3)
        ax.axvline(init / 1000, color=RED, lw=1.5, linestyle="--", label=f"Initial ${init/1000:.0f}k")
        ax.axvline(np.percentile(finals, 5) / 1000, color=ORANGE, lw=1.2, linestyle=":", label="5th pct")
        ax.axvline(np.median(finals) / 1000, color=GREEN, lw=1.5, linestyle="-", label="Median")
        ax.set_xlabel("Final Equity ($k)"); ax.set_ylabel("Frequency")
        ax.legend(fontsize=7, facecolor=PAN, labelcolor=TEXT)
        style_ax(ax, f"Final Equity Distribution — Bar MC (N={res_bar['n_sim']:,})")

        # ── Panel 2: Max DD distribution (bar-level) ─────────────
        ax = fig.add_subplot(gs[0, 1])
        max_dds = res_bar["_max_dds"]
        ax.hist(max_dds, bins=60, color=RED, alpha=0.75, edgecolor=BG, linewidth=0.3)
        ax.axvline(np.percentile(max_dds, 5), color=ORANGE, lw=1.5, linestyle=":", label="5th pct")
        ax.axvline(np.median(max_dds), color=GREEN, lw=1.5, linestyle="-", label="Median")
        ax.axvline(-50, color=RED, lw=1.5, linestyle="--", label="Ruin -50%")
        ax.set_xlabel("Max Drawdown (%)"); ax.set_ylabel("Frequency")
        ax.legend(fontsize=7, facecolor=PAN, labelcolor=TEXT)
        style_ax(ax, "Max Drawdown Distribution — Bar MC")

        # ── Panel 3: CAGR distribution ───────────────────────────
        ax = fig.add_subplot(gs[1, 0])
        cagrs = res_bar["_cagrs"]
        ax.hist(cagrs, bins=80, color=GREEN, alpha=0.75, edgecolor=BG, linewidth=0.3)
        ax.axvline(0, color=RED, lw=1.5, linestyle="--", label="0%")
        ax.axvline(np.percentile(cagrs, 5), color=ORANGE, lw=1.2, linestyle=":", label="5th pct")
        ax.axvline(np.median(cagrs), color=BLUE, lw=1.5, linestyle="-", label="Median")
        ax.set_xlabel("CAGR (%)"); ax.set_ylabel("Frequency")
        ax.legend(fontsize=7, facecolor=PAN, labelcolor=TEXT)
        style_ax(ax, "CAGR Distribution — Bar MC")

        # ── Panel 4: Sharpe distribution ─────────────────────────
        ax = fig.add_subplot(gs[1, 1])
        sharpes = res_bar["_sharpes"]
        ax.hist(sharpes, bins=60, color=ORANGE, alpha=0.75, edgecolor=BG, linewidth=0.3)
        ax.axvline(0, color=RED, lw=1.5, linestyle="--", label="0")
        ax.axvline(1.0, color=GREEN, lw=1.5, linestyle="-", label="Sharpe=1.0")
        ax.axvline(np.median(sharpes), color=BLUE, lw=1.5, linestyle=":", label="Median")
        ax.set_xlabel("Sharpe Ratio"); ax.set_ylabel("Frequency")
        ax.legend(fontsize=7, facecolor=PAN, labelcolor=TEXT)
        style_ax(ax, "Sharpe Distribution — Bar MC")

        # ── Panel 5: Trade-level final equity ────────────────────
        ax = fig.add_subplot(gs[2, 0])
        if res_trade is not None:
            t_finals = res_trade["_finals"]
            ax.hist(t_finals / 1000, bins=50, color="#F0883E", alpha=0.75, edgecolor=BG, linewidth=0.3)
            ax.axvline(init / 1000, color=RED, lw=1.5, linestyle="--", label=f"Initial")
            ax.axvline(np.median(t_finals) / 1000, color=GREEN, lw=1.5, linestyle="-", label="Median")
            ax.set_xlabel("Final Equity ($k)"); ax.set_ylabel("Frequency")
            ax.legend(fontsize=7, facecolor=PAN, labelcolor=TEXT)
        style_ax(ax, f"Final Equity — Trade MC (N={res_trade['n_sim'] if res_trade else 0:,})")

        # ── Panel 6: Sample equity curves ────────────────────────
        ax = fig.add_subplot(gs[2, 1])
        rng  = np.random.default_rng(99)
        bar_rets = res_bar.get("_bar_returns", None)
        if bar_rets is not None:
            n = len(bar_rets)
            for _ in range(200):
                s  = rng.choice(bar_rets, size=n, replace=True)
                eq = init * np.cumprod(1 + s)
                ax.plot(np.linspace(0, 1, n), eq / 1000, lw=0.3, alpha=0.15, color=BLUE)
            ax.axhline(init / 1000, color=RED, lw=1, linestyle="--")
            ax.set_xlabel("Time (normalized)"); ax.set_ylabel("Equity ($k)")
        style_ax(ax, "200 Sample Equity Paths")

        plt.savefig(CHART_PATH, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close()
        log.info("Chart saved → %s", CHART_PATH)

    except ImportError as e:
        log.warning("matplotlib tidak tersedia: %s", e)
    except Exception as e:
        log.warning("Chart generation gagal: %s", e)


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def save_results(results: dict, path: Path) -> None:
    """Save simulation summary statistics ke CSV."""
    # Exclude private arrays
    data = {k: v for k, v in results.items() if not k.startswith("_")}
    pd.DataFrame([data]).to_csv(path, index=False)
    log.info("Results saved → %s", path)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(n_sim: int      = DEFAULT_N_SIM,
         ruin: float     = DEFAULT_RUIN_THRESH,
         bar_only: bool  = False,
         trade_only: bool = False,
         no_plot: bool   = False) -> dict:

    if not BACKTEST_PATH.exists():
        log.error("File tidak ditemukan: %s", BACKTEST_PATH)
        log.error("Jalankan backtest_engine.py terlebih dahulu.")
        raise SystemExit(1)

    all_results = {}

    # ── Bar-level MC ──────────────────────────────────────────
    if not trade_only:
        bar_returns = load_bar_returns(BACKTEST_PATH)
        res_bar     = run_monte_carlo(bar_returns, n_sim=n_sim,
                                      ruin_thresh=ruin, label="BAR-LEVEL")
        res_bar["_bar_returns"] = bar_returns   # simpan untuk chart
        tests_bar   = evaluate_results(res_bar)
        print_results(res_bar, tests_bar)
        save_results(res_bar, MC_BAR_PATH)
        all_results["bar"] = res_bar
    else:
        res_bar = None

    # ── Trade-level MC ────────────────────────────────────────
    if not bar_only:
        trade_returns, _ = extract_trade_returns(BACKTEST_PATH)
        if len(trade_returns) >= 20:
            res_trade   = run_monte_carlo(trade_returns, n_sim=n_sim,
                                          ruin_thresh=ruin, label="TRADE-LEVEL")
            tests_trade = evaluate_results(res_trade)
            print_results(res_trade, tests_trade)
            save_results(res_trade, MC_TRADE_PATH)
            all_results["trade"] = res_trade
        else:
            log.warning("Trade count (%d) terlalu sedikit untuk Trade-level MC. Min 20.", len(trade_returns))
            res_trade = None
    else:
        res_trade = None

    # ── Charts ────────────────────────────────────────────────
    if not no_plot and res_bar is not None:
        save_charts(res_bar, res_trade)

    # ── Comparison summary ────────────────────────────────────
    if res_bar is not None and res_trade is not None:
        print("\n" + "─" * 68)
        print("  COMPARISON: BAR-LEVEL vs TRADE-LEVEL")
        print("─" * 68)
        for label, r in [("BAR-LEVEL", res_bar), ("TRADE-LEVEL", res_trade)]:
            print(f"  {label:<14}  Median EQ: ${r['eq_median']:>12,.0f}  "
                  f"DD p5: {r['dd_p5']:>7.1f}%  "
                  f"Ruin: {r['ruin_pct']:>5.2f}%  "
                  f"PF: {r['pf_median']:>5.3f}")
        print("─" * 68 + "\n")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo Simulation — BTC Hybrid Model")
    parser.add_argument("-n", "--n-sim", type=int, default=DEFAULT_N_SIM,
                        help=f"Jumlah simulasi (default: {DEFAULT_N_SIM:,})")
    parser.add_argument("--ruin", type=float, default=DEFAULT_RUIN_THRESH,
                        help=f"Ruin threshold DD% (default: {DEFAULT_RUIN_THRESH})")
    parser.add_argument("--bar-only",   action="store_true", help="Hanya bar-level MC")
    parser.add_argument("--trade-only", action="store_true", help="Hanya trade-level MC")
    parser.add_argument("--no-plot",    action="store_true", help="Skip chart generation")
    args = parser.parse_args()

    main(n_sim=args.n_sim, ruin=args.ruin,
         bar_only=args.bar_only, trade_only=args.trade_only,
         no_plot=args.no_plot)
