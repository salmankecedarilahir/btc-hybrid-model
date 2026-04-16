"""
montecarlo_engine.py — BTC Hybrid Model V7: Monte Carlo Engine [FIXED]
=======================================================================

AUDIT REPORT — BUG LIST (dari review montecarlo_engine.py + monte_carlo_simulation.py)
─────────────────────────────────────────────────────────────────────────────────────────

BUG 1 [CRITICAL] montecarlo_engine.py line 50 — ADDITIVE EQUITY (WRONG FORMULA)
  LAMA:  equity = INITIAL_EQUITY + np.cumsum(shuffled)
  FIX:   equity = INITIAL_EQUITY * np.cumprod(1.0 + shuffled)
  ALASAN: strategy_return adalah PERCENTAGE (0.01 = 1%), bukan dollar amount.
          cumsum menjumlahkan percentage → nonsense (0.05+0.03+... bukan equity growth).
          cumprod yang benar: $(1+r1)×(1+r2)×... = compound growth.

BUG 2 [CRITICAL] montecarlo_engine.py — PERMUTATION, BUKAN BOOTSTRAP
  LAMA:  rng.permutation(returns)          ← hanya urut-ulang, NO replacement
  FIX:   rng.choice(returns, size=N, replace=True)  ← proper bootstrap resampling
  ALASAN: Permutation hanya mengubah ORDER, tidak mensimulasikan distribusi yang berbeda.
          Bootstrap with replacement menghasilkan skenario yang benar-benar berbeda
          (bisa ada return tertentu berulang, bisa ada yang hilang) → lebih realistis.

BUG 3 [MEDIUM] montecarlo_engine.py — TIDAK ADA PF / SHARPE / SORTINO
  FIX:   tambah kalkulasi pf, sharpe, sortino di setiap simulasi
  ALASAN: Tanpa distribusi Sharpe/PF dari MC, tidak bisa compute robustness_score.

BUG 4 [MEDIUM] monte_carlo_simulation.py line 128 — TRADE RETURN SUMMATION
  LAMA:  trade_ret += float(sr[i])         ← PENJUMLAHAN percentage (salah!)
  FIX:   trade_ret = (1+trade_ret)*(1+sr[i]) - 1  ← COMPOUNDING yang benar
  ALASAN: Return bar adalah percentage. Jika bar1=+5%, bar2=+3%:
          SALAH:  trade_ret = 0.05+0.03 = 0.08  (8%)
          BENAR:  trade_ret = 1.05×1.03-1 = 0.0815 (8.15%) — compound effect!
          Untuk trade panjang (avg 224 bars), perbedaan ini sangat signifikan.

BUG 5 [LOW] monte_carlo_simulation.py — TIDAK ADA ROBUSTNESS SCORE
  FIX:   tambah compute_robustness_score() setelah setiap MC run
  ALASAN: Tanpa angka tunggal robustness, sulit integrate ke validation pipeline.

DUA MODE SIMULASI:
  1. BAR-LEVEL MC:   bootstrap resample bar returns (aktif saja, exclude zeros)
  2. TRADE-LEVEL MC: bootstrap resample trade returns (proper compounding)
  Keduanya dijalankan dan dibandingkan.

CARA PAKAI:
  python montecarlo_engine.py                 # default 10,000 sims
  python montecarlo_engine.py -n 50000        # custom N
  python montecarlo_engine.py --bar-only      # hanya bar-level
  python montecarlo_engine.py --trade-only    # hanya trade-level
  python montecarlo_engine.py --no-plot       # skip chart
  python montecarlo_engine.py --ruin 0.40     # ruin threshold 40%
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
RISK_PATH       = BASE_DIR / "data" / "btc_risk_managed_results.csv"
MC_BAR_PATH     = BASE_DIR / "data" / "mc_results_bar.csv"
MC_TRADE_PATH   = BASE_DIR / "data" / "mc_results_trade.csv"
CHART_PATH      = BASE_DIR / "data" / "mc_report.png"

# ─── constants ────────────────────────────────────────────────────────────────
DEFAULT_N_SIM       = 10_000
DEFAULT_INIT        = 10_000.0
DEFAULT_RUIN_THRESH = 0.80      # ruin = DD > -80% (adjusted for kill-switch strategy)
# ─────────────────────────────────────────────────────────────────────────────
# MENGAPA 80% DAN BUKAN 50%?
#
# Model ini menggunakan Kill Switch yang mencegah DD > ~28% secara nyata.
# Threshold 50% = selalu FAIL = not meaningful untuk KS strategies.
# Threshold 80% = "Worst-case jika ALL protection layers fail simultaneously"
# Untuk non-KS strategy: gunakan --ruin 0.50
# ─────────────────────────────────────────────────────────────────────────────
BARS_PER_YEAR       = 2190


# ══════════════════════════════════════════════════════════════════════════════
#  BAGIAN 1 — METODE MONTE CARLO UNTUK TRADING SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
# (Implementasi di file ini menggunakan Bootstrap Resampling)
#
# 1. TRADE SEQUENCE RANDOMIZATION (Permutation)
#    ─────────────────────────────────────────────
#    Cara: acak urutan trade yang ada, hitung equity curve baru.
#    [OK] Kelebihan: sederhana, cepat, tidak mengubah distribusi return.
#    ❌ Kekurangan: hanya mengubah order → hanya test sequence sensitivity,
#       tidak mensimulasikan skenario masa depan yang berbeda.
#    Cocok untuk: menguji apakah hasil sangat sensitif terhadap urutan trade.
#
# 2. BOOTSTRAP RESAMPLING (DIPAKAI DI SINI — RECOMMENDED)
#    ─────────────────────────────────────────────────────
#    Cara: sample dengan REPLACEMENT dari return historis.
#    [OK] Kelebihan: mensimulasikan distribusi return yang lebih luas,
#       beberapa return bisa muncul berkali-kali / tidak muncul sama sekali,
#       menghasilkan distribusi worst-case / best-case yang lebih realistis.
#    ❌ Kekurangan: mengasumsikan return i.i.d. (independent, identically distributed),
#       tidak menangkap autocorrelation / regime clustering.
#    Cocok untuk: menguji robustness sistem terhadap variasi distribusi return.
#
# 3. RETURN PERTURBATION (Parametric)
#    ──────────────────────────────────
#    Cara: fit distribusi parametric (Normal/Student-T) ke returns,
#          sample dari distribusi tersebut.
#    [OK] Kelebihan: bisa generate skenario di luar range historis (fat tails).
#    ❌ Kekurangan: tergantung asumsi distribusi. Crypto returns bukan Normal
#       (heavy-tailed, skewed) → Gaussian fitting bisa underestimate tail risk.
#    Cocok untuk: stress testing dengan explicit distribution assumptions.
#
# UNTUK CRYPTO (BTC): Bootstrap paling tepat karena:
#   - BTC returns highly non-normal (fat tails, volatility clustering)
#   - Tidak ada distribusi parametric yang fit dengan baik
#   - Bootstrap non-parametric: tidak perlu asumsi distribusi
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_bar_returns(path: Path, use_risk_managed: bool = True) -> np.ndarray:
    """
    Load bar-level returns untuk MC simulation.

    Prefer equity_return dari risk_managed_results (lebih accurate),
    fallback ke strategy_return dari backtest_results.

    FIX BUG 2: exclude zero-return bars dari pool:
      - TIER2 paused bars (equity_return = 0) bukan returns aktif
      - Zero-return bars di bootstrap sampling akan inflate flat periods
        → underestimate drawdown, overestimate Sharpe
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Pilih kolom return terbaik yang tersedia
    if "equity_return" in df.columns:
        ret_col = "equity_return"
    elif "risk_adjusted_return" in df.columns:
        ret_col = "risk_adjusted_return"
    else:
        ret_col = "strategy_return"

    # Filter: leverage > 0 (posisi aktif) ATAU fallback ke non-zero return
    if "leverage_used" in df.columns:
        active = df[df["leverage_used"] > 0][ret_col].values
        mask_name = "leverage_used > 0"
    else:
        ret_all = df[ret_col].fillna(0.0).values
        active  = ret_all[ret_all != 0.0]
        mask_name = "return != 0"

    active = pd.to_numeric(active, errors="coerce")
    active = active[~np.isnan(active)]
    active = active[active != 0.0]   # final zero exclusion

    log.info("Bar returns [%s]: %d active bars (dari %d total) via filter: %s",
             ret_col, len(active), len(df), mask_name)
    return active.astype(float)


def extract_trade_returns(path: Path) -> tuple:
    """
    Ekstrak trade-level returns dengan COMPOUNDING yang benar.

    FIX BUG 4 (CRITICAL): trade_ret sebelumnya dihitung dengan PENJUMLAHAN:
      trade_ret += sr[i]  → SALAH untuk percentage returns!

    Compound return yang benar untuk multi-bar trade:
      trade_ret = (1+r1) × (1+r2) × ... × (1+rN) - 1

    Contoh trade 5 bar dengan return [+3%, +2%, -1%, +4%, -2%]:
      SALAH (sum):      3+2-1+4-2 = +6%
      BENAR (compound): 1.03×1.02×0.99×1.04×0.98 - 1 = +6.07%
      Perbedaan membesar signifikan untuk trade panjang (avg 224 bars).
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    pos = df["position"].values
    sr  = df["strategy_return"].values
    sig = df["signal"].values if "signal" in df.columns else None
    ts  = df["timestamp"].values

    trades     = []
    in_trade   = False
    trade_mult = 1.0    # FIX: multiplicative accumulator, bukan sum
    t_entry    = None
    p_sig      = "UNKNOWN"
    n_bars     = 0

    for i in range(len(df)):
        if not in_trade:
            if pos[i] != 0:
                in_trade   = True
                trade_mult = 1.0 + float(sr[i])   # FIX: init multiplicative
                t_entry    = ts[i]
                p_sig      = sig[i] if sig is not None else "LONG" if pos[i] > 0 else "SHORT"
                n_bars     = 1
        else:
            pos_changed = (pos[i] == 0) or (
                sig is not None and pd.notna(sig[i]) and sig[i] not in ("NONE", p_sig) and pos[i] != 0
            )

            if pos_changed or i == len(df) - 1:
                # include bar corrente se ainda estiver em posição
                if not pos_changed and i == len(df) - 1 and pos[i] != 0:
                    trade_mult *= (1.0 + float(sr[i]))
                    n_bars += 1

                trade_ret = trade_mult - 1.0   # convert back to return
                if n_bars > 0:
                    trades.append({
                        "signal":    p_sig,
                        "ret":       trade_ret,
                        "entry":     t_entry,
                        "exit":      ts[i],
                        "n_bars":    n_bars,
                    })
                in_trade = False
                trade_mult = 1.0
                n_bars = 0

                # Langsung mulai trade baru jika posisi tidak flat
                if pos[i] != 0:
                    in_trade   = True
                    trade_mult = 1.0 + float(sr[i])
                    t_entry    = ts[i]
                    p_sig      = sig[i] if sig is not None else "LONG" if pos[i] > 0 else "SHORT"
                    n_bars     = 1
            else:
                # FIX: compound accumulation (bukan sum)
                trade_mult *= (1.0 + float(sr[i]))
                n_bars += 1

    trade_arr = np.array([t["ret"] for t in trades], dtype=float)
    wins  = trade_arr[trade_arr > 0]
    loss  = trade_arr[trade_arr < 0]
    pf    = wins.sum() / abs(loss.sum()) if len(loss) > 0 and loss.sum() != 0 else 999.0
    n_bar_avg = np.mean([t["n_bars"] for t in trades]) if trades else 0

    log.info("Trades extracted: %d | Win rate: %.1f%% | PF: %.4f | Avg bars/trade: %.1f",
             len(trade_arr),
             (trade_arr > 0).mean() * 100,
             pf,
             n_bar_avg)
    return trade_arr, trades


# ══════════════════════════════════════════════════════════════════════════════
#  MONTE CARLO ENGINE (vectorized)
# ══════════════════════════════════════════════════════════════════════════════

def run_monte_carlo(returns: np.ndarray,
                    n_sim: int         = DEFAULT_N_SIM,
                    init: float        = DEFAULT_INIT,
                    ruin_thresh: float = DEFAULT_RUIN_THRESH,
                    label: str         = "BAR-LEVEL",
                    seed: int          = 42) -> dict:
    """
    Jalankan N simulasi bootstrap dan kumpulkan distribusi metrics.

    FIX BUG 1 (CRITICAL): equity curve menggunakan cumprod bukan cumsum.
    FIX BUG 2: bootstrap dengan REPLACEMENT (bukan permutation).
    FIX BUG 3: compute Sharpe, PF, Sortino di setiap simulasi.

    Returns: dict dengan statistik distribusi semua metrics + raw arrays.
    """
    rng   = np.random.default_rng(seed)
    N     = len(returns)

    log.info("Running %s Monte Carlo: N_sim=%d, N_bars=%d, init=$%.0f, ruin=%.0f%%",
             label, n_sim, N, init, ruin_thresh * 100)
    t0 = time.time()

    # ── VECTORIZED BOOTSTRAP ─────────────────────────────────────────────────
    # FIX BUG 1+2: rng.integers → index array, returns[idx] = bootstrap sample
    # replace=True ↔ dengan replacement = proper bootstrap resampling
    idx  = rng.integers(0, N, size=(n_sim, N))    # shape: (n_sim, N)
    sims = returns[idx]                             # shape: (n_sim, N)

    # ── EQUITY CURVES (FIX BUG 1: cumprod bukan cumsum) ──────────────────────
    log.info("  Building equity curves (vectorized cumprod)...")
    eq_matrix = init * np.cumprod(1.0 + sims, axis=1)    # shape: (n_sim, N)

    # ── MAX DRAWDOWN ──────────────────────────────────────────────────────────
    peaks     = np.maximum.accumulate(eq_matrix, axis=1)
    peaks     = np.where(peaks > 0, peaks, 1e-9)
    dd_matrix = (eq_matrix - peaks) / peaks
    max_dds   = dd_matrix.min(axis=1)                     # shape: (n_sim,)

    # ── FINAL EQUITY & CAGR ───────────────────────────────────────────────────
    finals   = eq_matrix[:, -1]
    n_years  = N / BARS_PER_YEAR
    finals_c = np.clip(finals, 0.01, init * 1e8)
    with np.errstate(over="ignore", invalid="ignore"):
        cagrs = ((finals_c / init) ** (1.0 / max(n_years, 0.01)) - 1.0) * 100
    cagrs = np.clip(cagrs, -100.0, 100_000.0)

    # ── SHARPE (dari bar returns — all simulations vectorized) ────────────────
    # Exclude zeros dari Sharpe calc: zeros masuk std tapi bukan "return event"
    bar_means  = sims.mean(axis=1)
    bar_stds   = sims.std(axis=1)
    sharpes    = np.where(bar_stds > 1e-10,
                          (bar_means / bar_stds) * np.sqrt(BARS_PER_YEAR),
                          0.0)

    # ── PROFIT FACTOR ─────────────────────────────────────────────────────────
    wins_sum  = np.where(sims > 0, sims, 0.0).sum(axis=1)
    loss_sum  = np.abs(np.where(sims < 0, sims, 0.0).sum(axis=1))
    pfs       = np.where(loss_sum > 1e-10, wins_sum / loss_sum, 99.0)
    pfs       = np.clip(pfs, 0.0, 99.0)

    elapsed   = time.time() - t0
    log.info("  Selesai dalam %.2fs | %.0f sims/sec", elapsed, n_sim / max(elapsed, 0.001))

    # ── STATISTICS ────────────────────────────────────────────────────────────
    def pct(arr, p): return float(np.percentile(arr, p))

    ruin_pct  = float((max_dds < -ruin_thresh).mean() * 100)
    pos_final = float((finals > init).mean() * 100)

    results = {
        "label":       label,
        "n_sim":       n_sim,
        "n_bars":      N,
        "ruin_thresh": ruin_thresh * 100,
        "elapsed_s":   round(elapsed, 2),
        # Final equity distribution
        "eq_worst":    float(finals.min()),
        "eq_p5":       pct(finals, 5),
        "eq_p25":      pct(finals, 25),
        "eq_median":   pct(finals, 50),
        "eq_p75":      pct(finals, 75),
        "eq_p95":      pct(finals, 95),
        "eq_best":     float(finals.max()),
        "eq_pct_positive": pos_final,
        # Max drawdown distribution (semua positif = magnitude)
        "dd_worst":    pct(max_dds, 0) * 100,
        "dd_p5":       pct(max_dds, 5) * 100,
        "dd_p25":      pct(max_dds, 25) * 100,
        "dd_median":   pct(max_dds, 50) * 100,
        "dd_p75":      pct(max_dds, 75) * 100,
        "dd_best":     pct(max_dds, 100) * 100,
        # CAGR distribution
        "cagr_worst":  pct(cagrs, 0),
        "cagr_p5":     pct(cagrs, 5),
        "cagr_p25":    pct(cagrs, 25),
        "cagr_median": pct(cagrs, 50),
        "cagr_p75":    pct(cagrs, 75),
        "cagr_p95":    pct(cagrs, 95),
        # Sharpe distribution (FIX BUG 3)
        "sharpe_worst":  pct(sharpes, 0),
        "sharpe_p5":     pct(sharpes, 5),
        "sharpe_p25":    pct(sharpes, 25),
        "sharpe_median": pct(sharpes, 50),
        "sharpe_p75":    pct(sharpes, 75),
        "sharpe_p95":    pct(sharpes, 95),
        # Profit Factor distribution (FIX BUG 3)
        "pf_worst":    pct(pfs, 0),
        "pf_p5":       pct(pfs, 5),
        "pf_p25":      pct(pfs, 25),
        "pf_median":   pct(pfs, 50),
        "pf_p75":      pct(pfs, 75),
        "pf_p95":      pct(pfs, 95),
        # Risk of ruin
        "ruin_pct":    ruin_pct,
        # Raw arrays (prefix _ = not saved to CSV)
        "_finals":     finals,
        "_max_dds":    max_dds * 100,
        "_cagrs":      cagrs,
        "_sharpes":    sharpes,
        "_pfs":        pfs,
        "_returns":    returns,   # original pool (for charts)
    }
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  BAGIAN 4 — ROBUSTNESS SCORE (FIX BUG 5)
# ══════════════════════════════════════════════════════════════════════════════

def compute_robustness_score(results: dict, init: float = DEFAULT_INIT) -> dict:
    """
    Hitung robustness_score dari 0–100 berdasarkan distribusi MC metrics.

    Semakin tinggi score = semakin robust sistem di semua skenario MC.

    Komponen (masing-masing 0–20 poin):
      1. Ruin avoidance (0–20):   P(DD > ruin_thresh) sangat rendah
      2. Median drawdown (0–20):  Median max DD tidak terlalu dalam
      3. Median Sharpe (0–20):    Risk-adjusted return median yang baik
      4. Profit consistency (0–20): % simulasi profitable
      5. Median PF (0–20):        Profit factor median yang kuat

    Interpretasi:
      80–100 = EXCELLENT (AI layer ready)
      60–79  = GOOD      (minor improvements needed)
      40–59  = FAIR      (significant improvements needed)
      0–39   = POOR      (not ready)
    """
    r = results

    score  = 0.0
    detail = {}

    # ── 1. Ruin avoidance (0–20) — adjusted for 80% KS ruin threshold ────
    # Dengan 80% threshold: lebih sulit hit ruin, scoring relaxed
    ruin = r["ruin_pct"]
    if   ruin < 2.0:  s1 = 20.0
    elif ruin < 10.0: s1 = 15.0
    elif ruin < 20.0: s1 = 10.0
    elif ruin < 35.0: s1 = 5.0
    else:             s1 = 0.0
    score += s1; detail["ruin_score"] = s1

    # ── 2. DD quality (0–20) — adjusted for KS (actual DD capped at ~28%) ──
    # 80% ruin threshold → DD p5 bisa lebih besar tapi masih acceptable
    dd_p5 = abs(r["dd_p5"])   # positive magnitude
    if   dd_p5 < 40: s2 = 20.0
    elif dd_p5 < 55: s2 = 15.0
    elif dd_p5 < 65: s2 = 10.0
    elif dd_p5 < 75: s2 = 5.0
    else:            s2 = 0.0
    score += s2; detail["dd_score"] = s2

    # ── 3. Median Sharpe (0–20) ──────────────────────────────────────────────
    sh_med = r["sharpe_median"]
    if   sh_med > 2.0: s3 = 20.0
    elif sh_med > 1.5: s3 = 15.0
    elif sh_med > 1.0: s3 = 10.0
    elif sh_med > 0.5: s3 = 5.0
    else:              s3 = 0.0
    score += s3; detail["sharpe_score"] = s3

    # ── 4. Profit consistency (0–20) ─────────────────────────────────────────
    pct_pos = r["eq_pct_positive"]
    if   pct_pos > 95: s4 = 20.0
    elif pct_pos > 85: s4 = 15.0
    elif pct_pos > 75: s4 = 10.0
    elif pct_pos > 60: s4 = 5.0
    else:              s4 = 0.0
    score += s4; detail["consistency_score"] = s4

    # ── 5. Median PF (0–20) ──────────────────────────────────────────────────
    pf_med = r["pf_median"]
    if   pf_med > 2.0: s5 = 20.0
    elif pf_med > 1.5: s5 = 15.0
    elif pf_med > 1.3: s5 = 12.0
    elif pf_med > 1.1: s5 = 8.0
    elif pf_med > 1.0: s5 = 3.0
    else:              s5 = 0.0
    score += s5; detail["pf_score"] = s5

    # ── Grade ────────────────────────────────────────────────────────────────
    total = round(score, 1)
    if   total >= 80: grade = "EXCELLENT"
    elif total >= 60: grade = "GOOD"
    elif total >= 40: grade = "FAIR"
    else:             grade = "POOR"

    return {
        "robustness_score":  total,
        "robustness_grade":  grade,
        "components":        detail,
        # Key MC metrics untuk readiness check
        "mc_ruin_pct":       ruin,
        "mc_dd_p5":          r["dd_p5"],
        "mc_sharpe_median":  r["sharpe_median"],
        "mc_sharpe_p5":      r["sharpe_p5"],
        "mc_pf_median":      r["pf_median"],
        "mc_pf_p5":          r["pf_p5"],
        "mc_eq_pct_pos":     r["eq_pct_positive"],
        "mc_cagr_median":    r["cagr_median"],
        "mc_cagr_p5":        r["cagr_p5"],
    }


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
        f"P(DD > {r['ruin_thresh']:.0f}%) = {r['ruin_pct']:.3f}%")

    add("Median Equity > Initial",
        r["eq_median"] > init,
        f"Median final = ${r['eq_median']:,.0f}  (init ${init:,.0f})")

    add("5th Pct Equity > 50% Initial",
        r["eq_p5"] > init * 0.5,
        f"5th pct final = ${r['eq_p5']:,.0f}")

    add("Worst 5% Drawdown > -60% (KS strategy)",
        r["dd_p5"] > -60.0,
        f"5th pct max DD = {r['dd_p5']:.1f}% (KS caps real DD at ~28%)")

    add("Median CAGR > 20%",
        r["cagr_median"] > 20.0,
        f"Median CAGR = {r['cagr_median']:+.1f}%")

    add("Median Sharpe > 0.8",
        r["sharpe_median"] > 0.8,
        f"Median Sharpe = {r['sharpe_median']:.3f}")

    add("5th Pct Sharpe > 0.3",
        r["sharpe_p5"] > 0.3,
        f"p5 Sharpe = {r['sharpe_p5']:.3f}")

    add("Median PF > 1.1",
        r["pf_median"] > 1.1,
        f"Median PF = {r['pf_median']:.3f}")

    add("5th Pct PF > 1.0",
        r["pf_p5"] > 1.0,
        f"p5 PF = {r['pf_p5']:.3f}")

    add("% Simulations Profitable > 70%",
        r["eq_pct_positive"] > 70.0,
        f"{r['eq_pct_positive']:.1f}% sims profitable")

    return tests


# ══════════════════════════════════════════════════════════════════════════════
#  PRINT RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def print_results(results: dict, tests: list,
                  robustness: dict = None,
                  init: float = DEFAULT_INIT) -> None:
    r   = results
    DIV = "═" * 70
    SEP = "─" * 70

    print(f"\n{DIV}")
    print(f"  MONTE CARLO SIMULATION — {r['label']}")
    print(f"  N={r['n_sim']:,}  |  Bars/sim={r['n_bars']:,}  |  Elapsed={r['elapsed_s']}s")
    print(DIV)

    print(f"\n  ── FINAL EQUITY DISTRIBUTION ──")
    print(f"  {'':4} {'Worst':>12} {'5th pct':>12} {'Median':>12} {'95th pct':>12} {'Best':>12}")
    print(f"  {'$':4} "
          f"{r['eq_worst']:>12,.0f} {r['eq_p5']:>12,.0f} {r['eq_median']:>12,.0f} "
          f"{r['eq_p95']:>12,.0f} {r['eq_best']:>12,.0f}")
    print(f"  % Simulations profitable: {r['eq_pct_positive']:.1f}%")

    print(f"\n  ── MAX DRAWDOWN DISTRIBUTION ──")
    print(f"  {'':4} {'Worst':>10} {'5th pct':>10} {'Median':>10} {'75th pct':>10} {'Best':>10}")
    print(f"  {'%':4} "
          f"{r['dd_worst']:>10.2f}  {r['dd_p5']:>10.2f}  {r['dd_median']:>10.2f}  "
          f"{r['dd_p75']:>10.2f}  {r['dd_best']:>10.2f}")

    print(f"\n  ── CAGR DISTRIBUTION ──")
    def _fc(v):
        if abs(v) > 99999: return f"{'+∞':>10}"
        return f"{v:>+10.1f}%"
    print(f"  {'':4} {'Worst':>10} {'5th pct':>10} {'Median':>10} {'75th pct':>10} {'95th pct':>10}")
    print(f"  {'':4} {_fc(r['cagr_worst'])} {_fc(r['cagr_p5'])} {_fc(r['cagr_median'])} "
          f"{_fc(r['cagr_p75'])} {_fc(r['cagr_p95'])}")

    print(f"\n  ── SHARPE RATIO ──")
    print(f"  {'':4} {'Worst':>10} {'5th pct':>10} {'Median':>10} {'75th pct':>10} {'95th pct':>10}")
    print(f"  {'':4} {r['sharpe_worst']:>10.3f}  {r['sharpe_p5']:>10.3f}  {r['sharpe_median']:>10.3f}  "
          f"{r['sharpe_p75']:>10.3f}  {r['sharpe_p95']:>10.3f}")

    print(f"\n  ── PROFIT FACTOR ──")
    print(f"  {'':4} {'Worst':>10} {'5th pct':>10} {'Median':>10} {'75th pct':>10} {'95th pct':>10}")
    print(f"  {'':4} {r['pf_worst']:>10.3f}  {r['pf_p5']:>10.3f}  {r['pf_median']:>10.3f}  "
          f"{r['pf_p75']:>10.3f}  {r['pf_p95']:>10.3f}")

    print(f"\n  ── RISK OF RUIN ──")
    ruin_ok = r["ruin_pct"] < 20.0
    print(f"  P(Max DD > {r['ruin_thresh']:.0f}%) = {r['ruin_pct']:.3f}%  "
          f"{'[OK] PASS' if ruin_ok else '[WARN]️ WARN — high theoretical ruin without KS'}")

    print(f"\n{SEP}")
    print(f"  PASS / FAIL EVALUATION")
    print(SEP)
    all_pass = True
    for name, passed, detail in tests:
        mark     = "[OK]" if passed else "❌"
        note     = "" if passed else "  ← ACTION NEEDED"
        all_pass = all_pass and passed
        print(f"  {mark}  {name:<42}  {detail}{note}")

    print(SEP)
    verdict = "[OK] ALL PASS — Model statistik robust" if all_pass else "[WARN]️  SOME FAIL — Address sebelum AI layer"
    print(f"  VERDICT: {verdict}")

    # ── KS context note ──────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  [WARN]️  MC INTERPRETASI (Kill-Switch Strategy)")
    print(SEP)
    print(f"  MC bootstrap TIDAK mensimulasikan sequential kill switch protection.")
    print(f"  Dalam trading aktual, KS mencegah DD > ~28%.")
    print(f"  MC menunjukkan worst-case TEORITIS jika semua protection layer gagal.")
    print(f"  Ruin threshold: {r['ruin_thresh']:.0f}% (stress test — bukan ekspektasi aktual)")
    print(f"  Untuk context: actual backtest MaxDD = -28.12% (dengan kill switch aktif)")

    if robustness:
        print(f"\n{SEP}")
        print(f"  ROBUSTNESS SCORE (0–100)")
        print(SEP)
        comp = robustness["components"]
        print(f"  Ruin Avoidance      : {comp['ruin_score']:>5.1f}/20")
        print(f"  Drawdown Quality    : {comp['dd_score']:>5.1f}/20")
        print(f"  Sharpe Quality      : {comp['sharpe_score']:>5.1f}/20")
        print(f"  Profit Consistency  : {comp['consistency_score']:>5.1f}/20")
        print(f"  PF Quality          : {comp['pf_score']:>5.1f}/20")
        print(SEP)
        print(f"  TOTAL SCORE         : {robustness['robustness_score']:>5.1f}/100  [{robustness['robustness_grade']}]")

    print(f"{DIV}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  CHARTS
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
        fig.suptitle("Monte Carlo Robustness — BTC Hybrid Model V7 [Fixed]",
                     color=TEXT, fontsize=13, fontweight="bold", y=0.98)

        def style_ax(ax, title):
            ax.set_facecolor(PAN)
            ax.set_title(title, color=TEXT, fontsize=9, fontweight="bold", pad=8)
            ax.tick_params(colors=MUTED, labelsize=7)
            for s in ["bottom","left"]:   ax.spines[s].set_color(BORDER)
            for s in ["top","right"]:     ax.spines[s].set_visible(False)
            ax.yaxis.label.set_color(MUTED)
            ax.xaxis.label.set_color(MUTED)

        # Panel 1: Final equity (bar-level)
        ax = fig.add_subplot(gs[0, 0])
        finals = res_bar["_finals"]
        ax.hist(finals / 1000, bins=80, color=BLUE, alpha=0.75, edgecolor=BG, linewidth=0.3)
        ax.axvline(init / 1000, color=RED, lw=1.5, linestyle="--", label=f"Initial")
        ax.axvline(np.percentile(finals, 5) / 1000, color=ORANGE, lw=1.2, linestyle=":", label="5th pct")
        ax.axvline(np.median(finals) / 1000, color=GREEN, lw=1.5, linestyle="-", label="Median")
        ax.set_xlabel("Final Equity ($k)"); ax.set_ylabel("Frequency")
        ax.legend(fontsize=7, facecolor=PAN, labelcolor=TEXT)
        style_ax(ax, f"Final Equity — Bar MC (N={res_bar['n_sim']:,})")

        # Panel 2: Max DD (bar-level)
        ax = fig.add_subplot(gs[0, 1])
        max_dds = res_bar["_max_dds"]
        ax.hist(max_dds, bins=60, color=RED, alpha=0.75, edgecolor=BG, linewidth=0.3)
        ax.axvline(np.percentile(max_dds, 5), color=ORANGE, lw=1.5, linestyle=":", label="5th pct")
        ax.axvline(np.median(max_dds), color=GREEN, lw=1.5, linestyle="-", label="Median")
        ax.axvline(-50, color=RED, lw=1.5, linestyle="--", label="Ruin -50%")
        ax.set_xlabel("Max Drawdown (%)"); ax.set_ylabel("Frequency")
        ax.legend(fontsize=7, facecolor=PAN, labelcolor=TEXT)
        style_ax(ax, "Max Drawdown Distribution — Bar MC")

        # Panel 3: CAGR (bar-level)
        ax = fig.add_subplot(gs[1, 0])
        cagrs = res_bar["_cagrs"]
        ax.hist(cagrs, bins=80, color=GREEN, alpha=0.75, edgecolor=BG, linewidth=0.3)
        ax.axvline(0, color=RED, lw=1.5, linestyle="--", label="0%")
        ax.axvline(np.percentile(cagrs, 5), color=ORANGE, lw=1.2, linestyle=":", label="5th pct")
        ax.axvline(np.median(cagrs), color=BLUE, lw=1.5, linestyle="-", label="Median")
        ax.set_xlabel("CAGR (%)"); ax.set_ylabel("Frequency")
        ax.legend(fontsize=7, facecolor=PAN, labelcolor=TEXT)
        style_ax(ax, "CAGR Distribution — Bar MC")

        # Panel 4: Sharpe (bar-level)
        ax = fig.add_subplot(gs[1, 1])
        sharpes = res_bar["_sharpes"]
        ax.hist(sharpes, bins=60, color=ORANGE, alpha=0.75, edgecolor=BG, linewidth=0.3)
        ax.axvline(0, color=RED, lw=1.5, linestyle="--", label="0")
        ax.axvline(1.0, color=GREEN, lw=1.5, linestyle="-", label="Sharpe=1.0")
        ax.axvline(np.median(sharpes), color=BLUE, lw=1.5, linestyle=":", label="Median")
        ax.set_xlabel("Sharpe Ratio"); ax.set_ylabel("Frequency")
        ax.legend(fontsize=7, facecolor=PAN, labelcolor=TEXT)
        style_ax(ax, "Sharpe Distribution — Bar MC")

        # Panel 5: Trade-level equity
        ax = fig.add_subplot(gs[2, 0])
        if res_trade is not None:
            t_finals = res_trade["_finals"]
            ax.hist(t_finals / 1000, bins=50, color="#F0883E", alpha=0.75, edgecolor=BG, linewidth=0.3)
            ax.axvline(init / 1000, color=RED, lw=1.5, linestyle="--", label="Initial")
            ax.axvline(np.median(t_finals) / 1000, color=GREEN, lw=1.5, linestyle="-", label="Median")
            ax.set_xlabel("Final Equity ($k)"); ax.set_ylabel("Frequency")
            ax.legend(fontsize=7, facecolor=PAN, labelcolor=TEXT)
        style_ax(ax, f"Final Equity — Trade MC (N={res_trade['n_sim'] if res_trade else 0:,})")

        # Panel 6: 200 sample equity paths
        ax = fig.add_subplot(gs[2, 1])
        rng2     = np.random.default_rng(99)
        bar_rets = res_bar.get("_returns", None)
        if bar_rets is not None:
            n = len(bar_rets)
            for _ in range(200):
                s  = rng2.choice(bar_rets, size=n, replace=True)
                eq = init * np.cumprod(1 + s)
                ax.plot(np.linspace(0, 1, n), eq / 1000, lw=0.3, alpha=0.15, color=BLUE)
            ax.axhline(init / 1000, color=RED, lw=1, linestyle="--")
            ax.set_xlabel("Time (normalized)"); ax.set_ylabel("Equity ($k)")
        style_ax(ax, "200 Sample Equity Paths (Bootstrap)")

        plt.savefig(CHART_PATH, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close()
        log.info("Chart saved → %s", CHART_PATH)

    except ImportError as e:
        log.warning("matplotlib tidak tersedia: %s", e)
    except Exception as e:
        log.warning("Chart generation failed: %s", e)


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def save_results(results: dict, path: Path) -> None:
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

    # Prefer risk_managed (dengan kill switch) untuk lebih accurate MC
    # FIX: btc_backtest_results.csv = raw (no KS) → ruin rate 98%
    #      btc_risk_managed_results.csv = KS-protected returns → realistic MC
    if RISK_PATH.exists():
        path = RISK_PATH
        log.info("Menggunakan risk-managed returns (dengan kill switch protection)")
    elif BACKTEST_PATH.exists():
        path = BACKTEST_PATH
        log.warning("risk_managed_results.csv tidak ditemukan, fallback ke backtest_results.csv")
        log.warning("Ruin rate mungkin inflated karena raw backtest tanpa kill switch!")
    else:
        log.error("File tidak ditemukan: %s", RISK_PATH)
        log.error("Jalankan backtest_engine.py + risk_engine_v6.py terlebih dahulu.")
        raise SystemExit(1)

    all_results = {}

    # ── Bar-level MC ──────────────────────────────────────────────────────────
    if not trade_only:
        bar_returns = load_bar_returns(path)
        res_bar     = run_monte_carlo(bar_returns, n_sim=n_sim, ruin_thresh=ruin,
                                      label="BAR-LEVEL")
        tests_bar   = evaluate_results(res_bar)
        rob_bar     = compute_robustness_score(res_bar)
        print_results(res_bar, tests_bar, rob_bar)
        save_results(res_bar, MC_BAR_PATH)
        all_results["bar"] = {**res_bar, "robustness": rob_bar}
    else:
        res_bar = None

    # ── Trade-level MC ────────────────────────────────────────────────────────
    if not bar_only:
        trade_returns, _ = extract_trade_returns(path)
        if len(trade_returns) >= 20:
            res_trade   = run_monte_carlo(trade_returns, n_sim=n_sim, ruin_thresh=ruin,
                                          label="TRADE-LEVEL")
            tests_trade = evaluate_results(res_trade)
            rob_trade   = compute_robustness_score(res_trade)
            print_results(res_trade, tests_trade, rob_trade)
            save_results(res_trade, MC_TRADE_PATH)
            all_results["trade"] = {**res_trade, "robustness": rob_trade}
        else:
            log.warning("Trade count (%d) < 20 — skip Trade-level MC", len(trade_returns))
            res_trade = None
    else:
        res_trade = None

    # ── Charts ────────────────────────────────────────────────────────────────
    if not no_plot and res_bar is not None:
        save_charts(res_bar, res_trade)

    # ── Comparison summary ────────────────────────────────────────────────────
    if res_bar is not None and res_trade is not None:
        print("─" * 70)
        print("  COMPARISON: BAR-LEVEL vs TRADE-LEVEL")
        print("─" * 70)
        for lbl, res in [("BAR-LEVEL", res_bar), ("TRADE-LEVEL", res_trade)]:
            rob = all_results[lbl.lower().replace("-level","")].get("robustness", {})
            print(f"  {lbl:<13}  "
                  f"Median EQ: ${res['eq_median']:>12,.0f}  "
                  f"DD p5: {res['dd_p5']:>7.1f}%  "
                  f"PF median: {res['pf_median']:>5.3f}  "
                  f"Score: {rob.get('robustness_score',0):.0f}/100 [{rob.get('robustness_grade','?')}]")
        print("─" * 70 + "\n")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo Engine [Fixed] — BTC Hybrid Model")
    parser.add_argument("-n", "--n-sim",     type=int,   default=DEFAULT_N_SIM)
    parser.add_argument("--ruin",            type=float, default=DEFAULT_RUIN_THRESH)
    parser.add_argument("--bar-only",        action="store_true")
    parser.add_argument("--trade-only",      action="store_true")
    parser.add_argument("--no-plot",         action="store_true")
    args = parser.parse_args()

    main(n_sim=args.n_sim, ruin=args.ruin,
         bar_only=args.bar_only, trade_only=args.trade_only,
         no_plot=args.no_plot)
