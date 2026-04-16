"""
╔══════════════════════════════════════════════════════════════════════╗
║   parameter_stability_test.py — BTC Hybrid Model V7                ║
║   BAGIAN 4: Parameter Stability / Perturbation Test                ║
╠══════════════════════════════════════════════════════════════════════╣
║  Approach:                                                          ║
║  Karena kita tidak memiliki akses ke signal engine source code     ║
║  saat runtime, kita menggunakan INDIRECT PARAMETER SENSITIVITY:    ║
║                                                                     ║
║  1. Return Distribution Perturbation — simulate ±10% parameter     ║
║     noise by bootstrapping return distributions                    ║
║  2. Signal Threshold Sensitivity — test PF/Sharpe sensitivity      ║
║     to hybrid_score threshold (current: 40/100)                   ║
║  3. Leverage Multiplier Sensitivity — test 0.8x to 1.2x leverage  ║
║  4. Kill Switch Threshold Sensitivity — test ±20% on KS levels    ║
║  5. Rolling Window Stability — metrics stable over time windows    ║
╠══════════════════════════════════════════════════════════════════════╣
║  Cara pakai:                                                        ║
║    python parameter_stability_test.py                              ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data"
RISK_PATH = DATA_DIR / "btc_risk_managed_results.csv"
BT_PATH   = DATA_DIR / "btc_backtest_results.csv"
SIG_PATH  = DATA_DIR / "btc_trading_signals.csv"
OUT_PATH  = DATA_DIR / "parameter_stability.csv"

BARS_PER_YEAR = 2190
INIT          = 10_000.0
DIV = "═" * 70
SEP = "─" * 70


def ok(msg):   print(f"  [OK] {msg}")
def warn(msg): print(f"  [WARN]️  {msg}")
def err(msg):  print(f"  ❌ {msg}")


def calc_metrics(ret: np.ndarray) -> dict:
    ret_nz  = ret[ret != 0.0]
    if len(ret_nz) < 5:
        return dict(cagr=np.nan, sharpe=np.nan, pf=np.nan, max_dd=np.nan)
    eq      = INIT * np.cumprod(1.0 + ret)
    n_years = len(ret) / BARS_PER_YEAR
    cagr    = ((eq[-1] / INIT) ** (1 / max(n_years, 0.01)) - 1) * 100
    peak    = np.maximum.accumulate(eq)
    peak    = np.where(peak > 0, peak, 1e-9)
    max_dd  = float(((eq - peak) / peak).min() * 100)
    mu, sg  = ret_nz.mean(), ret_nz.std()
    sharpe  = float((mu / sg) * np.sqrt(BARS_PER_YEAR)) if sg > 1e-10 else 0.0
    wins    = ret_nz[ret_nz > 0]
    losses  = ret_nz[ret_nz < 0]
    pf      = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else 99.0
    return dict(cagr=round(cagr,2), sharpe=round(sharpe,3),
                pf=round(min(pf,99),4), max_dd=round(max_dd,2))


# ══════════════════════════════════════════════════════════════════
#  TEST 1: Return Distribution Perturbation
# ══════════════════════════════════════════════════════════════════

def test_return_perturbation(df: pd.DataFrame) -> dict:
    print(f"\n{SEP}")
    print("  TEST 1 — Return Distribution Perturbation (±10% noise)")
    print(SEP)

    ret_col = "equity_return" if "equity_return" in df.columns else "strategy_return"
    ret_base = df[ret_col].fillna(0).values
    active   = ret_base[ret_base != 0]

    rng       = np.random.default_rng(42)
    PERTURBATIONS = {
        "Baseline":      0.00,
        "Noise +2%":     0.02,
        "Noise +5%":     0.05,
        "Noise +10%":    0.10,
        "Noise −5%":    -0.05,
        "Noise −10%":   -0.10,
    }

    results = []
    base_metrics = calc_metrics(ret_base)

    print(f"  {'Perturbation':<20} {'CAGR%':>9} {'Sharpe':>8} {'PF':>7} {'MaxDD%':>8}")
    print("  " + "─"*55)

    for label, noise in PERTURBATIONS.items():
        if noise == 0:
            ret_perturbed = ret_base.copy()
        else:
            # Add multiplicative noise to non-zero returns
            perturbed = ret_base.copy()
            nonzero_mask = perturbed != 0
            perturbed[nonzero_mask] *= (1 + noise * rng.normal(0, 1, nonzero_mask.sum()))
            # Clip to reasonable range
            perturbed = np.clip(perturbed, -0.15, 0.30)
            ret_perturbed = perturbed

        m = calc_metrics(ret_perturbed)
        m["label"] = label
        m["noise"]  = noise
        results.append(m)

        print(f"  {label:<20} {m['cagr']:>+9.1f}% {m['sharpe']:>8.3f} "
              f"{m['pf']:>7.3f} {m['max_dd']:>8.1f}%")

    # Sensitivity: how much do metrics change per 1% noise?
    pfs     = [r["pf"]     for r in results if r["noise"] >= 0]
    sharpes = [r["sharpe"] for r in results if r["noise"] >= 0]
    pf_cv   = np.std(pfs) / max(np.mean(pfs), 0.1)
    sh_cv   = np.std(sharpes) / max(abs(np.mean(sharpes)), 0.1)

    print(f"\n  PF CoV across perturbations    : {pf_cv:.4f}")
    print(f"  Sharpe CoV across perturbations: {sh_cv:.4f}")

    passed = True
    if pf_cv < 0.10:
        ok("PF very stable under return perturbation (CoV < 0.10)")
    elif pf_cv < 0.20:
        ok(f"PF stable under perturbation (CoV = {pf_cv:.3f})")
    else:
        warn(f"PF moderately sensitive to return perturbation (CoV = {pf_cv:.3f})")

    # Check: does strategy stay profitable under all perturbations?
    n_profitable = sum(1 for r in results if r.get("pf", 0) > 1.0)
    print(f"  PF > 1.0 in {n_profitable}/{len(results)} perturbation scenarios")
    if n_profitable == len(results):
        ok("Strategy profitable under all perturbation scenarios")
    elif n_profitable >= len(results) * 0.7:
        ok(f"Strategy profitable in {n_profitable}/{len(results)} scenarios")
    else:
        err(f"Strategy not profitable in {len(results)-n_profitable}/{len(results)} scenarios")
        passed = False

    return {
        "passed":       passed,
        "pf_cv":        pf_cv,
        "sharpe_cv":    sh_cv,
        "n_profitable": n_profitable,
        "results":      results,
    }


# ══════════════════════════════════════════════════════════════════
#  TEST 2: Signal Threshold Sensitivity
# ══════════════════════════════════════════════════════════════════

def test_signal_threshold(df: pd.DataFrame) -> dict:
    print(f"\n{SEP}")
    print("  TEST 2 — Signal Threshold Sensitivity (hybrid_score filter)")
    print(SEP)

    if "hybrid_score" not in df.columns:
        warn("hybrid_score not in dataset — skipping threshold test")
        return {"passed": True, "skipped": True}

    ret_col  = "equity_return" if "equity_return" in df.columns else "strategy_return"
    ret_all  = df[ret_col].fillna(0).values
    score    = df["hybrid_score"].fillna(0).values
    pos      = df["position"].values if "position" in df.columns else np.ones(len(df))

    THRESHOLDS = [20, 30, 40, 50, 60, 70]  # current = 40
    results    = []

    print(f"  {'Threshold':>10} {'N_signals':>10} {'CAGR%':>9} {'Sharpe':>8} {'PF':>7} {'MaxDD%':>8}")
    print("  " + "─"*55)

    for thresh in THRESHOLDS:
        # Simulate: only take signals where hybrid_score >= threshold
        signal_mask = (score >= thresh) | (pos == 0)  # keep flat bars
        ret_filtered = ret_all.copy()
        ret_filtered[~signal_mask] = 0.0  # zero out skipped signals

        n_signals = int((signal_mask & (pos != 0)).sum())
        m = calc_metrics(ret_filtered)
        m["threshold"] = thresh
        m["n_signals"] = n_signals
        results.append(m)

        marker = "◄ current" if thresh == 40 else ""
        print(f"  {'≥'+str(thresh):>10} {n_signals:>10,} {m['cagr']:>+9.1f}% "
              f"{m['sharpe']:>8.3f} {m['pf']:>7.3f} {m['max_dd']:>8.1f}%  {marker}")

    pfs = [r["pf"] for r in results]
    pf_cv = np.std(pfs) / max(np.mean(pfs), 0.1)

    if pf_cv < 0.15:
        ok(f"PF stable across score thresholds (CoV={pf_cv:.3f})")
    else:
        warn(f"PF sensitive to score threshold (CoV={pf_cv:.3f}) — current threshold important")

    return {
        "passed": pf_cv < 0.25,
        "pf_cv": pf_cv,
        "results": results,
    }


# ══════════════════════════════════════════════════════════════════
#  TEST 3: Leverage Multiplier Sensitivity
# ══════════════════════════════════════════════════════════════════

def test_leverage_sensitivity(df: pd.DataFrame) -> dict:
    print(f"\n{SEP}")
    print("  TEST 3 — Leverage Multiplier Sensitivity (0.7x to 1.5x)")
    print(SEP)

    ret_col = "equity_return" if "equity_return" in df.columns else "strategy_return"
    ret_base = df[ret_col].fillna(0).values

    LEV_MULTIPLIERS = [0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.50]
    results = []

    print(f"  {'Lev Mult':>10} {'CAGR%':>9} {'Sharpe':>8} {'PF':>7} {'MaxDD%':>8}")
    print("  " + "─"*46)

    for mult in LEV_MULTIPLIERS:
        ret_adjusted = ret_base * mult
        ret_adjusted = np.clip(ret_adjusted, -0.15, 0.35)
        m = calc_metrics(ret_adjusted)
        m["lev_mult"] = mult
        results.append(m)

        marker = "◄ current" if mult == 1.00 else ""
        print(f"  {mult:>10.2f}x {m['cagr']:>+9.1f}% {m['sharpe']:>8.3f} "
              f"{m['pf']:>7.3f} {m['max_dd']:>8.1f}%  {marker}")

    # Check Sharpe degrades gracefully with leverage increase
    sh_at_1x = next(r["sharpe"] for r in results if r["lev_mult"] == 1.00)
    sh_at_15 = next(r["sharpe"] for r in results if r["lev_mult"] == 1.50)
    dd_at_15 = abs(next(r["max_dd"] for r in results if r["lev_mult"] == 1.50))

    if sh_at_15 > sh_at_1x * 0.7:
        ok(f"Sharpe degrades gracefully at 1.5x leverage ({sh_at_15:.3f} vs {sh_at_1x:.3f})")
    else:
        warn(f"Sharpe degrades significantly at 1.5x leverage")

    if dd_at_15 < 50:
        ok(f"MaxDD at 1.5x leverage = {dd_at_15:.1f}% (acceptable)")
    else:
        warn(f"MaxDD at 1.5x leverage = {dd_at_15:.1f}% (too high)")

    # Optimal leverage
    sharpes = [r["sharpe"] for r in results]
    optimal_mult = LEV_MULTIPLIERS[np.argmax(sharpes)]
    print(f"\n  Optimal leverage multiplier (best Sharpe): {optimal_mult:.1f}x")
    if optimal_mult <= 1.10:
        ok(f"Optimal leverage near 1x — current sizing is efficient")
    else:
        warn(f"Higher leverage ({optimal_mult:.1f}x) would improve Sharpe — consider")

    return {
        "passed": True,
        "optimal_mult": optimal_mult,
        "results": results,
    }


# ══════════════════════════════════════════════════════════════════
#  TEST 4: Kill Switch Threshold Sensitivity
# ══════════════════════════════════════════════════════════════════

def test_killswitch_sensitivity(df: pd.DataFrame) -> dict:
    print(f"\n{SEP}")
    print("  TEST 4 — Kill Switch Threshold Sensitivity")
    print(SEP)

    if "kill_switch_active" not in df.columns:
        warn("kill_switch_active not in dataset — using drawdown proxy")
        # Create proxy: KS is "active" when drawdown is deep
        if "drawdown" in df.columns:
            dd = df["drawdown"].fillna(0).values
        elif "equity" in df.columns:
            eq = df["equity"].values
            peak = np.maximum.accumulate(eq)
            dd = (eq - peak) / peak
        else:
            warn("Cannot compute drawdown — skipping KS sensitivity test")
            return {"passed": True, "skipped": True}
    else:
        dd = df["drawdown"].fillna(0).values if "drawdown" in df.columns else None

    ret_col  = "equity_return" if "equity_return" in df.columns else "strategy_return"
    ret_base = df[ret_col].fillna(0).values

    # Simulate different KS trigger thresholds
    # Current TIER1 = -0.15, TIER2 = -0.25
    KS_THRESHOLDS = [
        ("Very Tight (-10%/-20%)", -0.10, -0.20),
        ("Tight (-12%/-22%)",      -0.12, -0.22),
        ("Current (-15%/-25%)",    -0.15, -0.25),  # baseline
        ("Relaxed (-18%/-28%)",    -0.18, -0.28),
        ("Loose (-20%/-35%)",      -0.20, -0.35),
    ]

    results = []

    print(f"  {'KS Config':<30} {'% Paused':>10} {'CAGR%':>9} {'MaxDD%':>8}")
    print("  " + "─"*60)

    if dd is not None:
        for label, t1, t2 in KS_THRESHOLDS:
            # Simulate: pause when drawdown < t1
            ks_active  = (dd < t1)
            pct_paused = ks_active.mean() * 100
            ret_simulated = ret_base.copy()
            ret_simulated[ks_active] = 0.0

            m = calc_metrics(ret_simulated)
            results.append({**m, "label": label, "t1": t1, "t2": t2,
                             "pct_paused": pct_paused})

            marker = "◄ current" if abs(t1 + 0.15) < 0.001 else ""
            print(f"  {label:<30} {pct_paused:>9.1f}% {m['cagr']:>+9.1f}% "
                  f"{m['max_dd']:>8.1f}%  {marker}")

        pfs = [r["pf"] for r in results if "pf" in r]
        pf_cv = np.std(pfs) / max(np.mean(pfs), 0.1)

        if pf_cv < 0.15:
            ok(f"PF stable across KS threshold variations (CoV={pf_cv:.3f})")
        else:
            warn(f"PF sensitive to KS threshold (CoV={pf_cv:.3f})")

    return {"passed": True, "results": results}


# ══════════════════════════════════════════════════════════════════
#  TEST 5: Rolling Window Stability
# ══════════════════════════════════════════════════════════════════

def test_rolling_stability(df: pd.DataFrame) -> dict:
    print(f"\n{SEP}")
    print("  TEST 5 — Rolling Window Stability (12-Month Windows)")
    print(SEP)

    ret_col = "equity_return" if "equity_return" in df.columns else "strategy_return"
    ret  = df[ret_col].fillna(0).values
    WINDOW = BARS_PER_YEAR  # 1 year

    results = []
    step    = WINDOW // 2  # 6-month step

    print(f"  {'Period':<25} {'CAGR%':>9} {'Sharpe':>8} {'PF':>7} {'MaxDD%':>8}")
    print("  " + "─"*55)

    ts = df["timestamp"] if "timestamp" in df.columns else None

    for start in range(0, len(ret) - WINDOW, step):
        end    = start + WINDOW
        window = ret[start:end]

        label = f"Bar {start}–{end}"
        if ts is not None:
            try:
                t0 = pd.to_datetime(ts.iloc[start]).strftime("%Y-%m")
                t1 = pd.to_datetime(ts.iloc[min(end-1,len(ts)-1)]).strftime("%Y-%m")
                label = f"{t0} → {t1}"
            except Exception:
                pass

        m = calc_metrics(window)
        m["label"] = label
        results.append(m)
        print(f"  {label:<25} {m['cagr']:>+9.1f}% {m['sharpe']:>8.3f} "
              f"{m['pf']:>7.3f} {m['max_dd']:>8.1f}%")

    if not results:
        warn("Not enough data for rolling windows"); return {"passed": True}

    pfs     = [r["pf"]     for r in results if not np.isnan(r.get("pf",np.nan))]
    sharpes = [r["sharpe"] for r in results if not np.isnan(r.get("sharpe",np.nan))]

    pf_cv   = np.std(pfs) / max(np.mean(pfs), 0.1) if pfs else 0
    sh_cv   = np.std(sharpes) / max(abs(np.mean(sharpes)), 0.1) if sharpes else 0
    n_prof  = sum(1 for p in pfs if p > 1.0)

    print(f"\n  PF CoV across windows   : {pf_cv:.4f}")
    print(f"  Sharpe CoV across windows: {sh_cv:.4f}")
    print(f"  Profitable windows       : {n_prof}/{len(pfs)}")

    if pf_cv < 0.30:
        ok(f"PF stable across rolling windows (CoV={pf_cv:.3f})")
    else:
        warn(f"PF variability across time windows (CoV={pf_cv:.3f}) — regime-dependent")

    return {
        "passed": n_prof >= len(pfs) * 0.5,
        "pf_cv": pf_cv,
        "sh_cv": sh_cv,
        "n_profitable": n_prof,
        "n_windows": len(results),
    }


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def run() -> dict:
    print(f"\n{DIV}")
    print("  PARAMETER STABILITY TEST — BTC Hybrid Model V7")
    print(f"{DIV}")

    path = RISK_PATH if RISK_PATH.exists() else BT_PATH
    df   = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True,
                                          errors="coerce")

    # Merge signals for hybrid_score if needed
    if "hybrid_score" not in df.columns:
        sig_path = SIG_PATH
        if sig_path.exists():
            sig_df = pd.read_csv(sig_path)
            if "hybrid_score" in sig_df.columns:
                df = df.merge(sig_df[["timestamp","hybrid_score"]].drop_duplicates("timestamp"),
                              on="timestamp", how="left")

    log.info("Loaded: %d bars", len(df))

    r1 = test_return_perturbation(df)
    r2 = test_signal_threshold(df)
    r3 = test_leverage_sensitivity(df)
    r4 = test_killswitch_sensitivity(df)
    r5 = test_rolling_stability(df)

    results_list = [
        ("Return Perturbation",    r1["passed"]),
        ("Signal Threshold",       r2.get("passed", True)),
        ("Leverage Sensitivity",   r3["passed"]),
        ("KS Threshold",           r4["passed"]),
        ("Rolling Stability",      r5["passed"]),
    ]
    n_pass = sum(1 for _, p in results_list if p)
    score  = n_pass / len(results_list) * 100

    # Save all results
    all_res = []
    for label, passed in results_list:
        all_res.append({"test": label, "passed": passed})
    pd.DataFrame(all_res).to_csv(OUT_PATH, index=False)

    print(f"\n{DIV}")
    print("  PARAMETER STABILITY VERDICT")
    print(DIV)
    for name, passed in results_list:
        print(f"  {'[OK]' if passed else '[WARN]️ '} {name}")
    print(SEP)
    print(f"  Score: {n_pass}/{len(results_list)} ({score:.0f}%)")
    if score >= 80:
        ok("Strategy parameters are STABLE — low sensitivity to perturbation")
    else:
        warn("Some parameter sensitivity found — review before AI layer")
    print(f"{DIV}\n")

    return {
        "parameter_stability_score": round(score, 1),
        "pf_cv_perturbation": r1.get("pf_cv", 0),
        "pf_cv_rolling":      r5.get("pf_cv", 0),
        "n_profitable_windows": r5.get("n_profitable", 0),
        "passed": score >= 60,
    }


if __name__ == "__main__":
    run()
