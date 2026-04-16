"""
╔══════════════════════════════════════════════════════════════════════╗
║   out_of_sample_validation.py — BTC Hybrid Model V7                ║
║   BAGIAN 5: Out-of-Sample Validation                               ║
╠══════════════════════════════════════════════════════════════════════╣
║  Data split:                                                        ║
║    Training IS : 60% — 2017-01 → ~2022-01                         ║
║    Validation  : 20% — ~2022-01 → ~2024-01                        ║
║    True OOS    : 20% — ~2024-01 → 2026-03 (never seen)            ║
║                                                                     ║
║  Hypothesis test:                                                   ║
║    H0: OOS performance is NOT significantly worse than IS          ║
║    H1: OOS performance degrades significantly (overfitting)        ║
║                                                                     ║
║  Metrics per split:                                                 ║
║    CAGR, Sharpe, PF, MaxDD, WinRate, Calmar                       ║
╠══════════════════════════════════════════════════════════════════════╣
║  Cara pakai:                                                        ║
║    python out_of_sample_validation.py                              ║
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
OUT_PATH  = DATA_DIR / "oos_validation.csv"

BARS_PER_YEAR = 2190
INIT          = 10_000.0

# Split fractions
IS_FRAC  = 0.60
VAL_FRAC = 0.20
OOS_FRAC = 0.20

DIV = "═" * 70
SEP = "─" * 70


def ok(msg):   print(f"  [OK] {msg}")
def warn(msg): print(f"  [WARN]️  {msg}")
def err(msg):  print(f"  ❌ {msg}")


def calc_metrics(ret: np.ndarray, label: str = "") -> dict:
    ret_nz = ret[ret != 0.0]
    if len(ret_nz) < 10:
        return dict(label=label, n_bars=len(ret), n_active=len(ret_nz),
                    cagr=np.nan, sharpe=np.nan, pf=np.nan, max_dd=np.nan,
                    calmar=np.nan, win_rate=np.nan)

    eq      = INIT * np.cumprod(1.0 + ret)
    n_years = len(ret) / BARS_PER_YEAR
    final   = float(eq[-1])
    cagr    = ((final / INIT) ** (1 / max(n_years, 0.01)) - 1) * 100
    peak    = np.maximum.accumulate(eq)
    peak    = np.where(peak > 0, peak, 1e-9)
    max_dd  = float(((eq - peak) / peak).min() * 100)
    calmar  = abs(cagr / max_dd) if max_dd < 0 else 99.0
    mu, sg  = ret_nz.mean(), ret_nz.std()
    sharpe  = float((mu / sg) * np.sqrt(BARS_PER_YEAR)) if sg > 1e-10 else 0.0
    neg_r   = ret_nz[ret_nz < 0]
    sortino = float((mu / neg_r.std()) * np.sqrt(BARS_PER_YEAR)) \
              if (len(neg_r) > 0 and neg_r.std() > 0) else 0.0
    wins    = ret_nz[ret_nz > 0]
    losses  = ret_nz[ret_nz < 0]
    pf      = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else 99.0
    wr      = float(len(wins) / (len(wins)+len(losses))) * 100

    return dict(label=label, n_bars=len(ret), n_active=len(ret_nz),
                cagr=round(cagr,2), sharpe=round(sharpe,3), sortino=round(sortino,3),
                pf=round(min(pf,99),4), max_dd=round(max_dd,2),
                calmar=round(min(calmar,99),3), win_rate=round(wr,2))


# ══════════════════════════════════════════════════════════════════
#  SPLIT DATA
# ══════════════════════════════════════════════════════════════════

def split_data(df: pd.DataFrame) -> tuple:
    n   = len(df)
    i_is  = int(n * IS_FRAC)
    i_val = int(n * (IS_FRAC + VAL_FRAC))

    is_df  = df.iloc[:i_is].copy()
    val_df = df.iloc[i_is:i_val].copy()
    oos_df = df.iloc[i_val:].copy()

    def date_range(sub):
        if "timestamp" in sub.columns and len(sub) > 0:
            ts = pd.to_datetime(sub["timestamp"], utc=True, errors="coerce")
            return f"{ts.iloc[0].strftime('%Y-%m-%d')} → {ts.iloc[-1].strftime('%Y-%m-%d')}"
        return f"bars {sub.index[0]}–{sub.index[-1]}"

    log.info("IS  split: %d bars (%s)", len(is_df), date_range(is_df))
    log.info("VAL split: %d bars (%s)", len(val_df), date_range(val_df))
    log.info("OOS split: %d bars (%s)", len(oos_df), date_range(oos_df))

    return is_df, val_df, oos_df


# ══════════════════════════════════════════════════════════════════
#  PERFORMANCE COMPARISON
# ══════════════════════════════════════════════════════════════════

def compare_splits(df: pd.DataFrame) -> dict:
    print(f"\n{SEP}")
    print("  OOS VALIDATION — Performance Across Splits")
    print(SEP)

    ret_col = "equity_return" if "equity_return" in df.columns else "strategy_return"
    is_df, val_df, oos_df = split_data(df)

    # Get date ranges
    def get_dates(sub):
        if "timestamp" not in sub.columns or len(sub) == 0:
            return ("?", "?")
        ts = pd.to_datetime(sub["timestamp"], utc=True, errors="coerce")
        return (ts.iloc[0].strftime("%Y-%m-%d"), ts.iloc[-1].strftime("%Y-%m-%d"))

    splits = [
        ("Training IS (60%)", is_df),
        ("Validation (20%)",  val_df),
        ("True OOS (20%)",    oos_df),
        ("FULL DATA",         df),
    ]

    metrics_list = []
    for label, sub in splits:
        ret = sub[ret_col].fillna(0).values
        m   = calc_metrics(ret, label)
        d0, d1 = get_dates(sub)
        m["date_start"] = d0
        m["date_end"]   = d1
        metrics_list.append(m)

    # Print table
    print(f"\n  {'Split':<22} {'Period':>25} {'CAGR%':>9} {'Sharpe':>8} "
          f"{'PF':>7} {'MaxDD%':>8} {'WR%':>7}")
    print("  " + "─" * 70)

    for m in metrics_list:
        period = f"{m['date_start']}→{m['date_end']}"
        cagr   = f"{m['cagr']:>+9.1f}%" if not np.isnan(m.get("cagr",np.nan)) else "     N/A"
        sh     = f"{m['sharpe']:>8.3f}" if not np.isnan(m.get("sharpe",np.nan)) else "     N/A"
        pf     = f"{m['pf']:>7.3f}"    if not np.isnan(m.get("pf",np.nan))    else "    N/A"
        dd     = f"{m['max_dd']:>8.1f}%" if not np.isnan(m.get("max_dd",np.nan)) else "     N/A"
        wr     = f"{m['win_rate']:>7.1f}%" if not np.isnan(m.get("win_rate",np.nan)) else "    N/A"
        print(f"  {m['label']:<22} {period:>25} {cagr} {sh} {pf} {dd} {wr}")

    return metrics_list, (is_df, val_df, oos_df)


# ══════════════════════════════════════════════════════════════════
#  DEGRADATION ANALYSIS
# ══════════════════════════════════════════════════════════════════

def analyze_degradation(metrics_list: list) -> dict:
    print(f"\n{SEP}")
    print("  DEGRADATION ANALYSIS (IS → OOS)")
    print(SEP)

    is_m  = next(m for m in metrics_list if "Training" in m["label"])
    oos_m = next(m for m in metrics_list if "OOS" in m["label"])
    val_m = next(m for m in metrics_list if "Validation" in m["label"])

    def deg_ratio(is_val, oos_val, metric):
        if np.isnan(is_val) or np.isnan(oos_val): return np.nan
        if abs(is_val) < 1e-6: return 0.0
        return oos_val / abs(is_val)

    metrics_to_check = ["cagr","sharpe","pf","max_dd"]
    degradations = {}

    print(f"\n  {'Metric':<15} {'IS':>10} {'VAL':>10} {'OOS':>10} "
          f"{'Deg Ratio':>11} {'Status':>12}")
    print("  " + "─"*63)

    passed = True
    for met in metrics_to_check:
        is_val  = is_m.get(met, np.nan)
        val_val = val_m.get(met, np.nan)
        oos_val = oos_m.get(met, np.nan)

        # For MaxDD: less negative OOS = better
        if met == "max_dd":
            deg  = abs(oos_val) / max(abs(is_val), 1e-6) if not (np.isnan(is_val) or np.isnan(oos_val)) else np.nan
            ok_  = deg < 2.0  # OOS DD < 2x IS DD
        else:
            deg  = oos_val / max(abs(is_val), 1e-6) if not (np.isnan(is_val) or np.isnan(oos_val)) else np.nan
            ok_  = deg > 0.5  # OOS at least 50% of IS performance

        status = "[OK] OK" if ok_ else "[WARN]️ DEGRADE"
        if not ok_:
            if met in ["sharpe","pf"]:
                passed = False

        degradations[met] = {"ratio": deg, "ok": ok_}

        is_s  = f"{is_val:+.3f}"  if not np.isnan(is_val)  else "N/A"
        val_s = f"{val_val:+.3f}" if not np.isnan(val_val) else "N/A"
        oos_s = f"{oos_val:+.3f}" if not np.isnan(oos_val) else "N/A"
        deg_s = f"{deg:+.3f}"     if not np.isnan(deg)     else "N/A"
        print(f"  {met:<15} {is_s:>10} {val_s:>10} {oos_s:>10} {deg_s:>11} {status:>12}")

    print()

    # Overfitting test: sharpe IS vs OOS
    sh_ratio = degradations.get("sharpe", {}).get("ratio", 0)
    if sh_ratio > 0.7:
        ok(f"Sharpe ratio OOS/IS = {sh_ratio:.2f} (>0.7 = acceptable degradation)")
    elif sh_ratio > 0.4:
        warn(f"Sharpe degrades to {sh_ratio:.2f}x IS level — moderate overfitting risk")
    else:
        err(f"Sharpe degrades to {sh_ratio:.2f}x IS level — significant overfitting")
        passed = False

    pf_ratio = degradations.get("pf", {}).get("ratio", 0)
    if pf_ratio > 0.7:
        ok(f"PF ratio OOS/IS = {pf_ratio:.2f} (>0.7 = acceptable)")
    elif pf_ratio > 0.5:
        warn(f"PF degrades to {pf_ratio:.2f}x IS level")
    else:
        err(f"PF degrades severely OOS ({pf_ratio:.2f}x)")
        passed = False

    return {
        "passed": passed,
        "sh_ratio": sh_ratio,
        "pf_ratio": pf_ratio,
        "degradations": degradations,
    }


# ══════════════════════════════════════════════════════════════════
#  BOOTSTRAP SIGNIFICANCE TEST
# ══════════════════════════════════════════════════════════════════

def bootstrap_significance(ret_is: np.ndarray,
                             ret_oos: np.ndarray,
                             n_boot: int = 1000) -> dict:
    print(f"\n{SEP}")
    print("  BOOTSTRAP SIGNIFICANCE TEST")
    print(SEP)

    rng = np.random.default_rng(42)

    def sharpe_boot(ret):
        ret_nz = ret[ret != 0]
        if len(ret_nz) < 5: return 0.0
        mu, sg = ret_nz.mean(), ret_nz.std()
        return (mu / sg) * np.sqrt(BARS_PER_YEAR) if sg > 1e-10 else 0.0

    # Bootstrap distribution for IS
    is_sharpes  = []
    oos_sharpes = []

    n_is  = len(ret_is)
    n_oos = len(ret_oos)

    log.info("Running %d bootstrap iterations...", n_boot)
    for _ in range(n_boot):
        idx_is  = rng.integers(0, n_is,  n_is)
        idx_oos = rng.integers(0, n_oos, n_oos)
        is_sharpes.append(sharpe_boot(ret_is[idx_is]))
        oos_sharpes.append(sharpe_boot(ret_oos[idx_oos]))

    is_mean  = np.mean(is_sharpes)
    oos_mean = np.mean(oos_sharpes)
    is_std   = np.std(is_sharpes)
    oos_std  = np.std(oos_sharpes)

    # One-sided test: is OOS Sharpe significantly less than IS Sharpe?
    diff = np.array(is_sharpes) - np.array(oos_sharpes)
    p_value = (diff <= 0).mean()  # proportion where IS ≤ OOS

    print(f"  Bootstrap N         : {n_boot}")
    print(f"  IS  Sharpe dist     : {is_mean:.3f} ± {is_std:.3f}")
    print(f"  OOS Sharpe dist     : {oos_mean:.3f} ± {oos_std:.3f}")
    print(f"  p-value (IS≤OOS)    : {p_value:.3f}")

    if p_value > 0.30:
        ok(f"OOS performance NOT significantly worse than IS (p={p_value:.3f})")
        ok("H0 accepted: No evidence of overfitting")
        overfitting = False
    elif p_value > 0.10:
        warn(f"Marginal performance difference (p={p_value:.3f}) — monitor")
        overfitting = False
    else:
        warn(f"IS significantly outperforms OOS (p={p_value:.3f}) — potential overfitting")
        overfitting = True

    return {
        "passed": not overfitting,
        "p_value": p_value,
        "is_sharpe_mean": is_mean,
        "oos_sharpe_mean": oos_mean,
        "overfitting_detected": overfitting,
    }


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def run() -> dict:
    print(f"\n{DIV}")
    print("  OUT-OF-SAMPLE VALIDATION — BTC Hybrid Model V7")
    print(f"  Split: IS={IS_FRAC*100:.0f}% / VAL={VAL_FRAC*100:.0f}% / OOS={OOS_FRAC*100:.0f}%")
    print(DIV)

    path = RISK_PATH if RISK_PATH.exists() else BT_PATH
    df   = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True,
                                          errors="coerce")
    log.info("Loaded: %d bars", len(df))

    ret_col  = "equity_return" if "equity_return" in df.columns else "strategy_return"

    metrics_list, (is_df, val_df, oos_df) = compare_splits(df)
    deg_result = analyze_degradation(metrics_list)

    ret_is  = is_df[ret_col].fillna(0).values
    ret_oos = oos_df[ret_col].fillna(0).values
    boot_result = bootstrap_significance(ret_is, ret_oos, n_boot=1000)

    # Save results
    results_df = pd.DataFrame(metrics_list)
    results_df.to_csv(OUT_PATH, index=False)
    log.info("OOS results saved → %s", OUT_PATH)

    # Final verdict
    oos_m    = next(m for m in metrics_list if "OOS" in m["label"])
    oos_pass = (not np.isnan(oos_m.get("pf",np.nan)) and oos_m["pf"] > 1.0
                and not np.isnan(oos_m.get("sharpe",np.nan)) and oos_m["sharpe"] > 0.5)

    results = [
        ("Degradation Analysis",  deg_result["passed"]),
        ("Bootstrap Significance",boot_result["passed"]),
        ("OOS PF > 1.0",          oos_m.get("pf",0) > 1.0),
        ("OOS Sharpe > 0.5",      oos_m.get("sharpe",0) > 0.5),
    ]
    n_pass = sum(1 for _, p in results if p)
    score  = n_pass / len(results) * 100

    print(f"\n{DIV}")
    print("  OOS VALIDATION VERDICT")
    print(DIV)
    for name, passed in results:
        print(f"  {'[OK]' if passed else '[WARN]️ '} {name}")
    print(SEP)
    print(f"  OOS Sharpe   : {oos_m.get('sharpe',np.nan):.3f}")
    print(f"  OOS PF       : {oos_m.get('pf',np.nan):.4f}")
    print(f"  OOS CAGR     : {oos_m.get('cagr',np.nan):+.1f}%")
    print(f"  Sharpe ratio OOS/IS : {deg_result.get('sh_ratio',0):.3f}")
    print(SEP)
    if score >= 75:
        ok("OUT-OF-SAMPLE validation PASSED — strategy generalizes well")
    elif score >= 50:
        warn("Partial OOS pass — some degradation expected in live trading")
    else:
        err("OOS validation FAILED — significant overfitting risk")
    print(f"{DIV}\n")

    return {
        "oos_validation_score": round(score, 1),
        "oos_sharpe": oos_m.get("sharpe", np.nan),
        "oos_pf":     oos_m.get("pf", np.nan),
        "oos_cagr":   oos_m.get("cagr", np.nan),
        "sh_ratio":   deg_result.get("sh_ratio", 0),
        "pf_ratio":   deg_result.get("pf_ratio", 0),
        "p_value":    boot_result.get("p_value", 0),
        "passed":     score >= 50,
    }


if __name__ == "__main__":
    run()
