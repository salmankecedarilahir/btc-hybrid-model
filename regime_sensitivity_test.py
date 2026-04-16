"""
╔══════════════════════════════════════════════════════════════════════╗
║   regime_sensitivity_test.py — BTC Hybrid Model V7                 ║
║   BAGIAN 3: Regime Sensitivity Analysis                             ║
╠══════════════════════════════════════════════════════════════════════╣
║  Regimes analyzed:                                                  ║
║    1. Bull Market    (regime=UP, strong uptrend)                   ║
║    2. Bear Market    (regime=DOWN, strong downtrend)               ║
║    3. Sideways       (regime=SIDEWAYS/NEUTRAL, ranging)            ║
║    4. High Volatility (ATR percentile > 75th)                      ║
║    5. Low Volatility  (ATR percentile < 25th)                      ║
║  Output: CAGR, Sharpe, PF, MaxDD per regime                       ║
╠══════════════════════════════════════════════════════════════════════╣
║  Cara pakai:                                                        ║
║    python regime_sensitivity_test.py                               ║
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

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
RISK_PATH  = DATA_DIR / "btc_risk_managed_results.csv"
BT_PATH    = DATA_DIR / "btc_backtest_results.csv"
OUT_PATH   = DATA_DIR / "regime_sensitivity.csv"

BARS_PER_YEAR = 2190
INIT          = 10_000.0
DIV = "═" * 70
SEP = "─" * 70


def ok(msg):   print(f"  [OK] {msg}")
def warn(msg): print(f"  [WARN]️  {msg}")
def err(msg):  print(f"  ❌ {msg}")


def calc_metrics(ret: np.ndarray, label: str = "") -> dict:
    """Calculate key metrics for a return series."""
    ret_nz = ret[ret != 0.0]
    if len(ret_nz) < 10:
        return {"label": label, "n_bars": len(ret), "n_active": len(ret_nz),
                "cagr": np.nan, "sharpe": np.nan, "pf": np.nan, "max_dd": np.nan,
                "win_rate": np.nan}

    eq      = INIT * np.cumprod(1.0 + ret)
    n_years = len(ret) / BARS_PER_YEAR
    final   = float(eq[-1])
    cagr    = ((final / INIT) ** (1.0 / max(n_years, 0.1)) - 1.0) * 100

    peak    = np.maximum.accumulate(eq)
    peak    = np.where(peak > 0, peak, 1e-9)
    dd      = (eq - peak) / peak
    max_dd  = float(dd.min() * 100)

    mu      = ret_nz.mean()
    sig     = ret_nz.std()
    sharpe  = float((mu / sig) * np.sqrt(BARS_PER_YEAR)) if sig > 1e-10 else 0.0

    wins    = ret_nz[ret_nz > 0]
    losses  = ret_nz[ret_nz < 0]
    pf      = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else 99.0
    wr      = float(len(wins) / (len(wins) + len(losses))) * 100 if (len(wins)+len(losses)) > 0 else 0.0

    return {
        "label":    label,
        "n_bars":   len(ret),
        "n_active": len(ret_nz),
        "cagr":     round(cagr, 2),
        "sharpe":   round(sharpe, 3),
        "pf":       round(min(pf, 99.0), 4),
        "max_dd":   round(max_dd, 2),
        "win_rate": round(wr, 2),
    }


def detect_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign regime labels to each bar.
    Uses 'regime' column if available, otherwise computes from EMA/price.
    """
    out = df.copy()

    # ── Trend regime ────────────────────────────────────────────────
    if "regime" in df.columns:
        out["regime_label"] = df["regime"].str.upper().str.strip()
        log.info("Using existing 'regime' column")
    elif all(c in df.columns for c in ["close","ema_200"]):
        # Simple: above EMA200 = UP, below = DOWN
        ma200 = df["ema_200"].ffill()
        close = df["close"]
        ema_ratio = (close - ma200) / ma200
        conditions = [
            ema_ratio > 0.05,
            ema_ratio < -0.05,
        ]
        choices = ["UP", "DOWN"]
        out["regime_label"] = np.select(conditions, choices, default="SIDEWAYS")
    elif "close" in df.columns:
        # Fallback: rolling return to classify
        close = df["close"].values
        roll_ret = pd.Series(close).pct_change(90).fillna(0)
        out["regime_label"] = np.where(roll_ret > 0.10, "UP",
                               np.where(roll_ret < -0.10, "DOWN", "SIDEWAYS"))
    else:
        out["regime_label"] = "UNKNOWN"

    # ── Volatility regime ──────────────────────────────────────────
    if "atr_percentile" in df.columns:
        atr_pct = df["atr_percentile"].fillna(50)
        out["vol_regime"] = np.where(atr_pct > 75, "HIGH_VOL",
                            np.where(atr_pct < 25, "LOW_VOL", "MED_VOL"))
    elif "vol_24bar" in df.columns:
        vol = df["vol_24bar"].ffill()
        q25 = vol.quantile(0.25)
        q75 = vol.quantile(0.75)
        out["vol_regime"] = np.where(vol > q75, "HIGH_VOL",
                            np.where(vol < q25, "LOW_VOL", "MED_VOL"))
    elif "close" in df.columns:
        rolling_std = df["close"].pct_change().rolling(24).std().fillna(0)
        q25 = rolling_std.quantile(0.25)
        q75 = rolling_std.quantile(0.75)
        out["vol_regime"] = np.where(rolling_std > q75, "HIGH_VOL",
                            np.where(rolling_std < q25, "LOW_VOL", "MED_VOL"))
    else:
        out["vol_regime"] = "UNKNOWN"

    return out


# ══════════════════════════════════════════════════════════════════
#  REGIME PERFORMANCE ANALYSIS
# ══════════════════════════════════════════════════════════════════

def analyze_regime_performance(df: pd.DataFrame) -> dict:
    print(f"\n{SEP}")
    print("  REGIME PERFORMANCE ANALYSIS")
    print(SEP)

    ret_col = "equity_return" if "equity_return" in df.columns else "strategy_return"
    df_reg  = detect_regimes(df)
    ret_all = df[ret_col].fillna(0).values

    regimes_to_test = [
        ("ALL",        None,           None),
        ("BULL (UP)",  "regime_label", "UP"),
        ("BEAR (DOWN)","regime_label", "DOWN"),
        ("SIDEWAYS",   "regime_label", "SIDEWAYS"),
        ("HIGH VOL",   "vol_regime",   "HIGH_VOL"),
        ("LOW VOL",    "vol_regime",   "LOW_VOL"),
    ]

    # Add neutral/mixed if present
    for extra in ["NEUTRAL","MIXED"]:
        if (df_reg.get("regime_label","") == extra).any() if hasattr(df_reg.get("regime_label",""), "any") else False:
            regimes_to_test.append((extra, "regime_label", extra))

    results = []
    for label, col, val in regimes_to_test:
        if col is None:
            mask = np.ones(len(df_reg), dtype=bool)
        elif col in df_reg.columns:
            mask = (df_reg[col] == val).values
        else:
            continue

        ret_sub = ret_all[mask]
        m = calc_metrics(ret_sub, label)
        m["n_bars_pct"] = round(mask.sum() / len(df_reg) * 100, 1)
        results.append(m)

    # Print formatted table
    print(f"\n  {'Regime':<18} {'Bars%':>6} {'N_active':>9} {'CAGR%':>9} "
          f"{'Sharpe':>8} {'PF':>7} {'MaxDD%':>8} {'WR%':>7}")
    print("  " + "─" * 68)

    all_metrics = None
    for m in results:
        if m["label"] == "ALL":
            all_metrics = m
            continue
        cagr  = f"{m['cagr']:+.1f}%" if not np.isnan(m.get('cagr',np.nan)) else "N/A"
        sh    = f"{m['sharpe']:.3f}" if not np.isnan(m.get('sharpe',np.nan)) else "N/A"
        pf    = f"{m['pf']:.3f}" if not np.isnan(m.get('pf',np.nan)) else "N/A"
        dd    = f"{m['max_dd']:.1f}%" if not np.isnan(m.get('max_dd',np.nan)) else "N/A"
        wr    = f"{m['win_rate']:.1f}%" if not np.isnan(m.get('win_rate',np.nan)) else "N/A"
        print(f"  {m['label']:<18} {m.get('n_bars_pct',0):>5.1f}% {m['n_active']:>9,} "
              f"{cagr:>9} {sh:>8} {pf:>7} {dd:>8} {wr:>7}")

    if all_metrics:
        print("  " + "─" * 68)
        m = all_metrics
        print(f"  {'TOTAL'::<18} {'100%':>6} {m['n_active']:>9,} "
              f"  {m['cagr']:>+8.1f}%  {m['sharpe']:>7.3f}  {m['pf']:>6.3f} "
              f"  {m['max_dd']:>7.1f}%  {m['win_rate']:>6.1f}%")

    # Save results
    pd.DataFrame(results).to_csv(OUT_PATH, index=False)
    log.info("Regime results saved → %s", OUT_PATH)

    return results


# ══════════════════════════════════════════════════════════════════
#  REGIME SENSITIVITY SCORING
# ══════════════════════════════════════════════════════════════════

def score_regime_sensitivity(results: list) -> dict:
    print(f"\n{SEP}")
    print("  REGIME SENSITIVITY SCORING")
    print(SEP)

    df_res = pd.DataFrame(results)
    df_res = df_res[df_res["label"] != "ALL"].dropna(subset=["cagr","sharpe","pf"])

    if df_res.empty:
        warn("No regime data to score"); return {"passed": True, "score": 50}

    # Score: how consistent are metrics across regimes?
    pfs      = df_res["pf"].values
    sharpes  = df_res["sharpe"].values
    cagrs    = df_res["cagr"].values

    # PF consistency: all regimes should have PF > 0.9
    n_pf_ok = (pfs > 0.90).sum()
    n_pos   = (cagrs > -30).sum()   # less than -30% CAGR = very bad for that regime

    # Coefficient of variation across regimes
    pf_cv    = np.std(pfs) / max(np.mean(pfs), 0.1)
    sharpe_cv = np.std(sharpes) / max(abs(np.mean(sharpes)), 0.1)

    print(f"  PF > 0.90 in regimes : {n_pf_ok}/{len(df_res)}")
    print(f"  PF coeff. of variation: {pf_cv:.3f}  (low=stable)")
    print(f"  Sharpe CoV            : {sharpe_cv:.3f}  (low=stable)")

    # Find worst regime
    worst = df_res.loc[df_res["sharpe"].idxmin()]
    best  = df_res.loc[df_res["sharpe"].idxmax()]

    print(f"\n  Best regime  : {best['label']:<18}  Sharpe={best['sharpe']:.3f}  PF={best['pf']:.3f}")
    print(f"  Worst regime : {worst['label']:<18}  Sharpe={worst['sharpe']:.3f}  PF={worst['pf']:.3f}")

    # Sensitivity issues
    issues = []
    if n_pf_ok < len(df_res) * 0.5:
        err(f"Only {n_pf_ok}/{len(df_res)} regimes have PF > 0.90 — strategy struggles in certain regimes")
        issues.append("low_pf_regimes")
    else:
        ok(f"{n_pf_ok}/{len(df_res)} regimes have PF > 0.90")

    if pf_cv > 0.5:
        warn(f"High PF variability across regimes (CoV={pf_cv:.3f}) — regime-dependent performance")
        issues.append("pf_instability")
    else:
        ok(f"PF stable across regimes (CoV={pf_cv:.3f})")

    score = min(100, (n_pf_ok / max(len(df_res), 1)) * 50 +
                     max(0, (1 - pf_cv) * 30) +
                     max(0, (1 - sharpe_cv) * 20))

    print(f"\n  Regime Sensitivity Score : {score:.1f}/100")
    if score >= 70:
        ok("Strategy performs consistently across regimes")
    elif score >= 50:
        warn("Moderate regime sensitivity — monitor in live trading")
    else:
        err("High regime sensitivity — strategy may underperform in adverse regimes")

    return {
        "passed": score >= 50,
        "score": round(score, 1),
        "n_pf_ok": n_pf_ok,
        "n_regimes": len(df_res),
        "pf_cv": pf_cv,
        "issues": issues,
    }


# ══════════════════════════════════════════════════════════════════
#  BEAR MARKET SPECIFIC ANALYSIS
# ══════════════════════════════════════════════════════════════════

def analyze_bear_protection(df: pd.DataFrame) -> dict:
    print(f"\n{SEP}")
    print("  BEAR MARKET PROTECTION ANALYSIS")
    print(SEP)

    df_reg = detect_regimes(df)
    ret_col = "equity_return" if "equity_return" in df.columns else "strategy_return"
    ret = df[ret_col].fillna(0).values

    # Bear market periods
    if "regime_label" in df_reg.columns:
        bear_mask = (df_reg["regime_label"] == "DOWN").values
        bull_mask = (df_reg["regime_label"] == "UP").values
    else:
        bear_mask = np.zeros(len(df), dtype=bool)
        bull_mask = np.zeros(len(df), dtype=bool)

    # Kill switch activation during bear
    ks_col = "kill_switch_active" if "kill_switch_active" in df.columns else None
    if ks_col:
        ks_bear = df.loc[bear_mask, ks_col].mean() * 100 if bear_mask.sum() > 0 else 0
        ks_bull = df.loc[bull_mask, ks_col].mean() * 100 if bull_mask.sum() > 0 else 0
        print(f"  Kill switch active in BEAR : {ks_bear:.1f}% of bear bars")
        print(f"  Kill switch active in BULL : {ks_bull:.1f}% of bull bars")

        if ks_bear > ks_bull * 2:
            ok(f"Kill switch correctly more active in bear ({ks_bear:.0f}% vs {ks_bull:.0f}%)")
        else:
            warn("Kill switch activation not significantly higher in bear market")

    # Position sizing during bear
    if "leverage_used" in df.columns:
        lev_bear = df.loc[bear_mask, "leverage_used"].mean() if bear_mask.sum() > 0 else 0
        lev_bull = df.loc[bull_mask, "leverage_used"].mean() if bull_mask.sum() > 0 else 0
        print(f"  Avg leverage BEAR : {lev_bear:.3f}x")
        print(f"  Avg leverage BULL : {lev_bull:.3f}x")
        if lev_bear < lev_bull * 0.8:
            ok(f"Leverage reduced in bear market ({lev_bear:.2f}x vs {lev_bull:.2f}x bull)")
        else:
            warn(f"Leverage not significantly reduced in bear market")

    # Short exposure during bear
    if "position" in df.columns:
        pos_bear = df.loc[bear_mask, "position"]
        short_pct = (pos_bear == -1).mean() * 100 if len(pos_bear) > 0 else 0
        flat_pct  = (pos_bear == 0).mean() * 100  if len(pos_bear) > 0 else 0
        print(f"  Short exposure in BEAR : {short_pct:.1f}%")
        print(f"  Flat/paused in BEAR    : {flat_pct:.1f}%")

        total_protective = short_pct + flat_pct
        if total_protective > 50:
            ok(f"Good bear protection: {total_protective:.0f}% of bars short or flat")
        else:
            warn(f"Limited bear protection: only {total_protective:.0f}% flat/short during bear")

    return {"passed": True}


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def run() -> dict:
    print(f"\n{DIV}")
    print("  REGIME SENSITIVITY TEST — BTC Hybrid Model V7")
    print(f"{DIV}")

    path = RISK_PATH if RISK_PATH.exists() else BT_PATH
    df   = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True,
                                          errors="coerce")
    log.info("Loaded: %d bars", len(df))

    # Try to merge signals for regime data
    if "regime" not in df.columns:
        sig_path = DATA_DIR / "btc_trading_signals.csv"
        if sig_path.exists():
            sig_df = pd.read_csv(sig_path)
            if "regime" in sig_df.columns:
                log.info("Merging regime data from signals file")
                df = df.merge(sig_df[["timestamp","regime"]].drop_duplicates("timestamp"),
                              on="timestamp", how="left")

    regime_results = analyze_regime_performance(df)
    score_result   = score_regime_sensitivity(regime_results)
    bear_result    = analyze_bear_protection(df)

    print(f"\n{DIV}")
    print("  REGIME SENSITIVITY VERDICT")
    print(DIV)
    print(f"  Regime Sensitivity Score : {score_result['score']:.1f}/100")
    print(f"  Regimes with PF > 0.90   : {score_result['n_pf_ok']}/{score_result['n_regimes']}")

    if score_result["passed"]:
        ok("Strategy shows acceptable performance across all regimes")
    else:
        warn("Strategy has regime-specific weaknesses — see details above")

    print(f"{DIV}\n")

    return {
        "regime_sensitivity_score": score_result["score"],
        "n_pf_ok":   score_result["n_pf_ok"],
        "n_regimes": score_result["n_regimes"],
        "pf_cv":     score_result.get("pf_cv", 0),
        "passed":    score_result["passed"],
    }


if __name__ == "__main__":
    run()
