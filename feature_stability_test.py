"""
╔══════════════════════════════════════════════════════════════════════╗
║   feature_stability_test.py — BTC Hybrid Model V7                  ║
║   BAGIAN 2: Feature Stability Test                                  ║
╠══════════════════════════════════════════════════════════════════════╣
║  Checks:                                                            ║
║    1. Feature correlation matrix (detect redundancy / multicollin.) ║
║    2. Feature variance stability (split-half comparison)           ║
║    3. Feature stationarity check (ADF-like rolling test)           ║
║    4. Feature NaN rate per column                                  ║
║    5. Feature importance stability (signal correlation)            ║
╠══════════════════════════════════════════════════════════════════════╣
║  Cara pakai:                                                        ║
║    python feature_stability_test.py                                ║
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
AI_PATH   = DATA_DIR / "ai_training_dataset.csv"
SIG_PATH  = DATA_DIR / "btc_trading_signals.csv"
OUT_CORR  = DATA_DIR / "feature_correlation.csv"
OUT_STAB  = DATA_DIR / "feature_stability.csv"

DIV = "═" * 65
SEP = "─" * 65

# Features to include in analysis (numeric, meaningful for AI)
FEATURE_GROUPS = {
    "trend":       ["trend_score", "trend_score_norm", "ema20_vs_ema50",
                    "price_vs_ema20", "price_vs_ema50", "price_vs_ema200"],
    "momentum":    ["momentum_24", "momentum_72", "ret_1bar", "ret_4bar",
                    "ret_12bar", "ret_24bar"],
    "volatility":  ["vol_24bar", "vol_72bar", "vol_126bar", "vol_ratio"],
    "regime":      ["regime_encoded", "regime_up", "regime_down", "regime_sideways"],
    "derivatives": ["funding_rate", "funding_zscore", "derivatives_score",
                    "oi_zscore", "funding_extreme"],
    "signal":      ["hybrid_score", "signal_encoded", "signal_score",
                    "signal_strength", "signal_quality"],
    "risk":        ["drawdown", "equity_return", "leverage_used",
                    "kill_switch_active", "risk_state"],
}


def ok(msg):   print(f"  [OK] {msg}")
def warn(msg): print(f"  [WARN]️  {msg}")
def err(msg):  print(f"  ❌ {msg}")


def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    df = pd.read_csv(path)
    log.info("Loaded: %d rows × %d cols", len(df), len(df.columns))
    return df


def get_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return only numeric columns that are features (not targets/meta)."""
    exclude = {"timestamp","signal","regime","target_ret_1bar","target_ret_4bar",
               "target_ret_12bar","target_direction_1bar","target_profitable_trade",
               "open","high","low","close","volume","equity","shadow_equity",
               "running_max_equity"}
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in num_cols if c not in exclude
                 and not c.startswith("target_")]
    return df[feat_cols].copy()


# ══════════════════════════════════════════════════════════════════
#  TEST 1: Feature Correlation Matrix
# ══════════════════════════════════════════════════════════════════

def test_correlation(feat_df: pd.DataFrame) -> dict:
    print(f"\n{SEP}")
    print("  TEST 1 — Feature Correlation Matrix")
    print(SEP)

    # Drop columns with too many NaN
    valid = feat_df.loc[:, feat_df.isna().mean() < 0.3]
    valid = valid.fillna(valid.median())

    corr = valid.corr(method="pearson")
    corr.to_csv(OUT_CORR)
    log.info("Correlation matrix saved → %s", OUT_CORR)

    # Find highly correlated pairs (>0.90 absolute)
    high_corr_pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            v = abs(corr.iloc[i, j])
            if v > 0.90 and not np.isnan(v):
                high_corr_pairs.append((cols[i], cols[j], v))

    high_corr_pairs.sort(key=lambda x: -x[2])

    print(f"  Features analyzed : {len(valid.columns)}")
    print(f"  High corr pairs (>0.90): {len(high_corr_pairs)}")

    if high_corr_pairs:
        warn(f"  {len(high_corr_pairs)} highly correlated feature pairs:")
        for f1, f2, v in high_corr_pairs[:10]:
            print(f"      {f1:<30} ↔ {f2:<30}  r={v:.3f}")
        if len(high_corr_pairs) > 10:
            print(f"      ... and {len(high_corr_pairs)-10} more")
    else:
        ok("No highly correlated feature pairs (r>0.90)")

    # Correlation cluster analysis
    avg_abs_corr = (corr.abs().sum().sum() - len(cols)) / (len(cols) * (len(cols) - 1))
    print(f"  Avg abs correlation : {avg_abs_corr:.4f}")
    if avg_abs_corr < 0.3:
        ok(f"Low average correlation ({avg_abs_corr:.4f}) — diverse feature set")
    elif avg_abs_corr < 0.5:
        warn(f"Moderate correlation ({avg_abs_corr:.4f}) — consider feature selection")
    else:
        warn(f"High average correlation ({avg_abs_corr:.4f}) — feature redundancy possible")

    return {
        "passed": True,
        "n_high_corr": len(high_corr_pairs),
        "avg_abs_corr": avg_abs_corr,
        "high_corr_pairs": high_corr_pairs[:5],
    }


# ══════════════════════════════════════════════════════════════════
#  TEST 2: Feature Variance Stability
# ══════════════════════════════════════════════════════════════════

def test_variance_stability(feat_df: pd.DataFrame) -> dict:
    print(f"\n{SEP}")
    print("  TEST 2 — Feature Variance Stability (Split-Half)")
    print(SEP)

    n = len(feat_df)
    h = n // 2
    first_half  = feat_df.iloc[:h]
    second_half = feat_df.iloc[h:]

    stability_results = []

    for col in feat_df.columns:
        s1 = feat_df[col].iloc[:h].dropna()
        s2 = feat_df[col].iloc[h:].dropna()
        if len(s1) < 10 or len(s2) < 10:
            continue

        var1 = float(s1.var())
        var2 = float(s2.var())
        if var1 < 1e-10 and var2 < 1e-10:
            continue  # constant feature

        # Variance ratio (stable = ratio close to 1.0)
        var_ratio = max(var1, var2) / max(min(var1, var2), 1e-10)

        # Mean shift
        mean1 = float(s1.mean())
        mean2 = float(s2.mean())
        std_pool = (s1.std() + s2.std()) / 2
        mean_shift = abs(mean2 - mean1) / max(std_pool, 1e-10)

        stability_results.append({
            "feature":    col,
            "var_h1":     var1,
            "var_h2":     var2,
            "var_ratio":  var_ratio,
            "mean_h1":    mean1,
            "mean_h2":    mean2,
            "mean_shift": mean_shift,
            "stable":     var_ratio < 5.0 and mean_shift < 2.0,
        })

    stab_df = pd.DataFrame(stability_results)
    if stab_df.empty:
        warn("No features to analyze"); return {"passed": True}

    n_stable   = stab_df["stable"].sum()
    n_unstable = len(stab_df) - n_stable
    pct_stable = n_stable / len(stab_df) * 100

    print(f"  Features tested   : {len(stab_df)}")
    print(f"  Stable features   : {n_stable} ({pct_stable:.1f}%)")
    print(f"  Unstable features : {n_unstable}")

    if n_unstable > 0:
        unstable = stab_df[~stab_df["stable"]].sort_values("var_ratio", ascending=False)
        warn(f"Unstable features (var_ratio > 5x or mean_shift > 2σ):")
        for _, row in unstable.head(8).iterrows():
            print(f"      {row['feature']:<35}  var_ratio={row['var_ratio']:.1f}x  "
                  f"mean_shift={row['mean_shift']:.2f}σ")
    else:
        ok("All features variance-stable across first/second half")

    stab_df.to_csv(OUT_STAB, index=False)
    log.info("Stability results saved → %s", OUT_STAB)

    return {
        "passed": pct_stable >= 70,
        "pct_stable": pct_stable,
        "n_stable": n_stable,
        "n_unstable": n_unstable,
    }


# ══════════════════════════════════════════════════════════════════
#  TEST 3: Feature Stationarity (Rolling ADF proxy)
# ══════════════════════════════════════════════════════════════════

def test_stationarity(feat_df: pd.DataFrame) -> dict:
    print(f"\n{SEP}")
    print("  TEST 3 — Feature Stationarity Check (Rolling Mean/Std)")
    print(SEP)

    # Proxy for ADF: if rolling mean drifts consistently, feature is non-stationary
    WINDOW = min(500, len(feat_df) // 4)
    results = []

    # Priority features to check
    priority = [c for c in ["trend_score_norm","price_vs_ema200","momentum_72",
                              "vol_72bar","regime_encoded","signal_encoded",
                              "equity_return","leverage_used","drawdown"]
                if c in feat_df.columns]

    for col in priority:
        series = feat_df[col].ffill().fillna(0)
        if series.std() < 1e-10:
            continue

        roll_mean = series.rolling(WINDOW).mean().dropna()
        roll_std  = series.rolling(WINDOW).std().dropna()

        # Stationarity proxy: std of rolling means (stationary → low drift)
        mean_drift  = roll_mean.std() / max(series.std(), 1e-10)
        std_drift   = roll_std.std() / max(roll_std.mean(), 1e-10)
        stationary  = mean_drift < 0.5

        results.append({
            "feature":    col,
            "mean_drift": mean_drift,
            "std_drift":  std_drift,
            "stationary": stationary,
        })

    if not results:
        warn("No features to check"); return {"passed": True}

    res_df = pd.DataFrame(results)
    n_stat   = res_df["stationary"].sum()
    n_nonstat = len(res_df) - n_stat

    print(f"  Features checked  : {len(res_df)}")
    print(f"  {'Feature':<35} {'Mean Drift':>12} {'Stationary':>12}")
    print("  " + "─"*60)
    for _, row in res_df.iterrows():
        mark = "[OK]" if row["stationary"] else "[WARN]️ "
        print(f"  {mark} {row['feature']:<33} {row['mean_drift']:>12.4f} "
              f"{'YES' if row['stationary'] else 'TRENDING':>12}")

    if n_nonstat > 0:
        warn(f"{n_nonstat} features show non-stationary behavior (mean drift)")
        warn("Consider normalizing these features or using first-differences")
    else:
        ok("All checked features appear stationary")

    return {
        "passed": n_nonstat <= 2,
        "n_stationary": n_stat,
        "n_nonstationary": n_nonstat,
        "details": results,
    }


# ══════════════════════════════════════════════════════════════════
#  TEST 4: Feature NaN Rate
# ══════════════════════════════════════════════════════════════════

def test_nan_rate(feat_df: pd.DataFrame) -> dict:
    print(f"\n{SEP}")
    print("  TEST 4 — Feature NaN Rate Analysis")
    print(SEP)

    nan_rates = (feat_df.isna().mean() * 100).sort_values(ascending=False)
    high_nan  = nan_rates[nan_rates > 5.0]
    zero_var  = feat_df.select_dtypes(include=np.number).var()
    const_feat = zero_var[zero_var < 1e-10].index.tolist()

    print(f"  Total features    : {len(feat_df.columns)}")
    print(f"  Features NaN>5%   : {len(high_nan)}")
    print(f"  Constant features : {len(const_feat)}")

    if len(high_nan) > 0:
        warn("Features with >5% NaN:")
        for feat, rate in high_nan.head(10).items():
            print(f"      {feat:<40} {rate:.1f}%")
    else:
        ok("All features have NaN < 5%")

    if const_feat:
        warn(f"Constant/zero-variance features: {const_feat[:5]}")
    else:
        ok("No constant features")

    return {
        "passed": len(high_nan) == 0,
        "n_high_nan": len(high_nan),
        "n_constant": len(const_feat),
        "max_nan_pct": float(nan_rates.max()),
    }


# ══════════════════════════════════════════════════════════════════
#  TEST 5: Feature-Signal Correlation (Predictive Power Check)
# ══════════════════════════════════════════════════════════════════

def test_predictive_power(df: pd.DataFrame) -> dict:
    print(f"\n{SEP}")
    print("  TEST 5 — Feature Predictive Power (Signal Correlation)")
    print(SEP)

    target_col = None
    for c in ["target_ret_1bar","target_ret_4bar","equity_return"]:
        if c in df.columns:
            target_col = c
            break

    if target_col is None:
        warn("No target column found — skipping predictive power test")
        return {"passed": True}

    target = pd.to_numeric(df[target_col], errors="coerce").fillna(0)

    # Test correlation of each feature with future return
    feature_candidates = [c for c in df.select_dtypes(include=np.number).columns
                          if c not in {"equity","drawdown","shadow_equity",
                                       "running_max_equity","timestamp"}
                          and not c.startswith("target_")]

    corrs = {}
    for col in feature_candidates:
        try:
            feat = pd.to_numeric(df[col], errors="coerce").fillna(0)
            if feat.std() < 1e-10:
                continue
            c = float(np.corrcoef(feat.values, target.values)[0,1])
            if not np.isnan(c):
                corrs[col] = abs(c)
        except Exception:
            pass

    corr_series = pd.Series(corrs).sort_values(ascending=False)

    print(f"  Target column     : {target_col}")
    print(f"  Features tested   : {len(corrs)}")
    print(f"\n  Top 10 predictive features:")
    for feat, corr_val in corr_series.head(10).items():
        bar = "█" * int(corr_val * 40)
        print(f"    {feat:<35} {corr_val:.4f}  {bar}")

    # Features with meaningful signal (|corr| > 0.02)
    n_predictive = (corr_series > 0.02).sum()
    print(f"\n  Features with |corr| > 0.02 (weak signal): {n_predictive}/{len(corrs)}")

    if n_predictive >= 5:
        ok(f"{n_predictive} features have at least weak predictive power")
    else:
        warn(f"Only {n_predictive} predictive features — AI layer may struggle")

    return {
        "passed": n_predictive >= 5,
        "n_predictive": n_predictive,
        "top_features": corr_series.head(10).to_dict(),
    }


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def run() -> dict:
    print(f"\n{DIV}")
    print("  FEATURE STABILITY TEST — BTC Hybrid Model V7")
    print(f"{DIV}")

    # Load AI training dataset (has all features)
    path = AI_PATH if AI_PATH.exists() else SIG_PATH
    df   = load_features(path)
    feat_df = get_numeric_features(df)

    log.info("Feature analysis on %d features", len(feat_df.columns))

    # Run all tests
    r1 = test_correlation(feat_df)
    r2 = test_variance_stability(feat_df)
    r3 = test_stationarity(feat_df)
    r4 = test_nan_rate(feat_df)
    r5 = test_predictive_power(df)

    results = [
        ("Correlation Matrix",     r1["passed"]),
        ("Variance Stability",     r2["passed"]),
        ("Stationarity Check",     r3["passed"]),
        ("NaN Rate",               r4["passed"]),
        ("Predictive Power",       r5["passed"]),
    ]
    n_pass = sum(1 for _, p in results if p)

    print(f"\n{DIV}")
    print("  FEATURE STABILITY VERDICT")
    print(DIV)
    for name, passed in results:
        print(f"  {'[OK]' if passed else '[WARN]️ '} {name}")

    pct = n_pass / len(results) * 100
    print(SEP)
    print(f"  Score: {n_pass}/{len(results)} ({pct:.0f}%)")
    if pct >= 80:
        print("  [OK] Feature pipeline STABLE — ready for AI training")
    else:
        print("  [WARN]️  Feature pipeline has instabilities — review before AI training")
    print(f"{DIV}\n")

    return {
        "feature_stability_score": round(pct, 1),
        "n_high_corr":   r1.get("n_high_corr", 0),
        "pct_stable":    r2.get("pct_stable", 0),
        "n_nonstat":     r3.get("n_nonstationary", 0),
        "n_predictive":  r5.get("n_predictive", 0),
        "passed":        pct >= 60,
    }


if __name__ == "__main__":
    run()
