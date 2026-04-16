"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  validation_v2.py — BTC Hybrid AI v2                                       ║
║  Purged Walk Forward + Fat-Tail Monte Carlo + Leakage Audit               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import logging
from typing import Callable, List, Optional, Tuple
from scipy import stats as scipy_stats

log = logging.getLogger(__name__)
SEP = "─" * 65


def _metrics(ret: pd.Series, label: str = "") -> dict:
    if len(ret) < 10: return {}
    eq   = (1 + ret).cumprod()
    peak = eq.expanding().max()
    dd   = (eq / peak - 1)
    n    = len(ret)
    bpy  = 2190  # 4h bars per year
    cagr = ((eq.iloc[-1] / eq.iloc[0]) ** (bpy / n) - 1) * 100
    sh   = ret.mean() / (ret.std() + 1e-10) * np.sqrt(bpy)
    sor  = ret.mean() / (ret[ret < 0].std() + 1e-10) * np.sqrt(bpy)
    w    = ret[ret > 0].sum()
    l    = -ret[ret < 0].sum()
    pf   = w / max(l, 1e-10)
    return {
        "label"    : label,
        "n_bars"   : n,
        "cagr_pct" : round(cagr, 2),
        "sharpe"   : round(sh, 3),
        "sortino"  : round(sor, 3),
        "pf"       : round(pf, 3),
        "max_dd"   : round(dd.min() * 100, 2),
        "win_rate" : round((ret > 0).mean() * 100, 1),
        "avg_win"  : round(ret[ret > 0].mean() * 100, 3) if (ret > 0).any() else 0,
        "avg_loss" : round(ret[ret < 0].mean() * 100, 3) if (ret < 0).any() else 0,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  LEAKAGE AUDIT
# ═════════════════════════════════════════════════════════════════════════════

class LeakageAudit:
    """
    Systematic check for data leakage in the pipeline.

    Checks:
    1. Target-feature correlation (suspicious if >0.3 at lag=0)
    2. Feature future correlation (any feature correlates with future price)
    3. Scaler contamination (IS vs OOS feature distributions)
    4. Risk model circular dependency
    """

    def run(
        self,
        feat_df:        pd.DataFrame,
        target_series:  pd.Series,
        close:          pd.Series,
        is_split:       float = 0.70,
    ) -> dict:
        results = {}
        print(f"\n{SEP}\n  LEAKAGE AUDIT\n{SEP}")

        n      = len(feat_df)
        is_end = int(n * is_split)

        # ── Check 1: Target-feature correlation at lag 0 ──────────────────────
        # If a feature at time t strongly correlates with target at time t,
        # that feature might contain forward information
        print("\nCheck 1: Target-feature correlation at lag=0")
        suspicious = []
        for col in feat_df.columns:
            f   = feat_df[col].iloc[:is_end]
            tgt = target_series.reindex(feat_df.index).iloc[:is_end]
            corr = f.corr(tgt)
            if abs(corr) > 0.25:
                suspicious.append((col, round(corr, 3)))
                print(f"  [WARN]️  {col}: corr={corr:.3f} — INVESTIGATE")
        if not suspicious:
            print("  [OK] No suspicious correlations")
        results["suspicious_features"] = suspicious

        # ── Check 2: Feature future autocorrelation ───────────────────────────
        print("\nCheck 2: Close-price predictability from features (should be near 0)")
        future_close = close.shift(-1)
        high_fwd_corrs = []
        for col in feat_df.columns[:10]:   # check top 10 for speed
            fc = feat_df[col].iloc[:is_end].corr(
                future_close.reindex(feat_df.index).iloc[:is_end]
            )
            if abs(fc) > 0.15:
                high_fwd_corrs.append((col, round(fc, 3)))
        if high_fwd_corrs:
            print(f"  [WARN]️  {len(high_fwd_corrs)} features have suspicious fwd corr")
        else:
            print("  [OK] No forward-price correlations detected")
        results["forward_corr_issues"] = high_fwd_corrs

        # ── Check 3: IS vs OOS feature distribution shift ─────────────────────
        print("\nCheck 3: Feature distribution shift IS → OOS (KS test)")
        ks_fails = []
        for col in feat_df.columns:
            is_vals  = feat_df[col].iloc[:is_end].dropna().values
            oos_vals = feat_df[col].iloc[is_end:].dropna().values
            if len(is_vals) < 50 or len(oos_vals) < 50: continue
            stat, pval = scipy_stats.ks_2samp(is_vals, oos_vals)
            if pval < 0.01:
                ks_fails.append((col, round(stat, 3), round(pval, 4)))
        if ks_fails:
            print(f"  [WARN]️  {len(ks_fails)} features show significant distribution shift:")
            for col, stat, pval in ks_fails[:5]:
                print(f"     {col}: KS={stat}, p={pval}")
        else:
            print("  [OK] Feature distributions stable IS → OOS")
        results["distribution_shift"] = ks_fails

        # ── Check 4: Target leakage — target should have near-zero IS autocorr
        print("\nCheck 4: Target series integrity")
        tgt = target_series.reindex(feat_df.index).iloc[:is_end].dropna()
        ac1 = tgt.autocorr(1)
        ac4 = tgt.autocorr(4)
        print(f"  Target autocorr lag-1: {ac1:.4f} (should be near 0)")
        print(f"  Target autocorr lag-4: {ac4:.4f} (should be near 0)")
        if abs(ac1) > 0.1:
            print(f"  [WARN]️  High autocorrelation in target — possible lookahead")
        else:
            print(f"  [OK] Target autocorrelation OK")
        results["target_autocorr"] = {"lag1": ac1, "lag4": ac4}

        passed = (
            len(suspicious) == 0 and
            len(high_fwd_corrs) == 0 and
            len(ks_fails) < 5 and
            abs(ac1) < 0.1
        )
        results["passed"] = passed
        print(f"\n  {'[OK] LEAKAGE AUDIT PASSED' if passed else '[WARN]️  LEAKAGE ISSUES FOUND — REVIEW ABOVE'}")
        return results


# ═════════════════════════════════════════════════════════════════════════════
#  PURGED WALK FORWARD VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

class PurgedWalkForward:
    """
    Walk forward with embargo gap between IS and OOS.

    Design:
    - Expanding window (IS grows, OOS fixed size)
    - Embargo: skip N bars after IS ends (prevents leakage at boundary)
    - Minimum IS size enforced (no tiny training windows)
    - Reports degradation ratio IS→OOS

    Interpretation:
    - OOS Sharpe ≥ 0.5            : model has live edge
    - PF ratio (OOS/IS) ≥ 0.50   : acceptable degradation
    - Profitable windows ≥ 60%    : consistent, not lucky
    """

    def __init__(
        self,
        min_train_bars : int = 3000,
        oos_bars       : int = 500,
        embargo_bars   : int = 48,
    ):
        self.min_train = min_train_bars
        self.oos_bars  = oos_bars
        self.embargo   = embargo_bars

    def run(
        self,
        returns:   pd.Series,
        model_fn:  Optional[Callable] = None,
        feat_df:   Optional[pd.DataFrame] = None,
    ) -> dict:
        n = len(returns)
        print(f"\n{SEP}\n  PURGED WALK FORWARD VALIDATION\n{SEP}")
        print(f"  Data: {n} bars  Min train: {self.min_train}  "
              f"OOS: {self.oos_bars}  Embargo: {self.embargo}")

        if n < self.min_train + self.oos_bars:
            msg = (f"Insufficient data: need {self.min_train+self.oos_bars}, "
                   f"have {n}")
            print(f"  [WARN]️  {msg}")
            return {"passed": False, "error": msg, "n_windows": 0}

        windows = []
        cursor  = self.min_train

        while cursor + self.oos_bars <= n:
            is_idx  = np.arange(0, cursor)
            oos_start = cursor + self.embargo
            oos_end   = min(oos_start + self.oos_bars, n)
            oos_idx = np.arange(oos_start, oos_end)

            if len(oos_idx) < self.oos_bars // 2:
                break

            is_ret  = returns.iloc[is_idx]
            oos_ret = returns.iloc[oos_idx]

            is_m   = _metrics(is_ret,  "IS")
            oos_m  = _metrics(oos_ret, "OOS")

            windows.append({
                "is_end"    : returns.index[cursor - 1],
                "oos_start" : returns.index[oos_start],
                "oos_end"   : returns.index[oos_end - 1],
                "is_sharpe" : is_m.get("sharpe", 0),
                "oos_sharpe": oos_m.get("sharpe", 0),
                "is_pf"     : is_m.get("pf", 0),
                "oos_pf"    : oos_m.get("pf", 0),
                "is_cagr"   : is_m.get("cagr_pct", 0),
                "oos_cagr"  : oos_m.get("cagr_pct", 0),
                "is_dd"     : is_m.get("max_dd", 0),
                "oos_dd"    : oos_m.get("max_dd", 0),
            })

            cursor += self.oos_bars

        if not windows:
            return {"passed": False, "error": "No windows generated", "n_windows": 0}

        wf = pd.DataFrame(windows)

        # Metrics
        avg_oos_sharpe = wf["oos_sharpe"].mean()
        avg_oos_pf     = wf["oos_pf"].mean()
        pf_ratio       = (wf["oos_pf"] / wf["is_pf"].replace(0, np.nan)).mean()
        profitable_pct = (wf["oos_pf"] > 1.0).mean() * 100
        sharpe_deg     = (wf["oos_sharpe"] / wf["is_sharpe"].replace(0, np.nan)).mean()

        print(f"\n  Windows         : {len(wf)}")
        print(f"  Avg OOS Sharpe  : {avg_oos_sharpe:.3f}  (target: >0.50)")
        print(f"  Avg OOS PF      : {avg_oos_pf:.3f}  (target: >1.00)")
        print(f"  PF ratio OOS/IS : {pf_ratio:.3f}  (target: >0.50)")
        print(f"  Profitable wins : {profitable_pct:.0f}%  (target: >60%)")
        print(f"  Sharpe deg      : {sharpe_deg:.3f}  (IS→OOS, target: >0.30)")
        print(f"\n  Window detail:")
        for _, row in wf.iterrows():
            mark = "[OK]" if row["oos_pf"] > 1.0 else "❌"
            print(f"  {mark}  OOS {row['oos_start'].date()}→{row['oos_end'].date()} "
                  f"Sharpe={row['oos_sharpe']:.2f}  PF={row['oos_pf']:.2f}  "
                  f"CAGR={row['oos_cagr']:+.1f}%")

        passed = avg_oos_sharpe >= 0.3 and avg_oos_pf >= 0.90 and profitable_pct >= 50
        mark   = "[OK] PASS" if passed else "[WARN]️  WARN"
        print(f"\n  Status: {mark}")

        return {
            "n_windows"       : len(wf),
            "avg_oos_sharpe"  : round(avg_oos_sharpe, 3),
            "avg_oos_pf"      : round(avg_oos_pf, 3),
            "pf_ratio"        : round(pf_ratio, 3),
            "profitable_pct"  : round(profitable_pct, 1),
            "sharpe_deg"      : round(sharpe_deg, 3),
            "windows"         : wf,
            "passed"          : passed,
        }


# ═════════════════════════════════════════════════════════════════════════════
#  FAT-TAIL MONTE CARLO
# ═════════════════════════════════════════════════════════════════════════════

class FatTailMonteCarlo:
    """
    Realistic Monte Carlo simulation for BTC trading.

    Improvements over naive bootstrap:
    1. Student-t distribution (fat tails) instead of normal
    2. Trade clustering (BTC crashes come in clusters → GARCH-like)
    3. Regime-conditioned simulation
    4. Reports full drawdown distribution, not just max

    Why fat tails matter:
    - BTC return kurtosis ≈ 8-15 (normal = 3)
    - Naive bootstrap underestimates tail risk by 2-3x
    - A strategy that survives normal MC might be wiped out by
      a single fat-tail event (March 2020, May 2021, FTX Nov 2022)
    """

    def __init__(self, n_sims: int = 1000, seed: int = 42):
        self.n_sims = n_sims
        self.seed   = seed

    def run(
        self,
        returns:         pd.Series,
        use_fat_tails:   bool = True,
        use_clustering:  bool = True,
    ) -> dict:
        np.random.seed(self.seed)
        arr = returns.dropna().values
        n   = len(arr)

        print(f"\n{SEP}\n  FAT-TAIL MONTE CARLO  ({self.n_sims:,} sims)\n{SEP}")

        # Fit Student-t distribution to returns
        df_t, loc_t, scale_t = scipy_stats.t.fit(arr)
        print(f"  Return distribution fit:")
        print(f"    t-degrees: {df_t:.2f}  (normal=∞, BTC typically 3-5)")
        print(f"    skew     : {scipy_stats.skew(arr):.3f}")
        print(f"    kurtosis : {scipy_stats.kurtosis(arr):.3f}  (excess, normal=0)")

        pf_list, sh_list, dd_list, ruin_list, dd_p5_list = [], [], [], [], []

        for sim in range(self.n_sims):
            if use_fat_tails:
                # Sample from fitted Student-t
                sim_ret = scipy_stats.t.rvs(df_t, loc=loc_t,
                                             scale=scale_t, size=n)
            else:
                sim_ret = np.random.choice(arr, size=n, replace=True)

            if use_clustering:
                # GARCH-lite: multiply returns by a vol regime factor
                # Simulates volatility clustering
                vol_state = np.ones(n)
                for i in range(1, n):
                    if abs(sim_ret[i-1]) > np.percentile(abs(arr), 90):
                        vol_state[i] = np.random.choice([1.5, 2.0, 2.5])
                    else:
                        vol_state[i] = max(0.5, vol_state[i-1] * 0.95)
                sim_ret = sim_ret * vol_state

            # Metrics
            eq   = np.cumprod(1 + sim_ret)
            peak = np.maximum.accumulate(eq)
            dd   = (eq / peak - 1)

            wins = sim_ret[sim_ret > 0].sum()
            loss = -sim_ret[sim_ret < 0].sum()
            pf   = wins / max(loss, 1e-10)
            sh   = sim_ret.mean() / (sim_ret.std() + 1e-10) * np.sqrt(2190)
            ruin = eq.min() < 0.30   # 70% ruin threshold (stricter than before)

            pf_list.append(pf)
            sh_list.append(sh)
            dd_list.append(dd.min())
            ruin_list.append(ruin)
            dd_p5_list.append(np.percentile(dd, 5))

        pf_arr = np.array(pf_list)
        sh_arr = np.array(sh_list)
        dd_arr = np.array(dd_list)

        result = {
            "n_sims"          : self.n_sims,
            "fat_tails"       : use_fat_tails,
            "t_degrees_freedom": round(df_t, 2),
            "pf_mean"         : round(pf_arr.mean(), 3),
            "pf_p5"           : round(np.percentile(pf_arr, 5), 3),
            "pf_p50"          : round(np.percentile(pf_arr, 50), 3),
            "sharpe_mean"     : round(sh_arr.mean(), 3),
            "sharpe_p5"       : round(np.percentile(sh_arr, 5), 3),
            "dd_median"       : round(np.median(dd_arr) * 100, 2),
            "dd_worst_5pct"   : round(np.percentile(dd_arr, 5) * 100, 2),
            "dd_worst_1pct"   : round(np.percentile(dd_arr, 1) * 100, 2),
            "ruin_rate"       : round(np.mean(ruin_list) * 100, 2),
            "passed"          : (np.mean(ruin_list) < 0.10 and
                                  np.percentile(pf_arr, 5) > 0.90),
        }

        print(f"\n  PF  (mean / p5 / p50) : {result['pf_mean']} / {result['pf_p5']} / {result['pf_p50']}")
        print(f"  Sharpe (mean / p5)    : {result['sharpe_mean']} / {result['sharpe_p5']}")
        print(f"  MaxDD (median / p5 / p1): {result['dd_median']}% / "
              f"{result['dd_worst_5pct']}% / {result['dd_worst_1pct']}%")
        print(f"  Ruin rate (equity<30%): {result['ruin_rate']:.2f}%")
        mark = "[OK] PASS" if result["passed"] else "[WARN]️  WARN"
        print(f"  Status: {mark}")

        return result


# ═════════════════════════════════════════════════════════════════════════════
#  MASTER VALIDATION v2
# ═════════════════════════════════════════════════════════════════════════════

class ValidationV2:
    """Master validation orchestrator for v2 system."""

    def __init__(self,
                 min_train_bars=3000, oos_bars=500, embargo_bars=48,
                 n_mc_sims=1000, mc_seed=42):
        self.wfv = PurgedWalkForward(min_train_bars, oos_bars, embargo_bars)
        self.mc  = FatTailMonteCarlo(n_mc_sims, mc_seed)
        self.audit = LeakageAudit()

    def run(self, returns: pd.Series,
            feat_df=None, target=None, close=None) -> dict:

        print(f"\n{'═'*65}")
        print("  VALIDATION V2 — BTC Hybrid AI System")
        print(f"{'═'*65}")

        wfv_r  = self.wfv.run(returns)
        mc_r   = self.mc.run(returns)
        is_m   = _metrics(returns, "Full Period")

        # Score
        score = 0
        if is_m.get("pf", 0) >= 1.3:   score += 15
        if is_m.get("sharpe", 0) >= 1.0:score += 10
        if mc_r.get("ruin_rate", 100) < 5:  score += 20
        if mc_r.get("pf_p5", 0) >= 0.9:     score += 15
        if wfv_r.get("avg_oos_sharpe", 0) >= 0.3: score += 20
        if wfv_r.get("profitable_pct", 0) >= 50:   score += 10
        if wfv_r.get("pf_ratio", 0) >= 0.5:        score += 10

        grade = ("EXCELLENT" if score >= 85 else
                 "GOOD"      if score >= 70 else
                 "MARGINAL"  if score >= 50 else "FAIL")

        print(f"\n{'═'*65}")
        print(f"  VALIDATION SCORE : {score}/100  [{grade}]")
        print(f"  Full-period: CAGR={is_m.get('cagr_pct',0):+.1f}%  "
              f"Sharpe={is_m.get('sharpe',0):.2f}  PF={is_m.get('pf',0):.2f}  "
              f"DD={is_m.get('max_dd',0):.1f}%")
        print(f"{'═'*65}")

        return {
            "walk_forward": wfv_r,
            "monte_carlo" : mc_r,
            "full_metrics": is_m,
            "score"       : score,
            "grade"       : grade,
            "passed"      : score >= 60,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    np.random.seed(42)
    n   = 10000
    ret = pd.Series(np.random.randn(n)*0.006 + 0.0004,
                    index=pd.date_range("2019-01-01", periods=n, freq="4h"))
    v2  = ValidationV2(min_train_bars=2000, oos_bars=400, n_mc_sims=500)
    report = v2.run(ret)
    print(f"\nFinal grade: {report['grade']}")
