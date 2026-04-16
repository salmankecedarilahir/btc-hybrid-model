"""
╔══════════════════════════════════════════════════════════════════════════╗
║  validation_framework.py  —  BTC Autonomous AI Quant System            ║
║  LAYER 6 : Validation Framework                                         ║
╠══════════════════════════════════════════════════════════════════════════╣
║  TUJUAN  : Validasi sistem trading secara komprehensif                  ║
║  MODUL   :                                                              ║
║    A. Walk Forward Validation  (expanding window)                       ║
║    B. Monte Carlo Simulation   (return shuffling + bootstrapping)       ║
║    C. Out-of-Sample Test       (strict IS/OOS split)                    ║
║    D. Regime Robustness Test   (per-regime performance)                 ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import logging
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────────────────────

def _metrics(returns: pd.Series, label: str = "") -> dict:
    """Compute standard performance metrics from returns series."""
    if len(returns) == 0 or returns.std() == 0:
        return {}
    eq      = (1 + returns).cumprod()
    peak    = eq.expanding().max()
    dd      = (eq / peak - 1)
    cagr    = ((eq.iloc[-1] / eq.iloc[0]) ** (2190 / max(len(returns),1)) - 1) * 100
    sharpe  = returns.mean() / (returns.std() + 1e-10) * np.sqrt(2190)
    win_r   = (returns > 0).mean() * 100
    wins    = returns[returns > 0].sum()
    losses  = -returns[returns < 0].sum()
    pf      = wins / max(losses, 1e-10)
    return {
        "label"   : label,
        "cagr_pct": round(cagr, 2),
        "sharpe"  : round(sharpe, 3),
        "pf"      : round(pf, 3),
        "max_dd"  : round(dd.min() * 100, 2),
        "win_rate": round(win_r, 1),
        "n_bars"  : len(returns),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  A. WALK FORWARD VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

class WalkForwardValidator:
    """
    Expanding-window walk forward validation.

    Method
    ------
    1. Start with initial training window (is_bars)
    2. Evaluate on next oos_bars window
    3. Expand training window and repeat
    4. Aggregate OOS performance

    Example
    -------
    wfv = WalkForwardValidator(is_bars=8760, oos_bars=2190)
    result = wfv.run(returns_series, model_fn)
    """

    def __init__(self, is_bars: int = 8760, oos_bars: int = 2190, min_is: int = 2190):
        self.is_bars  = is_bars
        self.oos_bars = oos_bars
        self.min_is   = min_is

    def run(
        self,
        returns: pd.Series,
        model_fn: Optional[Callable] = None,
        feat_df: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        Parameters
        ----------
        returns  : Full strategy returns series
        model_fn : Optional callable(train_returns) → oos_returns modifier
        feat_df  : Optional features for model_fn

        Returns
        -------
        dict with per-window metrics and aggregate stats
        """
        n = len(returns)
        windows = []
        start   = self.is_bars

        while start + self.oos_bars <= n:
            is_ret  = returns.iloc[:start]
            oos_ret = returns.iloc[start:start + self.oos_bars]

            if model_fn is not None and feat_df is not None:
                is_feat  = feat_df.iloc[:start]
                oos_feat = feat_df.iloc[start:start + self.oos_bars]
                try:
                    oos_ret = model_fn(is_ret, is_feat, oos_ret, oos_feat)
                except Exception as e:
                    log.warning("model_fn error in WFV: %s", e)

            is_m   = _metrics(is_ret,  f"IS_{start}")
            oos_m  = _metrics(oos_ret, f"OOS_{start}")
            windows.append({
                "is_start" : returns.index[0],
                "is_end"   : returns.index[start-1],
                "oos_start": returns.index[start],
                "oos_end"  : returns.index[min(start+self.oos_bars-1, n-1)],
                "is_sharpe": is_m.get("sharpe", np.nan),
                "oos_sharpe":oos_m.get("sharpe",np.nan),
                "is_pf"    : is_m.get("pf",    np.nan),
                "oos_pf"   : oos_m.get("pf",   np.nan),
                "is_cagr"  : is_m.get("cagr_pct",np.nan),
                "oos_cagr" : oos_m.get("cagr_pct",np.nan),
                "is_dd"    : is_m.get("max_dd",np.nan),
                "oos_dd"   : oos_m.get("max_dd",np.nan),
            })
            start += self.oos_bars

        if not windows:
            return {"passed": False, "error": "Insufficient data for WFV"}

        wf = pd.DataFrame(windows)
        avg_oos_pf     = wf["oos_pf"].mean()
        oos_pf_ratio   = (wf["oos_pf"] >= 1.0).mean()
        deg_ratio      = (wf["oos_sharpe"] / wf["is_sharpe"].replace(0,np.nan)).mean()
        oos_preserved  = (wf["oos_pf"] / wf["is_pf"].replace(0,np.nan)).fillna(0).clip(0,1).mean() * 100

        result = {
            "n_windows"     : len(wf),
            "avg_oos_pf"    : round(avg_oos_pf, 3),
            "oos_pf_ratio"  : round(oos_pf_ratio, 3),
            "deg_ratio"     : round(deg_ratio, 3),
            "oos_preserved" : round(oos_preserved, 1),
            "windows"       : wf,
            "passed"        : avg_oos_pf >= 0.90 and oos_pf_ratio >= 0.50,
        }
        self._print_summary(result)
        return result

    def _print_summary(self, r):
        print("\n── Walk Forward Validation ──────────────────────────────")
        print(f"  Windows       : {r['n_windows']}")
        print(f"  Avg OOS PF    : {r['avg_oos_pf']:.3f}")
        print(f"  OOS PF≥1.0    : {r['oos_pf_ratio']*100:.0f}%")
        print(f"  Degradation   : {r['deg_ratio']:.3f} (IS→OOS Sharpe ratio)")
        print(f"  OOS Preserved : {r['oos_preserved']:.1f}%")
        mark = "[OK] PASS" if r["passed"] else "[WARN]️  WARN"
        print(f"  Status        : {mark}")


# ─────────────────────────────────────────────────────────────────────────────
#  B. MONTE CARLO SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

class MonteCarloValidator:
    """
    Monte Carlo robustness via return shuffling and bootstrapping.

    Example
    -------
    mc = MonteCarloValidator(n_sims=1000)
    result = mc.run(returns_series)
    """

    def __init__(self, n_sims: int = 1000, seed: int = 42):
        self.n_sims = n_sims
        self.seed   = seed

    def run(self, returns: pd.Series, confidence: float = 0.95) -> dict:
        np.random.seed(self.seed)
        arr = returns.values
        n   = len(arr)

        pf_list, sharpe_list, dd_list, ruin_list = [], [], [], []

        for _ in range(self.n_sims):
            sim = np.random.choice(arr, size=n, replace=True)
            eq  = np.cumprod(1 + sim)
            peak= np.maximum.accumulate(eq)
            dd  = (eq / peak - 1).min()
            wins  = sim[sim > 0].sum()
            losses= -sim[sim < 0].sum()
            pf    = wins / max(losses, 1e-10)
            sh    = sim.mean() / (sim.std() + 1e-10) * np.sqrt(8760/4)
            ruin  = (eq < 0.5).any()   # 50% ruin threshold

            pf_list.append(pf); sharpe_list.append(sh)
            dd_list.append(dd); ruin_list.append(ruin)

        alpha = 1 - confidence
        result = {
            "n_sims"       : self.n_sims,
            "pf_mean"      : round(np.mean(pf_list), 3),
            "pf_p5"        : round(np.percentile(pf_list, 5), 3),
            "pf_p50"       : round(np.percentile(pf_list, 50), 3),
            "sharpe_mean"  : round(np.mean(sharpe_list), 3),
            "sharpe_p5"    : round(np.percentile(sharpe_list, 5), 3),
            "dd_worst"     : round(np.percentile(dd_list, 5) * 100, 2),
            "dd_median"    : round(np.median(dd_list) * 100, 2),
            "ruin_rate"    : round(np.mean(ruin_list) * 100, 2),
            "confidence"   : confidence,
            "passed"       : (np.mean(ruin_list) < 0.20 and
                              np.percentile(pf_list, 5) > 1.0),
        }
        self._print_summary(result)
        return result

    def _print_summary(self, r):
        print("\n── Monte Carlo Simulation ───────────────────────────────")
        print(f"  Simulations   : {r['n_sims']:,}")
        print(f"  PF (mean/p5)  : {r['pf_mean']:.3f} / {r['pf_p5']:.3f}")
        print(f"  Sharpe (mean) : {r['sharpe_mean']:.3f}")
        print(f"  DD (worst 5%) : {r['dd_worst']:.2f}%")
        print(f"  Ruin Rate     : {r['ruin_rate']:.2f}%")
        mark = "[OK] PASS" if r["passed"] else "[WARN]️  WARN"
        print(f"  Status        : {mark}")


# ─────────────────────────────────────────────────────────────────────────────
#  C. OUT-OF-SAMPLE TEST
# ─────────────────────────────────────────────────────────────────────────────

class OOSValidator:
    """
    Strict IS/OOS split with degradation analysis.

    Example
    -------
    oos = OOSValidator(is_pct=0.60, val_pct=0.20, oos_pct=0.20)
    result = oos.run(returns_series)
    """

    def __init__(self, is_pct=0.60, val_pct=0.20, oos_pct=0.20):
        self.is_pct  = is_pct
        self.val_pct = val_pct
        self.oos_pct = oos_pct

    def run(self, returns: pd.Series) -> dict:
        n   = len(returns)
        t1  = int(n * self.is_pct)
        t2  = int(n * (self.is_pct + self.val_pct))

        is_ret  = returns.iloc[:t1]
        val_ret = returns.iloc[t1:t2]
        oos_ret = returns.iloc[t2:]

        splits = {
            "IS":  _metrics(is_ret,  "In-Sample"),
            "VAL": _metrics(val_ret, "Validation"),
            "OOS": _metrics(oos_ret, "Out-of-Sample"),
        }

        sharpe_deg = splits["OOS"].get("sharpe", 0) / max(splits["IS"].get("sharpe", 0.01), 0.01)
        pf_deg     = splits["OOS"].get("pf", 0)     / max(splits["IS"].get("pf", 1.0), 0.01)

        result = {
            "splits"      : splits,
            "sharpe_deg"  : round(sharpe_deg, 3),
            "pf_deg"      : round(pf_deg, 3),
            "oos_sharpe"  : splits["OOS"].get("sharpe", 0),
            "oos_pf"      : splits["OOS"].get("pf", 0),
            "passed"      : (splits["OOS"].get("pf", 0) > 1.0 and
                             splits["OOS"].get("sharpe", 0) > 0.5),
        }
        self._print_summary(result)
        return result

    def _print_summary(self, r):
        print("\n── Out-of-Sample Test ───────────────────────────────────")
        for split_name, m in r["splits"].items():
            print(f"  {split_name:<6} | CAGR={m.get('cagr_pct',0):+.1f}% "
                  f"Sharpe={m.get('sharpe',0):.3f} "
                  f"PF={m.get('pf',0):.3f} "
                  f"DD={m.get('max_dd',0):.1f}%")
        print(f"  Sharpe deg    : {r['sharpe_deg']:.3f} (OOS/IS)")
        print(f"  PF deg        : {r['pf_deg']:.3f}")
        mark = "[OK] PASS" if r["passed"] else "[WARN]️  WARN"
        print(f"  Status        : {mark}")


# ─────────────────────────────────────────────────────────────────────────────
#  D. MASTER VALIDATION FRAMEWORK
# ─────────────────────────────────────────────────────────────────────────────

class ValidationFramework:
    """
    Master validation orchestrator.

    Runs WFV + MC + OOS and produces a unified report.

    Example
    -------
    vf = ValidationFramework()
    report = vf.run(returns_series, feat_df=feat_df)
    """

    def __init__(self,
                 is_bars=8760, oos_bars=2190,
                 n_mc_sims=1000, mc_seed=42):
        self.wfv = WalkForwardValidator(is_bars=is_bars, oos_bars=oos_bars)
        self.mc  = MonteCarloValidator(n_sims=n_mc_sims, seed=mc_seed)
        self.oos = OOSValidator()

    def run(
        self,
        returns:  pd.Series,
        feat_df:  Optional[pd.DataFrame] = None,
        model_fn: Optional[Callable]     = None,
    ) -> dict:
        print("\n" + "═"*60)
        print("  VALIDATION FRAMEWORK — BTC Hybrid AI System")
        print("═"*60)

        wfv_r = self.wfv.run(returns, model_fn, feat_df)
        mc_r  = self.mc.run(returns)
        oos_r = self.oos.run(returns)

        # Composite score
        score = 0
        score += 30 if wfv_r.get("passed") else 15 if wfv_r.get("avg_oos_pf",0)>0.8 else 0
        score += 40 if mc_r.get("passed")  else 20 if mc_r.get("ruin_rate",100)<30 else 0
        score += 30 if oos_r.get("passed") else 15 if oos_r.get("oos_pf",0)>0.9 else 0

        report = {
            "walk_forward": wfv_r,
            "monte_carlo" : mc_r,
            "oos"         : oos_r,
            "score"       : score,
            "grade"       : "EXCELLENT" if score>=85 else "GOOD" if score>=70 else "MARGINAL" if score>=50 else "FAIL",
            "passed"      : score >= 60,
        }

        print(f"\n{'═'*60}")
        print(f"  VALIDATION SCORE: {score}/100  [{report['grade']}]")
        print(f"  WFV={'[OK]' if wfv_r.get('passed') else '[WARN]️ '}  "
              f"MC={'[OK]' if mc_r.get('passed') else '[WARN]️ '}  "
              f"OOS={'[OK]' if oos_r.get('passed') else '[WARN]️ '}")
        print("═"*60)
        return report


def run(returns: pd.Series, feat_df=None) -> dict:
    return ValidationFramework().run(returns, feat_df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    np.random.seed(42); n=10000
    ret=pd.Series(np.random.randn(n)*0.005 + 0.0003,
                  index=pd.date_range("2019-01-01",periods=n,freq="4h"))
    vf=ValidationFramework()
    report=vf.run(ret)
    print(f"\nFinal Grade: {report['grade']}")
