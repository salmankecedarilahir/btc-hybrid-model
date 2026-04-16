"""
╔══════════════════════════════════════════════════════════════════════╗
║   robustness_score_engine.py — BTC Hybrid Model V7                 ║
║   BAGIAN 6+7+8: Robustness Score + Model Readiness + AI Prep      ║
╠══════════════════════════════════════════════════════════════════════╣
║  Bagian 6: Aggregate all hardening test results → single score     ║
║  Bagian 7: MODEL_READY_FOR_AI_LAYER gate                          ║
║  Bagian 8: AI dataset preparation guide & validation               ║
╠══════════════════════════════════════════════════════════════════════╣
║  Cara pakai:                                                        ║
║    python robustness_score_engine.py                               ║
║    python robustness_score_engine.py --run-all     (full hardening)║
╚══════════════════════════════════════════════════════════════════════╝
"""

import argparse
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
AI_PATH   = DATA_DIR / "ai_training_dataset.csv"
MC_PATH   = DATA_DIR / "mc_results_bar.csv"
WF_PATH   = DATA_DIR / "walkforward_results.csv"
OOS_PATH  = DATA_DIR / "oos_validation.csv"
REG_PATH  = DATA_DIR / "regime_sensitivity.csv"
PARAM_PATH = DATA_DIR / "parameter_stability.csv"
FINAL_OUT = DATA_DIR / "final_readiness_report.csv"

BARS_PER_YEAR = 2190
INIT          = 10_000.0

DIV = "═" * 70
SEP = "─" * 70


def ok(msg):   print(f"  [OK] {msg}")
def warn(msg): print(f"  [WARN]️  {msg}")
def err(msg):  print(f"  ❌ {msg}")


# ══════════════════════════════════════════════════════════════════
#  BAGIAN 6: ROBUSTNESS SCORE ENGINE
# ══════════════════════════════════════════════════════════════════

class RobustnessScoreEngine:
    """
    Aggregate all hardening test results into a single robustness score.

    Score components (total = 100 pts):
      A. Quant Performance (25pts)  — backtest metrics
      B. Dataset Integrity (15pts)  — data quality
      C. Monte Carlo (20pts)        — worst-case distributional robustness
      D. Walk-Forward (15pts)       — out-of-sample consistency
      E. Regime Sensitivity (10pts) — cross-regime performance
      F. Parameter Stability (10pts)— perturbation resistance
      G. OOS Validation (5pts)      — true holdout performance
    """

    WEIGHTS = {
        "quant_performance": 25,
        "dataset_integrity": 15,
        "monte_carlo":       20,
        "walk_forward":      15,
        "regime_sensitivity": 10,
        "parameter_stability":10,
        "oos_validation":     5,
    }

    def __init__(self):
        self.scores   = {}
        self.details  = {}
        self.errors   = []

    def score_quant_performance(self) -> float:
        """Score based on backtest metrics from risk_managed."""
        path = RISK_PATH if RISK_PATH.exists() else BT_PATH
        if not path.exists():
            self.errors.append("No backtest data")
            return 0.0

        df      = pd.read_csv(path)
        ret_col = "equity_return" if "equity_return" in df.columns else "strategy_return"
        ret     = df[ret_col].fillna(0).values
        ret_nz  = ret[ret != 0]

        if len(ret_nz) < 100:
            return 0.0

        eq      = INIT * np.cumprod(1.0 + ret)
        n_years = len(ret) / BARS_PER_YEAR
        cagr    = ((eq[-1] / INIT) ** (1 / max(n_years, 0.01)) - 1) * 100
        peak    = np.maximum.accumulate(eq)
        peak    = np.where(peak > 0, peak, 1e-9)
        max_dd  = float(((eq - peak) / peak).min() * 100)
        mu, sg  = ret_nz.mean(), ret_nz.std()
        sharpe  = (mu / sg) * np.sqrt(BARS_PER_YEAR) if sg > 1e-10 else 0.0
        wins    = ret_nz[ret_nz > 0]
        losses  = ret_nz[ret_nz < 0]
        pf      = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else 99.0

        self.details["bt_cagr"]   = round(cagr, 2)
        self.details["bt_sharpe"] = round(sharpe, 3)
        self.details["bt_pf"]     = round(min(pf, 99), 4)
        self.details["bt_max_dd"] = round(max_dd, 2)

        # Score each metric
        s_cagr   = 7 if cagr > 100 else 5 if cagr > 50 else 3 if cagr > 20 else 1
        s_sharpe = 7 if sharpe > 2.0 else 5 if sharpe > 1.5 else 3 if sharpe > 1.0 else 0
        s_pf     = 6 if pf > 1.5 else 4 if pf > 1.3 else 2 if pf > 1.1 else 0
        s_dd     = 5 if abs(max_dd) < 25 else 3 if abs(max_dd) < 35 else 1 if abs(max_dd) < 50 else 0

        score = s_cagr + s_sharpe + s_pf + s_dd
        log.info("Quant Performance: CAGR=%.1f%% Sharpe=%.3f PF=%.3f MaxDD=%.1f%% → %d/25",
                 cagr, sharpe, pf, max_dd, score)
        return float(score)

    def score_dataset_integrity(self) -> float:
        """Score based on dataset audit results (if available)."""
        # Try to load audit results; if not available compute from data
        score = 15.0  # assume clean unless evidence otherwise

        path = RISK_PATH if RISK_PATH.exists() else BT_PATH
        if not path.exists():
            return 0.0

        df = pd.read_csv(path)

        # NaN check
        ret_col = "equity_return" if "equity_return" in df.columns else "strategy_return"
        n_nan = df[ret_col].isna().sum()
        if n_nan > 0:
            score -= min(5, n_nan // 100)

        # Duplicate timestamps
        if "timestamp" in df.columns:
            n_dup = df["timestamp"].duplicated().sum()
            if n_dup > 0:
                score -= min(5, n_dup)

        # Equity monotonic at TIER2 paused bars
        if "kill_switch_active" in df.columns and "equity" in df.columns:
            ks    = df["kill_switch_active"].values
            eq    = df["equity"].values
            paused_diff = np.diff(eq)[ks[1:] == 1]
            n_moving_paused = (np.abs(paused_diff) > 1.0).sum()
            if n_moving_paused > 10:
                score -= 3

        self.details["dataset_integrity_score"] = score
        log.info("Dataset Integrity score: %.0f/15", score)
        return max(0.0, score)

    def score_monte_carlo(self) -> float:
        """Score based on MC robustness results."""
        if not MC_PATH.exists():
            warn("MC results not found — run validation_framework.py first")
            return 10.0  # partial credit

        mc_df = pd.read_csv(MC_PATH)
        if mc_df.empty:
            return 0.0

        r = mc_df.iloc[0].to_dict()
        ruin_pct      = r.get("ruin_pct", 100)
        pf_median     = r.get("pf_median", 0)
        pf_p5         = r.get("pf_p5", 0)
        sharpe_median = r.get("sharpe_median", 0)
        eq_pos_pct    = r.get("eq_pct_positive", 0)

        self.details["mc_ruin_pct"]      = ruin_pct
        self.details["mc_pf_median"]     = pf_median
        self.details["mc_pf_p5"]         = pf_p5
        self.details["mc_sharpe_median"] = sharpe_median
        self.details["mc_eq_pos_pct"]    = eq_pos_pct

        s_ruin   = 6 if ruin_pct < 5 else 4 if ruin_pct < 15 else 2 if ruin_pct < 30 else 0
        s_pf_med = 6 if pf_median > 1.3 else 4 if pf_median > 1.1 else 2 if pf_median > 1.0 else 0
        s_sharpe = 5 if sharpe_median > 1.5 else 3 if sharpe_median > 1.0 else 1 if sharpe_median > 0.5 else 0
        s_cons   = 3 if eq_pos_pct > 95 else 2 if eq_pos_pct > 80 else 0

        score = s_ruin + s_pf_med + s_sharpe + s_cons
        log.info("Monte Carlo: ruin=%.2f%% pf_med=%.3f sh_med=%.3f → %d/20",
                 ruin_pct, pf_median, sharpe_median, score)
        return float(min(score, 20))

    def score_walk_forward(self) -> float:
        """Score based on walk-forward results."""
        if not WF_PATH.exists():
            warn("Walk-forward results not found — run validation_framework.py first")
            return 7.0  # partial credit

        wf_df = pd.read_csv(WF_PATH)
        if wf_df.empty:
            return 0.0

        # OOS preserved windows (PF≥1.0 or CAGR>-5%)
        if "oos_pf" in wf_df.columns and "oos_cagr" in wf_df.columns:
            oos_preserved = ((wf_df["oos_pf"] >= 1.0) | (wf_df["oos_cagr"] > -5.0)).mean() * 100
        elif "oos_cagr" in wf_df.columns:
            oos_preserved = (wf_df["oos_cagr"] > 0).mean() * 100
        else:
            oos_preserved = 50.0

        avg_pf_ratio  = wf_df["pf_ratio"].mean() if "pf_ratio" in wf_df.columns else 0.5
        median_oos_pf = wf_df["oos_pf"].median() if "oos_pf" in wf_df.columns else 0.9

        self.details["wf_oos_preserved"] = round(oos_preserved, 1)
        self.details["wf_pf_ratio"]      = round(avg_pf_ratio, 3)
        self.details["wf_median_oos_pf"] = round(median_oos_pf, 3)

        s_pos    = 6 if oos_preserved >= 66 else 4 if oos_preserved >= 50 else 2 if oos_preserved >= 33 else 0
        s_ratio  = 5 if avg_pf_ratio >= 0.80 else 3 if avg_pf_ratio >= 0.60 else 1
        s_oos_pf = 4 if median_oos_pf >= 1.0 else 2 if median_oos_pf >= 0.90 else 0

        score = s_pos + s_ratio + s_oos_pf
        log.info("Walk-Forward: oos_preserved=%.0f%% pf_ratio=%.3f → %d/15",
                 oos_preserved, avg_pf_ratio, score)
        return float(min(score, 15))

    def score_regime_sensitivity(self) -> float:
        """Score based on regime analysis."""
        if not REG_PATH.exists():
            return 7.0  # partial credit if not run

        reg_df = pd.read_csv(REG_PATH)
        if reg_df.empty or "pf" not in reg_df.columns:
            return 7.0

        reg_df = reg_df[reg_df["label"] != "ALL"].dropna(subset=["pf"])
        if reg_df.empty:
            return 7.0

        pfs     = reg_df["pf"].values
        n_pf_ok = (pfs > 0.90).sum()
        pf_cv   = np.std(pfs) / max(np.mean(pfs), 0.1)

        s_ok = 6 if n_pf_ok == len(pfs) else 4 if n_pf_ok >= len(pfs)*0.7 else 2
        s_cv = 4 if pf_cv < 0.20 else 2 if pf_cv < 0.40 else 0

        score = s_ok + s_cv
        log.info("Regime Sensitivity: %d/%d regimes PF>0.90, CoV=%.3f → %d/10",
                 n_pf_ok, len(pfs), pf_cv, score)
        return float(min(score, 10))

    def score_parameter_stability(self) -> float:
        """Score based on parameter perturbation tests."""
        if not PARAM_PATH.exists():
            return 7.0  # partial credit if not run

        param_df = pd.read_csv(PARAM_PATH)
        n_pass   = param_df["passed"].sum() if "passed" in param_df.columns else 3
        n_total  = len(param_df) if len(param_df) > 0 else 5

        score = (n_pass / max(n_total, 1)) * 10
        log.info("Parameter Stability: %d/%d tests pass → %.0f/10", n_pass, n_total, score)
        return float(min(score, 10))

    def score_oos_validation(self) -> float:
        """Score based on true OOS performance."""
        if not OOS_PATH.exists():
            return 3.0  # partial credit

        oos_df = pd.read_csv(OOS_PATH)
        if oos_df.empty:
            return 0.0

        # Get OOS row
        oos_row = oos_df[oos_df["label"].str.contains("OOS", case=False, na=False)]
        if oos_row.empty:
            return 3.0

        oos_pf     = float(oos_row["pf"].iloc[0]) if "pf" in oos_row.columns else 0
        oos_sharpe = float(oos_row["sharpe"].iloc[0]) if "sharpe" in oos_row.columns else 0

        s_pf     = 3 if oos_pf > 1.1 else 2 if oos_pf > 1.0 else 0
        s_sharpe = 2 if oos_sharpe > 0.8 else 1 if oos_sharpe > 0.3 else 0

        score = s_pf + s_sharpe
        log.info("OOS Validation: pf=%.3f sharpe=%.3f → %d/5", oos_pf, oos_sharpe, score)
        return float(min(score, 5))

    def compute_total(self) -> dict:
        log.info("Computing robustness scores...")

        self.scores["quant_performance"]  = self.score_quant_performance()
        self.scores["dataset_integrity"]  = self.score_dataset_integrity()
        self.scores["monte_carlo"]        = self.score_monte_carlo()
        self.scores["walk_forward"]       = self.score_walk_forward()
        self.scores["regime_sensitivity"] = self.score_regime_sensitivity()
        self.scores["parameter_stability"]= self.score_parameter_stability()
        self.scores["oos_validation"]     = self.score_oos_validation()

        total = sum(self.scores.values())
        max_t = sum(self.WEIGHTS.values())

        if   total >= 85: grade = "EXCELLENT"
        elif total >= 70: grade = "GOOD"
        elif total >= 55: grade = "FAIR"
        elif total >= 40: grade = "POOR"
        else:             grade = "CRITICAL"

        return {
            "total_score":  round(total, 1),
            "max_score":    max_t,
            "grade":        grade,
            "components":   self.scores,
            "details":      self.details,
        }

    def print_report(self, result: dict) -> None:
        print(f"\n{DIV}")
        print("  BAGIAN 6 — ROBUSTNESS SCORE ENGINE")
        print(DIV)
        print(f"\n  {'Component':<25} {'Score':>8} {'Max':>6} {'Pct':>8}")
        print("  " + "─"*50)
        for key, score in result["components"].items():
            max_s = self.WEIGHTS[key]
            pct   = score / max_s * 100
            bar   = "█" * int(pct / 10)
            print(f"  {key.replace('_',' ').title():<25} "
                  f"{score:>7.1f}  {max_s:>5}  {pct:>6.0f}%  {bar}")
        print("  " + "─"*50)
        total = result["total_score"]
        print(f"  {'TOTAL ROBUSTNESS SCORE':<25} {total:>7.1f} / {result['max_score']}")
        print(f"  {'GRADE':<25} {result['grade']}")

        # Key details
        d = result["details"]
        if d:
            print(f"\n  Key Metrics:")
            if "bt_cagr" in d:   print(f"    Backtest CAGR      : {d['bt_cagr']:+.2f}%")
            if "bt_sharpe" in d: print(f"    Backtest Sharpe    : {d['bt_sharpe']:.3f}")
            if "bt_pf" in d:     print(f"    Backtest PF        : {d['bt_pf']:.4f}")
            if "bt_max_dd" in d: print(f"    Backtest MaxDD     : {d['bt_max_dd']:.2f}%")
            if "mc_ruin_pct" in d: print(f"    MC Ruin Rate       : {d['mc_ruin_pct']:.3f}%")
            if "mc_pf_median" in d: print(f"    MC PF Median       : {d['mc_pf_median']:.3f}")
            if "wf_oos_preserved" in d: print(f"    WF OOS Preserved   : {d['wf_oos_preserved']:.0f}%")
        print(f"{DIV}")


# ══════════════════════════════════════════════════════════════════
#  BAGIAN 7: MODEL READINESS CHECK
# ══════════════════════════════════════════════════════════════════

class ModelReadinessChecker:
    """
    Definitive GO / NO-GO gate untuk AI layer integration.

    10 Blocking criteria — semua harus PASS.
    4  Advisory criteria — warning only.
    """

    BLOCKING = {
        "bt_pf_gte_1_4":        ("BT Profit Factor ≥ 1.4",                1.4,  "pf",  ">="),
        "bt_sharpe_gte_1_3":    ("BT Sharpe Ratio ≥ 1.3",                1.3,  "sharpe", ">="),
        "bt_max_dd_gte_neg40":  ("BT MaxDD ≥ -40% (risk_managed)",       -40.0,"max_dd",">="),
        "bt_cagr_gte_50":       ("BT CAGR ≥ 50%",                         50.0,"cagr", ">="),
        "mc_pf_p5_gt_1":        ("MC p5 PF > 1.0  (worst 5% sims)",       1.0, "mc_pf_p5",">="),
        "mc_sharpe_med_gt_08":  ("MC Median Sharpe > 0.8",                0.8, "mc_sharpe_median",">="),
        "mc_ruin_lt_20":        ("MC Ruin Rate < 20% (KS adj, 80% thresh)",20.0,"mc_ruin_pct","<"),
        "wf_oos_pres_gte_33":   ("WF OOS Preserved ≥ 33%",               33.0,"wf_oos_preserved",">="),
        "wf_median_oos_pf_09":  ("WF Median OOS PF > 0.90",               0.90,"wf_median_oos_pf",">="),
        "rob_score_gte_55":     ("Total Robustness Score ≥ 55",           55.0,"total_score",">="),
    }

    ADVISORY = {
        "bt_sortino_gte_1":     ("BT Sortino ≥ 1.0",        1.0, "sortino", ">="),
        "wf_pf_ratio_gte_07":   ("WF avg PF ratio ≥ 0.70",  0.70,"wf_pf_ratio",">="),
        "regime_pf_cv_lt_05":   ("Regime PF CoV < 0.50",    0.50,"regime_pf_cv","<"),
        "param_stable":         ("Parameter Stability ≥ 70%",70.0,"param_score",">="),
    }

    def _eval(self, val, thresh, op):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return False
        if op == ">=": return val >= thresh
        if op == ">":  return val > thresh
        if op == "<":  return val < thresh
        if op == "<=": return val <= thresh
        return False

    def run(self, robustness_result: dict) -> dict:
        print(f"\n{DIV}")
        print("  BAGIAN 7 — MODEL READINESS CHECK — AI LAYER GATE")
        print(DIV)

        # Gather all values
        d     = robustness_result.get("details", {})
        comp  = robustness_result.get("components", {})
        total = robustness_result.get("total_score", 0)

        values = {
            "pf":                  d.get("bt_pf"),
            "sharpe":              d.get("bt_sharpe"),
            "max_dd":              d.get("bt_max_dd"),
            "cagr":                d.get("bt_cagr"),
            "mc_pf_p5":            d.get("mc_pf_p5"),
            "mc_sharpe_median":    d.get("mc_sharpe_median"),
            "mc_ruin_pct":         d.get("mc_ruin_pct"),
            "wf_oos_preserved":    d.get("wf_oos_preserved"),
            "wf_median_oos_pf":    d.get("wf_median_oos_pf"),
            "wf_pf_ratio":         d.get("wf_pf_ratio"),
            "total_score":         total,
            "sortino":             None,  # not directly tracked
            "regime_pf_cv":        None,
            "param_score":         comp.get("parameter_stability", 0) / 0.10
                                   if comp.get("parameter_stability") is not None else None,
        }

        # Try to compute sortino from data
        path = RISK_PATH if RISK_PATH.exists() else BT_PATH
        if path.exists():
            df     = pd.read_csv(path)
            ret_col = "equity_return" if "equity_return" in df.columns else "strategy_return"
            ret_nz  = df[ret_col].fillna(0)
            ret_nz  = ret_nz[ret_nz != 0].values
            if len(ret_nz) > 10:
                mu     = ret_nz.mean()
                neg_r  = ret_nz[ret_nz < 0]
                if len(neg_r) > 0 and neg_r.std() > 1e-10:
                    values["sortino"] = float((mu / neg_r.std()) * np.sqrt(BARS_PER_YEAR))

        # Evaluate blocking
        print(f"\n  ── BLOCKING CRITERIA (all must PASS) ──")
        n_block_pass = 0
        blocking_results = {}
        for key, (label, thresh, val_key, op) in self.BLOCKING.items():
            val    = values.get(val_key)
            passed = self._eval(val, thresh, op)
            n_block_pass += passed
            blocking_results[key] = {"passed": passed, "value": val}
            val_s = f"{val:.2f}" if val is not None and not np.isnan(float(val)) else "N/A"
            mark  = "[OK] PASS" if passed else "❌ FAIL"
            print(f"  {mark}  {label:<48}  (value={val_s})")

        # Evaluate advisory
        print(f"\n  ── ADVISORY CRITERIA (warning only) ──")
        n_adv_pass = 0
        for key, (label, thresh, val_key, op) in self.ADVISORY.items():
            val    = values.get(val_key)
            passed = self._eval(val, thresh, op)
            n_adv_pass += passed
            val_s = f"{val:.3f}" if val is not None and not (isinstance(val,float) and np.isnan(val)) else "N/A"
            mark  = "[OK]" if passed else "[WARN]️ "
            print(f"  {mark}  {label:<48}  (value={val_s})")

        all_block_pass = (n_block_pass == len(self.BLOCKING))

        print(f"\n{SEP}")
        print(f"  Blocking: {n_block_pass}/{len(self.BLOCKING)} PASS")
        print(f"  Advisory: {n_adv_pass}/{len(self.ADVISORY)} PASS")
        print(SEP)

        if all_block_pass:
            if n_adv_pass >= 3:
                status = "MODEL_READY_FOR_AI_LAYER"
                emoji  = "[GREEN]"
            else:
                status = "MODEL_READY_FOR_AI_LAYER (advisory warnings)"
                emoji  = "[YELLOW]"
        else:
            n_fail = len(self.BLOCKING) - n_block_pass
            status = f"MODEL_NOT_READY ({n_fail} blocking criteria failed)"
            emoji  = "[RED]"

        print(f"\n  {emoji}  STATUS: {status}")
        print(f"{DIV}")

        return {
            "status":         status,
            "ready":          all_block_pass,
            "n_block_pass":   n_block_pass,
            "n_block_total":  len(self.BLOCKING),
            "n_adv_pass":     n_adv_pass,
            "blocking":       blocking_results,
        }


# ══════════════════════════════════════════════════════════════════
#  BAGIAN 8: AI INTEGRATION PREPARATION
# ══════════════════════════════════════════════════════════════════

def prepare_ai_integration(readiness_result: dict) -> dict:
    print(f"\n{DIV}")
    print("  BAGIAN 8 — AI INTEGRATION PREPARATION")
    print(DIV)

    if not readiness_result.get("ready", False):
        print(f"\n  [WARN]️  Model belum READY — selesaikan blocking criteria dulu")
        print(f"  Tapi kita bisa validasi dataset AI tetap tersedia...")

    # ── Check AI dataset ─────────────────────────────────────────
    print(f"\n  ── AI DATASET CHECK ──")
    ai_ready = False
    if AI_PATH.exists():
        ai_df   = pd.read_csv(AI_PATH)
        n_rows  = len(ai_df)
        n_cols  = len(ai_df.columns)

        # Check target variables
        target_cols = [c for c in ai_df.columns if c.startswith("target_")]

        # Check feature NaN
        num_feats = ai_df.select_dtypes(include=np.number)
        nan_pct   = num_feats.isna().mean().max() * 100

        # Class balance (for target_direction_1bar)
        if "target_direction_1bar" in ai_df.columns:
            balance = ai_df["target_direction_1bar"].value_counts(normalize=True) * 100
        else:
            balance = pd.Series()

        print(f"  Dataset size        : {n_rows:,} rows × {n_cols} cols")
        print(f"  Target variables    : {target_cols}")
        print(f"  Max feature NaN     : {nan_pct:.2f}%")
        if not balance.empty:
            print(f"  Target distribution :")
            for cls, pct in balance.items():
                print(f"    {cls}: {pct:.1f}%")

        ai_ready = (n_rows >= 5000 and len(target_cols) >= 1 and nan_pct < 10)
    else:
        print(f"  ❌ ai_training_dataset.csv not found")
        print(f"     Run: python ai_dataset_builder.py")

    # ── AI Model recommendations ─────────────────────────────────
    print(f"\n  ── RECOMMENDED AI ARCHITECTURE ──")
    print(f"""
  TASK TYPE: Supervised Learning — Signal Enhancement
  ─────────────────────────────────────────────────────────────────
  Primary Task    : Classification — will next trade be profitable?
                    Target: target_profitable_trade (binary: 0/1)

  Secondary Task  : Regression — expected return next N bars
                    Target: target_ret_4bar, target_ret_12bar

  ─────────────────────────────────────────────────────────────────
  RECOMMENDED MODELS:
  1. LightGBM / XGBoost    ← best for tabular, handles NaN
  2. Random Forest         ← robust, interpretable, no overfitting
  3. LSTM / Transformer    ← if temporal patterns are important
  4. Ensemble (1+2)        ← most robust for production

  ─────────────────────────────────────────────────────────────────
  FEATURE ENGINEERING RECOMMENDATIONS:
  • Lag features: add ret_1bar_lag1, ret_1bar_lag2, ..., lag5
  • Rolling stats: 24-bar rolling mean/std of each feature
  • Interaction: regime × signal_encoded
  • Normalize: StandardScaler or MinMaxScaler per feature
  • Remove: high-corr features (r>0.90) — keep lower NaN rate one
  • Exclude: equity, shadow_equity, running_max_equity (leakage!)

  ─────────────────────────────────────────────────────────────────
  TRAINING STRATEGY:
  • Use TimeSeriesSplit (NOT random train/test split!)
  • Walk-forward validation with expanding window
  • Min train set: 2 years (~17,520 bars)
  • Feature selection: use SHAP values after first training
  • Regularization: high dropout / L2 to prevent overfitting
  • Early stopping: validate on forward-looking 6-month window

  ─────────────────────────────────────────────────────────────────
  INTEGRATION WITH QUANT CORE:
  Quant Core Signal (current) → AI Filter → Enhanced Signal
                          ↑
                   AI acts as a FILTER, not a replacement:
                   • If quant says LONG & AI says "low confidence" → reduce position
                   • If quant says LONG & AI says "high confidence" → full position
                   • Never override quant core kill switch
                   • AI confidence threshold: 0.55 minimum (avoid overfit)

  ─────────────────────────────────────────────────────────────────
  VALIDATION BEFORE DEPLOYMENT:
  1. Backtest with AI filter: CAGR must be ≥ quant-only CAGR
  2. MC simulation with AI filter: ruin rate must not increase
  3. Paper trade 30 days before live
  4. Monitor: feature drift (PSI > 0.2 = retrain)
  5. Retrain frequency: every 6 months or after major regime shift
    """)

    # ── Dataset preparation steps ─────────────────────────────────
    print(f"  ── DATASET PREPARATION STEPS ──")
    print(f"""
  1. Run full pipeline:  .\\run_pipeline.bat
  2. Build AI dataset:   python ai_dataset_builder.py
  3. Run audit:          python dataset_audit.py
  4. Split data:
       IS  (train)   : 2017-01 → 2022-01  (~60%)
       VAL (tune)    : 2022-01 → 2024-01  (~20%)
       OOS (test)    : 2024-01 → 2026-03  (~20%)
  5. Feature selection:
       python feature_stability_test.py
       Remove features with variance CoV > 5x or NaN > 5%
  6. Final training:     Use IS + VAL for final model
  7. Final evaluation:   Evaluate ONLY on OOS (never use for tuning!)
    """)

    return {
        "ai_dataset_ready": ai_ready,
        "n_rows": n_rows if AI_PATH.exists() else 0,
        "n_targets": len(target_cols) if AI_PATH.exists() else 0,
    }


# ══════════════════════════════════════════════════════════════════
#  MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════

def run(run_all: bool = False) -> dict:
    """
    run_all=True: run all hardening modules first then score
    run_all=False: use existing results from previous runs
    """

    if run_all:
        log.info("Running all hardening modules...")
        try:
            from dataset_audit          import run as run_da
            from feature_stability_test import run as run_fs
            from regime_sensitivity_test import run as run_rs
            from parameter_stability_test import run as run_ps
            from out_of_sample_validation import run as run_oos
            run_da(); run_fs(); run_rs(); run_ps(); run_oos()
        except ImportError as e:
            log.warning("Could not import sub-modules: %s", e)

    print(f"\n{DIV}")
    print("  FINAL QUANT CORE HARDENING — BTC Hybrid Model V7")
    print(f"{DIV}")

    # Bagian 6: Compute robustness score
    engine  = RobustnessScoreEngine()
    rob_res = engine.compute_total()
    engine.print_report(rob_res)

    # Bagian 7: Model readiness check
    checker     = ModelReadinessChecker()
    ready_res   = checker.run(rob_res)

    # Bagian 8: AI preparation
    ai_res = prepare_ai_integration(ready_res)

    # Save final report
    report_row = {
        "timestamp":       pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_score":     rob_res["total_score"],
        "grade":           rob_res["grade"],
        "status":          ready_res["status"],
        "ready":           ready_res["ready"],
        "n_block_pass":    ready_res["n_block_pass"],
        "ai_dataset_ready":ai_res["ai_dataset_ready"],
    }
    for k, v in rob_res["components"].items():
        report_row[f"score_{k}"] = v
    for k, v in rob_res["details"].items():
        report_row[k] = v

    pd.DataFrame([report_row]).to_csv(FINAL_OUT, index=False)
    log.info("Final report saved → %s", FINAL_OUT)

    print(f"\n{DIV}")
    print("  FINAL SUMMARY")
    print(DIV)
    print(f"  Robustness Score : {rob_res['total_score']:.1f}/100  [{rob_res['grade']}]")
    print(f"  Blocking Pass    : {ready_res['n_block_pass']}/{ready_res['n_block_total']}")
    print(f"  AI Dataset Ready : {'[OK] YES' if ai_res['ai_dataset_ready'] else '❌ NO'}")
    print()
    emoji = "[GREEN]" if ready_res["ready"] else "[RED]"
    print(f"  {emoji}  FINAL STATUS: {ready_res['status']}")
    print(DIV)

    return {
        "total_score":   rob_res["total_score"],
        "grade":         rob_res["grade"],
        "status":        ready_res["status"],
        "ready":         ready_res["ready"],
        "ai_ready":      ai_res["ai_dataset_ready"],
        "details":       rob_res,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-all", action="store_true",
                        help="Run all sub-modules before scoring")
    args = parser.parse_args()
    run(run_all=args.run_all)
