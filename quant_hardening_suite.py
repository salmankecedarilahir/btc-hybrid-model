"""
╔══════════════════════════════════════════════════════════════════════╗
║   quant_hardening_suite.py — BTC Hybrid Model V7                   ║
║   MASTER RUNNER: Final Quant Core Hardening Suite                  ║
╠══════════════════════════════════════════════════════════════════════╣
║  Menjalankan semua 8 bagian secara berurutan:                       ║
║                                                                     ║
║  Bagian 1: Dataset Integrity Audit     → dataset_audit.py          ║
║  Bagian 2: Feature Stability Test      → feature_stability_test.py ║
║  Bagian 3: Regime Sensitivity Analysis → regime_sensitivity_test.py║
║  Bagian 4: Parameter Stability Test    → parameter_stability_test.py║
║  Bagian 5: Out-of-Sample Validation    → out_of_sample_validation.py║
║  Bagian 6: Robustness Score Engine     → robustness_score_engine.py║
║  Bagian 7: Model Readiness Check       → (dalam robustness_score)  ║
║  Bagian 8: AI Integration Preparation → (dalam robustness_score)   ║
╠══════════════════════════════════════════════════════════════════════╣
║  Cara pakai:                                                        ║
║    python quant_hardening_suite.py               # full suite      ║
║    python quant_hardening_suite.py --fast        # skip slow tests ║
║    python quant_hardening_suite.py --stage 1     # single stage    ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

DIV = "═" * 70
SEP = "─" * 70


def run_stage(name: str, fn, results: dict) -> dict:
    print(f"\n{DIV}")
    print(f"  RUNNING: {name}")
    print(DIV)
    t0 = time.time()
    try:
        result = fn()
        elapsed = time.time() - t0
        status  = "[OK] PASS" if result.get("passed", True) else "[WARN]️  WARN"
        log.info("%s completed in %.1fs — %s", name, elapsed, status)
        return result
    except Exception as e:
        elapsed = time.time() - t0
        log.error("%s FAILED in %.1fs: %s", name, elapsed, e)
        import traceback
        traceback.print_exc()
        return {"passed": False, "error": str(e)}


def main(stages: list = None, fast: bool = False):
    t_total = time.time()

    print(f"\n{DIV}")
    print("  BTC HYBRID MODEL V7 — FINAL QUANT CORE HARDENING SUITE")
    print(f"  {'FAST MODE (skip heavy tests)' if fast else 'FULL MODE (all tests)'}")
    print(DIV)

    results = {}
    all_stages = [1, 2, 3, 4, 5, 6]
    if stages:
        run_stages = stages
    else:
        run_stages = all_stages

    # ── Stage 1: Dataset Integrity ────────────────────────────────
    if 1 in run_stages:
        from dataset_audit import run as run_s1
        results["dataset_audit"] = run_stage("Stage 1: Dataset Integrity Audit", run_s1, results)

    # ── Stage 2: Feature Stability ────────────────────────────────
    if 2 in run_stages:
        from feature_stability_test import run as run_s2
        results["feature_stability"] = run_stage("Stage 2: Feature Stability Test", run_s2, results)

    # ── Stage 3: Regime Sensitivity ──────────────────────────────
    if 3 in run_stages:
        from regime_sensitivity_test import run as run_s3
        results["regime_sensitivity"] = run_stage("Stage 3: Regime Sensitivity Analysis", run_s3, results)

    # ── Stage 4: Parameter Stability ──────────────────────────────
    if 4 in run_stages:
        from parameter_stability_test import run as run_s4
        results["parameter_stability"] = run_stage("Stage 4: Parameter Stability Test", run_s4, results)

    # ── Stage 5: OOS Validation ───────────────────────────────────
    if 5 in run_stages:
        from out_of_sample_validation import run as run_s5
        results["oos_validation"] = run_stage("Stage 5: Out-of-Sample Validation", run_s5, results)

    # ── Stage 6+7+8: Robustness Score + Readiness + AI Prep ──────
    if 6 in run_stages:
        from robustness_score_engine import run as run_s6
        results["robustness_engine"] = run_stage(
            "Stage 6+7+8: Robustness Score + Readiness + AI Prep",
            lambda: run_s6(run_all=False),
            results
        )

    # ── Final Summary ─────────────────────────────────────────────
    elapsed = time.time() - t_total

    print(f"\n{DIV}")
    print("  QUANT HARDENING SUITE — COMPLETE SUMMARY")
    print(DIV)

    stage_names = {
        "dataset_audit":      "1. Dataset Integrity",
        "feature_stability":  "2. Feature Stability",
        "regime_sensitivity": "3. Regime Sensitivity",
        "parameter_stability":"4. Parameter Stability",
        "oos_validation":     "5. OOS Validation",
        "robustness_engine":  "6-8. Robustness+Readiness+AI",
    }

    all_pass = True
    for key, name in stage_names.items():
        if key not in results:
            continue
        r = results[key]
        passed = r.get("passed", True) and "error" not in r
        all_pass = all_pass and passed

        score_str = ""
        for score_key in ["dataset_audit_score","feature_stability_score",
                          "regime_sensitivity_score","parameter_stability_score",
                          "oos_validation_score","total_score"]:
            if score_key in r:
                score_str = f"  score={r[score_key]:.1f}"
                break

        mark = "[OK]" if passed else ("❌" if "error" in r else "[WARN]️ ")
        print(f"  {mark}  {name:<35}{score_str}")

    # Final status
    rob = results.get("robustness_engine", {})
    print(f"\n{SEP}")
    if "total_score" in rob:
        print(f"  Total Robustness Score : {rob['total_score']:.1f}/100  [{rob.get('grade','?')}]")
    print(f"  Total elapsed          : {elapsed:.1f}s")
    print()

    status = rob.get("status", "UNKNOWN")
    emoji  = "[GREEN]" if rob.get("ready", False) else "[RED]"
    print(f"  {emoji}  FINAL STATUS: {status}")
    print(DIV)

    print(f"""
  NEXT STEPS:
  ──────────────────────────────────────────────────────────────
  {'[OK] Model is ready! Proceed to AI layer:' if rob.get('ready') else '❌ Fix blocking criteria first, then:'}
    1. python ai_dataset_builder.py         (build training data)
    2. python feature_stability_test.py     (validate features)
    3. Train AI model (LightGBM recommended)
    4. Backtest with AI filter vs without
    5. MC simulation with AI filter
    6. Paper trade 30+ days
    7. Live deployment with kill switch active
  ──────────────────────────────────────────────────────────────
  """)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BTC Hybrid Model V7 — Quant Core Hardening Suite"
    )
    parser.add_argument("--stage", type=int, nargs="+",
                        help="Specific stages to run (1-6)")
    parser.add_argument("--fast", action="store_true",
                        help="Skip computationally heavy tests")
    args = parser.parse_args()

    main(stages=args.stage, fast=args.fast)
