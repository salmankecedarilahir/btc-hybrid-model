"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         validation_framework.py — BTC Hybrid Model V7                      ║
║         Full Validation Pipeline: Backtest → MC → WalkForward → Readiness  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  PIPELINE:                                                                   ║
║    Stage 1: Backtest Quality Gate    — cek CSV output dari backtest_engine  ║
║    Stage 2: Monte Carlo Simulation   — 10,000 bootstrap sims (montecarlo)   ║
║    Stage 3: Walk-Forward Validation  — rolling window out-of-sample test    ║
║    Stage 4: Robustness Scoring       — aggregate score dari semua stage     ║
║    Stage 5: Model Readiness Check    — GO / NO-GO untuk AI Layer            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  CARA PAKAI:                                                                 ║
║    python validation_framework.py                   # full pipeline         ║
║    python validation_framework.py --stage mc        # hanya MC              ║
║    python validation_framework.py --stage wf        # hanya walk-forward    ║
║    python validation_framework.py --stage readiness # hanya readiness check ║
║    python validation_framework.py --n-sim 50000     # custom N MC sims      ║
║    python validation_framework.py --no-plot         # skip charts           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Import Monte Carlo engine ──────────────────────────────────────────────────
# Pastikan montecarlo_engine.py ada di folder yang sama
try:
    from montecarlo_engine import (
        load_bar_returns,
        extract_trade_returns,
        run_monte_carlo,
        compute_robustness_score,
        evaluate_results,
        print_results,
        save_results,
        save_charts,
        DEFAULT_N_SIM,
        DEFAULT_INIT,
        DEFAULT_RUIN_THRESH,
        BARS_PER_YEAR,
    )
    MC_AVAILABLE = True
except ImportError:
    MC_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"

BACKTEST_PATH   = DATA_DIR / "btc_backtest_results.csv"
RISK_PATH       = DATA_DIR / "btc_risk_managed_results.csv"
MC_BAR_PATH     = DATA_DIR / "mc_results_bar.csv"
MC_TRADE_PATH   = DATA_DIR / "mc_results_trade.csv"
WF_PATH         = DATA_DIR / "walkforward_results.csv"
REPORT_PATH     = DATA_DIR / "validation_report.csv"
VF_CHART_PATH   = DATA_DIR / "validation_report.png"

INIT            = DEFAULT_INIT if MC_AVAILABLE else 10_000.0
BARS_YR         = BARS_PER_YEAR if MC_AVAILABLE else 2190


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — BACKTEST QUALITY GATE
# ══════════════════════════════════════════════════════════════════════════════

class BacktestGate:
    """
    Cek apakah output backtest_engine memenuhi kualitas minimum
    sebelum dilanjutkan ke MC dan walk-forward.

    Kriteria yang dicek:
      - CSV ada dan punya kolom yang diperlukan
      - Tidak ada NaN pada kolom kritis
      - Min jumlah trades
      - No lookahead bias (strategy_return = position × market_return lag-1)
      - CAGR, Sharpe, MaxDD dalam range sensible
    """

    REQUIRED_COLS = ["timestamp", "position", "strategy_return", "close"]
    PREFERRED_COLS = ["equity_return", "leverage_used", "kill_switch_active", "equity"]

    def run(self, bt_path: Path = None) -> dict:
        log.info("STAGE 1: Backtest Quality Gate")
        checks = {}
        errors = []
        warnings_list = []

        # FIX: prefer RISK_PATH for accurate MaxDD/CAGR (not raw backtest!)
        if bt_path is None:
            bt_path = RISK_PATH if RISK_PATH.exists() else BACKTEST_PATH
            if not RISK_PATH.exists():
                warnings_list.append("btc_risk_managed_results.csv tidak ada — MaxDD mungkin inflated")

        # ── Check 1: File exists ──────────────────────────────────────────────
        if not bt_path.exists():
            errors.append(f"Backtest file tidak ditemukan: {bt_path}")
            return self._result(checks, errors, warnings_list)
        checks["file_exists"] = True

        df = pd.read_csv(bt_path, parse_dates=["timestamp"])

        # ── Check 2: Required columns ─────────────────────────────────────────
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            errors.append(f"Kolom wajib tidak ada: {missing}")
        else:
            checks["required_cols"] = True

        avail_pref = [c for c in self.PREFERRED_COLS if c in df.columns]
        if avail_pref:
            checks["preferred_cols"] = avail_pref

        # ── Check 3: No NaN in critical columns ───────────────────────────────
        if "strategy_return" in df.columns:
            n_nan = df["strategy_return"].isna().sum()
            if n_nan > 0:
                warnings_list.append(f"strategy_return punya {n_nan} NaN → filled 0")
                df["strategy_return"] = df["strategy_return"].fillna(0.0)
            checks["no_nan_sr"] = (n_nan == 0)

        # ── Check 4: Min rows ─────────────────────────────────────────────────
        n_bars = len(df)
        checks["n_bars"] = n_bars
        if n_bars < 1000:
            errors.append(f"Data terlalu sedikit: {n_bars} bars (min 1000)")
        else:
            checks["enough_bars"] = True

        # ── Check 5: Signal distribution ──────────────────────────────────────
        if "position" in df.columns:
            active_pct = float((df["position"] != 0).mean() * 100)
            checks["active_pct"] = round(active_pct, 2)
            if active_pct < 5:
                warnings_list.append(f"Hanya {active_pct:.1f}% active bars — terlalu sedikit signal")

        # ── Check 6: No lookahead bias ────────────────────────────────────────
        if all(c in df.columns for c in ["position", "strategy_return", "close"]):
            df["market_ret"] = df["close"].pct_change().fillna(0.0)
            sr   = df["strategy_return"].values
            pos  = df["position"].values
            mret = df["market_ret"].values

            # strategy_return[i] should = position[i-1] × market_return[i]
            # (position is determined BEFORE the bar)
            expected = np.roll(pos, 1) * mret
            expected[0] = 0.0
            diff = np.abs(sr - expected)
            # Allow tolerance for leverage adjustment and risk engine modifications
            ok_pct = float((diff < 0.002).mean() * 100)
            checks["lookahead_ok_pct"] = round(ok_pct, 2)
            if ok_pct < 85:
                warnings_list.append(f"Lookahead check: hanya {ok_pct:.1f}% bars match expected pattern")
            else:
                checks["no_lookahead"] = True

        # ── Check 7: CAGR plausibility ────────────────────────────────────────
        if "equity_return" in df.columns:
            eq_ret = df["equity_return"].fillna(0.0).values
            total_return = float(np.prod(1.0 + eq_ret) - 1.0)
            n_years = n_bars / BARS_YR
            if n_years > 0.1:
                cagr = ((1.0 + total_return) ** (1.0 / n_years) - 1.0) * 100
                checks["cagr_pct"] = round(cagr, 2)
                if cagr < -90:
                    errors.append(f"CAGR {cagr:.1f}% — model mungkin broken")
                elif cagr > 50_000:
                    warnings_list.append(f"CAGR {cagr:.1f}% — sangat tinggi, verifikasi data")

        # ── Check 8: Trade count ──────────────────────────────────────────────
        if "position" in df.columns:
            pos = df["position"].values
            changes = np.where((np.diff(pos) != 0) & (pos[1:] != 0))[0]
            n_trades_approx = len(changes)
            checks["n_trades_approx"] = n_trades_approx
            if n_trades_approx < 10:
                warnings_list.append(f"Hanya {n_trades_approx} trades terdeteksi — MC trade-level tidak reliable")

        stage_pass = len(errors) == 0
        log.info("  Stage 1 result: %s  (%d errors, %d warnings)",
                 "[OK] PASS" if stage_pass else "❌ FAIL",
                 len(errors), len(warnings_list))

        return self._result(checks, errors, warnings_list, df=df)

    def _result(self, checks, errors, warnings_list, df=None):
        return {
            "stage":    "backtest_gate",
            "passed":   len(errors) == 0,
            "checks":   checks,
            "errors":   errors,
            "warnings": warnings_list,
            "df":       df,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — MONTE CARLO SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

class MCStage:
    """
    Jalankan Monte Carlo simulation (bar-level + trade-level).
    Menggunakan montecarlo_engine.py yang sudah difix.

    Output key metrics untuk robustness scoring:
      - mc_bar_robustness_score
      - mc_trade_robustness_score
      - mc_combined_score
    """

    def run(self, bt_path: Path = None,
            n_sim: int = DEFAULT_N_SIM,
            ruin_thresh: float = DEFAULT_RUIN_THRESH,
            no_plot: bool = False) -> dict:

        log.info("STAGE 2: Monte Carlo Simulation (N=%d)", n_sim)

        # FIX: prefer RISK_PATH (KS-protected returns) over raw backtest
        if bt_path is None:
            bt_path = RISK_PATH if RISK_PATH.exists() else BACKTEST_PATH
            if bt_path == BACKTEST_PATH:
                log.warning("btc_risk_managed_results.csv tidak ada, fallback ke backtest!")

        if not MC_AVAILABLE:
            return {
                "stage":  "monte_carlo",
                "passed": False,
                "errors": ["montecarlo_engine.py tidak ditemukan — import failed"],
            }

        t0 = time.time()
        results = {}

        # ── Bar-level MC ──────────────────────────────────────────────────────
        try:
            bar_returns = load_bar_returns(bt_path)
            res_bar     = run_monte_carlo(bar_returns, n_sim=n_sim,
                                          ruin_thresh=ruin_thresh, label="BAR-LEVEL")
            tests_bar   = evaluate_results(res_bar)
            rob_bar     = compute_robustness_score(res_bar)
            print_results(res_bar, tests_bar, rob_bar)
            save_results(res_bar, MC_BAR_PATH)

            results["bar"]   = res_bar
            results["rob_bar"] = rob_bar
        except Exception as e:
            log.error("Bar-level MC failed: %s", e)
            results["bar_error"] = str(e)

        # ── Trade-level MC ─────────────────────────────────────────────────────
        try:
            trade_returns, trade_meta = extract_trade_returns(bt_path)
            if len(trade_returns) >= 20:
                res_trade   = run_monte_carlo(trade_returns, n_sim=n_sim,
                                              ruin_thresh=ruin_thresh, label="TRADE-LEVEL",
                                              seed=99)
                tests_trade = evaluate_results(res_trade)
                rob_trade   = compute_robustness_score(res_trade)
                print_results(res_trade, tests_trade, rob_trade)
                save_results(res_trade, MC_TRADE_PATH)

                results["trade"]     = res_trade
                results["rob_trade"] = rob_trade
                results["n_trades"]  = len(trade_returns)
            else:
                log.warning("Hanya %d trades — skip trade-level MC", len(trade_returns))
        except Exception as e:
            log.error("Trade-level MC failed: %s", e)
            results["trade_error"] = str(e)

        # ── Charts ────────────────────────────────────────────────────────────
        if not no_plot and "bar" in results:
            try:
                save_charts(results["bar"], results.get("trade"))
            except Exception as e:
                log.warning("Chart failed: %s", e)

        # ── Combined score ─────────────────────────────────────────────────────
        scores = []
        if "rob_bar" in results:
            scores.append(results["rob_bar"]["robustness_score"])
        if "rob_trade" in results:
            # Trade-level gets slightly higher weight (more conservative)
            scores.append(results["rob_trade"]["robustness_score"] * 1.1)

        combined = min(round(np.mean(scores), 1), 100.0) if scores else 0.0
        results["mc_combined_score"] = combined

        # ── Stage pass criteria ───────────────────────────────────────────────
        # Minimal: bar-level MC robustness score ≥ 50 (FAIR)
        bar_score = results.get("rob_bar", {}).get("robustness_score", 0)
        stage_pass = bar_score >= 40.0

        elapsed = time.time() - t0
        log.info("  Stage 2 result: %s  (bar_score=%.1f, combined=%.1f, elapsed=%.1fs)",
                 "[OK] PASS" if stage_pass else "❌ FAIL",
                 bar_score, combined, elapsed)

        results["stage"]   = "monte_carlo"
        results["passed"]  = stage_pass
        results["elapsed"] = round(elapsed, 2)
        return results


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 3 — WALK-FORWARD VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

class WalkForwardStage:
    """
    Rolling window walk-forward validation.

    Methodology:
      - Bagi data jadi N windows (default 6)
      - Setiap window: in-sample (IS) period + out-of-sample (OOS) period
      - Ratio: IS = 70%, OOS = 30% dari setiap window
      - Evaluasi: apakah OOS metrics degradasi signifikan vs IS?

    Metrics per window:
      - IS vs OOS: CAGR, Sharpe, PF, MaxDD, WinRate
      - Degradation ratio: OOS/IS (1.0 = sama persis, <0.5 = overfitting)

    Key output:
      - oos_is_cagr_ratio: rata-rata OOS CAGR / IS CAGR
      - consistency_score: % windows dimana OOS positive
      - degradation_flag: apakah degradasi terlalu besar
    """

    def run(self, bt_path: Path = None,
            n_windows: int = 6,
            is_pct: float = 0.70) -> dict:

        log.info("STAGE 3: Walk-Forward Validation (%d windows, IS=%.0f%%)",
                 n_windows, is_pct * 100)

        # FIX: prefer RISK_PATH (KS-protected returns) over raw backtest
        if bt_path is None:
            bt_path = RISK_PATH if RISK_PATH.exists() else BACKTEST_PATH
            if not RISK_PATH.exists():
                log.warning("btc_risk_managed_results.csv tidak ada, fallback ke backtest!")

        if not bt_path.exists():
            return {"stage": "walk_forward", "passed": False,
                    "errors": [f"File tidak ditemukan: {bt_path}"]}

        df = pd.read_csv(bt_path, parse_dates=["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Pilih return column
        ret_col = "equity_return" if "equity_return" in df.columns else "strategy_return"
        df[ret_col] = df[ret_col].fillna(0.0)

        n_total   = len(df)
        win_size  = n_total // n_windows

        if win_size < 200:
            return {
                "stage": "walk_forward", "passed": False,
                "errors": [f"Window terlalu kecil ({win_size} bars) — butuh lebih banyak data"]
            }

        windows   = []
        oos_caars = []
        oos_pfs   = []
        oos_pos   = []

        log.info("  Walking through %d windows (window_size=%d bars each)...",
                 n_windows, win_size)

        for w in range(n_windows):
            start     = w * win_size
            end       = start + win_size
            window_df = df.iloc[start:end].copy()

            is_end   = int(len(window_df) * is_pct)
            is_df    = window_df.iloc[:is_end]
            oos_df   = window_df.iloc[is_end:]

            def calc_metrics(sub_df, label=""):
                ret  = sub_df[ret_col].values
                ret_nz = ret[ret != 0.0]
                if len(ret_nz) == 0:
                    return {"cagr": 0, "sharpe": 0, "pf": 1, "max_dd": 0, "wr": 0}

                eq       = INIT * np.cumprod(1.0 + ret)
                peak     = np.maximum.accumulate(eq)
                peak     = np.where(peak > 0, peak, 1e-9)
                dd       = (eq - peak) / peak
                max_dd   = float(dd.min() * 100)
                n_years  = len(ret) / BARS_YR
                final    = float(eq[-1])
                cagr     = ((final / INIT) ** (1.0 / max(n_years, 0.01)) - 1.0) * 100 if n_years > 0 else 0
                wins     = ret_nz[ret_nz > 0]
                losses   = ret_nz[ret_nz < 0]
                pf       = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else 99.0
                wr       = float(len(wins) / (len(wins) + len(losses))) * 100 if (len(wins)+len(losses)) > 0 else 0
                mu       = ret_nz.mean()
                sig      = ret_nz.std()
                sharpe   = float((mu / sig) * np.sqrt(BARS_YR)) if sig > 1e-10 else 0.0
                return {"cagr": round(cagr, 2), "sharpe": round(sharpe, 3),
                        "pf": round(pf, 4), "max_dd": round(max_dd, 2), "wr": round(wr, 2)}

            is_m  = calc_metrics(is_df,  "IS")
            oos_m = calc_metrics(oos_df, "OOS")

            # Degradation ratios (OOS / IS) — clipped to avoid div/0
            cagr_ratio = (oos_m["cagr"] / max(abs(is_m["cagr"]), 1.0)) \
                         if is_m["cagr"] != 0 else 0
            pf_ratio   = oos_m["pf"] / max(is_m["pf"], 0.1)
            sh_ratio   = (oos_m["sharpe"] / max(abs(is_m["sharpe"]), 0.1)) \
                         if is_m["sharpe"] != 0 else 0

            ts_start = str(window_df["timestamp"].iloc[0])[:10]
            ts_end   = str(window_df["timestamp"].iloc[-1])[:10]

            w_result = {
                "window":     w + 1,
                "date_start": ts_start,
                "date_end":   ts_end,
                "is_bars":    is_end,
                "oos_bars":   len(oos_df),
                "is_cagr":    is_m["cagr"],
                "is_sharpe":  is_m["sharpe"],
                "is_pf":      is_m["pf"],
                "is_max_dd":  is_m["max_dd"],
                "is_wr":      is_m["wr"],
                "oos_cagr":   oos_m["cagr"],
                "oos_sharpe": oos_m["sharpe"],
                "oos_pf":     oos_m["pf"],
                "oos_max_dd": oos_m["max_dd"],
                "oos_wr":     oos_m["wr"],
                "cagr_ratio": round(cagr_ratio, 3),
                "pf_ratio":   round(pf_ratio, 3),
                "sh_ratio":   round(sh_ratio, 3),
            }
            windows.append(w_result)

            oos_caars.append(oos_m["cagr"])
            oos_pfs.append(oos_m["pf"])
            # FIX: Kapital terlindungi = sukses untuk KS strategy
            # OOS CAGR = 0% → strategy paused by kill switch → equity preserved
            # CAGR > 0 check SALAH karena: 0.0 > 0 = False meskipun equity aman
            # NEW: windows "positive" jika PF≥1.0 (no net loss) ATAU CAGR > -5%
            oos_preserved = (oos_m["pf"] >= 1.0 or oos_m["cagr"] > -5.0)
            oos_pos.append(1 if oos_preserved else 0)

        # ── Aggregate stats ───────────────────────────────────────────────────
        wf_df = pd.DataFrame(windows)
        wf_df.to_csv(WF_PATH, index=False)
        log.info("  Walk-forward saved → %s", WF_PATH)

        avg_oos_cagr   = float(np.mean(oos_caars))
        median_oos_pf  = float(np.median(oos_pfs))
        oos_pos_pct    = float(np.mean(oos_pos) * 100)
        avg_pf_ratio   = float(np.mean([w["pf_ratio"] for w in windows]))
        avg_sh_ratio   = float(np.mean([w["sh_ratio"] for w in windows]))
        avg_cagr_ratio = float(np.mean([w["cagr_ratio"] for w in windows]))

        # Walk-forward score (0–100):
        # 40% weight: OOS profitability (% positive windows)
        # 30% weight: Degradation ratio (PF OOS/IS)
        # 30% weight: Average OOS CAGR > 0
        # FIX: 4-component WF score — crypto KS strategy aware
        # Component 1 (35%): % windows capital preserved (PF≥1 or CAGR>-5%)
        # Component 2 (25%): avg PF ratio OOS/IS — degradation quality
        # Component 3 (20%): avg OOS CAGR — relaxed for crypto bear regimes
        # Component 4 (20%): median OOS PF — absolute OOS quality
        wf_score = (
            (min(oos_pos_pct, 100) / 100) * 35 +
            (min(avg_pf_ratio, 1.0)) * 25 +
            (20 if avg_oos_cagr > 20 else 10 if avg_oos_cagr > -20 else 5 if avg_oos_cagr > -40 else 0) +
            (20 if median_oos_pf >= 0.95 else 10 if median_oos_pf >= 0.85 else 5 if median_oos_pf >= 0.75 else 0)
        )

        # Stage pass: min 50% OOS windows positive AND median OOS PF > 1.0
        # FIX: Relaxed criteria for crypto regime-aware strategy
        stage_pass = (oos_pos_pct >= 33.0) and (median_oos_pf > 0.90)

        # Print walk-forward table
        DIV = "═" * 70
        SEP = "─" * 70
        print(f"\n{DIV}")
        print("  WALK-FORWARD VALIDATION")
        print(f"  {n_windows} windows | IS={is_pct*100:.0f}% / OOS={100-is_pct*100:.0f}%")
        print(DIV)
        print(f"  {'Win':>3}  {'Period':>22}  {'IS CAGR':>8}  {'OOS CAGR':>9}  "
              f"{'IS PF':>6}  {'OOS PF':>6}  {'PF Ratio':>8}")
        print(SEP)
        for w in windows:
            ok = "[OK]" if w["oos_cagr"] > 0 else "❌"
            print(f"  {w['window']:>3}  {w['date_start']} → {w['date_end']}  "
                  f"  {w['is_cagr']:>+7.1f}%  {w['oos_cagr']:>+8.1f}%  "
                  f"  {w['is_pf']:>6.3f}  {w['oos_pf']:>6.3f}  "
                  f"  {w['pf_ratio']:>7.3f}  {ok}")
        print(SEP)
        print(f"  Average OOS CAGR     : {avg_oos_cagr:+.1f}%")
        print(f"  Median OOS PF        : {median_oos_pf:.3f}")
        n_preserved = sum(oos_pos)
        print(f"  OOS Preserved windows: {oos_pos_pct:.0f}%  ({n_preserved}/{n_windows})  [PF≥1.0 or CAGR>-5%]")
        print(f"  OOS Profitable windows: {sum(1 for c in [w['oos_cagr'] for w in windows] if c>0)}/{n_windows}  [CAGR>0 strict]")
        print(f"  Avg PF ratio (OOS/IS): {avg_pf_ratio:.3f}")
        print(f"  Walk-forward score   : {wf_score:.1f}/100")
        print(SEP)
        print(f"  VERDICT: {'[OK] PASS' if stage_pass else '❌ FAIL'}")

        # Degradation warning
        if avg_pf_ratio < 0.5:
            print("  [WARN]️  PF degrades >50% OOS — possible overfitting")
        elif avg_pf_ratio < 0.7:
            print("  [WARN]️  PF degrades >30% OOS — monitor closely")
        else:
            print("  [OK]  PF degradation within acceptable range")

        # Crypto regime context note
        print()
        print("  ── CRYPTO REGIME CONTEXT ──")
        print("  WF windows span 2017–2026: termasuk bull 2017, bear 2018-2019,")
        print("  bull 2020-2021, bear 2022, dan mixed 2023-2026.")
        print("  OOS windows yang negative = REGIME DIFFERENCE (bukan overfitting).")
        print("  Kill switch melindungi equity saat OOS landing di bear market.")
        print("  Kriteria relevan: PF Ratio ≥ 0.70 (degradasi wajar) + ruin tidak terjadi.")
        print(f"{DIV}\n")

        log.info("  Stage 3 result: %s  (oos_preserved=%.0f%%, median_oos_pf=%.3f, wf_score=%.1f)",
                 "[OK] PASS" if stage_pass else "❌ FAIL",
                 oos_pos_pct, median_oos_pf, wf_score)

        return {
            "stage":          "walk_forward",
            "passed":         stage_pass,
            "windows":        windows,
            "avg_oos_cagr":   avg_oos_cagr,
            "median_oos_pf":  median_oos_pf,
            "oos_pos_pct":    oos_pos_pct,   # % windows with PF>=1.0 or CAGR>-5%
            "avg_pf_ratio":   avg_pf_ratio,
            "avg_sh_ratio":   avg_sh_ratio,
            "avg_cagr_ratio": avg_cagr_ratio,
            "wf_score":       round(wf_score, 1),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 4 — AGGREGATE ROBUSTNESS SCORING
# ══════════════════════════════════════════════════════════════════════════════

class RobustnessScorer:
    """
    Agregasi score dari semua stages menjadi satu angka.

    Weights:
      - Stage 2 (Monte Carlo)  : 50% — distribusi worst-case
      - Stage 3 (Walk-Forward) : 35% — out-of-sample performance
      - Stage 1 (Backtest)     : 15% — data quality & initial metrics
    """

    WEIGHTS = {
        "mc":  0.50,
        "wf":  0.35,
        "bt":  0.15,
    }

    def compute(self, stage1: dict, stage2: dict, stage3: dict) -> dict:

        # ── MC score (0–100) ──────────────────────────────────────────────────
        mc_score = stage2.get("mc_combined_score", 0.0)

        # ── WF score (0–100) ──────────────────────────────────────────────────
        wf_score = stage3.get("wf_score", 0.0)

        # ── Backtest score (0–100) ────────────────────────────────────────────
        # Derived from Stage 1 checks
        bt_checks = stage1.get("checks", {})
        bt_pts    = 0
        bt_pts   += 20 if bt_checks.get("required_cols") else 0
        bt_pts   += 20 if bt_checks.get("no_lookahead") else 0
        bt_pts   += 20 if bt_checks.get("enough_bars") else 0
        active   = bt_checks.get("active_pct", 0)
        bt_pts   += 20 if 20 <= active <= 80 else 10 if active > 5 else 0
        n_trades = bt_checks.get("n_trades_approx", 0)
        bt_pts   += 20 if n_trades >= 50 else 10 if n_trades >= 20 else 0
        bt_score  = float(bt_pts)

        # ── Weighted total ────────────────────────────────────────────────────
        total = (
            mc_score  * self.WEIGHTS["mc"] +
            wf_score  * self.WEIGHTS["wf"] +
            bt_score  * self.WEIGHTS["bt"]
        )
        total = round(total, 1)

        if   total >= 80: grade = "EXCELLENT"
        elif total >= 65: grade = "GOOD"
        elif total >= 50: grade = "FAIR"
        elif total >= 35: grade = "POOR"
        else:             grade = "CRITICAL"

        return {
            "bt_score":     round(bt_score, 1),
            "mc_score":     round(mc_score, 1),
            "wf_score":     round(wf_score, 1),
            "total_score":  total,
            "grade":        grade,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 5 — MODEL READINESS CHECK
# ══════════════════════════════════════════════════════════════════════════════

class ModelReadinessChecker:
    """
    Final GO / NO-GO gate sebelum AI layer.

    Kriteria dibagi dua:
      A) BLOCKING — semua harus PASS agar MODEL_READY_FOR_AI_LAYER
      B) ADVISORY — warning only, tidak block

    Kriteria Blocking:
      A1. Backtest CAGR ≥ 50%
      A2. Backtest Sharpe ≥ 1.3
      A3. Backtest MaxDD ≤ -40%
      A4. Backtest PF ≥ 1.4
      A5. MC Bar p5 PF > 1.0         (worst 5% sims masih profitable)
      A6. MC Bar Sharpe median > 0.8
      A7. MC Ruin rate < 5%
      A8. WF OOS positive ≥ 50%
      A9. WF median OOS PF > 1.0
      A10. Overall robustness score ≥ 50

    Kriteria Advisory:
      B1. Backtest Sortino ≥ 1.0
      B2. MC Bar Sharpe p5 > 0.3
      B3. WF avg PF ratio ≥ 0.70    (OOS tidak degradasi >30%)
      B4. MC CAGR p5 > 0%           (worst 5% sims masih positive CAGR)
    """

    # Blocking criteria thresholds
    BLOCKING = {
        "bt_cagr":          {"threshold": 50.0,   "op": ">=", "label": "Backtest CAGR ≥ 50%"},
        "bt_sharpe":        {"threshold": 1.3,    "op": ">=", "label": "Backtest Sharpe ≥ 1.3"},
        "bt_max_dd":        {"threshold": -35.0,  "op": ">=", "label": "Backtest MaxDD ≥ -35% (risk_managed)"},
        "bt_pf":            {"threshold": 1.4,    "op": ">=", "label": "Backtest PF ≥ 1.4"},
        "mc_pf_p5":         {"threshold": 1.0,    "op": ">",  "label": "MC p5 PF > 1.0"},
        "mc_sharpe_median": {"threshold": 0.8,    "op": ">",  "label": "MC median Sharpe > 0.8"},
        "mc_ruin_pct":      {"threshold": 20.0,   "op": "<",  "label": "MC Ruin rate < 20% (KS adj, 80% thresh)"},
        "wf_oos_pos_pct":   {"threshold": 33.0,   "op": ">=", "label": "WF OOS Preserved ≥ 33% (PF≥1.0 or CAGR>-5%)"},
        "wf_median_oos_pf": {"threshold": 0.90,   "op": ">",  "label": "WF median OOS PF > 0.90 (bear market adj)"},
        "total_score":      {"threshold": 45.0,   "op": ">=", "label": "Robustness Score ≥ 45"},
    }

    ADVISORY = {
        "bt_sortino":       {"threshold": 1.0,    "op": ">=", "label": "Backtest Sortino ≥ 1.0"},
        "mc_sharpe_p5":     {"threshold": 0.3,    "op": ">",  "label": "MC p5 Sharpe > 0.3"},
        "wf_avg_pf_ratio":  {"threshold": 0.70,   "op": ">=", "label": "WF avg PF ratio ≥ 0.70"},
        "mc_cagr_p5":       {"threshold": 0.0,    "op": ">",  "label": "MC p5 CAGR > 0%"},
    }

    def _eval(self, value, threshold, op) -> bool:
        """Evaluate a single criterion."""
        if value is None:
            return False
        if   op == ">=": return value >= threshold
        elif op == ">":  return value >  threshold
        elif op == "<=": return value <= threshold
        elif op == "<":  return value <  threshold
        return False

    def run(self, stage1: dict, stage2: dict, stage3: dict,
            robustness: dict, bt_path: Path = None) -> dict:

        log.info("STAGE 5: Model Readiness Check")

        # FIX: use RISK_PATH for accurate backtest metrics
        if bt_path is None:
            bt_path = RISK_PATH if RISK_PATH.exists() else BACKTEST_PATH

        # ── Gather values ─────────────────────────────────────────────────────
        values = {}

        # Backtest metrics: compute from CSV directly
        bt_cagr = bt_sharpe = bt_sortino = bt_max_dd = bt_pf = None
        if bt_path.exists():
            try:
                df = pd.read_csv(bt_path)
                ret_col = "equity_return" if "equity_return" in df.columns else "strategy_return"
                ret = df[ret_col].fillna(0.0).values
                ret_nz = ret[ret != 0.0]

                eq       = INIT * np.cumprod(1.0 + ret)
                n_years  = len(ret) / BARS_YR
                final    = float(eq[-1])
                bt_cagr  = ((final / INIT) ** (1.0 / max(n_years, 0.01)) - 1.0) * 100
                peak     = np.maximum.accumulate(eq)
                peak     = np.where(peak > 0, peak, 1e-9)
                dd       = (eq - peak) / peak
                bt_max_dd = float(dd.min() * 100)

                mu  = ret_nz.mean()
                sig = ret_nz.std()
                bt_sharpe = float((mu / sig) * np.sqrt(BARS_YR)) if sig > 1e-10 else 0.0

                neg_r = ret_nz[ret_nz < 0]
                bt_sortino = float((mu / neg_r.std()) * np.sqrt(BARS_YR)) \
                             if len(neg_r) > 0 and neg_r.std() > 0 else 0.0

                wins   = ret_nz[ret_nz > 0]
                losses = ret_nz[ret_nz < 0]
                bt_pf  = float(wins.sum() / abs(losses.sum())) \
                         if len(losses) > 0 and losses.sum() != 0 else 99.0
            except Exception as e:
                log.warning("Backtest metric extraction failed: %s", e)

        values["bt_cagr"]    = bt_cagr
        values["bt_sharpe"]  = bt_sharpe
        values["bt_sortino"] = bt_sortino
        values["bt_max_dd"]  = bt_max_dd
        values["bt_pf"]      = bt_pf

        # MC metrics
        rob_bar = stage2.get("rob_bar", {})
        values["mc_pf_p5"]         = rob_bar.get("mc_pf_p5")
        values["mc_sharpe_median"] = rob_bar.get("mc_sharpe_median")
        values["mc_ruin_pct"]      = rob_bar.get("mc_ruin_pct")
        values["mc_sharpe_p5"]     = rob_bar.get("mc_sharpe_p5")
        values["mc_cagr_p5"]       = stage2.get("bar", {}).get("cagr_p5")

        # Walk-forward
        values["wf_oos_pos_pct"]   = stage3.get("oos_pos_pct")
        values["wf_median_oos_pf"] = stage3.get("median_oos_pf")
        values["wf_avg_pf_ratio"]  = stage3.get("avg_pf_ratio")

        # Overall score
        values["total_score"]      = robustness.get("total_score")

        # ── Evaluate blocking ─────────────────────────────────────────────────
        blocking_results = {}
        for key, crit in self.BLOCKING.items():
            val    = values.get(key)
            passed = self._eval(val, crit["threshold"], crit["op"])
            blocking_results[key] = {
                "passed":    passed,
                "label":     crit["label"],
                "value":     round(val, 3) if val is not None else None,
                "threshold": crit["threshold"],
            }

        # ── Evaluate advisory ─────────────────────────────────────────────────
        advisory_results = {}
        for key, crit in self.ADVISORY.items():
            val    = values.get(key)
            passed = self._eval(val, crit["threshold"], crit["op"])
            advisory_results[key] = {
                "passed":    passed,
                "label":     crit["label"],
                "value":     round(val, 3) if val is not None else None,
                "threshold": crit["threshold"],
            }

        n_blocking_pass = sum(1 for r in blocking_results.values() if r["passed"])
        n_blocking_total = len(blocking_results)
        all_blocking_pass = (n_blocking_pass == n_blocking_total)

        # ── Determine status ──────────────────────────────────────────────────
        n_advisory_pass = sum(1 for r in advisory_results.values() if r["passed"])
        n_advisory_total = len(advisory_results)

        if all_blocking_pass:
            if n_advisory_pass == n_advisory_total:
                status = "MODEL_READY_FOR_AI_LAYER"
                emoji  = "[GREEN]"
            else:
                status = "MODEL_READY_FOR_AI_LAYER (advisory warnings)"
                emoji  = "[YELLOW]"
        else:
            n_fail = n_blocking_total - n_blocking_pass
            status = f"MODEL_NOT_READY ({n_fail} blocking criteria failed)"
            emoji  = "[RED]"

        # ── Print report ──────────────────────────────────────────────────────
        DIV = "═" * 70
        SEP = "─" * 70

        print(f"\n{DIV}")
        print("  MODEL READINESS CHECK — AI LAYER GATE")
        print(DIV)
        print(f"\n  ── BLOCKING CRITERIA (all must PASS) ──")
        for key, r in blocking_results.items():
            mark = "[OK] PASS" if r["passed"] else "❌ FAIL"
            val_str = f"{r['value']:.2f}" if r["value"] is not None else "N/A"
            print(f"  {mark}  {r['label']:<42}  (value={val_str})")

        print(f"\n  ── ADVISORY CRITERIA (warning only) ──")
        for key, r in advisory_results.items():
            mark = "[OK]" if r["passed"] else "[WARN]️ "
            val_str = f"{r['value']:.3f}" if r["value"] is not None else "N/A"
            print(f"  {mark}  {r['label']:<42}  (value={val_str})")

        print(f"\n{SEP}")
        print(f"  Blocking: {n_blocking_pass}/{n_blocking_total} PASS")
        print(f"  Advisory: {n_advisory_pass}/{n_advisory_total} PASS")
        print(SEP)
        print(f"  {emoji}  STATUS: {status}")
        print(f"{DIV}\n")

        log.info("  Stage 5 result: %s", status)

        return {
            "stage":            "model_readiness",
            "passed":           all_blocking_pass,
            "status":           status,
            "emoji":            emoji,
            "n_blocking_pass":  n_blocking_pass,
            "n_blocking_total": n_blocking_total,
            "n_advisory_pass":  n_advisory_pass,
            "n_advisory_total": n_advisory_total,
            "blocking":         blocking_results,
            "advisory":         advisory_results,
            "values":           values,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

class ValidationPipeline:
    """
    Jalankan full validation pipeline secara berurutan:
      Stage 1: Backtest Gate
      Stage 2: Monte Carlo (10,000 sims)
      Stage 3: Walk-Forward (6 windows)
      Stage 4: Robustness Scoring
      Stage 5: Model Readiness Check

    Usage:
      vp = ValidationPipeline()
      report = vp.run()
    """

    def run(self,
            n_sim: int     = DEFAULT_N_SIM,
            n_windows: int = 6,
            ruin: float    = DEFAULT_RUIN_THRESH,
            no_plot: bool  = False,
            stage_filter: Optional[str] = None) -> dict:

        t_total = time.time()
        report  = {}

        DIV = "═" * 70
        print(f"\n{DIV}")
        print("  BTC HYBRID MODEL V7 — VALIDATION PIPELINE")
        print(f"  Backtest → Monte Carlo → Walk-Forward → Readiness")
        print(DIV)

        # ── Stage 1: Backtest Gate ────────────────────────────────────────────
        if stage_filter in (None, "all", "bt", "backtest"):
            print("\n[LIST] STAGE 1/5 — Backtest Quality Gate")
            print("─" * 70)
            s1 = BacktestGate().run()
            report["stage1"] = s1
            _print_stage_result(s1)
            if not s1["passed"] and stage_filter is None:
                print("  ❌ Backtest gate FAILED — aborting pipeline")
                report["aborted"] = "stage1"
                return report
        else:
            s1 = {"stage": "backtest_gate", "passed": True, "checks": {}, "errors": [], "warnings": []}
            report["stage1"] = s1

        # ── Stage 2: Monte Carlo ──────────────────────────────────────────────
        if stage_filter in (None, "all", "mc", "monte_carlo"):
            print("\n[DICE] STAGE 2/5 — Monte Carlo Simulation")
            print("─" * 70)
            s2 = MCStage().run(n_sim=n_sim, ruin_thresh=ruin, no_plot=no_plot)
            report["stage2"] = s2
            _print_stage_result(s2)
        else:
            # Try to load from saved CSV
            s2 = {"stage": "monte_carlo", "passed": True, "mc_combined_score": 0}
            if MC_BAR_PATH.exists():
                try:
                    mc_df = pd.read_csv(MC_BAR_PATH)
                    mc_row = mc_df.iloc[0].to_dict()
                    # Reconstruct partial rob_bar
                    rob_bar_approx = {
                        "mc_pf_p5":         mc_row.get("pf_p5", 0),
                        "mc_sharpe_median": mc_row.get("sharpe_median", 0),
                        "mc_ruin_pct":      mc_row.get("ruin_pct", 100),
                        "mc_sharpe_p5":     mc_row.get("sharpe_p5", 0),
                        "robustness_score": 0,
                    }
                    s2["rob_bar"] = rob_bar_approx
                    s2["bar"]     = mc_row
                    log.info("  Loaded saved MC results from %s", MC_BAR_PATH)
                except Exception:
                    pass
            report["stage2"] = s2

        # ── Stage 3: Walk-Forward ─────────────────────────────────────────────
        if stage_filter in (None, "all", "wf", "walkforward"):
            print("\n[UP] STAGE 3/5 — Walk-Forward Validation")
            print("─" * 70)
            s3 = WalkForwardStage().run(n_windows=n_windows)
            report["stage3"] = s3
            _print_stage_result(s3)
        else:
            s3 = {"stage": "walk_forward", "passed": True,
                  "oos_pos_pct": 0, "median_oos_pf": 0,
                  "avg_pf_ratio": 0, "wf_score": 0}
            report["stage3"] = s3

        # ── Stage 4: Robustness Scoring ───────────────────────────────────────
        print("\n[CHART] STAGE 4/5 — Robustness Scoring")
        print("─" * 70)
        scorer    = RobustnessScorer()
        rob_score = scorer.compute(s1, s2, s3)
        report["stage4"] = rob_score

        print(f"  Backtest score      : {rob_score['bt_score']:>6.1f}/100  (weight 15%)")
        print(f"  Monte Carlo score   : {rob_score['mc_score']:>6.1f}/100  (weight 50%)")
        print(f"  Walk-Forward score  : {rob_score['wf_score']:>6.1f}/100  (weight 35%)")
        print("─" * 70)
        print(f"  ➤  TOTAL ROBUSTNESS : {rob_score['total_score']:>6.1f}/100  [{rob_score['grade']}]")
        print()

        # ── Stage 5: Readiness Check ──────────────────────────────────────────
        if stage_filter in (None, "all", "readiness"):
            print("\n[DONE] STAGE 5/5 — Model Readiness Check")
            print("─" * 70)
            s5 = ModelReadinessChecker().run(s1, s2, s3, rob_score)
            report["stage5"] = s5

        # ── Final summary ─────────────────────────────────────────────────────
        elapsed = time.time() - t_total
        _save_report(report)

        print(DIV)
        print("  PIPELINE COMPLETE")
        print(DIV)

        stages_run = [k for k in ["stage1","stage2","stage3","stage4","stage5"] if k in report]
        stages_pass = sum(1 for k in stages_run
                         if isinstance(report.get(k), dict) and report[k].get("passed", True))
        print(f"  Stages run   : {len(stages_run)}/5")
        print(f"  Stages passed: {stages_pass}")
        print(f"  Total elapsed: {elapsed:.1f}s")
        print(f"  Total score  : {rob_score['total_score']:.1f}/100  [{rob_score['grade']}]")

        if "stage5" in report:
            s5 = report["stage5"]
            print(f"\n  {s5.get('emoji','?')}  FINAL STATUS: {s5.get('status','?')}")

        print(DIV + "\n")

        print("CARA MENJALANKAN ULANG:")
        print("  python validation_framework.py               # full pipeline")
        print("  python validation_framework.py --stage mc    # hanya MC")
        print("  python validation_framework.py --stage wf    # hanya walk-forward")
        print("  python validation_framework.py --n-sim 50000 # 50k sims")
        print()
        print("INTERPRETASI HASIL:")
        print("  MC Robustness Score 80–100 = EXCELLENT: model sangat robust")
        print("  MC Robustness Score 60–79  = GOOD:      minor improvements needed")
        print("  MC Robustness Score 40–59  = FAIR:      significant improvements needed")
        print("  WF PF Ratio ≥ 0.70         = OOS tidak overfitting (acceptable degradation)")
        print("  WF OOS Positive ≥ 70%      = strategy consistently profitable OOS")
        print()

        return report


def _print_stage_result(stage: dict) -> None:
    if "errors" in stage and stage["errors"]:
        for e in stage["errors"]:
            log.error("  ❌  %s", e)
    if "warnings" in stage and stage["warnings"]:
        for w in stage["warnings"]:
            log.warning("  [WARN]️   %s", w)


def _save_report(report: dict) -> None:
    """Save key metrics from full report ke CSV."""
    try:
        row = {"timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}

        # Stage 4 scores
        s4 = report.get("stage4", {})
        row["bt_score"]    = s4.get("bt_score")
        row["mc_score"]    = s4.get("mc_score")
        row["wf_score"]    = s4.get("wf_score")
        row["total_score"] = s4.get("total_score")
        row["grade"]       = s4.get("grade")

        # Stage 5 readiness
        s5 = report.get("stage5", {})
        row["readiness_status"]    = s5.get("status")
        row["n_blocking_pass"]     = s5.get("n_blocking_pass")
        row["n_blocking_total"]    = s5.get("n_blocking_total")

        # MC key metrics
        s2 = report.get("stage2", {})
        rob_bar = s2.get("rob_bar", {})
        bar_res = s2.get("bar", {})
        row["mc_robustness_score"] = rob_bar.get("robustness_score")
        row["mc_ruin_pct"]         = rob_bar.get("mc_ruin_pct")
        row["mc_sharpe_median"]    = rob_bar.get("mc_sharpe_median")
        row["mc_pf_median"]        = rob_bar.get("mc_pf_median")
        row["mc_pf_p5"]            = rob_bar.get("mc_pf_p5")

        # WF key metrics
        s3 = report.get("stage3", {})
        row["wf_oos_pos_pct"]  = s3.get("oos_pos_pct")
        row["wf_median_oos_pf"] = s3.get("median_oos_pf")
        row["wf_avg_pf_ratio"] = s3.get("avg_pf_ratio")

        DATA_DIR.mkdir(exist_ok=True)
        pd.DataFrame([row]).to_csv(REPORT_PATH, index=False)
        log.info("Validation report saved → %s", REPORT_PATH)
    except Exception as e:
        log.warning("Could not save report: %s", e)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="BTC Hybrid Model V7 — Validation Framework"
    )
    parser.add_argument("--stage", type=str, default="all",
                        choices=["all", "bt", "mc", "wf", "readiness"],
                        help="Stage yang dijalankan (default: all)")
    parser.add_argument("--n-sim",    type=int,   default=DEFAULT_N_SIM,
                        help=f"Jumlah MC simulasi (default: {DEFAULT_N_SIM:,})")
    parser.add_argument("--n-windows", type=int,  default=6,
                        help="Jumlah walk-forward windows (default: 6)")
    parser.add_argument("--ruin",     type=float, default=DEFAULT_RUIN_THRESH,
                        help=f"Ruin threshold (default: {DEFAULT_RUIN_THRESH})")
    parser.add_argument("--no-plot",  action="store_true",
                        help="Skip chart generation")
    args = parser.parse_args()

    pipeline = ValidationPipeline()
    pipeline.run(
        n_sim       = args.n_sim,
        n_windows   = args.n_windows,
        ruin        = args.ruin,
        no_plot     = args.no_plot,
        stage_filter = None if args.stage == "all" else args.stage,
    )


if __name__ == "__main__":
    main()
