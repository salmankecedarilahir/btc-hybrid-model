"""
╔══════════════════════════════════════════════════════════════════════════╗
║  autonomous_research_loop.py  —  BTC Autonomous AI Quant System        ║
║  LAYER 10 : Autonomous Quant Research Loop                              ║
╠══════════════════════════════════════════════════════════════════════════╣
║  TUJUAN  : Otomasi penuh research → training → validasi → deploy       ║
║                                                                         ║
║  WORKFLOW:                                                              ║
║    1. Data Ingestion          (load + validate OHLCV)                   ║
║    2. Feature Engineering     (FeatureEngine)                           ║
║    3. Regime Detection Train  (RegimeDetectionModel)                    ║
║    4. AI Dataset Build        (AITrainingDatasetBuilder)                ║
║    5. Signal Quality Train    (SignalQualityModel)                      ║
║    6. Risk Allocation Train   (RiskAllocationAI)                        ║
║    7. Backtesting             (AITradingPipeline)                       ║
║    8. Monte Carlo             (MonteCarloValidator)                     ║
║    9. Walk Forward            (WalkForwardValidator)                    ║
║   10. OOS Validation          (OOSValidator)                            ║
║   11. Model Selection         (ModelSelectionEngine)                    ║
║   12. Registry + Logging      (ModelRegistry + ExperimentLogger)        ║
║   13. Deployment Gate         (check score ≥ threshold)                 ║
╠══════════════════════════════════════════════════════════════════════════╣
║  CATATAN PENTING:                                                       ║
║    Modul validasi AI menggunakan ai_validation_framework.py             ║
║    (BUKAN validation_framework.py yang sudah ada untuk quant core)      ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import logging, time
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)
DIV = "═" * 68
SEP = "─" * 68


class AutonomousResearchLoop:
    """
    Full autonomous research + deployment pipeline.

    Cara pakai
    ----------
    loop = AutonomousResearchLoop(
        data_path     = "data/btc_backtest_results.csv",
        backtest_path = "data/btc_risk_managed_results.csv",
    )
    loop.run_single()        # satu cycle
    loop.run(n_cycles=3)     # tiga cycle
    """

    def __init__(
        self,
        data_path        : str   = "data/btc_backtest_results.csv",
        backtest_path    : str   = "data/btc_risk_managed_results.csv",
        output_dir       : str   = "models",
        deploy_threshold : float = 60.0,
        sq_threshold     : float = 0.35,   # 0.35 — cukup longgar agar sinyal lolos
        n_mc_sims        : int   = 1000,
        is_bars          : int   = 2000,   # disesuaikan dengan OOS 6021 bars
        oos_bars         : int   = 1500,   # → 2 windows WFV minimal
        train_pct        : float = 0.70,
        seed             : int   = 42,
    ):
        self.data_path        = data_path
        self.backtest_path    = backtest_path
        self.output_dir       = Path(output_dir)
        self.deploy_threshold = deploy_threshold
        self.sq_threshold     = sq_threshold
        self.n_mc_sims        = n_mc_sims
        self.is_bars          = is_bars
        self.oos_bars         = oos_bars
        self.train_pct        = train_pct
        self.seed             = seed
        self._cycle_results   = []

    # ── MAIN ENTRY POINTS ─────────────────────────────────────────────────────

    def run(self, n_cycles: int = 1) -> list:
        """Run N independent research cycles."""
        print(f"\n{DIV}")
        print(f"  AUTONOMOUS RESEARCH LOOP — {n_cycles} cycle(s)")
        print(DIV)
        all_results = []
        for i in range(n_cycles):
            print(f"\n{'='*50}")
            print(f"  CYCLE {i+1}/{n_cycles}  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*50}")
            result = self.run_single(cycle_id=i+1)
            all_results.append(result)
            self._cycle_results.append(result)
        self._print_final_summary(all_results)
        return all_results

    def run_single(self, cycle_id: int = 1) -> dict:
        """Run one complete research cycle."""
        t0     = time.time()
        result = {}

        try:
            # ── STEP 1: Data Ingestion ─────────────────────────────────────
            ohlcv_df, backtest_df = self._step_data_ingestion()
            result["n_bars"] = len(ohlcv_df)

            # ── STEP 2: Feature Engineering ───────────────────────────────
            feat_df = self._step_feature_engineering(ohlcv_df)
            result["n_features"] = feat_df.shape[1]

            # ── STEP 3: Regime Detection ───────────────────────────────────
            regime_model, regime_labels = self._step_regime_detection(feat_df)
            feat_df["regime"] = regime_labels
            result["regime_model"] = regime_model

            # ── STEP 4: Build AI Dataset ───────────────────────────────────
            dataset = self._step_build_dataset(feat_df, backtest_df, regime_labels)
            result["n_samples"] = len(dataset)

            # ── STEP 5: Train Signal Quality ───────────────────────────────
            sq_model = self._step_train_signal_quality(dataset)
            result["sq_model"] = sq_model
            result["cv_auc"]   = np.mean(sq_model.cv_scores_)

            # ── STEP 6: Train Risk Allocation ──────────────────────────────
            risk_model = self._step_train_risk_allocation(dataset)
            result["risk_model"] = risk_model

            # ── STEP 7: Backtest ───────────────────────────────────────────
            bt_returns, bt_metrics = self._step_backtest(
                ohlcv_df, feat_df, backtest_df, sq_model, risk_model, regime_model
            )
            result["bt_metrics"] = bt_metrics

            # ── STEP 8: Monte Carlo ────────────────────────────────────────
            mc_result = self._step_monte_carlo(bt_returns)
            result["mc_result"] = mc_result

            # ── STEP 9: Walk Forward ───────────────────────────────────────
            wfv_result = self._step_walk_forward(bt_returns)
            result["wfv_result"] = wfv_result

            # ── STEP 10: OOS Validation ────────────────────────────────────
            oos_result = self._step_oos_validation(bt_returns)
            result["oos_result"] = oos_result

            # ── STEP 11: Score & Select ────────────────────────────────────
            val_score, val_grade = self._step_score(bt_metrics, mc_result, wfv_result, oos_result)
            result["validation_score"] = val_score
            result["validation_grade"] = val_grade

            # ── STEP 12: Registry + Logging ────────────────────────────────
            model_id = self._step_register_and_log(
                cycle_id, regime_model, sq_model, risk_model,
                bt_metrics, mc_result, wfv_result, oos_result,
                sq_model.cv_scores_, dataset.columns.tolist(), val_score, val_grade
            )
            result["model_id"] = model_id

            # ── STEP 13: Deployment Gate ───────────────────────────────────
            deployed = self._step_deployment_gate(val_score, model_id)
            result["deployed"] = deployed

            result["elapsed"] = round(time.time() - t0, 1)
            result["status"]  = "SUCCESS"

        except Exception as e:
            log.error("Research cycle failed: %s", e, exc_info=True)
            result["status"]  = "FAILED"
            result["error"]   = str(e)
            result["elapsed"] = round(time.time() - t0, 1)

        self._print_cycle_summary(result, cycle_id)
        return result

    # ── STEP IMPLEMENTATIONS ──────────────────────────────────────────────────

    def _step_data_ingestion(self):
        print(f"\n{SEP}\n  STEP 1: Data Ingestion\n{SEP}")

        # ── Load OHLCV / backtest utama ────────────────────────────────────
        if Path(self.data_path).exists():
            ohlcv = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
            # Normalisasi nama kolom ke lowercase
            ohlcv.columns = [c.lower() for c in ohlcv.columns]
            log.info("Loaded data: %s", ohlcv.shape)
        else:
            log.warning("File %s tidak ditemukan — pakai synthetic data", self.data_path)
            ohlcv = self._generate_synthetic_ohlcv(5000)

        # Pastikan kolom volume ada (beberapa backtest CSV tidak punya)
        if "volume" not in ohlcv.columns:
            log.warning("Kolom 'volume' tidak ada — diisi dummy 1000")
            ohlcv["volume"] = 1000.0

        # Pastikan OHLC ada semua
        for col in ["open", "high", "low", "close"]:
            if col not in ohlcv.columns and "close" in ohlcv.columns:
                ohlcv[col] = ohlcv["close"]

        # ── Load backtest risk managed (opsional, untuk target labels) ─────
        backtest = None
        if Path(self.backtest_path).exists():
            backtest = pd.read_csv(self.backtest_path, index_col=0, parse_dates=True)
            backtest.columns = [c.lower() for c in backtest.columns]
            log.info("Loaded backtest: %s", backtest.shape)
        else:
            log.warning("File backtest %s tidak ditemukan — target dibuat otomatis", self.backtest_path)

        print(f"  OHLCV     : {ohlcv.shape[0]:,} bars  ({ohlcv.index[0].date()} → {ohlcv.index[-1].date()})")
        if backtest is not None:
            print(f"  Backtest  : {backtest.shape[0]:,} rows × {backtest.shape[1]} cols")

        # ── TRAIN / OOS SPLIT ─────────────────────────────────────────────────
        # Simpan split index agar model hanya dilatih di IS, backtest di OOS
        n = len(ohlcv)
        split_idx = int(n * self.train_pct)
        self._split_idx   = split_idx
        self._ohlcv_train = ohlcv.iloc[:split_idx]
        self._ohlcv_oos   = ohlcv.iloc[split_idx:]
        self._bt_train    = backtest.iloc[:split_idx] if backtest is not None else None
        self._bt_oos      = backtest.iloc[split_idx:] if backtest is not None else None
        print(f"  Train     : {len(self._ohlcv_train):,} bars → {self._ohlcv_train.index[-1].date()}")
        print(f"  OOS       : {len(self._ohlcv_oos):,} bars → {self._ohlcv_oos.index[-1].date()} (backtest ONLY here)")

        return ohlcv, backtest

    def _step_feature_engineering(self, ohlcv_df):
        print(f"\n{SEP}\n  STEP 2: Feature Engineering\n{SEP}")
        from feature_engine import FeatureEngine
        engine   = FeatureEngine()
        feat_df  = engine.transform(ohlcv_df)
        # Simpan split untuk training vs OOS
        self._feat_train = feat_df.iloc[:self._split_idx]
        self._feat_oos   = feat_df.iloc[self._split_idx:]
        print(f"  Features  : {feat_df.shape[1]}")
        print(f"  Feat train: {len(self._feat_train):,} rows")
        print(f"  Feat OOS  : {len(self._feat_oos):,} rows")
        return feat_df

    def _step_regime_detection(self, feat_df):
        print(f"\n{SEP}\n  STEP 3: Regime Detection Training\n{SEP}")
        from regime_detection_model import RegimeDetectionModel, REGIME_NAMES
        model = RegimeDetectionModel(seed=self.seed)
        # Fit HANYA pada training data
        model.fit(self._feat_train)
        # Predict pada seluruh data (features are backward-looking, no leakage)
        labels, _ = model.predict(feat_df)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(self.output_dir / "regime_model.pkl"))
        dist = labels.value_counts().rename(REGIME_NAMES)
        print(f"  Regime distribution (full):\n{dist.to_string()}")
        return model, labels

    def _step_build_dataset(self, feat_df, backtest_df, regime_labels):
        print(f"\n{SEP}\n  STEP 4: AI Dataset Build\n{SEP}")
        from ai_training_dataset_builder import AITrainingDatasetBuilder
        builder = AITrainingDatasetBuilder()
        output_path = "data/ai_training_dataset.csv"
        # [WARN]️  HANYA gunakan training split — BUKAN full data
        # Ini mencegah data leakage ke model
        regime_train = regime_labels.iloc[:self._split_idx]
        ds = builder.build(
            feat_df       = self._feat_train,
            backtest_df   = self._bt_train,
            regime_labels = regime_train,
            output_path   = output_path,
        )
        print(f"  [WARN]️  Dataset build dari TRAIN only ({len(ds):,} rows) — OOS reserved untuk backtest")
        return ds

    def _step_train_signal_quality(self, dataset):
        print(f"\n{SEP}\n  STEP 5: Signal Quality AI Training\n{SEP}")
        from signal_quality_model import SignalQualityModel

        # Bersihkan dataset — paksa semua kolom ke numeric
        ds = dataset.copy()
        for col in ds.columns:
            ds[col] = pd.to_numeric(ds[col], errors="coerce")
        ds = ds.fillna(0)

        model  = SignalQualityModel(n_estimators=200, seed=self.seed)
        target = "target_profitable_trade"
        if target not in ds.columns:
            log.warning("Kolom target '%s' tidak ada — dibuat dari target_ret_4bar", target)
            ds[target] = (ds.get("target_ret_4bar", pd.Series(0, index=ds.index)) > 0).astype(int)

        model.fit(ds, target_col=target)
        model.save(str(self.output_dir / "signal_quality_model.pkl"))
        print(f"  CV AUC    : {np.mean(model.cv_scores_):.4f} ± {np.std(model.cv_scores_):.4f}")
        return model

    def _step_train_risk_allocation(self, dataset):
        print(f"\n{SEP}\n  STEP 6: Risk Allocation AI Training\n{SEP}")
        from risk_allocation_ai import RiskAllocationAI

        # Bersihkan dataset — paksa semua kolom ke numeric sebelum masuk model
        ds = dataset.copy()
        for col in ds.columns:
            ds[col] = pd.to_numeric(ds[col], errors="coerce")
        ds = ds.fillna(0)

        model = RiskAllocationAI(seed=self.seed)
        model.fit(ds)
        model.save(str(self.output_dir / "risk_alloc_model.pkl"))
        print(f"  Risk model trained [OK]")
        return model

    def _step_backtest(self, ohlcv_df, feat_df, backtest_df,
                       sq_model, risk_model, regime_model):
        print(f"\n{SEP}\n  STEP 7: Backtesting with AI Filter (OOS ONLY)\n{SEP}")

        # [WARN]️  GUNAKAN HANYA OOS DATA untuk backtest — model belum pernah lihat ini
        ohlcv_oos = self._ohlcv_oos
        bt_oos    = self._bt_oos

        if bt_oos is not None and "position" in bt_oos.columns:
            quant_signal = bt_oos["position"].reindex(ohlcv_oos.index).ffill().fillna(0)
        elif bt_oos is not None and "signal" in bt_oos.columns:
            quant_signal = bt_oos["signal"].reindex(ohlcv_oos.index).ffill().fillna(0)
        else:
            log.warning("Kolom position tidak ada di OOS — gunakan semua signal")
            quant_signal = pd.Series(1, index=ohlcv_oos.index)

        print(f"  OOS period: {ohlcv_oos.index[0].date()} → {ohlcv_oos.index[-1].date()}")
        print(f"  OOS bars  : {len(ohlcv_oos):,}")

        from ai_trading_pipeline import AITradingPipeline
        pipeline = AITradingPipeline(
            regime_model    = regime_model,
            signal_quality  = sq_model,
            risk_allocation = risk_model,
            sq_threshold    = self.sq_threshold,
        )
        results = pipeline.run(ohlcv_oos, quant_signal)
        metrics = pipeline.summary_report(results)

        ret    = results["strategy_return"]
        wins   = ret[ret > 0].sum()
        losses = -ret[ret < 0].sum()
        metrics["pf"]     = round(wins / max(losses, 1e-10), 3)
        metrics["sharpe"] = round(ret.mean() / (ret.std() + 1e-10) * np.sqrt(8760 / 4), 3)

        print(f"  CAGR (OOS): {metrics.get('cagr_pct', 0):+.1f}%")
        print(f"  Sharpe    : {metrics.get('sharpe', 0):.3f}")
        print(f"  PF        : {metrics.get('pf', 0):.3f}")
        print(f"  Max DD    : {metrics.get('max_dd_pct', 0):.1f}%")
        print(f"  Filtered  : {metrics.get('filter_rate_pct', 0):.1f}% of signals removed")
        if metrics.get('filter_rate_pct', 0) > 80:
            print(f"  [WARN]️  Filter rate >80% — pertimbangkan turunkan sq_threshold lebih lanjut")

        return ret, metrics

    def _step_monte_carlo(self, returns):
        print(f"\n{SEP}\n  STEP 8: Monte Carlo Validation\n{SEP}")
        from ai_validation_framework import MonteCarloValidator
        # Filter returns yang valid (bukan 0 semua)
        active = returns[returns != 0]
        if len(active) < 50:
            log.warning("Terlalu sedikit return aktif (%d) untuk MC reliable", len(active))
            returns_mc = returns
        else:
            returns_mc = active
        mc = MonteCarloValidator(n_sims=self.n_mc_sims, seed=self.seed)
        return mc.run(returns_mc)

    def _step_walk_forward(self, returns):
        print(f"\n{SEP}\n  STEP 9: Walk Forward Validation\n{SEP}")
        from ai_validation_framework import WalkForwardValidator

        n = len(returns)
        # Adaptif: sesuaikan window dengan panjang data aktual
        # Minimal harus ada 2 windows → is + oos < n/2
        is_bars  = min(self.is_bars,  int(n * 0.40))
        oos_bars = min(self.oos_bars, int(n * 0.25))

        if is_bars + oos_bars >= n:
            is_bars  = int(n * 0.40)
            oos_bars = int(n * 0.25)

        print(f"  WFV adaptive: n={n} is_bars={is_bars} oos_bars={oos_bars}")
        wfv = WalkForwardValidator(is_bars=is_bars, oos_bars=oos_bars)
        return wfv.run(returns)

    def _step_oos_validation(self, returns):
        print(f"\n{SEP}\n  STEP 10: Out-of-Sample Validation\n{SEP}")
        # [WARN]️  Import dari ai_validation_framework, BUKAN validation_framework
        from ai_validation_framework import OOSValidator
        return OOSValidator().run(returns)

    def _step_score(self, bt, mc, wf, oo):
        """
        Composite validation score 0-100.
        Bobot: BT=25, MC=25, WFV=25, OOS=25
        Kalau WFV tidak bisa jalan (windows=0), bobotnya dialihkan ke OOS.
        """
        score = 0

        # ── BT performance (25 pts) ──────────────────────────────────────────
        if bt.get("pf", 0) >= 1.5:      score += 15
        elif bt.get("pf", 0) >= 1.2:    score += 10
        elif bt.get("pf", 0) >= 1.0:    score += 5
        if bt.get("sharpe", 0) >= 2.0:  score += 10
        elif bt.get("sharpe", 0) >= 1.0:score += 7
        elif bt.get("sharpe", 0) >= 0.5:score += 3

        # ── MC (25 pts) ───────────────────────────────────────────────────────
        ruin = mc.get("ruin_rate", 100)
        if ruin < 1:    score += 15
        elif ruin < 5:  score += 10
        elif ruin < 15: score += 5
        pf_p5 = mc.get("pf_p5", 0)
        if pf_p5 >= 1.0:   score += 10
        elif pf_p5 >= 0.8: score += 6
        elif pf_p5 >= 0.6: score += 2

        # ── WFV (25 pts) — skip jika windows=0, alihkan ke OOS ───────────────
        wf_windows = wf.get("n_windows", 0)
        if wf_windows == 0:
            # WFV tidak bisa jalan → OOS dapat bobot ekstra
            oos_bonus = True
            print("  ℹ️  WFV skip (data OOS terlalu pendek) → OOS dapat bobot extra")
        else:
            oos_bonus = False
            if wf.get("avg_oos_pf", 0) >= 1.0:    score += 15
            elif wf.get("avg_oos_pf", 0) >= 0.85: score += 10
            elif wf.get("avg_oos_pf", 0) >= 0.70: score += 5
            if wf.get("oos_preserved", 0) >= 50:  score += 10
            elif wf.get("oos_preserved", 0) >= 30:score += 5

        # ── OOS (25 pts, atau 50 pts jika WFV skip) ───────────────────────────
        oos_max = 50 if oos_bonus else 25
        oos_score = 0
        oos_pf = oo.get("oos_pf", 0)
        if oos_pf >= 1.2:    oos_score += 15
        elif oos_pf >= 1.0:  oos_score += 10
        elif oos_pf >= 0.9:  oos_score += 5
        oos_sh = oo.get("oos_sharpe", 0)
        if oos_sh >= 1.0:    oos_score += 10
        elif oos_sh >= 0.5:  oos_score += 6
        elif oos_sh >= 0.1:  oos_score += 2
        # Scale ke max bobot
        if oos_bonus:
            oos_score = int(oos_score * 50 / 25)
        score += oos_score

        grade = ("EXCELLENT" if score >= 85 else
                 "GOOD"      if score >= 70 else
                 "MARGINAL"  if score >= 50 else "FAIL")

        print(f"  Score breakdown: BT={min(score,25)}  MC=..  OOS={oos_score}  Total={score}/100")
        return score, grade

    def _step_register_and_log(self, cycle_id, regime_model, sq_model, risk_model,
                                bt, mc, wf, oo, cv_scores, features,
                                val_score, val_grade):
        from model_retraining_engine import ModelRegistry
        from experiment_logger import ExperimentLogger

        registry = ModelRegistry(str(self.output_dir / "registry"))
        logger   = ExperimentLogger()
        version  = f"v{datetime.now().strftime('%Y%m%d_%H%M')}_c{cycle_id}"

        registry.register(regime_model, version=version + "_regime",
                          model_type="regime", performance={})
        mid = registry.register(sq_model, version=version + "_sq",
                                 model_type="signal_quality",
                                 performance={"bt_pf": bt.get("pf", 0),
                                              "oos_pf": oo.get("oos_pf", 0)})
        registry.register(risk_model, version=version + "_risk",
                          model_type="risk_allocation", performance={})

        logger.log_from_results(
            model_version    = version,
            model_type       = "signal_quality",
            bt_metrics       = bt,
            mc_result        = mc,
            wfv_result       = wf,
            oos_result       = oo,
            cv_scores        = cv_scores,
            features         = features[:20],
            validation_score = val_score,
            validation_grade = val_grade,
        )
        return mid

    def _step_deployment_gate(self, val_score, model_id):
        print(f"\n{SEP}\n  STEP 13: Deployment Gate\n{SEP}")
        if val_score >= self.deploy_threshold:
            print(f"  [GREEN] APPROVED  — score={val_score:.0f} ≥ threshold={self.deploy_threshold:.0f}")
            print(f"  Model {model_id} promoted to CHAMPION")
            return True
        else:
            print(f"  [RED] BLOCKED   — score={val_score:.0f} < threshold={self.deploy_threshold:.0f}")
            print(f"  Fix issues and re-run research loop.")
            return False

    # ── REPORTING ─────────────────────────────────────────────────────────────

    def _print_cycle_summary(self, result, cycle_id):
        print(f"\n{DIV}")
        print(f"  CYCLE {cycle_id} SUMMARY")
        print(DIV)
        if result.get("status") == "FAILED":
            print(f"  ❌ FAILED: {result.get('error', 'unknown')}")
        else:
            print(f"  Status          : {result.get('status', '?')}")
            print(f"  Validation Score: {result.get('validation_score', 0):.0f}/100  [{result.get('validation_grade', '?')}]")
            print(f"  CV AUC          : {result.get('cv_auc', 0):.4f}")
            bt = result.get("bt_metrics", {})
            print(f"  BT  — PF={bt.get('pf',0):.3f}  Sharpe={bt.get('sharpe',0):.3f}  DD={bt.get('max_dd_pct',0):.1f}%")
            mc = result.get("mc_result", {})
            print(f"  MC  — Ruin={mc.get('ruin_rate',0):.2f}%  PF_p5={mc.get('pf_p5',0):.3f}")
            wf = result.get("wfv_result", {})
            print(f"  WFV — OOS_PF={wf.get('avg_oos_pf',0):.3f}  Windows={wf.get('n_windows',0)}")
            print(f"  Deployed        : {'[OK] YES' if result.get('deployed') else '❌ NO'}")
            print(f"  Elapsed         : {result.get('elapsed', 0):.1f}s")
        print(DIV)

    def _print_final_summary(self, all_results):
        print(f"\n{DIV}")
        print("  AUTONOMOUS RESEARCH LOOP — FINAL SUMMARY")
        print(DIV)
        for i, r in enumerate(all_results, 1):
            status = r.get("status", "?")
            score  = r.get("validation_score", 0)
            grade  = r.get("validation_grade", "?")
            dep    = "[OK]" if r.get("deployed") else "❌"
            print(f"  Cycle {i}: {status}  score={score:.0f} [{grade}]  deployed={dep}")
        print(DIV)

    @staticmethod
    def _generate_synthetic_ohlcv(n=5000) -> pd.DataFrame:
        np.random.seed(42)
        dates = pd.date_range("2019-01-01", periods=n, freq="4h")
        price = 10000 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
        price = np.maximum(price, 1.0)
        return pd.DataFrame({
            "open"  : price,
            "close" : price * (1 + np.random.randn(n) * 0.003),
            "high"  : price * 1.01,
            "low"   : price * 0.99,
            "volume": np.abs(np.random.randn(n) * 1000 + 500),
        }, index=dates)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    loop = AutonomousResearchLoop(
        data_path        = "data/btc_backtest_results.csv",
        backtest_path    = "data/btc_risk_managed_results.csv",
        output_dir       = "models",
        deploy_threshold = 60,
        n_mc_sims        = 1000,   # 1000 sims untuk MC yang credible
        sq_threshold     = 0.45,   # 0.45 — tidak over-filter sinyal
        train_pct        = 0.70,   # 70% train, 30% OOS backtest
    )

    # Run 2 cycles: cycle 1 = baseline, cycle 2 = retrain dengan data terbaru
    results = loop.run(n_cycles=2)

    # Ringkasan semua cycles
    print(f"\n{'='*50}")
    for i, r in enumerate(results, 1):
        print(f"Cycle {i}: {r.get('status')}  "
              f"Score={r.get('validation_score',0):.0f}/100  "
              f"[{r.get('validation_grade','?')}]  "
              f"Deployed={'YES' if r.get('deployed') else 'NO'}")
    print(f"{'='*50}")
