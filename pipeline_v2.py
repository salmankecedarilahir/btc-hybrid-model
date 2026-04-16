"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  pipeline_v2.py — BTC Hybrid AI v2                                         ║
║  Complete Redesigned Production Pipeline                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  RUN:  python pipeline_v2.py                                               ║
║  or import and call: PipelineV2(...).run()                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import logging, time
from pathlib import Path
from datetime import datetime
from typing import Optional

from feature_engine_v2 import FeatureEngineV2
from signal_model_v2   import SignalModelV2
from risk_model_v2     import RiskModelV2
from validation_v2     import ValidationV2, LeakageAudit, _metrics

log = logging.getLogger(__name__)
DIV = "═" * 68
SEP = "─" * 68


class PipelineV2:
    """
    Complete v2 pipeline — leakage-free, regime-aware, Sharpe-optimized.

    Data splits (chronological, NEVER shuffled):
      IS   (In-Sample train) : 0% → 60%
      VAL  (Validation)      : 60% → 80%   ← hyperparameter tuning
      OOS  (Out-of-Sample)   : 80% → 100%  ← final evaluation ONLY

    Usage
    -----
    pipeline = PipelineV2(
        data_path     = "data/btc_backtest_results.csv",
        backtest_path = "data/btc_risk_managed_results.csv",
    )
    report = pipeline.run()
    """

    def __init__(
        self,
        data_path          : str   = "data/btc_backtest_results.csv",
        backtest_path      : str   = "data/btc_risk_managed_results.csv",
        output_dir         : str   = "models",
        is_pct             : float = 0.60,
        val_pct            : float = 0.20,
        target_lookahead   : int   = 4,     # bars ahead for return target
        target_vol         : float = 0.25,  # annual vol target for risk model
        signal_threshold_pct: float = 65.0, # percentile threshold for signals
        n_mc_sims          : int   = 1000,
        min_wfv_train_bars : int   = 3000,
        seed               : int   = 42,
    ):
        self.data_path     = data_path
        self.backtest_path = backtest_path
        self.output_dir    = Path(output_dir)
        self.is_pct        = is_pct
        self.val_pct       = val_pct
        self.la            = target_lookahead
        self.target_vol    = target_vol
        self.sig_pct       = signal_threshold_pct
        self.n_mc_sims     = n_mc_sims
        self.min_wfv_train = min_wfv_train_bars
        self.seed          = seed

        # Components
        self.feature_engine = FeatureEngineV2()
        self.signal_model   = SignalModelV2(n_estimators=300, seed=seed)
        self.risk_model     = RiskModelV2(target_vol=target_vol)
        self.validator      = ValidationV2(
            min_train_bars = min_wfv_train_bars,
            oos_bars       = 500,
            embargo_bars   = 48,
            n_mc_sims      = n_mc_sims,
        )

    # ── MAIN RUN ─────────────────────────────────────────────────────────────

    def run(self) -> dict:
        t0 = time.time()
        print(f"\n{DIV}")
        print("  BTC HYBRID AI SYSTEM v2 — FULL PIPELINE")
        print(DIV)

        # ── STEP 1: Load + split data ─────────────────────────────────────────
        ohlcv, backtest = self._load_data()
        n     = len(ohlcv)
        is_end  = int(n * self.is_pct)
        val_end = int(n * (self.is_pct + self.val_pct))

        ohlcv_is  = ohlcv.iloc[:is_end]
        ohlcv_val = ohlcv.iloc[is_end:val_end]
        ohlcv_oos = ohlcv.iloc[val_end:]

        print(f"\n  IS  : {len(ohlcv_is):>6,} bars  "
              f"({ohlcv_is.index[0].date()} → {ohlcv_is.index[-1].date()})")
        print(f"  VAL : {len(ohlcv_val):>6,} bars  "
              f"({ohlcv_val.index[0].date()} → {ohlcv_val.index[-1].date()})")
        print(f"  OOS : {len(ohlcv_oos):>6,} bars  "
              f"({ohlcv_oos.index[0].date()} → {ohlcv_oos.index[-1].date()})")

        # ── STEP 2: Build target (BEFORE feature engineering) ────────────────
        # [WARN]️  Target must be computed from raw prices, not features
        print(f"\n{SEP}\n  STEP 2: Target Engineering\n{SEP}")
        target = self._build_target(ohlcv, backtest)
        print(f"  Target: expected_return_{self.la}bar")
        print(f"  Shape : {target.shape}  Mean={target.mean()*100:.3f}%  "
              f"Std={target.std()*100:.3f}%")
        print(f"  Positive rate: {(target>0).mean()*100:.1f}%")
        if abs(target.autocorr(1)) > 0.1:
            print(f"  [WARN]️  Target autocorr={target.autocorr(1):.3f} — check for leakage")
        else:
            print(f"  [OK] Target autocorr OK ({target.autocorr(1):.3f})")

        # ── STEP 3: Feature engineering (scaler fit on IS only) ──────────────
        print(f"\n{SEP}\n  STEP 3: Feature Engineering (v2 — 25 features)\n{SEP}")
        feat_is  = self.feature_engine.fit_transform(ohlcv_is)   # fit + transform IS
        feat_val = self.feature_engine.transform(ohlcv_val)       # transform only
        feat_oos = self.feature_engine.transform(ohlcv_oos)       # transform only
        feat_all = self.feature_engine.transform(ohlcv)           # for regime + backtest

        print(f"  [OK] Scaler fitted on IS ({len(ohlcv_is):,} bars) ONLY")
        print(f"  Features: {feat_is.shape[1]}  |  IS={feat_is.shape[0]}  "
              f"VAL={feat_val.shape[0]}  OOS={feat_oos.shape[0]}")

        # ── STEP 4: Regime detection (fit on IS) ─────────────────────────────
        print(f"\n{SEP}\n  STEP 4: Regime Detection\n{SEP}")
        regime_all, regime_conf_all = self._detect_regime(feat_all, ohlcv)

        # ── STEP 5: Leakage audit ─────────────────────────────────────────────
        print(f"\n{SEP}\n  STEP 5: Leakage Audit\n{SEP}")
        audit_result = LeakageAudit().run(
            feat_all, target, ohlcv["close"], is_split=self.is_pct
        )

        # ── STEP 6: Train signal model on IS only ────────────────────────────
        print(f"\n{SEP}\n  STEP 6: Signal Model Training (IS only)\n{SEP}")
        target_is = target.reindex(feat_is.index)
        regime_is = regime_all.reindex(feat_is.index).fillna(2)

        self.signal_model.fit(feat_is, target_is, regime_is)
        print(f"  CV Sharpe: {np.mean(self.signal_model.cv_sharpes_):.3f} "
              f"± {np.std(self.signal_model.cv_sharpes_):.3f}")

        # ── STEP 7: Generate signals for VAL (hyperparameter check) ──────────
        print(f"\n{SEP}\n  STEP 7: Signal Generation on VAL\n{SEP}")
        val_signals = self.signal_model.get_signal(
            feat_val,
            regime_all.reindex(feat_val.index),
            regime_conf_all.reindex(feat_val.index),
            threshold_pct=self.sig_pct,
        )
        val_pass_rate = (val_signals["signal"] != 0).mean() * 100
        print(f"  VAL pass rate: {val_pass_rate:.1f}%  (target: 30-60%)")
        if val_pass_rate < 20:
            print(f"  [WARN]️  Too few signals — lower threshold_pct from {self.sig_pct}")
        elif val_pass_rate > 65:
            print(f"  [WARN]️  Too many signals — raise threshold_pct from {self.sig_pct}")
        else:
            print(f"  [OK] Pass rate in healthy range")

        # ── STEP 8: OOS backtest ──────────────────────────────────────────────
        print(f"\n{SEP}\n  STEP 8: OOS Backtest (TRUE out-of-sample)\n{SEP}")
        oos_signals = self.signal_model.get_signal(
            feat_oos,
            regime_all.reindex(feat_oos.index),
            regime_conf_all.reindex(feat_oos.index),
            threshold_pct=self.sig_pct,
        )

        oos_results = self._simulate(ohlcv_oos, oos_signals)
        oos_metrics = _metrics(oos_results["strategy_return"], "OOS")

        print(f"  CAGR    : {oos_metrics.get('cagr_pct',0):+.1f}%")
        print(f"  Sharpe  : {oos_metrics.get('sharpe',0):.3f}")
        print(f"  Sortino : {oos_metrics.get('sortino',0):.3f}")
        print(f"  PF      : {oos_metrics.get('pf',0):.3f}")
        print(f"  MaxDD   : {oos_metrics.get('max_dd',0):.1f}%")
        print(f"  WinRate : {oos_metrics.get('win_rate',0):.1f}%")

        # ── STEP 9: Full IS+VAL+OOS backtest for validation framework ─────────
        print(f"\n{SEP}\n  STEP 9: Walk Forward + Monte Carlo Validation\n{SEP}")
        # Retrain on IS+VAL for full signal
        feat_isvl = pd.concat([feat_is, feat_val])
        tgt_isvl  = target.reindex(feat_isvl.index)
        reg_isvl  = regime_all.reindex(feat_isvl.index).fillna(2)

        final_model = SignalModelV2(n_estimators=300, seed=self.seed)
        final_model.fit(feat_isvl, tgt_isvl, reg_isvl, verbose=False)

        all_signals = final_model.get_signal(
            feat_all,
            regime_all,
            regime_conf_all,
            threshold_pct=self.sig_pct,
        )
        all_results  = self._simulate(ohlcv, all_signals)
        all_returns  = all_results["strategy_return"]

        val_report = self.validator.run(all_returns)

        # ── STEP 10: Save models ──────────────────────────────────────────────
        self.output_dir.mkdir(parents=True, exist_ok=True)
        final_model.save(str(self.output_dir / "signal_model_v2.pkl"))
        log.info("Models saved to %s", self.output_dir)

        # ── Final report ──────────────────────────────────────────────────────
        elapsed = time.time() - t0
        report = {
            "oos_metrics"  : oos_metrics,
            "validation"   : val_report,
            "audit"        : audit_result,
            "cv_sharpe"    : round(np.mean(self.signal_model.cv_sharpes_), 3),
            "oos_pass_rate": round(val_pass_rate, 1),
            "elapsed"      : round(elapsed, 1),
            "grade"        : val_report.get("grade", "UNKNOWN"),
            "score"        : val_report.get("score", 0),
        }

        print(f"\n{DIV}")
        print("  PIPELINE v2 — COMPLETE")
        print(f"  Score    : {report['score']}/100  [{report['grade']}]")
        print(f"  OOS Sharpe: {oos_metrics.get('sharpe',0):.3f}")
        print(f"  Elapsed  : {elapsed:.1f}s")
        print(DIV)

        return report

    # ── HELPERS ───────────────────────────────────────────────────────────────

    def _load_data(self):
        print(f"\n{SEP}\n  STEP 1: Data Load\n{SEP}")
        if Path(self.data_path).exists():
            ohlcv = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
            ohlcv.columns = [c.lower() for c in ohlcv.columns]
        else:
            log.warning("Data not found — using synthetic")
            ohlcv = self._synthetic_ohlcv(10000)

        for col in ["open","high","low","close","volume"]:
            if col not in ohlcv.columns:
                ohlcv[col] = ohlcv.get("close", ohlcv.iloc[:,0])
        if "volume" not in ohlcv.columns:
            ohlcv["volume"] = 1000.0

        backtest = None
        if Path(self.backtest_path).exists():
            backtest = pd.read_csv(self.backtest_path, index_col=0, parse_dates=True)
            backtest.columns = [c.lower() for c in backtest.columns]
        print(f"  Loaded {len(ohlcv):,} bars")
        return ohlcv, backtest

    def _build_target(self, ohlcv, backtest=None) -> pd.Series:
        """
        Build forward return target — STRICTLY no lookahead.

        target[t] = close[t + la] / close[t] - 1

        If backtest has position column, we weight by the direction:
        target[t] = position[t] × (close[t+la] / close[t] - 1)

        This gives the ACTUAL P&L of following the quant signal,
        which is what the model should learn to predict.
        """
        close  = ohlcv["close"]
        fwd_ret = (close.shift(-self.la) / close - 1)

        if backtest is not None and "position" in backtest.columns:
            pos = backtest["position"].reindex(ohlcv.index).ffill().fillna(0)
            # Only keep bars where there's an active position
            target = (pos * fwd_ret).where(pos != 0, fwd_ret)
        else:
            target = fwd_ret

        # Drop last `la` bars (they have NaN forward returns — no leakage)
        target = target.iloc[:-self.la]
        target = pd.to_numeric(target, errors="coerce").fillna(0)
        return target.rename("target")

    def _detect_regime(self, feat_all, ohlcv):
        """
        Simple regime detection using rule-based approach.
        Returns (regime_series, confidence_series).
        """
        close  = ohlcv["close"]
        ret    = close.pct_change()
        ema20  = close.ewm(20, adjust=False).mean()
        ema50  = close.ewm(50, adjust=False).mean()
        ema200 = close.ewm(200, adjust=False).mean()
        vol20  = ret.rolling(20).std() * np.sqrt(2190)
        vol60  = ret.rolling(60).std() * np.sqrt(2190)

        # Regime rules:
        # BULL(0): price > EMA200, EMA20 > EMA50, vol not extreme
        # BEAR(1): price < EMA200 or significant downtrend
        # CHOP(2): else (ranging, high vol, mixed signals)
        vol_high = vol20 > vol60 * 1.5

        regime = pd.Series(2, index=close.index)  # default CHOP
        bull_mask = (close > ema200) & (ema20 > ema50) & (~vol_high)
        bear_mask = (close < ema200) & (ema20 < ema50)
        regime[bull_mask] = 0
        regime[bear_mask] = 1

        # Confidence: how strongly each condition is met
        trend_score = ((close > ema20).astype(float) +
                       (close > ema50).astype(float) +
                       (close > ema200).astype(float) +
                       (ema20 > ema50).astype(float) +
                       (ema50 > ema200).astype(float)) / 5.0
        confidence = (trend_score * 2 - 1).abs().clip(0, 1)

        dist = regime.value_counts().rename({0:"BULL",1:"BEAR",2:"CHOP"})
        print(f"  Regime distribution:\n{dist.to_string()}")

        return regime, confidence

    def _simulate(self, ohlcv, signals_df) -> pd.DataFrame:
        """Simple backtest simulation on a given OHLCV + signals dataset."""
        close = ohlcv["close"]
        ret   = close.pct_change().fillna(0)

        signal   = signals_df["signal"].reindex(close.index).fillna(0)
        convict  = signals_df.get("conviction", pd.Series(0.5, index=close.index))
        regime_s = signals_df.get("regime",    pd.Series(0, index=close.index))

        # Compute position sizes
        equity_dummy = pd.Series(np.ones(len(close)) * 10000, index=close.index)
        sizing = self.risk_model.compute(close, signal, convict,
                                          regime_s, equity_dummy)

        strategy_ret = signal.shift(1).fillna(0) * ret * sizing["position_size_mult"].shift(1).fillna(0)
        equity = (1 + strategy_ret).cumprod() * 10000

        out = pd.DataFrame(index=close.index)
        out["strategy_return"]   = strategy_ret
        out["equity"]            = equity
        out["signal"]            = signal
        out["position_size_mult"]= sizing["position_size_mult"]
        return out

    @staticmethod
    def _synthetic_ohlcv(n=10000) -> pd.DataFrame:
        np.random.seed(42)
        dates = pd.date_range("2019-01-01", periods=n, freq="4h")
        price = 10000 * np.exp(np.cumsum(np.random.randn(n)*0.01))
        return pd.DataFrame({
            "open":price,"close":price*(1+np.random.randn(n)*0.003),
            "high":price*1.01,"low":price*0.99,
            "volume":np.abs(np.random.randn(n)*1000+500),
        }, index=dates)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    pipeline = PipelineV2(
        data_path    = "data/btc_backtest_results.csv",
        backtest_path= "data/btc_risk_managed_results.csv",
        output_dir   = "models",
        is_pct       = 0.60,
        val_pct      = 0.20,
        target_vol   = 0.25,
        signal_threshold_pct = 65.0,
        n_mc_sims    = 1000,
        min_wfv_train_bars = 3000,
    )

    report = pipeline.run()
