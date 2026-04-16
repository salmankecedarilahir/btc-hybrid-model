"""
╔══════════════════════════════════════════════════════════════════════════╗
║  ai_trading_pipeline.py  —  BTC Autonomous AI Quant System             ║
║  LAYER 9 : AI Trading Pipeline (Full Integration)                       ║
╠══════════════════════════════════════════════════════════════════════════╣
║  TUJUAN  : Integrasikan semua layer menjadi satu pipeline              ║
║                                                                         ║
║  FLOW:                                                                  ║
║    MarketData → FeatureEngine → RegimeDetection → QuantSignal          ║
║    → SignalQualityAI → RiskAllocationAI → TradeExecution               ║
║    → PerformanceLogger → ValidationFramework                           ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import logging, json
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  PERFORMANCE LOGGER
# ─────────────────────────────────────────────────────────────────────────────

class PerformanceLogger:
    """Real-time trade and bar-level performance tracker."""

    def __init__(self, log_path="logs/performance.csv"):
        self.log_path = log_path
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        self._trades: list = []
        self._bars:   list = []

    def log_bar(self, ts, position, ret, equity, regime, sq_proba, pos_size):
        self._bars.append({
            "timestamp":ts,"position":position,"return":ret,
            "equity":equity,"regime":regime,
            "sq_proba":sq_proba,"pos_size":pos_size
        })

    def log_trade(self, entry_ts, exit_ts, direction, entry_px, exit_px,
                  pnl_pct, regime, sq_proba, pos_size):
        self._trades.append({
            "entry_ts":entry_ts,"exit_ts":exit_ts,"direction":direction,
            "entry_px":entry_px,"exit_px":exit_px,"pnl_pct":pnl_pct,
            "regime":regime,"sq_proba":sq_proba,"pos_size":pos_size,
            "win":(pnl_pct > 0)
        })

    def flush(self):
        if self._bars:
            pd.DataFrame(self._bars).to_csv(self.log_path, mode="a",
                                             header=not Path(self.log_path).exists(),
                                             index=False)
            self._bars.clear()

    def get_recent_metrics(self, n_trades=20) -> dict:
        if not self._trades: return {}
        t = self._trades[-n_trades:]
        pnls    = [x["pnl_pct"] for x in t]
        wins    = [x["win"] for x in t]
        wins_v  = [p for p in pnls if p > 0]
        loss_v  = [abs(p) for p in pnls if p <= 0]
        pf      = sum(wins_v) / max(sum(loss_v), 1e-10)
        sharpe  = np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(252)
        return {
            "recent_win_rate": np.mean(wins),
            "recent_pf":       pf,
            "recent_sharpe":   sharpe,
            "n_recent":        len(t),
        }

    def get_drawdown(self, equity_series: pd.Series) -> float:
        if len(equity_series) == 0: return 0.0
        peak = equity_series.expanding().max()
        return float((equity_series / peak - 1).iloc[-1])


# ─────────────────────────────────────────────────────────────────────────────
#  TRADE EXECUTION ENGINE (Simulation)
# ─────────────────────────────────────────────────────────────────────────────

class TradeExecutionEngine:
    """
    Simulated trade execution with slippage and commission.
    In live mode: connects to exchange API.
    """

    def __init__(self, slippage_pct=0.0005, commission_pct=0.0004):
        self.slippage   = slippage_pct
        self.commission = commission_pct
        self._position  = 0
        self._entry_px  = None
        self._entry_ts  = None

    def execute(self, signal: int, price: float, ts,
                pos_size_mult: float = 1.0) -> dict:
        """
        Parameters
        ----------
        signal       : +1 LONG, -1 SHORT, 0 FLAT
        price        : current market price
        pos_size_mult: from RiskAllocationAI

        Returns
        -------
        dict with action taken and net position
        """
        action = "HOLD"
        pnl    = 0.0

        if signal != self._position:
            # Close existing
            if self._position != 0 and self._entry_px:
                raw_pnl = (price - self._entry_px) / self._entry_px * self._position
                pnl     = raw_pnl - self.slippage - self.commission
                action  = "CLOSE"

            # Open new
            if signal != 0:
                exec_px       = price * (1 + signal * self.slippage)
                self._entry_px= exec_px
                self._entry_ts= ts
                self._position= signal
                action = "OPEN_LONG" if signal == 1 else "OPEN_SHORT"
            else:
                self._position = 0
                self._entry_px = None
                self._entry_ts = None
                action = "FLAT"

        return {
            "action"      : action,
            "position"    : self._position,
            "price"       : price,
            "pos_size_mult": pos_size_mult,
            "pnl"         : pnl,
            "timestamp"   : ts,
        }

    def get_position(self): return self._position


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN AI TRADING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class AITradingPipeline:
    """
    Master pipeline integrating all AI layers.

    Architecture
    ------------
    OHLCV Data
      → [1] FeatureEngine         (85+ features)
      → [2] RegimeDetectionModel  (0-4 regime label)
      → [3] Quant Signal          (position: -1/0/+1 from existing logic)
      → [4] SignalQualityModel    (probability 0-1)
      → [5] RiskAllocationAI      (pos_size_mult 0-1.5)
      → [6] TradeExecutionEngine  (execute / simulate)
      → [7] PerformanceLogger     (log bars + trades)

    Example
    -------
    pipeline = AITradingPipeline.from_saved_models(
        regime_path = "models/regime_model.pkl",
        signal_path = "models/signal_quality_model.pkl",
        risk_path   = "models/risk_alloc_model.pkl",
    )
    results = pipeline.run(ohlcv_df, quant_signal_series)
    """

    def __init__(
        self,
        feature_engine    = None,
        regime_model      = None,
        signal_quality    = None,
        risk_allocation   = None,
        exec_engine       = None,
        perf_logger       = None,
        sq_threshold      : float = 0.55,
        regime_filter     : bool  = True,
    ):
        from feature_engine import FeatureEngine
        self.feature_engine  = feature_engine  or FeatureEngine()
        self.regime_model    = regime_model
        self.signal_quality  = signal_quality
        self.risk_allocation = risk_allocation
        self.exec_engine     = exec_engine     or TradeExecutionEngine()
        self.perf_logger     = perf_logger     or PerformanceLogger()
        self.sq_threshold    = sq_threshold
        self.regime_filter   = regime_filter

    @classmethod
    def from_saved_models(cls, regime_path: str, signal_path: str,
                          risk_path: str, **kwargs) -> "AITradingPipeline":
        from regime_detection_model import RegimeDetectionModel
        from signal_quality_model   import SignalQualityModel
        from risk_allocation_ai     import RiskAllocationAI
        return cls(
            regime_model    = RegimeDetectionModel.load(regime_path),
            signal_quality  = SignalQualityModel.load(signal_path),
            risk_allocation = RiskAllocationAI.load(risk_path),
            **kwargs,
        )

    # ── FULL PIPELINE ─────────────────────────────────────────────────────────
    def run(
        self,
        ohlcv_df: pd.DataFrame,
        quant_signal: pd.Series,        # Raw signal from quant core: -1/0/+1
        initial_equity: float = 10_000,
    ) -> pd.DataFrame:
        """
        Run full AI pipeline on historical or live data.

        Parameters
        ----------
        ohlcv_df      : OHLCV DataFrame
        quant_signal  : Series of -1/0/+1 from quant core
        initial_equity: Starting capital

        Returns
        -------
        results_df with all pipeline columns
        """
        log.info("AITradingPipeline.run() — %d bars", len(ohlcv_df))

        # ── STEP 1: Feature Engineering ───────────────────────────────────────
        feat_df = self.feature_engine.transform(ohlcv_df)
        feat_df = feat_df.ffill().fillna(0)

        # ── STEP 2: Regime Detection ──────────────────────────────────────────
        if self.regime_model and self.regime_model.is_fitted:
            regime_labels, regime_proba = self.regime_model.predict(feat_df)
        else:
            regime_labels = pd.Series(2, index=feat_df.index, name="regime")
            regime_proba  = pd.DataFrame()
        feat_df["regime"] = regime_labels

        # ── STEP 3: Quant Signal (already computed, attach it) ────────────────
        feat_df["quant_signal"]    = quant_signal.reindex(feat_df.index).ffill().fillna(0)
        feat_df["signal_direction"]= feat_df["quant_signal"]
        feat_df["signal_strength"] = feat_df["quant_signal"].abs()

        # ── STEP 4: Signal Quality ────────────────────────────────────────────
        if self.signal_quality and self.signal_quality.is_fitted:
            try:
                sq_proba = self.signal_quality.predict_proba(feat_df)
            except Exception as e:
                log.warning("SignalQuality predict failed: %s — using 0.6", e)
                sq_proba = pd.Series(0.6, index=feat_df.index)
        else:
            sq_proba = pd.Series(0.6, index=feat_df.index)
        feat_df["signal_quality_proba"] = sq_proba

        # ── STEP 5: Signal Filtering ──────────────────────────────────────────
        feat_df["signal_approved"] = (
            (sq_proba >= self.sq_threshold) | (feat_df["quant_signal"] == 0)
        ).astype(int)

        # Suppress signal if low quality
        feat_df["filtered_signal"] = feat_df["quant_signal"] * feat_df["signal_approved"]

        # ── STEP 6: Risk Allocation ───────────────────────────────────────────
        # Add performance state for risk model
        feat_df = self._attach_perf_state(feat_df, initial_equity)

        if self.risk_allocation and self.risk_allocation.is_fitted:
            try:
                risk_out = self.risk_allocation.transform(feat_df)
                feat_df["position_size_mult"] = pd.to_numeric(
                    risk_out["position_size_mult"], errors="coerce"
                ).fillna(1.0)
            except Exception as e:
                log.warning("RiskAlloc transform failed: %s — using default 1.0", e)
                feat_df["position_size_mult"] = 1.0
        else:
            feat_df["position_size_mult"] = 1.0

        # ── STEP 7: Simulate Execution ────────────────────────────────────────
        feat_df = self._simulate_execution(feat_df, ohlcv_df, initial_equity)

        # ── STEP 8: Performance Logging ───────────────────────────────────────
        self.perf_logger.flush()

        log.info("Pipeline complete | final_equity=%.2f | n_bars=%d",
                 feat_df["equity"].iloc[-1], len(feat_df))
        return feat_df

    def _attach_perf_state(self, feat_df, initial_equity):
        """Attach rolling performance features for risk model."""
        if "strategy_return" not in feat_df.columns:
            feat_df["recent_win_rate"] = 0.5
            feat_df["recent_pf"]       = 1.5
            feat_df["current_drawdown"]= 0.0
            return feat_df

        sr = feat_df["strategy_return"]
        feat_df["recent_win_rate"] = (sr > 0).rolling(20).mean().fillna(0.5)
        wins   = sr.clip(lower=0).rolling(20).sum()
        losses = (-sr.clip(upper=0)).rolling(20).sum()
        feat_df["recent_pf"] = (wins / losses.replace(0, np.nan)).fillna(1.5)
        eq   = (1 + sr).cumprod() * initial_equity
        peak = eq.expanding().max()
        feat_df["current_drawdown"] = ((eq/peak - 1).clip(-1, 0)).fillna(0)
        return feat_df

    def _simulate_execution(self, feat_df, ohlcv_df, initial_equity):
        """Bar-by-bar execution simulation."""
        close  = ohlcv_df["close"].reindex(feat_df.index)
        equity = initial_equity
        equities, returns, positions = [], [], []

        for ts, row in feat_df.iterrows():
            try:
                signal   = int(row.get("filtered_signal", 0) or 0)
                price    = float(close.loc[ts])
                ps_mult  = float(row.get("position_size_mult", 1.0) or 1.0)
                sq_proba = float(row.get("signal_quality_proba", 0.5) or 0.5)
                regime   = int(row.get("regime", 2) or 2)
                if np.isnan(price): price = close.ffill().loc[ts]
            except Exception:
                signal, ps_mult, sq_proba, regime = 0, 1.0, 0.5, 2

            exec_result = self.exec_engine.execute(signal, price, ts, ps_mult)
            bar_ret     = exec_result["pnl"] * ps_mult
            if np.isnan(bar_ret): bar_ret = 0.0
            equity      = equity * (1 + bar_ret)

            equities.append(equity)
            returns.append(bar_ret)
            positions.append(exec_result["position"])

            self.perf_logger.log_bar(ts, signal, bar_ret, equity,
                                      regime, sq_proba, ps_mult)

        feat_df["equity"]          = equities
        feat_df["strategy_return"] = returns
        feat_df["position"]        = positions
        return feat_df

    # ── SINGLE BAR (LIVE) ─────────────────────────────────────────────────────
    def step(self, ohlcv_row: pd.Series, quant_signal: int,
             recent_feat_df: pd.DataFrame) -> dict:
        """
        Process single bar in live trading.

        Parameters
        ----------
        ohlcv_row     : latest OHLCV bar as Series
        quant_signal  : current signal from quant core
        recent_feat_df: last N bars of feature data (for context)

        Returns
        -------
        dict with signal, position_size, action
        """
        feat = self.feature_engine.transform(
            pd.DataFrame([ohlcv_row]).rename(columns=str.lower)
        )
        # Regime
        regime = 2
        if self.regime_model and self.regime_model.is_fitted and len(recent_feat_df) > 10:
            labels, _ = self.regime_model.predict(recent_feat_df.tail(50))
            regime = int(labels.iloc[-1])
        feat["regime"] = regime

        # SQ proba
        sq_proba = 0.6
        if self.signal_quality and self.signal_quality.is_fitted:
            feat["signal_direction"] = quant_signal
            feat["signal_strength"]  = abs(quant_signal)
            sq_proba = float(self.signal_quality.predict_proba(feat).iloc[0])

        # Signal filter
        approved = sq_proba >= self.sq_threshold or quant_signal == 0
        final_sig = quant_signal if approved else 0

        # Position size
        pos_mult = 1.0
        if self.risk_allocation and self.risk_allocation.is_fitted:
            feat["signal_quality_proba"] = sq_proba
            pos_mult = self.risk_allocation.predict(feat.iloc[0])

        return {
            "timestamp"          : ohlcv_row.name,
            "quant_signal"       : quant_signal,
            "signal_quality_proba": sq_proba,
            "signal_approved"    : approved,
            "final_signal"       : final_sig,
            "position_size_mult" : pos_mult,
            "regime"             : regime,
        }

    # ── REPORT ────────────────────────────────────────────────────────────────
    def summary_report(self, results_df: pd.DataFrame) -> dict:
        ret  = results_df["strategy_return"]
        eq   = results_df["equity"]
        peak = eq.expanding().max()
        dd   = (eq / peak - 1).min() * 100
        cagr = (eq.iloc[-1] / eq.iloc[0]) ** (2190/max(len(ret),1)) - 1

        # Filter stats
        n_quant   = (results_df["quant_signal"] != 0).sum()
        n_filtered= (results_df["filtered_signal"] != 0).sum()
        filter_rate= (1 - n_filtered/max(n_quant,1)) * 100

        return {
            "cagr_pct"    : round(cagr*100, 2),
            "sharpe"      : round(ret.mean()/(ret.std()+1e-10)*np.sqrt(8760/4), 3),
            "max_dd_pct"  : round(dd, 2),
            "final_equity": round(float(eq.iloc[-1]), 2),
            "n_quant_signals": int(n_quant),
            "n_filtered_signals": int(n_filtered),
            "filter_rate_pct": round(filter_rate, 1),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    np.random.seed(42); n=2000
    dates=pd.date_range("2021-01-01",periods=n,freq="4h")
    price=30000*np.exp(np.cumsum(np.random.randn(n)*0.01))
    ohlcv=pd.DataFrame({"open":price,"close":price*(1+np.random.randn(n)*0.003),
                         "high":price*1.01,"low":price*0.99,
                         "volume":np.abs(np.random.randn(n)*1000+500)},index=dates)
    signal=pd.Series(np.random.choice([-1,0,1],n,p=[0.3,0.4,0.3]),index=dates)

    pipeline=AITradingPipeline()
    results=pipeline.run(ohlcv, signal)
    report=pipeline.summary_report(results)
    print("\n[OK] Pipeline Complete:")
    for k,v in report.items(): print(f"  {k}: {v}")
