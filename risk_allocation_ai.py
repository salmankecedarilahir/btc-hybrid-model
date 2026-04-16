"""
╔══════════════════════════════════════════════════════════════════════════╗
║  risk_allocation_ai.py  —  BTC Autonomous AI Quant System              ║
║  LAYER 4 : Dynamic Risk Allocation AI                                   ║
╠══════════════════════════════════════════════════════════════════════════╣
║  TUJUAN  : Tentukan ukuran posisi optimal secara adaptif                ║
║  INPUT   : regime + signal_quality_proba + volatility + perf_state     ║
║  OUTPUT  : position_size_multiplier (0.0 – 1.5)                        ║
║  METODE  : Rule-based Kelly + ML regression + regime-scaled caps        ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import pickle, logging
from pathlib import Path
from typing import Optional

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

log = logging.getLogger(__name__)

REGIME_CAPS = {
    0: 1.20,   # TRENDING_UP   — allow larger positions
    1: 0.70,   # TRENDING_DOWN — reduce exposure
    2: 0.80,   # RANGING       — moderate
    3: 0.50,   # HIGH_VOL      — small positions
    4: 0.90,   # LOW_VOL       — normal
}

REGIME_NAMES = {0:"TRENDING_UP",1:"TRENDING_DOWN",2:"RANGING",3:"HIGH_VOL",4:"LOW_VOL"}


class RiskAllocationAI:
    """
    Dynamic position sizer combining:
      1. Fractional Kelly Criterion (base sizing)
      2. Regime-based cap            (regime safety gate)
      3. ML regression adjustment    (learned from performance)
      4. Volatility scaling          (ATR normalization)
      5. Drawdown guard              (reduce on drawdown)

    Output: position_size_multiplier ∈ [0.0, 1.5]
            Multiply by base_position_pct to get final size.

    Example
    -------
    ai   = RiskAllocationAI()
    ai.fit(hist_df)
    size = ai.predict(live_row)     # single bar
    df   = ai.transform(live_df)    # entire dataframe
    """

    def __init__(self, kelly_fraction=0.25, base_risk=0.01,
                 max_multiplier=1.5, min_multiplier=0.0, seed=42):
        self.kelly_fraction   = kelly_fraction
        self.base_risk        = base_risk
        self.max_mult         = max_multiplier
        self.min_mult         = min_multiplier
        self.seed             = seed

        self.scaler           = StandardScaler()
        self.ml_model         = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=3,
            min_samples_leaf=20, subsample=0.8, random_state=seed
        )
        self.is_fitted        = False
        self._perf_stats      = {}

    # ── INPUTS REQUIRED ───────────────────────────────────────────────────────
    # Each row of input df must have:
    #  signal_quality_proba   : float 0-1
    #  regime                 : int 0-4
    #  vol_real_20            : float (annualized vol)
    #  atr_14_pct             : float
    #  recent_win_rate        : float 0-1 (rolling 20-trade win rate)
    #  recent_pf              : float (rolling 20-trade profit factor)
    #  current_drawdown       : float (-0.3 = 30% DD)

    FEATURES = [
        "signal_quality_proba","regime","vol_real_20","atr_14_pct",
        "recent_win_rate","recent_pf","current_drawdown",
        "rsi_14","bb_width","vol_ratio_10_30",
    ]

    # ── TRAIN ─────────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame, target_col="optimal_size") -> "RiskAllocationAI":
        """
        Train ML adjustment model.
        If target_col missing, generate synthetic targets via Kelly.
        """
        if target_col not in df.columns:
            log.info("Target '%s' not found — generating Kelly targets", target_col)
            df = df.copy()
            df[target_col] = self._kelly_targets(df)

        X, y = self._prep(df, target_col, fit=True)
        log.info("RiskAlloc train: %d samples × %d features", len(y), X.shape[1])

        tscv = TimeSeriesSplit(n_splits=4, gap=20)
        r2s  = []
        for _, (tr, va) in enumerate(tscv.split(X)):
            self.ml_model.fit(X[tr], y[tr])
            r2 = self.ml_model.score(X[va], y[va])
            r2s.append(r2)
        log.info("RiskAlloc CV R²: %.3f ± %.3f", np.mean(r2s), np.std(r2s))

        self.ml_model.fit(X, y)

        # Cache performance stats for adaptive sizing
        if "recent_win_rate" in df.columns:
            self._perf_stats = {
                "mean_wr": df["recent_win_rate"].mean(),
                "mean_pf": df.get("recent_pf", pd.Series([1.5])).mean(),
            }

        self.is_fitted = True
        log.info("RiskAlloc model fitted [OK]")
        return self

    def _kelly_targets(self, df):
        """Generate Kelly-based optimal size targets."""
        wr  = df.get("recent_win_rate", pd.Series(np.full(len(df),0.5), index=df.index))
        pf  = df.get("recent_pf",       pd.Series(np.full(len(df),1.5), index=df.index))
        vol = df.get("vol_real_20",      pd.Series(np.full(len(df),0.5), index=df.index))
        sq  = df.get("signal_quality_proba", pd.Series(np.full(len(df),0.5), index=df.index))

        b       = pf - 1                          # avg win / avg loss ratio
        kelly   = (wr * b - (1 - wr)) / b.clip(lower=0.01)
        fractional = (kelly * self.kelly_fraction).clip(0, 1)
        vol_scale  = (0.3 / vol.clip(lower=0.1)).clip(0.3, 2.0)
        return (fractional * vol_scale * sq).clip(0, 1.5)

    # ── PREDICT ───────────────────────────────────────────────────────────────
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute position_size_multiplier for each row.
        Returns df with added columns.
        """
        out = df.copy()
        out["pos_size_kelly"]   = self._kelly_component(df)
        out["pos_size_regime"]  = df.get("regime", pd.Series(2, index=df.index)).map(REGIME_CAPS).fillna(0.8)
        out["pos_size_dd"]      = self._drawdown_guard(df)
        out["pos_size_vol"]     = self._volatility_scale(df)

        if self.is_fitted:
            X = self._prep_X_live(df)
            ml_adj = self.ml_model.predict(X)
            out["pos_size_ml_adj"] = np.clip(ml_adj, 0, 1.5)
        else:
            out["pos_size_ml_adj"] = 1.0

        # Final combination
        final = (
            out["pos_size_kelly"] *
            out["pos_size_regime"] *
            out["pos_size_dd"] *
            out["pos_size_vol"] *
            out["pos_size_ml_adj"].clip(0.5, 1.5)
        )
        # Floor: kalau ada signal aktif, minimum size 0.10 (10%)
        # Ini mencegah semua posisi jadi nol karena Kelly belum warm-up
        has_signal = (df.get("quant_signal", pd.Series(0, index=df.index)).abs() > 0).values
        final_arr  = np.clip(final.values, self.min_mult, self.max_mult)
        final_arr[has_signal] = np.maximum(final_arr[has_signal], 0.10)
        out["position_size_mult"] = final_arr
        out["regime_name"] = df.get("regime", pd.Series(2, index=df.index)).map(REGIME_NAMES).fillna("RANGING")
        return out

    def predict(self, row: pd.Series) -> float:
        """Single-row prediction."""
        df = pd.DataFrame([row])
        return float(self.transform(df)["position_size_mult"].iloc[0])

    # ── COMPONENTS ────────────────────────────────────────────────────────────
    def _kelly_component(self, df):
        sq  = pd.to_numeric(df.get("signal_quality_proba", pd.Series(0.5, index=df.index)), errors="coerce").fillna(0.5)
        wr  = pd.to_numeric(df.get("recent_win_rate",       pd.Series(0.5, index=df.index)), errors="coerce").fillna(0.5)
        pf  = pd.to_numeric(df.get("recent_pf",             pd.Series(1.5, index=df.index)), errors="coerce").fillna(1.5)

        # Pastikan win rate punya nilai minimum yang sensible
        wr  = wr.clip(0.20, 0.80)
        pf  = pf.clip(0.50, 5.0)
        b   = (pf - 1).clip(lower=0.01)
        k   = (wr * b - (1 - wr)) / b
        kelly_raw = (k * self.kelly_fraction * sq).clip(0, 1)

        # Floor minimum 0.20 saat ada signal aktif
        # Ini agar posisi tidak selalu nol di awal karena history belum ada
        active = sq > 0.3
        kelly_raw = kelly_raw.where(~active, kelly_raw.clip(lower=0.20))
        return kelly_raw

    @staticmethod
    def _drawdown_guard(df):
        dd = df.get("current_drawdown", pd.Series(0.0, index=df.index)).fillna(0)
        # Linear reduction: 0% DD → 1.0, -20% DD → 0.5, -40% DD → 0.0
        return (1 + dd * 2.5).clip(0, 1)

    @staticmethod
    def _volatility_scale(df):
        vol = df.get("vol_real_20", pd.Series(0.5, index=df.index)).fillna(0.5)
        # Target vol = 30%; scale inversely
        return (0.30 / vol.clip(lower=0.05)).clip(0.25, 2.0)

    # ── DATA PREP ─────────────────────────────────────────────────────────────
    def _prep(self, df, target_col, fit=False):
        avail = [c for c in self.FEATURES if c in df.columns]

        # Paksa numeric — buang kolom string
        X = df[avail].copy()
        for col in list(X.columns):
            X[col] = pd.to_numeric(X[col], errors="coerce")
        X = X.dropna(axis=1, how="all")

        if fit: self._feat_cols = list(X.columns)
        X = X.replace([np.inf,-np.inf],np.nan).ffill().bfill().fillna(0)

        # Target — handle NaN dan string
        y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).values.astype(float)

        if fit: Xs = self.scaler.fit_transform(X)
        else:   Xs = self.scaler.transform(X)
        return Xs, y

    def _prep_X_live(self, df):
        avail = [c for c in self._feat_cols if c in df.columns]
        X = df[avail].copy()
        for col in list(X.columns):
            X[col] = pd.to_numeric(X[col], errors="coerce")
        X = X.replace([np.inf,-np.inf],np.nan).ffill().bfill().fillna(0)
        return self.scaler.transform(X)

    # ── PERSIST ───────────────────────────────────────────────────────────────
    def save(self, path="models/risk_alloc_model.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path,"wb") as f: pickle.dump(self,f)
        log.info("RiskAlloc model saved → %s", path)

    @classmethod
    def load(cls, path="models/risk_alloc_model.pkl"):
        with open(path,"rb") as f: m = pickle.load(f)
        return m


def run(df: pd.DataFrame) -> dict:
    model = RiskAllocationAI()
    model.fit(df)
    out = model.transform(df)
    log.info("RiskAlloc output: mean_size=%.3f", out["position_size_mult"].mean())
    return {"model":model,"output":out,"passed":True}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    np.random.seed(42); n=2000
    demo=pd.DataFrame({
        "signal_quality_proba":np.random.uniform(0.3,0.9,n),
        "regime":np.random.randint(0,5,n),
        "vol_real_20":np.abs(np.random.randn(n)*0.3+0.5),
        "atr_14_pct":np.abs(np.random.randn(n)*0.5+2),
        "recent_win_rate":np.random.uniform(0.3,0.7,n),
        "recent_pf":np.abs(np.random.randn(n)*0.5+1.5),
        "current_drawdown":np.random.uniform(-0.3,0,n),
        "rsi_14":np.random.uniform(20,80,n),
        "bb_width":np.abs(np.random.randn(n)+3),
        "vol_ratio_10_30":np.abs(np.random.randn(n)*0.3+1),
    })
    ai=RiskAllocationAI(); ai.fit(demo)
    out=ai.transform(demo)
    print(f"\n[OK] Position Size Summary:")
    print(out[["pos_size_kelly","pos_size_regime","pos_size_dd","pos_size_vol","position_size_mult"]].describe())
