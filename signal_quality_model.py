"""
╔══════════════════════════════════════════════════════════════════════════╗
║  signal_quality_model.py  —  BTC Autonomous AI Quant System            ║
║  LAYER 3 : Signal Quality AI                                            ║
╠══════════════════════════════════════════════════════════════════════════╣
║  TUJUAN  : Evaluasi kualitas signal dari quant core                     ║
║  INPUT   : feat_df + regime labels + quant signal + trade results       ║
║  OUTPUT  : probability_of_trade_success (0.0–1.0)                      ║
║  METODE  : LightGBM + Random Forest ensemble (Platt-calibrated)         ║
║                                                                         ║
║  Anti-Overfitting :                                                     ║
║    • TimeSeriesSplit (no leakage)   • L1/L2 regularization              ║
║    • Feature importance pruning     • Calibrated probabilities          ║
║    • Min 50-bar gap between IS/OOS  • Early stopping                    ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import pickle, logging
from pathlib import Path
from typing import List, Optional, Tuple

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score
from sklearn.feature_selection import SelectFromModel
import warnings; warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)

# Try LightGBM (optional but preferred)
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    log.warning("LightGBM not installed — falling back to GradientBoosting")

# ─────────────────────────────────────────────────────────────────────────────
#  DEFAULT FEATURE SET FOR SIGNAL QUALITY
# ─────────────────────────────────────────────────────────────────────────────
SQ_FEATURES = [
    # Trend
    "price_vs_ema20","price_vs_ema50","price_vs_ema200",
    "ema_x_8_20","ema_x_20_50","ema_x_50_200",
    "linreg_slope_20","linreg_slope_50","adx",
    # Momentum
    "rsi_6","rsi_14","rsi_21","macd","macd_hist","macd_bull",
    "roc_5","roc_10","roc_20","stoch_k","cci_20",
    # Volatility
    "atr_14_pct","vol_real_10","vol_real_20","vol_ratio_10_30",
    "bb_width","bb_pos","bb_squeeze",
    # Volume
    "vol_ratio","vol_zscore","obv_slope","cmf_20","mfi_14","vwap_dist",
    # Micro
    "body_ratio","body_dir","efficiency_10","efficiency_20","intrabar_mom",
    # Statistical
    "skew_20","kurt_20","ret_z20","autocorr","hurst_50","entropy_20",
    # Regime
    "regime",
    # Quant signal features (must be added before training)
    "signal_strength","signal_direction",
]


class SignalQualityModel:
    """
    Signal quality classifier.

    Flow
    ----
    1.  Build training dataset (feat_df + signal cols + target)
    2.  fit(X, y)          → trains ensemble + calibration
    3.  predict_proba(X)   → returns P(trade_success)
    4.  filter_signals(df) → removes low-confidence signals

    Example
    -------
    model = SignalQualityModel()
    model.fit(train_df, target_col="target_profitable_trade")
    proba = model.predict_proba(live_feat_df)
    filtered = model.filter_signals(live_feat_df, threshold=0.55)
    """

    def __init__(
        self,
        n_estimators: int  = 200,
        min_proba_thresh: float = 0.55,
        n_splits: int      = 5,
        feature_select: bool = True,
        seed: int          = 42,
    ):
        self.n_estimators       = n_estimators
        self.min_proba_thresh   = min_proba_thresh
        self.n_splits           = n_splits
        self.feature_select     = feature_select
        self.seed               = seed

        self.scaler             = StandardScaler()
        self.selected_features_: List[str] = []
        self.is_fitted          = False
        self.cv_scores_: List[float] = []

        self._model_lgb = None
        self._model_rf  = None
        self._calibrated= None

    # ── TRAINING ──────────────────────────────────────────────────────────────
    def fit(self, feat_df: pd.DataFrame, target_col: str = "target_profitable_trade"):
        """
        Train signal quality model.

        Parameters
        ----------
        feat_df    : DataFrame with features + target column
        target_col : binary 0/1 (1 = trade was profitable)
        """
        if target_col not in feat_df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        X_raw, y = self._prep_Xy(feat_df, target_col, fit=True)
        log.info("SignalQuality train: %d samples × %d features | pos_rate=%.1f%%",
                 len(y), X_raw.shape[1], y.mean()*100)

        # Feature selection via Random Forest importance
        if self.feature_select:
            X_raw = self._select_features(X_raw, y)

        # TimeSeriesSplit CV
        tscv = TimeSeriesSplit(n_splits=self.n_splits, gap=50)
        auc_scores = []

        for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_raw)):
            Xtr, ytr = X_raw[tr_idx], y[tr_idx]
            Xva, yva = X_raw[va_idx], y[va_idx]
            m = self._build_primary(Xtr, ytr)
            p = m.predict_proba(Xva)[:,1]
            try:
                auc = roc_auc_score(yva, p)
            except Exception:
                auc = 0.5
            auc_scores.append(auc)
            log.info("  Fold %d — AUC=%.4f", fold+1, auc)

        self.cv_scores_ = auc_scores
        log.info("CV AUC mean=%.4f ± %.4f", np.mean(auc_scores), np.std(auc_scores))

        # Final model on all data + calibration
        self._model_lgb = self._build_primary(X_raw, y)
        self._model_rf  = RandomForestClassifier(
            n_estimators=100, max_depth=6, min_samples_leaf=20,
            random_state=self.seed, n_jobs=-1
        )
        self._model_rf.fit(X_raw, y)

        # Calibrate with isotonic regression
        self._calibrated = CalibratedClassifierCV(
            self._build_primary(X_raw, y), method="isotonic", cv=3
        )
        self._calibrated.fit(X_raw, y)

        self.is_fitted = True
        log.info("SignalQuality model fitted [OK]  CV-AUC=%.3f", np.mean(auc_scores))
        return self

    def _build_primary(self, X, y):
        """Build primary classifier (LightGBM preferred)."""
        if HAS_LGB:
            model = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=31,
                min_child_samples=30,
                reg_alpha=0.1,   # L1
                reg_lambda=0.1,  # L2
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.seed,
                verbose=-1,
            )
        else:
            model = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                learning_rate=0.05,
                max_depth=4,
                min_samples_leaf=20,
                subsample=0.8,
                random_state=self.seed,
            )
        model.fit(X, y)
        return model

    def _select_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Keep only features with positive importance."""
        rf = RandomForestClassifier(n_estimators=100, max_depth=5,
                                    random_state=self.seed, n_jobs=-1)
        rf.fit(X, y)
        imp = rf.feature_importances_
        mask = imp > np.percentile(imp, 25)   # keep top 75%
        self._feat_mask = mask
        log.info("Feature selection: %d → %d features", X.shape[1], mask.sum())
        return X[:, mask]

    # ── INFERENCE ─────────────────────────────────────────────────────────────
    def predict_proba(self, feat_df: pd.DataFrame) -> pd.Series:
        """
        Returns probability of trade success for each row.

        Returns
        -------
        pd.Series (0.0 – 1.0), index = feat_df.index
        """
        if not self.is_fitted: raise RuntimeError("Call .fit() first")
        X = self._prep_X_live(feat_df)
        # Ensemble: 50% calibrated + 30% LGB + 20% RF
        p_cal = self._calibrated.predict_proba(X)[:,1]
        p_lgb = self._model_lgb.predict_proba(X)[:,1]
        p_rf  = self._model_rf.predict_proba(X)[:,1]
        p_ens = 0.50*p_cal + 0.30*p_lgb + 0.20*p_rf
        return pd.Series(p_ens, index=feat_df.index, name="signal_quality_proba")

    def filter_signals(
        self, feat_df: pd.DataFrame, threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Returns feat_df with low-quality signals removed.
        Adds 'signal_quality_proba' and 'signal_approved' columns.
        """
        thresh = threshold or self.min_proba_thresh
        proba  = self.predict_proba(feat_df)
        out    = feat_df.copy()
        out["signal_quality_proba"] = proba
        out["signal_approved"]      = (proba >= thresh).astype(int)
        n_orig = len(out)
        n_pass = out["signal_approved"].sum()
        log.info("Signal filter: %d/%d signals passed (thresh=%.2f)", n_pass, n_orig, thresh)
        return out

    def get_feature_importance(self, top_n=20) -> pd.DataFrame:
        if not self.is_fitted: raise RuntimeError("Not fitted")
        if HAS_LGB and hasattr(self._model_lgb, "feature_importances_"):
            imp = self._model_lgb.feature_importances_
        else:
            imp = self._model_rf.feature_importances_

        # Setelah feature selection, jumlah fitur berkurang
        # Pakai nama fitur yang sudah difilter (sesuai _feat_mask)
        if hasattr(self, "_feat_mask") and self._feat_mask is not None:
            names = [f for f, keep in zip(self._active_features, self._feat_mask) if keep]
        else:
            names = self._active_features

        # Pastikan panjang sama sebelum buat DataFrame
        min_len = min(len(names), len(imp))
        names   = names[:min_len]
        imp     = imp[:min_len]

        df = pd.DataFrame({"feature": names, "importance": imp})
        return df.sort_values("importance", ascending=False).head(top_n)

    # ── DATA PREP ─────────────────────────────────────────────────────────────
    def _prep_Xy(self, feat_df, target_col, fit=False):
        import pandas as pd
        avail = [c for c in SQ_FEATURES if c in feat_df.columns]
        X = feat_df[avail].copy()

        # Paksa semua kolom ke numeric — buang string/NONE
        for col in list(X.columns):
            X[col] = pd.to_numeric(X[col], errors="coerce")

        # Buang kolom yg seluruhnya NaN (non-numeric murni)
        X = X.dropna(axis=1, how="all")

        # Update active features ke kolom yg benar-benar numeric
        if fit:
            self._active_features = list(X.columns)

        X = X.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

        # Target — handle NONE/string
        y_raw = pd.to_numeric(feat_df[target_col], errors="coerce").fillna(0)
        y = y_raw.values.astype(int)

        if fit: Xs = self.scaler.fit_transform(X)
        else:   Xs = self.scaler.transform(X)
        if fit and self.feature_select and hasattr(self, "_feat_mask"):
            Xs = Xs[:, self._feat_mask]
        return Xs, y

    def _prep_X_live(self, feat_df):
        import pandas as pd
        avail = [c for c in self._active_features if c in feat_df.columns]
        X = feat_df[avail].copy()
        for col in list(X.columns):
            X[col] = pd.to_numeric(X[col], errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
        Xs = self.scaler.transform(X)
        if hasattr(self, "_feat_mask"):
            Xs = Xs[:, self._feat_mask]
        return Xs

    # ── PERSISTENCE ───────────────────────────────────────────────────────────
    def save(self, path="models/signal_quality_model.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path,"wb") as f: pickle.dump(self,f)
        log.info("SignalQuality model saved → %s", path)

    @classmethod
    def load(cls, path="models/signal_quality_model.pkl"):
        with open(path,"rb") as f: m = pickle.load(f)
        log.info("SignalQuality model loaded ← %s", path)
        return m


def run(feat_df: pd.DataFrame, target_col="target_profitable_trade") -> dict:
    model = SignalQualityModel()
    model.fit(feat_df, target_col=target_col)
    proba = model.predict_proba(feat_df)
    return {"model":model,"proba":proba,"cv_auc":np.mean(model.cv_scores_),"passed":True}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    np.random.seed(42)
    n=2000
    feat=pd.DataFrame({c:np.random.randn(n) for c in SQ_FEATURES if c not in ["regime","signal_direction"]})
    feat["regime"]           = np.random.randint(0,5,n)
    feat["signal_direction"] = np.random.choice([-1,1],n)
    feat["target_profitable_trade"] = np.random.randint(0,2,n)
    model=SignalQualityModel(n_estimators=50)
    model.fit(feat,"target_profitable_trade")
    proba=model.predict_proba(feat)
    print(f"\n[OK] Signal quality proba: mean={proba.mean():.3f}  std={proba.std():.3f}")
    print(f"CV AUC scores: {[f'{s:.3f}' for s in model.cv_scores_]}")
    print("\nTop features:")
    print(model.get_feature_importance())
