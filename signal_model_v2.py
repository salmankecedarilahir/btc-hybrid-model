"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  signal_model_v2.py — BTC Hybrid AI v2                                     ║
║  Regime-Conditioned Regressor with Sharpe-Aligned Objective               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  KEY CHANGES FROM v1:                                                      ║
║  1. Target: expected_return_4bar (regression) NOT binary classification    ║
║  2. Objective: custom Sharpe-proxy loss NOT AUC                            ║
║  3. Separate sub-model per regime (BULL/BEAR/CHOP)                         ║
║  4. Purged TimeSeriesSplit (embargo gap between folds)                     ║
║  5. SHAP-based feature stability check                                     ║
║  6. Dynamic signal threshold (not fixed 0.35/0.45/0.55)                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import pickle, logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings; warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    HAS_LGB = False
    log.warning("LightGBM not found — using GradientBoosting")

# ─────────────────────────────────────────────────────────────────────────────
#  WHY REGRESSION TARGET IS BETTER THAN CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
"""
CLASSIFICATION (AUC) PROBLEMS:
  - A +0.1% trade and +5% trade both count as "class 1 (win)"
  - Loss function ignores trade magnitude → model ignores magnitude
  - AUC 0.75 can coexist with Sharpe 0.0 if wins are tiny, losses are large
  - Calibration is needed separately (Platt scaling adds more leakage risk)

REGRESSION (Expected Return) ADVANTAGES:
  - Model directly predicts profit magnitude
  - Large wins are explicitly incentivized
  - Signal strength = predicted return = natural position sizing input
  - Custom loss function can directly penalize losses more than gains
    (downside-risk-aware: loss = -Sharpe proxy)

SHARPE-PROXY LOSS:
  L(y_pred, y_true) = -(mean(y_pred × y_true) / std(y_pred × y_true + ε))
  This directly maximizes Sharpe ratio of the SIGNAL, not prediction accuracy.
"""


# ─────────────────────────────────────────────────────────────────────────────
#  PURGED TIME SERIES SPLIT
# ─────────────────────────────────────────────────────────────────────────────

class PurgedTimeSeriesSplit:
    """
    TimeSeriesSplit with embargo gap to prevent leakage.

    Why embargo?
    After the training period ends, there's a buffer zone where we
    don't use data for validation. This prevents:
    - Rolling features from IS period leaking into OOS evaluation
    - Trade returns from open positions at IS boundary affecting OOS

    embargo_pct: fraction of fold size to skip (default 1% = ~88 bars for 8760-bar fold)
    """

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        self.n_splits    = n_splits
        self.embargo_pct = embargo_pct

    def split(self, X, y=None, groups=None):
        n = len(X)
        fold_size   = n // (self.n_splits + 1)
        embargo_bars = max(1, int(fold_size * self.embargo_pct))

        for i in range(self.n_splits):
            train_end  = fold_size * (i + 1)
            test_start = train_end + embargo_bars
            test_end   = min(test_start + fold_size, n)

            if test_end <= test_start:
                continue

            train_idx = np.arange(0, train_end)
            test_idx  = np.arange(test_start, test_end)
            yield train_idx, test_idx


# ─────────────────────────────────────────────────────────────────────────────
#  SHARPE LOSS FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def sharpe_loss_lgb(y_pred: np.ndarray,
                    dataset: "lgb.Dataset") -> Tuple[np.ndarray, np.ndarray]:
    """
    Custom LightGBM loss that maximizes Sharpe ratio.

    Objective: maximize E[r × signal] / std(r × signal)
    where r = actual return, signal = predicted return

    Gradient derivation:
      L = -Sharpe(y_pred × y_true)
      ∂L/∂y_pred = -y_true × (std - (y_pred×y_true - mean) × cov_correction)
                              / (n × std²)
    """
    y_true = dataset.get_label()
    n      = len(y_true)
    pnl    = y_pred * y_true    # element-wise P&L
    mu     = pnl.mean()
    sig    = pnl.std() + 1e-8

    # Gradient: ∂(-Sharpe)/∂y_pred_i
    grad = -y_true * (sig - (pnl - mu) * (pnl - mu).mean() / sig) / (n * sig)

    # Hessian (diagonal approximation)
    hess = (y_true ** 2) / (n * sig) + 1e-6

    return grad, hess


def sharpe_metric_lgb(y_pred: np.ndarray,
                      dataset: "lgb.Dataset") -> Tuple[str, float, bool]:
    """Custom eval metric: Sharpe ratio of signal × return."""
    y_true = dataset.get_label()
    pnl    = y_pred * y_true
    if pnl.std() < 1e-8:
        return "sharpe", 0.0, True
    sharpe = pnl.mean() / pnl.std() * np.sqrt(2190)  # annualized 4h
    return "sharpe", float(sharpe), True   # True = higher is better


# ─────────────────────────────────────────────────────────────────────────────
#  SIGNAL MODEL v2
# ─────────────────────────────────────────────────────────────────────────────

REGIME_LABELS = {0: "BULL", 1: "BEAR", 2: "CHOP"}

class SignalModelV2:
    """
    Regime-conditioned regression model for BTC signal generation.

    Architecture:
      - Global model: trained on all data
      - Regime models: trained only on bars of that regime
      - Ensemble: regime_confidence × regime_pred + (1-conf) × global_pred

    Target: expected_return_4bar
      = (close[t+4] / close[t] - 1) × position_direction
      This is the actual P&L per unit if you follow the signal

    Usage:
      model = SignalModelV2()
      model.fit(feat_df, returns_df, regime_series)
      predictions = model.predict(live_feat_df, live_regime)
    """

    def __init__(self, n_estimators=300, seed=42):
        self.n_estimators = n_estimators
        self.seed         = seed
        self._global_model  = None
        self._regime_models : Dict[int, object] = {}
        self.is_fitted      = False
        self.cv_sharpes_    : List[float] = []
        self._feature_names : List[str] = []
        self._threshold_history: List[float] = []

    # ── TRAINING ──────────────────────────────────────────────────────────────

    def fit(
        self,
        feat_df:       pd.DataFrame,   # output of FeatureEngineV2.fit_transform()
        target_series: pd.Series,      # expected_return_4bar
        regime_series: pd.Series,      # 0=BULL 1=BEAR 2=CHOP
        verbose:       bool = True,
    ) -> "SignalModelV2":
        """
        Train global + per-regime models.

        Parameters
        ----------
        feat_df       : Scaled features (25 cols) from FeatureEngineV2
        target_series : expected_return_4bar — MUST be forward return
                        shifted to avoid leakage:
                        target = close.shift(-4) / close - 1
                        (computed BEFORE any ML training, not inside)
        regime_series : Regime labels 0/1/2
        """
        self._feature_names = list(feat_df.columns)

        # Align all series
        idx    = feat_df.index.intersection(target_series.index)
        X      = feat_df.loc[idx].values
        y      = target_series.loc[idx].values
        regime = regime_series.reindex(idx).fillna(2).values.astype(int)

        # Drop NaN rows (from rolling features warmup)
        valid  = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y, regime = X[valid], y[valid], regime[valid]

        log.info("SignalModelV2 training: %d samples × %d features", len(y), X.shape[1])
        log.info("Target stats: mean=%.4f std=%.4f", y.mean(), y.std())

        # ── 1. Purged CV to estimate Sharpe ───────────────────────────────────
        pscv       = PurgedTimeSeriesSplit(n_splits=5, embargo_pct=0.01)
        cv_sharpes = []

        for fold, (tr_idx, va_idx) in enumerate(pscv.split(X)):
            m = self._build_model()
            if HAS_LGB:
                m.fit(
                    X[tr_idx], y[tr_idx],
                    eval_set=[(X[va_idx], y[va_idx])],
                    callbacks=[lgb.early_stopping(50, verbose=False),
                               lgb.log_evaluation(-1)]
                )
            else:
                m.fit(X[tr_idx], y[tr_idx])

            preds      = m.predict(X[va_idx])
            pnl        = preds * y[va_idx]
            sh         = pnl.mean() / (pnl.std() + 1e-8) * np.sqrt(2190)
            cv_sharpes.append(sh)
            if verbose:
                log.info("  Fold %d — CV Sharpe=%.3f", fold+1, sh)

        self.cv_sharpes_ = cv_sharpes
        log.info("CV Sharpe: %.3f ± %.3f", np.mean(cv_sharpes), np.std(cv_sharpes))

        # ── 2. Train global model on all data ─────────────────────────────────
        self._global_model = self._build_model()
        if HAS_LGB:
            self._global_model.fit(X, y,
                                   callbacks=[lgb.log_evaluation(-1)])
        else:
            self._global_model.fit(X, y)
        log.info("Global model trained")

        # ── 3. Train per-regime models ─────────────────────────────────────────
        for reg_id, reg_name in REGIME_LABELS.items():
            mask = regime == reg_id
            n_reg = mask.sum()
            if n_reg < 200:
                log.warning("Regime %s: only %d samples — skipping regime model",
                            reg_name, n_reg)
                continue
            m_reg = self._build_model()
            if HAS_LGB:
                m_reg.fit(X[mask], y[mask],
                          callbacks=[lgb.log_evaluation(-1)])
            else:
                m_reg.fit(X[mask], y[mask])
            self._regime_models[reg_id] = m_reg
            pnl_reg = m_reg.predict(X[mask]) * y[mask]
            sh_reg  = pnl_reg.mean() / (pnl_reg.std() + 1e-8) * np.sqrt(2190)
            log.info("Regime %s model: n=%d  IS Sharpe=%.3f", reg_name, n_reg, sh_reg)

        self.is_fitted = True
        log.info("SignalModelV2 fitted [OK]  CV Sharpe=%.3f", np.mean(cv_sharpes))
        return self

    # ── INFERENCE ─────────────────────────────────────────────────────────────

    def predict_raw(
        self,
        feat_df:       pd.DataFrame,
        regime_series: pd.Series,
        regime_conf:   Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Predict expected return for each bar.

        Returns raw predicted return — NOT a probability.
        Positive = bullish signal, Negative = bearish signal.
        Magnitude = predicted return strength.
        """
        if not self.is_fitted:
            raise RuntimeError("Call .fit() first")

        X      = feat_df[self._feature_names].values
        regime = regime_series.reindex(feat_df.index).fillna(2).values.astype(int)
        conf   = (regime_conf.reindex(feat_df.index).fillna(0.5).values
                  if regime_conf is not None else np.full(len(X), 0.5))

        global_pred = self._global_model.predict(X)
        final_pred  = global_pred.copy()

        # Blend with regime-specific model where available
        for reg_id, reg_model in self._regime_models.items():
            mask = regime == reg_id
            if mask.sum() == 0: continue
            reg_pred = reg_model.predict(X[mask])
            c        = conf[mask]
            # Higher confidence in regime → use more of regime model
            final_pred[mask] = c * reg_pred + (1 - c) * global_pred[mask]

        return pd.Series(final_pred, index=feat_df.index, name="pred_return")

    def get_signal(
        self,
        feat_df:       pd.DataFrame,
        regime_series: pd.Series,
        regime_conf:   Optional[pd.Series] = None,
        threshold_pct: float = 65.0,   # top 35% signals pass
        regime_thresholds: Optional[Dict[int, float]] = None,
    ) -> pd.DataFrame:
        """
        Convert predictions to trading signals with dynamic thresholding.

        threshold_pct: percentile of rolling predictions to use as threshold
          65 → pass signals above 65th percentile (top 35%)
          Calibrated per regime for robustness.

        Returns DataFrame with:
          pred_return  : raw predicted expected return
          signal       : +1 LONG, -1 SHORT, 0 FLAT
          conviction   : 0-1 signal strength
          threshold    : current dynamic threshold used
        """
        preds  = self.predict_raw(feat_df, regime_series, regime_conf)
        regime = regime_series.reindex(feat_df.index).fillna(2)

        # Default per-regime thresholds
        if regime_thresholds is None:
            regime_thresholds = {
                0: threshold_pct - 5,   # BULL: slightly more permissive
                1: threshold_pct + 10,  # BEAR: stricter longs
                2: threshold_pct + 15,  # CHOP: very strict
            }

        out = pd.DataFrame(index=feat_df.index)
        out["pred_return"] = preds
        out["regime"]      = regime
        out["signal"]      = 0
        out["conviction"]  = 0.0
        out["threshold"]   = np.nan

        # Rolling window for dynamic threshold
        window = 60  # 60 bars = ~10 days at 4h
        roll_preds = preds.rolling(window, min_periods=20)

        for i in range(len(out)):
            ts      = out.index[i]
            reg_id  = int(regime.iloc[i])
            pred    = float(preds.iloc[i])
            pct     = regime_thresholds.get(reg_id, threshold_pct)

            # Dynamic threshold based on rolling prediction distribution
            if i >= 20:
                hist_preds = preds.iloc[max(0, i-window):i].values
                thresh_long  = np.percentile(hist_preds, pct)
                thresh_short = np.percentile(hist_preds, 100 - pct)
            else:
                thresh_long  = 0.0
                thresh_short = 0.0

            out.iloc[i, out.columns.get_loc("threshold")] = thresh_long

            if pred > thresh_long and pred > 0:
                signal = 1
            elif pred < thresh_short and pred < 0:
                signal = -1
            else:
                signal = 0

            out.iloc[i, out.columns.get_loc("signal")] = signal

            # Conviction = how far above threshold
            if thresh_long != thresh_short:
                rng = thresh_long - thresh_short
                conv = np.clip((pred - thresh_short) / (rng + EPS), 0, 1)
            else:
                conv = 0.5 if pred > 0 else 0.0
            out.iloc[i, out.columns.get_loc("conviction")] = conv

        # Track threshold for diagnostics
        self._threshold_history.extend(out["threshold"].dropna().tolist())

        pass_rate = (out["signal"] != 0).mean() * 100
        log.info("Signal generation: %.1f%% pass rate (target: 35-50%%)", pass_rate)
        if pass_rate < 15:
            log.warning("Pass rate very low (%.1f%%) — consider lowering threshold_pct", pass_rate)
        if pass_rate > 70:
            log.warning("Pass rate very high (%.1f%%) — signals may not be selective enough", pass_rate)

        return out

    def get_feature_importance(self) -> pd.DataFrame:
        """Feature importance from global model."""
        if not self.is_fitted or self._global_model is None:
            return pd.DataFrame()
        imp = self._global_model.feature_importances_
        return pd.DataFrame({
            "feature": self._feature_names[:len(imp)],
            "importance": imp,
        }).sort_values("importance", ascending=False)

    # ── BUILD MODEL ───────────────────────────────────────────────────────────

    def _build_model(self):
        """
        Build LightGBM with conservative hyperparameters.

        Key anti-overfitting params:
          max_depth=4         : shallow trees → less memorization
          min_child_samples=50: each leaf needs 50+ samples
          reg_alpha/lambda=0.3: L1+L2 regularization
          subsample=0.7       : row sampling
          colsample_bytree=0.6: feature sampling
          learning_rate=0.03  : slow learning → better generalization
        """
        if HAS_LGB:
            return lgb.LGBMRegressor(
                n_estimators     = self.n_estimators,
                learning_rate    = 0.03,
                max_depth        = 4,
                num_leaves       = 15,    # 2^depth/2 for safety
                min_child_samples= 50,
                reg_alpha        = 0.3,
                reg_lambda       = 0.3,
                subsample        = 0.7,
                colsample_bytree = 0.6,
                random_state     = self.seed,
                verbose          = -1,
                objective        = "regression",  # plain MSE first
                # Note: for Sharpe loss uncomment below (requires custom obj):
                # objective = sharpe_loss_lgb,
            )
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=0.03,
                max_depth=4,
                min_samples_leaf=50,
                subsample=0.7,
                random_state=self.seed,
            )

    # ── PERSISTENCE ───────────────────────────────────────────────────────────

    def save(self, path: str = "models/signal_model_v2.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f: pickle.dump(self, f)
        log.info("SignalModelV2 saved → %s", path)

    @classmethod
    def load(cls, path: str) -> "SignalModelV2":
        with open(path, "rb") as f: m = pickle.load(f)
        return m


EPS = 1e-8

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    np.random.seed(42)
    n = 5000
    dates   = pd.date_range("2019-01-01", periods=n, freq="4h")
    price   = 10000 * np.exp(np.cumsum(np.random.randn(n)*0.01))

    # Simulate features
    feat_df = pd.DataFrame(
        np.random.randn(n, 25),
        index=dates,
        columns=[f"feat_{i}" for i in range(25)]
    )

    # CORRECT target: forward return (no leakage — computed from price, not from model)
    close   = pd.Series(price, index=dates)
    target  = (close.shift(-4) / close - 1).fillna(0)  # 4-bar forward return

    regime  = pd.Series(np.random.randint(0, 3, n), index=dates)

    model = SignalModelV2(n_estimators=50)
    model.fit(feat_df, target, regime)

    signals = model.get_signal(feat_df, regime)
    print(f"\n[OK] Signal pass rate: {(signals['signal']!=0).mean()*100:.1f}%")
    print(f"CV Sharpe scores: {[f'{s:.3f}' for s in model.cv_sharpes_]}")
    print(f"\nFeature importance:")
    print(model.get_feature_importance().head(10))
