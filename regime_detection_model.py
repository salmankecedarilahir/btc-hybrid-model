"""
╔══════════════════════════════════════════════════════════════════════════╗
║  regime_detection_model.py  —  BTC Autonomous AI Quant System          ║
║  LAYER 2 : Market Regime Detection AI                                   ║
╠══════════════════════════════════════════════════════════════════════════╣
║  TUJUAN  : Identifikasi kondisi market secara otomatis                  ║
║  INPUT   : feat_df dari FeatureEngine                                   ║
║  OUTPUT  : regime label (0-4) + probabilitas per regime                 ║
║  METODE  : Gaussian Mixture Model + Rule-based override + Smoothing     ║
║                                                                         ║
║  Labels  :  0=TRENDING_UP  1=TRENDING_DOWN  2=RANGING                  ║
║             3=HIGH_VOL     4=LOW_VOL                                    ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import pickle, logging
from pathlib import Path
from typing import Tuple

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

REGIME_NAMES = {0:"TRENDING_UP", 1:"TRENDING_DOWN", 2:"RANGING",
                3:"HIGH_VOL",    4:"LOW_VOL"}

REGIME_FEATURES = [
    "ema_50_slope","ema_20_slope","price_vs_ema200",
    "vol_real_20","vol_ratio_10_30","rsi_14","adx",
    "bb_width","ret_z20","macd_hist",
]


class RegimeDetectionModel:
    """
    Market Regime Detector.

    Pipeline : features → StandardScaler → GMM(5 components)
               → semantic label assignment → rule-override → mode-smooth

    Example
    -------
    model = RegimeDetectionModel()
    model.fit(feat_df)
    labels, proba_df = model.predict(feat_df)
    model.save("models/regime_model.pkl")
    """

    def __init__(self, n_regimes=5, smooth_window=5, seed=42):
        self.n_regimes    = n_regimes
        self.smooth_window= smooth_window
        self.scaler  = StandardScaler()
        self.gmm     = GaussianMixture(n_components=n_regimes, covariance_type="full",
                                       n_init=5, max_iter=300, random_state=seed)
        self.is_fitted    = False
        self._regime_map  = {}
        self._feat_cols   = []

    # ── TRAIN ─────────────────────────────────────────────────────────────────
    def fit(self, feat_df: pd.DataFrame) -> "RegimeDetectionModel":
        X = self._prep(feat_df, fit=True)
        Xs = self.scaler.fit_transform(X)
        self.gmm.fit(Xs)
        raw = self.gmm.predict(Xs)
        self._regime_map = self._assign_labels(feat_df, raw)
        self.is_fitted = True
        log.info("Regime model fitted | map=%s", self._regime_map)
        return self

    def _assign_labels(self, feat_df, raw_labels):
        """Map GMM cluster IDs to semantic regime labels."""
        mapping, stats = {}, {}
        for c in range(self.n_regimes):
            sub = feat_df.iloc[raw_labels == c]
            if len(sub) < 5: mapping[c]=2; continue
            safe_mean = lambda col: sub[col].mean() if col in sub.columns and len(sub) > 0 else 0.0
            stats[c] = {
                "slope"  : safe_mean("ema_50_slope"),
                "vol"    : safe_mean("vol_real_20"),
                "vol_rat": safe_mean("vol_ratio_10_30"),
                "rsi"    : safe_mean("rsi_14"),
                "bb_w"   : safe_mean("bb_width"),
            }
        sr = sorted(stats, key=lambda c: stats[c]["vol_rat"], reverse=True)
        mapping[sr[0]] = 3
        # LOW_VOL → lowest realized vol
        for c in sorted(stats, key=lambda c: stats[c]["vol"]):
            if c not in mapping: mapping[c]=4; break
        # TRENDING_UP → highest slope among remaining
        rem = [c for c in stats if c not in mapping]
        if rem:
            best = max(rem, key=lambda c: stats[c]["slope"])
            mapping[best] = 0
            rem.remove(best)
        # TRENDING_DOWN → lowest slope
        if rem:
            worst = min(rem, key=lambda c: stats[c]["slope"])
            mapping[worst] = 1
            rem.remove(worst)
        # RANGING → rest
        for c in rem: mapping[c] = 2
        for c in range(self.n_regimes):
            if c not in mapping: mapping[c]=2
        return mapping

    # ── PREDICT ───────────────────────────────────────────────────────────────
    def predict(self, feat_df: pd.DataFrame,
                smooth=True, override=True) -> Tuple[pd.Series, pd.DataFrame]:
        if not self.is_fitted: raise RuntimeError("Call .fit() first")
        X  = self._prep(feat_df)
        Xs = self.scaler.transform(X)
        raw   = self.gmm.predict(Xs)
        proba = self.gmm.predict_proba(Xs)
        sem   = np.array([self._regime_map.get(r, 2) for r in raw])
        if override: sem = self._override(feat_df, sem)
        if smooth:   sem = self._smooth(pd.Series(sem, index=feat_df.index)).values
        labels = pd.Series(sem, index=feat_df.index, name="regime")
        cols   = [f"prob_{REGIME_NAMES[i]}" for i in range(self.n_regimes)]
        proba_df = pd.DataFrame(proba, index=feat_df.index, columns=cols)
        proba_df["regime"]      = labels
        proba_df["regime_name"] = labels.map(REGIME_NAMES)
        return labels, proba_df

    def _override(self, feat_df, sem):
        """Rule-based override for extreme conditions."""
        sem = sem.copy()
        vr  = feat_df.get("vol_ratio_10_30", pd.Series(np.ones(len(feat_df)), index=feat_df.index)).values
        vl  = feat_df.get("vol_real_20",     pd.Series(np.full(len(feat_df),.3), index=feat_df.index)).values
        sl  = feat_df.get("ema_50_slope",    pd.Series(np.zeros(len(feat_df)), index=feat_df.index)).values
        rs  = feat_df.get("rsi_14",          pd.Series(np.full(len(feat_df),50), index=feat_df.index)).values
        bw  = feat_df.get("bb_width",        pd.Series(np.full(len(feat_df),3), index=feat_df.index)).values
        vl_thresh = np.nanpercentile(vl, 85)
        bw_low    = np.nanpercentile(bw, 15)
        for i in range(len(sem)):
            if   vr[i] > 1.8 or vl[i] > vl_thresh:              sem[i] = 3
            elif bw[i] < bw_low and abs(sl[i]) < 0.1:            sem[i] = 4
            elif sl[i] >  0.3  and rs[i] > 60:                   sem[i] = 0
            elif sl[i] < -0.3  and rs[i] < 40:                   sem[i] = 1
        return sem

    def _smooth(self, labels: pd.Series) -> pd.Series:
        """Mode filter to eliminate noisy regime flips."""
        out = labels.copy()
        w   = self.smooth_window
        for i in range(w, len(labels)):
            window = labels.iloc[i-w+1:i+1]
            mode_result = window.mode()
            if len(mode_result) > 0:
                out.iloc[i] = mode_result.iloc[0]
        return out

    # ── UTILS ─────────────────────────────────────────────────────────────────
    def _prep(self, feat_df, fit=False):
        avail = [c for c in REGIME_FEATURES if c in feat_df.columns]
        if fit: self._feat_cols = avail
        X = feat_df[self._feat_cols].copy().replace([np.inf,-np.inf],np.nan).ffill().bfill().fillna(0)
        return X.values

    def get_regime_stats(self, feat_df):
        labels, _ = self.predict(feat_df)
        rows = []
        for rid, name in REGIME_NAMES.items():
            mask = labels==rid
            if not mask.any(): continue
            sub = feat_df[mask]
            rows.append({"regime":rid,"name":name,"count":mask.sum(),"pct":mask.mean()*100,
                         "avg_vol":sub.get("vol_real_20",pd.Series([np.nan])).mean(),
                         "avg_slope":sub.get("ema_50_slope",pd.Series([np.nan])).mean()})
        return pd.DataFrame(rows).set_index("regime")

    def regime_performance(self, feat_df, returns):
        labels, _ = self.predict(feat_df)
        rows = []
        for rid, name in REGIME_NAMES.items():
            mask = labels==rid
            if not mask.any(): continue
            r = returns[mask]
            rows.append({"regime":name,"n_bars":mask.sum(),
                         "mean_ret":r.mean()*100,
                         "sharpe":r.mean()/(r.std()+1e-10)*np.sqrt(2190),
                         "win_rate":(r>0).mean()*100})
        return pd.DataFrame(rows)

    def save(self, path="models/regime_model.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path,"wb") as f: pickle.dump(self,f)
        log.info("Regime model saved → %s", path)

    @classmethod
    def load(cls, path="models/regime_model.pkl"):
        with open(path,"rb") as f: m = pickle.load(f)
        log.info("Regime model loaded ← %s", path)
        return m


def run(feat_df: pd.DataFrame) -> dict:
    model = RegimeDetectionModel()
    model.fit(feat_df)
    labels, proba_df = model.predict(feat_df)
    log.info("Regime distribution:\n%s", labels.value_counts().rename(REGIME_NAMES).to_string())
    return {"model":model,"labels":labels,"proba_df":proba_df,"passed":True}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    np.random.seed(42)
    n=3000; dates=pd.date_range("2019-01-01",periods=n,freq="4h")
    feat=pd.DataFrame({"ema_50_slope":np.random.randn(n)*0.2,
                        "ema_20_slope":np.random.randn(n)*0.3,
                        "price_vs_ema200":np.random.randn(n)*5,
                        "vol_real_20":np.abs(np.random.randn(n)*0.5+0.5),
                        "vol_ratio_10_30":np.abs(np.random.randn(n)*0.3+1),
                        "rsi_14":np.random.uniform(20,80,n),
                        "adx":np.abs(np.random.randn(n)*15+25),
                        "bb_width":np.abs(np.random.randn(n)+3),
                        "ret_z20":np.random.randn(n),
                        "macd_hist":np.random.randn(n)*100},index=dates)
    model=RegimeDetectionModel(); model.fit(feat)
    labels,proba_df=model.predict(feat)
    print("\n[OK] Regime Distribution:")
    print(labels.value_counts().rename(REGIME_NAMES))
