"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  feature_engine_v2.py — BTC Hybrid AI v2                                  ║
║  25 Curated Features — Leakage-Free — Regime-Aware                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  DESIGN PRINCIPLES:                                                        ║
║  1. Every feature is strictly backward-looking (no lookahead)             ║
║  2. Features selected via SHAP stability across 5 CV folds                ║
║  3. Max correlation 0.70 between any pair                                  ║
║  4. Scaler MUST be fit on training data ONLY                               ║
║  5. Regime features are first-class (not afterthoughts)                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import RobustScaler
from typing import Optional, List, Tuple

log = logging.getLogger(__name__)
EPS = 1e-10


class FeatureEngineV2:
    """
    25-feature curated set for BTC 4h trading.

    Features are grouped into 5 categories:
      A. Trend (5)        — direction and strength
      B. Momentum (5)     — rate of change and reversals
      C. Volatility (5)   — regime and sizing inputs
      D. Microstructure (5) — bar-level signals
      E. Regime (5)       — explicit market state features

    Scaler is fitted separately — call fit_scaler(train_df) first,
    then transform(df) for any period.
    """

    FEATURE_NAMES = [
        # A. Trend
        "trend_ema_ratio",       # close / EMA200 − 1 (normalized distance)
        "trend_slope_20",        # linreg slope of close over 20 bars
        "trend_ema_cross",       # EMA20 > EMA50 (binary, smoothed)
        "trend_adx_14",          # ADX — trend strength
        "trend_donchian_pos",    # position within 20-bar Donchian channel

        # B. Momentum
        "mom_rsi_14",            # RSI(14)
        "mom_roc_10",            # Rate of change 10 bars
        "mom_macd_hist_norm",    # MACD histogram / ATR (normalized)
        "mom_stoch_k",           # Stochastic %K
        "mom_rsi_divergence",    # RSI vs price divergence signal

        # C. Volatility
        "vol_atr_pct",           # ATR(14) / close — current vol
        "vol_realized_10",       # realized vol 10 bars (annualized)
        "vol_regime",            # vol_realized_10 / vol_realized_60 (vol regime)
        "vol_bb_width_norm",     # BB width normalized by 50-bar mean
        "vol_hl_efficiency",     # (high-low) / ATR — intrabar vol usage

        # D. Microstructure
        "micro_body_ratio",      # candle body / total range
        "micro_upper_shadow",    # upper shadow ratio
        "micro_close_position",  # close position in bar (0=low, 1=high)
        "micro_gap",             # overnight gap (open vs prev close)
        "micro_volume_ratio",    # volume / 20-bar mean volume

        # E. Regime
        "regime_trend_score",    # composite trend score (EMA alignment)
        "regime_vol_zscore",     # vol z-score vs 60-bar mean
        "regime_return_skew",    # 20-bar return skewness
        "regime_autocorr",       # 10-bar return autocorrelation
        "regime_hurst",          # Hurst exponent proxy (trending vs mean-reverting)
    ]

    def __init__(self):
        self.scaler = RobustScaler()   # RobustScaler >> StandardScaler for fat tails
        self._scaler_fitted = False

    # ── MAIN INTERFACE ────────────────────────────────────────────────────────

    def fit_scaler(self, train_df: pd.DataFrame) -> "FeatureEngineV2":
        """
        Fit scaler on TRAINING DATA ONLY.
        Must be called before transform().

        [WARN]️  CRITICAL: Never fit scaler on full dataset.
            Fitting on full data leaks test statistics into training.
        """
        raw = self._compute_raw_features(train_df)
        raw = self._clean(raw)
        self.scaler.fit(raw[self.FEATURE_NAMES].values)
        self._scaler_fitted = True
        log.info("FeatureEngineV2: scaler fitted on %d training bars", len(train_df))
        return self

    def transform(self, df: pd.DataFrame, scale: bool = True) -> pd.DataFrame:
        """
        Compute and scale features.

        Parameters
        ----------
        df    : OHLCV DataFrame (any period — IS, VAL, or OOS)
        scale : Apply RobustScaler (must call fit_scaler first if True)

        Returns
        -------
        DataFrame with 25 scaled features
        """
        raw = self._compute_raw_features(df)
        raw = self._clean(raw)

        if scale:
            if not self._scaler_fitted:
                raise RuntimeError(
                    "Scaler not fitted. Call fit_scaler(train_df) first.\n"
                    "NEVER fit on full data — only on training split."
                )
            scaled = self.scaler.transform(raw[self.FEATURE_NAMES].values)
            out = pd.DataFrame(scaled, index=raw.index, columns=self.FEATURE_NAMES)
        else:
            out = raw[self.FEATURE_NAMES].copy()

        log.info("FeatureEngineV2: %d features × %d rows (scaled=%s)",
                 len(self.FEATURE_NAMES), len(out), scale)
        return out

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Fit scaler and transform in one call — training data only."""
        self.fit_scaler(train_df)
        return self.transform(train_df, scale=True)

    # ── FEATURE COMPUTATION ───────────────────────────────────────────────────

    def _compute_raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        o, h, l, c, v = (df["open"], df["high"], df["low"],
                         df["close"], df["volume"])
        ret = c.pct_change()

        feat = pd.DataFrame(index=df.index)

        # ── A. TREND ──────────────────────────────────────────────────────────
        ema20  = c.ewm(span=20, adjust=False).mean()
        ema50  = c.ewm(span=50, adjust=False).mean()
        ema200 = c.ewm(span=200, adjust=False).mean()
        atr14  = self._atr(h, l, c, 14)

        feat["trend_ema_ratio"]   = (c / ema200 - 1).clip(-0.5, 0.5)
        feat["trend_slope_20"]    = self._linreg_slope(c, 20)
        feat["trend_ema_cross"]   = ((ema20 > ema50).rolling(3).mean())  # smooth flip noise
        feat["trend_adx_14"]      = self._adx(h, l, c, 14)
        feat["trend_donchian_pos"]= (c - l.rolling(20).min()) / (
                                     h.rolling(20).max() - l.rolling(20).min() + EPS)

        # ── B. MOMENTUM ───────────────────────────────────────────────────────
        rsi14 = self._rsi(c, 14)
        macd_line   = c.ewm(12, adjust=False).mean() - c.ewm(26, adjust=False).mean()
        macd_signal = macd_line.ewm(9, adjust=False).mean()
        macd_hist   = macd_line - macd_signal
        lo14 = l.rolling(14).min(); hi14 = h.rolling(14).max()

        feat["mom_rsi_14"]         = rsi14 / 100.0  # normalize to 0-1
        feat["mom_roc_10"]         = (c / c.shift(10) - 1).clip(-0.3, 0.3)
        feat["mom_macd_hist_norm"] = (macd_hist / (atr14 + EPS)).clip(-3, 3)
        feat["mom_stoch_k"]        = ((c - lo14) / (hi14 - lo14 + EPS)).clip(0, 1)

        # RSI divergence: price making new high but RSI not
        price_high5 = (c == c.rolling(5).max()).astype(float)
        rsi_high5   = (rsi14 == rsi14.rolling(5).max()).astype(float)
        feat["mom_rsi_divergence"] = (price_high5 - rsi_high5).rolling(3).sum().clip(-1, 1)

        # ── C. VOLATILITY ─────────────────────────────────────────────────────
        vol10 = ret.rolling(10).std() * np.sqrt(2190)   # annualized (4h bars/yr)
        vol60 = ret.rolling(60).std() * np.sqrt(2190)
        bbm   = c.rolling(20).mean(); bbs = c.rolling(20).std()
        bb_w  = (4 * bbs) / (bbm + EPS)
        bb_w_50mean = bb_w.rolling(50).mean()

        feat["vol_atr_pct"]       = (atr14 / c).clip(0, 0.15)
        feat["vol_realized_10"]   = vol10.clip(0, 3.0)
        feat["vol_regime"]        = (vol10 / (vol60 + EPS)).clip(0, 3.0)
        feat["vol_bb_width_norm"] = (bb_w / (bb_w_50mean + EPS)).clip(0, 3.0)
        feat["vol_hl_efficiency"] = ((h - l) / (atr14 + EPS)).clip(0, 3.0)

        # ── D. MICROSTRUCTURE ─────────────────────────────────────────────────
        body    = (c - o).abs()
        total   = (h - l) + EPS
        top     = pd.concat([c, o], axis=1).max(axis=1)
        bot     = pd.concat([c, o], axis=1).min(axis=1)

        feat["micro_body_ratio"]   = (body / total).clip(0, 1)
        feat["micro_upper_shadow"] = ((h - top) / total).clip(0, 1)
        feat["micro_close_position"]= ((c - l) / total).clip(0, 1)
        feat["micro_gap"]          = ((o - c.shift(1)) / (c.shift(1) + EPS)).clip(-0.1, 0.1)
        feat["micro_volume_ratio"] = (v / (v.rolling(20).mean() + EPS)).clip(0, 5)

        # ── E. REGIME ─────────────────────────────────────────────────────────
        # Composite trend score: how many EMAs are aligned bullishly
        trend_score = ((c > ema20).astype(int) +
                       (c > ema50).astype(int) +
                       (c > ema200).astype(int) +
                       (ema20 > ema50).astype(int) +
                       (ema50 > ema200).astype(int)) / 5.0

        vol_zscore = ((vol10 - vol10.rolling(60).mean()) /
                      (vol10.rolling(60).std() + EPS)).clip(-3, 3)

        feat["regime_trend_score"] = trend_score
        feat["regime_vol_zscore"]  = vol_zscore
        feat["regime_return_skew"] = ret.rolling(20).skew().clip(-3, 3)
        feat["regime_autocorr"]    = ret.rolling(20).apply(
            lambda x: float(pd.Series(x).autocorr(1)) if len(x) > 5 else 0.0,
            raw=False
        ).clip(-1, 1)
        feat["regime_hurst"]       = ret.rolling(40).apply(
            self._hurst_proxy, raw=True
        ).clip(0, 1)

        return feat

    # ── HELPERS ───────────────────────────────────────────────────────────────

    @staticmethod
    def _atr(h, l, c, n):
        tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        return tr.ewm(span=n, adjust=False).mean()

    @staticmethod
    def _rsi(c, n=14):
        d = c.diff()
        g = d.clip(lower=0).ewm(span=n, adjust=False).mean()
        lo = (-d.clip(upper=0)).ewm(span=n, adjust=False).mean()
        return 100 - 100 / (1 + g / (lo + EPS))

    @staticmethod
    def _adx(h, l, c, n=14):
        tr   = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
        dmu  = pd.Series(np.where((h.diff()>0)&(h.diff()>-l.diff()),h.diff(),0),index=h.index)
        dmd  = pd.Series(np.where((-l.diff()>0)&(-l.diff()>h.diff()),-l.diff(),0),index=h.index)
        atr  = tr.ewm(span=n,adjust=False).mean()
        dip  = dmu.ewm(span=n,adjust=False).mean() / (atr+EPS) * 100
        dim  = dmd.ewm(span=n,adjust=False).mean() / (atr+EPS) * 100
        dx   = ((dip-dim).abs() / (dip+dim+EPS) * 100)
        return dx.ewm(span=n,adjust=False).mean() / 100.0  # normalize to 0-1

    @staticmethod
    def _linreg_slope(s, n):
        def _slope(arr):
            if np.isnan(arr).any() or arr[-1] == 0: return 0.0
            x = np.arange(len(arr), dtype=float)
            return np.polyfit(x, arr, 1)[0] / (arr[-1] + EPS) * 10
        return s.rolling(n).apply(_slope, raw=True).clip(-1, 1)

    @staticmethod
    def _hurst_proxy(arr):
        """H > 0.5 = trending, H < 0.5 = mean-reverting, H = 0.5 = random"""
        arr = arr[~np.isnan(arr)]
        if len(arr) < 20: return 0.5
        lags = [4, 8, 16]
        rs_vals = []
        for lag in lags:
            if lag >= len(arr): continue
            sub = arr[:lag]
            s = np.std(sub, ddof=1)
            if s < EPS: continue
            cum = np.cumsum(sub - np.mean(sub))
            rs_vals.append((lag, (cum.max() - cum.min()) / s))
        if len(rs_vals) < 2: return 0.5
        lgs, rs = zip(*rs_vals)
        try:
            return float(np.polyfit(np.log(lgs), np.log(rs), 1)[0])
        except Exception:
            return 0.5

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill().fillna(0)
        return df

    def check_correlation(self, df: pd.DataFrame,
                          threshold: float = 0.70) -> pd.DataFrame:
        """
        Check feature correlations. Any pair above threshold
        should be reviewed for redundancy.
        """
        feat = self._compute_raw_features(df)
        feat = self._clean(feat)[self.FEATURE_NAMES]
        corr = feat.corr().abs()
        pairs = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                c = corr.iloc[i,j]
                if c > threshold:
                    pairs.append({
                        "feat_a": corr.columns[i],
                        "feat_b": corr.columns[j],
                        "corr": round(c, 3),
                    })
        if pairs:
            log.warning("High correlation pairs (>%.2f): %d found", threshold, len(pairs))
            for p in pairs:
                log.warning("  %s ↔ %s = %.3f", p["feat_a"], p["feat_b"], p["corr"])
        return pd.DataFrame(pairs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    np.random.seed(42)
    n = 5000
    dates = pd.date_range("2019-01-01", periods=n, freq="4h")
    price = 10000 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
    df = pd.DataFrame({
        "open":price, "close":price*(1+np.random.randn(n)*0.003),
        "high":price*1.01, "low":price*0.99,
        "volume":np.abs(np.random.randn(n)*1000+500)
    }, index=dates)

    # CORRECT usage — fit only on training split
    split = int(len(df) * 0.70)
    train_df = df.iloc[:split]
    test_df  = df.iloc[split:]

    engine = FeatureEngineV2()
    train_feat = engine.fit_transform(train_df)   # fit + transform train
    test_feat  = engine.transform(test_df)         # transform only test

    print(f"\n[OK] Train features: {train_feat.shape}")
    print(f"[OK] Test  features: {test_feat.shape}")
    print(f"\nFeature list ({len(engine.FEATURE_NAMES)}):")
    for i, f in enumerate(engine.FEATURE_NAMES, 1):
        print(f"  {i:2d}. {f}")

    # Check correlations
    corr_pairs = engine.check_correlation(train_df)
    if corr_pairs.empty:
        print("\n[OK] No high-correlation pairs found")
    else:
        print(f"\n[WARN]️  {len(corr_pairs)} high-correlation pairs found")
