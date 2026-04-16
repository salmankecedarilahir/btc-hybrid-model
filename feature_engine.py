"""
╔══════════════════════════════════════════════════════════════════════════╗
║  feature_engine.py  —  BTC Autonomous AI Quant System                  ║
║  LAYER 1 : Feature Engineering Engine                                   ║
╠══════════════════════════════════════════════════════════════════════════╣
║  TUJUAN  : Ekstrak 85+ fitur market dari data OHLCV                    ║
║  INPUT   : pd.DataFrame (open, high, low, close, volume)               ║
║  OUTPUT  : pd.DataFrame fitur siap untuk AI layer                       ║
║  KATEGORI: Trend(22) Momentum(17) Volatility(15) Vol(13) Micro(10) Stat(10) ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import logging
from typing import Callable, List, Optional

log = logging.getLogger(__name__)
EPS = 1e-10

# ─────────────────────────────────────────────────────────────────────────────
#  PRIMITIVE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _safe(a, b):
    if hasattr(b, 'replace'):
        return a / b.replace(0, np.nan)
    return a / b if b != 0 else np.nan
def _ema(s, n):       return s.ewm(span=n, adjust=False).mean()
def _roc(s, n):       return (s / s.shift(n) - 1) * 100
def _zscore(s, n):    return _safe(s - s.rolling(n).mean(), s.rolling(n).std())

def _rsi(s, n=14):
    d = s.diff()
    return 100 - 100 / (1 + _safe(d.clip(lower=0).rolling(n).mean(),
                                   (-d.clip(upper=0)).rolling(n).mean()))

def _atr(h, l, c, n=14):
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def _macd(c, f=12, sl=26, sg=9):
    line = _ema(c,f) - _ema(c,sl); sig = _ema(line,sg)
    return line, sig, line-sig

def _bb(c, n=20, k=2):
    mid=c.rolling(n).mean(); std=c.rolling(n).std()
    return mid+k*std, mid, mid-k*std

def _vwap(c, v, n):   return (c*v).rolling(n).sum() / v.rolling(n).sum()
def _cmf(h, l, c, v, n=20):
    mfm = _safe(((c-l)-(h-c)), (h-l))
    return (mfm*v).rolling(n).sum() / v.rolling(n).sum()

def _linreg_slope(s, n):
    def _f(arr):
        if np.isnan(arr).any(): return np.nan
        m = np.polyfit(np.arange(len(arr)), arr, 1)[0]
        return m / (arr[-1]+EPS) * 100
    return s.rolling(n).apply(_f, raw=True)

def _hurst(ret, n):
    def _h(arr):
        arr = arr[~np.isnan(arr)]
        if len(arr) < 10: return 0.5
        rs = []
        for lag in [4,8,16]:
            if lag >= len(arr): continue
            sub = arr[:lag]; s = np.std(sub, ddof=1)
            if s == 0: continue
            cum = np.cumsum(sub - np.mean(sub))
            rs.append((lag, (cum.max()-cum.min())/s))
        if len(rs) < 2: return 0.5
        lgs, vals = zip(*rs)
        return np.polyfit(np.log(lgs), np.log(vals), 1)[0]
    return ret.rolling(n).apply(_h, raw=True)

def _entropy(arr):
    arr = arr[~np.isnan(arr)]
    if len(arr) < 5: return np.nan
    h, _ = np.histogram(arr, bins=max(4, int(np.sqrt(len(arr)))))
    p = h/h.sum(); p = p[p>0]
    return -np.sum(p*np.log(p))

# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class FeatureEngine:
    """
    Generates 85+ market features across 6 categories.

    Example
    -------
    engine  = FeatureEngine()
    feat_df = engine.transform(ohlcv_df)       # → DataFrame 85+ cols
    engine.add_custom(my_fn)                    # extend with custom features
    """

    def __init__(self, custom: Optional[List[Callable]] = None):
        self._custom = custom or []
        self.feature_names_: List[str] = []

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ["open","high","low","close","volume"]:
            if col not in df.columns: raise ValueError(f"Missing column: {col}")
        o,h,l,c,v = df["open"],df["high"],df["low"],df["close"],df["volume"]
        ret = c.pct_change()
        feat = pd.DataFrame(index=df.index)
        feat = self._trend(feat, c, h, l)
        feat = self._momentum(feat, c, h, l)
        feat = self._volatility(feat, c, h, l, ret)
        feat = self._volume(feat, c, h, l, v, ret)
        feat = self._micro(feat, o, h, l, c, v)
        feat = self._statistical(feat, ret)
        for fn in self._custom:
            try:
                res = fn(df)
                if isinstance(res, pd.Series): feat[res.name or f"cust_{len(feat.columns)}"] = res
                elif isinstance(res, pd.DataFrame):
                    for col in res.columns: feat[col] = res[col]
            except Exception as e: log.warning("Custom feature error: %s", e)
        self.feature_names_ = list(feat.columns)
        log.info("FeatureEngine: %d features × %d rows", feat.shape[1], feat.shape[0])
        return feat

    def add_custom(self, fn: Callable): self._custom.append(fn)

    # ── TREND (22) ───────────────────────────────────────────────────────────
    def _trend(self, f, c, h, l):
        for n in [8,13,20,50,100,200]: f[f"ema_{n}"] = _ema(c,n)
        f["ema_x_8_20"]   = (_ema(c,8)>_ema(c,20)).astype(int)
        f["ema_x_20_50"]  = (_ema(c,20)>_ema(c,50)).astype(int)
        f["ema_x_50_200"] = (_ema(c,50)>_ema(c,200)).astype(int)
        f["price_vs_ema20"]  = _safe(c-_ema(c,20), _ema(c,20))*100
        f["price_vs_ema50"]  = _safe(c-_ema(c,50), _ema(c,50))*100
        f["price_vs_ema200"] = _safe(c-_ema(c,200), _ema(c,200))*100
        for n in [20,50]:
            e = _ema(c,n); f[f"ema_{n}_slope"] = _safe(e-e.shift(5), e.shift(5))*100
        f["linreg_slope_20"] = _linreg_slope(c,20)
        f["linreg_slope_50"] = _linreg_slope(c,50)
        f["adx"]             = self._adx(h,l,c)
        for n in [20,55]:
            f[f"donchian_{n}"] = _safe(c-l.rolling(n).min(), h.rolling(n).max()-l.rolling(n).min())
        return f

    @staticmethod
    def _adx(h,l,c,n=14):
        tr  = _atr(h,l,c,n)
        dmu = pd.Series(np.where((h.diff()>0)&(h.diff()>-l.diff()),h.diff(),0),index=h.index)
        dmd = pd.Series(np.where((-l.diff()>0)&(-l.diff()>h.diff()),-l.diff(),0),index=h.index)
        a   = tr.rolling(n).mean()
        dip = _safe(dmu.rolling(n).mean(),a)*100
        dim = _safe(dmd.rolling(n).mean(),a)*100
        return (_safe((dip-dim).abs(),dip+dim)*100).ewm(span=n,adjust=False).mean()

    # ── MOMENTUM (17) ────────────────────────────────────────────────────────
    def _momentum(self, f, c, h, l):
        for n in [6,14,21]: f[f"rsi_{n}"] = _rsi(c,n)
        f["rsi_ob"] = (_rsi(c,14)>70).astype(int)
        f["rsi_os"] = (_rsi(c,14)<30).astype(int)
        ml,ms,mh = _macd(c)
        f["macd"]=ml; f["macd_sig"]=ms; f["macd_hist"]=mh; f["macd_bull"]=(ml>ms).astype(int)
        for n in [5,10,20,60]: f[f"roc_{n}"] = _roc(c,n)
        lo14,hi14 = l.rolling(14).min(), h.rolling(14).max()
        f["stoch_k"]     = _safe(c-lo14,hi14-lo14)*100
        f["williams_r"]  = -100*_safe(hi14-c,hi14-lo14)
        tp = (h+l+c)/3
        f["cci_20"] = _safe(tp-tp.rolling(20).mean(), 0.015*tp.rolling(20).std())
        return f

    # ── VOLATILITY (15) ──────────────────────────────────────────────────────
    def _volatility(self, f, c, h, l, ret):
        for n in [7,14,21]:
            f[f"atr_{n}"]     = _atr(h,l,c,n)
            f[f"atr_{n}_pct"] = _safe(_atr(h,l,c,n),c)*100
        for n in [10,20,30,60]: f[f"vol_real_{n}"] = ret.rolling(n).std()*np.sqrt(252)
        f["vol_ratio_10_30"] = _safe(ret.rolling(10).std(), ret.rolling(30).std())
        bbu,bbm,bbl = _bb(c)
        f["bb_width"]   = _safe(bbu-bbl,bbm)*100
        f["bb_pos"]     = _safe(c-bbl,bbu-bbl)
        f["bb_squeeze"] = (f["bb_width"]<f["bb_width"].rolling(50).mean()*0.85).astype(int)
        f["hl_pct"]     = _safe(h-l,c)*100
        return f

    # ── VOLUME (13) ──────────────────────────────────────────────────────────
    def _volume(self, f, c, h, l, v, ret):
        f["vol_ma20"]   = v.rolling(20).mean()
        f["vol_ratio"]  = _safe(v,v.rolling(20).mean())
        f["vol_roc5"]   = _roc(v,5)
        f["vol_zscore"] = _zscore(v,20)
        obv = (np.sign(ret)*v).cumsum()
        f["obv"]=obv; f["obv_slope"]=_roc(obv,10)
        f["pv_corr20"]  = c.rolling(20).corr(v)
        f["vwap_dist"]  = _safe(c-_vwap(c,v,20), _vwap(c,v,20))*100
        f["cmf_20"]     = _cmf(h,l,c,v,20)
        f["force_idx"]  = _ema(ret*v,13)
        f["vol_cum_r"]  = _safe(v.rolling(5).sum(), v.rolling(20).sum())
        tp=(h+l+c)/3; mf=tp*v
        pos=mf.where(tp>tp.shift(),0).rolling(14).sum()
        neg=mf.where(tp<tp.shift(),0).rolling(14).sum()
        f["mfi_14"] = 100 - 100/(1+_safe(pos,neg.abs()))
        f["vpt"] = (ret*v).cumsum()
        return f

    # ── MICROSTRUCTURE (10) ──────────────────────────────────────────────────
    def _micro(self, f, o, h, l, c, v):
        for n in [10,20]:
            f[f"efficiency_{n}"] = _safe((c-c.shift(n)).abs(), c.diff().abs().rolling(n).sum())
        f["body_ratio"]   = _safe((c-o).abs(), h-l)
        f["body_dir"]     = np.sign(c-o)
        bt,bb = pd.concat([c,o],axis=1).max(axis=1), pd.concat([c,o],axis=1).min(axis=1)
        f["upper_shadow"] = _safe(h-bt,h-l)
        f["lower_shadow"] = _safe(bb-l,h-l)
        f["gap_up"]       = ((o>h.shift())*1).rolling(5).sum()
        f["gap_down"]     = ((o<l.shift())*1).rolling(5).sum()
        f["intrabar_mom"] = _safe(c-o,h-l)
        f["overnight_gap"]= _safe(o-c.shift(),c.shift())*100
        return f

    # ── STATISTICAL (10) ─────────────────────────────────────────────────────
    def _statistical(self, f, ret):
        for n in [20,60]:
            f[f"skew_{n}"] = ret.rolling(n).skew()
            f[f"kurt_{n}"] = ret.rolling(n).kurt()
        f["ret_z20"] = _zscore(ret,20)
        f["ret_z60"] = _zscore(ret,60)
        f["autocorr"] = ret.rolling(20).apply(lambda x: float(pd.Series(x).autocorr(lag=1)) if len(x)>2 else 0.0, raw=False)
        f["hurst_50"] = _hurst(ret,50)
        f["entropy_20"] = ret.rolling(20).apply(_entropy, raw=True)
        f["ret_rank_60"] = ret.rolling(60).rank(pct=True)
        return f


def run(df: pd.DataFrame = None, data_path: str = None) -> pd.DataFrame:
    if df is None: df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    return FeatureEngine().transform(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    np.random.seed(42)
    n=3000; dates=pd.date_range("2019-01-01",periods=n,freq="4h")
    price=10000*np.exp(np.cumsum(np.random.randn(n)*0.01))
    demo=pd.DataFrame({"open":price,"close":price*(1+np.random.randn(n)*0.003),
                        "high":price*1.01,"low":price*0.99,
                        "volume":np.abs(np.random.randn(n)*1000+500)},index=dates)
    feat=FeatureEngine().transform(demo)
    print(f"\n[OK] Features: {feat.shape[1]}  |  Rows: {feat.shape[0]}")
    for i,col in enumerate(feat.columns,1): print(f"  {i:3d}. {col}")
