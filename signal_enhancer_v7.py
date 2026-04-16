"""
╔══════════════════════════════════════════════════════════════════════════════╗
║            signal_enhancer_v7.py  —  BTC Hybrid Model V7                   ║
║            Signal Quality Enhancement Layer                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  POSISI DALAM PIPELINE:                                                      ║
║    hybrid_engine.py  →  [signal_enhancer_v7.py]  →  backtest/live          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  5 FILTER MODULES (setiap filter memberi score 0-20, total 0-100):          ║
║                                                                              ║
║  1. Volatility Filter     (ATR + Realized Vol)         max 20 pts           ║
║  2. Trend Strength Filter (EMA slope + ADX)            max 20 pts           ║
║  3. Liquidity Filter      (Volume vs average)          max 20 pts           ║
║  4. Momentum Confirmation (RSI + MACD + ROC)           max 20 pts           ║
║  5. Market Structure      (BB position + OBV)          max 20 pts           ║
║                                                                              ║
║  Score → Trade Decision:                                                     ║
║    80-100  EXCELLENT  → leverage boost × 1.20                               ║
║    60-79   GOOD       → leverage normal × 1.00                              ║
║    40-59   FAIR       → leverage reduce × 0.80                              ║
║    20-39   WEAK       → leverage reduce × 0.60                              ║
║     0-19   SKIP       → signal ditolak                                      ║
║                                                                              ║
║  OUTPUT COLUMNS BARU (ditambahkan ke CSV):                                  ║
║    f1_volatility  f2_trend  f3_liquidity  f4_momentum  f5_structure         ║
║    signal_score  signal_quality  lev_mult  rsi_14  macd_line  macd_signal   ║
║    macd_hist  bb_upper  bb_mid  bb_lower  bb_width  bb_position             ║
║    obv  obv_ema20  adx  stoch_k  stoch_d  atr_zscore                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

CARA PAKAI:
  python signal_enhancer_v7.py                  # process default signals file
  python signal_enhancer_v7.py --threshold 50   # custom score threshold
  python signal_enhancer_v7.py --report         # tampilkan detail report
  python signal_enhancer_v7.py --plot           # save chart ke data/

INTEGRASI:
  import signal_enhancer_v7 as se
  df = se.run(df)                                # tambah semua kolom ke DataFrame
  threshold = se.DEFAULT_THRESHOLD               # score minimum = 40
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ─── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── paths ────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
SIGNALS_PATH   = BASE_DIR / "data" / "btc_trading_signals.csv"
OUTPUT_PATH    = BASE_DIR / "data" / "btc_trading_signals.csv"   # overwrite in-place
CHART_DIR      = BASE_DIR / "data"

# ─── thresholds ───────────────────────────────────────────────────────────────
DEFAULT_THRESHOLD = 40        # signal diterima jika score >= 40
SCORE_MAX         = 100
BARS_PER_YEAR     = 2190      # 4H timeframe


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def _ema(series: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(series).ewm(span=span, adjust=False).mean().values


def _sma(series: np.ndarray, period: int) -> np.ndarray:
    return pd.Series(series).rolling(period, min_periods=1).mean().values


def _std(series: np.ndarray, period: int) -> np.ndarray:
    return pd.Series(series).rolling(period, min_periods=2).std().values


def _rank(x: float, low: float, high: float) -> float:
    """Normalize x ke 0-1 range [low, high]."""
    if high == low:
        return 0.5
    return float(np.clip((x - low) / (high - low), 0.0, 1.0))


# ══════════════════════════════════════════════════════════════════════════════
#  INDICATOR LIBRARY
# ══════════════════════════════════════════════════════════════════════════════

def calc_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI menggunakan Wilder's EMA (identik dengan TradingView)."""
    delta = np.diff(close, prepend=close[0])
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(com=period - 1, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(com=period - 1, min_periods=period, adjust=False).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_l != 0, avg_g / avg_l, 100.0)
    return np.clip(100.0 - 100.0 / (1.0 + rs), 0, 100)


def calc_macd(close: np.ndarray,
              fast: int = 12, slow: int = 26, sig: int = 9
              ) -> tuple:
    """MACD line, signal line, histogram."""
    macd_line = _ema(close, fast) - _ema(close, slow)
    macd_sig  = _ema(macd_line, sig)
    return macd_line, macd_sig, macd_line - macd_sig


def calc_bollinger(close: np.ndarray,
                   period: int = 20, mult: float = 2.0) -> tuple:
    """BB upper/mid/lower + normalized width + price position."""
    mid   = _sma(close, period)
    std   = _std(close, period)
    upper = mid + mult * std
    lower = mid - mult * std
    rang  = np.where((upper - lower) > 0, upper - lower, 1.0)
    width = rang / np.where(mid > 0, mid, 1.0)
    pos   = np.clip((close - lower) / rang, 0.0, 1.0)
    return upper, mid, lower, width, pos


def calc_obv(close: np.ndarray, volume: np.ndarray) -> tuple:
    """On-Balance Volume + EMA20."""
    obv = np.zeros(len(close))
    for i in range(1, len(close)):
        obv[i] = obv[i - 1] + (volume[i] if close[i] >= close[i - 1] else -volume[i])
    return obv, _ema(obv, 20)


def calc_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
             period: int = 14) -> tuple:
    """ADX + DI+ + DI-  (Wilder's method)."""
    n   = len(close)
    tr  = np.zeros(n)
    pdm = np.zeros(n)
    ndm = np.zeros(n)

    for i in range(1, n):
        hl  = high[i] - low[i]
        hpc = abs(high[i] - close[i - 1])
        lpc = abs(low[i]  - close[i - 1])
        tr[i] = max(hl, hpc, lpc)
        up   = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        pdm[i] = up   if (up > down and up > 0)   else 0.0
        ndm[i] = down if (down > up and down > 0) else 0.0

    atr14 = pd.Series(tr).ewm(com=period - 1, min_periods=period, adjust=False).mean().values
    pdi   = 100 * pd.Series(pdm).ewm(com=period-1, min_periods=period, adjust=False).mean().values / np.where(atr14 > 0, atr14, 1)
    ndi   = 100 * pd.Series(ndm).ewm(com=period-1, min_periods=period, adjust=False).mean().values / np.where(atr14 > 0, atr14, 1)
    dx    = 100 * np.abs(pdi - ndi) / np.where((pdi + ndi) > 0, pdi + ndi, 1)
    adx   = pd.Series(dx).ewm(com=period - 1, min_periods=period, adjust=False).mean().values
    return adx, pdi, ndi


def calc_stochrsi(close: np.ndarray,
                  rsi_p: int = 14, stoch_p: int = 14,
                  k_p: int = 3, d_p: int = 3) -> tuple:
    """Stochastic RSI."""
    rsi_s = pd.Series(calc_rsi(close, rsi_p))
    lo    = rsi_s.rolling(stoch_p, min_periods=1).min()
    hi    = rsi_s.rolling(stoch_p, min_periods=1).max()
    rang  = (hi - lo).replace(0, 1.0)
    k     = ((rsi_s - lo) / rang * 100).clip(0, 100)
    stoch_k = k.rolling(k_p, min_periods=1).mean().values
    stoch_d = pd.Series(stoch_k).rolling(d_p, min_periods=1).mean().values
    return stoch_k, stoch_d


def calc_roc(close: np.ndarray, period: int = 10) -> np.ndarray:
    """Rate of Change (%)."""
    shifted = np.roll(close, period)
    shifted[:period] = close[:period]
    with np.errstate(divide="ignore", invalid="ignore"):
        roc = np.where(shifted != 0, (close - shifted) / shifted * 100, 0.0)
    return roc


def calc_realized_vol(returns: np.ndarray, period: int = 20) -> np.ndarray:
    """Realized volatility (annualized)."""
    return pd.Series(returns).rolling(period, min_periods=5).std().values * np.sqrt(BARS_PER_YEAR)


def calc_atr_zscore(atr: np.ndarray, period: int = 100) -> np.ndarray:
    """ATR z-score: seberapa jauh dari rata-rata historis."""
    mu  = _sma(atr, period)
    sig = _std(atr, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sig > 0, (atr - mu) / sig, 0.0)


def calc_ema_slope(ema: np.ndarray, period: int = 5) -> np.ndarray:
    """EMA slope: perubahan rata-rata per bar (normalized by EMA value)."""
    slope = np.diff(ema, prepend=ema[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ema > 0, slope / ema, 0.0)


# ══════════════════════════════════════════════════════════════════════════════
#  FILTER 1 — VOLATILITY (max 20 pts)
# ══════════════════════════════════════════════════════════════════════════════

def filter_volatility(atr: np.ndarray,
                      atr_pct: np.ndarray,
                      close: np.ndarray,
                      signal: np.ndarray) -> np.ndarray:
    """
    Logika:
      - ATR percentile 20-65: OPTIMAL volatility zone → full score
      - ATR percentile 65-80: ELEVATED → partial score (masuk trade tapi caution)
      - ATR percentile > 80:  EXTREME  → low score (avoid new entries)
      - ATR percentile < 20:  TOO QUIET → low score (signal less reliable)

    Returns: score array [0-20]
    """
    scores = np.zeros(len(signal))
    for i in range(len(signal)):
        if signal[i] == "NONE":
            continue
        ap = float(atr_pct[i])
        if 20 <= ap <= 65:
            scores[i] = 20.0                          # optimal range
        elif 65 < ap <= 75:
            scores[i] = 15.0                          # elevated but ok
        elif 75 < ap <= 85:
            scores[i] = 8.0                           # high volatility caution
        elif ap > 85:
            scores[i] = 2.0                           # extreme vol — avoid
        else:                                          # < 20: too quiet
            scores[i] = 10.0                          # signal less reliable
    return scores


# ══════════════════════════════════════════════════════════════════════════════
#  FILTER 2 — TREND STRENGTH (max 20 pts)
# ══════════════════════════════════════════════════════════════════════════════

def filter_trend(close: np.ndarray,
                 ema20: np.ndarray,
                 ema50: np.ndarray,
                 adx: np.ndarray,
                 pdi: np.ndarray,
                 ndi: np.ndarray,
                 trend_score: np.ndarray,
                 signal: np.ndarray) -> np.ndarray:
    """
    Logika:
      LONG:
        + EMA20 > EMA50 (bullish alignment)
        + EMA20 slope positive
        + ADX > 25 (trend ada)
        + DI+ > DI- (directional confirmation)
        + trend_score >= 1 (hybrid engine confirmation)

      SHORT: mirror
    """
    slope20 = calc_ema_slope(ema20)
    scores  = np.zeros(len(signal))

    for i in range(len(signal)):
        sig = signal[i]
        if sig == "NONE":
            continue

        pts = 0.0
        if sig == "LONG":
            if ema20[i] > ema50[i]:                pts += 5.0
            if slope20[i] > 0:                     pts += 3.0
            if float(adx[i]) > 25:                 pts += 4.0
            if float(pdi[i]) > float(ndi[i]):      pts += 4.0
            if float(trend_score[i]) >= 1:          pts += 4.0
        elif sig == "SHORT":
            if ema20[i] < ema50[i]:                pts += 5.0
            if slope20[i] < 0:                     pts += 3.0
            if float(adx[i]) > 25:                 pts += 4.0
            if float(ndi[i]) > float(pdi[i]):      pts += 4.0
            if float(trend_score[i]) <= -1:         pts += 4.0

        scores[i] = min(pts, 20.0)
    return scores


# ══════════════════════════════════════════════════════════════════════════════
#  FILTER 3 — LIQUIDITY (max 20 pts)
# ══════════════════════════════════════════════════════════════════════════════

def filter_liquidity(volume: np.ndarray,
                     signal: np.ndarray,
                     vol_period: int = 30) -> np.ndarray:
    """
    Logika:
      - Volume relative to 30-bar average
      - Volume > 150% avg: HIGH liquidity → full score
      - Volume 80-150% avg: NORMAL → medium score
      - Volume < 80% avg:  LOW → low score (higher slippage risk)
      - Volume = 0: ILLIQUID → skip
    """
    vol_avg = _sma(volume, vol_period)
    scores  = np.zeros(len(signal))

    for i in range(len(signal)):
        if signal[i] == "NONE":
            continue
        avg = float(vol_avg[i])
        vol = float(volume[i])
        if avg <= 0 or vol <= 0:
            scores[i] = 0.0
            continue
        ratio = vol / avg
        if ratio >= 1.5:
            scores[i] = 20.0     # high liquidity
        elif ratio >= 1.0:
            scores[i] = 15.0     # normal
        elif ratio >= 0.8:
            scores[i] = 10.0     # slightly low
        elif ratio >= 0.5:
            scores[i] = 5.0      # low
        else:
            scores[i] = 0.0      # illiquid
    return scores


# ══════════════════════════════════════════════════════════════════════════════
#  FILTER 4 — MOMENTUM CONFIRMATION (max 20 pts)
# ══════════════════════════════════════════════════════════════════════════════

def filter_momentum(rsi14: np.ndarray,
                    macd_line: np.ndarray,
                    macd_sig: np.ndarray,
                    macd_hist: np.ndarray,
                    roc10: np.ndarray,
                    signal: np.ndarray) -> np.ndarray:
    """
    Logika per komponen:
      RSI (0-7pts):
        LONG: RSI 35-72 (tidak extreme) → 7pts; RSI 30-80 → 4pts; else 0pts
        SHORT: RSI 28-65 → 7pts; RSI 20-72 → 4pts; else 0pts

      MACD (0-7pts):
        LONG: MACD > Signal → 4pts; Histogram accelerating → 3pts
        SHORT: MACD < Signal → 4pts; Histogram decelerating → 3pts

      ROC (0-6pts):
        LONG: ROC > 0 → 3pts; ROC > 2% → 6pts
        SHORT: ROC < 0 → 3pts; ROC < -2% → 6pts
    """
    macd_accel = macd_hist > np.roll(macd_hist, 1)
    scores     = np.zeros(len(signal))

    for i in range(len(signal)):
        sig = signal[i]
        if sig == "NONE":
            continue

        pts = 0.0
        r   = float(rsi14[i])
        roc = float(roc10[i])

        if sig == "LONG":
            # RSI
            if 35 <= r <= 72:      pts += 7.0
            elif 30 <= r <= 80:    pts += 4.0
            # MACD
            if macd_line[i] > macd_sig[i]:  pts += 4.0
            if macd_accel[i]:               pts += 3.0
            # ROC
            if roc > 2.0:  pts += 6.0
            elif roc > 0:  pts += 3.0

        elif sig == "SHORT":
            # RSI
            if 28 <= r <= 65:      pts += 7.0
            elif 20 <= r <= 72:    pts += 4.0
            # MACD
            if macd_line[i] < macd_sig[i]:  pts += 4.0
            if not macd_accel[i]:           pts += 3.0
            # ROC
            if roc < -2.0: pts += 6.0
            elif roc < 0:  pts += 3.0

        scores[i] = min(pts, 20.0)
    return scores


# ══════════════════════════════════════════════════════════════════════════════
#  FILTER 5 — MARKET STRUCTURE (max 20 pts)
# ══════════════════════════════════════════════════════════════════════════════

def filter_structure(close: np.ndarray,
                     bb_mid: np.ndarray,
                     bb_pos: np.ndarray,
                     obv: np.ndarray,
                     obv_ema: np.ndarray,
                     signal: np.ndarray) -> np.ndarray:
    """
    Logika:
      Bollinger Position (0-10pts):
        LONG: price > BB mid → 5pts; BB pos > 0.6 (upper zone) → 10pts
        SHORT: price < BB mid → 5pts; BB pos < 0.4 (lower zone) → 10pts

      OBV (0-10pts):
        LONG: OBV > OBV_EMA → 5pts; OBV slope positive (3-bar) → 5pts
        SHORT: OBV < OBV_EMA → 5pts; OBV slope negative → 5pts
    """
    obv_slope = (obv - np.roll(obv, 3)) > 0
    scores    = np.zeros(len(signal))

    for i in range(len(signal)):
        sig = signal[i]
        if sig == "NONE":
            continue

        pts = 0.0
        bp  = float(bb_pos[i])

        if sig == "LONG":
            if close[i] > bb_mid[i]:   pts += 5.0
            if bp > 0.6:               pts += 5.0   # upper half bias
            if obv[i] > obv_ema[i]:    pts += 5.0
            if obv_slope[i]:           pts += 5.0
        elif sig == "SHORT":
            if close[i] < bb_mid[i]:   pts += 5.0
            if bp < 0.4:               pts += 5.0   # lower half bias
            if obv[i] < obv_ema[i]:    pts += 5.0
            if not obv_slope[i]:       pts += 5.0

        scores[i] = min(pts, 20.0)
    return scores


# ══════════════════════════════════════════════════════════════════════════════
#  SCORE → QUALITY LABEL + LEVERAGE MULTIPLIER
# ══════════════════════════════════════════════════════════════════════════════

def score_to_quality(score: float) -> str:
    if score >= 80:   return "EXCELLENT"
    elif score >= 60: return "GOOD"
    elif score >= 40: return "FAIR"
    elif score >= 20: return "WEAK"
    else:             return "SKIP"


def score_to_leverage_mult(score: float, signal: str) -> float:
    """
    Score menentukan leverage multiplier.
    Digunakan oleh risk engine untuk scale position size.
    """
    if signal == "NONE" or score < DEFAULT_THRESHOLD:
        return 0.0        # trade ditolak
    if score >= 80:   return 1.20    # EXCELLENT: boost 20%
    elif score >= 60: return 1.00    # GOOD: normal
    elif score >= 40: return 0.80    # FAIR: reduce 20%
    else:             return 0.60    # WEAK: reduce 40%


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN COMPUTE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators dan tambahkan ke DataFrame."""
    close  = df["close"].values
    high   = df["high"].values   if "high"   in df.columns else close
    low    = df["low"].values    if "low"    in df.columns else close
    volume = df["volume"].values if "volume" in df.columns else np.ones(len(close))

    log.info("Menghitung RSI(14)...")
    df["rsi_14"] = np.round(calc_rsi(close, 14), 4)

    log.info("Menghitung MACD(12,26,9)...")
    ml, ms, mh = calc_macd(close)
    df["macd_line"]   = np.round(ml, 6)
    df["macd_signal"] = np.round(ms, 6)
    df["macd_hist"]   = np.round(mh, 6)

    log.info("Menghitung Bollinger Bands(20,2)...")
    bu, bm, bl, bw, bp = calc_bollinger(close)
    df["bb_upper"]    = np.round(bu, 4)
    df["bb_mid"]      = np.round(bm, 4)
    df["bb_lower"]    = np.round(bl, 4)
    df["bb_width"]    = np.round(bw, 6)
    df["bb_position"] = np.round(bp, 6)

    log.info("Menghitung OBV...")
    obv, obv_ema = calc_obv(close, volume)
    df["obv"]      = np.round(obv, 2)
    df["obv_ema20"] = np.round(obv_ema, 2)

    log.info("Menghitung ADX(14)...")
    adx, pdi, ndi = calc_adx(high, low, close, 14)
    df["adx"]  = np.round(adx, 4)
    df["pdi"]  = np.round(pdi, 4)
    df["ndi"]  = np.round(ndi, 4)

    log.info("Menghitung StochRSI(14,14,3,3)...")
    sk, sd = calc_stochrsi(close)
    df["stoch_k"] = np.round(sk, 4)
    df["stoch_d"] = np.round(sd, 4)

    log.info("Menghitung ROC(10)...")
    df["roc_10"] = np.round(calc_roc(close, 10), 4)

    log.info("Menghitung ATR z-score...")
    atr = df["atr_14"].values if "atr_14" in df.columns else np.ones(len(close))
    df["atr_zscore"] = np.round(calc_atr_zscore(atr), 4)

    log.info("Menghitung Realized Vol...")
    mkt_ret = np.diff(close, prepend=close[0]) / np.where(np.concatenate([[close[0]], close[:-1]]) > 0, np.concatenate([[close[0]], close[:-1]]), 1)
    df["realized_vol_20"] = np.round(calc_realized_vol(mkt_ret, 20), 4)

    return df


def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Hitung semua 5 filter scores dan composite signal_score."""
    sig       = df["signal"].values
    close     = df["close"].values
    volume    = df["volume"].values if "volume" in df.columns else np.ones(len(close))
    atr       = df["atr_14"].values if "atr_14" in df.columns else np.ones(len(close))
    atr_pct   = df["atr_percentile"].values if "atr_percentile" in df.columns else np.full(len(close), 50.)
    ema20     = df["ema_20"].values  if "ema_20"  in df.columns else _ema(close, 20)
    ema50     = df["ema_50"].values  if "ema_50"  in df.columns else _ema(close, 50)
    ts        = df["trend_score"].values if "trend_score" in df.columns else np.zeros(len(close))

    log.info("Menghitung Filter 1: Volatility...")
    f1 = filter_volatility(atr, atr_pct, close, sig)

    log.info("Menghitung Filter 2: Trend Strength...")
    f2 = filter_trend(close, ema20, ema50,
                      df["adx"].values, df["pdi"].values, df["ndi"].values,
                      ts, sig)

    log.info("Menghitung Filter 3: Liquidity...")
    f3 = filter_liquidity(volume, sig)

    log.info("Menghitung Filter 4: Momentum...")
    f4 = filter_momentum(df["rsi_14"].values,
                         df["macd_line"].values, df["macd_signal"].values,
                         df["macd_hist"].values, df["roc_10"].values, sig)

    log.info("Menghitung Filter 5: Market Structure...")
    f5 = filter_structure(close,
                          df["bb_mid"].values, df["bb_position"].values,
                          df["obv"].values, df["obv_ema20"].values, sig)

    composite = f1 + f2 + f3 + f4 + f5

    df["f1_volatility"]  = np.round(f1, 2)
    df["f2_trend"]       = np.round(f2, 2)
    df["f3_liquidity"]   = np.round(f3, 2)
    df["f4_momentum"]    = np.round(f4, 2)
    df["f5_structure"]   = np.round(f5, 2)
    df["signal_score"]   = np.round(composite, 2)
    df["signal_quality"] = [score_to_quality(s) if sig[i] != "NONE" else "NONE"
                            for i, s in enumerate(composite)]
    df["lev_mult"]       = [score_to_leverage_mult(composite[i], sig[i])
                            for i in range(len(sig))]

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def print_report(df: pd.DataFrame) -> None:
    sig_mask = df["signal"] != "NONE"
    active   = df[sig_mask]
    total_a  = len(active)

    div  = "═" * 65
    sep  = "─" * 65
    print(f"\n{div}")
    print("  SIGNAL ENHANCER V7 — QUALITY REPORT")
    print(div)
    print(f"  Total bars           : {len(df):>10,}")
    print(f"  Active signal bars   : {total_a:>10,}")
    print(sep)
    print(f"  {'Quality':<14} {'Count':>8} {'%':>8} {'AvgScore':>10} {'LevMult':>10}")
    print(sep)

    for q in ["EXCELLENT", "GOOD", "FAIR", "WEAK", "SKIP"]:
        mask  = active["signal_quality"] == q
        cnt   = mask.sum()
        pct   = cnt / max(total_a, 1) * 100
        avg_s = active.loc[mask, "signal_score"].mean() if cnt > 0 else 0
        avg_l = active.loc[mask, "lev_mult"].mean()     if cnt > 0 else 0
        bar   = "▓" * int(pct / 3)
        print(f"  {q:<14} {cnt:>8,} {pct:>7.1f}% {avg_s:>10.1f} {avg_l:>10.2f}  {bar}")

    print(sep)

    # Filter breakdown per signal type
    for sig_type in ["LONG", "SHORT"]:
        sub  = active[active["signal"] == sig_type]
        if len(sub) == 0:
            continue
        pass_mask = sub["lev_mult"] > 0
        print(f"\n  {sig_type} signals:")
        print(f"    Total          : {len(sub):,}")
        print(f"    Pass threshold : {pass_mask.sum():,}  ({pass_mask.mean()*100:.1f}%)")
        print(f"    Avg score      : {sub['signal_score'].mean():.1f}")
        print(f"    Avg f1 (vol)   : {sub['f1_volatility'].mean():.1f}/20")
        print(f"    Avg f2 (trend) : {sub['f2_trend'].mean():.1f}/20")
        print(f"    Avg f3 (liq)   : {sub['f3_liquidity'].mean():.1f}/20")
        print(f"    Avg f4 (mom)   : {sub['f4_momentum'].mean():.1f}/20")
        print(f"    Avg f5 (struct): {sub['f5_structure'].mean():.1f}/20")

    # Last bar
    last = df.iloc[-1]
    print(f"\n{sep}")
    print(f"  LAST BAR  [{str(last['timestamp'])[:19]}]")
    print(f"    Signal   : {last.get('signal','—')}  |  Score: {last['signal_score']:.1f}  |  Quality: {last['signal_quality']}")
    print(f"    RSI      : {last['rsi_14']:.2f}")
    print(f"    MACD hist: {last['macd_hist']:.6f}")
    print(f"    BB pos   : {last['bb_position']:.3f}")
    print(f"    OBV vs EM: {'↑ BULL' if last['obv'] > last['obv_ema20'] else '↓ BEAR'}")
    print(f"    ADX      : {last['adx']:.2f}  (DI+={last['pdi']:.1f}  DI-={last['ndi']:.1f})")
    print(f"    Lev mult : {last['lev_mult']:.2f}×")
    print(f"{div}\n")


def save_chart(df: pd.DataFrame, out_dir: Path) -> None:
    """Save signal quality charts."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        fig.patch.set_facecolor("#0D1117")
        for ax in axes:
            ax.set_facecolor("#161B22")

        active = df[df["signal"] != "NONE"].copy()

        # Panel 1: Score distribution
        ax = axes[0]
        colors_map = {"EXCELLENT": "#3FB950", "GOOD": "#58A6FF",
                      "FAIR": "#D29922", "WEAK": "#E3B341", "SKIP": "#F85149"}
        for q, col in colors_map.items():
            sub = active[active["signal_quality"] == q]["signal_score"]
            if len(sub) > 0:
                ax.hist(sub, bins=20, alpha=0.7, color=col, label=q, edgecolor="#30363D")
        ax.axvline(DEFAULT_THRESHOLD, color="#F85149", lw=1.5, linestyle="--", label=f"Threshold {DEFAULT_THRESHOLD}")
        ax.set_title("Signal Score Distribution", color="#E6EDF3", fontsize=11)
        ax.set_xlabel("Score (0-100)", color="#8B949E")
        ax.set_ylabel("Count", color="#8B949E")
        ax.tick_params(colors="#8B949E")
        ax.legend(facecolor="#161B22", labelcolor="#E6EDF3", fontsize=8)
        ax.spines["bottom"].set_color("#30363D"); ax.spines["left"].set_color("#30363D")
        ax.spines["top"].set_visible(False);     ax.spines["right"].set_visible(False)

        # Panel 2: Score over time (rolling avg)
        ax = axes[1]
        df["ts_plot"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None) if df["timestamp"].dtype == object else df["timestamp"]
        score_roll = df.set_index("ts_plot")["signal_score"].replace(0, np.nan).rolling("30D").mean()
        ax.plot(score_roll.index, score_roll.values, color="#58A6FF", lw=1.2, label="30D avg score")
        ax.axhline(DEFAULT_THRESHOLD, color="#F85149", lw=1, linestyle="--")
        ax.set_title("Signal Score over Time (30D rolling avg)", color="#E6EDF3", fontsize=11)
        ax.set_ylabel("Score", color="#8B949E")
        ax.tick_params(colors="#8B949E")
        ax.spines["bottom"].set_color("#30363D"); ax.spines["left"].set_color("#30363D")
        ax.spines["top"].set_visible(False);     ax.spines["right"].set_visible(False)

        # Panel 3: Filter contribution breakdown (stacked bar by quality)
        ax = axes[2]
        quality_order = ["EXCELLENT", "GOOD", "FAIR", "WEAK"]
        f_cols = ["f1_volatility", "f2_trend", "f3_liquidity", "f4_momentum", "f5_structure"]
        f_labels = ["Volatility", "Trend", "Liquidity", "Momentum", "Structure"]
        f_colors = ["#58A6FF", "#3FB950", "#D29922", "#E3B341", "#F0883E"]

        x     = np.arange(len(quality_order))
        width = 0.5
        bottoms = np.zeros(len(quality_order))

        for fi, (fc, fl, fco) in enumerate(zip(f_cols, f_labels, f_colors)):
            vals = []
            for q in quality_order:
                sub = active[active["signal_quality"] == q]
                vals.append(sub[fc].mean() if len(sub) > 0 else 0)
            ax.bar(x, vals, width, bottom=bottoms, label=fl, color=fco, alpha=0.85, edgecolor="#0D1117")
            bottoms += np.array(vals)

        ax.set_xticks(x)
        ax.set_xticklabels(quality_order, color="#E6EDF3")
        ax.set_title("Avg Filter Score by Quality Tier", color="#E6EDF3", fontsize=11)
        ax.set_ylabel("Points", color="#8B949E")
        ax.tick_params(colors="#8B949E")
        ax.legend(facecolor="#161B22", labelcolor="#E6EDF3", fontsize=8)
        ax.spines["bottom"].set_color("#30363D"); ax.spines["left"].set_color("#30363D")
        ax.spines["top"].set_visible(False);     ax.spines["right"].set_visible(False)

        plt.tight_layout(pad=2)
        out_path = out_dir / "signal_quality_report.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
        plt.close()
        log.info("Chart saved → %s", out_path)
    except ImportError:
        log.warning("matplotlib tidak terinstall, skip chart generation")


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def run(df: pd.DataFrame = None, threshold: int = DEFAULT_THRESHOLD,
        report: bool = False, plot: bool = False) -> pd.DataFrame:
    """
    Fungsi utama — bisa dipanggil sebagai API atau standalone.

    Parameters:
        df        : DataFrame input (jika None, load dari SIGNALS_PATH)
        threshold : score minimum untuk menerima trade (default 40)
        report    : print quality report ke terminal
        plot      : save chart ke data/

    Returns:
        DataFrame dengan kolom indikator dan score tambahan
    """
    log.info("═" * 65)
    log.info("Signal Enhancer V7 — BTC Hybrid Model")
    log.info("  Filters: Volatility | Trend | Liquidity | Momentum | Structure")
    log.info("  Score threshold: %d / 100", threshold)
    log.info("═" * 65)

    # ── Load data ────────────────────────────────────────────────
    if df is None:
        if not SIGNALS_PATH.exists():
            log.error("File tidak ditemukan: %s", SIGNALS_PATH)
            log.error("Jalankan hybrid_engine.py terlebih dahulu.")
            sys.exit(1)
        df = pd.read_csv(SIGNALS_PATH, parse_dates=["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        log.info("Loaded: %d baris | %s → %s",
                 len(df),
                 df["timestamp"].iloc[0].strftime("%Y-%m-%d"),
                 df["timestamp"].iloc[-1].strftime("%Y-%m-%d"))

    # ── Compute indicators ───────────────────────────────────────
    df = compute_all_indicators(df)

    # ── Compute scores ───────────────────────────────────────────
    df = compute_scores(df)

    # ── Apply threshold ──────────────────────────────────────────
    # Signal yang tidak lolos threshold di-set ke NONE
    # Original signal disimpan di signal_original
    df["signal_original"] = df["signal"].copy()
    filtered_mask = (df["signal"] != "NONE") & (df["signal_score"] < threshold)
    n_filtered = filtered_mask.sum()
    df.loc[filtered_mask, "signal"] = "NONE"
    df.loc[filtered_mask, "lev_mult"] = 0.0
    log.info("Signal difilter (score < %d): %d bars dari %d active (%.1f%%)",
             threshold, n_filtered,
             (df["signal_original"] != "NONE").sum(),
             n_filtered / max((df["signal_original"] != "NONE").sum(), 1) * 100)

    # ── Report ───────────────────────────────────────────────────
    if report:
        print_report(df)

    # ── Chart ────────────────────────────────────────────────────
    if plot:
        save_chart(df, CHART_DIR)

    # ── Save ─────────────────────────────────────────────────────
    df.to_csv(OUTPUT_PATH, index=False)
    log.info("Tersimpan → %s  (%d baris, %d kolom)",
             OUTPUT_PATH, len(df), len(df.columns))
    log.info("═" * 65)

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Signal Enhancer V7 — BTC Hybrid Model")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD,
                        help=f"Score minimum untuk menerima trade (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--report", action="store_true",
                        help="Tampilkan quality report detail ke terminal")
    parser.add_argument("--plot", action="store_true",
                        help="Save chart ke data/signal_quality_report.png")
    args = parser.parse_args()

    run(threshold=args.threshold, report=args.report, plot=args.plot)
