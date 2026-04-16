"""
indicators.py — Technical indicator functions (pure pandas, no lookahead bias).

Semua fungsi:
  - Menerima DataFrame dengan kolom OHLCV standar
  - Mengembalikan pd.Series dengan index yang sama
  - Tidak menggunakan data masa depan (no lookahead bias)
  - Menggunakan pandas ewm/rolling yang by default causal (tidak shift)
"""

import pandas as pd
import numpy as np


# ── EMA ───────────────────────────────────────────────────────────────────────

def calculate_ema(df: pd.DataFrame, period: int, column: str = "close") -> pd.Series:
    """
    Exponential Moving Average (EMA).

    Menggunakan pandas ewm dengan adjust=False — identik dengan formula EMA standar:
        EMA_t = price_t * k + EMA_{t-1} * (1 - k)
        k = 2 / (period + 1)

    Tidak ada lookahead bias karena ewm hanya melihat data masa lalu.

    Parameters
    ----------
    df     : DataFrame dengan kolom `column`
    period : window EMA (misal: 20, 50, 200)
    column : kolom sumber, default "close"

    Returns
    -------
    pd.Series bernama f"ema_{period}"
    """
    if column not in df.columns:
        raise ValueError(f"Kolom '{column}' tidak ditemukan di DataFrame.")

    ema = df[column].ewm(span=period, adjust=False).mean()
    ema.name = f"ema_{period}"
    return ema


# ── ATR ───────────────────────────────────────────────────────────────────────

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range (ATR) — Wilder's smoothing method.

    True Range = max(
        high - low,
        |high - prev_close|,
        |low  - prev_close|
    )

    ATR = Wilder EMA dari True Range dengan period window.
    Wilder EMA ≡ ewm(alpha = 1/period, adjust=False)

    Tidak ada lookahead bias — semua kalkulasi hanya melihat past data.

    Parameters
    ----------
    df     : DataFrame dengan kolom 'high', 'low', 'close'
    period : ATR period, default 14

    Returns
    -------
    pd.Series bernama f"atr_{period}"
    """
    for col in ["high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan di DataFrame.")

    high       = df["high"]
    low        = df["low"]
    prev_close = df["close"].shift(1)   # shift(1) = lihat candle sebelumnya, no lookahead

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    tr.name = "true_range"

    # Wilder smoothing = EMA dengan alpha = 1/period
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    atr.name = f"atr_{period}"
    return atr


# ── ATR Percentile (rolling, no lookahead) ────────────────────────────────────

def calculate_atr_percentile(atr: pd.Series, window: int = 100) -> pd.Series:
    """
    Rolling percentile rank dari ATR dalam window candle terakhir.

    Untuk setiap candle t, hitung: percentile rank ATR_t
    di antara ATR[t-window+1 ... t].

    Tidak ada lookahead bias karena:
    - Hanya menggunakan window candle masa lalu (termasuk t saat ini)
    - Tidak ada future data yang dipakai

    Parameters
    ----------
    atr    : pd.Series hasil calculate_atr()
    window : jumlah candle lookback, default 100

    Returns
    -------
    pd.Series nilai 0.0–100.0, bernama "atr_percentile"
    """
    def _percentile_rank(arr: np.ndarray) -> float:
        """Rank nilai terakhir dalam array (index -1) vs seluruh array."""
        if len(arr) == 0 or np.isnan(arr[-1]):
            return np.nan
        return float(np.sum(arr <= arr[-1])) / len(arr) * 100.0

    pct = atr.rolling(window=window, min_periods=window).apply(
        _percentile_rank, raw=True
    )
    pct.name = "atr_percentile"
    return pct
