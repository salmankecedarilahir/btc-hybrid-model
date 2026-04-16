"""
config.py — Central configuration for BTC Hybrid Model
Exchange: Bitget Futures (USDT-M Perpetual)
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

OHLCV_PATH   = DATA_DIR / "btc_4h_ohlcv.csv"
FUNDING_PATH = DATA_DIR / "funding_4h.csv"
OI_PATH      = DATA_DIR / "oi_4h.csv"

# ── Exchange (Bitget Futures) ──────────────────────────────────────────────────
SYMBOL_REST  = "BTCUSDT"
PRODUCT_TYPE = "usdt-futures"
TIMEFRAME    = "4h"
LIMIT        = 200        # max candles per Bitget request

# ── History windows (Bitget futures data mulai ~2021) ─────────────────────────
OHLCV_YEARS   = 2         # Yahoo Finance 1H limit: max 730 hari (2 tahun)
FUNDING_YEARS = 3
OI_YEARS      = 3

# ── Bitget REST API ───────────────────────────────────────────────────────────
BITGET_BASE      = "https://api.bitget.com"
FUNDING_ENDPOINT = "/api/v2/mix/market/history-fund-rate"
OI_ENDPOINT      = "/api/v2/mix/market/open-interest"

# ── Misc ──────────────────────────────────────────────────────────────────────
REQUEST_PAUSE = 0.5   # detik antar request pagination

# ── Global Constants ──────────────────────────────────────────────────────────
BARS_PER_YEAR = 2190  # 6 bars/day * 365 days for 4H timeframe
