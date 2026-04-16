"""
paper_trader.py — BTC Hybrid Model: Live Paper Trading Engine + AI Filter + Telegram + Dashboard + News Sentiment.
"""

import argparse
import json
import logging
import time
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    RICH = True
    console = Console()
except ImportError:
    RICH = False

try:
    import ccxt
    CCXT = True
except ImportError:
    CCXT = False

try:
    from agent.local_ai_agent import LocalTradingAgent
    AI_AGENT = True
except ImportError:
    AI_AGENT = False
    print("⚠️  WARNING: AI Agent tidak ditemukan. Trading tanpa AI filter.")

try:
    from telegram_notifier import send_telegram_message, _load_last_notif, _save_last_notif
    TELEGRAM = True
except ImportError:
    TELEGRAM = False
    print("ℹ️  Telegram notifier tidak ditemukan. Notifikasi disabled.")

try:
    from news_sentiment import fetch_all_news_sentiment
    NEWS_ENABLED = True
except ImportError:
    NEWS_ENABLED = False
    print("ℹ️  News sentiment disabled. Install: pip install requests")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════════

BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
LOG_PATH    = DATA_DIR / "paper_trading_log.csv"
STATE_PATH  = DATA_DIR / "paper_trading_state.json"
DATA_DIR.mkdir(exist_ok=True)

INITIAL_EQUITY   = 100.0
SYMBOL           = "BTC/USDT"
TIMEFRAME        = "4h"
EXCHANGE_ID      = "binance"
CANDLES_NEEDED   = 300

# AI Agent settings
ENABLE_AI_FILTER   = True
AI_CONFIDENCE_MIN  = 0.65
AI_MODEL           = "qwen2.5-coder:14b"

ai_agent = None
if AI_AGENT and ENABLE_AI_FILTER:
    try:
        ai_agent = LocalTradingAgent(model=AI_MODEL)
        log.info("✓ AI Agent initialized: %s", AI_MODEL)
    except Exception as e:
        log.warning("⚠️  Gagal init AI Agent: %s. Trading tanpa AI filter.", e)
        ai_agent = None
else:
    log.info("ℹ️  AI Filter disabled. Using quant signals only.")

# Telegram settings
ENABLE_TELEGRAM      = True
TELEGRAM_MODE        = "pipeline"
NOTIFY_ON_AI_DECIDE  = True

# Dashboard settings
ENABLE_DASHBOARD_EXPORT = True

# News sentiment settings
ENABLE_NEWS_SENTIMENT  = True
NEWS_UPDATE_INTERVAL   = 3600
last_news_update       = 0
news_sentiment_data    = {}

# Risk engine parameters
TARGET_VOL       = 1.00
MAX_LEVERAGE     = 5.0
KS_TIER1_DD      = -0.15
KS_TIER2_DD      = -0.25
KS_RESUME_DD     = -0.10
TIER1_SCALE      = 0.50
BAR_LOSS_LIMIT   = -0.12
BAR_GAIN_LIMIT   = +0.25
VOL_WINDOW       = 126
BARS_PER_YEAR    = 2190

TIER2_GAIN_CAP   = +0.12
TIER2_LOSS_CAP   = -0.10

BULL_MULT        = 1.30
BEAR_MULT        = 1.00

EMA_FAST         = 20
EMA_SLOW         = 50
ATR_PERIOD       = 14
MIN_TREND_SCORE  = 2

ENABLE_15M       = True
CANDLES_15M      = 120


# ════════════════════════════════════════════════════════════════════
#  EXCHANGE DATA
# ════════════════════════════════════════════════════════════════════

def get_exchange():
    if not CCXT:
        raise ImportError("ccxt tidak terinstall.\nJalankan: pip install ccxt")
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })
    return exchange


def fetch_ohlcv(exchange, symbol: str, timeframe: str, limit: int = 300) -> pd.DataFrame:
    log.info("Fetching %d candles %s %s ...", limit, symbol, timeframe)
    raw = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df  = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.iloc[:-1]
    log.info("Fetched %d closed candles. Latest: %s  price=$%.2f",
             len(df), df["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M"),
             df["close"].iloc[-1])
    return df


def fetch_ohlcv_15m(exchange, symbol: str = "BTC/USDT", limit: int = 120) -> pd.DataFrame:
    try:
        raw = exchange.fetch_ohlcv(symbol, "15m", limit=limit)
        df  = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df.iloc[:-1]
    except Exception as e:
        log.warning("Gagal fetch 15m (non-critical): %s", e)
        return pd.DataFrame()


# ════════════════════════════════════════════════════════════════════
#  1D TREND
# ════════════════════════════════════════════════════════════════════

def compute_1d_trend(df_4h: pd.DataFrame) -> dict:
    try:
        df_1d = (df_4h.set_index("timestamp")
                      .resample("1D")
                      .agg({"close": "last"})
                      .dropna()
                      .reset_index())
        df_1d["ema10"]  = df_1d["close"].ewm(span=10,  adjust=False).mean()
        df_1d["ema30"]  = df_1d["close"].ewm(span=30,  adjust=False).mean()
        df_1d["sma200"] = df_1d["close"].rolling(200, min_periods=50).mean()

        last = df_1d.iloc[-1]
        ema10  = float(last["ema10"])
        ema30  = float(last["ema30"])
        sma200 = float(last["sma200"]) if pd.notna(last["sma200"]) else 0.0
        close  = float(last["close"])

        bull = (ema10 > ema30) and (close > sma200)
        bear = (ema10 < ema30) and (close < sma200)
        strength = int(ema10 > ema30) + int(close > sma200)

        trend = "BULLISH" if bull else ("BEARISH" if bear else "NEUTRAL")
        return {
            "trend_1d":   trend,
            "bull_1d":    1 if bull else 0,
            "str_1d":     strength,
            "ema10_1d":   round(ema10, 2),
            "ema30_1d":   round(ema30, 2),
            "sma200_1d":  round(sma200, 2),
        }
    except Exception as e:
        log.warning("Gagal hitung 1D trend: %s", e)
        return {"trend_1d": "NEUTRAL", "bull_1d": 0, "str_1d": 1}


# ════════════════════════════════════════════════════════════════════
#  15M ENTRY OPTIMIZER
# ════════════════════════════════════════════════════════════════════

def compute_15m_entry(df_15m: pd.DataFrame, signal_4h: str) -> dict:
    empty = {"entry_quality": "NO_DATA", "entry_score": 0,
             "rsi_15m": 0, "dist_ema21": 0, "vol_ratio": 1.0}

    if len(df_15m) < 55 or signal_4h == "NONE":
        return empty

    try:
        df = df_15m.copy()
        df["ema8"]   = df["close"].ewm(span=8,  adjust=False).mean()
        df["ema21"]  = df["close"].ewm(span=21, adjust=False).mean()
        df["ema55"]  = df["close"].ewm(span=55, adjust=False).mean()

        delta = df["close"].diff()
        gain  = delta.clip(lower=0); loss = (-delta).clip(lower=0)
        avg_g = gain.ewm(com=13, adjust=False).mean()
        avg_l = loss.ewm(com=13, adjust=False).mean()
        df["rsi"] = 100 - (100 / (1 + avg_g / avg_l.clip(lower=1e-10)))

        last = df.iloc[-1]
        cur   = float(last["close"])
        ema21 = float(last["ema21"])
        rsi   = float(last["rsi"])
        dist  = (cur - ema21) / ema21 * 100
        score = 0

        if signal_4h == "LONG":
            if -1.5 <= dist <= 0.5:  score += 3
            if 35 <= rsi <= 55:       score += 2
        else:
            if -0.5 <= dist <= 1.5:  score += 3
            if 45 <= rsi <= 65:       score += 2

        quality = "EXCELLENT" if score >= 7 else "GOOD" if score >= 4 else "WAIT" if score >= 1 else "NO_ENTRY"

        return {
            "entry_quality": quality,
            "entry_score":   score,
            "rsi_15m":       round(rsi, 1),
            "dist_ema21":    round(dist, 2),
        }
    except Exception as e:
        log.warning("Gagal hitung 15m entry: %s", e)
        return empty


# ════════════════════════════════════════════════════════════════════
#  INDIKATOR
# ════════════════════════════════════════════════════════════════════

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ema_fast"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()

    high_low   = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close  = (df["low"]  - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = tr.ewm(span=ATR_PERIOD, min_periods=ATR_PERIOD, adjust=False).mean()
    df["atr_percentile"] = df["atr_14"].rolling(252, min_periods=50).rank(pct=True) * 100

    price_above_ema_fast = (df["close"] > df["ema_fast"]).astype(int)
    ema_fast_above_slow  = (df["ema_fast"] > df["ema_slow"]).astype(int)
    momentum_positive    = (df["close"].pct_change(10) > 0).astype(int)
    df["trend_score"] = price_above_ema_fast + ema_fast_above_slow + momentum_positive

    sma_200 = df["close"].rolling(200, min_periods=50).mean()
    df["bear_market"] = (df["close"] < sma_200).astype(int)

    conditions = [df["trend_score"] >= 2, df["trend_score"] <= 1]
    choices = ["BULLISH", "BEARISH"]
    df["regime"] = np.select(conditions, choices, default="NEUTRAL")

    roc_fast  = df["close"].pct_change(6)
    roc_slow  = df["close"].pct_change(24)
    df["funding_zscore"]      = (roc_fast / roc_fast.rolling(100).std()).fillna(0)
    df["deriv_extreme_short"] = (df["funding_zscore"] < -2).astype(int)
    df["hybrid_score"] = df["trend_score"].clip(0, 5)

    return df


# ════════════════════════════════════════════════════════════════════
#  SIGNAL GENERATION
# ════════════════════════════════════════════════════════════════════

def generate_signal(row: pd.Series, prev_position: int) -> tuple:
    trend  = int(row.get("trend_score", 0))
    bear   = int(row.get("bear_market", 0))
    ema_f  = float(row.get("ema_fast", 0))
    ema_s  = float(row.get("ema_slow", 0))
    deriv  = int(row.get("deriv_extreme_short", 0))
    hybrid = float(row.get("hybrid_score", 0))

    long_cond = (trend >= MIN_TREND_SCORE and bear == 0 and ema_f > ema_s and not (deriv == 1))
    short_cond = (trend <= 1 or bear == 1 or (deriv == 1 and trend <= 2))

    if long_cond:
        return "LONG", f"{hybrid:.2f}", 1
    elif short_cond:
        return "SHORT", f"{hybrid:.2f}", -1
    else:
        return "NONE", "NONE", prev_position


# ════════════════════════════════════════════════════════════════════
#  AI FILTER
# ════════════════════════════════════════════════════════════════════

def apply_ai_filter(quant_signal: str, row: pd.Series, 
                    ctx_1d: dict, ctx_15m: dict) -> dict:
    if not AI_AGENT or not ENABLE_AI_FILTER or ai_agent is None:
        return {"decision": quant_signal, "confidence": 1.0, "reason": "AI filter disabled"}
    
    signal_data = {
        "quant_signal": quant_signal,
        "price": float(row["close"]),
        "trend_score": int(row.get("trend_score", 0)),
        "hybrid_score": float(row.get("hybrid_score", 0)),
        "regime": str(row.get("regime", "NEUTRAL")),
        "bear_market": int(row.get("bear_market", 0)),
        "1d_trend": ctx_1d.get("trend_1d", "NEUTRAL"),
        "15m_entry_quality": ctx_15m.get("entry_quality", "NO_DATA"),
    }
    
    ai_decision = ai_agent.analyze_signal(signal_data)
    
    log.info("🤖 AI Decision: %s (confidence: %.2f) - %s",
             ai_decision.get("decision", "HOLD"),
             ai_decision.get("confidence", 0),
             ai_decision.get("reason", ""))
    
    return ai_decision


# ════════════════════════════════════════════════════════════════════
#  VOLATILITY SCALE
# ════════════════════════════════════════════════════════════════════

def compute_vol_scale(strategy_returns: list, regime_bull: bool = False) -> float:
    if len(strategy_returns) < 10:
        base_sc = 1.0
    else:
        sr  = pd.Series(strategy_returns[-VOL_WINDOW:])
        rv  = float(sr.std() * np.sqrt(BARS_PER_YEAR))
        rv  = max(rv, 0.05)
        base_sc = TARGET_VOL / rv
    mult = BULL_MULT if regime_bull else 1.0
    return float(np.clip(base_sc * mult, 0.30, MAX_LEVERAGE))


# ════════════════════════════════════════════════════════════════════
#  STATE MANAGEMENT
# ════════════════════════════════════════════════════════════════════

def load_state() -> dict:
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            state = json.load(f)
        log.info("State loaded: equity=$%.2f | position=%s | tier=%d",
                 state["equity"], state["position"], state["tier"])
        return state
    return _default_state()


def _default_state() -> dict:
    return {
        "equity":            INITIAL_EQUITY,
        "max_equity":        INITIAL_EQUITY,
        "shadow_equity":     INITIAL_EQUITY,
        "position":          0,
        "tier":              0,
        "total_trades":      0,
        "winning_trades":    0,
        "strategy_returns":  [],
        "last_bar_ts":       None,
        "entry_price":       None,
        "entry_equity":      INITIAL_EQUITY,
        "peak_equity":       INITIAL_EQUITY,
        "started_at":        datetime.now(timezone.utc).isoformat(),
    }


def save_state(state: dict) -> None:
    if NEWS_ENABLED and ENABLE_NEWS_SENTIMENT:
        state["news_sentiment"] = news_sentiment_data.get("sentiment_label", "N/A")
        state["news_score"] = news_sentiment_data.get("sentiment_score", 0)
        state["news_updated"] = news_sentiment_data.get("last_updated", "")
    
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)


def update_state_live_fields(state: dict, price: float, bar_ret: float,
                              leverage: float, vol_scale: float,
                              signal: str, trend_1d: str, bull_1d: int,
                              entry_quality: str) -> None:
    state["last_price"]      = round(price, 2)
    state["last_bar_return"] = round(bar_ret * 100, 4)
    state["last_leverage"]   = round(leverage, 4)
    state["last_vol_scale"]  = round(vol_scale, 4)
    state["last_signal"]     = signal
    state["trend_1d"]        = trend_1d
    state["bull_1d"]         = int(bull_1d)
    state["entry_quality"]   = entry_quality
    state["last_updated"]    = datetime.now(timezone.utc).isoformat()


# ════════════════════════════════════════════════════════════════════
#  TRADE LOGGER - FIXED! ✅
# ════════════════════════════════════════════════════════════════════

def log_bar(ts, signal, position, price, equity, drawdown,
            vol_scale, leverage, tier, bar_ret, strategy_ret,
            trend_1d="NEUTRAL", bull_1d=0, entry_quality="NO_DATA",
            regime_bull=False) -> None:
    row = {
        "timestamp":      ts,
        "signal":         signal,
        "position":       position,
        "price":          round(price, 2),
        "equity":         round(equity, 4),
        "drawdown_pct":   round(drawdown * 100, 4),
        "bar_return_pct": round(bar_ret * 100, 4),
        "strategy_ret":   round(strategy_ret * 100, 6),
        "vol_scale":      round(vol_scale, 4),
        "leverage_used":  round(leverage, 4),
        "kill_tier":      tier,
        "trend_1d":       trend_1d,
        "bull_1d":        bull_1d,
        "regime_bull":    int(regime_bull),
        "entry_quality":  entry_quality,
    }
    df_row = pd.DataFrame([row])
    if LOG_PATH.exists():
        df_row.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        df_row.to_csv(LOG_PATH, index=False)


# ════════════════════════════════════════════════════════════════════
#  CORE PROCESSING - FIXED! ✅
# ════════════════════════════════════════════════════════════════════

def process_bar_with_signal(row: pd.Series, next_close: float, 
                            state: dict, forced_pos: int, 
                            forced_signal: str, 
                            regime_bull: bool = False) -> tuple:
    cur_close  = float(row["close"])
    market_ret = (next_close - cur_close) / cur_close
    raw_pos = forced_pos
    signal = forced_signal
    strength = str(row.get("hybrid_score", 0))
    strategy_ret = raw_pos * market_ret

    state["strategy_returns"].append(strategy_ret)
    if len(state["strategy_returns"]) > VOL_WINDOW * 2:
        state["strategy_returns"] = state["strategy_returns"][-VOL_WINDOW:]

    vol_scale = compute_vol_scale(state["strategy_returns"], regime_bull=regime_bull)

    tier      = state["tier"]
    cur_eq    = state["equity"]
    max_eq    = state["max_equity"]
    shadow    = state["shadow_equity"]
    bar_ret   = 0.0
    leverage  = 0.0

    if tier == 2:
        strategy_ret_capped = float(np.clip(strategy_ret, TIER2_LOSS_CAP, TIER2_GAIN_CAP))
        shadow = max(shadow * (1 + strategy_ret_capped), 0.01)
        state["shadow_equity"] = shadow
        if (shadow - max_eq) / max_eq > KS_RESUME_DD:
            tier = 0
            cur_eq = shadow
    else:
        eff_scale = vol_scale * (TIER1_SCALE if tier == 1 else 1.0)
        if raw_pos != 0:
            bar_ret = strategy_ret * eff_scale
            bar_ret = max(bar_ret, BAR_LOSS_LIMIT)
            bar_ret = min(bar_ret, BAR_GAIN_LIMIT)
            leverage = eff_scale
        else:
            bar_ret = 0.0

        cur_eq  = max(cur_eq * (1 + bar_ret), 0.01)
        shadow  = cur_eq
        if cur_eq > max_eq:
            max_eq = cur_eq
        dd = (cur_eq - max_eq) / max_eq

        if tier == 0 and dd <= KS_TIER1_DD:
            tier = 1
            log.warning("[ALERT] Tier1 (half size) | eq=$%.2f | dd=%.1f%%", cur_eq, dd * 100)
        elif tier == 1:
            if dd <= KS_TIER2_DD:
                tier = 2
                shadow = cur_eq
                log.warning("[WARN] Tier2 (PAUSED) | eq=$%.2f | dd=%.1f%%", cur_eq, dd * 100)
            elif dd > KS_TIER1_DD * 0.5:
                tier = 0

        prev_pos = state["position"]
        if raw_pos != prev_pos and prev_pos != 0:
            state["total_trades"] += 1
            if cur_eq > state.get("entry_equity", cur_eq):
                state["winning_trades"] += 1
            state["entry_equity"] = cur_eq if raw_pos != 0 else None

    dd_final = (cur_eq - max_eq) / max_eq if max_eq > 0 else 0.0

    state.update({
        "equity":        cur_eq,
        "max_equity":    max_eq,
        "shadow_equity": shadow,
        "position":      raw_pos,
        "tier":          tier,
        "last_bar_ts":   str(row["timestamp"]),
        "entry_price":   cur_close if signal in ("LONG", "SHORT") else state.get("entry_price"),
        "peak_equity":   max_eq,
    })

    # ✅ FIXED: Semua argument sebagai keyword arguments
    log_bar(
        ts=ts,
        signal=signal,
        position=raw_pos,
        price=cur_close,
        equity=cur_eq,
        drawdown=dd_final,
        vol_scale=vol_scale,
        leverage=leverage,
        tier=tier,
        bar_ret=bar_ret,
        strategy_ret=strategy_ret,
        trend_1d=state.get("trend_1d", "NEUTRAL"),
        bull_1d=int(regime_bull),
        entry_quality=state.get("entry_quality", "NO_DATA"),
        regime_bull=regime_bull
    )

    update_state_live_fields(
        state=state,
        price=cur_close,
        bar_ret=bar_ret,
        leverage=leverage,
        vol_scale=vol_scale,
        signal=signal,
        trend_1d=state.get("trend_1d", "NEUTRAL"),
        bull_1d=int(regime_bull),
        entry_quality=state.get("entry_quality", "NO_DATA")
    )

    return state, {
        "signal":       signal,
        "strength":     strength,
        "position":     raw_pos,
        "price":        cur_close,
        "equity":       cur_eq,
        "drawdown":     dd_final,
        "bar_ret":      bar_ret,
        "vol_scale":    vol_scale,
        "leverage":     leverage,
        "tier":         tier,
        "market_ret":   market_ret,
        "strategy_ret": strategy_ret,
        "regime_bull":  regime_bull,
    }


def _pos_to_signal(pos: int) -> str:
    return {1: "LONG", -1: "SHORT", 0: "NONE"}.get(pos, "NONE")


# ════════════════════════════════════════════════════════════════════
#  DASHBOARD TERMINAL
# ════════════════════════════════════════════════════════════════════

def print_dashboard(state: dict, last_bar: dict, df: pd.DataFrame,
                    ctx_1d: dict = None, ctx_15m: dict = None) -> None:
    equity     = state["equity"]
    initial    = INITIAL_EQUITY
    pnl        = equity - initial
    pnl_pct    = (equity - initial) / initial * 100
    dd         = (equity - state["max_equity"]) / state["max_equity"] * 100 if state["max_equity"] > 0 else 0
    win_rate   = (state["winning_trades"] / max(state["total_trades"], 1)) * 100
    n_years    = len(df) / BARS_PER_YEAR if len(df) > 0 else 0.001
    cagr       = ((equity / initial) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

    tier_label = {0: "NORMAL ✓", 1: "TIER1 [ALERT] (half)", 2: "TIER2 [WARN] (paused)"}
    pos_label  = {1: "LONG  ▲", -1: "SHORT ▼", 0: "FLAT  —"}
    sig_color  = {"LONG": "[GREEN]", "SHORT": "[RED]", "NONE": "⬜"}

    regime_bull  = last_bar.get("regime_bull", False)
    regime_label = "🔥 BULL BOOST (×1.3)" if regime_bull else "NORMAL"

    trend_1d = ctx_1d.get("trend_1d", "NEUTRAL") if ctx_1d else state.get("trend_1d","NEUTRAL")

    eq_raw   = ctx_15m.get("entry_quality", "NO_DATA") if ctx_15m else "NO_DATA"
    eq_score = ctx_15m.get("entry_score", 0) if ctx_15m else 0

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    div = "═" * 62

    if RICH:
        t = Table(title=f"[CHART] BTC Paper Trader — {now}", box=box.DOUBLE_EDGE,
                  style="bold", title_style="bold cyan")
        t.add_column("Metric", style="dim", width=28)
        t.add_column("Value", justify="right", width=22)

        sign = "+" if pnl >= 0 else ""
        t.add_row("💰 Virtual Equity",  f"[bold green]${equity:.4f}[/]")
        t.add_row("P&L", f"[{'green' if pnl>=0 else 'red'}]{sign}${pnl:.4f} ({sign}{pnl_pct:.2f}%)[/]")
        t.add_row("Drawdown", f"[{'red' if dd < -10 else 'yellow'}]{dd:.2f}%[/]")
        t.add_row("CAGR (ann.)", f"{cagr:+.2f}%")
        t.add_row("", "")
        t.add_row("🎯 Signal", f"{sig_color.get(last_bar.get('signal',''),'⬜')} {last_bar.get('signal','—')}")
        t.add_row("📍 Position", pos_label.get(last_bar.get("position", 0), "—"))
        t.add_row("💲 BTC Price", f"${last_bar.get('price', 0):,.2f}")
        t.add_row("Kill Switch", tier_label.get(last_bar.get("tier", 0), "—"))
        t.add_row("", "")
        
        if NEWS_ENABLED and ENABLE_NEWS_SENTIMENT:
            news_sent = state.get("news_sentiment", "N/A")
            news_score = state.get("news_score", 0)
            news_color = "green" if news_sent == "BULLISH" else "red" if news_sent == "BEARISH" else "yellow"
            t.add_row("📰 News Sentiment", f"[{news_color}]{news_sent} ({news_score:.2f})[/]")
        
        if AI_AGENT and ENABLE_AI_FILTER and ai_agent is not None:
            ai_dec = state.get("ai_decision", "N/A")
            ai_conf = state.get("ai_confidence", 0)
            ai_color = "green" if ai_conf >= AI_CONFIDENCE_MIN else "red"
            t.add_row("🤖 AI Decision", f"[{ai_color}]{ai_dec} ({ai_conf:.2f})[/]")
        t.add_row("", "")
        t.add_row("⏱ 15m Entry Quality", f"{eq_raw} ({eq_score}/10)")
        t.add_row("🔧 Vol Scale", f"{last_bar.get('vol_scale', 1):.3f}x")
        t.add_row("🔧 Leverage Used", f"{last_bar.get('leverage', 0):.3f}x")
        t.add_row("Bar Return", f"{last_bar.get('bar_ret', 0)*100:+.3f}%")
        t.add_row("", "")
        t.add_row("🏆 Total Trades", str(state["total_trades"]))
        t.add_row("🏆 Win Rate", f"{win_rate:.1f}%")
        console.print(t)
    else:
        print(f"\n{div}")
        print(f"  [CHART] BTC PAPER TRADER — {now}")
        print(div)
        print(f"  Virtual Equity    : ${equity:.4f}")
        print(f"  P&L               : ${pnl:+.4f} ({pnl_pct:+.2f}%)")
        print(f"  Drawdown          : {dd:.2f}%")
        if NEWS_ENABLED and ENABLE_NEWS_SENTIMENT:
            print(f"  News Sentiment    : {state.get('news_sentiment', 'N/A')}")
        print(f"  Total Trades      : {state['total_trades']}")
        print(f"  Win Rate          : {win_rate:.1f}%")
        print(div)


# ════════════════════════════════════════════════════════════════════
#  TELEGRAM NOTIFICATION
# ════════════════════════════════════════════════════════════════════

def send_telegram_notification(state: dict, ai_decision: dict = None):
    if not TELEGRAM or not ENABLE_TELEGRAM:
        return
    
    try:
        last_notif = _load_last_notif() if '_load_last_notif' in globals() else {}
        signal_changed = int(state.get("position", 0)) != int(last_notif.get("position", -99))
        tier_changed = int(state.get("tier", 0)) != int(last_notif.get("tier", -99))
        
        if NOTIFY_ON_AI_DECIDE and ai_decision and ai_agent is not None:
            ai_conf = ai_decision.get("confidence", 0)
            ai_dec = ai_decision.get("decision", "N/A")
            if ai_conf >= AI_CONFIDENCE_MIN and ai_dec in ["BUY", "LONG", "SELL", "SHORT"]:
                ai_alert_msg = (
                    f"🤖 <b>AI TRADE APPROVED</b>\n"
                    f"{'─' * 28}\n"
                    f"Decision: <b>{ai_dec}</b>\n"
                    f"Confidence: <b>{ai_conf:.2f}</b>\n"
                    f"Reason: {ai_decision.get('reason', 'N/A')}\n"
                    f"{'─' * 28}\n"
                    f"💰 Equity: ${state.get('equity', 0):.4f}\n"
                    f"💲 BTC: ${state.get('last_price', 0):,.2f}\n"
                    f"<i>BTC Hybrid Model + AI</i>"
                )
                send_telegram_message(ai_alert_msg)
                log.info("✓ Telegram AI alert sent")
        
        if TELEGRAM_MODE == "pipeline":
            from telegram_notifier import run_notify
            run_notify(mode="pipeline")
            log.info("✓ Telegram pipeline notification sent")
        elif TELEGRAM_MODE == "full":
            from telegram_notifier import run_notify
            run_notify(mode="full")
            log.info("✓ Telegram full notification sent")
            
    except Exception as e:
        log.warning("⚠️  Telegram notification failed: %s", e)


# ════════════════════════════════════════════════════════════════════
#  DASHBOARD AUTO-EXPORT
# ════════════════════════════════════════════════════════════════════

def export_dashboard_auto():
    if not DASHBOARD_EXPORT or not ENABLE_DASHBOARD_EXPORT:
        return
    
    try:
        log.info("📊 Auto-exporting dashboard data...")
        subprocess.run(
            ["python", "export_dashboard_data.py"],
            capture_output=True,
            text=True,
            timeout=30
        )
        log.info("✅ Dashboard data exported")
    except Exception as e:
        log.warning("⚠️  Dashboard export failed: %s", e)


# ════════════════════════════════════════════════════════════════════
#  NEWS SENTIMENT
# ════════════════════════════════════════════════════════════════════

def update_news_sentiment() -> dict:
    global news_sentiment_data, last_news_update
    
    if not NEWS_ENABLED or not ENABLE_NEWS_SENTIMENT:
        return {}
    
    current_time = time.time()
    
    if current_time - last_news_update < NEWS_UPDATE_INTERVAL:
        return news_sentiment_data
    
    try:
        log.info("📰 Updating news sentiment...")
        news_sentiment_data = fetch_all_news_sentiment()
        last_news_update = current_time
        log.info("✅ News sentiment updated")
    except Exception as e:
        log.warning("⚠️  News sentiment update failed: %s", e)
    
    return news_sentiment_data


# ════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ════════════════════════════════════════════════════════════════════

def run_once(exchange=None) -> None:
    if exchange is None:
        exchange = get_exchange()

    df = fetch_ohlcv(exchange, SYMBOL, TIMEFRAME, limit=CANDLES_NEEDED)
    df = compute_indicators(df)
    ctx_1d = compute_1d_trend(df)

    ctx_15m = {}
    if ENABLE_15M:
        df_15m  = fetch_ohlcv_15m(exchange, SYMBOL, limit=CANDLES_15M)
        if len(df_15m) > 0:
            cur_signal = _pos_to_signal(load_state().get("position", 0))
            ctx_15m    = compute_15m_entry(df_15m, cur_signal)

    update_news_sentiment()

    state = load_state()
    state["trend_1d"]      = ctx_1d.get("trend_1d", "NEUTRAL")
    state["bull_1d"]       = ctx_1d.get("bull_1d", 0)
    state["entry_quality"] = ctx_15m.get("entry_quality", "NO_DATA")

    latest_ts = str(df["timestamp"].iloc[-1])
    if state.get("last_bar_ts") == latest_ts:
        log.info("Bar %s sudah diproses sebelumnya. Menunggu bar baru...", latest_ts)
        last_bar = {
            "signal":      _pos_to_signal(state["position"]),
            "position":    state["position"],
            "price":       float(df["close"].iloc[-1]),
            "tier":        state["tier"],
            "vol_scale":   compute_vol_scale(state["strategy_returns"], regime_bull=bool(ctx_1d.get("bull_1d",0))),
            "leverage":    0.0,
            "bar_ret":     0.0,
            "regime_bull": bool(ctx_1d.get("bull_1d", 0)),
        }
        print_dashboard(state, last_bar, df, ctx_1d, ctx_15m)
        return

    regime_bull = bool(ctx_1d.get("bull_1d", 0))
    if len(df) >= 2:
        prev_row   = df.iloc[-2]
        next_close = float(df["close"].iloc[-1])
        
        quant_signal, strength, raw_pos = generate_signal(prev_row, state["position"])
        ai_decision = apply_ai_filter(quant_signal, prev_row, ctx_1d, ctx_15m)
        
        if ai_decision.get("confidence", 0) < AI_CONFIDENCE_MIN:
            log.warning("⚠️  AI confidence rendah (%.2f), skip trade", ai_decision.get("confidence", 0))
            raw_pos = 0
            quant_signal = "NONE"
        elif ai_decision.get("decision") == "HOLD":
            log.info("ℹ️  AI recommend HOLD, maintaining position")
            raw_pos = state["position"]
        
        state, last_bar = process_bar_with_signal(
            prev_row, next_close, state, raw_pos, quant_signal, regime_bull=regime_bull
        )
        
        state["ai_decision"] = ai_decision.get("decision", "N/A")
        state["ai_confidence"] = ai_decision.get("confidence", 0)
        state["ai_reason"] = ai_decision.get("reason", "")

    state["last_bar_ts"] = latest_ts
    save_state(state)
    
    send_telegram_notification(state, ai_decision)
    export_dashboard_auto()
    
    print_dashboard(state, last_bar, df, ctx_1d, ctx_15m)


def run_backfill(n_bars: int = 200) -> None:
    exchange = get_exchange()
    df = fetch_ohlcv(exchange, SYMBOL, TIMEFRAME, limit=n_bars + 50)
    df = compute_indicators(df)
    ctx_1d = compute_1d_trend(df)

    log.info("Memulai backfill %d bars...", n_bars)
    state = _default_state()

    for i in range(len(df) - n_bars, len(df) - 1):
        if i < 1: continue
        row        = df.iloc[i]
        next_close = float(df["close"].iloc[i + 1])
        state, last_bar = process_bar_with_signal(row, next_close, state, 
                                                   state["position"], 
                                                   _pos_to_signal(state["position"]),
                                                   regime_bull=bool(ctx_1d.get("bull_1d", 0)))

    state["last_bar_ts"]  = str(df["timestamp"].iloc[-2])
    state["trend_1d"]     = ctx_1d.get("trend_1d", "NEUTRAL")
    state["bull_1d"]      = ctx_1d.get("bull_1d", 0)
    state["entry_quality"]= "NO_DATA"
    save_state(state)

    log.info("Backfill selesai! Equity: $%.4f | Trades: %d",
             state["equity"], state["total_trades"])
    print_dashboard(state, last_bar, df.iloc[:-1], ctx_1d, {})


def run_live() -> None:
    log.info("═" * 58)
    log.info("  BTC Paper Trader — LIVE MODE + AI + TELEGRAM + NEWS")
    log.info("  Symbol    : %s", SYMBOL)
    log.info("  Timeframe : %s", TIMEFRAME)
    log.info("  Equity    : $%.2f (virtual)", INITIAL_EQUITY)
    log.info("  AI Filter : %s", "ENABLED (Qwen2.5-coder:14b)" if ai_agent else "DISABLED")
    log.info("  Telegram  : %s", "ENABLED" if TELEGRAM and ENABLE_TELEGRAM else "DISABLED")
    log.info("  Dashboard : %s", "AUTO-EXPORT ENABLED" if ENABLE_DASHBOARD_EXPORT else "DISABLED")
    log.info("  News API  : %s", "ENABLED (Tavily)" if NEWS_ENABLED and ENABLE_NEWS_SENTIMENT else "DISABLED")
    log.info("  Log file  : %s", LOG_PATH)
    log.info("  Ctrl+C untuk berhenti")
    log.info("═" * 58)

    exchange = get_exchange()

    try:
        while True:
            try:
                run_once(exchange)
            except Exception as e:
                log.error("Error saat proses bar: %s", e)
                log.info("Retry dalam 60 detik...")

            log.info("Menunggu 5 menit untuk cek bar baru...")
            time.sleep(5 * 60)

    except KeyboardInterrupt:
        log.info("Paper trader dihentikan.")
        _print_final_summary()


def _print_final_summary() -> None:
    if not LOG_PATH.exists():
        return
    df = pd.read_csv(LOG_PATH)
    if len(df) == 0:
        return
    state = load_state()
    eq    = state["equity"]
    init  = INITIAL_EQUITY
    pnl   = eq - init
    print(f"\n{'═'*50}")
    print(f"  PAPER TRADING SUMMARY")
    print(f"{'═'*50}")
    print(f"  Initial Equity : ${init:.2f}")
    print(f"  Final Equity   : ${eq:.4f}")
    print(f"  Total P&L      : ${pnl:+.4f} ({(pnl/init*100):+.2f}%)")
    print(f"  Total Bars     : {len(df)}")
    print(f"  Total Trades   : {state['total_trades']}")
    print(f"  Log saved to   : {LOG_PATH}")
    print(f"{'═'*50}")


# ════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BTC Hybrid Paper Trader + AI + Telegram + News",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh:
  python paper_trader.py              # live mode (default)
  python paper_trader.py --backfill   # warm-up dengan 200 bar historis
  python paper_trader.py --once       # cek kondisi sekarang
  python paper_trader.py --reset      # reset state ke awal ($100)
        """
    )
    parser.add_argument("--backfill", action="store_true", help="Proses 200 bar historis")
    parser.add_argument("--once",     action="store_true", help="Proses satu bar")
    parser.add_argument("--reset",    action="store_true", help="Reset state")
    args = parser.parse_args()

    if args.reset:
        state = _default_state()
        save_state(state)
        if LOG_PATH.exists():
            LOG_PATH.rename(LOG_PATH.with_suffix(".bak.csv"))
        log.info("State direset. Equity=$%.2f", INITIAL_EQUITY)
    elif args.backfill:
        run_backfill(n_bars=200)
    elif args.once:
        run_once()
    else:
        run_live()
