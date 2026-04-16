"""
export_dashboard_data.py — Export paper trading data untuk dashboard HTML.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
STATE_PATH = DATA_DIR / "paper_trading_state.json"
LOG_PATH = DATA_DIR / "paper_trading_log.csv"
OUTPUT_PATH = DATA_DIR / "dashboard_data.json"
SENTIMENT_PATH = DATA_DIR / "news_sentiment.json"

def load_state() -> dict:
    if not STATE_PATH.exists():
        log.warning("State file tidak ditemukan. Jalankan paper_trader.py dulu.")
        return {}
    with open(STATE_PATH, encoding="utf-8") as f:
        return json.load(f)

def load_log() -> pd.DataFrame:
    if not LOG_PATH.exists():
        log.warning("Log file tidak ditemukan. Jalankan paper_trader.py dulu.")
        return pd.DataFrame()
    return pd.read_csv(LOG_PATH)

def load_news_sentiment() -> dict:
    """Load news sentiment dari file."""
    if not SENTIMENT_PATH.exists():
        log.warning("News sentiment file tidak ditemukan. Jalankan news_sentiment.py dulu.")
        return {}
    try:
        with open(SENTIMENT_PATH, encoding="utf-8") as f:
            data = json.load(f)
        log.info("✅ Loaded news sentiment: %s (Score: %.2f)", 
                 data.get("sentiment_label", "N/A"), 
                 data.get("sentiment_score", 0))
        return data
    except Exception as e:
        log.warning("⚠️  Failed to load news sentiment: %s", e)
        return {}

def load_raw_candles(limit: int = 300) -> list:
    try:
        import ccxt
        exchange = ccxt.binance({"enableRateLimit": True})
        raw = exchange.fetch_ohlcv("BTC/USDT", "4h", limit=limit)
        candles = []
        for r in raw:
            candles.append({
                "timestamp": r[0],
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
                "volume": float(r[5]),
            })
        log.info("✅ Loaded %d candles from Binance", len(candles))
        return candles
    except Exception as e:
        log.error("❌ Gagal fetch dari Binance: %s", e)
        return []

def export_dashboard_data():
    log.info("Loading data untuk dashboard...")
    
    state = load_state()
    log_df = load_log()
    candles = load_raw_candles()
    news_sentiment = load_news_sentiment()
    
    if not state:
        log.error("State kosong. Jalankan paper_trader.py --once dulu.")
        return False, None
    
    initial_equity = 100.0
    equity = state.get("equity", initial_equity)
    pnl = equity - initial_equity
    pnl_pct = (pnl / initial_equity) * 100
    max_equity = state.get("max_equity", initial_equity)
    drawdown = ((equity - max_equity) / max_equity) * 100 if max_equity > 0 else 0
    total_trades = state.get("total_trades", 0)
    winning_trades = state.get("winning_trades", 0)
    win_rate = (winning_trades / max(total_trades, 1)) * 100
    ai_confidence = state.get("ai_confidence", 0)
    ai_decision = state.get("ai_decision", "N/A")
    
    candle_data = []
    for c in candles:
        try:
            ts_seconds = int(c["timestamp"]) // 1000
            candle_data.append({
                "time": ts_seconds,
                "open": c["open"],
                "high": c["high"],
                "low": c["low"],
                "close": c["close"],
            })
        except Exception as e:
            log.warning("Skip candle: %s", e)
    
    equity_curve = []
    if not log_df.empty:
        for _, row in log_df.iterrows():
            try:
                ts = row.get("timestamp", "")
                if isinstance(ts, str):
                    dt = pd.to_datetime(ts)
                    ts_seconds = int(dt.timestamp())
                else:
                    ts_seconds = int(ts) // 1000 if ts > 1e12 else int(ts)
                
                equity_curve.append({
                    "time": ts_seconds,
                    "value": float(row.get("equity", initial_equity)),
                })
            except Exception as e:
                log.warning("Skip equity row: %s", e)
    
    trades = []
    if not log_df.empty:
        for _, row in log_df.iterrows():
            try:
                ts = row.get("timestamp", "")
                if isinstance(ts, str):
                    dt = pd.to_datetime(ts)
                    ts_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    ts_str = str(ts)
                
                trades.append({
                    "timestamp": ts_str,
                    "signal": str(row.get("signal", "NONE")),
                    "position": int(row.get("position", 0)),
                    "price": float(row.get("price", 0)),
                    "equity": float(row.get("equity", initial_equity)),
                    "ai_decision": ai_decision,
                })
            except Exception as e:
                log.warning("Skip trade row: %s", e)
    
    output = {
        "equity": float(equity),
        "initial_equity": float(initial_equity),
        "pnl": float(pnl),
        "pnl_pct": float(pnl_pct),
        "drawdown": float(drawdown),
        "total_trades": int(total_trades),
        "winning_trades": int(winning_trades),
        "win_rate": float(win_rate),
        "ai_confidence": float(ai_confidence) if ai_confidence else 0,
        "ai_decision": str(ai_decision),
        "news_sentiment": news_sentiment.get("sentiment_label", "N/A"),
        "news_score": news_sentiment.get("sentiment_score", 0),
        "news_positive": news_sentiment.get("positive_count", 0),
        "news_negative": news_sentiment.get("negative_count", 0),
        "news_neutral": news_sentiment.get("neutral_count", 0),
        "news_total": news_sentiment.get("total_articles", 0),
        "news_updated": news_sentiment.get("last_updated", ""),
        "news_articles": news_sentiment.get("top_titles", [])[:5],
        "candles": candle_data,
        "equity_curve": equity_curve,
        "trades": trades,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }
    
    DATA_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    
    log.info("✅ Dashboard data exported to: %s", OUTPUT_PATH)
    log.info("   Candles: %d | Equity points: %d | Trades: %d", 
             len(candle_data), len(equity_curve), len(trades))
    log.info("   Equity: $%.4f | P&L: %+.2f%% | Trades: %d", equity, pnl_pct, total_trades)
    log.info("   News Sentiment: %s (Score: %.2f)", 
             output["news_sentiment"], output["news_score"])
    
    return True, output

if __name__ == "__main__":
    success, output = export_dashboard_data()
    if success and output:
        print("\n" + "=" * 60)
        print("  ✅ DASHBOARD DATA EXPORTED")
        print("=" * 60)
        print(f"  File: {OUTPUT_PATH}")
        print(f"  Equity: ${output['equity']:.4f}")
        print(f"  News Sentiment: {output['news_sentiment']} (Score: {output['news_score']:.2f})")
        print("=" * 60)
        print("\n🌐 Buka: http://localhost:8000/dashboard_tradingview.html")
        print("💡 Tekan Ctrl+Shift+R untuk hard refresh browser!")
    else:
        print("\n❌ Export gagal. Pastikan paper_trader.py sudah dijalankan dulu.")
