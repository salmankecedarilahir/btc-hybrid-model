"""
live_signal_runner.py — Phase 7: Live Signal Runner.

Mode pengiriman:
  - LONG/SHORT       : kirim sinyal trading + alert
  - NEUTRAL/SIDEWAYS : kirim market update (kondisi terkini)

Duplikat dicegah berdasarkan timestamp — tidak kirim bar yang sama 2x.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

from telegram_notifier import send_telegram_message

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR         = Path(__file__).parent
SIGNALS_PATH     = BASE_DIR / "data" / "btc_trading_signals.csv"
LAST_SIGNAL_PATH = BASE_DIR / "data" / "last_signal.txt"


# ── Load ──────────────────────────────────────────────────────────────────────

def load_signals(path: Path) -> pd.DataFrame:
    if not path.exists():
        log.error("File tidak ditemukan: %s", path)
        sys.exit(1)
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


# ── Duplicate Prevention ──────────────────────────────────────────────────────

def load_last_sent() -> str:
    if not LAST_SIGNAL_PATH.exists():
        return ""
    return LAST_SIGNAL_PATH.read_text().strip()


def save_last_sent(ts: str) -> None:
    LAST_SIGNAL_PATH.parent.mkdir(exist_ok=True)
    LAST_SIGNAL_PATH.write_text(ts)
    log.info("Timestamp disimpan: %s", ts)


def is_duplicate(bar_ts: pd.Timestamp) -> bool:
    return str(bar_ts) == load_last_sent()


# ── Message Formatter ─────────────────────────────────────────────────────────

def format_message(bar: pd.Series) -> str:
    signal   = bar.get("signal", "NONE")
    strength = bar.get("signal_strength", "NONE")
    regime   = bar.get("regime", "N/A")
    close    = float(bar.get("close", 0))
    hybrid   = bar.get("hybrid_score", 0)
    deriv    = bar.get("derivatives_score", 0)
    atr_pct  = float(bar.get("atr_percentile", 0))
    ts       = bar["timestamp"].strftime("%Y-%m-%d %H:%M UTC")

    # Header & signal line
    if signal == "LONG":
        header   = "🚀 <b>BTC SIGNAL — LONG</b>"
        sig_line = "[GREEN] <b>Signal    :</b> LONG"
        action   = "[OK] Pertimbangkan posisi <b>LONG / BUY</b>"
    elif signal == "SHORT":
        header   = "🔻 <b>BTC SIGNAL — SHORT</b>"
        sig_line = "[RED] <b>Signal    :</b> SHORT"
        action   = "[OK] Pertimbangkan posisi <b>SHORT / SELL</b>"
    else:
        header   = "[CHART] <b>BTC MARKET UPDATE</b>"
        sig_line = "⚪ <b>Signal    :</b> NONE"
        action   = "⏳ <b>Tunggu sinyal</b> — jangan buka posisi dulu"

    # Strength badge
    strength_str = {
        "STRONG": "💪 STRONG",
        "NORMAL": "[OK] NORMAL",
        "WEAK":   "[WARN]️ WEAK",
        "NONE":   "─ N/A",
    }.get(strength, strength)

    # Regime dengan emoji
    regime_str = {
        "UP":       "[UP] UP",
        "DOWN":     "[DOWN] DOWN",
        "SIDEWAYS": "↔️ SIDEWAYS",
        "NEUTRAL":  "⏸ NEUTRAL",
    }.get(regime, regime)

    return (
        f"{header}\n"
        f"{'─' * 30}\n"
        f"{sig_line}\n"
        f"[CHART] <b>Strength  :</b> {strength_str}\n"
        f"{'─' * 30}\n"
        f"⏰ <b>Time      :</b> {ts}\n"
        f"💰 <b>Price     :</b> ${close:,.2f}\n"
        f"🔄 <b>Regime    :</b> {regime_str}\n"
        f"[UP] <b>Hybrid    :</b> {hybrid}\n"
        f"[DOWN] <b>Deriv     :</b> {deriv}\n"
        f"🌊 <b>ATR Pct   :</b> {atr_pct:.1f}%\n"
        f"{'─' * 30}\n"
        f"{action}\n"
        f"{'─' * 30}\n"
        f"<i>BTC Hybrid Model v1.0 | Auto update 4H</i>"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> None:
    log.info("═" * 55)
    log.info("Live Signal Runner — BTC Hybrid Model")
    log.info("═" * 55)

    df  = load_signals(SIGNALS_PATH)
    bar = df.iloc[-1]

    signal = bar.get("signal", "NONE")
    bar_ts = bar["timestamp"]

    log.info("Latest bar  : %s", bar_ts)
    log.info("Signal      : %s (%s)", signal, bar.get("signal_strength", "NONE"))
    log.info("Regime      : %s", bar.get("regime", "N/A"))
    log.info("Hybrid Score: %s", bar.get("hybrid_score", 0))

    # Skip jika bar yang sama sudah dikirim
    if is_duplicate(bar_ts):
        log.info("Bar ini sudah dikirim sebelumnya (%s) — skip.", bar_ts)
        return

    # Kirim apapun kondisi market
    msg     = format_message(bar)
    success = send_telegram_message(msg)

    if success:
        save_last_sent(str(bar_ts))
        log.info("✓ Update berhasil dikirim ke Telegram.")
    else:
        log.error("✗ Gagal kirim — akan retry di run berikutnya.")

    log.info("═" * 55)


if __name__ == "__main__":
    run()
