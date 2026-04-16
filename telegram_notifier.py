"""
telegram_notifier.py — Telegram Bot Notification Module.

Credentials dibaca dari file .env di folder project:
  TELEGRAM_BOT_TOKEN=your_token
  TELEGRAM_CHAT_ID=your_chat_id
"""

import argparse
import csv
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
TIMEOUT      = 10
ENV_PATH     = Path(__file__).parent / ".env"

# ── Paths paper trader ────────────────────────────────────────────
DATA_DIR        = Path(__file__).parent / "data"
STATE_PATH      = DATA_DIR / "paper_trading_state.json"
LOG_PATH        = DATA_DIR / "paper_trading_log.csv"
LAST_NOTIF_PATH = DATA_DIR / "last_notification.json"
INITIAL_EQUITY  = 100.0
BARS_PER_YEAR   = 2190


# ════════════════════════════════════════════════════════════════════
#  .env loader (tanpa library tambahan) — TIDAK DIUBAH
# ════════════════════════════════════════════════════════════════════

def load_env(path: Path = ENV_PATH) -> None:
    """
    Baca file .env dan set ke os.environ.
    Format yang didukung:
        KEY=value
        KEY="value"
        KEY='value'
    Baris kosong dan komentar (#) diabaikan.
    """
    if not path.exists():
        log.warning(".env file tidak ditemukan di: %s", path)
        return

    loaded = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("setx "):
                log.warning("Format .env salah (ada 'setx') — abaikan baris: %s", line[:40])
                continue
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, val)
                loaded += 1

    if loaded:
        log.info(".env loaded: %d variabel dari %s", loaded, path)


# Load .env saat module di-import
load_env()


# ════════════════════════════════════════════════════════════════════
#  Credentials — TIDAK DIUBAH
# ════════════════════════════════════════════════════════════════════

def _get_credentials() -> tuple[Optional[str], Optional[str]]:
    token   = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    return token, chat_id


# ════════════════════════════════════════════════════════════════════
#  Core sender — TIDAK DIUBAH
# ════════════════════════════════════════════════════════════════════

def send_telegram_message(text: str, parse_mode: str = "HTML") -> bool:
    """
    Kirim pesan ke Telegram.
    Returns True jika berhasil, False jika gagal.
    """
    token, chat_id = _get_credentials()

    if not token:
        log.error("TELEGRAM_BOT_TOKEN tidak ditemukan.")
        log.error("Pastikan file .env berisi: TELEGRAM_BOT_TOKEN=your_token")
        return False

    if not chat_id:
        log.error("TELEGRAM_CHAT_ID tidak ditemukan.")
        log.error("Pastikan file .env berisi: TELEGRAM_CHAT_ID=your_chat_id")
        return False

    url     = TELEGRAM_API.format(token=token)
    payload = {
        "chat_id":    chat_id,
        "text":       text,
        "parse_mode": parse_mode,
    }

    try:
        resp = requests.post(url, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        if data.get("ok"):
            msg_id = data.get("result", {}).get("message_id", "?")
            log.info("Telegram ✓ (message_id=%s)", msg_id)
            return True
        else:
            log.error("Telegram API error: %s", data.get("description", "unknown"))
            return False

    except requests.exceptions.ConnectionError:
        log.error("Gagal konek ke Telegram — cek koneksi internet / VPN.")
        return False
    except requests.exceptions.Timeout:
        log.error("Telegram request timeout (%ds).", TIMEOUT)
        return False
    except requests.exceptions.RequestException as e:
        log.error("Telegram request error: %s", e)
        return False


# ════════════════════════════════════════════════════════════════════
#  Signal formatter — TIDAK DIUBAH
# ════════════════════════════════════════════════════════════════════

def send_signal_alert(signal_data: dict) -> bool:
    signal   = signal_data.get("signal", "NONE")
    strength = signal_data.get("signal_strength", "NONE")

    emoji = {"LONG": "🚀", "SHORT": "🔻"}.get(signal, "⏸")
    color = {"LONG": "[GREEN]", "SHORT": "[RED]"}.get(signal, "⚪")
    strength_badge = {
        "STRONG": "💪 STRONG",
        "NORMAL": "[OK] NORMAL",
        "WEAK":   "[WARN]️ WEAK",
        "NONE":   "─ NONE",
    }.get(strength, strength)

    close        = signal_data.get("close", 0)
    ts           = signal_data.get("timestamp", "N/A")
    regime       = signal_data.get("regime", "N/A")
    deriv_score  = signal_data.get("derivatives_score", 0)
    hybrid_score = signal_data.get("hybrid_score", 0)
    atr_pct      = signal_data.get("atr_percentile", 0)

    text = (
        f"{emoji} <b>BTC TRADING SIGNAL</b>\n"
        f"{'─' * 28}\n"
        f"{color} <b>Signal    :</b> {signal}\n"
        f"[CHART] <b>Strength  :</b> {strength_badge}\n"
        f"{'─' * 28}\n"
        f"⏰ <b>Time      :</b> {ts}\n"
        f"💰 <b>Price     :</b> ${float(close):,.2f}\n"
        f"🔄 <b>Regime    :</b> {regime}\n"
        f"[UP] <b>Hybrid    :</b> {hybrid_score}\n"
        f"[DOWN] <b>Deriv     :</b> {deriv_score}\n"
        f"🌊 <b>ATR Pct   :</b> {float(atr_pct):.1f}%\n"
        f"{'─' * 28}\n"
        f"<i>BTC Hybrid Model v1.0</i>"
    )

    return send_telegram_message(text)


# ════════════════════════════════════════════════════════════════════
#  PAPER TRADER — DATA LOADERS  (BARU)
# ════════════════════════════════════════════════════════════════════

def _load_state() -> dict:
    if not STATE_PATH.exists():
        log.warning("State file tidak ada: %s", STATE_PATH)
        return {}
    with open(STATE_PATH, encoding="utf-8") as f:
        return json.load(f)


def _load_last_log_row() -> dict:
    """Baca baris terakhir dari paper_trading_log.csv."""
    if not LOG_PATH.exists():
        return {}
    try:
        with open(LOG_PATH, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return rows[-1] if rows else {}
    except Exception:
        return {}


def _count_log_bars() -> int:
    if not LOG_PATH.exists():
        return 0
    try:
        with open(LOG_PATH, encoding="utf-8") as f:
            return sum(1 for _ in f) - 1   # minus header
    except Exception:
        return 0


def _load_last_notif() -> dict:
    if not LAST_NOTIF_PATH.exists():
        return {}
    try:
        with open(LAST_NOTIF_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_last_notif(data: dict) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    with open(LAST_NOTIF_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


# ════════════════════════════════════════════════════════════════════
#  PAPER TRADER — MESSAGE BUILDERS  (BARU)
# ════════════════════════════════════════════════════════════════════



def _load_signal_data() -> dict:
    """Baca baris terakhir dari btc_trading_signals.csv untuk market data."""
    try:
        sig_path = DATA_DIR / "btc_trading_signals.csv"
        if not sig_path.exists():
            return {}
        with open(sig_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return rows[-1] if rows else {}
    except Exception:
        return {}

def _get_live_btc_price() -> float:
    """
    Fetch harga BTC/USDT LIVE dari Binance public API.
    Tidak butuh API key — data publik.
    """
    try:
        url  = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return float(resp.json().get("price", 0))
    except Exception:
        pass
    # Fallback 1: Binance Futures
    try:
        url  = "https://fapi.binance.com/fapi/v1/ticker/price?symbol=BTCUSDT"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return float(resp.json().get("price", 0))
    except Exception:
        pass
    # Fallback 2: baca dari CSV lokal jika API gagal
    try:
        for csv_name in ["btc_4h_raw.csv", "btc_risk_managed_results.csv"]:
            p = DATA_DIR / csv_name
            if p.exists():
                with open(p, encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                price = float(rows[-1].get("close", 0)) if rows else 0.0
                if price > 0:
                    log.warning("API gagal — pakai harga CSV: $%.2f (mungkin tidak live)", price)
                    return price
    except Exception:
        pass
    return 0.0


def _get_latest_btc_price() -> float:
    """Alias untuk backward compatibility."""
    return _get_live_btc_price()

def _build_paper_message(state: dict, last_row: dict, mode: str = "full") -> str:
    """
    Buat pesan Telegram dari state paper trader.
    mode: 'full' | 'signal_change' | 'tier_alert' | 'test'
    """
    now        = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    equity     = float(state.get("equity",       INITIAL_EQUITY))
    max_eq     = float(state.get("max_equity",   INITIAL_EQUITY))
    position   = int(state.get("position",       0))
    tier       = int(state.get("tier",           0))
    total_tr   = int(state.get("total_trades",   0))
    winning_tr = int(state.get("winning_trades", 0))
    trend_1d   = state.get("trend_1d",     "NEUTRAL")
    bull_1d    = int(state.get("bull_1d",   0))
    eq_qual    = state.get("entry_quality", "NO_DATA")
    started_at = str(state.get("started_at", ""))[:10]

    pnl      = equity - INITIAL_EQUITY
    pnl_pct  = pnl / INITIAL_EQUITY * 100
    dd       = (equity - max_eq) / max_eq * 100 if max_eq > 0 else 0.0
    win_rate = winning_tr / max(total_tr, 1) * 100
    # Hitung ny dari started_at (calendar days) bukan dari jumlah log bars
    try:
        started_dt = datetime.fromisoformat(str(state.get("started_at", "")).replace("Z", "+00:00"))
        now_dt     = datetime.now(timezone.utc)
        elapsed_days = max((now_dt - started_dt).total_seconds() / 86400, 1)
        ny = elapsed_days / 365.25
    except Exception:
        ny = max(_count_log_bars() / BARS_PER_YEAR, 1/365.25)
    ny   = max(ny, 1/365.25)  # minimum 1 hari agar tidak overflow
    # Clip CAGR agar tidak overflow (max 10000%)
    raw_cagr = ((equity / INITIAL_EQUITY) ** (1 / ny) - 1) * 100
    cagr     = max(min(raw_cagr, 10000.0), -100.0)

    pos_emoji  = {1: "[GREEN] LONG  ▲", -1: "[RED] SHORT ▼", 0: "⬜ FLAT  —"}
    tier_emoji = {0: "[OK] NORMAL", 1: "[ALERT] TIER1 (half size)", 2: "🚨 TIER2 (PAUSED)"}
    t1d_emoji  = {"BULLISH": "[UP]", "BEARISH": "[DOWN]", "NEUTRAL": "➡️"}
    eq_emoji   = {"EXCELLENT": "[GREEN]", "GOOD": "[YELLOW]", "WAIT": "🔵",
                  "NO_ENTRY":  "[RED]", "NO_DATA": "⬜"}
    pnl_sign   = "[UP]" if pnl >= 0 else "[DOWN]"
    reg_label  = "🔥 BULL BOOST ×1.3" if bull_1d else "— Normal ×1.0"

    # Baca dari state (field baru dari paper_trader fix) → fallback ke log → fallback ke 0
    btc_price = float(state.get("last_price",
                last_row.get("price", _get_latest_btc_price())))
    bar_ret   = float(state.get("last_bar_return",
                last_row.get("bar_return_pct",  0)))
    lev_used  = float(state.get("last_leverage",
                last_row.get("leverage_used",   0)))
    vol_scale = float(state.get("last_vol_scale",
                last_row.get("vol_scale",       1)))
    last_sig  = str(state.get("last_signal",
                last_row.get("signal",          "NONE")))
    # Overwrite trend/regime dari state jika ada (lebih fresh)
    if state.get("trend_1d"):
        trend_1d = state.get("trend_1d", "NEUTRAL")
    if state.get("bull_1d") is not None:
        bull_1d  = int(state.get("bull_1d", 0))
        reg_label = "🔥 BULL BOOST ×1.3" if bull_1d else "— Normal ×1.0"
    if state.get("entry_quality"):
        eq_qual  = state.get("entry_quality", "NO_DATA")

    # ── Load market data dari signals CSV ───────────────────────────
    sig = _load_signal_data()
    sig_close      = float(sig.get("close", 0))
    sig_ts         = str(sig.get("timestamp", "N/A"))[:16].replace("T", " ")
    sig_signal     = str(sig.get("signal", "NONE"))
    sig_strength   = str(sig.get("signal_strength", "NONE"))
    sig_regime     = str(sig.get("regime", "NEUTRAL"))
    sig_hybrid     = float(sig.get("hybrid_score", 0))
    sig_deriv      = float(sig.get("derivatives_score", 0))
    sig_atr        = float(sig.get("atr_percentile", 0))

    # SELALU fetch harga live dari Binance — override semua sumber lain
    live_price = _get_live_btc_price()
    if live_price > 0:
        btc_price = live_price
    elif btc_price == 0 and sig_close > 0:
        btc_price = sig_close  # fallback ke CSV jika API gagal

    # Rekomendasi berdasarkan signal
    sig_emoji = {"LONG": "🚀", "SHORT": "🔻"}.get(sig_signal, "⏸")
    strength_badge = {
        "STRONG": "💪 STRONG", "NORMAL": "[OK] NORMAL",
        "WEAK": "[WARN]️ WEAK",   "NONE": "─ NONE",
    }.get(sig_strength, sig_strength)
    if sig_signal == "LONG":
        rekomendasi = "[GREEN] Sinyal LONG aktif — perhatikan entry"
    elif sig_signal == "SHORT":
        rekomendasi = "[RED] Sinyal SHORT aktif — perhatikan entry"
    else:
        rekomendasi = "⏳ Tunggu sinyal — jangan buka posisi dulu"

    # ── TEST ─────────────────────────────────────────────────────
    if mode == "test":
        return (
            "🤖 <b>BTC Hybrid Model — Test OK</b>\n"
            f"⏰ {now}\n\n"
            "[OK] Telegram bot berhasil terhubung!\n"
            "Notifikasi pipeline siap digunakan."
        )

    # ── TIER ALERT ───────────────────────────────────────────────
    if mode == "tier_alert":
        tier_desc = {
            0: "Kill switch kembali NORMAL. Trading dilanjutkan penuh.",
            1: "Drawdown mencapai -15%. Posisi dikurangi 50%.",
            2: "🚨 Drawdown -25%! Trading DIPAUSED sampai recovery.",
        }
        header = "🚨" if tier >= 2 else "[ALERT]"
        return (
            f"{header} <b>KILL SWITCH ALERT</b>\n"
            f"{'─' * 32}\n"
            f"Status  : <b>{tier_emoji.get(tier, '—')}</b>\n"
            f"{tier_desc.get(tier, '')}\n\n"
            f"💰 Equity   : <b>${equity:.4f}</b>\n"
            f"[DOWN] Drawdown : <b>{dd:.2f}%</b>\n"
            f"💲 BTC      : <b>${btc_price:,.2f}</b>\n"
            f"{'─' * 32}\n"
            f"<i>BTC Hybrid Model</i>"
        )

    # ── SIGNAL CHANGE ────────────────────────────────────────────
    if mode == "signal_change":
        sig_line = {
            "LONG":  "🚀 <b>SINYAL BARU: LONG</b> — BTC diprojeksikan naik",
            "SHORT": "🔻 <b>SINYAL BARU: SHORT</b> — BTC diprojeksikan turun",
            "NONE":  "⏸ <b>HOLD</b> — Tidak ada perubahan posisi",
        }.get(last_sig, f"Signal: {last_sig}")
        return (
            f"🎯 <b>SIGNAL UPDATE — BTC Hybrid Model</b>\n"
            f"{'─' * 32}\n"
            f"{sig_line}\n\n"
            f"📍 Posisi     : <b>{pos_emoji.get(position, '—')}</b>\n"
            f"💲 BTC Price  : <b>${btc_price:,.2f}</b>\n"
            f"[ALERT] Kill Switch: <b>{tier_emoji.get(tier, '—')}</b>\n\n"
            f"🔥 Regime     : <b>{sig_regime}</b>\n"
            f"{t1d_emoji.get(trend_1d,'')} 1D Trend : <b>{trend_1d}</b>\n"
            f"⏱ 15m Entry  : <b>{eq_emoji.get(eq_qual,'')} {eq_qual}</b>\n\n"
            f"💰 Equity : <b>${equity:.4f}</b>  ({pnl_sign} {pnl_pct:+.2f}%)\n"
            f"[DOWN] DD     : <b>{dd:.2f}%</b>\n"
            f"{'─' * 32}\n"
            f"<i>BTC Hybrid Model</i>"
        )

    # ── FULL UPDATE — GABUNGAN market + paper trader ──────────────
    # Format CAGR: tampilkan hanya jika sudah > 30 hari
    elapsed_days_approx = ny * 365.25
    if elapsed_days_approx < 30:
        cagr_line = f"  CAGR     : <b>N/A</b> (baru {int(elapsed_days_approx)}h, min 30h)\n"
    else:
        cagr_line = f"  CAGR     : <b>{cagr:+.2f}%</b>\n"

    return (
        f"[CHART] <b>BTC Hybrid Model — Update 4H</b>\n"
        f"⏰ {now}\n"
        f"{'─' * 32}\n"
        f"\n"
        f"<b>📡 MARKET ({sig_ts} UTC)</b>\n"
        f"  {sig_emoji} Signal   : <b>{sig_signal}</b>  {strength_badge}\n"
        f"  💲 BTC Price : <b>${btc_price:,.2f}</b>\n"
        f"  🔄 Regime    : <b>{sig_regime}</b>\n"
        f"  {t1d_emoji.get(trend_1d,'')} 1D Trend : <b>{trend_1d}</b>\n"
        f"  [UP] Hybrid    : <b>{sig_hybrid:.0f}</b>  |  [DOWN] Deriv: <b>{sig_deriv:.0f}</b>\n"
        f"  🌊 ATR Pct   : <b>{sig_atr:.1f}%</b>\n"
        f"  ⏱ 15m Entry : <b>{eq_emoji.get(eq_qual,'')} {eq_qual}</b>\n"
        f"\n"
        f"<b>💼 PAPER TRADER</b>\n"
        f"  💰 Equity    : <b>${equity:.4f}</b>\n"
        f"  [UP] P&L       : {pnl_sign} <b>${pnl:+.4f} ({pnl_pct:+.2f}%)</b>\n"
        f"  [DOWN] Drawdown  : <b>{dd:.2f}%</b>\n"
        f"{cagr_line}"
        f"\n"
        f"<b>🎯 POSISI SAAT INI</b>\n"
        f"  Posisi     : <b>{pos_emoji.get(position, '—')}</b>\n"
        f"  Bar Return : <b>{bar_ret:+.3f}%</b>\n"
        f"  Leverage   : <b>{lev_used:.2f}x</b>  |  Vol: <b>{vol_scale:.3f}x</b>\n"
        f"  Kill Switch: <b>{tier_emoji.get(tier, '—')}</b>\n"
        f"\n"
        f"<b>[UP] STATISTIK</b>\n"
        f"  Total Trades : <b>{total_tr}</b>  |  Win Rate: <b>{win_rate:.1f}%</b>\n"
        f"  Started      : {started_at}\n"
        f"{'─' * 32}\n"
        f"[TIP] {rekomendasi}\n"
        f"<i>BTC Hybrid Model</i>"
    )



# ════════════════════════════════════════════════════════════════════
#  PAPER TRADER — CHANGE DETECTION  (BARU)
# ════════════════════════════════════════════════════════════════════

def _has_signal_changed(state: dict, last_notif: dict) -> bool:
    return int(state.get("position", 0)) != int(last_notif.get("position", -99))


def _has_tier_changed(state: dict, last_notif: dict) -> bool:
    return int(state.get("tier", 0)) != int(last_notif.get("tier", -99))


# ════════════════════════════════════════════════════════════════════
#  PAPER TRADER — NOTIFICATION ORCHESTRATOR  (BARU)
# ════════════════════════════════════════════════════════════════════

def run_notify(mode: str = "full") -> None:
    """
    Kirim notifikasi paper trader ke Telegram.

    mode:
      'full'     — full update tanpa syarat (dipanggil pipeline)
      'signal'   — hanya jika posisi berubah (LONG/SHORT/FLAT)
      'alert'    — hanya jika kill switch tier berubah
      'pipeline' — full update + auto alert jika ada perubahan
      'test'     — pesan test koneksi
    """
    state      = _load_state()
    last_notif = _load_last_notif()
    last_row   = _load_last_log_row()

    if not state and mode != "test":
        log.warning("State kosong — jalankan paper_trader.py dulu.")
        return

    sent = False

    if mode == "test":
        sent = send_telegram_message(
            _build_paper_message(state, last_row, mode="test")
        )

    elif mode == "signal":
        if _has_signal_changed(state, last_notif):
            sent = send_telegram_message(
                _build_paper_message(state, last_row, mode="signal_change")
            )
            log.info("Signal berubah → notif dikirim")
        else:
            log.info("Tidak ada perubahan sinyal. Skip.")

    elif mode == "alert":
        if _has_tier_changed(state, last_notif):
            sent = send_telegram_message(
                _build_paper_message(state, last_row, mode="tier_alert")
            )
            log.info("Tier berubah → alert dikirim")
        else:
            log.info("Tidak ada perubahan tier. Skip.")

    elif mode == "pipeline":
        # 1. Selalu kirim full update
        sent = send_telegram_message(
            _build_paper_message(state, last_row, mode="full")
        )
        # 2. Bonus: kill switch alert jika tier berubah
        if _has_tier_changed(state, last_notif):
            send_telegram_message(
                _build_paper_message(state, last_row, mode="tier_alert")
            )
            log.info("Bonus: tier alert dikirim")
        # 3. Bonus: signal change jika posisi berubah
        if _has_signal_changed(state, last_notif):
            send_telegram_message(
                _build_paper_message(state, last_row, mode="signal_change")
            )
            log.info("Bonus: signal change alert dikirim")

    else:  # full
        sent = send_telegram_message(
            _build_paper_message(state, last_row, mode="full")
        )

    # Simpan state notif agar detect perubahan di run berikutnya
    if sent:
        _save_last_notif({
            "position": state.get("position", 0),
            "tier":     state.get("tier",     0),
            "equity":   state.get("equity",   INITIAL_EQUITY),
            "sent_at":  datetime.now(timezone.utc).isoformat(),
            "mode":     mode,
        })


# ════════════════════════════════════════════════════════════════════
#  ENTRY POINT  (diperluas dari versi lama)
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="BTC Hybrid Model — Telegram Notifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh:
  python telegram_notifier.py               # full update paper trader sekarang
  python telegram_notifier.py --test        # test koneksi bot
  python telegram_notifier.py --signal      # kirim hanya jika sinyal berubah
  python telegram_notifier.py --alert       # kirim hanya jika kill switch berubah
  python telegram_notifier.py --pipeline    # mode pipeline (full + auto alert)
        """,
    )
    parser.add_argument("--test",     action="store_true", help="Test koneksi bot")
    parser.add_argument("--signal",   action="store_true", help="Kirim hanya jika sinyal berubah")
    parser.add_argument("--alert",    action="store_true", help="Kirim hanya jika tier berubah")
    parser.add_argument("--pipeline", action="store_true", help="Mode pipeline (full + auto alert)")
    args = parser.parse_args()

    token, chat_id = _get_credentials()
    log.info("Token  : %s", "✓ ditemukan" if token else "✗ TIDAK DITEMUKAN")
    log.info("Chat ID: %s", "✓ ditemukan" if chat_id else "✗ TIDAK DITEMUKAN")

    if args.test:
        run_notify("test")
    elif args.signal:
        run_notify("signal")
    elif args.alert:
        run_notify("alert")
    elif args.pipeline:
        run_notify("pipeline")
    else:
        run_notify("full")
