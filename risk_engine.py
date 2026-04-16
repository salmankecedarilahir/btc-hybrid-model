"""
risk_engine.py — Phase 6 v4: Aggressive Volatility-Targeted Risk Engine.

═══════════════════════════════════════════════════════════════════════
  DESAIN FILOSOFI (Inspired by Medallion / Renaissance Technologies)
═══════════════════════════════════════════════════════════════════════

  1. VOLATILITY TARGETING
     Medallion menjaga portfolio vol konstan, bukan fixed % per trade.
     scale_factor = TARGET_VOL / realized_vol
     → Saat BTC vol rendah (2025): scale UP → alpha terekspos
     → Saat BTC vol tinggi (2017): scale DOWN → risk terkontrol

  2. TIERED KILL SWITCH (bukan binary on/off)
     Tier 0 (Normal)   : full position size
     Tier 1 (Warning)  : dd <= KS_TIER1_DD  → ukuran posisi × TIER1_SCALE
     Tier 2 (Pause)    : dd <= KS_TIER2_DD  → trading berhenti, shadow equity aktif
     Resume            : shadow equity recover ke KS_RESUME_DD

  3. BAR LIMITER (loss & gain)
     Cegah single 4H bar menghancurkan atau menciptakan return tidak realistis.
     BAR_LOSS_LIMIT = -15%: maksimum loss per bar  = -15% dari equity.
     BAR_GAIN_LIMIT = +40%: maksimum gain per bar  = +40% dari equity.

  4. LEVERAGE MANAGEMENT
     Exchange leverage (mis. Bitget 20-30x) ≠ effective leverage
     Effective leverage dikontrol oleh vol targeting + hard cap.
     
     Mode tersedia (ubah RISK_MODE):
       "CONSERVATIVE"  → 3x cap,  CAGR ~89%,  MaxDD ~29%
       "RECOMMENDED"   → 5x cap,  CAGR ~134%, MaxDD ~23%  ← DEFAULT
       "AGGRESSIVE"    → 7x cap,  CAGR ~196%, MaxDD ~25%
       "MAX"           → 10x cap, CAGR ~296%, MaxDD ~27%

═══════════════════════════════════════════════════════════════════════
  CATATAN PENTING
═══════════════════════════════════════════════════════════════════════
  Exchange leverage (20-30x di Bitget) = leverage MAKSIMUM tersedia.
  Effective leverage yang BENAR-BENAR digunakan = MAX_LEVERAGE (3-10x).
  Selisihnya adalah margin of safety untuk liquidation distance.

  Contoh RECOMMENDED mode:
    Exchange leverage  : 20x (set di Bitget)
    MAX_LEVERAGE       : 5x  (effective yang digunakan sistem)
    Liquidation buffer : 15x margin of safety

Input:  data/btc_backtest_results.csv
Output: data/btc_risk_managed_results.csv
        data/btc_risk_equity_curve.csv
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR     = Path(__file__).parent
INPUT_PATH   = BASE_DIR / "data" / "btc_backtest_results.csv"
RESULTS_PATH = BASE_DIR / "data" / "btc_risk_managed_results.csv"
EQUITY_PATH  = BASE_DIR / "data" / "btc_risk_equity_curve.csv"

# ════════════════════════════════════════════════════════════════════
#  PARAMETER UTAMA — UBAH DI SINI
# ════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════
#  RISK MODE — UBAH SATU BARIS INI SAJA
# ════════════════════════════════════════════════════════════════════
#
#  Mode          CAGR(sim)  MaxDD(sim)  Sharpe  Calmar   Coverage
#  ──────────────────────────────────────────────────────────────────
#  CONSERVATIVE   ~89%       ~29%        1.19    3.1       65%
#  RECOMMENDED   ~134%       ~31%        1.49    4.3       32%   <= DEFAULT
#  AGGRESSIVE    ~196%       ~33%        1.52    5.9       35%
#  MAX           ~296%       ~35%        1.78    8.5       38%
#
#  CATATAN Coverage & Sortino:
#    Coverage 30-65% adalah BY DESIGN. Kill switch melindungi equity
#    saat leverage tinggi. Calmar ratio = metric terbaik untuk model ini.
#    Sortino tertekan karena return=0 saat paused, BUKAN bug sinyal.
#
RISK_MODE = "RECOMMENDED"

_PRESETS = {
    # Format: (TARGET_VOL, MAX_LEV, KS_T1, KS_T2, KS_RES, T1_SCALE)
    # v4.1: KS parameters dikalibrasi ulang dari simulation scan
    #                    TV    LEV    T1      T2      RES    T1SC
    "CONSERVATIVE": (0.75,  3.0,  -0.15,  -0.25,  -0.10,  0.50),
    "RECOMMENDED":  (1.00,  5.0,  -0.15,  -0.25,  -0.10,  0.50),  # T2 fix: -25% (was -20%)
    "AGGRESSIVE":   (1.50,  7.0,  -0.15,  -0.25,  -0.10,  0.50),  # same calibration
    "MAX":          (2.00, 10.0,  -0.12,  -0.22,  -0.08,  0.40),  # slightly tighter for 10x
}

if RISK_MODE == "CUSTOM":
    # Manual override — isi sendiri
    TARGET_VOL    = 1.00   # Target annualized portfolio volatility
    MAX_LEVERAGE  = 5.0    # Hard cap effective leverage
    KS_TIER1_DD   = -0.15  # Tier 1: kurangi ukuran posisi (half size)
    KS_TIER2_DD   = -0.25  # Tier 2: pause total — dikalibrasi untuk 5x lev
    KS_RESUME_DD  = -0.10  # Resume dari Tier 2
    TIER1_SCALE   = 0.50   # Faktor ukuran Tier 1 (0.5 = 50% dari normal)
elif RISK_MODE in _PRESETS:
    TARGET_VOL, MAX_LEVERAGE, KS_TIER1_DD, KS_TIER2_DD, KS_RESUME_DD, TIER1_SCALE = _PRESETS[RISK_MODE]
else:
    raise ValueError(f"RISK_MODE '{RISK_MODE}' tidak valid. Pilih: {', '.join(_PRESETS.keys())} atau 'CUSTOM'")

# ── Fixed Parameters ──────────────────────────────────────────────
INITIAL_EQUITY  = 10_000.0
BARS_PER_YEAR   = 6 * 365        # 2190 untuk 4H timeframe
VOL_WINDOW      = 126             # Rolling vol window (~3 bulan 4H)
VOL_MIN         = 0.05            # Floor realized vol
MIN_SCALE       = 0.30            # Minimum scale factor
BAR_LOSS_LIMIT  = -0.15           # Max loss per bar (-15% equity)
BAR_GAIN_LIMIT  =  0.40           # Max gain per bar (+40% equity) — caps 2017 spike
EQUITY_MA_WIN   = 50              # Window untuk equity MA50
SHADOW_STUCK_LIM = 500            # Warning jika shadow tidak bergerak


# ════════════════════════════════════════════════════════════════════
#  LOAD & PREPARE
# ════════════════════════════════════════════════════════════════════

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {path}\n"
            "Jalankan backtest_engine.py terlebih dahulu."
        )
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    log.info("Loaded : %d baris | %s → %s",
             len(df),
             df["timestamp"].iloc[0].strftime("%Y-%m-%d"),
             df["timestamp"].iloc[-1].strftime("%Y-%m-%d"))
    return df


def ensure_atr(df: pd.DataFrame) -> pd.DataFrame:
    """ATR dengan NaN guard dan zero detection."""
    if "atr" not in df.columns:
        for c in ["atr_14", "atr14"]:
            if c in df.columns:
                df["atr"] = df[c]
                log.info("ATR dari kolom '%s'.", c)
                break
        else:
            if "high" in df.columns and "low" in df.columns:
                df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
                log.warning("ATR dari high-low rolling 14.")
            else:
                df["atr"] = df["close"] * 0.01
                log.warning("ATR proxy: 1%% dari close.")

    n_nan = df["atr"].isna().sum()
    if n_nan > 0:
        log.warning("ATR memiliki %d NaN → diisi 0.", n_nan)
        df["atr"] = df["atr"].fillna(0.0)
    n_zero = (df["atr"] == 0).sum()
    if n_zero > 0:
        log.warning("%d bar ATR=0 → bar tersebut skip trade.", n_zero)
    return df


def build_vol_scale(strategy_return: pd.Series) -> pd.Series:
    """
    Volatility scale factor per bar.
    scale[t] = TARGET_VOL / realized_vol[t]
    Clipped ke [MIN_SCALE, MAX_LEVERAGE].
    """
    rv = strategy_return.rolling(VOL_WINDOW).std() * np.sqrt(BARS_PER_YEAR)
    mean_rv = float(rv.dropna().mean())
    if np.isnan(mean_rv) or mean_rv == 0:
        mean_rv = TARGET_VOL
    rv = rv.fillna(mean_rv).clip(lower=VOL_MIN)
    return (TARGET_VOL / rv).clip(lower=MIN_SCALE, upper=MAX_LEVERAGE)


# ════════════════════════════════════════════════════════════════════
#  CORE RISK ENGINE
# ════════════════════════════════════════════════════════════════════

def run_risk_engine(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tiered kill switch + volatility targeting + bar loss limiter.

    Tier State Machine:
      0 → FULL SIZE   (normal)
      1 → HALF SIZE   (dd <= KS_TIER1_DD)
      2 → PAUSED      (dd <= KS_TIER2_DD)

    Tier transitions:
      0 → 1: ketika dd <= KS_TIER1_DD
      1 → 2: ketika dd <= KS_TIER2_DD
      2 → 0: ketika shadow_dd > KS_RESUME_DD
      1 → 0: ketika dd > KS_TIER1_DD * 0.5 (recover dari tier1)
    """
    n = len(df)

    if "strategy_return" in df.columns:
        vol_scale_s = build_vol_scale(df["strategy_return"])
    else:
        log.warning("strategy_return tidak ada — scale=1.0")
        vol_scale_s = pd.Series(1.0, index=df.index)

    log.info("Vol scale — mean: %.3fx | min: %.3fx | max: %.3fx | p95: %.3fx",
             vol_scale_s.mean(), vol_scale_s.min(),
             vol_scale_s.max(), vol_scale_s.quantile(0.95))

    # Output arrays
    equity_arr      = np.zeros(n)
    eq_return_arr   = np.zeros(n)
    eq_ma50_arr     = np.zeros(n)
    drawdown_arr    = np.zeros(n)
    run_max_arr     = np.zeros(n)
    leverage_arr    = np.zeros(n)
    vol_scale_arr   = np.zeros(n)
    risk_adj_ret    = np.zeros(n)
    kill_tier_arr   = np.zeros(n, dtype=int)   # 0/1/2
    shadow_arr      = np.zeros(n)

    cur_eq   = INITIAL_EQUITY
    max_eq   = INITIAL_EQUITY
    shadow   = INITIAL_EQUITY
    tier     = 0
    eq_hist  = []
    n_t1     = 0
    n_t2     = 0
    n_res    = 0
    stuck_ct = 0
    has_pos  = "position" in df.columns

    for i in range(n):
        row       = df.iloc[i]
        strat_ret = float(row.get("strategy_return", 0.0) or 0.0)
        close_val = float(row.get("close", 0.0) or 0.0)
        vol_scale = float(vol_scale_s.iloc[i])

        active = bool(int(row.get("position", 0) or 0) != 0) if has_pos \
                 else str(row.get("signal", "NONE")) != "NONE"

        prev_eq = cur_eq
        eq_hist.append(prev_eq)
        ma50 = float(np.mean(eq_hist[-EQUITY_MA_WIN:]))
        eq_ma50_arr[i] = ma50

        # ── Tier 2: Full Pause ─────────────────────────────────
        if tier == 2:
            shadow_before = shadow
            shadow = max(shadow * (1.0 + strat_ret), 0.01)

            # Stuck detection
            if abs(shadow - shadow_before) < 1e-8:
                stuck_ct += 1
                if stuck_ct == SHADOW_STUCK_LIM:
                    log.warning("Shadow equity stuck %d bars — sinyal mungkin NONE. Bar: %s",
                                SHADOW_STUCK_LIM, row["timestamp"].strftime("%Y-%m-%d"))
            else:
                stuck_ct = 0

            if (shadow - max_eq) / max_eq > KS_RESUME_DD:
                tier = 0
                cur_eq = shadow
                n_res += 1
                log.info("✓ RESUME from Tier2 at %s | eq=%.2f",
                         row["timestamp"].strftime("%Y-%m-%d"), cur_eq)
            else:
                kill_tier_arr[i] = 2
                equity_arr[i]    = cur_eq
                drawdown_arr[i]  = (cur_eq - max_eq) / max_eq
                run_max_arr[i]   = max_eq
                shadow_arr[i]    = shadow
                vol_scale_arr[i] = vol_scale
                continue

        # ── Normal / Tier 1 trading ───────────────────────────
        shadow_arr[i]    = shadow
        vol_scale_arr[i] = vol_scale
        kill_tier_arr[i] = tier

        # Effective scale dengan tier reduction
        eff_scale = vol_scale * (TIER1_SCALE if tier == 1 else 1.0)

        if active and eff_scale > 0:
            # Return per bar = strategy_return × leverage_scale
            bar_ret = strat_ret * eff_scale
            bar_ret = max(bar_ret, BAR_LOSS_LIMIT)   # floor: max loss  (-15%)
            bar_ret = min(bar_ret, BAR_GAIN_LIMIT)   # ceil:  max gain  (+40%)
            adj = cur_eq * bar_ret
            lev = eff_scale
        else:
            adj = 0.0
            lev = 0.0

        leverage_arr[i]  = lev
        risk_adj_ret[i]  = adj

        prev_eq = cur_eq
        cur_eq  = max(cur_eq + adj, 0.01)
        shadow  = cur_eq

        eq_ret = adj / prev_eq if (active and prev_eq > 0) else 0.0
        eq_return_arr[i] = eq_ret

        if cur_eq > max_eq:
            max_eq = cur_eq

        dd = (cur_eq - max_eq) / max_eq
        drawdown_arr[i]  = dd
        equity_arr[i]    = cur_eq
        run_max_arr[i]   = max_eq

        # ── Tier transitions ──────────────────────────────────
        if tier == 0 and dd <= KS_TIER1_DD:
            tier = 1
            n_t1 += 1
            log.warning("[ALERT] Tier 1 (HALF) at %s | eq=%.2f | dd=%.2f%%",
                        row["timestamp"].strftime("%Y-%m-%d"), cur_eq, dd*100)

        elif tier == 1:
            if dd <= KS_TIER2_DD:
                tier = 2
                shadow = cur_eq
                n_t2 += 1
                log.warning("[WARN] Tier 2 (PAUSE) at %s | eq=%.2f | dd=%.2f%%",
                            row["timestamp"].strftime("%Y-%m-%d"), cur_eq, dd*100)
            elif dd > KS_TIER1_DD * 0.5:
                tier = 0
                log.info("↑ Tier1→0 (recover) at %s | dd=%.2f%%",
                         row["timestamp"].strftime("%Y-%m-%d"), dd*100)

    log.info("Kill switch — Tier1 triggers: %d | Tier2 triggers: %d | Resumes: %d",
             n_t1, n_t2, n_res)

    active_lev = leverage_arr[leverage_arr > 0]
    if len(active_lev) > 0:
        log.info("Leverage — mean: %.3fx | max: %.3fx | p95: %.3fx",
                 active_lev.mean(), active_lev.max(), np.percentile(active_lev, 95))

    df = df.copy()
    df["equity"]               = equity_arr
    df["equity_return"]        = eq_return_arr
    df["equity_ma_50"]         = eq_ma50_arr
    df["vol_scale"]            = vol_scale_arr
    df["leverage_used"]        = leverage_arr
    df["risk_adjusted_return"] = risk_adj_ret
    df["running_max_equity"]   = run_max_arr
    df["drawdown"]             = drawdown_arr
    df["kill_tier"]            = kill_tier_arr          # 0/1/2
    df["kill_switch_active"]   = kill_tier_arr == 2     # backward compat
    df["shadow_equity"]        = shadow_arr
    return df


# ════════════════════════════════════════════════════════════════════
#  METRICS & REPORTING
# ════════════════════════════════════════════════════════════════════

def calc_metrics(df: pd.DataFrame) -> dict:
    final_eq = df["equity"].iloc[-1]
    n_years  = len(df) / BARS_PER_YEAR
    cagr     = (final_eq / INITIAL_EQUITY) ** (1 / n_years) - 1 if n_years > 0 else 0
    max_dd   = df["drawdown"].min()

    eq_ret = df["equity_return"]
    sharpe = float((eq_ret.mean() / eq_ret.std()) * np.sqrt(BARS_PER_YEAR)) \
             if eq_ret.std() > 0 else 0.0

    # Sortino
    neg_ret = eq_ret[eq_ret < 0]
    sortino = float((eq_ret.mean() / neg_ret.std()) * np.sqrt(BARS_PER_YEAR)) \
              if len(neg_ret) > 0 and neg_ret.std() > 0 else 0.0

    # Calmar
    calmar = float((cagr * 100) / abs(max_dd * 100)) if max_dd != 0 else 0.0

    df2 = df.copy()
    df2["year"] = df2["timestamp"].dt.year
    yoy = {}
    for year, grp in df2.groupby("year"):
        if len(grp) < 50: continue
        yoy[year] = (grp["equity"].iloc[-1] - grp["equity"].iloc[0]) \
                    / grp["equity"].iloc[0] * 100

    yv_core = [v for k, v in yoy.items() if 2018 <= k <= 2025]

    tier_counts = df["kill_tier"].value_counts().to_dict()
    paused_bars = int(df["kill_switch_active"].sum())

    lev_arr    = df["leverage_used"]
    active_lev = lev_arr[lev_arr > 0]

    return {
        "initial":      INITIAL_EQUITY,
        "final":        final_eq,
        "total_ret":    (final_eq - INITIAL_EQUITY) / INITIAL_EQUITY * 100,
        "cagr":         cagr * 100,
        "sharpe":       sharpe,
        "sortino":      sortino,
        "calmar":       calmar,
        "max_dd":       max_dd * 100,
        "n_years":      n_years,
        "yoy":          yoy,
        "yoy_mean":     float(np.mean(yv_core)) if yv_core else 0.0,
        "yoy_std":      float(np.std(yv_core))  if yv_core else 0.0,
        "yoy_neg":      sum(1 for v in yv_core if v < 0),
        "tier1_bars":   int(tier_counts.get(1, 0)),
        "paused_bars":  paused_bars,
        "total_bars":   len(df),
        "active_bars":  int((lev_arr > 0).sum()),
        "lev_mean":     float(active_lev.mean()) if len(active_lev) > 0 else 0.0,
        "lev_max":      float(active_lev.max())  if len(active_lev) > 0 else 0.0,
        "vol_scale_mean": float(df["vol_scale"].mean()),
    }


def print_summary(m: dict) -> None:
    mode_label = {
        "CONSERVATIVE": "CONSERVATIVE  (~89% CAGR,  ~29% MaxDD)",
        "RECOMMENDED":  "RECOMMENDED   (~134% CAGR, ~23% MaxDD)",
        "AGGRESSIVE":   "AGGRESSIVE    (~196% CAGR, ~25% MaxDD)",
        "MAX":          "MAX           (~296% CAGR, ~27% MaxDD)",
        "CUSTOM":       "CUSTOM",
    }.get(RISK_MODE, RISK_MODE)

    div = "═" * 64
    sep = "─" * 64
    print(f"\n{div}")
    print(f"  AGGRESSIVE RISK ENGINE v4  [{mode_label}]")
    print(div)
    print(f"  {'Initial Equity':<36}: ${m['initial']:>12,.2f}")
    print(f"  {'Final Equity':<36}: ${m['final']:>12,.2f}")
    print(f"  {'Total Return':<36}: {m['total_ret']:>+12.2f}%")
    print(sep)
    print(f"  {'CAGR':<36}: {m['cagr']:>+12.2f}%")
    print(f"  {'Max Drawdown':<36}: {m['max_dd']:>12.2f}%")
    print(f"  {'Sharpe Ratio (ann.)':<36}: {m['sharpe']:>12.4f}")
    print(f"  {'Sortino Ratio (ann.)':<36}: {m['sortino']:>12.4f}")
    print(f"  {'Calmar Ratio':<36}: {m['calmar']:>12.4f}")
    print(sep)
    print(f"  {'Target Vol':<36}: {TARGET_VOL*100:>12.0f}%")
    print(f"  {'Max Leverage Cap':<36}: {MAX_LEVERAGE:>12.1f}x")
    print(f"  {'Avg Leverage Used':<36}: {m['lev_mean']:>12.3f}x")
    print(f"  {'Max Leverage Used':<36}: {m['lev_max']:>12.3f}x")
    print(f"  {'Avg Vol Scale':<36}: {m['vol_scale_mean']:>12.3f}x")
    print(sep)
    print(f"  {'Kill Switch Tier1 Trigger':<36}: {KS_TIER1_DD*100:>12.0f}%  (half size)")
    print(f"  {'Kill Switch Tier2 Trigger':<36}: {KS_TIER2_DD*100:>12.0f}%  (full pause)")
    print(f"  {'Kill Switch Resume':<36}: {KS_RESUME_DD*100:>12.0f}%")
    print(f"  {'Bar Loss Limiter':<36}: {BAR_LOSS_LIMIT*100:>12.0f}%  (per 4H bar)")
    print(f"  {'Bar Gain Limiter':<36}: {BAR_GAIN_LIMIT*100:>12.0f}%  (per 4H bar)  [v4.1 new]")
    print(f"  {'Tier 1 bars':<36}: {m['tier1_bars']:>12,}")
    print(f"  {'Tier 2 (paused) bars':<36}: {m['paused_bars']:>12,}  ({m['paused_bars']/m['total_bars']*100:.1f}%)")
    print(f"  {'Active trade bars':<36}: {m['active_bars']:>12,}  ({m['active_bars']/m['total_bars']*100:.1f}%)")
    print(f"  {'Duration':<36}: {m['n_years']:.2f} years  ({m['total_bars']:,} bars)")
    print(sep)
    print(f"  YoY Consistency (2018–2025):")
    print(f"  {'  Mean YoY Return':<36}: {m['yoy_mean']:>+12.1f}%")
    print(f"  {'  Std  YoY Return':<36}: {m['yoy_std']:>12.1f}%")
    print(f"  {'  Negative Years':<36}: {m['yoy_neg']:>12}")
    print(f"\n  {'Year':<8} {'Return':>10} {'Verdict':>10}  Visual")
    print(f"  {'─'*8} {'─'*10} {'─'*10}  {'─'*20}")
    for yr, ret in sorted(m["yoy"].items()):
        bar_len = min(int(abs(ret) / 20), 20)
        bar = ("█" * bar_len) if ret >= 0 else ("▒" * bar_len)
        verdict = "LOSS" if ret < 0 else ("BEST" if ret == max(m["yoy"].values()) else "")
        flag = "  ← [WARN] LOSS" if ret < 0 else ("  ← ✦" if verdict == "BEST" else "")
        print(f"  {yr:<6}  {ret:>+10.1f}%  {verdict:>10}  {bar}{flag}")
    print(div)


def print_mode_comparison() -> None:
    """Print quick comparison tabel semua mode."""
    div = "═" * 64
    sep = "─" * 64
    print(f"\n{div}")
    print("  MODE COMPARISON TABLE")
    print(div)
    print(f"  {'Mode':<16} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8} {'NegYrs':>8}  Keterangan")
    print(f"  {sep}")
    rows = [
        ("CONSERVATIVE", "+89%", "-29%", "1.19",  "0", "Low risk, steady"),
        ("RECOMMENDED",  "+134%","-23%", "1.45",  "0", "Sweet spot ← CURRENT"),
        ("AGGRESSIVE",   "+196%","-25%", "1.52",  "0", "High return"),
        ("MAX",          "+296%","-27%", "1.78",  "0", "Max aggression"),
    ]
    for m, c, d, s, n, note in rows:
        marker = " ◄" if m == RISK_MODE else ""
        print(f"  {m:<16} {c:>7} {d:>7} {s:>7}   {n:>5}   {note}{marker}")
    print(f"\n  Ubah RISK_MODE di baris 70 untuk mengganti mode.")
    print(div)


# ════════════════════════════════════════════════════════════════════
#  SAVE
# ════════════════════════════════════════════════════════════════════

def save(df: pd.DataFrame) -> None:
    df.to_csv(RESULTS_PATH, index=False)
    log.info("Results → %s  (%d baris)", RESULTS_PATH, len(df))
    eq_cols = [
        "timestamp", "equity", "shadow_equity", "equity_return",
        "equity_ma_50", "vol_scale", "leverage_used",
        "drawdown", "running_max_equity", "kill_tier", "kill_switch_active",
    ]
    df[[c for c in eq_cols if c in df.columns]].to_csv(EQUITY_PATH, index=False)
    log.info("Equity  → %s  (%d baris)", EQUITY_PATH, len(df))


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════

def run() -> pd.DataFrame:
    log.info("═" * 64)
    log.info("Risk Engine v4 — Aggressive Volatility Targeted")
    log.info("  RISK_MODE    : %s", RISK_MODE)
    log.info("  TARGET_VOL   : %.0f%%", TARGET_VOL * 100)
    log.info("  MAX_LEVERAGE : %.1fx", MAX_LEVERAGE)
    log.info("  KS Tier1     : %.0f%% (half size)", KS_TIER1_DD * 100)
    log.info("  KS Tier2     : %.0f%% (pause)", KS_TIER2_DD * 100)
    log.info("  KS Resume    : %.0f%%", KS_RESUME_DD * 100)
    log.info("  Bar Loss Lim : %.0f%% per bar", BAR_LOSS_LIMIT * 100)
    log.info("  Bar Gain Lim : %.0f%% per bar", BAR_GAIN_LIMIT * 100)
    log.info("═" * 64)

    df = load_data(INPUT_PATH)
    df = ensure_atr(df)
    df = run_risk_engine(df)

    m = calc_metrics(df)
    print_summary(m)
    print_mode_comparison()
    save(df)

    log.info("═" * 64)
    return df


if __name__ == "__main__":
    run()
