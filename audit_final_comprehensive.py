"""
audit_final_comprehensive.py — Full Anomaly Audit Post-Fix.

Checks:
  1.  Profit factor vs expectancy consistency  [FIXED: pakai dataset sama]
  2.  Sharpe vs win rate plausibility
  3.  Kill switch behavior (pause & resume)    [FIXED: TIER2 spike detection]
  4.  Equity curve continuity                  [FIXED: threshold + root cause]
  5.  Signal distribution sanity
  6.  LONG vs SHORT return asymmetry
  7.  Drawdown calculation correctness
  8.  Backtest vs risk engine CAGR gap
  9.  Lookahead bias check
  10. Yearly return plausibility
  11. TIER2 Resume Spike Check                 [FIXED v2: signed spike + cap verify]

Changelog vs versi lama:
  - Audit 1:  PF & Expectancy sekarang pakai dataset yang SAMA (equity returns)
              Bug lama: PF dari leverage returns, Expectancy dari raw strategy_return
              Bug v2:   Zero-return TIER2 paused bars ikut dihitung sebagai "active"
                        → inflate denominator WR, distort expectancy jadi palsu negatif
              Fix v2:   Exclude bars di mana equity_return == 0 (paused = flat equity)
  - Audit 3:  Tambah deteksi spike saat resume TIER2
              Fix v2:   Bedakan TIER1 vs TIER2 resume, gunakan shadow_equity jika ada
  - Audit 4:  Threshold 50% → 35%, tambah root cause analysis
              Fix v2:   max_bar exclude zero-return TIER2 flat bars dari calculation
  - Audit 10: Flag return ekstrem, threshold 2000% → 1000%
  - Audit 11: Deteksi TIER2 resume spike secara eksplisit
              Fix v2:   Tambah signed spike, verify resume cap (≤12%) sudah berfungsi
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

BASE_DIR      = Path(__file__).parent
BACKTEST_PATH = BASE_DIR / "data" / "btc_backtest_results.csv"
RISK_PATH     = BASE_DIR / "data" / "btc_risk_managed_results.csv"
SIGNALS_PATH  = BASE_DIR / "data" / "btc_trading_signals.csv"
BARS_PER_YEAR = 6 * 365
INITIAL_EQ    = 10_000.0


def load(path, label):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def ok(msg):   print(f"  [OK] {msg}")
def warn(msg): print(f"  [WARN]️  {msg}")
def err(msg):  print(f"  ❌ {msg}")
def div(n=62): return "═" * n
def sep(n=62): return "─" * n


# ════════════════════════════════════════════════════════════════════
#  AUDIT 1: Profit Factor vs Expectancy  [FIXED]
# ════════════════════════════════════════════════════════════════════

def audit_1_profit_factor(risk_df):
    print(f"\n{div()}")
    print("  AUDIT 1 — Profit Factor vs Expectancy Consistency")
    print(div())

    # ── FIX: Pakai equity_return (bukan strategy_return) untuk keduanya ──
    # Bug lama: PF dihitung dari strategy_return (leverage returns)
    #           Expectancy dihitung dari trade_return / strategy_return (raw)
    #           → Dataset berbeda = hasil inkonsisten secara definisi
    # Fix: gunakan equity_return untuk SEMUA metric di audit ini

    col = "equity_return" if "equity_return" in risk_df.columns else "strategy_return"
    if col == "strategy_return":
        warn("Kolom equity_return tidak ada, fallback ke strategy_return")

    # Filter hanya bar aktif (posisi != 0)
    if "position" in risk_df.columns:
        active_df = risk_df[risk_df["position"] != 0].copy()
    elif "signal" in risk_df.columns:
        active_df = risk_df[risk_df["signal"] != "NONE"].copy()
    else:
        active_df = risk_df.copy()

    ret = pd.to_numeric(active_df[col], errors="coerce").dropna()

    # ── FIX v2: Exclude zero-return bars (TIER2 paused bars) ─────────────────
    # Root cause Audit 1 inkonsisten:
    #   Saat TIER2 pause, equity FLAT → equity_return = persis 0.0
    #   Tapi position column masih != 0 (signal forward-fill tetap jalan)
    #   Sehingga filter `position != 0` tidak menyaring paused bars
    #   Efek: zero bars masuk ke `ret`, inflate len(ret) tanpa tambah wins/losses
    #         → win_rate = len(wins)/len(ret) DEFLATED secara artifisial
    #         → (1-wr) terlalu besar → expectancy palsu negatif meski PF > 1
    # Fix: exclude bars dengan equity_return == 0 (paused = zero contribution)
    #      Lebih presisi: gunakan leverage_used > 0 jika tersedia
    if "leverage_used" in risk_df.columns:
        # Cara paling akurat: leverage > 0 berarti posisi betul-betul dibuka
        lev_active = risk_df["leverage_used"] > 0
        ret = pd.to_numeric(risk_df.loc[lev_active, col], errors="coerce").dropna()
        print(f"  Filter mode      : leverage_used > 0  ({lev_active.sum():,} bars)")
    else:
        # Fallback: exclude zero-return (TIER2 paused bars punya return persis 0)
        n_before = len(ret)
        ret = ret[ret != 0.0]
        n_excluded = n_before - len(ret)
        if n_excluded > 0:
            print(f"  Filter mode      : position!=0 & equity_return!=0")
            print(f"  Excluded         : {n_excluded:,} zero-return bars (TIER2 paused)")
        else:
            print(f"  Filter mode      : position!=0")

    # ── FIX v3: exclude zero-return bars dari SEMUA metric ──────────────────
    # Root cause inkonsistensi PF vs Expectancy (post leverage_used filter):
    #
    # Saat posisi open (lev>0) tapi market candle = flat (market_return≈0),
    # bar tersebut punya equity_return ≈ 0. Jumlahnya bisa mencapai 70%+ dari
    # total leverage>0 bars! Efek pada kalkulasi:
    #
    #   WR = len(wins) / len(ret)   ← len(ret) include zeros → WR deflated
    #   PF = wins.sum() / |losses.sum()| ← tidak terpengaruh zeros
    #
    # Contoh actual: len(ret)=5302, wins=793, losses=673, zeros=3836
    #   WR(wrong)   = 793/5302 = 14.96%  → Exp = 14.96%×2.53% + 85.04%×(-2.00%) = -1.32%
    #   WR(correct) = 793/1466 = 54.1%   → Exp = 54.1%×2.53% + 45.9%×(-2.00%) = +0.45%
    #
    # Fix: gunakan len(wins)+len(losses) sebagai denominator (exclude zeros).
    # Ini MATHEMATICALLY EQUIVALENT dengan cara PF dihitung.

    n_before_zero = len(ret)
    ret_nonzero   = ret[ret != 0.0]   # exclude flat-candle zero-return bars
    n_zeros       = n_before_zero - len(ret_nonzero)

    wins   = ret_nonzero[ret_nonzero > 0]
    losses = ret_nonzero[ret_nonzero < 0]
    n_nonzero = len(wins) + len(losses)

    if n_nonzero == 0:
        err("Tidak ada trade data"); return

    # WR sekarang dihitung dari wins/(wins+losses) — konsisten dengan PF
    wr        = len(wins) / n_nonzero
    avg_win   = wins.mean()   if len(wins)   > 0 else 0.0
    avg_loss  = losses.mean() if len(losses) > 0 else 0.0

    # PF dan Expectancy — HARUS konsisten secara matematis
    pf_manual  = wins.sum() / abs(losses.sum()) \
                 if len(losses) > 0 and losses.sum() != 0 else float("inf")
    expectancy = wr * avg_win + (1 - wr) * avg_loss

    print(f"  Dataset used     : {col}  (active bars only)")
    print(f"  Total active bars: {n_before_zero:,}  ({n_zeros:,} zero-return excluded)")
    print(f"  Non-zero bars    : {n_nonzero:,}  (wins={len(wins)}, losses={len(losses)})")
    print(f"  Win Rate         : {wr*100:.2f}%  (dari {n_nonzero:,} non-zero bars)")
    print(f"  Avg Win          : {avg_win*100:+.4f}%")
    print(f"  Avg Loss         : {avg_loss*100:+.4f}%")
    print(f"  Profit Factor    : {pf_manual:.4f}")
    print(f"  Expectancy/bar   : {expectancy*100:+.6f}%")
    print(sep())

    # Sekarang konsistensi harus terjamin (pakai dataset sama)
    if pf_manual > 1 and expectancy > 0:
        ok(f"Konsisten: PF={pf_manual:.3f}>1 dan Expectancy={expectancy*100:+.4f}%>0")
    elif pf_manual > 1 and expectancy < 0:
        err(f"INKONSISTEN: PF={pf_manual:.3f}>1 tapi Expectancy={expectancy*100:+.4f}%<0")
        err("  Dataset sama tapi hasil bertentangan — kemungkinan data korup")
        err("  Cek kolom equity_return apakah sudah terisi dengan benar")
    elif pf_manual < 1 and expectancy < 0:
        ok(f"Konsisten: PF={pf_manual:.3f}<1 dan Expectancy={expectancy*100:+.4f}%<0")
        warn("  Model tidak profitable di timeframe ini")
    else:
        warn(f"PF={pf_manual:.3f}, Expectancy={expectancy*100:+.4f}% — perlu monitoring")

    print(f"\n  NOTE (karakteristik model momentum/trend-following):")
    print(f"    Low win rate ({wr*100:.1f}%) + PF>1 adalah NORMAL jika:")
    print(f"    avg_win >> avg_loss (win jauh lebih besar dari loss)")
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    print(f"    Win/Loss ratio saat ini: {win_loss_ratio:.2f}x {'[OK] baik' if win_loss_ratio>1 else '❌ perlu perbaikan'}")


# ════════════════════════════════════════════════════════════════════
#  AUDIT 2: Sharpe vs Win Rate Plausibility
# ════════════════════════════════════════════════════════════════════

def audit_2_sharpe_plausibility(risk_df):
    print(f"\n{div()}")
    print("  AUDIT 2 — Sharpe vs Win Rate Plausibility")
    print(div())

    eq_ret = risk_df["equity_return"].fillna(0)
    sharpe = (eq_ret.mean() / eq_ret.std()) * np.sqrt(BARS_PER_YEAR) \
             if eq_ret.std() > 0 else 0

    # Win rate dari equity returns (konsisten)
    active_ret = eq_ret[eq_ret != 0]
    wr = (active_ret > 0).mean() if len(active_ret) > 0 else 0

    # Sortino
    neg_r   = eq_ret[eq_ret < 0]
    sortino = float((eq_ret.mean() / neg_r.std()) * np.sqrt(BARS_PER_YEAR)) \
              if len(neg_r) > 0 and neg_r.std() > 0 else 0.0

    print(f"  Sharpe (ann.)  : {sharpe:.4f}")
    print(f"  Sortino (ann.) : {sortino:.4f}")
    print(f"  Win Rate       : {wr*100:.2f}%")
    print(sep())

    if sharpe > 1.0 and wr < 0.25:
        ok(f"Sharpe {sharpe:.2f} dengan WR {wr*100:.1f}% — plausible (momentum system)")
        ok("  Low WR + High Sharpe normal untuk trend-following")
    elif sharpe > 2.0:
        warn(f"Sharpe {sharpe:.2f} sangat tinggi — cek lookahead bias")
    elif sharpe > 0 and wr > 0:
        ok(f"Sharpe {sharpe:.2f} dan WR {wr*100:.1f}% dalam range normal")
    else:
        err(f"Sharpe {sharpe:.2f} negatif dengan WR {wr*100:.1f}%")

    if sortino >= 1.0:
        ok(f"Sortino {sortino:.4f} ≥ 1.0 — PASS")
    elif sortino >= 0.7:
        warn(f"Sortino {sortino:.4f} < 1.0 — WARN, pertimbangkan tighten BAR_GAIN/LOSS_LIMIT")
    else:
        err(f"Sortino {sortino:.4f} < 0.7 — perlu perbaikan signifikan")


# ════════════════════════════════════════════════════════════════════
#  AUDIT 3: Kill Switch Behavior  [FIXED: TIER2 spike detection]
# ════════════════════════════════════════════════════════════════════

def audit_3_kill_switch(risk_df):
    print(f"\n{div()}")
    print("  AUDIT 3 — Kill Switch Behavior (Pause & Resume)")
    print(div())

    if "kill_switch_active" not in risk_df.columns:
        warn("Kolom kill_switch_active tidak ditemukan"); return

    ks      = risk_df["kill_switch_active"]
    n_total = len(risk_df)
    has_tier= "kill_tier" in risk_df.columns

    if has_tier:
        tier1_bars = (risk_df["kill_tier"] == 1).sum()
        tier2_bars = (risk_df["kill_tier"] == 2).sum()
        print(f"  Kill Switch Mode   : TIERED (v4+)")
        print(f"  Tier 1 (half size) : {tier1_bars:,}  ({tier1_bars/n_total*100:.1f}%)")
        print(f"  Tier 2 (paused)    : {tier2_bars:,}  ({tier2_bars/n_total*100:.1f}%)")
    else:
        n_paused = ks.sum()
        print(f"  Kill Switch Mode   : BINARY (v2/v3)")
        print(f"  Total paused bars  : {n_paused:,}  ({n_paused/n_total*100:.1f}%)")

    # ── FIX v2: Bedakan TIER1 vs TIER2 resume ────────────────────────────────
    # Bug lama: kill_switch_active = 0/1 conflate TIER1 (half size) + TIER2 (paused)
    #           Transisi TIER1→TIER0 bukan resume TIER2 — tidak seharusnya ada spike
    #           False positive: resume TIER1 bisa ter-flag sebagai TIER2 spike
    # Fix v2:   Jika shadow_equity tersedia, gunakan itu untuk deteksi presisi
    #           Jika tidak, tetap pakai ks diff tapi catat keterbatasannya
    has_shadow = "shadow_equity" in risk_df.columns
    ks_arr   = ks.values
    triggers = np.where(np.diff(ks_arr.astype(int)) == 1)[0] + 1
    resumes  = np.where(np.diff(ks_arr.astype(int)) == -1)[0] + 1

    print(f"  Tier2 Trigger events : {len(triggers)}")
    print(f"  Resume events        : {len(resumes)}")
    if not has_shadow:
        print(f"  NOTE: shadow_equity tidak tersedia — resume termasuk TIER1+TIER2")
    print(sep())

    for i, t in enumerate(triggers):
        ts = risk_df["timestamp"].iloc[t].strftime("%Y-%m-%d")
        eq = risk_df["equity"].iloc[t]
        dd = risk_df["drawdown"].iloc[t]
        print(f"  Trigger {i+1:>2}: {ts}  equity=${eq:,.0f}  dd={dd*100:.1f}%")

    # ── FIX v2: Spike detection dengan signed check ───────────────────────────
    spike_threshold = 0.20   # 20% dalam 1 bar = suspicious
    spike_found     = False
    print()
    for i, r in enumerate(resumes):
        ts  = risk_df["timestamp"].iloc[r].strftime("%Y-%m-%d")
        eq  = risk_df["equity"].iloc[r]
        dd  = risk_df["drawdown"].iloc[r]
        if r > 0:
            prev_eq = risk_df["equity"].iloc[r - 1]
            jump    = (eq - prev_eq) / prev_eq if prev_eq > 0 else 0.0
            # FIX v2: signed spike (positif DAN negatif)
            abs_jump = abs(jump)
            if abs_jump > spike_threshold:
                direction = "+" if jump > 0 else ""
                spike_flag = f"  [WARN] TIER2 SPIKE {direction}{jump*100:.1f}%"
                spike_found = True
            else:
                spike_flag = ""
            print(f"  Resume  {i+1:>2}: {ts}  equity=${eq:,.0f}  dd={dd*100:.1f}%{spike_flag}")
        else:
            print(f"  Resume  {i+1:>2}: {ts}  equity=${eq:,.0f}  dd={dd*100:.1f}%")

    print(sep())
    if spike_found:
        err("TIER2 resume spike terdeteksi!")
        err("  Shadow equity terakumulasi tanpa cap saat paused")
        err("  FIX: Tambah TIER2_GAIN_CAP dan TIER2_LOSS_CAP di risk engine")
        err("  Sudah diperbaiki di risk_engine_v6.py (resume jump cap)")
    elif len(resumes) > 0:
        ok("Kill switch pause & resume berfungsi dengan benar")
        ok("Tidak ada TIER2 resume spike terdeteksi")
    else:
        warn("Kill switch triggered tapi tidak pernah resume")


# ════════════════════════════════════════════════════════════════════
#  AUDIT 4: Equity Curve Continuity  [FIXED: threshold + root cause]
# ════════════════════════════════════════════════════════════════════

def audit_4_equity_continuity(risk_df):
    print(f"\n{div()}")
    print("  AUDIT 4 — Equity Curve Continuity")
    print(div())

    eq     = risk_df["equity"]
    # ── FIX v2: exclude zero-return bars (TIER2 flat) dari max_bar detection ──
    # Bug lama: eq.pct_change() mencakup semua bar termasuk TIER2 paused (return=0)
    #           Bukan bug untuk deteksi negatif, tapi max_bar harus dari bar aktif saja
    # Fix: gunakan leverage_used > 0 jika ada, fallback ke equity_return != 0
    if "leverage_used" in risk_df.columns:
        active_mask = risk_df["leverage_used"] > 0
        eq_ret_all  = eq.pct_change().fillna(0)
        eq_ret      = eq_ret_all[active_mask]
    elif "equity_return" in risk_df.columns:
        eq_ret_all  = risk_df["equity_return"].fillna(0)
        active_mask = eq_ret_all != 0
        eq_ret      = eq_ret_all[active_mask]
    else:
        eq_ret = eq.pct_change().fillna(0)

    # Cek negative equity
    neg = (eq <= 0).sum()
    if neg > 0:
        err(f"Ada {neg} bar dengan equity <= 0!")
    else:
        ok("Tidak ada equity negatif")

    if len(eq_ret) == 0:
        warn("Tidak ada active bar untuk max/min bar calculation")
        return

    max_jump = eq_ret.max()
    max_drop = eq_ret.min()
    jump_idx = eq_ret.idxmax()
    drop_idx = eq_ret.idxmin()

    print(f"  Largest single bar gain : {max_jump*100:+.2f}%"
          f"  @ {risk_df['timestamp'].iloc[jump_idx].strftime('%Y-%m-%d')}")
    print(f"  Largest single bar drop : {max_drop*100:+.2f}%"
          f"  @ {risk_df['timestamp'].iloc[drop_idx].strftime('%Y-%m-%d')}")

    # ── FIX: threshold 50% → 35%, tambah root cause analysis ─────
    if max_jump > 0.35:
        err(f"Single bar gain {max_jump*100:.1f}% > 35% — anomaly!")
        # Cek apakah spike ini terjadi saat resume TIER2
        if "kill_switch_active" in risk_df.columns:
            prev_ks = risk_df["kill_switch_active"].shift(1).fillna(False)
            cur_ks  = risk_df["kill_switch_active"]
            resume_mask = (prev_ks == True) & (cur_ks == False)
            if resume_mask.iloc[jump_idx]:
                err("  Root cause: TIER2 resume spike (shadow tanpa cap)")
                err("  Fix: Gunakan TIER2_GAIN_CAP di risk_engine_v6.py")
            else:
                err("  Bukan dari TIER2 resume — cek position sizing / leverage cap")
    elif max_jump > 0.20:
        warn(f"Single bar gain {max_jump*100:.1f}% — cukup tinggi, monitor")
    else:
        ok(f"Single bar gains dalam range wajar ({max_jump*100:.1f}% < 20%)")

    if abs(max_drop) > 0.35:
        err(f"Single bar drop {abs(max_drop)*100:.1f}% > 35% — cek BAR_LOSS_LIMIT")
    elif abs(max_drop) > 0.20:
        warn(f"Single bar drop {abs(max_drop)*100:.1f}% — pertimbangkan tighten limit")
    else:
        ok(f"Single bar drops dalam range wajar ({abs(max_drop)*100:.1f}% < 20%)")

    # Cek NaN
    nan_eq = eq.isna().sum()
    if nan_eq > 0:
        err(f"Ada {nan_eq} NaN di kolom equity!")
    else:
        ok("Tidak ada NaN di equity curve")


# ════════════════════════════════════════════════════════════════════
#  AUDIT 5: Signal Distribution Sanity
# ════════════════════════════════════════════════════════════════════

def audit_5_signal_distribution(sig_df):
    print(f"\n{div()}")
    print("  AUDIT 5 — Signal Distribution Sanity")
    print(div())

    n       = len(sig_df)
    long_n  = (sig_df["signal"] == "LONG").sum()
    short_n = (sig_df["signal"] == "SHORT").sum()
    none_n  = (sig_df["signal"] == "NONE").sum()
    active  = long_n + short_n

    print(f"  LONG  : {long_n:>8,}  ({long_n/n*100:.1f}%)")
    print(f"  SHORT : {short_n:>8,}  ({short_n/n*100:.1f}%)")
    print(f"  NONE  : {none_n:>8,}  ({none_n/n*100:.1f}%)")
    print(sep())

    ratio = long_n / short_n if short_n > 0 else float("inf")
    print(f"  LONG/SHORT ratio : {ratio:.2f}x")

    if active / n < 0.10:
        warn(f"Model terlalu jarang signal ({active/n*100:.1f}%) → CAGR rendah")
    elif active / n > 0.80:
        warn(f"Model terlalu sering signal ({active/n*100:.1f}%) → overtrading risk")
    else:
        ok(f"Active signal ratio {active/n*100:.1f}% dalam range normal")

    if ratio > 5:
        warn(f"LONG/SHORT ratio {ratio:.1f}x sangat tidak seimbang")
    elif ratio < 0.5:
        warn(f"SHORT lebih dominan ({ratio:.1f}x) — cek bias SHORT")
    else:
        ok(f"LONG/SHORT ratio {ratio:.2f}x cukup seimbang")


# ════════════════════════════════════════════════════════════════════
#  AUDIT 6: LONG vs SHORT Return Asymmetry
# ════════════════════════════════════════════════════════════════════

def audit_6_long_short_asymmetry(bt_df):
    print(f"\n{div()}")
    print("  AUDIT 6 — LONG vs SHORT Return Asymmetry")
    print(div())

    if "position" not in bt_df.columns:
        warn("Kolom 'position' tidak ditemukan"); return
    if "strategy_return" not in bt_df.columns:
        warn("Kolom 'strategy_return' tidak ditemukan"); return

    long_ret  = bt_df[bt_df["position"] ==  1]["strategy_return"]
    short_ret = bt_df[bt_df["position"] == -1]["strategy_return"]

    print(f"  {'':10} {'Count':>8} {'Sum%':>10} {'Mean%':>10} {'Std%':>10}")
    print(f"  {'─'*10} {'─'*8} {'─'*10} {'─'*10} {'─'*10}")
    for label, r in [("LONG", long_ret), ("SHORT", short_ret)]:
        if len(r) == 0: continue
        print(f"  {label:<10} {len(r):>8,} {r.sum()*100:>+9.2f}% "
              f"{r.mean()*100:>+9.4f}% {r.std()*100:>9.4f}%")

    print(sep())
    if len(long_ret) > 0 and len(short_ret) > 0:
        if long_ret.mean() > 0 and short_ret.mean() < 0:
            err("SHORT bars rata-rata merugi — SHORT mungkin kontra-produktif")
        elif long_ret.mean() > 0 and short_ret.mean() > 0:
            ok("Keduanya LONG dan SHORT rata-rata profit")
        elif long_ret.mean() < 0:
            err("LONG bars rata-rata merugi — cek signal logic")
        else:
            warn("SHORT mean positif tapi perlu monitoring")


# ════════════════════════════════════════════════════════════════════
#  AUDIT 7: Drawdown Calculation Correctness
# ════════════════════════════════════════════════════════════════════

def audit_7_drawdown(risk_df):
    print(f"\n{div()}")
    print("  AUDIT 7 — Drawdown Calculation Correctness")
    print(div())

    eq     = risk_df["equity"]
    dd_col = risk_df["drawdown"]

    roll_max  = eq.cummax()
    dd_recalc = (eq - roll_max) / roll_max

    max_dd_reported = dd_col.min()
    max_dd_recalc   = dd_recalc.min()
    diff            = abs(max_dd_reported - max_dd_recalc)

    print(f"  Max DD (reported)    : {max_dd_reported*100:.4f}%")
    print(f"  Max DD (recalculated): {max_dd_recalc*100:.4f}%")
    print(f"  Difference           : {diff*100:.6f}%")

    if diff < 0.001:
        ok("Drawdown calculation konsisten (diff < 0.1%)")
    else:
        err(f"Drawdown tidak konsisten! diff={diff*100:.4f}%")

    pos_dd = (dd_col > 0.001).sum()
    if pos_dd > 0:
        err(f"Ada {pos_dd} bar dengan drawdown positif (impossible!)")
    else:
        ok("Tidak ada drawdown positif")


# ════════════════════════════════════════════════════════════════════
#  AUDIT 8: Backtest vs Risk Engine CAGR Gap
# ════════════════════════════════════════════════════════════════════

def audit_8_cagr_gap(bt_df, risk_df):
    print(f"\n{div()}")
    print("  AUDIT 8 — Backtest vs Risk Engine CAGR Gap")
    print(div())

    n_years    = len(bt_df) / BARS_PER_YEAR
    bt_final   = bt_df["equity"].iloc[-1]
    risk_final = risk_df["equity"].iloc[-1]
    bt_cagr    = (bt_final   / INITIAL_EQ) ** (1 / n_years) - 1
    risk_cagr  = (risk_final / INITIAL_EQ) ** (1 / n_years) - 1
    gap        = bt_cagr - risk_cagr

    print(f"  Backtest CAGR    : {bt_cagr*100:>+8.2f}%  (final=${bt_final:,.0f})")
    print(f"  Risk Eng CAGR    : {risk_cagr*100:>+8.2f}%  (final=${risk_final:,.0f})")
    print(f"  Gap              : {gap*100:>+8.2f}%")
    print(sep())

    if gap < -10:
        ok(f"Gap CAGR {gap*100:.1f}% — risk engine outperform backtest (vol targeting aktif)")
    elif gap > 50:
        err(f"Gap CAGR {gap*100:.1f}% — risk engine underperform signifikan")
        warn("  Cek kill switch terlalu sering trigger atau leverage cap terlalu kecil")
    elif gap > 15:
        warn(f"Gap CAGR {gap*100:.1f}% — wajar karena risk engine punya kill switch")
    else:
        ok(f"Gap CAGR {gap*100:.1f}% dalam range wajar")

    if "position" in risk_df.columns:
        if "leverage_used" in risk_df.columns:
            risk_active = (risk_df["leverage_used"] > 0).sum()
        elif "position_size" in risk_df.columns:
            risk_active = (risk_df["position_size"] > 0).sum()
        else:
            risk_active = (risk_df["position"] != 0).sum()
        bt_active = (bt_df["position"] != 0).sum()
        coverage  = risk_active / bt_active if bt_active > 0 else 0
        print(f"\n  Backtest active bars : {bt_active:,}")
        print(f"  Risk active bars     : {risk_active:,}  ({coverage*100:.1f}% coverage)")
        if coverage < 0.8:
            warn(f"Coverage {coverage*100:.1f}% — risk engine missing some held bars")
        else:
            ok(f"Coverage {coverage*100:.1f}% — risk engine mengikuti state machine dengan baik")


# ════════════════════════════════════════════════════════════════════
#  AUDIT 9: Lookahead Bias Check
# ════════════════════════════════════════════════════════════════════

def audit_9_lookahead(bt_df):
    print(f"\n{div()}")
    print("  AUDIT 9 — Lookahead Bias Check")
    print(div())

    if "market_return" in bt_df.columns and "strategy_return" in bt_df.columns:
        if "position" in bt_df.columns:
            expected = bt_df["position"] * bt_df["market_return"]
            actual   = bt_df["strategy_return"]
            diff     = (expected - actual).abs().max()
            if diff < 1e-10:
                ok("strategy_return = position × market_return ✓ (no lookahead)")
            else:
                err(f"Mismatch strategy_return! max_diff={diff:.2e}")
        else:
            warn("Kolom 'position' tidak ada untuk validasi")
    else:
        warn("Kolom market_return/strategy_return tidak ditemukan")

    if "market_return" in bt_df.columns:
        last_mr = bt_df["market_return"].iloc[-1]
        if abs(last_mr) < 1e-10:
            ok("Last bar market_return = 0 ✓ (shift(-1) correct)")
        else:
            err(f"Last bar market_return = {last_mr:.6f} (should be 0!)")


# ════════════════════════════════════════════════════════════════════
#  AUDIT 10: Yearly Return Plausibility  [FIXED: threshold 1000%]
# ════════════════════════════════════════════════════════════════════

def audit_10_yearly(risk_df):
    print(f"\n{div()}")
    print("  AUDIT 10 — Yearly Return Plausibility")
    print(div())

    risk_df      = risk_df.copy()
    risk_df["year"] = risk_df["timestamp"].dt.year

    print(f"  {'Year':<6} {'Start':>10} {'End':>10} {'Return':>10} {'MaxDD':>9} {'Active%':>9}")
    print(f"  {'─'*6} {'─'*10} {'─'*10} {'─'*10} {'─'*9} {'─'*9}")

    anomaly_found = False
    for year, grp in risk_df.groupby("year"):
        s_eq   = grp["equity"].iloc[0]
        e_eq   = grp["equity"].iloc[-1]
        yr_ret = (e_eq - s_eq) / s_eq * 100
        yr_dd  = grp["drawdown"].min() * 100
        active = (grp["signal"] != "NONE").mean() * 100

        flag = ""
        if yr_ret > 1000:   # FIX: 2000% → 1000% (lebih sensitif)
            flag = " ← [WARN] sangat tinggi"
            anomaly_found = True
        elif yr_ret < -80:
            flag = " ← [WARN] sangat rendah"
            anomaly_found = True

        print(f"  {year:<6} ${s_eq:>9,.0f} ${e_eq:>9,.0f} "
              f"{yr_ret:>+9.1f}% {yr_dd:>8.1f}% {active:>8.1f}%{flag}")

    print(sep())
    if anomaly_found:
        warn("Ada tahun dengan return ekstrem — perlu cek position sizing")
        warn("  Return >1000% biasanya dari leverage tinggi di low-vol period (e.g. 2017)")
    else:
        ok("Yearly returns dalam range plausible")


# ════════════════════════════════════════════════════════════════════
#  AUDIT 11: TIER2 Resume Spike — BARU
# ════════════════════════════════════════════════════════════════════

def audit_11_tier2_spike(risk_df):
    print(f"\n{div()}")
    print("  AUDIT 11 — TIER2 Resume Spike Check  [FIXED v2]")
    print(div())
    print("  Cek apakah shadow equity terakumulasi tanpa batas saat TIER2 paused")
    print("  dan menyebabkan spike besar saat resume (root cause +65.58% di audit lama)")
    print(sep())

    if "kill_switch_active" not in risk_df.columns:
        warn("Kolom kill_switch_active tidak ada — skip"); return

    ks      = risk_df["kill_switch_active"].values
    eq      = risk_df["equity"].values
    ts_arr  = risk_df["timestamp"]

    SPIKE_THRESHOLD = 0.20  # 20% = suspicious
    MAX_ACCEPTABLE  = 0.35  # >35% = bug pasti
    EXPECTED_CAP    = 0.12  # ≤12% = sesuai TIER2_GAIN_CAP RECOMMENDED (post-fix)
    # FIX v3: add epsilon tolerance untuk floating point boundary
    # Cap di risk_engine set shadow = cur × 1.12 → equity_return = exactly 0.12
    # Tapi floating point: 0.12000000001 > 0.12 → false violation
    # Tolerance 1e-6 (0.0001%) aman karena bukan nilai meaningful
    CAP_EPSILON     = 1e-6

    spikes = []
    for i in range(1, len(ks)):
        # FIX v2: gunakan integer comparison (ks adalah 0/1 int, bukan bool)
        if int(ks[i-1]) == 1 and int(ks[i]) == 0:   # resume event
            jump = (eq[i] - eq[i-1]) / eq[i-1] if eq[i-1] > 0 else 0
            spikes.append({
                "ts":   ts_arr.iloc[i].strftime("%Y-%m-%d"),
                "prev": eq[i-1],
                "cur":  eq[i],
                "jump": jump,
            })

    if not spikes:
        ok("Tidak ada resume event ditemukan")
        return

    print(f"  Total resume events : {len(spikes)}")
    print(f"  Spike threshold     : >{SPIKE_THRESHOLD*100:.0f}% per resume")
    print(f"  Expected cap (V6)   : ≤{EXPECTED_CAP*100:.0f}% (TIER2_GAIN_CAP RECOMMENDED)")
    print(sep())

    has_bug      = False
    cap_violated = False
    max_spike    = 0.0
    for s in spikes:
        flag     = ""
        abs_jump = abs(s["jump"])
        if abs_jump > MAX_ACCEPTABLE:
            flag      = "  ← ❌ BUG (TIER2 shadow tanpa cap)"
            has_bug   = True
        elif abs_jump > EXPECTED_CAP + CAP_EPSILON:
            flag      = "  ← [WARN] tinggi (>cap, perlu cek)"
            cap_violated = True
        elif abs_jump > SPIKE_THRESHOLD:
            flag      = "  ← [WARN] tinggi"
        max_spike = max(max_spike, abs_jump)
        # FIX v2: signed jump display
        sign = "+" if s["jump"] >= 0 else ""
        print(f"  {s['ts']}  ${s['prev']:>12,.0f} → ${s['cur']:>12,.0f}"
              f"  jump={sign}{s['jump']*100:>6.2f}%{flag}")

    print(sep())
    if has_bug:
        err(f"TIER2 spike bug TERKONFIRMASI! Max resume jump: {max_spike*100:.2f}%")
        err("  Shadow equity terakumulasi tanpa batas saat TIER2 paused")
        err("  Fix: TIER2_GAIN_CAP + TIER2_LOSS_CAP di risk_engine_v6.py")
        err("  Update paper_trader.py: BAR_GAIN_LIMIT=0.25, TIER2_GAIN_CAP=0.12")
    elif cap_violated:
        warn(f"Resume jumps ada yang > expected cap {EXPECTED_CAP*100:.0f}%")
        warn("  Cek apakah resume jump cap sudah diapply di risk_engine_v6.py")
    elif max_spike > SPIKE_THRESHOLD:
        warn(f"Resume jumps ada yang > {SPIKE_THRESHOLD*100:.0f}% tapi ≤ {EXPECTED_CAP*100:.0f}%")
        warn("  Resume cap sudah berfungsi — nilai masih sedikit tinggi, monitor")
    else:
        ok(f"Semua resume jumps dalam range normal (max {max_spike*100:.2f}% ≤ {EXPECTED_CAP*100:.0f}%)")
        ok("TIER2 shadow cap + resume jump cap sudah berfungsi dengan benar")


# ════════════════════════════════════════════════════════════════════
#  FINAL VERDICT
# ════════════════════════════════════════════════════════════════════

def print_verdict(risk_df):
    print(f"\n{div()}")
    print("  FINAL SYSTEM VERDICT")
    print(div())

    n_years  = len(risk_df) / BARS_PER_YEAR
    final_eq = risk_df["equity"].iloc[-1]
    cagr     = (final_eq / INITIAL_EQ) ** (1 / n_years) - 1
    eq_ret   = risk_df["equity_return"].fillna(0)
    sharpe   = (eq_ret.mean() / eq_ret.std()) * np.sqrt(BARS_PER_YEAR) \
               if eq_ret.std() > 0 else 0
    max_dd   = risk_df["drawdown"].min()

    neg_ret = eq_ret[eq_ret < 0]
    sortino = float((eq_ret.mean() / neg_ret.std()) * np.sqrt(BARS_PER_YEAR)) \
              if len(neg_ret) > 0 and neg_ret.std() > 0 else 0.0
    calmar  = float((cagr * 100) / abs(max_dd * 100)) if max_dd != 0 else 0.0

    # ── FIX v2: Max single bar — exclude zero-return TIER2 flat bars ─────────
    # Bug lama: eq.pct_change() mencakup SEMUA bar termasuk TIER2 paused (return=0)
    #           Padahal TIER2 flat bars bukan "active bar" — equity-nya frozen
    #           Tidak ada efek pada max positif, tapi menjaga konsistensi dengan Audit 1 & 4
    eq      = risk_df["equity"]
    if "leverage_used" in risk_df.columns:
        active_mask = risk_df["leverage_used"] > 0
        eq_pct_all  = eq.pct_change().fillna(0)
        max_bar     = eq_pct_all[active_mask].max() * 100
    elif "equity_return" in risk_df.columns:
        eq_ret_full = risk_df["equity_return"].fillna(0)
        max_bar     = eq_ret_full[eq_ret_full != 0].max() * 100
    else:
        max_bar = eq.pct_change().fillna(0).max() * 100

    def v(val, t1, t2, l1, l2, lfail):
        return l1 if val >= t1 else (l2 if val >= t2 else lfail)

    print(f"  CAGR        : {cagr*100:>+8.2f}%   " + v(cagr, 0.50, 0.25, "PASS >50%",  "PASS >25%", "FAIL <25%"))
    print(f"  Sharpe      : {sharpe:>8.4f}   " + ("PASS" if sharpe  > 1.0 else "WARN") + " (>1.0)")
    print(f"  Sortino     : {sortino:>8.4f}   " + ("PASS" if sortino > 1.0 else "WARN") + " (>1.0)")
    print(f"  Calmar      : {calmar:>8.4f}   " + ("PASS" if calmar  > 1.0 else "WARN") + " (>1.0)")
    print(f"  Max DD      : {max_dd*100:>8.2f}%   " + v(max_dd, -0.25, -0.35, "PASS <25%", "PASS <35%", "WARN >35%"))
    print(f"  Max Bar     : {max_bar:>8.2f}%   " + ("PASS" if max_bar < 35 else "WARN") + " (<35%)")
    print(div())


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════

def run():
    print(f"\n{div()}")
    print("  FINAL COMPREHENSIVE AUDIT — BTC Hybrid Model  [V6 — All Bugs Fixed]")
    print(div())

    bt_df   = load(BACKTEST_PATH,  "backtest")
    risk_df = load(RISK_PATH,      "risk")
    sig_df  = load(SIGNALS_PATH,   "signals")

    audit_1_profit_factor(risk_df)
    audit_2_sharpe_plausibility(risk_df)
    audit_3_kill_switch(risk_df)
    audit_4_equity_continuity(risk_df)
    audit_5_signal_distribution(sig_df)
    audit_6_long_short_asymmetry(bt_df)
    audit_7_drawdown(risk_df)
    audit_8_cagr_gap(bt_df, risk_df)
    audit_9_lookahead(bt_df)
    audit_10_yearly(risk_df)
    audit_11_tier2_spike(risk_df)  # NEW
    print_verdict(risk_df)

    print(f"\n  AUDIT SELESAI\n{div()}\n")


if __name__ == "__main__":
    run()
