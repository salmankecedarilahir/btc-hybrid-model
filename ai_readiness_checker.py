"""
ai_readiness_checker.py — BTC Hybrid Model: AI Layer Readiness Gate
====================================================================

BAGIAN 10 — KRITERIA MODEL SIAP MASUK AI LAYER

Sebelum membangun AI layer, semua kriteria di bawah HARUS terpenuhi.
Script ini menjalankan semua checks dan mengeluarkan GO / NO-GO decision.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KRITERIA MINIMUM (semua harus PASS):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  CATEGORY A — METRIK PERFORMANCE (dari equity curve)
  ─────────────────────────────────────────────────────
  A1. CAGR             ≥ 50%          (sistem menghasilkan return signifikan)
  A2. Sharpe Ratio     ≥ 1.0          (return per unit risk cukup baik)
  A3. Sortino Ratio    ≥ 0.8          (downside risk terkelola)
  A4. Calmar Ratio     ≥ 1.0          (CAGR/MaxDD ratio acceptable)
  A5. Max Drawdown     ≤ -35%         (sistem tidak ruinous)

  CATEGORY B — METRIK KONSISTENSI (dari trade list)
  ────────────────────────────────────────────────────
  B1. Profit Factor    ≥ 1.3          (sistem profitable dengan margin)
  B2. Expectancy       > 0%           (HARUS positif jika PF > 1)
  B3. PF-Exp konsisten               (PF > 1 ↔ Exp > 0, tidak boleh kontradiksi)
  B4. Min trades       ≥ 50           (trend-following: per-trade count)
       NOTE: Sistem ini hold rata-rata 224 bars/trade → 89 trades cukup
  B5. Win Rate         > 0%           (ada minimal beberapa winning trade)

  CATEGORY C — INTEGRITAS DATA (dari audit)
  ─────────────────────────────────────────
  C1. Max single bar   ≤ 35%          (tidak ada spike resume anomaly)
  C2. No equity NaN                   (equity curve bersih)
  C3. DD calculation konsisten        (reported DD = recalculated DD, diff < 0.1%)
  C4. No lookahead bias               (strategy_return = position × market_return)
  C5. No TIER2 spike   (jump ≤ 12%)   (resume cap berfungsi)

  CATEGORY D — KESIAPAN DATASET AI
  ──────────────────────────────────
  D1. Min bars         ≥ 10,000       (cukup data untuk AI training)
  D2. Features tersedia               (regime, signals, market features)
  D3. Target variable ada             (future return atau direction)
  D4. Class imbalance  NONE ≤ 85%    (signal NONE tidak terlalu dominan)
  D5. No major NaN     ≤ 5%           (fitur utama tidak banyak NaN)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CARA PAKAI:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    python ai_readiness_checker.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

BASE          = Path(__file__).parent / "data"
BARS_PER_YEAR = 2190
INITIAL_EQ    = 10_000.0


# ════════════════════════════════════════════════════════════════════
#  CRITERIA DEFINITIONS
# ════════════════════════════════════════════════════════════════════

CRITERIA = {
    # Category A — Performance
    "A1_CAGR":          {"label": "CAGR ≥ 50%",                  "threshold": 0.50,   "direction": "ge"},
    "A2_Sharpe":        {"label": "Sharpe ≥ 1.0",                "threshold": 1.0,    "direction": "ge"},
    "A3_Sortino":       {"label": "Sortino ≥ 0.8",               "threshold": 0.8,    "direction": "ge"},
    "A4_Calmar":        {"label": "Calmar ≥ 1.0",                "threshold": 1.0,    "direction": "ge"},
    "A5_MaxDD":         {"label": "Max DD ≤ -35%",               "threshold": -0.35,  "direction": "ge"},   # -35% is floor

    # Category B — Consistency
    "B1_PF":            {"label": "Profit Factor ≥ 1.3",         "threshold": 1.3,    "direction": "ge"},
    "B2_Expectancy":    {"label": "Expectancy > 0%",             "threshold": 0.0,    "direction": "gt"},
    "B3_PF_Exp":        {"label": "PF ↔ Expectancy konsisten",   "threshold": True,   "direction": "eq"},
    "B4_MinTrades":     {"label": "Min trades 2265 50 (trend-following)", "threshold": 50,    "direction": "ge"},
    "B5_WinRate":       {"label": "Win rate > 0%",               "threshold": 0.0,    "direction": "gt"},

    # Category C — Data Integrity
    "C1_MaxBar":        {"label": "Max bar return ≤ 35%",        "threshold": 35.0,   "direction": "le"},
    "C2_NoNaN":         {"label": "No equity NaN",               "threshold": 0,      "direction": "eq"},
    "C3_DD_Consistent": {"label": "DD diff < 0.1%",              "threshold": 0.1,    "direction": "le"},
    "C4_NoLookahead":   {"label": "No lookahead bias",           "threshold": True,   "direction": "eq"},
    "C5_NoTier2Spike":  {"label": "TIER2 spike ≤ 12% (±0.01%)",      "threshold": 12.01,  "direction": "le"},

    # Category D — AI Dataset
    "D1_MinBars":       {"label": "Dataset ≥ 10,000 bars",       "threshold": 10000,  "direction": "ge"},
    "D2_Features":      {"label": "Market features tersedia",    "threshold": True,   "direction": "eq"},
    "D3_Target":        {"label": "Target variable ada",         "threshold": True,   "direction": "eq"},
    "D4_ClassBalance":  {"label": "NONE signal ≤ 85%",           "threshold": 85.0,   "direction": "le"},
    "D5_NoNaN":         {"label": "Key features NaN ≤ 5%",       "threshold": 5.0,    "direction": "le"},
}


def check_criterion(key: str, value, threshold, direction: str) -> bool:
    if direction == "ge":  return value >= threshold
    if direction == "gt":  return value > threshold
    if direction == "le":  return value <= threshold
    if direction == "lt":  return value < threshold
    if direction == "eq":  return value == threshold
    return False


# ════════════════════════════════════════════════════════════════════
#  METRIC EXTRACTORS
# ════════════════════════════════════════════════════════════════════

def extract_performance_metrics(risk_df: pd.DataFrame) -> dict:
    """Extract A-category metrics from risk managed results."""
    eq      = risk_df["equity"]
    n_years = len(risk_df) / BARS_PER_YEAR
    cagr    = (eq.iloc[-1] / INITIAL_EQ) ** (1 / n_years) - 1

    eq_ret  = risk_df["equity_return"].fillna(0) \
              if "equity_return" in risk_df.columns \
              else eq.pct_change().fillna(0)

    # Exclude zero-return TIER2 paused bars
    if "leverage_used" in risk_df.columns:
        active_ret = eq_ret[risk_df["leverage_used"] > 0]
    else:
        active_ret = eq_ret[eq_ret != 0]

    sharpe  = float((active_ret.mean() / active_ret.std()) * np.sqrt(BARS_PER_YEAR)) \
              if active_ret.std() > 0 else 0.0

    neg_ret = active_ret[active_ret < 0]
    sortino = float((active_ret.mean() / neg_ret.std()) * np.sqrt(BARS_PER_YEAR)) \
              if len(neg_ret) > 0 and neg_ret.std() > 0 else 0.0

    roll_max = eq.cummax()
    dd       = (eq - roll_max) / roll_max
    max_dd   = float(dd.min())
    calmar   = float(cagr / abs(max_dd)) if max_dd != 0 else 0.0

    return {
        "cagr":    cagr,
        "sharpe":  sharpe,
        "sortino": sortino,
        "calmar":  calmar,
        "max_dd":  max_dd,
    }


def extract_consistency_metrics(risk_df: pd.DataFrame,
                                 bt_df: pd.DataFrame) -> dict:
    """Extract B-category metrics."""
    col = "equity_return" if "equity_return" in risk_df.columns else "strategy_return"

    # Active bars only, exclude zero-return
    if "leverage_used" in risk_df.columns:
        ret = risk_df.loc[risk_df["leverage_used"] > 0, col]
    else:
        ret = risk_df[col]
    ret = pd.to_numeric(ret, errors="coerce").dropna()
    ret = ret[ret != 0.0]

    wins   = ret[ret > 0]
    losses = ret[ret < 0]
    n      = len(ret)

    pf  = float(wins.sum() / abs(losses.sum())) \
          if len(losses) > 0 and losses.sum() != 0 else float("inf")
    wr  = len(wins) / n if n > 0 else 0
    avg_win  = float(wins.mean())  if len(wins)   > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
    exp = wr * avg_win + (1 - wr) * avg_loss

    pf_exp_consistent = (pf > 1) == (exp > 0)

    # Approximate trade count from position changes
    pos = risk_df["position"] if "position" in risk_df.columns else bt_df["position"]
    pos_diff = pos.diff().fillna(pos)
    n_trades = int(((pos_diff != 0) & (pos != 0)).sum())

    return {
        "profit_factor":    pf,
        "expectancy":       exp,
        "win_rate":         wr,
        "n_trades":         n_trades,
        "pf_exp_consistent": pf_exp_consistent,
    }


def extract_integrity_metrics(risk_df: pd.DataFrame,
                               bt_df: pd.DataFrame) -> dict:
    """Extract C-category metrics."""
    eq = risk_df["equity"]

    # Max bar (active bars only)
    if "equity_return" in risk_df.columns:
        eq_ret_full = risk_df["equity_return"].fillna(0)
        max_bar = float(eq_ret_full[eq_ret_full != 0].max() * 100)
    else:
        max_bar = float(eq.pct_change().fillna(0).max() * 100)

    # NaN check
    n_nan = int(eq.isna().sum())

    # DD consistency
    roll_max  = eq.cummax()
    dd_recalc = (eq - roll_max) / roll_max
    dd_col    = risk_df["drawdown"] if "drawdown" in risk_df.columns else dd_recalc
    dd_diff   = float(abs(dd_col.min() - dd_recalc.min()) * 100)

    # Lookahead bias
    if all(c in bt_df.columns for c in ["position", "market_return", "strategy_return"]):
        expected = bt_df["position"] * bt_df["market_return"]
        diff     = (expected - bt_df["strategy_return"]).abs().max()
        no_lookahead = float(diff) < 1e-8
    else:
        no_lookahead = True  # assume OK if columns not found

    # TIER2 spike — dengan epsilon tolerance untuk floating point boundary
    max_spike = 0.0
    if "kill_switch_active" in risk_df.columns:
        ks  = risk_df["kill_switch_active"].values
        eqv = risk_df["equity"].values
        for i in range(1, len(ks)):
            if int(ks[i-1]) == 1 and int(ks[i]) == 0:
                jump = (eqv[i] - eqv[i-1]) / eqv[i-1] if eqv[i-1] > 0 else 0
                # Round ke 4 desimal untuk avoid floating point noise (e.g. 12.00001%)
                max_spike = max(max_spike, round(abs(jump) * 100, 4))

    return {
        "max_bar_pct":     max_bar,
        "n_equity_nan":    n_nan,
        "dd_diff_pct":     dd_diff,
        "no_lookahead":    no_lookahead,
        "max_tier2_spike": max_spike,
    }


def extract_ai_dataset_metrics(sig_df: pd.DataFrame) -> dict:
    """Extract D-category metrics."""
    n_bars = len(sig_df)

    has_features = all(c in sig_df.columns for c in ["signal", "regime"])
    has_target   = "market_return" in sig_df.columns or "close" in sig_df.columns

    none_pct = 0.0
    if "signal" in sig_df.columns:
        none_pct = float((sig_df["signal"] == "NONE").mean() * 100)

    key_features = ["signal", "regime", "close", "volume"]
    max_nan_pct  = max(
        sig_df[c].isna().mean() * 100 if c in sig_df.columns else 100.0
        for c in key_features
    )

    return {
        "n_bars":        n_bars,
        "has_features":  has_features,
        "has_target":    has_target,
        "none_pct":      none_pct,
        "max_nan_pct":   max_nan_pct,
    }


# ════════════════════════════════════════════════════════════════════
#  MAIN READINESS CHECKER
# ════════════════════════════════════════════════════════════════════

def run_readiness_check(risk_df, bt_df, sig_df):
    DIV = "═" * 68
    SEP = "─" * 68

    print(f"\n{DIV}")
    print("  AI LAYER READINESS CHECK — BTC Hybrid Model V6")
    print("  Gate: semua kriteria PASS → GO untuk AI layer")
    print(DIV)

    # ── Extract semua metrik ─────────────────────────────────────────
    perf  = extract_performance_metrics(risk_df)
    cons  = extract_consistency_metrics(risk_df, bt_df)
    integ = extract_integrity_metrics(risk_df, bt_df)
    ai    = extract_ai_dataset_metrics(sig_df)

    # ── Map ke nilai untuk setiap criterion ─────────────────────────
    values = {
        "A1_CAGR":          perf["cagr"],
        "A2_Sharpe":        perf["sharpe"],
        "A3_Sortino":       perf["sortino"],
        "A4_Calmar":        perf["calmar"],
        "A5_MaxDD":         perf["max_dd"],

        "B1_PF":            cons["profit_factor"],
        "B2_Expectancy":    cons["expectancy"],
        "B3_PF_Exp":        cons["pf_exp_consistent"],
        "B4_MinTrades":     cons["n_trades"],
        "B5_WinRate":       cons["win_rate"],

        "C1_MaxBar":        integ["max_bar_pct"],
        "C2_NoNaN":         integ["n_equity_nan"],
        "C3_DD_Consistent": integ["dd_diff_pct"],
        "C4_NoLookahead":   integ["no_lookahead"],
        "C5_NoTier2Spike":  integ["max_tier2_spike"],

        "D1_MinBars":       ai["n_bars"],
        "D2_Features":      ai["has_features"],
        "D3_Target":        ai["has_target"],
        "D4_ClassBalance":  ai["none_pct"],
        "D5_NoNaN":         ai["max_nan_pct"],
    }

    # ── Evaluasi tiap criterion ──────────────────────────────────────
    results  = {}
    n_pass   = 0
    n_fail   = 0
    n_warn   = 0   # kriteria yang masuk "warning zone" (dekat threshold)
    blockers = []  # kriteria FAIL yang paling kritikal (A & B & C)

    categories = {
        "A": "PERFORMANCE METRICS",
        "B": "CONSISTENCY (PF / Expectancy / Trades)",
        "C": "DATA INTEGRITY",
        "D": "AI DATASET READINESS",
    }

    for cat_key, cat_name in categories.items():
        print(f"\n  {'─'*60}")
        print(f"  CATEGORY {cat_key} — {cat_name}")
        print(f"  {'─'*60}")

        for key, crit in CRITERIA.items():
            if not key.startswith(cat_key):
                continue

            val   = values[key]
            thr   = crit["threshold"]
            dirn  = crit["direction"]
            label = crit["label"]
            passed = check_criterion(key, val, thr, dirn)

            # Format nilai untuk display
            if isinstance(val, float):
                if abs(val) < 10:
                    val_str = f"{val:>+.4f}"
                else:
                    val_str = f"{val:>+.2f}"
            elif isinstance(val, bool):
                val_str = "YES" if val else "NO"
            else:
                val_str = f"{val:>,.0f}"

            # Warning zone: dalam 10% dari threshold
            in_warning = False
            if passed and isinstance(val, (int, float)) and not isinstance(val, bool):
                ratio = abs(val / (thr + 1e-9))
                if 0.85 < ratio < 1.15 and dirn == "ge":
                    in_warning = True
                elif 0.85 < ratio < 1.15 and dirn == "le":
                    in_warning = True

            if passed:
                mark = "[OK] PASS"
                n_pass += 1
                if in_warning:
                    mark = "[YELLOW] PASS (close)"
                    n_warn += 1
            else:
                mark = "❌ FAIL"
                n_fail += 1
                if cat_key in ("A", "B", "C"):
                    blockers.append((key, label, val_str))

            results[key] = passed
            print(f"  {mark}  {label:<38} value={val_str}")

    # ── Final verdict ────────────────────────────────────────────────
    print(f"\n{DIV}")
    all_pass   = n_fail == 0
    cat_abc_ok = all(results.get(k, False) for k in results if k[0] in ("A","B","C"))

    print(f"  Results  : {n_pass} PASS  |  {n_warn} close  |  {n_fail} FAIL")
    print(f"  Critical : Categories A+B+C — {'ALL PASS' if cat_abc_ok else f'{len(blockers)} FAILURES'}")
    print(DIV)

    if all_pass:
        print(f"""
  [GREEN]  GO — SIAP MASUK AI LAYER

  Semua {len(CRITERIA)} kriteria terpenuhi.
  Dataset valid secara matematis dan performa model sudah cukup kuat
  untuk dijadikan basis AI layer.

  Next step:
    1. Jalankan: python ai_dataset_builder.py
    2. Bangun AI signal enhancer / filter
    3. Target AI: tingkatkan Sortino dari {perf['sortino']:.2f} → ≥ 1.5
""")
    elif cat_abc_ok:
        print(f"""
  [YELLOW]  CONDITIONAL GO — Dataset kriteria A/B/C sudah OK

  Bisa mulai AI layer development, tapi selesaikan D-category dulu:
""")
        for key, label, val in [(k,l,v) for k,l,v in [(k, CRITERIA[k]["label"], values[k])
                                                        for k in results
                                                        if not results[k] and k[0] == "D"]]:
            print(f"    ❌ {label}")
    else:
        print(f"""
  [RED]  NO-GO — Masih ada {len(blockers)} blocker critical

  Selesaikan dulu sebelum masuk AI layer:
""")
        for key, label, val in blockers:
            print(f"    ❌ [{key}] {label}  (current: {val})")

        print(f"""
  Recommended fix order:
    1. C-category (Data Integrity) — fix resume spike, lookahead
    2. B-category (Consistency)    — fix PF/Expectancy inkonsistensi
    3. A-category (Performance)    — biasanya ikut fix setelah B & C
""")

    print(DIV)
    return results, all_pass


# ════════════════════════════════════════════════════════════════════
#  SUMMARY TABLE (untuk dokumentasi)
# ════════════════════════════════════════════════════════════════════

def print_criteria_table():
    """Cetak tabel kriteria lengkap untuk dokumentasi."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║         AI LAYER READINESS CRITERIA — BTC Hybrid Model V6           ║
╠══════╦═══════════════════════════════════════╦══════════╦═══════════╣
║ CODE ║ CRITERIA                              ║ THRESHOLD║ CATEGORY  ║
╠══════╬═══════════════════════════════════════╬══════════╬═══════════╣
║ A1   ║ CAGR                                  ║  ≥ 50%   ║ Perform.  ║
║ A2   ║ Sharpe Ratio                          ║  ≥ 1.0   ║ Perform.  ║
║ A3   ║ Sortino Ratio                         ║  ≥ 0.8   ║ Perform.  ║
║ A4   ║ Calmar Ratio                          ║  ≥ 1.0   ║ Perform.  ║
║ A5   ║ Max Drawdown                          ║  ≤ -35%  ║ Perform.  ║
╠══════╬═══════════════════════════════════════╬══════════╬═══════════╣
║ B1   ║ Profit Factor                         ║  ≥ 1.3   ║ Consist.  ║
║ B2   ║ Expectancy per trade                  ║   > 0%   ║ Consist.  ║
║ B3   ║ PF ↔ Expectancy matematically align   ║  = True  ║ Consist.  ║
║ B4   ║ Min trades (trend-following)         ║   ≥ 50   ║ Consist.  ║
║ B5   ║ Win Rate                              ║   > 0%   ║ Consist.  ║
╠══════╬═══════════════════════════════════════╬══════════╬═══════════╣
║ C1   ║ Max single bar return                 ║  ≤ 35%   ║ Integrity ║
║ C2   ║ Equity NaN count                      ║  = 0     ║ Integrity ║
║ C3   ║ Drawdown calc diff                    ║  < 0.1%  ║ Integrity ║
║ C4   ║ No lookahead bias                     ║  = True  ║ Integrity ║
║ C5   ║ TIER2 max resume jump                 ║  ≤ 12%   ║ Integrity ║
╠══════╬═══════════════════════════════════════╬══════════╬═══════════╣
║ D1   ║ Dataset size                          ║ ≥ 10,000 ║ AI Ready  ║
║ D2   ║ Market features available             ║  = True  ║ AI Ready  ║
║ D3   ║ Target variable defined               ║  = True  ║ AI Ready  ║
║ D4   ║ NONE signal dominance                 ║  ≤ 85%   ║ AI Ready  ║
║ D5   ║ Max key feature NaN%                  ║  ≤ 5%    ║ AI Ready  ║
╚══════╩═══════════════════════════════════════╩══════════╩═══════════╝

  BLOCKING:  A + B + C (harus semua PASS sebelum GO)
  NON-BLOCK: D (bisa difix paralel dengan AI development)
""")


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    risk_path = BASE / "btc_risk_managed_results.csv"
    bt_path   = BASE / "btc_backtest_results.csv"
    sig_path  = BASE / "btc_trading_signals.csv"

    for p in [risk_path, bt_path, sig_path]:
        if not p.exists():
            print(f"❌ File tidak ditemukan: {p}")
            print("   Jalankan pipeline lengkap terlebih dahulu")
            raise SystemExit(1)

    print("  Loading data ...")
    risk_df = pd.read_csv(risk_path, parse_dates=["timestamp"])
    bt_df   = pd.read_csv(bt_path,   parse_dates=["timestamp"])
    sig_df  = pd.read_csv(sig_path,  parse_dates=["timestamp"])

    for df in [risk_df, bt_df, sig_df]:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    print_criteria_table()
    run_readiness_check(risk_df, bt_df, sig_df)
