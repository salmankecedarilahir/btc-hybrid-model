"""
position_size_checker.py — BTC Hybrid Model: Position Size & Risk Auditor
==========================================================================

BAGIAN 8 — POSITION SIZE CHECK

Audit sebelumnya menemukan max bar return +45% (post-fix ~32%).
Module ini memeriksa:

  1. Position size per bar (leverage exposure)
  2. Risk per trade (berapa % equity yang di-risk)
  3. Max single bar gain/loss distribution
  4. Leverage distribution (histogram)
  5. TIER2 cap verification (max resume jump ≤ 12%)
  6. VaR (Value at Risk) — berapa max loss di 1% worst bars

Cara pakai:
    python position_size_checker.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

BASE           = Path(__file__).parent / "data"
INITIAL_EQ     = 10_000.0
BARS_PER_YEAR  = 2190

# Thresholds dari risk_engine_v6 RECOMMENDED preset
MAX_BAR_GAIN   = 0.25    # 25%
MAX_BAR_LOSS   = -0.12   # -12%
TIER2_GAIN_CAP = 0.12    # 12% max resume jump


# ════════════════════════════════════════════════════════════════════
#  POSITION SIZE CHECKER
# ════════════════════════════════════════════════════════════════════

class PositionSizeChecker:
    """
    Audit position sizing dan risk exposure dari risk-managed results.

    Expected input: btc_risk_managed_results.csv (output risk_engine_v6)
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy().sort_values("timestamp").reset_index(drop=True)
        self._detect_columns()

    def _detect_columns(self):
        """Detect kolom yang tersedia — risk_managed_results punya lebih banyak kolom."""
        self.has_leverage    = "leverage_used"    in self.df.columns
        self.has_eq_return   = "equity_return"    in self.df.columns
        self.has_kill_switch = "kill_switch_active" in self.df.columns
        self.has_shadow      = "shadow_equity"    in self.df.columns

        log.info("Columns available: leverage=%s, eq_return=%s, kill_switch=%s, shadow=%s",
                 self.has_leverage, self.has_eq_return,
                 self.has_kill_switch, self.has_shadow)

    def _ok(self, msg):  print(f"  [OK] {msg}")
    def _warn(self, msg): print(f"  [WARN]️  {msg}")
    def _err(self, msg):  print(f"  ❌ {msg}")

    # ─── CHECK 1: Leverage Distribution ───────────────────────────────────────
    def check_leverage(self) -> dict:
        print("\n  ── CHECK 1: Leverage Distribution ──")

        if not self.has_leverage:
            self._warn("Kolom leverage_used tidak ada — skip")
            return {}

        lev = self.df["leverage_used"]
        active_lev = lev[lev > 0]

        if len(active_lev) == 0:
            self._warn("Tidak ada bar aktif dengan leverage > 0")
            return {}

        stats = {
            "avg_leverage":    round(float(active_lev.mean()), 4),
            "max_leverage":    round(float(active_lev.max()),  4),
            "min_leverage":    round(float(active_lev.min()),  4),
            "p95_leverage":    round(float(active_lev.quantile(0.95)), 4),
            "p99_leverage":    round(float(active_lev.quantile(0.99)), 4),
            "n_active_bars":   len(active_lev),
        }

        print(f"    Avg leverage      : {stats['avg_leverage']:.4f}x")
        print(f"    Max leverage      : {stats['max_leverage']:.4f}x")
        print(f"    Min leverage      : {stats['min_leverage']:.4f}x")
        print(f"    95th pct leverage : {stats['p95_leverage']:.4f}x")
        print(f"    99th pct leverage : {stats['p99_leverage']:.4f}x")

        if stats["max_leverage"] > 5.0:
            self._err(f"Max leverage {stats['max_leverage']:.2f}x melebihi preset limit (5x)")
        elif stats["max_leverage"] > 3.0:
            self._warn(f"Max leverage {stats['max_leverage']:.2f}x cukup tinggi — monitor")
        else:
            self._ok(f"Leverage dalam range aman (max {stats['max_leverage']:.2f}x)")

        if stats["avg_leverage"] > 2.0:
            self._warn(f"Avg leverage {stats['avg_leverage']:.2f}x — pertimbangkan turunkan TARGET_VOL")
        else:
            self._ok(f"Avg leverage {stats['avg_leverage']:.2f}x — dalam range wajar")

        return stats

    # ─── CHECK 2: Single Bar Return Distribution ──────────────────────────────
    def check_bar_returns(self) -> dict:
        print("\n  ── CHECK 2: Single Bar Return Distribution ──")

        if self.has_eq_return:
            bar_ret = self.df["equity_return"].fillna(0)
        else:
            bar_ret = self.df["equity"].pct_change().fillna(0)

        # Hanya active bars
        if self.has_leverage:
            active_mask = self.df["leverage_used"] > 0
        else:
            active_mask = bar_ret != 0.0

        active_ret = bar_ret[active_mask]
        n_active   = len(active_ret)

        if n_active == 0:
            self._warn("Tidak ada active bar")
            return {}

        max_gain = float(active_ret.max())
        max_loss = float(active_ret.min())
        avg_ret  = float(active_ret.mean())
        std_ret  = float(active_ret.std())

        # VaR (1% worst)
        var_1pct = float(active_ret.quantile(0.01))
        # CVaR (Expected Shortfall)
        cvar_1pct = float(active_ret[active_ret <= var_1pct].mean())

        stats = {
            "max_gain_pct":   round(max_gain * 100, 4),
            "max_loss_pct":   round(max_loss * 100, 4),
            "avg_return_pct": round(avg_ret  * 100, 6),
            "std_return_pct": round(std_ret  * 100, 4),
            "var_1pct":       round(var_1pct * 100, 4),
            "cvar_1pct":      round(cvar_1pct* 100, 4),
        }

        print(f"    Max single bar GAIN  : {stats['max_gain_pct']:>+8.4f}%")
        print(f"    Max single bar LOSS  : {stats['max_loss_pct']:>+8.4f}%")
        print(f"    Avg bar return       : {stats['avg_return_pct']:>+8.6f}%")
        print(f"    Std bar return       : {stats['std_return_pct']:>8.4f}%")
        print(f"    VaR (1%)             : {stats['var_1pct']:>+8.4f}%  (worst 1% bar)")
        print(f"    CVaR/ES (1%)         : {stats['cvar_1pct']:>+8.4f}%  (avg of worst 1%)")

        # Threshold checks
        if max_gain > MAX_BAR_GAIN * 100:
            self._err(f"Max bar gain {max_gain*100:.2f}% melebihi BAR_GAIN_LIMIT ({MAX_BAR_GAIN*100:.0f}%)")
            self._err("  Kemungkinan resume spike belum ter-cap dengan benar")
        elif max_gain > 20:
            self._warn(f"Max bar gain {max_gain*100:.2f}% — cukup tinggi, monitor")
        else:
            self._ok(f"Max bar gain {max_gain*100:.2f}% dalam range aman (<{MAX_BAR_GAIN*100:.0f}%)")

        if abs(max_loss) > abs(MAX_BAR_LOSS) * 100:
            self._err(f"Max bar loss {max_loss*100:.2f}% melebihi BAR_LOSS_LIMIT ({MAX_BAR_LOSS*100:.0f}%)")
        else:
            self._ok(f"Max bar loss {max_loss*100:.2f}% dalam range aman (>{MAX_BAR_LOSS*100:.0f}%)")

        # Distribusi percentiles
        print(f"\n    Return percentiles (active bars):")
        for p in [1, 5, 25, 50, 75, 95, 99]:
            val = active_ret.quantile(p/100) * 100
            print(f"      {p:>3}th pct : {val:>+8.4f}%")

        return stats

    # ─── CHECK 3: Risk Per Trade ───────────────────────────────────────────────
    def check_risk_per_trade(self) -> dict:
        print("\n  ── CHECK 3: Risk Per Trade ──")

        # Risk per trade = max potential loss jika stop hit
        # Proxy: avg loss per active bar × avg n_bars per trade
        pos = self.df["position"]
        if self.has_eq_return:
            sr = self.df["equity_return"].fillna(0)
        else:
            sr = self.df["equity"].pct_change().fillna(0)

        active = self.df[pos != 0]
        if len(active) == 0:
            self._warn("Tidak ada active bar untuk risk calculation")
            return {}

        active_sr = sr[pos != 0]

        # Hitung trade boundaries
        pos_diff  = pos.diff().fillna(pos)
        entries   = pos_diff[(pos_diff != 0) & (pos != 0)]
        n_trades  = len(entries)
        n_bars    = len(active)
        avg_bars_per_trade = n_bars / n_trades if n_trades > 0 else 0

        # Risk exposure per bar
        losses    = active_sr[active_sr < 0]
        avg_loss_bar = float(losses.mean()) if len(losses) > 0 else 0.0

        # Proxy risk per trade
        risk_per_trade_proxy = avg_loss_bar * avg_bars_per_trade

        print(f"    N trades (approx)        : {n_trades:,}")
        print(f"    Avg bars per trade       : {avg_bars_per_trade:.1f}")
        print(f"    Avg loss per bar         : {avg_loss_bar*100:>+.4f}%")
        print(f"    Proxy risk/trade         : {risk_per_trade_proxy*100:>+.4f}%")

        # Risk/reward
        wins_bar = active_sr[active_sr > 0]
        avg_win_bar = float(wins_bar.mean()) if len(wins_bar) > 0 else 0.0
        if avg_loss_bar != 0:
            rr_ratio = abs(avg_win_bar / avg_loss_bar)
            print(f"    Risk/Reward ratio (bar)  : {rr_ratio:.4f}x")

        if abs(risk_per_trade_proxy) > 0.15:
            self._err(f"Proxy risk per trade {risk_per_trade_proxy*100:.2f}% > 15% — terlalu besar")
        elif abs(risk_per_trade_proxy) > 0.05:
            self._warn(f"Proxy risk per trade {risk_per_trade_proxy*100:.2f}% — monitor")
        else:
            self._ok(f"Risk per trade dalam range wajar ({risk_per_trade_proxy*100:.2f}%)")

        return {
            "n_trades":          n_trades,
            "avg_bars_trade":    avg_bars_per_trade,
            "avg_loss_bar_pct":  avg_loss_bar * 100,
            "proxy_risk_pct":    risk_per_trade_proxy * 100,
        }

    # ─── CHECK 4: TIER2 Resume Cap Verification ───────────────────────────────
    def check_tier2_resume_cap(self) -> dict:
        print("\n  ── CHECK 4: TIER2 Resume Cap Verification ──")

        if not self.has_kill_switch:
            self._warn("Kolom kill_switch_active tidak ada — skip")
            return {}

        ks  = self.df["kill_switch_active"].values
        eq  = self.df["equity"].values
        ts  = self.df["timestamp"]

        spikes = []
        for i in range(1, len(ks)):
            if int(ks[i-1]) == 1 and int(ks[i]) == 0:
                jump = (eq[i] - eq[i-1]) / eq[i-1] if eq[i-1] > 0 else 0
                spikes.append({"ts": str(ts.iloc[i])[:10], "jump": jump})

        if not spikes:
            self._ok("Tidak ada resume event")
            return {}

        jumps    = [s["jump"] for s in spikes]
        max_jump = max(abs(j) for j in jumps)
        n_over   = sum(1 for j in jumps if abs(j) > TIER2_GAIN_CAP + 1e-6)

        print(f"    Total resume events : {len(spikes)}")
        print(f"    Max resume jump     : {max_jump*100:>+.2f}%")
        print(f"    Expected cap        : ≤{TIER2_GAIN_CAP*100:.0f}%")
        print(f"    Violations (>cap)   : {n_over}")

        if n_over > 0:
            self._err(f"{n_over} resume events melebihi cap {TIER2_GAIN_CAP*100:.0f}%")
            self._err("  Fix belum efektif — cek resume bar skip logic di risk_engine_v6.py")
            for s in spikes:
                if abs(s["jump"]) > TIER2_GAIN_CAP + 1e-6:
                    print(f"    {s['ts']}  jump={s['jump']*100:>+7.2f}%  ← VIOLATION")
        else:
            self._ok(f"Semua resume jumps ≤ {TIER2_GAIN_CAP*100:.0f}% — cap berfungsi ✓")

        return {
            "n_resume_events": len(spikes),
            "max_jump_pct":    max_jump * 100,
            "n_violations":    n_over,
        }

    def run_all_checks(self) -> dict:
        DIV = "═" * 65
        print(f"\n{DIV}")
        print("  POSITION SIZE CHECKER — Full Audit")
        print(DIV)
        print(f"  Dataset: {len(self.df):,} bars")

        return {
            "leverage":     self.check_leverage(),
            "bar_returns":  self.check_bar_returns(),
            "risk_trade":   self.check_risk_per_trade(),
            "tier2_cap":    self.check_tier2_resume_cap(),
        }


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    path = BASE / "btc_risk_managed_results.csv"
    if not path.exists():
        print(f"❌ File tidak ditemukan: {path}")
        print("   Jalankan risk_engine_v6.py terlebih dahulu")
        raise SystemExit(1)

    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    checker = PositionSizeChecker(df)
    checker.run_all_checks()
