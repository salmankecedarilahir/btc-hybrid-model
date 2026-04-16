"""
trade_list_validator.py — BTC Hybrid Model: Trade List Validator
================================================================

Memeriksa integritas trade list:
  1. Trade overlap (posisi belum closed, sudah buka baru)
  2. Missing exit (entry tanpa exit di akhir data)
  3. Incorrect return calculation (return tidak cocok harga)
  4. Inconsistent trade count (jumlah entry ≠ exit)
  5. Zero-return contamination (TIER2 paused bars masuk trade)
  6. Sign convention error (SHORT return harus negatif saat harga naik)

Cara pakai:
    python trade_list_validator.py
    # atau
    from trade_list_validator import TradeListValidator
    v = TradeListValidator(df)
    v.run_all_checks()
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

BASE = Path(__file__).parent / "data"


# ════════════════════════════════════════════════════════════════════
#  VALIDATOR CLASS
# ════════════════════════════════════════════════════════════════════

class TradeListValidator:
    """
    Validasi trade list dari backtest_results.csv.

    Expected columns (minimal):
      timestamp, signal, position, market_return,
      strategy_return, equity, drawdown
    """

    def __init__(self, df: pd.DataFrame, verbose: bool = True):
        self.df      = df.copy()
        self.verbose = verbose
        self.errors  = []
        self.warns   = []
        self._ensure_columns()

    def _ensure_columns(self):
        required = ["timestamp", "signal", "position",
                    "market_return", "strategy_return", "equity"]
        missing  = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        self.df = self.df.sort_values("timestamp").reset_index(drop=True)

    def _ok(self, msg):
        if self.verbose: print(f"  [OK] {msg}")

    def _warn(self, msg):
        self.warns.append(msg)
        if self.verbose: print(f"  [WARN]️  {msg}")

    def _err(self, msg):
        self.errors.append(msg)
        if self.verbose: print(f"  ❌ {msg}")

    # ─── CHECK 1: Trade Overlap ────────────────────────────────────────────────
    def check_trade_overlap(self) -> bool:
        """
        Detect posisi berubah arah tanpa melalui FLAT/NONE.
        Contoh: LONG → SHORT tanpa FLAT bar di antara = overlap.
        """
        if self.verbose:
            print("\n  ── CHECK 1: Trade Overlap ──")

        pos    = self.df["position"]
        # Overlap = posisi langsung flip dari +1 ke -1 atau sebaliknya
        flip   = ((pos.shift(1) == 1) & (pos == -1)) | \
                 ((pos.shift(1) == -1) & (pos == 1))
        n_flip = flip.sum()

        if n_flip > 0:
            dates = self.df.loc[flip, "timestamp"].dt.strftime("%Y-%m-%d").tolist()
            self._warn(f"{n_flip} direct position flip (LONG↔SHORT tanpa FLAT): {dates[:5]}{'...' if len(dates)>5 else ''}")
            return False
        else:
            self._ok("Tidak ada trade overlap — semua flip melalui FLAT/NONE")
            return True

    # ─── CHECK 2: Missing Exit ────────────────────────────────────────────────
    def check_missing_exit(self) -> bool:
        """
        Pastikan trade terakhir di-close (posisi kembali ke 0 di akhir data).
        """
        if self.verbose:
            print("\n  ── CHECK 2: Missing Exit ──")

        last_pos = int(self.df["position"].iloc[-1])
        if last_pos != 0:
            self._warn(f"Posisi terakhir = {last_pos} (belum closed). "
                       f"Open trade di akhir data tidak dihitung return-nya.")
            return False
        else:
            self._ok("Semua trade closed — posisi terakhir = 0")
            return True

    # ─── CHECK 3: Return Calculation Correctness ──────────────────────────────
    def check_return_calculation(self) -> bool:
        """
        Verifikasi: strategy_return = position × market_return (no lookahead).
        Setiap mismatch > 1e-8 menunjukkan bug kalkulasi.
        """
        if self.verbose:
            print("\n  ── CHECK 3: Return Calculation Correctness ──")

        expected = self.df["position"] * self.df["market_return"]
        actual   = self.df["strategy_return"]
        diff     = (expected - actual).abs()
        max_diff = float(diff.max())
        n_mismatch = (diff > 1e-8).sum()

        if n_mismatch > 0:
            self._err(f"strategy_return mismatch: {n_mismatch} bars (max diff={max_diff:.2e})")
            self._err("  Kemungkinan bug: lookahead, leverage factor, atau fee tidak dikurangi")
            return False
        else:
            self._ok(f"strategy_return = position × market_return ✓ (max diff={max_diff:.2e})")
            return True

    # ─── CHECK 4: Trade Count Consistency ────────────────────────────────────
    def check_trade_count(self) -> dict:
        """
        Hitung jumlah trade dari signal changes.
        Bandingkan dengan active bar count.
        """
        if self.verbose:
            print("\n  ── CHECK 4: Trade Count Consistency ──")

        pos          = self.df["position"]
        # Entry = bar di mana posisi berubah dari 0/berbeda ke nilai baru
        pos_changed  = pos.diff().fillna(pos).ne(0) & (pos != 0)
        n_entries    = int(pos_changed.sum())

        # Exit = bar di mana posisi berubah ke 0 atau ke arah berbeda
        exits        = ((pos != 0) & (pos.shift(-1, fill_value=0) != pos))
        n_exits      = int(exits.sum())

        n_long_bars  = int((pos == 1).sum())
        n_short_bars = int((pos == -1).sum())
        n_flat_bars  = int((pos == 0).sum())
        n_active     = n_long_bars + n_short_bars

        balanced = (n_entries == n_exits)

        print(f"    Entries detected : {n_entries:,}")
        print(f"    Exits detected   : {n_exits:,}")
        print(f"    LONG bars        : {n_long_bars:,}")
        print(f"    SHORT bars       : {n_short_bars:,}")
        print(f"    FLAT bars        : {n_flat_bars:,}")
        print(f"    Active bars      : {n_active:,} ({n_active/len(pos)*100:.1f}%)")

        if balanced:
            self._ok(f"Entry/Exit balanced: {n_entries} entries = {n_exits} exits")
        else:
            self._err(f"Entry/Exit mismatch: {n_entries} entries ≠ {n_exits} exits")

        return {
            "n_entries": n_entries, "n_exits": n_exits,
            "n_long": n_long_bars, "n_short": n_short_bars,
            "n_flat": n_flat_bars, "balanced": balanced,
        }

    # ─── CHECK 5: Zero Return Contamination ───────────────────────────────────
    def check_zero_return_contamination(self) -> dict:
        """
        Deteksi berapa banyak zero-return bars yang ada di active position bars.
        Zero return saat active = TIER2 paused atau bug kalkulasi.
        """
        if self.verbose:
            print("\n  ── CHECK 5: Zero-Return Contamination ──")

        active = self.df[self.df["position"] != 0]
        sr     = active["strategy_return"]
        n_zero = (sr == 0.0).sum()
        pct    = n_zero / len(sr) * 100 if len(sr) > 0 else 0

        print(f"    Active bars total     : {len(active):,}")
        print(f"    Zero-return active    : {n_zero:,}  ({pct:.1f}%)")
        print(f"    Non-zero active       : {len(active) - n_zero:,}")

        if pct > 20:
            self._err(f"{pct:.1f}% active bars punya return=0 → kemungkinan TIER2 bars masuk filter")
            self._err("  Fix: gunakan leverage_used > 0 sebagai active filter")
        elif pct > 5:
            self._warn(f"{pct:.1f}% active bars punya return=0 → cek TIER2 pause logic")
        else:
            self._ok(f"Zero-return contamination rendah: {pct:.1f}%")

        # Efek pada Win Rate jika zero bars dimasukkan
        ret_with_zero    = sr
        ret_without_zero = sr[sr != 0.0]

        wr_with    = (ret_with_zero > 0).mean() * 100    if len(ret_with_zero)    > 0 else 0
        wr_without = (ret_without_zero > 0).mean() * 100 if len(ret_without_zero) > 0 else 0

        print(f"\n    Win Rate WITH zero-return bars    : {wr_with:.2f}%")
        print(f"    Win Rate WITHOUT zero-return bars : {wr_without:.2f}%")
        print(f"    Perbedaan WR                      : {wr_without - wr_with:+.2f}%")

        if abs(wr_without - wr_with) > 2:
            self._warn(f"Zero-return bars significantly distort WR by {wr_without-wr_with:+.2f}%")

        return {
            "n_zero": n_zero, "pct_zero": pct,
            "wr_with_zero": wr_with, "wr_without_zero": wr_without,
        }

    # ─── CHECK 6: Sign Convention ─────────────────────────────────────────────
    def check_sign_convention(self) -> bool:
        """
        SHORT position saat market naik harus menghasilkan negative return.
        Verifikasi: strategy_return SHORT bars saat market_return > 0 harus < 0.
        """
        if self.verbose:
            print("\n  ── CHECK 6: Sign Convention ──")

        short_up = self.df[(self.df["position"] == -1) & (self.df["market_return"] > 0.001)]
        short_dn = self.df[(self.df["position"] == -1) & (self.df["market_return"] < -0.001)]

        if len(short_up) > 0:
            pct_neg = (short_up["strategy_return"] < 0).mean() * 100
            if pct_neg < 95:
                self._err(f"SHORT dengan market_return>0: hanya {pct_neg:.1f}% punya return negatif")
                self._err("  Kemungkinan sign convention bug di strategy_return kalkulasi")
                return False
            else:
                self._ok(f"SHORT sign convention benar: {pct_neg:.1f}% negatif saat market naik")

        if len(short_dn) > 0:
            pct_pos = (short_dn["strategy_return"] > 0).mean() * 100
            if pct_pos < 95:
                self._warn(f"SHORT dengan market_return<0: hanya {pct_pos:.1f}% punya return positif")

        return True

    # ─── CHECK 7: Equity Monotonicity ─────────────────────────────────────────
    def check_equity_integrity(self) -> bool:
        """
        Equity tidak boleh negatif atau NaN.
        Drawdown tidak boleh positif (impossible).
        """
        if self.verbose:
            print("\n  ── CHECK 7: Equity Integrity ──")

        eq   = self.df["equity"]
        ok   = True

        n_neg = (eq <= 0).sum()
        n_nan = eq.isna().sum()

        if n_neg > 0:
            self._err(f"{n_neg} bars dengan equity <= 0 — floor logic mungkin buggy")
            ok = False
        else:
            self._ok("Tidak ada equity negatif atau zero")

        if n_nan > 0:
            self._err(f"{n_nan} NaN di equity curve")
            ok = False
        else:
            self._ok("Tidak ada NaN di equity")

        if "drawdown" in self.df.columns:
            dd = self.df["drawdown"]
            n_pos_dd = (dd > 0.001).sum()
            if n_pos_dd > 0:
                self._err(f"{n_pos_dd} bars dengan drawdown positif (mathematically impossible)")
                ok = False
            else:
                self._ok("Tidak ada drawdown positif")

        return ok

    # ─── FULL REPORT ──────────────────────────────────────────────────────────
    def run_all_checks(self) -> dict:
        DIV = "═" * 65
        print(f"\n{DIV}")
        print("  TRADE LIST VALIDATOR — Full Check")
        print(DIV)

        results = {
            "overlap":      self.check_trade_overlap(),
            "missing_exit": self.check_missing_exit(),
            "return_calc":  self.check_return_calculation(),
            "trade_count":  self.check_trade_count(),
            "zero_contam":  self.check_zero_return_contamination(),
            "sign_conv":    self.check_sign_convention(),
            "equity_ok":    self.check_equity_integrity(),
        }

        print(f"\n{DIV}")
        print("  SUMMARY")
        print(DIV)
        print(f"  Total errors   : {len(self.errors)}")
        print(f"  Total warnings : {len(self.warns)}")

        if self.errors:
            print("\n  ❌ ERRORS (harus difix sebelum lanjut):")
            for e in self.errors:
                print(f"    • {e}")

        if self.warns:
            print("\n  [WARN]️  WARNINGS (perlu monitoring):")
            for w in self.warns:
                print(f"    • {w}")

        if not self.errors:
            print("\n  [OK] Dataset valid — siap untuk metric calculation")

        print(DIV)
        return results


# ════════════════════════════════════════════════════════════════════
#  STANDALONE RUNNER
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    path = BASE / "btc_backtest_results.csv"
    if not path.exists():
        print(f"❌ File tidak ditemukan: {path}")
        raise SystemExit(1)

    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    v = TradeListValidator(df, verbose=True)
    v.run_all_checks()
