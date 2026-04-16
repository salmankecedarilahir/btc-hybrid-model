"""
metrics_calculator.py — BTC Hybrid Model: Trade-Level Metric Calculator
========================================================================

BAGIAN 1 — FORMULA MATEMATIKA YANG BENAR
─────────────────────────────────────────
Semua metrik harus dihitung dari TRADE LIST (bukan per-bar),
kecuali Sharpe/Sortino yang dihitung dari equity curve returns.

  Profit Factor  = Σ(wins) / |Σ(losses)|
                   → PF > 1 berarti profitable

  Expectancy     = WR × AvgWin + (1 - WR) × AvgLoss
                   → harus KONSISTEN dengan PF:
                     jika PF > 1 maka Expectancy PASTI > 0
                     jika Expectancy < 0 dengan PF > 1 → BUG

  Win Rate       = N_wins / N_trades            (per TRADE, bukan per BAR)

  Avg Win        = Σ(positive returns) / N_wins
  Avg Loss       = Σ(negative returns) / N_losses  (nilai negatif)

  Sharpe (ann.)  = (μ_r / σ_r) × √BPY
                   dimana r = equity daily/bar returns

  Sortino (ann.) = (μ_r / σ_downside) × √BPY
                   dimana σ_downside = std(r[r < 0])

  Max Drawdown   = min((equity - cummax(equity)) / cummax(equity))

BAGIAN 2 — BUG YANG PALING SERING TERJADI
──────────────────────────────────────────
  BUG 1: Return dihitung per BAR bukan per TRADE
    → Win rate 4.9% terjadi karena hanya "entry bar" yang dihitung
    → Bar yang sedang hold posisi tidak dihitung sebagai win/loss
    → Fix: akumulasi return dari entry sampai exit

  BUG 2: Dataset berbeda antara PF dan Expectancy
    → PF dari equity_return (leveraged), Expectancy dari strategy_return (raw)
    → Nilai berbeda karena leverage multiplier berbeda tiap bar
    → Fix: gunakan SATU dataset yang sama untuk semua metrik

  BUG 3: Zero-return bars masuk ke denominator
    → TIER2 paused bars punya equity_return = 0.0
    → Masuk ke len(ret) tapi tidak masuk wins/losses
    → Inflate denominator → WR deflated → Expectancy palsu negatif

  BUG 4: Trade tidak closed dengan benar
    → Open trade di akhir backtest tidak dihitung return-nya
    → Missing exit → return = 0 → masuk losses yang seharusnya tidak ada

  BUG 5: Overlapping trades
    → Posisi belum closed, sudah buka posisi baru
    → Double counting returns pada periode yang sama

Cara pakai:
    from metrics_calculator import MetricsCalculator
    calc = MetricsCalculator(trade_df, equity_df)
    report = calc.full_report()
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

BARS_PER_YEAR = 2190   # 4H bars per year


# ════════════════════════════════════════════════════════════════════
#  CORE METRIC FUNCTIONS
# ════════════════════════════════════════════════════════════════════

def calc_profit_factor(returns: pd.Series) -> float:
    """
    PF = Σ(wins) / |Σ(losses)|
    Input: series of trade returns (signed %)
    """
    wins   = returns[returns > 0]
    losses = returns[returns < 0]
    if len(losses) == 0 or losses.sum() == 0:
        return float("inf")
    return float(wins.sum() / abs(losses.sum()))


def calc_expectancy(returns: pd.Series) -> dict:
    """
    Expectancy = WR × AvgWin + (1-WR) × AvgLoss

    INVARIANT: Jika PF > 1, Expectancy PASTI > 0.
    Jika tidak, ada bug di dataset.

    Returns dict dengan semua komponen untuk debugging.
    """
    wins   = returns[returns > 0]
    losses = returns[returns < 0]
    n      = len(returns)

    if n == 0:
        return {"expectancy": 0, "win_rate": 0, "avg_win": 0, "avg_loss": 0}

    wr       = len(wins) / n
    avg_win  = float(wins.mean())  if len(wins)   > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
    exp      = wr * avg_win + (1 - wr) * avg_loss

    pf = calc_profit_factor(returns)

    # Konsistensi check
    pf_implies_positive = pf > 1
    exp_is_positive     = exp > 0
    consistent          = (pf_implies_positive == exp_is_positive)

    return {
        "expectancy":      round(exp * 100, 6),       # dalam %
        "win_rate":        round(wr * 100, 4),
        "avg_win":         round(avg_win * 100, 4),
        "avg_loss":        round(avg_loss * 100, 4),
        "profit_factor":   round(pf, 4),
        "n_trades":        n,
        "n_wins":          len(wins),
        "n_losses":        len(losses),
        "consistent":      consistent,
        "win_loss_ratio":  round(abs(avg_win / avg_loss), 4) if avg_loss != 0 else float("inf"),
    }


def calc_sharpe(equity_returns: pd.Series, bpy: int = BARS_PER_YEAR) -> float:
    """
    Sharpe = (μ / σ) × √BPY
    Input: bar-level equity returns (pct_change dari equity curve)
    """
    r = equity_returns.dropna()
    if r.std() == 0 or len(r) < 2:
        return 0.0
    return float((r.mean() / r.std()) * np.sqrt(bpy))


def calc_sortino(equity_returns: pd.Series, bpy: int = BARS_PER_YEAR) -> float:
    """
    Sortino = (μ / σ_downside) × √BPY
    σ_downside = std(r[r < 0])  → hanya downside volatility
    """
    r    = equity_returns.dropna()
    neg  = r[r < 0]
    if len(neg) == 0 or neg.std() == 0:
        return 0.0
    return float((r.mean() / neg.std()) * np.sqrt(bpy))


def calc_max_drawdown(equity: pd.Series) -> dict:
    """
    MaxDD = min((equity - cummax) / cummax)
    Returns MaxDD dan periode-nya.
    """
    roll_max = equity.cummax()
    dd       = (equity - roll_max) / roll_max
    max_dd   = float(dd.min())
    idx      = dd.idxmin()

    return {
        "max_drawdown_pct": round(max_dd * 100, 4),
        "max_dd_date":      str(equity.index[idx]) if hasattr(equity.index, '__getitem__') else str(idx),
        "no_positive_dd":   bool((dd > 0.001).sum() == 0),
    }


def calc_calmar(cagr_pct: float, max_dd_pct: float) -> float:
    """Calmar = CAGR% / |MaxDD%|"""
    if max_dd_pct == 0:
        return 0.0
    return round(cagr_pct / abs(max_dd_pct), 4)


# ════════════════════════════════════════════════════════════════════
#  MAIN CLASS
# ════════════════════════════════════════════════════════════════════

class MetricsCalculator:
    """
    Menghitung semua metrik dari trade list DAN equity curve.

    Input columns yang dibutuhkan:
      trade_df  : [entry_time, exit_time, direction, return_pct, entry_price, exit_price]
      equity_df : [timestamp, equity, drawdown]  (bar-level)
    """

    def __init__(self,
                 trade_df:  pd.DataFrame,
                 equity_df: pd.DataFrame,
                 initial_equity: float = 10_000.0,
                 bpy: int = BARS_PER_YEAR):
        self.trades  = trade_df.copy()
        self.equity  = equity_df.copy()
        self.init_eq = initial_equity
        self.bpy     = bpy
        self._validate_inputs()

    def _validate_inputs(self):
        required_trade  = {"return_pct"}
        required_equity = {"equity"}
        for col in required_trade:
            if col not in self.trades.columns:
                raise ValueError(f"trade_df missing column: {col}")
        for col in required_equity:
            if col not in self.equity.columns:
                raise ValueError(f"equity_df missing column: {col}")

    def _get_clean_trade_returns(self) -> pd.Series:
        """
        Ambil return per trade — exclude:
          1. NaN returns
          2. Zero returns dari paused bars / incomplete trades
          3. Outlier ekstrem (>500%) yang kemungkinan data error
        """
        ret = pd.to_numeric(self.trades["return_pct"], errors="coerce").dropna()
        n_before = len(ret)

        # Exclude zero returns (TIER2 paused / no position)
        ret = ret[ret != 0.0]
        n_zero = n_before - len(ret)

        # Exclude extreme outliers (> 500% per trade = almost certainly error)
        n_outlier_before = len(ret)
        ret = ret[ret.abs() <= 5.0]   # 500% max
        n_outlier = n_outlier_before - len(ret)

        if n_zero > 0:
            log.warning("Excluded %d zero-return records (paused/no-position bars)", n_zero)
        if n_outlier > 0:
            log.warning("Excluded %d extreme outlier trades (>500%%)", n_outlier)

        return ret

    def _get_equity_bar_returns(self) -> pd.Series:
        """
        Ambil bar-level equity returns untuk Sharpe/Sortino.
        Exclude zero-return TIER2 bars dari denominator.
        """
        eq  = self.equity["equity"]
        ret = eq.pct_change().fillna(0)

        # Jika leverage_used tersedia, pakai itu sebagai active mask
        if "leverage_used" in self.equity.columns:
            active = self.equity["leverage_used"] > 0
            return ret[active]

        # Fallback: exclude zero returns (TIER2 flat bars)
        return ret[ret != 0.0]

    def trade_metrics(self) -> dict:
        """Semua metrik berbasis trade list."""
        ret = self._get_clean_trade_returns()
        exp = calc_expectancy(ret)

        return {
            **exp,
            "note": "Dihitung dari trade return, exclude zero/paused bars",
        }

    def equity_metrics(self) -> dict:
        """Sharpe, Sortino, MaxDD dari equity curve."""
        bar_ret = self._get_equity_bar_returns()
        eq      = self.equity["equity"]

        # CAGR
        n_years = len(self.equity) / self.bpy
        final   = float(eq.iloc[-1])
        cagr    = ((final / self.init_eq) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

        sharpe  = calc_sharpe(bar_ret, self.bpy)
        sortino = calc_sortino(bar_ret, self.bpy)
        dd      = calc_max_drawdown(eq)
        calmar  = calc_calmar(cagr, dd["max_drawdown_pct"])

        return {
            "cagr_pct":     round(cagr, 4),
            "final_equity": round(final, 2),
            "sharpe":       round(sharpe, 4),
            "sortino":      round(sortino, 4),
            "calmar":       round(calmar, 4),
            **dd,
        }

    def consistency_check(self) -> dict:
        """
        Verifikasi konsistensi matematis PF vs Expectancy.

        INVARIANT:
          PF > 1  ↔  Expectancy > 0   (HARUS selalu berlaku)
          PF = 1  ↔  Expectancy = 0
          PF < 1  ↔  Expectancy < 0
        """
        tm  = self.trade_metrics()
        pf  = tm["profit_factor"]
        exp = tm["expectancy"]
        wr  = tm["win_rate"]

        # Cek konsistensi
        pf_positive  = pf > 1
        exp_positive = exp > 0
        consistent   = (pf_positive == exp_positive)

        # Cek plausibilitas win rate
        # Model momentum: WR rendah (5-25%) + avg_win >> avg_loss adalah NORMAL
        wlr = tm["win_loss_ratio"]
        wr_plausible = (
            (wr < 25 and wlr > 2.0) or   # momentum: WR rendah tapi ratio tinggi
            (wr >= 25 and wlr >= 1.0)     # balanced: WR cukup, ratio minimal 1:1
        )

        # Diagnosa bug jika inkonsisten
        diagnosis = []
        if not consistent:
            if pf > 1 and exp < 0:
                diagnosis.append("BUG: Zero-return bars inflate denominator WR")
                diagnosis.append("BUG: Dataset berbeda antara PF dan Expectancy")
                diagnosis.append("FIX: Exclude equity_return==0 dari active bar filter")
            else:
                diagnosis.append("BUG: PF < 1 tapi Expectancy positif — cek sign convention")

        return {
            "pf_exp_consistent": consistent,
            "wr_plausible":      wr_plausible,
            "diagnosis":         diagnosis,
            "profit_factor":     pf,
            "expectancy_pct":    exp,
            "win_rate_pct":      wr,
            "win_loss_ratio":    wlr,
        }

    def full_report(self, verbose: bool = True) -> dict:
        """Generate laporan lengkap semua metrik."""
        tm   = self.trade_metrics()
        em   = self.equity_metrics()
        chk  = self.consistency_check()

        report = {
            "trade_metrics":    tm,
            "equity_metrics":   em,
            "consistency":      chk,
        }

        if verbose:
            self._print_report(tm, em, chk)

        return report

    def _print_report(self, tm, em, chk):
        DIV = "═" * 65
        SEP = "─" * 65
        print(f"\n{DIV}")
        print("  METRICS CALCULATOR — Full Report")
        print(DIV)

        print(f"\n  ── TRADE-LEVEL METRICS ──")
        print(f"  Total trades     : {tm['n_trades']:,}")
        print(f"  Win Rate         : {tm['win_rate']:>10.4f}%")
        print(f"  Avg Win          : {tm['avg_win']:>+10.4f}%")
        print(f"  Avg Loss         : {tm['avg_loss']:>+10.4f}%")
        print(f"  Win/Loss Ratio   : {tm['win_loss_ratio']:>10.4f}x")
        print(f"  Profit Factor    : {tm['profit_factor']:>10.4f}")
        print(f"  Expectancy/trade : {tm['expectancy']:>+10.6f}%")

        cons_mark = "[OK]" if chk["pf_exp_consistent"] else "❌"
        print(f"\n  PF ↔ Expectancy  : {cons_mark} {'KONSISTEN' if chk['pf_exp_consistent'] else 'INKONSISTEN — BUG!'}")
        if chk["diagnosis"]:
            for d in chk["diagnosis"]:
                print(f"    → {d}")

        print(f"\n  ── EQUITY-LEVEL METRICS ──")
        print(f"  CAGR             : {em['cagr_pct']:>+10.4f}%")
        print(f"  Sharpe           : {em['sharpe']:>10.4f}  {'[OK]' if em['sharpe']  > 1.0 else '[WARN]️'} (>1.0)")
        print(f"  Sortino          : {em['sortino']:>10.4f}  {'[OK]' if em['sortino'] > 1.0 else '[WARN]️'} (>1.0)")
        print(f"  Calmar           : {em['calmar']:>10.4f}  {'[OK]' if em['calmar']  > 1.0 else '[WARN]️'} (>1.0)")
        print(f"  Max Drawdown     : {em['max_drawdown_pct']:>10.4f}%")
        print(f"  Final Equity     : ${em['final_equity']:>12,.2f}")
        print(f"{DIV}\n")


# ════════════════════════════════════════════════════════════════════
#  STANDALONE RUNNER
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    BASE = Path(__file__).parent / "data"

    # Load trade list dari backtest results
    trade_path  = BASE / "btc_backtest_results.csv"
    equity_path = BASE / "btc_risk_managed_results.csv"

    if not trade_path.exists():
        print(f"❌ File tidak ditemukan: {trade_path}")
        print("   Jalankan backtest_engine.py terlebih dahulu")
        raise SystemExit(1)
    if not equity_path.exists():
        print(f"❌ File tidak ditemukan: {equity_path}")
        print("   Jalankan risk_engine_v6.py terlebih dahulu")
        raise SystemExit(1)

    trade_df  = pd.read_csv(trade_path,  parse_dates=["timestamp"])
    equity_df = pd.read_csv(equity_path, parse_dates=["timestamp"])

    # Rename kolom agar sesuai expected format
    # trade_df pakai strategy_return sebagai return_pct
    if "return_pct" not in trade_df.columns:
        if "strategy_return" in trade_df.columns:
            trade_df["return_pct"] = trade_df["strategy_return"]
        elif "equity_return" in trade_df.columns:
            trade_df["return_pct"] = trade_df["equity_return"]

    calc   = MetricsCalculator(trade_df, equity_df)
    report = calc.full_report(verbose=True)
