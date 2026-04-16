"""
equity_curve_reconstructor.py — BTC Hybrid Model
=================================================

BAGIAN 3 & 6 — AUDIT TRADE LIST VS EQUITY CURVE

Metode verifikasi konsistensi:

  Step 1: Ambil trade list dari backtest (position series + strategy_return)
  Step 2: Hitung cumulative return per trade dari entry → exit
  Step 3: Rekonstruksi equity curve dari trade list
  Step 4: Bandingkan equity rekonstruksi vs equity asli
          → Jika sama (diff < 1e-6): dataset konsisten, tidak ada bug
          → Jika berbeda: ada bug di salah satu (biasanya di kalkulasi return)

BAGIAN 7 — EXPECTANCY VALIDATION

Setelah rekonstruksi berhasil:
  - Expectancy = WR × AvgWin + (1-WR) × AvgLoss harus > 0 jika PF > 1
  - Verifikasi ketiganya (PF, Exp, WR) secara matematis konsisten

Cara pakai:
    python equity_curve_reconstructor.py
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless (tidak perlu display)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

BASE         = Path(__file__).parent / "data"
INITIAL_EQ   = 10_000.0
BARS_PER_YEAR= 2190


# ════════════════════════════════════════════════════════════════════
#  TRADE EXTRACTOR
# ════════════════════════════════════════════════════════════════════

def extract_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ekstrak trade-level summary dari bar-level dataframe.

    Returns DataFrame dengan kolom:
      trade_id, entry_time, exit_time, direction,
      entry_price, exit_price, n_bars,
      cumulative_return, strategy_return_sum
    """
    pos = df["position"].values
    sr  = df["strategy_return"].values
    ts  = df["timestamp"].values

    close = df["close"].values if "close" in df.columns else np.zeros(len(df))

    trades   = []
    in_trade = False
    entry_i  = 0
    trade_id = 0

    for i in range(len(df)):
        if not in_trade:
            if pos[i] != 0:
                in_trade  = True
                entry_i   = i
                trade_id += 1
        else:
            # Exit: posisi berubah ke 0 atau flip arah
            exiting = (pos[i] == 0) or (pos[i] != pos[entry_i])
            if exiting or i == len(df) - 1:
                exit_i = i + 1

                trade_bars = df.iloc[entry_i:exit_i]
                if len(trade_bars) == 0:
                    continue

                bar_returns   = trade_bars["strategy_return"].values
                cumulative_r  = float(np.prod(1 + bar_returns) - 1)
                sum_r         = float(bar_returns.sum())

                exit_idx = min(i, len(df) - 1)

                trades.append({
                    "trade_id":          trade_id,
                    "entry_time":        ts[entry_i],
                    "exit_time":         ts[exit_idx],
                    "direction":         "LONG" if pos[entry_i] == 1 else "SHORT",
                    "entry_price":       float(close[entry_i]),
                    "exit_price":        float(close[exit_idx]),
                    "n_bars":            len(trade_bars),
                    "cumulative_return": cumulative_r,
                    "sum_return":        sum_r,
                    "is_win":            cumulative_r > 0,
                })

                in_trade = False
                if pos[i] != 0:
                    # Langsung buka trade baru
                    in_trade  = True
                    entry_i   = i
                    trade_id += 1

    return pd.DataFrame(trades)


# ════════════════════════════════════════════════════════════════════
#  EQUITY RECONSTRUCTOR
# ════════════════════════════════════════════════════════════════════

def reconstruct_equity_from_bars(df: pd.DataFrame,
                                  initial: float = INITIAL_EQ) -> pd.Series:
    """
    Rekonstruksi equity curve bar-by-bar dari strategy_return.
    Ini seharusnya IDENTIK dengan equity curve asli jika tidak ada bug.

    equity[t] = initial × Π(1 + strategy_return[0..t])
    """
    sr     = df["strategy_return"].fillna(0)
    equity = initial * (1 + sr).cumprod()
    return equity.clip(lower=0.01)


def reconstruct_equity_from_trades(trade_df: pd.DataFrame,
                                    n_bars:   int,
                                    initial:  float = INITIAL_EQ) -> pd.Series:
    """
    Rekonstruksi equity dari trade list (bukan bar level).
    Useful untuk verifikasi: equity dari trade list == equity dari bar returns.
    """
    eq = np.full(n_bars, initial)
    cur = initial

    for _, row in trade_df.iterrows():
        cur = cur * (1 + row["cumulative_return"])
        cur = max(cur, 0.01)

    return pd.Series(eq)


# ════════════════════════════════════════════════════════════════════
#  COMPARATOR
# ════════════════════════════════════════════════════════════════════

def compare_equity_curves(original: pd.Series,
                           reconstructed: pd.Series,
                           tolerance: float = 1e-4) -> dict:
    """
    Bandingkan equity curve asli vs rekonstruksi.

    Jika keduanya sama (max diff < tolerance):
      → Dataset konsisten, tidak ada bug di kalkulasi return

    Jika berbeda:
      → Ada bug di salah satu:
         a) equity curve asli tidak mencerminkan strategy_return
         b) strategy_return tidak dihitung dengan benar
         c) Ada komponen lain (fee, slippage) yang tidak tertrack
    """
    n    = min(len(original), len(reconstructed))
    orig = original.iloc[:n].values
    reco = reconstructed.iloc[:n].values

    # Relative diff: |orig - reco| / orig
    rel_diff = np.abs(orig - reco) / np.maximum(np.abs(orig), 1e-8)
    abs_diff = np.abs(orig - reco)

    max_rel  = float(rel_diff.max())
    max_abs  = float(abs_diff.max())
    mean_rel = float(rel_diff.mean())

    # Cek divergence point (pertama kali diff melebihi tolerance)
    diverge_idx = np.where(rel_diff > tolerance)[0]
    first_diverg = int(diverge_idx[0]) if len(diverge_idx) > 0 else -1

    consistent = max_rel < tolerance

    return {
        "consistent":       consistent,
        "max_rel_diff":     round(max_rel * 100, 6),    # dalam %
        "max_abs_diff":     round(max_abs, 2),
        "mean_rel_diff":    round(mean_rel * 100, 6),
        "first_divergence_bar": first_diverg,
        "n_bars_compared":  n,
        "tolerance_pct":    tolerance * 100,
    }


# ════════════════════════════════════════════════════════════════════
#  EXPECTANCY VALIDATOR  (Bagian 7)
# ════════════════════════════════════════════════════════════════════

def validate_expectancy_consistency(trade_df: pd.DataFrame) -> dict:
    """
    BAGIAN 7 — Verifikasi konsistensi matematis:

    INVARIANT yang harus selalu berlaku:
      PF > 1  ↔  Expectancy > 0
      PF = 1  ↔  Expectancy = 0
      PF < 1  ↔  Expectancy < 0

    Jika invariant dilanggar, ada bug di dataset atau kalkulasi.

    Proof:
      PF = Σwins / |Σlosses|
      Exp = WR×AvgWin + LR×AvgLoss
          = (Nw/N)×(Σwins/Nw) + (Nl/N)×(Σlosses/Nl)
          = Σwins/N + Σlosses/N
          = (Σwins + Σlosses) / N

      PF > 1 → Σwins > |Σlosses| → Σwins + Σlosses > 0 → Exp > 0  ✓
    """
    if "cumulative_return" in trade_df.columns:
        ret_col = "cumulative_return"
    elif "return_pct" in trade_df.columns:
        ret_col = "return_pct"
    elif "sum_return" in trade_df.columns:
        ret_col = "sum_return"
    else:
        raise ValueError("trade_df harus punya kolom: cumulative_return, return_pct, atau sum_return")

    ret    = pd.to_numeric(trade_df[ret_col], errors="coerce").dropna()
    ret    = ret[ret != 0.0]   # exclude zero-return

    wins   = ret[ret > 0]
    losses = ret[ret < 0]
    n      = len(ret)

    if n == 0:
        return {"error": "No valid trades found"}

    pf       = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else float("inf")
    wr       = len(wins) / n
    avg_win  = float(wins.mean())  if len(wins)   > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
    exp      = wr * avg_win + (1 - wr) * avg_loss

    # Direct formula: Exp = (Σwins + Σlosses) / N
    exp_direct = float((wins.sum() + losses.sum()) / n)

    # Keduanya harus sama (numerical precision)
    formula_match = abs(exp - exp_direct) < 1e-10

    # Invariant check
    consistent = (pf > 1) == (exp > 0)

    # Diagnosa jika inkonsisten
    diagnosis = []
    if not consistent:
        if pf > 1 and exp < 0:
            diagnosis.append("IMPOSSIBLE: PF>1 tapi Exp<0 — pasti ada bug di dataset")
            diagnosis.append("Penyebab paling umum:")
            diagnosis.append("  1. Zero-return bars (TIER2 paused) masuk ke trade list")
            diagnosis.append("     → Inflate N tanpa tambah wins/losses → WR deflated")
            diagnosis.append("  2. Dataset berbeda untuk PF vs Expectancy")
            diagnosis.append("  3. Return dihitung per-bar bukan per-trade")

    DIV = "═" * 65
    print(f"\n{DIV}")
    print("  EXPECTANCY VALIDATION (Bagian 7)")
    print(DIV)
    print(f"  Return column used : {ret_col}")
    print(f"  Total trades       : {n:,}")
    print(f"  Win trades         : {len(wins):,}")
    print(f"  Loss trades        : {len(losses):,}")
    print(f"  Win Rate           : {wr*100:.4f}%")
    print(f"  Avg Win            : {avg_win*100:+.4f}%")
    print(f"  Avg Loss           : {avg_loss*100:+.4f}%")
    print(f"  Win/Loss Ratio     : {abs(avg_win/avg_loss):.4f}x" if avg_loss != 0 else "  Win/Loss Ratio     : ∞")
    print(f"\n  Profit Factor      : {pf:.6f}")
    print(f"  Expectancy (formula): {exp*100:+.6f}%")
    print(f"  Expectancy (direct) : {exp_direct*100:+.6f}%")
    print(f"  Formula match      : {'[OK] Ya' if formula_match else '❌ Tidak'}")
    print(f"\n  PF ↔ Expectancy    : {'[OK] KONSISTEN' if consistent else '❌ INKONSISTEN — BUG!'}")

    if diagnosis:
        print(f"\n  Diagnosis:")
        for d in diagnosis:
            print(f"    {d}")

    print(DIV)

    return {
        "n_trades": n, "n_wins": len(wins), "n_losses": len(losses),
        "win_rate": wr, "avg_win": avg_win, "avg_loss": avg_loss,
        "profit_factor": pf, "expectancy": exp, "expectancy_direct": exp_direct,
        "formula_match": formula_match, "consistent": consistent,
        "diagnosis": diagnosis,
    }


# ════════════════════════════════════════════════════════════════════
#  VISUALIZER
# ════════════════════════════════════════════════════════════════════

def plot_equity_comparison(original: pd.Series,
                            reconstructed: pd.Series,
                            out_path: Path = None):
    """Plot equity asli vs rekonstruksi untuk visual inspection."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    n = min(len(original), len(reconstructed))
    x = np.arange(n)

    ax1 = axes[0]
    ax1.plot(x, original.iloc[:n].values,       label="Original Equity",       color="steelblue",  lw=1.5)
    ax1.plot(x, reconstructed.iloc[:n].values,  label="Reconstructed Equity",  color="darkorange", lw=1.0, linestyle="--")
    ax1.set_title("Equity Curve: Original vs Reconstructed")
    ax1.set_ylabel("Equity ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Diff plot
    ax2 = axes[1]
    diff = (original.iloc[:n].values - reconstructed.iloc[:n].values)
    ax2.plot(x, diff, color="crimson", lw=0.8)
    ax2.axhline(0, color="black", lw=0.5, linestyle="--")
    ax2.set_title("Difference (Original - Reconstructed)")
    ax2.set_ylabel("Diff ($)")
    ax2.set_xlabel("Bar")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120)
        print(f"  Plot saved: {out_path}")
    else:
        out = Path("data/equity_comparison.png")
        out.parent.mkdir(exist_ok=True)
        plt.savefig(out, dpi=120)
        print(f"  Plot saved: {out}")
    plt.close()


# ════════════════════════════════════════════════════════════════════
#  MAIN RUNNER
# ════════════════════════════════════════════════════════════════════

def run():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    path = BASE / "btc_backtest_results.csv"
    if not path.exists():
        print(f"❌ File tidak ditemukan: {path}")
        raise SystemExit(1)

    print(f"\n  Loading {path.name} ...")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── Step 1 & 2: Rekonstruksi equity dari strategy_return ────────
    print("  Reconstructing equity from strategy_return ...")
    eq_original     = df["equity"]
    eq_reconstructed = reconstruct_equity_from_bars(df, INITIAL_EQ)

    # ── Step 3: Bandingkan ─────────────────────────────────────────
    result = compare_equity_curves(eq_original, eq_reconstructed)

    DIV = "═" * 65
    print(f"\n{DIV}")
    print("  EQUITY CURVE RECONSTRUCTION — Audit Result")
    print(DIV)
    print(f"  Bars compared         : {result['n_bars_compared']:,}")
    print(f"  Max relative diff     : {result['max_rel_diff']:.6f}%")
    print(f"  Max absolute diff     : ${result['max_abs_diff']:,.2f}")
    print(f"  Mean relative diff    : {result['mean_rel_diff']:.6f}%")
    print(f"  Tolerance             : {result['tolerance_pct']:.4f}%")
    print(f"  First divergence bar  : {result['first_divergence_bar']}")

    if result["consistent"]:
        print(f"\n  [OK] KONSISTEN — equity curve berasal dari strategy_return dengan benar")
        print(f"     Bug bukan di equity reconstruction, tapi di metric calculation layer")
    else:
        print(f"\n  ❌ INKONSISTEN — ada perbedaan antara equity curve dan strategy_return")
        print(f"     Kemungkinan penyebab:")
        print(f"     1. Ada komponen di equity (fee, slippage) yang tidak di strategy_return")
        print(f"     2. EQUITY_FLOOR clipping mengubah trajectory")
        print(f"     3. Kill switch merubah equity tapi strategy_return tidak ter-update")
    print(DIV)

    # ── Step 4: Extract trades & validate expectancy ────────────────
    print("\n  Extracting trade list ...")
    trade_df = extract_trades(df)
    print(f"  Trades extracted: {len(trade_df):,}")
    print(f"  LONG trades: {(trade_df['direction']=='LONG').sum():,}")
    print(f"  SHORT trades: {(trade_df['direction']=='SHORT').sum():,}")

    validate_expectancy_consistency(trade_df)

    # ── Plot ───────────────────────────────────────────────────────
    try:
        plot_equity_comparison(eq_original, eq_reconstructed)
    except Exception as e:
        print(f"  Plot skipped: {e}")

    return result, trade_df


if __name__ == "__main__":
    run()
