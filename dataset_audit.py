"""
╔══════════════════════════════════════════════════════════════════════╗
║   dataset_audit.py — BTC Hybrid Model V7                           ║
║   BAGIAN 1: Dataset Integrity Audit                                ║
╠══════════════════════════════════════════════════════════════════════╣
║  Checks:                                                            ║
║    1. Trade list consistency (entry/exit pairing)                  ║
║    2. Equity curve reconstruction & reconciliation                  ║
║    3. Duplicate trade detection                                     ║
║    4. Trade overlap detection (LONG→SHORT without FLAT)            ║
║    5. Return calculation correctness (no lookahead)                ║
║    6. Timestamp continuity (no gaps, no duplicates)                ║
║    7. NaN / Inf contamination scan                                  ║
║    8. Data alignment (backtest vs risk_managed consistency)        ║
╠══════════════════════════════════════════════════════════════════════╣
║  Cara pakai:                                                        ║
║    python dataset_audit.py                                         ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data"
BT_PATH   = DATA_DIR / "btc_backtest_results.csv"
RISK_PATH = DATA_DIR / "btc_risk_managed_results.csv"
SIG_PATH  = DATA_DIR / "btc_trading_signals.csv"

BARS_PER_YEAR = 2190
INIT          = 10_000.0

DIV = "═" * 65
SEP = "─" * 65


# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════

def ok(msg):  print(f"  [OK] {msg}")
def warn(msg): print(f"  [WARN]️  {msg}")
def err(msg):  print(f"  ❌ {msg}")


def load(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)
    log.info("Loaded %s: %d rows × %d cols", label, len(df), len(df.columns))
    return df


def extract_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract trade list from bar-level dataframe.
    Uses compound accumulation (correct: multiplicative).
    Returns DataFrame with one row per trade.
    """
    pos    = df["position"].values
    sr     = df["strategy_return"].fillna(0).values
    sig    = df["signal"].values if "signal" in df.columns else np.full(len(df), "UNK")
    ts     = df["timestamp"].values if "timestamp" in df.columns else np.arange(len(df))

    trades = []
    in_tr  = False
    mult   = 1.0
    t_entry = None
    p_sig   = None
    i_entry = 0

    for i in range(len(df)):
        if not in_tr:
            if pos[i] != 0:
                in_tr   = True
                mult    = 1.0 + float(sr[i])
                t_entry = ts[i]
                p_sig   = sig[i]
                i_entry = i
        else:
            if pos[i] == 0 or (sig[i] not in ("NONE", p_sig) and sig[i] != "UNK" and pos[i] != 0):
                trade_ret = mult - 1.0
                trades.append({
                    "entry_ts":   t_entry,
                    "exit_ts":    ts[i],
                    "signal":     p_sig,
                    "ret":        trade_ret,
                    "n_bars":     i - i_entry,
                    "entry_idx":  i_entry,
                    "exit_idx":   i,
                })
                in_tr = False
                mult  = 1.0
                if pos[i] != 0:
                    in_tr   = True
                    mult    = 1.0 + float(sr[i])
                    t_entry = ts[i]
                    p_sig   = sig[i]
                    i_entry = i
            else:
                mult *= (1.0 + float(sr[i]))

    if in_tr:
        trade_ret = mult - 1.0
        trades.append({
            "entry_ts":  t_entry,
            "exit_ts":   ts[-1],
            "signal":    p_sig,
            "ret":       trade_ret,
            "n_bars":    len(df) - i_entry,
            "entry_idx": i_entry,
            "exit_idx":  len(df) - 1,
            "open":      True,
        })

    return pd.DataFrame(trades) if trades else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════
#  CHECK 1: Trade List Consistency
# ══════════════════════════════════════════════════════════════════

def check_trade_list(df: pd.DataFrame) -> dict:
    print(f"\n{SEP}")
    print("  CHECK 1 — Trade List Consistency")
    print(SEP)

    trades = extract_trades(df)
    if trades.empty:
        err("No trades extracted"); return {"passed": False}

    n_trades = len(trades)
    n_long   = (trades["signal"] == "LONG").sum() if "signal" in trades else 0
    n_short  = (trades["signal"] == "SHORT").sum() if "signal" in trades else 0
    n_open   = trades.get("open", pd.Series(dtype=bool)).sum() if "open" in trades.columns else 0
    avg_bars = trades["n_bars"].mean()

    wins   = trades[trades["ret"] > 0]
    losses = trades[trades["ret"] < 0]
    wr     = len(wins) / n_trades
    pf     = wins["ret"].sum() / abs(losses["ret"].sum()) if len(losses) > 0 else 99.0

    print(f"  Total trades     : {n_trades}")
    print(f"  LONG / SHORT     : {n_long} / {n_short}")
    print(f"  Open (unclosed)  : {n_open}")
    print(f"  Avg bars/trade   : {avg_bars:.1f}")
    print(f"  Win Rate         : {wr*100:.1f}%")
    print(f"  Profit Factor    : {pf:.4f}")

    passed = True

    if n_trades < 30:
        warn(f"Only {n_trades} trades — low for robust stats (need ≥30)")
        passed = False
    else:
        ok(f"{n_trades} trades — sufficient sample size")

    if n_open > 1:
        warn(f"{n_open} open trades at end of data (expected ≤1)")
    else:
        ok("Open trade count OK")

    if avg_bars < 1:
        err("Avg bars/trade < 1 — data extraction issue"); passed = False
    else:
        ok(f"Avg bars/trade = {avg_bars:.0f} (trend-following: normal)")

    if abs(pf - 1.0) < 0.001 and n_trades > 50:
        warn("PF ≈ 1.0 with many trades — possible return contamination")
    else:
        ok(f"PF = {pf:.4f} — consistent with strategy")

    return {"passed": passed, "n_trades": n_trades, "win_rate": wr,
            "profit_factor": pf, "trades_df": trades}


# ══════════════════════════════════════════════════════════════════
#  CHECK 2: Equity Curve Reconstruction
# ══════════════════════════════════════════════════════════════════

def check_equity_reconstruction(df: pd.DataFrame) -> dict:
    print(f"\n{SEP}")
    print("  CHECK 2 — Equity Curve Reconstruction")
    print(SEP)

    ret_col = "equity_return" if "equity_return" in df.columns else "strategy_return"
    ret     = df[ret_col].fillna(0.0).values
    eq_col  = df["equity"].values if "equity" in df.columns else None

    # Reconstruct equity from returns
    eq_recon = INIT * np.cumprod(1.0 + ret)

    passed = True

    if eq_col is not None:
        diff     = np.abs(eq_recon - eq_col)
        max_diff = diff.max()
        max_pct  = (max_diff / eq_col.max()) * 100

        print(f"  Max abs diff     : ${max_diff:,.2f}")
        print(f"  Max pct diff     : {max_pct:.6f}%")
        print(f"  Final recon      : ${eq_recon[-1]:,.2f}")
        print(f"  Final stored     : ${eq_col[-1]:,.2f}")

        if max_pct < 0.01:
            ok("Equity curve reconstruction matches stored (diff < 0.01%)")
        elif max_pct < 0.1:
            warn(f"Small reconstruction diff {max_pct:.4f}% — floating point OK")
        else:
            err(f"Reconstruction diff {max_pct:.2f}% — possible data issue")
            passed = False

        # Check no negative equity
        n_neg = (eq_col < 0).sum()
        if n_neg > 0:
            err(f"{n_neg} bars with negative equity"); passed = False
        else:
            ok("No negative equity bars")

        # Check max drawdown consistency
        peak   = np.maximum.accumulate(eq_col)
        dd     = (eq_col - peak) / peak
        max_dd = dd.min() * 100
        print(f"  Max Drawdown     : {max_dd:.2f}%")
        ok(f"MaxDD = {max_dd:.2f}%")
    else:
        warn("No stored equity column — reconstruction only")
        print(f"  Final equity     : ${eq_recon[-1]:,.2f}")

    return {"passed": passed, "final_equity": float(eq_recon[-1])}


# ══════════════════════════════════════════════════════════════════
#  CHECK 3: Duplicate Trade Detection
# ══════════════════════════════════════════════════════════════════

def check_duplicates(df: pd.DataFrame, trades_df: pd.DataFrame) -> dict:
    print(f"\n{SEP}")
    print("  CHECK 3 — Duplicate Trade Detection")
    print(SEP)

    passed = True

    # Bar-level timestamp duplicates
    n_ts_dup = df["timestamp"].duplicated().sum() if "timestamp" in df.columns else 0
    if n_ts_dup > 0:
        err(f"{n_ts_dup} duplicate timestamps in bar data"); passed = False
    else:
        ok("No duplicate timestamps in bar data")

    # Trade-level entry timestamp duplicates
    if not trades_df.empty and "entry_ts" in trades_df.columns:
        n_tr_dup = trades_df["entry_ts"].duplicated().sum()
        if n_tr_dup > 0:
            warn(f"{n_tr_dup} duplicate trade entry timestamps")
        else:
            ok("No duplicate trade entries")

    # Check for same-timestamp bars with different positions (flip)
    if "position" in df.columns and "timestamp" in df.columns:
        grp = df.groupby("timestamp")["position"].nunique()
        n_multi = (grp > 1).sum()
        if n_multi > 0:
            err(f"{n_multi} timestamps with multiple position values"); passed = False
        else:
            ok("No multi-position timestamps")

    return {"passed": passed, "n_ts_duplicates": n_ts_dup}


# ══════════════════════════════════════════════════════════════════
#  CHECK 4: Trade Overlap Detection
# ══════════════════════════════════════════════════════════════════

def check_trade_overlap(df: pd.DataFrame, trades_df: pd.DataFrame) -> dict:
    print(f"\n{SEP}")
    print("  CHECK 4 — Trade Overlap Detection")
    print(SEP)

    passed = True

    # Direct position flips LONG→SHORT without FLAT
    if "position" in df.columns and "signal" in df.columns:
        pos  = df["position"].values
        sig  = df["signal"].values
        flips = 0
        for i in range(1, len(pos)):
            if pos[i-1] == 1 and pos[i] == -1:
                flips += 1
            elif pos[i-1] == -1 and pos[i] == 1:
                flips += 1
        if flips > 0:
            warn(f"{flips} direct position flips (LONG↔SHORT without FLAT)")
        else:
            ok("No direct position flips")

    # Trade-level overlap check
    if not trades_df.empty and "entry_idx" in trades_df.columns:
        overlaps = 0
        for i in range(1, len(trades_df)):
            prev_exit = trades_df.iloc[i-1]["exit_idx"]
            curr_entry = trades_df.iloc[i]["entry_idx"]
            if curr_entry < prev_exit:
                overlaps += 1
        if overlaps > 0:
            err(f"{overlaps} overlapping trades"); passed = False
        else:
            ok(f"No overlapping trades ({len(trades_df)} trades checked)")

    return {"passed": passed}


# ══════════════════════════════════════════════════════════════════
#  CHECK 5: Return Calculation Correctness (No Lookahead)
# ══════════════════════════════════════════════════════════════════

def check_return_correctness(df: pd.DataFrame) -> dict:
    print(f"\n{SEP}")
    print("  CHECK 5 — Return Calculation & Lookahead Bias")
    print(SEP)

    passed = True

    if all(c in df.columns for c in ["position", "strategy_return", "close"]):
        sr    = df["strategy_return"].fillna(0).values
        pos   = df["position"].values
        close = df["close"].values

        # Market return should be close[i]/close[i-1] - 1
        mret = np.zeros(len(close))
        mret[1:] = close[1:] / close[:-1] - 1.0

        # strategy_return[i] = position[i-1] * market_return[i]  (no lookahead)
        expected = np.roll(pos, 1) * mret
        expected[0] = 0.0

        diff    = np.abs(sr - expected)
        ok_pct  = float((diff < 0.002).mean() * 100)  # 0.2% tolerance for leverage
        n_nonzero = (sr != 0).sum()

        print(f"  Bars matching expected pattern : {ok_pct:.1f}%")
        print(f"  Active bars                    : {n_nonzero:,}")

        # Lookahead bias would show correlation between sr[i] and close[i+1]
        future_corr = np.corrcoef(sr[:-1], close[1:])[0,1]
        print(f"  Corr(sr[t], close[t+1])        : {future_corr:.4f}  (should be ≈ 0)")

        if ok_pct >= 80:
            ok(f"{ok_pct:.1f}% bars match expected return pattern")
        else:
            warn(f"Only {ok_pct:.1f}% bars match — check leverage/slippage adjustments")

        if abs(future_corr) < 0.05:
            ok(f"No lookahead bias detected (future corr = {future_corr:.4f})")
        else:
            err(f"Possible lookahead bias: corr(sr, future_close) = {future_corr:.4f}")
            passed = False

        # Last bar market_return should be 0 (shift(-1) correct)
        last_sr = float(sr[-1])
        print(f"  Last bar strategy_return       : {last_sr:.6f}  (should be 0)")
        if abs(last_sr) < 1e-9:
            ok("Last bar return = 0 (shift correct)")
        else:
            warn(f"Last bar return ≠ 0 ({last_sr:.6f}) — check shift logic")

    else:
        warn("Missing columns for return check (position/strategy_return/close)")

    return {"passed": passed}


# ══════════════════════════════════════════════════════════════════
#  CHECK 6: Timestamp Continuity
# ══════════════════════════════════════════════════════════════════

def check_timestamp_continuity(df: pd.DataFrame) -> dict:
    print(f"\n{SEP}")
    print("  CHECK 6 — Timestamp Continuity")
    print(SEP)

    if "timestamp" not in df.columns:
        warn("No timestamp column"); return {"passed": True}

    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    ts_diff = ts.diff().dropna()

    expected_gap = pd.Timedelta("4h")
    normal  = (ts_diff == expected_gap).sum()
    gaps    = (ts_diff > expected_gap * 2).sum()
    overlap = (ts_diff <= pd.Timedelta(0)).sum()

    total   = len(ts_diff)
    date_range = f"{ts.min().date()} → {ts.max().date()}"

    print(f"  Date range       : {date_range}")
    print(f"  Total bars       : {len(df):,}")
    print(f"  Normal gaps      : {normal:,} / {total:,} ({normal/total*100:.1f}%)")
    print(f"  Large gaps (>8h) : {gaps:,}")
    print(f"  Overlaps (≤0)    : {overlap:,}")

    passed = True
    if overlap > 0:
        err(f"{overlap} overlapping/negative timestamps"); passed = False
    else:
        ok("No overlapping timestamps")

    if gaps > 50:
        warn(f"{gaps} large time gaps (expected for weekends/maintenance)")
    elif gaps > 0:
        ok(f"{gaps} gaps found (weekends/exchange maintenance — expected for crypto)")
    else:
        ok("Consistent 4H bar spacing")

    return {"passed": passed, "n_gaps": gaps, "date_range": date_range}


# ══════════════════════════════════════════════════════════════════
#  CHECK 7: NaN / Inf Contamination
# ══════════════════════════════════════════════════════════════════

def check_nan_inf(df: pd.DataFrame) -> dict:
    print(f"\n{SEP}")
    print("  CHECK 7 — NaN / Inf Contamination Scan")
    print(SEP)

    critical_cols = [c for c in ["position","strategy_return","equity",
                                  "market_return","equity_return","close"] if c in df.columns]
    issues = []

    for col in critical_cols:
        n_nan = df[col].isna().sum()
        n_inf = np.isinf(pd.to_numeric(df[col], errors="coerce").fillna(0)).sum()
        if n_nan > 0 or n_inf > 0:
            issues.append((col, n_nan, n_inf))
            warn(f"{col}: {n_nan} NaN, {n_inf} Inf")
        else:
            ok(f"{col}: clean")

    # Check for extreme return outliers (>50% in single bar)
    if "strategy_return" in df.columns:
        sr = df["strategy_return"].fillna(0).abs()
        n_extreme = (sr > 0.50).sum()
        if n_extreme > 0:
            warn(f"{n_extreme} bars with |return| > 50% — verify risk engine")
        else:
            ok("No extreme return outliers (>50% per bar)")

    passed = len(issues) == 0
    return {"passed": passed, "n_issues": len(issues)}


# ══════════════════════════════════════════════════════════════════
#  CHECK 8: Backtest vs Risk Managed Alignment
# ══════════════════════════════════════════════════════════════════

def check_data_alignment(bt_df: pd.DataFrame, risk_df: pd.DataFrame) -> dict:
    print(f"\n{SEP}")
    print("  CHECK 8 — Backtest vs Risk Managed Alignment")
    print(SEP)

    passed = True

    # Row count
    n_bt   = len(bt_df)
    n_risk = len(risk_df)
    print(f"  Backtest rows    : {n_bt:,}")
    print(f"  Risk managed rows: {n_risk:,}")

    if n_bt != n_risk:
        warn(f"Row count differs: {n_bt} vs {n_risk}")
    else:
        ok(f"Row counts match: {n_bt:,}")

    # Timestamp alignment
    if "timestamp" in bt_df.columns and "timestamp" in risk_df.columns:
        bt_ts   = pd.to_datetime(bt_df["timestamp"], utc=True, errors="coerce")
        risk_ts = pd.to_datetime(risk_df["timestamp"], utc=True, errors="coerce")
        common  = min(len(bt_ts), len(risk_ts))
        match   = (bt_ts.iloc[:common].values == risk_ts.iloc[:common].values).mean()
        print(f"  Timestamp match  : {match*100:.1f}%")
        if match > 0.999:
            ok("Timestamps perfectly aligned")
        else:
            warn(f"Timestamp alignment {match*100:.1f}% — check merge logic")

    # Position alignment
    if "position" in bt_df.columns and "position" in risk_df.columns:
        common  = min(len(bt_df), len(risk_df))
        pos_match = (bt_df["position"].iloc[:common].values ==
                     risk_df["position"].iloc[:common].values).mean()
        print(f"  Position match   : {pos_match*100:.1f}%")
        if pos_match > 0.95:
            ok(f"Positions aligned ({pos_match*100:.1f}%)")
        else:
            warn(f"Position alignment only {pos_match*100:.1f}%")

    return {"passed": passed}


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def run() -> dict:
    print(f"\n{DIV}")
    print("  DATASET INTEGRITY AUDIT — BTC Hybrid Model V7")
    print(f"{DIV}")

    # Load data
    bt_df = load(BT_PATH, "btc_backtest_results")

    risk_df = None
    if RISK_PATH.exists():
        risk_df = load(RISK_PATH, "btc_risk_managed_results")
    else:
        log.warning("risk_managed_results not found — using backtest only")

    primary = risk_df if risk_df is not None else bt_df

    # Run all checks
    r1 = check_trade_list(primary)
    r2 = check_equity_reconstruction(primary)
    r3 = check_duplicates(primary, r1.get("trades_df", pd.DataFrame()))
    r4 = check_trade_overlap(primary, r1.get("trades_df", pd.DataFrame()))
    r5 = check_return_correctness(bt_df)
    r6 = check_timestamp_continuity(primary)
    r7 = check_nan_inf(primary)
    r8 = check_data_alignment(bt_df, risk_df) if risk_df is not None else {"passed": True}

    # Final verdict
    results = [("Trade List Consistency",    r1["passed"]),
               ("Equity Reconstruction",     r2["passed"]),
               ("Duplicate Detection",       r3["passed"]),
               ("Trade Overlap",             r4["passed"]),
               ("Return Correctness",        r5["passed"]),
               ("Timestamp Continuity",      r6["passed"]),
               ("NaN/Inf Scan",              r7["passed"]),
               ("Data Alignment",            r8["passed"])]

    n_pass = sum(1 for _, p in results if p)

    print(f"\n{DIV}")
    print("  DATASET AUDIT FINAL VERDICT")
    print(DIV)
    for name, passed in results:
        print(f"  {'[OK]' if passed else '❌'}  {name}")
    print(SEP)

    if n_pass == len(results):
        verdict = "[OK] DATASET CLEAN — Ready for AI layer"
    elif n_pass >= 6:
        verdict = f"[WARN]️  MOSTLY CLEAN ({n_pass}/{len(results)}) — Review warnings"
    else:
        verdict = f"❌ ISSUES FOUND ({n_pass}/{len(results)}) — Fix before AI layer"

    print(f"  {n_pass}/{len(results)} checks passed")
    print(f"  VERDICT: {verdict}")
    print(f"{DIV}\n")

    return {
        "dataset_audit_score": round(n_pass / len(results) * 100, 1),
        "passed": n_pass == len(results),
        "details": {k: v for k, v in [("r1",r1),("r2",r2),("r3",r3),("r4",r4),
                                        ("r5",r5),("r6",r6),("r7",r7),("r8",r8)]},
        "n_trades": r1.get("n_trades", 0),
        "trade_pf": r1.get("profit_factor", 0),
    }


if __name__ == "__main__":
    run()
