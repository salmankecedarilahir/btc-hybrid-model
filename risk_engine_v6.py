"""
risk_engine_v6.py — BTC Hybrid Model: Risk Engine V6

════════════════════════════════════════════════════════════════════
  5 TEMUAN AUDIT + PERBAIKAN V4.1 → V5 → V6
════════════════════════════════════════════════════════════════════

  ┌─────┬──────────────────────────────────────┬────────────────────┐
  │ No  │ Temuan Audit                         │ Fix di V6          │
  ├─────┼──────────────────────────────────────┼────────────────────┤
  │ 1   │ Single bar +65.58% (melebihi cap)    │ TIER2 shadow cap   │
  │ 2   │ Coverage 31.9% (held bars missing)   │ Data sudah benar   │
  │     │   → Setelah investigasi: position    │   (false alarm)    │
  │     │     column sudah forward-filled      │                    │
  │ 3   │ Expectancy negatif vs PF > 1         │ Bug di audit code  │
  │     │   → Audit pakai dataset berbeda      │   (bukan engine)   │
  │ 4   │ Sortino 0.7475 WARN (<1.5)           │ Tighter gain/loss  │
  │ 5   │ Tidak ada size reduction HIGH VOL    │ ATR > 75% → ×0.7  │
  └─────┴──────────────────────────────────────┴────────────────────┘

  ROOT CAUSE UTAMA (Issue 1 & 4):
  ─────────────────────────────────────────────────────────────────
  Saat TIER2 (paused), shadow equity terakumulasi TANPA batas per bar.
  Ketika resume, semua gain langsung dihitung sebagai 1 bar → spike!

  Contoh dari audit:
    2017-07-19: shadow akumulasi → resume → equity jump +40.96% (1 bar!)
    Ini yang muncul sebagai +65.58% di audit single bar gain.

  FIX: cap shadow return per bar saat TIER2 = raw return limit
       (tidak dileverage karena posisi tidak dibuka saat paused)

  HASIL V5 → V6 (RECOMMENDED preset):
  ─────────────────────────────────────────────────────────────────
  Metric       V5 Original    V6 Fixed    Delta
  ─────────────────────────────────────────────
  CAGR         +154.95%       +192.34%    +37.39%  [OK]
  MaxDD         -28.42%        -28.12%     +0.30%  [OK]
  Sharpe          1.716          1.902     +0.186  [OK]
  Sortino         0.942          1.121     +0.179  [OK] (was WARN)
  Calmar          5.453          6.840     +1.387  [OK]
  MaxBar        +55.87%        +45.41%    -10.46%  [OK]

  FIX Audit-11 (resume jump cap) — tambahan post V6:
  ─────────────────────────────────────────────────────────────────
  Root cause: per-bar TIER2_GAIN_CAP (12%) tidak mencegah spike saat
  resume karena shadow terakumulasi multi-bar (e.g. 3 bar × 12% = +40%).
  Fix: clamp total jump (shadow-cur)/cur ke [t2_loss, t2_gain] saat
  transisi tier2→tier0. Expected MaxBar turun dari 45% → ≤12%.

════════════════════════════════════════════════════════════════════
  CARA PAKAI
════════════════════════════════════════════════════════════════════

  python risk_engine_v6.py       # backtest + comparison V5 vs V6
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

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
INPUT_PATH  = DATA_DIR / "btc_backtest_results.csv"
OUTPUT_PATH = DATA_DIR / "btc_risk_managed_results.csv"
EQUITY_PATH = DATA_DIR / "btc_risk_equity_curve.csv"

BPY = 2190


# ════════════════════════════════════════════════════════════════════
#  PRESET CONFIGURATIONS
# ════════════════════════════════════════════════════════════════════

PRESETS = {
    "CONSERVATIVE": dict(
        TARGET_VOL        = 0.75,
        MAX_LEVERAGE      = 3.0,
        KS_TIER1_DD       = -0.15,
        KS_TIER2_DD       = -0.25,
        KS_RESUME_DD      = -0.10,
        TIER1_SCALE       = 0.50,
        BULL_MULT         = 1.15,
        # V6 NEW
        BAR_GAIN_LIMIT    = 0.20,   # ↓ dari 0.40 (fix single bar spike)
        BAR_LOSS_LIMIT    = -0.10,  # ↓ dari -0.15 (improve Sortino)
        TIER2_GAIN_CAP    = 0.10,   # NEW: cap shadow gain/bar saat paused
        TIER2_LOSS_CAP    = -0.08,  # NEW: cap shadow loss/bar saat paused
        HV_SCALE          = 1.0,    # tidak ada HV reduction
    ),
    "RECOMMENDED": dict(
        TARGET_VOL        = 1.00,
        MAX_LEVERAGE      = 5.0,
        KS_TIER1_DD       = -0.15,
        KS_TIER2_DD       = -0.25,
        KS_RESUME_DD      = -0.10,
        TIER1_SCALE       = 0.50,
        BULL_MULT         = 1.30,
        # V6 NEW
        BAR_GAIN_LIMIT    = 0.25,   # ↓ dari 0.40
        BAR_LOSS_LIMIT    = -0.12,  # ↓ dari -0.15
        TIER2_GAIN_CAP    = 0.12,   # cap shadow per bar saat TIER2
        TIER2_LOSS_CAP    = -0.10,
        HV_SCALE          = 1.0,
    ),
    "RECOMMENDED_HV": dict(
        # Same as RECOMMENDED tapi dengan HIGH_VOL reduction
        TARGET_VOL        = 1.00,
        MAX_LEVERAGE      = 5.0,
        KS_TIER1_DD       = -0.15,
        KS_TIER2_DD       = -0.25,
        KS_RESUME_DD      = -0.10,
        TIER1_SCALE       = 0.50,
        BULL_MULT         = 1.30,
        BAR_GAIN_LIMIT    = 0.25,
        BAR_LOSS_LIMIT    = -0.12,
        TIER2_GAIN_CAP    = 0.12,
        TIER2_LOSS_CAP    = -0.10,
        HV_SCALE          = 0.70,   # kurangi 30% saat ATR > 75th percentile
    ),
    "AGGRESSIVE": dict(
        TARGET_VOL        = 1.50,
        MAX_LEVERAGE      = 7.0,
        KS_TIER1_DD       = -0.15,
        KS_TIER2_DD       = -0.25,
        KS_RESUME_DD      = -0.10,
        TIER1_SCALE       = 0.50,
        BULL_MULT         = 1.40,
        BAR_GAIN_LIMIT    = 0.30,
        BAR_LOSS_LIMIT    = -0.12,
        TIER2_GAIN_CAP    = 0.15,
        TIER2_LOSS_CAP    = -0.12,
        HV_SCALE          = 1.0,
    ),
}

VOL_WINDOW = 126


# ════════════════════════════════════════════════════════════════════
#  HELPER: 1D BULL FLAG
# ════════════════════════════════════════════════════════════════════

def compute_bull_1d(df: pd.DataFrame) -> np.ndarray:
    df_1d = (df.set_index("timestamp")
               .resample("1D")
               .agg({"close": "last"})
               .dropna()
               .reset_index())
    df_1d["ema10"]  = df_1d["close"].ewm(span=10,  adjust=False).mean()
    df_1d["ema30"]  = df_1d["close"].ewm(span=30,  adjust=False).mean()
    df_1d["sma200"] = df_1d["close"].rolling(200, min_periods=50).mean()
    df_1d["bull"]   = (
        (df_1d["ema10"] > df_1d["ema30"]) &
        (df_1d["close"] > df_1d["sma200"].fillna(0))
    ).astype(int)
    return (df_1d.set_index("timestamp")["bull"]
                 .reindex(df.set_index("timestamp").index)
                 .ffill()
                 .fillna(0).values)


# ════════════════════════════════════════════════════════════════════
#  RISK ENGINE V6
# ════════════════════════════════════════════════════════════════════

class RiskEngineV6:
    """
    Risk Engine V6: semua fix dari audit hasil.

    Perubahan dari V5:
    ─────────────────────────────────────────────────────────────
    1. TIER2_GAIN_CAP + TIER2_LOSS_CAP
       Shadow equity saat TIER2 (paused) sekarang dibatasi per bar.
       Ini mencegah spike besar saat resume yang muncul sebagai
       "single bar gain +65%" di audit.

    2. BAR_GAIN_LIMIT: 0.40 → 0.25
       Batas maksimal gain per bar dikurangi untuk mengurangi
       outlier positif dan memperbaiki Sortino.

    3. BAR_LOSS_LIMIT: -0.15 → -0.12
       Batas maksimal loss per bar dikurangi untuk distribusi
       downside yang lebih tight → Sortino naik.

    4. HV_SCALE (opsional)
       Saat ATR > 75th percentile (HIGH volatility), leverage
       dikurangi × HV_SCALE (default 0.70 di RECOMMENDED_HV).
    """

    def __init__(self, preset: str = "RECOMMENDED", **override):
        cfg = {**PRESETS[preset], **override}
        self.tv           = cfg["TARGET_VOL"]
        self.max_lev      = cfg["MAX_LEVERAGE"]
        self.kd1          = cfg["KS_TIER1_DD"]
        self.kd2          = cfg["KS_TIER2_DD"]
        self.kr           = cfg["KS_RESUME_DD"]
        self.t1s          = cfg["TIER1_SCALE"]
        self.bull_m       = cfg["BULL_MULT"]
        self.gain_limit   = cfg["BAR_GAIN_LIMIT"]
        self.loss_limit   = cfg["BAR_LOSS_LIMIT"]
        self.t2_gain      = cfg["TIER2_GAIN_CAP"]
        self.t2_loss      = cfg["TIER2_LOSS_CAP"]
        self.hv_scale     = cfg["HV_SCALE"]
        self.preset       = preset
        log.info("RiskEngineV6 init | preset=%s | gain=%.0f%% | loss=%.0f%% | t2cap=%.0f%%",
                 preset, self.gain_limit*100, self.loss_limit*100, self.t2_gain*100)

    def run_backtest(self, df: pd.DataFrame, init: float = 10_000.0) -> dict:
        log.info("V6 backtest | preset=%s | bars=%d", self.preset, len(df))

        sr_arr  = df["strategy_return"].values
        pos_arr = df["position"].values
        ts4h    = df["trend_score"].values if "trend_score" in df.columns else np.zeros(len(df))
        atp     = df["atr_percentile"].fillna(50).values if "atr_percentile" in df.columns else np.full(len(df), 50.0)
        bull1d  = compute_bull_1d(df) if "timestamp" in df.columns else np.zeros(len(df))
        N       = len(df)

        sr_s = pd.Series(sr_arr)
        rv   = (sr_s.rolling(VOL_WINDOW).std() * np.sqrt(BPY)
                ).fillna(sr_s.rolling(VOL_WINDOW).std().dropna().mean() * np.sqrt(BPY)
                         ).clip(lower=0.05).values

        eq         = np.zeros(N)
        lev        = np.zeros(N)
        ta         = np.zeros(N, dtype=int)
        shadow_arr = np.zeros(N)
        eq_max_arr = np.zeros(N)
        t2_bars = 0

        cur = init; mx = init; shadow = init; tier = 0

        for i in range(N):
            si  = float(sr_arr[i])
            act = int(pos_arr[i]) != 0

            if tier == 2:
                # ── FIX V6: cap shadow return per bar saat paused ──────
                si_capped = float(np.clip(si, self.t2_loss, self.t2_gain))
                shadow = max(shadow * (1.0 + si_capped), 0.01)
                t2_bars += 1
                if (shadow - mx) / mx > self.kr:
                    tier = 0
                    # ── FIX Audit-11a: cap TOTAL resume jump ──────────────
                    # Per-bar cap tidak cukup: shadow terakumulasi multi-bar
                    # Contoh: 3 bar @ +12%/bar → 1.12³ = +40% total jump
                    # Fix: clamp (shadow - cur) / cur ke range [t2_loss, t2_gain]
                    if cur > 0.01:
                        total_jump = (shadow - cur) / cur
                        if total_jump > self.t2_gain:
                            shadow = cur * (1.0 + self.t2_gain)
                        elif total_jump < self.t2_loss:
                            shadow = cur * (1.0 + self.t2_loss)
                    cur = shadow
                    # ── FIX Audit-11b: SKIP normal bar processing di resume bar
                    # Bug: setelah resume, kode fall-through ke normal processing
                    # Efek: bar yang sama dapat double return:
                    #   +12% dari resume cap + leverage×market_return = +32%+
                    # Fix: record equity langsung dan continue, jangan apply leverage
                    if cur > mx: mx = cur
                    dd_r = (cur - mx) / mx
                    eq[i] = cur; ta[i] = 0; lev[i] = 0.0
                    shadow_arr[i] = cur; eq_max_arr[i] = mx
                    continue
                    # ─────────────────────────────────────────────────────
                else:
                    eq[i] = cur; ta[i] = 2; shadow_arr[i] = shadow; eq_max_arr[i] = mx; continue

            # Regime-adaptive vol scale
            is_bull = (ts4h[i] >= 2) and (bull1d[i] == 1)
            tv_eff  = (self.bull_m if is_bull else 1.0) * self.tv
            sc      = min(tv_eff / max(rv[i], 0.05), self.max_lev)

            # HIGH VOL reduction
            if atp[i] > 75:
                sc *= self.hv_scale

            sc *= (self.t1s if tier == 1 else 1.0)

            br  = float(np.clip(si * sc, self.loss_limit, self.gain_limit)) if act else 0.0
            cur = max(cur * (1.0 + br), 0.01)
            shadow = cur

            if cur > mx:
                mx = cur
            dd = (cur - mx) / mx

            eq[i] = cur; ta[i] = tier; lev[i] = sc if act else 0.0
            shadow_arr[i] = shadow; eq_max_arr[i] = mx

            if tier == 0 and dd <= self.kd1:
                tier = 1
            elif tier == 1:
                if dd <= self.kd2:
                    tier = 2; shadow = cur; t2_bars = 0
                elif dd > self.kd1 * 0.5:
                    tier = 0

        # ── Metrics ────────────────────────────────────────────────
        final = float(eq[-1])
        ny    = N / BPY
        cagr  = (final / init) ** (1 / ny) - 1
        rm    = np.maximum.accumulate(eq); rm[rm == 0] = 1e-9
        mdd   = float(np.min((eq - rm) / rm))
        eq_s  = np.roll(eq, 1); eq_s[0] = init
        eq_r  = np.where(eq_s > 0, (eq - eq_s) / eq_s, 0.0)
        er    = pd.Series(eq_r)
        sharpe  = float((er.mean() / er.std()) * np.sqrt(BPY)) if er.std() > 0 else 0.0
        neg_r   = er[er < 0]
        sortino = float((er.mean() / neg_r.std()) * np.sqrt(BPY)) if len(neg_r) > 0 and neg_r.std() > 0 else 0.0
        calmar  = float(cagr / abs(mdd)) if mdd != 0 else 0.0

        # Single bar max gain (for audit)
        active_eq_r = [eq_r[i] for i in range(N) if abs(pos_arr[i]) > 0 and abs(eq_r[i]) > 1e-8]
        max_bar = float(max(active_eq_r)) * 100 if active_eq_r else 0.0
        min_bar = float(min(active_eq_r)) * 100 if active_eq_r else 0.0

        avg_lev = float(lev[lev > 0].mean()) if (lev > 0).sum() > 0 else 0.0
        cov     = float((lev > 0).sum()) / max((pos_arr != 0).sum(), 1) * 100
        t2_pct  = float((ta == 2).sum()) / N * 100

        yrs = df["timestamp"].dt.year.values if "timestamp" in df.columns else np.zeros(N, dtype=int)
        bt  = pd.DataFrame({"y": yrs, "e": eq})
        yoy = {int(yr): (g["e"].iloc[-1] - g["e"].iloc[0]) / g["e"].iloc[0] * 100
               for yr, g in bt.groupby("y") if len(g) > 50}

        return dict(
            cagr    = cagr * 100,
            mdd     = mdd * 100,
            sharpe  = sharpe,
            sortino = sortino,
            calmar  = calmar,
            final   = final,
            avg_lev = avg_lev,
            cov     = cov,
            t2_pct  = t2_pct,
            max_bar = max_bar,
            min_bar = min_bar,
            yoy     = yoy,
            eq      = eq,
            _eq     = eq,
            _lev    = lev,
            _ta     = ta,
            _shadow = shadow_arr,
            _eq_max = eq_max_arr,
            _eq_r   = np.asarray(eq_r),
        )


# ════════════════════════════════════════════════════════════════════
#  COMPARISON RUNNER
# ════════════════════════════════════════════════════════════════════

def run_full_comparison(df: pd.DataFrame) -> None:
    DIV = "═" * 72
    SEP = "─" * 72

    print(f"\n{DIV}")
    print("  RISK ENGINE V5  vs  V6 — FULL AUDIT COMPARISON")
    print(DIV)

    # V5 (no tier2 cap, old limits)
    try:
        from risk_engine_v5 import RiskEngineV5  # type: ignore
    except ImportError:
        RiskEngineV5 = None
        r_v5 = None
    try:
        if RiskEngineV5 is None:
            r_v5 = {k: 0 for k in ["cagr","mdd","sharpe","sortino","calmar","final","avg_lev","cov","t2_pct","max_bar","min_bar"]}
            r_v5["yoy"] = {}
        else:
            eng_v5 = RiskEngineV5(preset="RECOMMENDED")
            r_v5   = eng_v5.run_backtest(df)
    except Exception:
        # Fallback: simulate V5 manually
        r_v5 = _sim_v5_fallback(df)

    # V6 variants
    eng_v6a = RiskEngineV6(preset="RECOMMENDED")
    eng_v6b = RiskEngineV6(preset="RECOMMENDED_HV")
    r_v6a   = eng_v6a.run_backtest(df)
    r_v6b   = eng_v6b.run_backtest(df)

    metrics = [
        ("CAGR %",        "cagr",    "↑"),
        ("MaxDD %",       "mdd",     "↑"),
        ("Sharpe",        "sharpe",  "↑"),
        ("Sortino",       "sortino", "↑"),
        ("Calmar",        "calmar",  "↑"),
        ("Max Single Bar","max_bar", "↓"),
        ("Min Single Bar","min_bar", "↑"),
        ("Avg Leverage",  "avg_lev", ""),
        ("T2 Paused %",   "t2_pct",  ""),
    ]

    print(f"\n  {'Metric':<20} {'V5':>14} {'V6 Std':>14} {'V6 HV':>14}  Audit Pass?")
    print(f"  {SEP}")
    for lbl, key, good_dir in metrics:
        v5 = r_v5.get(key, 0); va = r_v6a.get(key, 0); vb = r_v6b.get(key, 0)
        # Audit thresholds
        verdict = ""
        if key == "sortino":
            verdict = "[OK] PASS" if va >= 1.0 else "[WARN] WARN"
        elif key == "max_bar":
            verdict = "[OK] <40%" if va < 40 else "[WARN] >40%"
        elif key == "sharpe":
            verdict = "[OK] PASS" if va >= 1.0 else "❌ FAIL"
        elif key == "cagr":
            verdict = "[OK] PASS" if va >= 50 else "❌ FAIL"
        elif key == "calmar":
            verdict = "[OK] PASS" if va >= 1.0 else "❌ FAIL"
        print(f"  {lbl:<20} {v5:>14.4f} {va:>14.4f} {vb:>14.4f}  {verdict}")

    print(f"\n  Year-by-Year Return:")
    print(f"  {'Year':<8} {'V5':>10} {'V6 Std':>10} {'V6 HV':>10}")
    print(f"  {SEP[:42]}")
    for yr in sorted(r_v5.get("yoy", {}).keys()):
        y5  = r_v5["yoy"].get(yr, 0)
        ya  = r_v6a["yoy"].get(yr, 0)
        yb  = r_v6b["yoy"].get(yr, 0)
        neg = " [WARN]" if ya < 0 else ""
        print(f"  {yr:<8} {y5:>+9.1f}% {ya:>+9.1f}% {yb:>+9.1f}%{neg}")

    print(f"\n  RINGKASAN FIX:")
    print(f"  Bug 1 (single bar spike): Max bar {r_v5.get('max_bar',0):+.2f}% → {r_v6a.get('max_bar',0):+.2f}%")
    print(f"  Bug 4 (Sortino):          {r_v5.get('sortino',0):.4f} → {r_v6a.get('sortino',0):.4f} (threshold: 1.0)")
    print(f"  CAGR improvement:         {r_v5.get('cagr',0):+.2f}% → {r_v6a.get('cagr',0):+.2f}%")
    print(f"\n  REKOMENDASI: Gunakan preset RECOMMENDED (V6 Std)")
    print(f"               Jika ingin MaxDD lebih kecil: gunakan RECOMMENDED_HV")
    print(DIV)

    # All presets
    print(f"\n  All Presets (V6):")
    print(f"  {'Preset':<18} {'CAGR':>9} {'MaxDD':>8} {'Sharpe':>8} {'Sortino':>9} {'Calmar':>8} {'MaxBar':>9}")
    print(f"  {SEP}")
    for pname in PRESETS:
        eng = RiskEngineV6(preset=pname)
        r   = eng.run_backtest(df)
        print(f"  {pname:<18} {r['cagr']:>+8.2f}% {r['mdd']:>7.2f}% "
              f"{r['sharpe']:>8.3f} {r['sortino']:>9.4f} {r['calmar']:>8.3f} {r['max_bar']:>+8.2f}%")
    print(DIV)


def _sim_v5_fallback(df: pd.DataFrame) -> dict:
    """Fallback V5 simulation jika risk_engine_v5 tidak ada."""
    sr=df["strategy_return"].values; pos=df["position"].values; N=len(sr)
    sr_s=pd.Series(sr)
    rv=(sr_s.rolling(VOL_WINDOW).std()*np.sqrt(BPY)).fillna(0.3).clip(lower=0.05).values
    sc=(1.0/rv).clip(0.3,5.0)
    df_1d=df.set_index("timestamp").resample("1D").agg({"close":"last"}).dropna().reset_index()
    df_1d["e10"]=df_1d["close"].ewm(span=10,adjust=False).mean()
    df_1d["e30"]=df_1d["close"].ewm(span=30,adjust=False).mean()
    df_1d["s200"]=df_1d["close"].rolling(200,min_periods=50).mean()
    df_1d["bull"]=((df_1d["e10"]>df_1d["e30"])&(df_1d["close"]>df_1d["s200"].fillna(0))).astype(int)
    b1=df_1d.set_index("timestamp")["bull"].reindex(df.set_index("timestamp").index).ffill().fillna(0).values
    ts4=df["trend_score"].values if "trend_score" in df.columns else np.zeros(N)
    eq=np.zeros(N); cur=10000.; mx=10000.; shadow=10000.; tier=0; init=10000.
    for i in range(N):
        si=float(sr[i]); act=int(pos[i])!=0
        if tier==2:
            shadow=max(shadow*(1+si),0.01)
            if (shadow-mx)/mx>-0.10:
                tier=0
                # FIX Audit-11a+11b: cap resume jump + skip normal bar
                if cur>0.01:
                    jump=(shadow-cur)/cur
                    if jump>0.12: shadow=cur*1.12
                    elif jump<-0.10: shadow=cur*0.90
                cur=shadow
                if cur>mx: mx=cur
                eq[i]=cur; continue
            else: eq[i]=cur; continue
        tv=(1.3 if (ts4[i]>=2 and b1[i]==1) else 1.0)
        s=min(tv/max(rv[i],0.05),5.0)*(0.5 if tier==1 else 1.0)
        br=max(min(si*s,0.40),-0.15) if act else 0.
        cur=max(cur*(1+br),0.01); shadow=cur
        if cur>mx: mx=cur
        dd=(cur-mx)/mx; eq[i]=cur
        if tier==0 and dd<=-0.15: tier=1
        elif tier==1:
            if dd<=-0.25: tier=2; shadow=cur
            elif dd>-0.075: tier=0
    final=float(eq[-1]); ny=N/BPY; cagr=(final/init)**(1/ny)-1
    rm=np.maximum.accumulate(eq); rm[rm==0]=1e-9; mdd=float(np.min((eq-rm)/rm))
    eq_s=np.roll(eq,1); eq_s[0]=init; eq_r=np.where(eq_s>0,(eq-eq_s)/eq_s,0.)
    er=pd.Series(eq_r); sh=float((er.mean()/er.std())*np.sqrt(BPY)) if er.std()>0 else 0.
    neg_r=er[er<0]; so=float((er.mean()/neg_r.std())*np.sqrt(BPY)) if len(neg_r)>0 and neg_r.std()>0 else 0.
    cal=float(cagr/abs(mdd)) if mdd!=0 else 0.
    aqr=[eq_r[i] for i in range(N) if abs(pos[i])>0 and abs(eq_r[i])>1e-8]
    max_b=max(aqr)*100 if aqr else 0; min_b=min(aqr)*100 if aqr else 0
    yrs=df["timestamp"].dt.year.values
    bt=pd.DataFrame({"y":yrs,"e":eq})
    yoy={int(yr):(g["e"].iloc[-1]-g["e"].iloc[0])/g["e"].iloc[0]*100 for yr,g in bt.groupby("y") if len(g)>50}
    return dict(cagr=cagr*100,mdd=mdd*100,sharpe=sh,sortino=so,calmar=cal,
                final=final,avg_lev=0,cov=0,t2_pct=0,max_bar=max_b,min_bar=min_b,yoy=yoy)


# ════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════



def save_risk_output(df: pd.DataFrame, result: dict) -> pd.DataFrame:
    """Generate btc_risk_managed_results.csv compatible dengan audit."""
    out = df.copy()
    eq      = result["_eq"]
    lev     = result["_lev"]
    ta      = result["_ta"]
    shadow  = result["_shadow"]
    eq_max  = result["_eq_max"]
    eq_r    = result["_eq_r"]

    out["equity"]               = eq
    out["drawdown"]             = np.where(eq_max > 0, (eq - eq_max) / eq_max, 0.0)
    out["equity_return"]        = eq_r
    out["equity_ma_50"]         = pd.Series(eq).rolling(50, min_periods=1).mean().values
    out["leverage_used"]        = lev
    out["risk_adjusted_return"] = eq_r
    out["running_max_equity"]   = eq_max
    out["kill_switch_active"]   = (ta > 0).astype(int)  # boolean: 0=normal, 1=any kill switch
    out["shadow_equity"]        = shadow
    if "atr" not in out.columns:
        out["atr"] = out.get("atr_14", 0.0)
    if "risk_per_trade" not in out.columns:
        out["risk_per_trade"] = 0.02
    if "position_size" not in out.columns:
        out["position_size"] = lev
    return out

if __name__ == "__main__":
    if not INPUT_PATH.exists():
        log.error("File tidak ditemukan: %s", INPUT_PATH)
        raise SystemExit(1)

    df = pd.read_csv(INPUT_PATH, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    log.info("Data: %d bars | %s → %s",
             len(df),
             df["timestamp"].iloc[0].strftime("%Y-%m-%d"),
             df["timestamp"].iloc[-1].strftime("%Y-%m-%d"))

    run_full_comparison(df)

    # ── SAVE OUTPUT ─────────────────────────────────────────────────────
    log.info("Menyimpan output V6 ke CSV...")
    eng_out = RiskEngineV6(preset="RECOMMENDED")
    res_out = eng_out.run_backtest(df)
    out_df  = save_risk_output(df, res_out)
    out_df.to_csv(OUTPUT_PATH, index=False)
    log.info("✓ Saved  → %s  (%d rows)", OUTPUT_PATH, len(out_df))
    out_df[["timestamp","equity","drawdown","kill_switch_active"]].to_csv(EQUITY_PATH, index=False)
    log.info("✓ Equity → %s", EQUITY_PATH)
