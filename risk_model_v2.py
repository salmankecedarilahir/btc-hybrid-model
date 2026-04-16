"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  risk_model_v2.py — BTC Hybrid AI v2                                       ║
║  Volatility Targeting + Drawdown-Aware Position Sizing                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  WHY KELLY FAILS HERE:                                                     ║
║                                                                            ║
║  Kelly formula: f* = (p*b - q) / b                                        ║
║  where p = win prob, b = win/loss ratio                                    ║
║                                                                            ║
║  Problems in practice:                                                     ║
║  1. Win rate estimated from recent 20 trades = too noisy (std ~±15%)      ║
║  2. Full Kelly = maximum geometric growth but also max drawdown            ║
║     Full Kelly → 50% drawdown is expected at some point                   ║
║  3. When model predictions are noisy (Sharpe 0.16), Kelly says "bet big" ║
║     because it assumes your edge estimate is precise                       ║
║  4. Kelly on correlated trades (4h BTC = autocorrelated) is wrong         ║
║     Kelly assumes independent bets                                         ║
║                                                                            ║
║  VOLATILITY TARGETING (replacement):                                       ║
║  size = (target_annual_vol / realized_vol_10bar) × conviction             ║
║  + drawdown multiplier × regime_cap                                        ║
║                                                                            ║
║  This is used by most institutional quant funds (risk parity, CTAs).      ║
║  Key properties:                                                           ║
║  - Size ∝ 1/vol → large in calm markets, small in volatile markets        ║
║  - No circular dependency on recent P&L                                    ║
║  - Predictable risk (target vol is the constraint)                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional

log = logging.getLogger(__name__)

REGIME_CAPS = {
    0: 1.00,   # BULL  → full size allowed
    1: 0.60,   # BEAR  → reduce to 60%
    2: 0.50,   # CHOP  → reduce to 50%
}

REGIME_NAMES = {0: "BULL", 1: "BEAR", 2: "CHOP"}


class RiskModelV2:
    """
    Production-grade position sizer using volatility targeting.

    Formula:
      raw_size = (target_vol / realized_vol) × conviction × regime_cap

      dd_factor = 1.0                    if DD < soft_limit
                = 0.5                    if soft_limit ≤ DD < hard_limit
                = 0.25                   if DD ≥ hard_limit
                = 0.0 (kill switch)      if DD ≥ kill_limit

      final_size = clip(raw_size × dd_factor, min_size, max_size)

    Parameters
    ----------
    target_vol   : Annual volatility target (default 0.25 = 25%)
    vol_lookback : Bars for realized vol calculation (default 10)
    soft_dd      : Drawdown where size halves (default 0.10 = 10%)
    hard_dd      : Drawdown where size quarters (default 0.20 = 20%)
    kill_dd      : Drawdown kill switch (default 0.35 = 35%)
    max_size     : Maximum position multiplier (default 1.0 = 100%)
    min_size     : Minimum position multiplier (default 0.10)
    """

    def __init__(
        self,
        target_vol  : float = 0.25,
        vol_lookback: int   = 10,
        soft_dd     : float = 0.10,
        hard_dd     : float = 0.20,
        kill_dd     : float = 0.35,
        max_size    : float = 1.00,
        min_size    : float = 0.10,
        bars_per_year: int  = 2190,   # 4h bars
    ):
        self.target_vol   = target_vol
        self.vol_lookback = vol_lookback
        self.soft_dd      = soft_dd
        self.hard_dd      = hard_dd
        self.kill_dd      = kill_dd
        self.max_size     = max_size
        self.min_size     = min_size
        self.bpy          = bars_per_year

        # State tracking
        self._equity_history : list = []
        self._peak_equity    : float = 1.0
        self._current_dd     : float = 0.0

    # ── MAIN INTERFACE ────────────────────────────────────────────────────────

    def compute(
        self,
        close:         pd.Series,
        signals:       pd.Series,        # +1/-1/0
        conviction:    Optional[pd.Series] = None,
        regime:        Optional[pd.Series] = None,
        equity_series: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Compute position sizes for full time series.

        Parameters
        ----------
        close      : Close price series
        signals    : Trading signals (+1/-1/0)
        conviction : Signal conviction 0-1 (from SignalModelV2)
        regime     : Regime labels 0=BULL, 1=BEAR, 2=CHOP
        equity_series : Running equity for drawdown calculation

        Returns
        -------
        DataFrame with sizing breakdown and final position_size_mult
        """
        ret = close.pct_change().fillna(0)
        n   = len(close)
        idx = close.index

        # Defaults
        if conviction is None:
            conviction = pd.Series(0.5, index=idx)
        if regime is None:
            regime = pd.Series(0, index=idx)
        if equity_series is None:
            equity_series = pd.Series(np.ones(n), index=idx)

        conviction = pd.to_numeric(conviction, errors="coerce").fillna(0.5).clip(0, 1)
        regime     = pd.to_numeric(regime, errors="coerce").fillna(0).astype(int)

        out = pd.DataFrame(index=idx)

        # ── Component 1: Volatility scaling ───────────────────────────────────
        realized_vol = ret.rolling(self.vol_lookback).std() * np.sqrt(self.bpy)
        realized_vol = realized_vol.clip(lower=0.01)  # floor at 1% to prevent infinite size

        vol_scale = (self.target_vol / realized_vol).clip(0.10, 2.0)
        out["vol_scale"]    = vol_scale
        out["realized_vol"] = realized_vol

        # ── Component 2: Conviction scaling ───────────────────────────────────
        # Map conviction 0-1 → size 0.5-1.0 (never go to zero based on conviction alone)
        conv_scale = 0.5 + 0.5 * conviction.clip(0, 1)
        out["conv_scale"] = conv_scale

        # ── Component 3: Regime cap ────────────────────────────────────────────
        regime_cap = regime.map(REGIME_CAPS).fillna(0.8)
        out["regime_cap"]  = regime_cap
        out["regime_name"] = regime.map(REGIME_NAMES).fillna("UNKNOWN")

        # ── Component 4: Drawdown factor ──────────────────────────────────────
        peak    = equity_series.expanding().max()
        dd      = (equity_series / peak - 1).clip(-1, 0)
        out["current_drawdown"] = dd

        dd_factor = pd.Series(1.0, index=idx)
        dd_factor = np.where(dd <= -self.kill_dd, 0.00,
                    np.where(dd <= -self.hard_dd,  0.25,
                    np.where(dd <= -self.soft_dd,  0.50, 1.0)))
        out["dd_factor"] = pd.Series(dd_factor, index=idx)

        # ── Combine all components ─────────────────────────────────────────────
        raw_size  = vol_scale * conv_scale * regime_cap * dd_factor
        final     = raw_size.clip(0, self.max_size)

        # Apply minimum only when there's an active signal
        has_signal = signals.abs() > 0
        final = final.where(~has_signal, final.clip(lower=self.min_size))

        # Zero out when no signal
        final = final.where(has_signal, 0.0)

        out["position_size_mult"] = final

        # ── Diagnostics ───────────────────────────────────────────────────────
        active_sizes = final[has_signal]
        if len(active_sizes) > 0:
            log.info(
                "RiskModelV2: mean_size=%.3f  median=%.3f  max=%.3f  "
                "kill_bars=%d  dd_range=[%.1f%%, %.1f%%]",
                active_sizes.mean(),
                active_sizes.median(),
                active_sizes.max(),
                (dd_factor == 0).sum(),
                dd.min() * 100,
                dd.max() * 100,
            )

        return out

    def compute_single(
        self,
        realized_vol_annual: float,
        conviction:          float,
        regime_id:           int,
        current_dd:          float,
    ) -> float:
        """
        Single-bar position size for live trading.

        Parameters
        ----------
        realized_vol_annual : Annualized realized vol (e.g. 0.45 = 45%)
        conviction          : Signal conviction 0-1
        regime_id           : 0=BULL, 1=BEAR, 2=CHOP
        current_dd          : Current drawdown (e.g. -0.08 = -8%)
        """
        realized_vol = max(realized_vol_annual, 0.01)
        vol_scale    = np.clip(self.target_vol / realized_vol, 0.10, 2.0)
        conv_scale   = 0.5 + 0.5 * np.clip(conviction, 0, 1)
        reg_cap      = REGIME_CAPS.get(regime_id, 0.8)

        if current_dd <= -self.kill_dd:
            dd_factor = 0.0
        elif current_dd <= -self.hard_dd:
            dd_factor = 0.25
        elif current_dd <= -self.soft_dd:
            dd_factor = 0.50
        else:
            dd_factor = 1.0

        raw  = vol_scale * conv_scale * reg_cap * dd_factor
        size = np.clip(raw, 0.0, self.max_size)
        if size > 0:
            size = max(size, self.min_size)
        return float(size)

    def get_sizing_report(self, sizing_df: pd.DataFrame) -> dict:
        """
        Summarize sizing behavior for diagnostics.
        """
        active = sizing_df[sizing_df["position_size_mult"] > 0]
        return {
            "n_active_bars"      : len(active),
            "mean_size"          : round(float(active["position_size_mult"].mean()), 3) if len(active) > 0 else 0,
            "median_size"        : round(float(active["position_size_mult"].median()), 3) if len(active) > 0 else 0,
            "kill_switch_bars"   : int((sizing_df["dd_factor"] == 0).sum()),
            "soft_limit_bars"    : int((sizing_df["dd_factor"] == 0.50).sum()),
            "hard_limit_bars"    : int((sizing_df["dd_factor"] == 0.25).sum()),
            "max_realized_vol"   : round(float(sizing_df["realized_vol"].max()), 3),
            "mean_regime_cap"    : round(float(sizing_df["regime_cap"].mean()), 3),
            "max_drawdown"       : round(float(sizing_df["current_drawdown"].min()), 3),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    np.random.seed(42)
    n     = 3000
    dates = pd.date_range("2020-01-01", periods=n, freq="4h")
    price = 30000 * np.exp(np.cumsum(np.random.randn(n)*0.012))
    close = pd.Series(price, index=dates)

    signals    = pd.Series(np.random.choice([-1,0,1], n, p=[0.25,0.50,0.25]), index=dates)
    conviction = pd.Series(np.random.uniform(0.3, 0.9, n), index=dates)
    regime     = pd.Series(np.random.choice([0,1,2], n, p=[0.4,0.3,0.3]), index=dates)
    equity     = pd.Series(10000 * np.exp(np.cumsum(np.random.randn(n)*0.005)), index=dates)

    risk = RiskModelV2(target_vol=0.25)
    sizing = risk.compute(close, signals, conviction, regime, equity)

    print("\n[OK] RiskModelV2 Output:")
    print(sizing[["vol_scale","conv_scale","regime_cap","dd_factor","position_size_mult"]].describe())
    print("\nSizing report:")
    for k, v in risk.get_sizing_report(sizing).items():
        print(f"  {k:<25}: {v}")
