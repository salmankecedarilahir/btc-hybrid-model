"""
╔══════════════════════════════════════════════════════════════════════════╗
║  ai_training_dataset_builder.py  —  BTC Autonomous AI Quant System    ║
║  LAYER 5 : AI Training Dataset Builder                                  ║
╠══════════════════════════════════════════════════════════════════════════╣
║  TUJUAN  : Bangun dataset ML dari hasil backtest/live trading           ║
║  INPUT   : backtest_df + risk_managed_df + feat_df + regime_labels     ║
║  OUTPUT  : ai_training_dataset.csv (features + targets)                ║
║                                                                         ║
║  Targets yang dibuat:                                                   ║
║    target_profitable_trade  : binary 0/1 (profit or not)               ║
║    target_ret_1bar          : return 1 bar ahead                        ║
║    target_ret_4bar          : return 4 bars ahead                       ║
║    target_ret_12bar         : return 12 bars ahead                      ║
║    target_direction_1bar    : sign of next bar return                   ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


class AITrainingDatasetBuilder:
    """
    Builds ML-ready training dataset from quant system outputs.

    Flow
    ----
    1. Merge feat_df + regime_labels + backtest results
    2. Generate forward-looking targets (properly shifted, no leakage)
    3. Add signal metadata columns
    4. Clean & validate dataset
    5. Save to CSV

    Example
    -------
    builder = AITrainingDatasetBuilder()
    dataset = builder.build(
        feat_df      = feature_df,
        backtest_df  = bt_df,
        regime_labels= regime_series,
        output_path  = "data/ai_training_dataset.csv"
    )
    """

    # Columns to exclude (leakage risk)
    LEAKAGE_COLS = [
        "equity","shadow_equity","running_max_equity",
        "pnl","cumulative_pnl","trade_pnl",
    ]

    def __init__(self, lookahead_bars: list = [1, 4, 12]):
        self.lookahead_bars = lookahead_bars

    # ── MAIN BUILD ────────────────────────────────────────────────────────────
    def build(
        self,
        feat_df:       pd.DataFrame,
        backtest_df:   Optional[pd.DataFrame] = None,
        regime_labels: Optional[pd.Series]    = None,
        signal_df:     Optional[pd.DataFrame] = None,
        output_path:   str = "data/ai_training_dataset.csv",
    ) -> pd.DataFrame:
        """
        Merge all sources and produce clean ML dataset.

        Parameters
        ----------
        feat_df       : Features from FeatureEngine
        backtest_df   : Full backtest dataframe (position, returns, etc.)
        regime_labels : Regime series from RegimeDetectionModel
        signal_df     : Optional signal metadata (signal_strength, direction)
        output_path   : Where to save the final CSV
        """
        log.info("Building AI training dataset...")
        ds = feat_df.copy()

        # ── 1. Attach regime ──────────────────────────────────────────────────
        if regime_labels is not None:
            ds["regime"] = regime_labels.reindex(ds.index).ffill()
        else:
            ds["regime"] = 2   # default RANGING

        # ── 2. Attach backtest columns ────────────────────────────────────────
        if backtest_df is not None:
            bt_cols = [c for c in backtest_df.columns
                       if c not in ds.columns and c not in self.LEAKAGE_COLS]
            for col in bt_cols:
                ds[col] = backtest_df[col].reindex(ds.index)

        # ── 3. Attach signal metadata ─────────────────────────────────────────
        if signal_df is not None:
            for col in ["signal_strength","signal_direction","signal_encoded"]:
                if col in signal_df.columns:
                    ds[col] = signal_df[col].reindex(ds.index)

        # ── 4. Add performance state features ─────────────────────────────────
        ds = self._add_performance_state(ds)

        # ── 5. Build forward-looking targets (NO LEAKAGE) ────────────────────
        ds = self._build_targets(ds)

        # ── 6. Remove leakage columns ─────────────────────────────────────────
        ds = self._remove_leakage(ds)

        # ── 7. Clean & validate ───────────────────────────────────────────────
        ds = self._clean(ds)

        # ── 8. Save ───────────────────────────────────────────────────────────
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        ds.to_csv(output_path)
        log.info("Dataset saved → %s  |  shape=%s", output_path, ds.shape)
        self._print_summary(ds)

        return ds

    # ── TARGET ENGINEERING ────────────────────────────────────────────────────
    def _build_targets(self, ds: pd.DataFrame) -> pd.DataFrame:
        """Build all forward-looking targets with proper shift."""
        # Close price for return calculation
        if "close" in ds.columns:
            c = ds["close"]
        else:
            # Try to reconstruct from strategy return
            c = None

        if c is not None:
            for n in self.lookahead_bars:
                fwd_ret = c.shift(-n) / c - 1
                ds[f"target_ret_{n}bar"] = fwd_ret

            ds["target_direction_1bar"] = np.sign(ds.get(f"target_ret_1bar", pd.Series(0, index=ds.index)))

        # Profitable trade target (based on position direction + forward return)
        if "position" in ds.columns and "target_ret_4bar" in ds.columns:
            pos = ds["position"]
            fwd = ds["target_ret_4bar"]
            ds["target_profitable_trade"] = ((pos * fwd) > 0).astype(int)
        elif "target_ret_4bar" in ds.columns:
            ds["target_profitable_trade"] = (ds["target_ret_4bar"] > 0).astype(int)
        else:
            ds["target_profitable_trade"] = np.nan

        # Signal-level target: was the signal profitable?
        if "strategy_return" in ds.columns:
            sr = ds["strategy_return"]
            ds["target_signal_profit"] = (sr.shift(-1) > 0).astype(float)

        log.info("Targets built: %s", [c for c in ds.columns if c.startswith("target_")])
        return ds

    def _add_performance_state(self, ds: pd.DataFrame) -> pd.DataFrame:
        """Add rolling performance metrics as features."""
        # Rolling win rate (20 bars)
        if "strategy_return" in ds.columns:
            sr = ds["strategy_return"]
            ds["recent_win_rate"] = (sr > 0).rolling(20).mean()
            # Rolling profit factor
            wins  = sr.clip(lower=0).rolling(20).sum()
            losses= (-sr.clip(upper=0)).rolling(20).sum()
            ds["recent_pf"] = wins / losses.replace(0, np.nan)

        # Current drawdown state
        if "equity" in ds.columns:
            eq = ds["equity"]
            peak = eq.expanding().max()
            ds["current_drawdown"] = (eq / peak - 1).clip(-1, 0)
        elif "equity_return" in ds.columns:
            cumret = (1 + ds["equity_return"]).cumprod()
            peak   = cumret.expanding().max()
            ds["current_drawdown"] = (cumret / peak - 1).clip(-1, 0)
        else:
            ds["current_drawdown"] = 0.0

        # Drawdown state categories — encode ke int agar tidak crash di scaler
        dd_series = ds.get("current_drawdown", pd.Series(0.0, index=ds.index))
        if isinstance(dd_series, pd.Series) and len(dd_series) > 1:
            dd_cut = pd.cut(
                dd_series,
                bins=[-np.inf, -0.20, -0.10, -0.05, 0],
                labels=[0, 1, 2, 3]
            )
            ds["drawdown_state"] = pd.to_numeric(dd_cut, errors="coerce").fillna(3).astype(int)
        else:
            ds["drawdown_state"] = 3

        return ds

    # ── UTILS ─────────────────────────────────────────────────────────────────
    def _remove_leakage(self, ds: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [c for c in self.LEAKAGE_COLS if c in ds.columns]
        if cols_to_drop:
            log.info("Dropping leakage columns: %s", cols_to_drop)
            ds = ds.drop(columns=cols_to_drop)
        return ds

    def _clean(self, ds: pd.DataFrame) -> pd.DataFrame:
        # Drop rows where ALL targets are NaN
        target_cols = [c for c in ds.columns if c.startswith("target_")]
        if target_cols:
            ds = ds.dropna(subset=target_cols, how="all")

        # Paksa semua non-target kolom ke numeric (buang string residual)
        non_target_cols = [c for c in ds.columns if not c.startswith("target_") and c != "timestamp"]
        for col in non_target_cols:
            ds[col] = pd.to_numeric(ds[col], errors="coerce")

        # Cap extreme values (winsorize at 1%/99%)
        num_cols = ds.select_dtypes(include=[np.number]).columns
        feat_cols= [c for c in num_cols if not c.startswith("target_")]
        for col in feat_cols:
            lo, hi = ds[col].quantile(0.01), ds[col].quantile(0.99)
            ds[col] = ds[col].clip(lo, hi)

        # Replace inf
        ds = ds.replace([np.inf,-np.inf], np.nan)

        # Fill remaining NaN dengan 0 di feature cols
        ds[feat_cols] = ds[feat_cols].fillna(0)

        log.info("Dataset cleaned: %d rows × %d cols", *ds.shape)
        return ds

    def _print_summary(self, ds: pd.DataFrame):
        target_cols = [c for c in ds.columns if c.startswith("target_")]
        print("\n" + "═"*60)
        print("  AI TRAINING DATASET SUMMARY")
        print("═"*60)
        print(f"  Rows         : {len(ds):,}")
        print(f"  Features     : {len([c for c in ds.columns if not c.startswith('target_')])}")
        print(f"  Targets      : {target_cols}")
        if "target_profitable_trade" in ds.columns:
            pt = ds["target_profitable_trade"].dropna()
            print(f"  Positive rate: {pt.mean()*100:.1f}% ({pt.sum():.0f}/{len(pt)})")
        print("═"*60)

    # ── SPLIT UTILITY ─────────────────────────────────────────────────────────
    def train_val_oos_split(self, ds: pd.DataFrame,
                             train_pct=0.60, val_pct=0.20):
        """
        Chronological split: train / validation / OOS.
        """
        n   = len(ds)
        t1  = int(n * train_pct)
        t2  = int(n * (train_pct + val_pct))
        return ds.iloc[:t1], ds.iloc[t1:t2], ds.iloc[t2:]


def run(feat_df, backtest_df=None, regime_labels=None,
        output_path="data/ai_training_dataset.csv"):
    builder = AITrainingDatasetBuilder()
    ds = builder.build(feat_df, backtest_df, regime_labels,
                       output_path=output_path)
    return {"dataset": ds, "shape": ds.shape, "passed": True}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    np.random.seed(42); n=3000
    dates=pd.date_range("2019-01-01",periods=n,freq="4h")
    price=10000*np.exp(np.cumsum(np.random.randn(n)*0.01))
    feat=pd.DataFrame({"close":price,"open":price,"high":price*1.01,"low":price*0.99,
                        "rsi_14":np.random.uniform(20,80,n),
                        "macd_hist":np.random.randn(n)*100,
                        "vol_real_20":np.abs(np.random.randn(n)*0.3+0.5),
                        "bb_width":np.abs(np.random.randn(n)+3)},index=dates)
    bt=pd.DataFrame({"position":np.random.choice([-1,0,1],n),
                      "strategy_return":np.random.randn(n)*0.005,
                      "equity":10000*np.exp(np.cumsum(np.random.randn(n)*0.005))},index=dates)
    regime=pd.Series(np.random.randint(0,5,n),index=dates,name="regime")
    builder=AITrainingDatasetBuilder()
    ds=builder.build(feat,bt,regime,"data/ai_training_dataset.csv")
    print(ds.tail())
