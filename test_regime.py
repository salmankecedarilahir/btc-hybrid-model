"""
test_regime.py — Validasi hasil Market Regime Engine.

Output:
  1. Distribusi jumlah & persentase regime
  2. 10 baris terakhir dengan semua kolom regime
  3. Plot: close price + EMA20 + EMA50, warna background per regime
  4. Plot: ATR percentile time series

Jalankan: python test_regime.py
"""

import sys
import logging
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (aman di semua OS)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).parent
INPUT_PATH  = BASE_DIR / "data" / "btc_4h_with_regime.csv"
CHART_PATH  = BASE_DIR / "data" / "regime_chart.png"

DIVIDER   = "═" * 65
SEPARATOR = "─" * 65

# ── Warna per regime ──────────────────────────────────────────────────────────
REGIME_COLORS = {
    "UP":       "#1a9641",   # hijau
    "DOWN":     "#d7191c",   # merah
    "SIDEWAYS": "#fdae61",   # oranye
    "NEUTRAL":  "#cccccc",   # abu
}


def load_data() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        log.error("File tidak ditemukan: %s", INPUT_PATH)
        log.error("Jalankan regime_engine.py terlebih dahulu.")
        sys.exit(1)

    df = pd.read_csv(INPUT_PATH, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def print_distribution(df: pd.DataFrame) -> None:
    print(f"\n{DIVIDER}")
    print("  Regime Distribution")
    print(DIVIDER)

    total         = len(df)
    valid         = df["regime"].notna().sum()
    warmup        = df["atr_percentile"].isna().sum()

    print(f"  Total candles    : {total:,}")
    print(f"  Warmup (no data) : {warmup:,}")
    print(f"  Valid candles    : {valid - warmup:,}")
    print()

    order = ["UP", "DOWN", "SIDEWAYS", "NEUTRAL"]
    for regime in order:
        count = (df["regime"] == regime).sum()
        pct   = count / total * 100
        bar   = "█" * int(pct / 2)
        print(f"  {regime:<10} {count:>5} candles  ({pct:5.1f}%)  {bar}")

    print()
    # trend_score stats untuk valid rows only
    valid_df = df.dropna(subset=["atr_percentile"])
    ts = valid_df["trend_score"]
    print(f"  Trend Score  mean={ts.mean():.3f}  std={ts.std():.3f}  "
          f"min={ts.min():.0f}  max={ts.max():.0f}")


def print_tail(df: pd.DataFrame, n: int = 10) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  Last {n} Candles")
    print(SEPARATOR)

    cols = ["timestamp", "close", "ema_20", "ema_50", "atr_14",
            "atr_percentile", "regime", "trend_score"]
    cols = [c for c in cols if c in df.columns]

    tail = df[cols].tail(n).copy()
    tail["timestamp"] = tail["timestamp"].dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M") if tail["timestamp"].dt.tz is not None else tail["timestamp"].dt.strftime("%Y-%m-%d %H:%M")

    # Format floats
    for col in ["close", "ema_20", "ema_50", "atr_14"]:
        if col in tail.columns:
            tail[col] = tail[col].map("{:.2f}".format)
    if "atr_percentile" in tail.columns:
        tail["atr_percentile"] = tail["atr_percentile"].map(
            lambda x: f"{x:.1f}" if pd.notna(x) else "NaN"
        )

    print(tail.to_string(index=False))


def plot_regime(df: pd.DataFrame) -> None:
    """
    Figure 1: Close price dengan EMA lines dan background warna per regime.
    Figure 2: ATR percentile dengan threshold line.
    """
    # Hanya plot baris yang punya regime valid
    plot_df = df.dropna(subset=["atr_percentile"]).copy()
    # Strip timezone → naive UTC agar matplotlib tidak error
    plot_df["timestamp_dt"] = plot_df["timestamp"].dt.tz_localize(None)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                    gridspec_kw={"height_ratios": [3, 1]},
                                    sharex=True)
    fig.patch.set_facecolor("#0d1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    # ── Panel 1: Close + EMA + regime background ──────────────────────────────
    ts = plot_df["timestamp_dt"]

    # Background shading per regime
    prev_regime = None
    seg_start   = None
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        regime = row["regime"]
        if regime != prev_regime:
            if prev_regime is not None:
                ax1.axvspan(seg_start, row["timestamp_dt"],
                            alpha=0.12, color=REGIME_COLORS.get(prev_regime, "#cccccc"),
                            linewidth=0)
            seg_start   = row["timestamp_dt"]
            prev_regime = regime
    # last segment
    if seg_start and prev_regime:
        ax1.axvspan(seg_start, ts.iloc[-1],
                    alpha=0.12, color=REGIME_COLORS.get(prev_regime, "#cccccc"),
                    linewidth=0)

    ax1.plot(ts, plot_df["close"],  color="#58a6ff", linewidth=0.9, label="Close", zorder=3)
    ax1.plot(ts, plot_df["ema_20"], color="#ffa657", linewidth=1.2,
             label=f"EMA20", linestyle="--", zorder=4)
    ax1.plot(ts, plot_df["ema_50"], color="#ff7b72", linewidth=1.5,
             label=f"EMA50", zorder=4)

    ax1.set_ylabel("Price (USD)", fontsize=11)
    ax1.set_title("BTC/USD 4H — Market Regime Engine", fontsize=13, pad=10)

    # Legend
    legend_handles = [
        mpatches.Patch(color=REGIME_COLORS["UP"],       alpha=0.6, label="UP"),
        mpatches.Patch(color=REGIME_COLORS["DOWN"],     alpha=0.6, label="DOWN"),
        mpatches.Patch(color=REGIME_COLORS["SIDEWAYS"], alpha=0.6, label="SIDEWAYS"),
        mpatches.Patch(color=REGIME_COLORS["NEUTRAL"],  alpha=0.6, label="NEUTRAL"),
    ]
    line_handles = [
        plt.Line2D([0], [0], color="#58a6ff", linewidth=1.5, label="Close"),
        plt.Line2D([0], [0], color="#ffa657", linewidth=1.5, linestyle="--", label="EMA20"),
        plt.Line2D([0], [0], color="#ff7b72", linewidth=1.5, label="EMA50"),
    ]
    ax1.legend(handles=legend_handles + line_handles,
               loc="upper left", fontsize=8,
               facecolor="#21262d", edgecolor="#30363d", labelcolor="white",
               ncol=4)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # ── Panel 2: ATR Percentile ───────────────────────────────────────────────
    ax2.fill_between(ts, plot_df["atr_percentile"],
                     alpha=0.5, color="#bc8cff", linewidth=0)
    ax2.plot(ts, plot_df["atr_percentile"],
             color="#bc8cff", linewidth=0.8)
    ax2.axhline(30, color="#fdae61", linewidth=1.2, linestyle="--",
                label="Sideways threshold (30)")
    ax2.set_ylabel("ATR Percentile", fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.legend(loc="upper left", fontsize=8,
               facecolor="#21262d", edgecolor="#30363d", labelcolor="white")

    # x-axis formatting
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right", color="white")

    plt.tight_layout(h_pad=0.5)
    plt.savefig(CHART_PATH, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    log.info("Chart tersimpan → %s", CHART_PATH)


def main() -> None:
    print(DIVIDER)
    print("  BTC Hybrid Model — Phase 2: Regime Validation")
    print(DIVIDER)

    df = load_data()

    log.info("Data loaded: %d baris", len(df))

    print_distribution(df)
    print_tail(df)
    plot_regime(df)

    print(f"\n{DIVIDER}")
    print("  Validation complete.")
    print(f"  Chart → {CHART_PATH}")
    print(DIVIDER)


if __name__ == "__main__":
    main()
