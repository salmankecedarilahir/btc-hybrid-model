"""
╔══════════════════════════════════════════════════════════════════════════╗
║  run_ai_system.py  —  BTC Autonomous AI Quant System                   ║
║  MASTER RUNNER                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Usage:                                                                 ║
║    python run_ai_system.py --mode full                                  ║
║    python run_ai_system.py --mode train_only                            ║
║    python run_ai_system.py --mode validate                              ║
║    python run_ai_system.py --mode research --cycles 3                   ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import argparse, logging, sys
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("logs/ai_system.log"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)
DIV = "═" * 68


def run_full_pipeline(data_path, backtest_path, cycles=1, deploy_threshold=75):
    """Run complete autonomous research loop."""
    from autonomous_research_loop import AutonomousResearchLoop
    loop = AutonomousResearchLoop(
        data_path        = data_path,
        backtest_path    = backtest_path,
        deploy_threshold = deploy_threshold,
    )
    return loop.run(n_cycles=cycles)


def run_feature_demo(data_path):
    """Run feature engineering only."""
    import pandas as pd
    from feature_engine import FeatureEngine
    df = pd.read_csv(data_path, index_col=0, parse_dates=True) if data_path else None
    if df is None:
        from autonomous_research_loop import AutonomousResearchLoop
        df = AutonomousResearchLoop._generate_synthetic_ohlcv(3000)
    engine = FeatureEngine()
    feat   = engine.transform(df)
    print(f"\n[OK] Features generated: {feat.shape[1]} features × {feat.shape[0]} rows")
    print("\nFeature list:")
    for i, c in enumerate(feat.columns, 1): print(f"  {i:3d}. {c}")
    return feat


def run_validation_only(returns_path):
    """Run validation framework on existing returns."""
    from validation_framework import ValidationFramework
    returns = pd.read_csv(returns_path, index_col=0, parse_dates=True).squeeze()
    vf = ValidationFramework()
    report = vf.run(returns)
    return report


def show_leaderboard():
    """Show experiment leaderboard."""
    from experiment_logger import ExperimentLogger
    logger = ExperimentLogger()
    logger.print_leaderboard()


def main():
    parser = argparse.ArgumentParser(description="BTC Autonomous AI Quant System")
    parser.add_argument("--mode", choices=["full","features","validate","leaderboard","research"],
                        default="full")
    parser.add_argument("--data",      default="data/raw/btc_ohlcv.csv")
    parser.add_argument("--backtest",  default="data/processed/btc_backtest_results.csv")
    parser.add_argument("--returns",   default="data/processed/returns.csv")
    parser.add_argument("--cycles",    type=int, default=1)
    parser.add_argument("--threshold", type=float, default=75.0)
    args = parser.parse_args()

    print(f"\n{DIV}")
    print("  BTC AUTONOMOUS AI QUANT SYSTEM")
    print(f"  Mode: {args.mode.upper()}")
    print(DIV)

    if args.mode == "full" or args.mode == "research":
        run_full_pipeline(args.data, args.backtest, args.cycles, args.threshold)
    elif args.mode == "features":
        run_feature_demo(args.data)
    elif args.mode == "validate":
        run_validation_only(args.returns)
    elif args.mode == "leaderboard":
        show_leaderboard()


if __name__ == "__main__":
    main()
