"""
╔══════════════════════════════════════════════════════════════════════════╗
║  experiment_logger.py  —  BTC Autonomous AI Quant System               ║
║  LAYER 7 : Experiment Logger                                            ║
╠══════════════════════════════════════════════════════════════════════════╣
║  TUJUAN  : Catat semua eksperimen training/validation ke database       ║
║  STORAGE : SQLite (lokal) + CSV backup                                  ║
║  DATA    : model_version, params, metrics, MC results, WFV results     ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import json, sqlite3, logging, hashlib, uuid
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict

log = logging.getLogger(__name__)

DB_PATH  = "research/experiments/experiments.db"
CSV_PATH = "research/experiments/experiments.csv"


# ─────────────────────────────────────────────────────────────────────────────
#  EXPERIMENT RECORD
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentRecord:
    # Identity
    experiment_id   : str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    model_version   : str = "v1.0"
    model_type      : str = "signal_quality"
    timestamp       : str = field(default_factory=lambda: datetime.now().isoformat())
    run_notes       : str = ""

    # Data info
    dataset_version : str = "latest"
    n_samples       : int = 0
    n_features      : int = 0
    features_used   : List[str] = field(default_factory=list)
    train_start     : str = ""
    train_end       : str = ""

    # Model params
    model_params    : Dict[str, Any] = field(default_factory=dict)
    hyperparams     : Dict[str, Any] = field(default_factory=dict)

    # Backtest metrics
    bt_cagr         : float = 0.0
    bt_sharpe       : float = 0.0
    bt_pf           : float = 0.0
    bt_max_dd       : float = 0.0
    bt_win_rate     : float = 0.0
    bt_sortino      : float = 0.0
    bt_n_trades     : int   = 0

    # CV metrics
    cv_auc_mean     : float = 0.0
    cv_auc_std      : float = 0.0
    cv_scores       : List[float] = field(default_factory=list)

    # Monte Carlo
    mc_ruin_rate    : float = 0.0
    mc_pf_mean      : float = 0.0
    mc_pf_p5        : float = 0.0
    mc_sharpe_mean  : float = 0.0
    mc_dd_worst     : float = 0.0
    mc_n_sims       : int   = 0

    # Walk Forward
    wf_avg_oos_pf   : float = 0.0
    wf_oos_preserved: float = 0.0
    wf_deg_ratio    : float = 0.0
    wf_n_windows    : int   = 0

    # OOS
    oos_sharpe      : float = 0.0
    oos_pf          : float = 0.0
    oos_cagr        : float = 0.0
    oos_max_dd      : float = 0.0

    # Final scoring
    validation_score: float = 0.0
    validation_grade: str   = ""
    is_deployed     : bool  = False
    deployed_at     : str   = ""

    # Status
    status          : str   = "completed"
    error_message   : str   = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        # Serialize lists/dicts to JSON strings for DB storage
        for key in ["features_used","model_params","hyperparams","cv_scores"]:
            d[key] = json.dumps(d[key])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentRecord":
        for key in ["features_used","model_params","hyperparams","cv_scores"]:
            if key in d and isinstance(d[key], str):
                try: d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError): d[key] = []
        return cls(**{k:v for k,v in d.items() if k in cls.__dataclass_fields__})

    def summary(self) -> str:
        return (
            f"[{self.experiment_id}] {self.model_version} | "
            f"PF={self.bt_pf:.3f} Sharpe={self.bt_sharpe:.3f} "
            f"DD={self.bt_max_dd:.1f}% | "
            f"MC_ruin={self.mc_ruin_rate:.1f}% | "
            f"WF_pf={self.wf_avg_oos_pf:.3f} | "
            f"Score={self.validation_score:.0f} [{self.validation_grade}]"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  EXPERIMENT LOGGER
# ─────────────────────────────────────────────────────────────────────────────

class ExperimentLogger:
    """
    Persistent experiment tracker with SQLite backend.

    Example
    -------
    logger = ExperimentLogger()

    # Log a complete experiment
    rec = ExperimentRecord(
        model_version = "v2.0",
        model_type    = "signal_quality",
        bt_sharpe     = 6.6,
        bt_pf         = 1.60,
        bt_max_dd     = -28.1,
        mc_ruin_rate  = 0.13,
        wf_avg_oos_pf = 0.97,
        validation_score = 83.0,
        validation_grade = "GOOD",
    )
    logger.log(rec)

    # Load history
    df = logger.load_all()
    best = logger.get_best(metric="validation_score")
    """

    def __init__(self, db_path: str = DB_PATH, csv_path: str = CSV_PATH):
        self.db_path  = db_path
        self.csv_path = csv_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create experiments table if not exists."""
        rec   = ExperimentRecord()
        d     = rec.to_dict()
        cols  = ", ".join(f'"{k}" TEXT' for k in d.keys())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"CREATE TABLE IF NOT EXISTS experiments ({cols})")
        log.info("ExperimentLogger initialized → %s", self.db_path)

    # ── WRITE ──────────────────────────────────────────────────────────────────
    def log(self, record: ExperimentRecord) -> str:
        d   = record.to_dict()
        cols= ", ".join(f'"{k}"' for k in d.keys())
        vals= ", ".join("?" for _ in d)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"INSERT INTO experiments ({cols}) VALUES ({vals})",
                         list(d.values()))
        self._append_csv(record)
        log.info("Experiment logged: %s", record.summary())
        return record.experiment_id

    def log_from_results(
        self,
        model_version: str,
        model_type: str,
        bt_metrics: dict    = None,
        mc_result: dict     = None,
        wfv_result: dict    = None,
        oos_result: dict    = None,
        cv_scores: list     = None,
        features: list      = None,
        model_params: dict  = None,
        notes: str          = "",
        validation_score: float = 0,
        validation_grade: str   = "",
    ) -> str:
        """Convenience: build ExperimentRecord from pipeline results dicts."""
        bt = bt_metrics or {}
        mc = mc_result  or {}
        wf = wfv_result or {}
        oo = oos_result or {}

        rec = ExperimentRecord(
            model_version    = model_version,
            model_type       = model_type,
            run_notes        = notes,
            features_used    = features or [],
            n_features       = len(features) if features else 0,
            model_params     = model_params or {},
            bt_cagr          = bt.get("cagr_pct", 0),
            bt_sharpe        = bt.get("sharpe", 0),
            bt_pf            = bt.get("pf", 0),
            bt_max_dd        = bt.get("max_dd", 0),
            bt_win_rate      = bt.get("win_rate", 0),
            bt_n_trades      = bt.get("n_trades", 0),
            cv_auc_mean      = float(np.mean(cv_scores)) if cv_scores else 0,
            cv_auc_std       = float(np.std(cv_scores))  if cv_scores else 0,
            cv_scores        = cv_scores or [],
            mc_ruin_rate     = mc.get("ruin_rate", 0),
            mc_pf_mean       = mc.get("pf_mean",   0),
            mc_pf_p5         = mc.get("pf_p5",     0),
            mc_sharpe_mean   = mc.get("sharpe_mean",0),
            mc_dd_worst      = mc.get("dd_worst",  0),
            mc_n_sims        = mc.get("n_sims",    0),
            wf_avg_oos_pf    = wf.get("avg_oos_pf",     0),
            wf_oos_preserved = wf.get("oos_preserved",  0),
            wf_deg_ratio     = wf.get("deg_ratio",      0),
            wf_n_windows     = wf.get("n_windows",      0),
            oos_sharpe       = oo.get("oos_sharpe",     0),
            oos_pf           = oo.get("oos_pf",         0),
            validation_score = validation_score,
            validation_grade = validation_grade,
        )
        return self.log(rec)

    def update_deployment(self, experiment_id: str, deployed: bool = True):
        ts = datetime.now().isoformat() if deployed else ""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                'UPDATE experiments SET is_deployed=?, deployed_at=? WHERE experiment_id=?',
                [str(deployed), ts, experiment_id]
            )
        log.info("Deployment status updated: %s → %s", experiment_id, deployed)

    # ── READ ───────────────────────────────────────────────────────────────────
    def load_all(self) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql("SELECT * FROM experiments ORDER BY timestamp DESC", conn)
        return df

    def load_by_type(self, model_type: str) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql(
                f"SELECT * FROM experiments WHERE model_type=? ORDER BY timestamp DESC",
                conn, params=[model_type]
            )
        return df

    def get_best(self, metric="validation_score", model_type=None) -> Optional[ExperimentRecord]:
        df = self.load_all() if model_type is None else self.load_by_type(model_type)
        if df.empty: return None
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        row = df.loc[df[metric].idxmax()]
        return ExperimentRecord.from_dict(row.to_dict())

    def get_deployed(self) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql(
                "SELECT * FROM experiments WHERE is_deployed='True'", conn
            )

    def compare(self, experiment_ids: List[str]) -> pd.DataFrame:
        metrics = ["model_version","bt_sharpe","bt_pf","bt_max_dd",
                   "mc_ruin_rate","wf_avg_oos_pf","oos_pf","validation_score"]
        df = self.load_all()
        sub = df[df["experiment_id"].isin(experiment_ids)][metrics]
        return sub.set_index("model_version")

    # ── CSV BACKUP ─────────────────────────────────────────────────────────────
    def _append_csv(self, record: ExperimentRecord):
        row = {k: v for k, v in asdict(record).items()}
        row["features_used"] = str(row["features_used"])
        row["cv_scores"]     = str(row["cv_scores"])
        row["model_params"]  = str(row["model_params"])
        df  = pd.DataFrame([row])
        hdr = not Path(self.csv_path).exists()
        df.to_csv(self.csv_path, mode="a", header=hdr, index=False)

    def print_leaderboard(self, top_n=10):
        df = self.load_all()
        if df.empty: print("No experiments logged yet."); return
        df["validation_score"] = pd.to_numeric(df["validation_score"], errors="coerce")
        top = df.nlargest(top_n, "validation_score")[[
            "experiment_id","model_version","timestamp",
            "bt_sharpe","bt_pf","mc_ruin_rate","wf_avg_oos_pf","validation_score","validation_grade"
        ]]
        print("\n── Experiment Leaderboard ───────────────────────────────")
        print(top.to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = ExperimentLogger(db_path="research/experiments/test.db",
                              csv_path="research/experiments/test.csv")
    for i in range(3):
        logger.log_from_results(
            model_version=f"v1.{i}",
            model_type="signal_quality",
            bt_metrics={"sharpe":6.5+i*0.3,"pf":1.5+i*0.1,"max_dd":-28+i,"win_rate":55},
            mc_result={"ruin_rate":0.13,"pf_mean":1.49,"pf_p5":0.90,"n_sims":1000},
            wfv_result={"avg_oos_pf":0.97,"oos_preserved":50,"deg_ratio":0.26,"n_windows":8},
            cv_scores=[0.58,0.61,0.59,0.62,0.60],
            validation_score=83.0+i, validation_grade="GOOD",
        )
    logger.print_leaderboard()
    best = logger.get_best()
    print(f"\nBest experiment: {best.summary()}")
