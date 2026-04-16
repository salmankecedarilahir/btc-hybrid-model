"""
╔══════════════════════════════════════════════════════════════════════════╗
║  model_retraining_engine.py  —  BTC Autonomous AI Quant System         ║
║  LAYER 8 : Model Retraining + Selection + Registry                      ║
╠══════════════════════════════════════════════════════════════════════════╣
║  MODULE A: ModelRetrainingEngine  — triggers + retraining logic         ║
║  MODULE B: ModelSelectionEngine   — pick best model from candidates     ║
║  MODULE C: ModelRegistry          — persistent model store              ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import json, pickle, logging, uuid, shutil
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

log = logging.getLogger(__name__)

REGISTRY_DIR  = Path("models/registry")
REGISTRY_JSON = REGISTRY_DIR / "registry.json"


# ═════════════════════════════════════════════════════════════════════════════
#  C. MODEL REGISTRY
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelEntry:
    model_id        : str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    model_version   : str = "v1.0"
    model_type      : str = "signal_quality"
    training_date   : str = field(default_factory=lambda: datetime.now().isoformat())
    dataset_version : str = "latest"
    model_path      : str = ""
    is_active       : bool = False
    is_champion     : bool = False
    performance     : Dict[str, float] = field(default_factory=dict)
    metadata        : Dict[str, Any]   = field(default_factory=dict)

    def score(self) -> float:
        """Composite score for ranking."""
        p = self.performance
        return (p.get("bt_pf", 0) * 20 +
                p.get("bt_sharpe", 0) * 10 +
                p.get("oos_pf", 0) * 30 -
                abs(p.get("mc_ruin_rate", 100)) * 2 +
                p.get("wf_avg_oos_pf", 0) * 20)


class ModelRegistry:
    """
    Persistent model store for all trained models.

    Example
    -------
    registry = ModelRegistry()
    model_id = registry.register(model_obj, version="v2.0",
                                  model_type="signal_quality",
                                  performance={"bt_pf":1.6})
    model = registry.load(model_id)
    champion = registry.get_champion("signal_quality")
    """

    def __init__(self, registry_dir: str = str(REGISTRY_DIR)):
        self.dir   = Path(registry_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._json = self.dir / "registry.json"
        self._load_registry()

    def _load_registry(self):
        if self._json.exists():
            with open(self._json) as f:
                raw = json.load(f)
            self._entries: Dict[str, ModelEntry] = {
                k: ModelEntry(**v) for k, v in raw.items()
            }
        else:
            self._entries = {}

    def _save_registry(self):
        with open(self._json, "w") as f:
            json.dump({k: asdict(v) for k, v in self._entries.items()}, f, indent=2)

    # ── REGISTER ──────────────────────────────────────────────────────────────
    def register(
        self,
        model_obj: Any,
        version: str,
        model_type: str = "signal_quality",
        performance: Dict[str, float] = None,
        dataset_version: str = "latest",
        metadata: dict = None,
    ) -> str:
        entry = ModelEntry(
            model_version   = version,
            model_type      = model_type,
            dataset_version = dataset_version,
            performance     = performance or {},
            metadata        = metadata or {},
        )
        # Save model pickle
        model_path = self.dir / f"{entry.model_id}_{version}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model_obj, f)
        entry.model_path = str(model_path)

        self._entries[entry.model_id] = entry
        self._save_registry()
        log.info("Model registered: id=%s version=%s type=%s",
                 entry.model_id, version, model_type)
        return entry.model_id

    # ── LOAD ──────────────────────────────────────────────────────────────────
    def load(self, model_id: str) -> Any:
        if model_id not in self._entries:
            raise KeyError(f"Model {model_id} not found in registry")
        path = self._entries[model_id].model_path
        with open(path, "rb") as f:
            return pickle.load(f)

    # ── CHAMPION ──────────────────────────────────────────────────────────────
    def promote_champion(self, model_id: str, model_type: str):
        """Set a model as champion (active production model)."""
        # Demote current champion
        for mid, entry in self._entries.items():
            if entry.model_type == model_type and entry.is_champion:
                entry.is_champion = False
                entry.is_active   = False
        # Promote new champion
        self._entries[model_id].is_champion = True
        self._entries[model_id].is_active   = True
        self._save_registry()
        log.info("Champion promoted: %s → %s", model_id, model_type)

    def get_champion(self, model_type: str) -> Optional[Tuple[str, Any]]:
        """Returns (model_id, model_obj) for current champion."""
        for mid, entry in self._entries.items():
            if entry.model_type == model_type and entry.is_champion:
                return mid, self.load(mid)
        return None, None

    # ── LIST ──────────────────────────────────────────────────────────────────
    def list_models(self, model_type: str = None) -> pd.DataFrame:
        rows = []
        for mid, e in self._entries.items():
            if model_type and e.model_type != model_type: continue
            row = asdict(e); row["score"] = e.score()
            rows.append(row)
        if not rows: return pd.DataFrame()
        df = pd.DataFrame(rows).sort_values("score", ascending=False)
        return df[["model_id","model_version","model_type","training_date",
                   "is_champion","is_active","score"]].reset_index(drop=True)

    def delete(self, model_id: str):
        if model_id not in self._entries: return
        path = Path(self._entries[model_id].model_path)
        if path.exists(): path.unlink()
        del self._entries[model_id]
        self._save_registry()
        log.info("Model deleted: %s", model_id)


# ═════════════════════════════════════════════════════════════════════════════
#  B. MODEL SELECTION ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class ModelSelectionEngine:
    """
    Select best model from multiple trained candidates.

    Scoring weights:
      PF (OOS) × 30 + Sharpe (BT) × 15 + WF avg OOS PF × 25
      - MC ruin rate × 20 + Val score × 10

    Example
    -------
    selector = ModelSelectionEngine()
    winner = selector.select([result1, result2, result3])
    """

    WEIGHTS = {
        "bt_pf"         : 15,
        "bt_sharpe"     : 10,
        "oos_pf"        : 30,
        "wf_avg_oos_pf" : 25,
        "mc_ruin_inv"   : 20,   # 1 - ruin_rate/100
    }

    MIN_REQUIREMENTS = {
        "bt_pf"      : 1.30,
        "bt_sharpe"  : 0.80,
        "oos_pf"     : 1.00,
        "mc_ruin_rate": 30.0,   # max ruin rate %
        "wf_avg_oos_pf": 0.80,
    }

    def select(self, candidates: List[dict], registry: Optional[ModelRegistry] = None) -> Optional[dict]:
        """
        Parameters
        ----------
        candidates : list of dicts, each having:
            model_id, model_obj, bt_metrics, mc_result, wfv_result, oos_result

        Returns
        -------
        Best candidate dict or None if none pass minimum requirements
        """
        scores = []
        for c in candidates:
            s = self._score(c)
            passes = self._check_requirements(c)
            scores.append({"candidate": c, "score": s, "passes": passes})
            log.info("Candidate %s: score=%.1f passes=%s",
                     c.get("model_id","?"), s, passes)

        valid = [s for s in scores if s["passes"]]
        if not valid:
            log.warning("No candidates pass minimum requirements!")
            # Fall back to least-bad
            valid = scores

        best = max(valid, key=lambda x: x["score"])
        winner = best["candidate"]
        log.info("🏆 Winner: %s (score=%.1f)", winner.get("model_id","?"), best["score"])

        # Auto-promote in registry
        if registry and "model_id" in winner:
            registry.promote_champion(winner["model_id"], winner.get("model_type","signal_quality"))

        return winner

    def _score(self, c: dict) -> float:
        bt  = c.get("bt_metrics",  {})
        mc  = c.get("mc_result",   {})
        wf  = c.get("wfv_result",  {})
        oo  = c.get("oos_result",  {})
        score = (
            bt.get("pf",      0)  * self.WEIGHTS["bt_pf"] +
            bt.get("sharpe",  0)  * self.WEIGHTS["bt_sharpe"] +
            oo.get("oos_pf",  0)  * self.WEIGHTS["oos_pf"] +
            wf.get("avg_oos_pf",0)* self.WEIGHTS["wf_avg_oos_pf"] +
            (1 - mc.get("ruin_rate",100)/100) * self.WEIGHTS["mc_ruin_inv"]
        )
        return score

    def _check_requirements(self, c: dict) -> bool:
        bt = c.get("bt_metrics",{}); mc = c.get("mc_result",{}); wf = c.get("wfv_result",{})
        oo = c.get("oos_result",{})
        return (
            bt.get("pf",       0) >= self.MIN_REQUIREMENTS["bt_pf"] and
            bt.get("sharpe",   0) >= self.MIN_REQUIREMENTS["bt_sharpe"] and
            oo.get("oos_pf",   0) >= self.MIN_REQUIREMENTS["oos_pf"] and
            mc.get("ruin_rate",100) <= self.MIN_REQUIREMENTS["mc_ruin_rate"] and
            wf.get("avg_oos_pf",0)  >= self.MIN_REQUIREMENTS["wf_avg_oos_pf"]
        )

    def comparison_table(self, candidates: List[dict]) -> pd.DataFrame:
        rows = []
        for c in candidates:
            bt = c.get("bt_metrics",{}); mc = c.get("mc_result",{})
            wf = c.get("wfv_result",{}); oo = c.get("oos_result",{})
            rows.append({
                "model_id"    : c.get("model_id","?"),
                "bt_pf"       : bt.get("pf",0),
                "bt_sharpe"   : bt.get("sharpe",0),
                "oos_pf"      : oo.get("oos_pf",0),
                "wf_oos_pf"   : wf.get("avg_oos_pf",0),
                "mc_ruin"     : mc.get("ruin_rate",0),
                "total_score" : self._score(c),
                "passes"      : self._check_requirements(c),
            })
        return pd.DataFrame(rows).sort_values("total_score",ascending=False)


# ═════════════════════════════════════════════════════════════════════════════
#  A. MODEL RETRAINING ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class ModelRetrainingEngine:
    """
    Automatic retraining trigger and orchestration.

    Triggers:
      - n_new_trades  : retrain after N new trades since last training
      - n_days        : retrain after N days
      - n_new_samples : retrain after N new data rows
      - performance   : retrain if live performance degrades below threshold

    Example
    -------
    engine = ModelRetrainingEngine(
        n_new_trades=100,
        n_days=180,
    )
    should_retrain, reason = engine.check_trigger(state)
    if should_retrain:
        new_model = engine.retrain(trainer_fn, dataset, ...)
    """

    def __init__(
        self,
        n_new_trades   : int   = 100,
        n_days         : int   = 180,
        n_new_samples  : int   = 5000,
        min_oos_pf     : float = 1.0,
        min_live_sharpe: float = 0.3,
    ):
        self.n_new_trades    = n_new_trades
        self.n_days          = n_days
        self.n_new_samples   = n_new_samples
        self.min_oos_pf      = min_oos_pf
        self.min_live_sharpe = min_live_sharpe
        self._last_train_state: dict = {}

    def check_trigger(self, state: dict) -> Tuple[bool, str]:
        """
        state = {
            "trades_since_train": int,
            "days_since_train":   int,
            "new_samples":        int,
            "live_sharpe_30d":    float,
            "live_pf_30d":        float,
        }
        """
        if state.get("trades_since_train", 0) >= self.n_new_trades:
            return True, f"Trade trigger: {state['trades_since_train']} new trades"

        if state.get("days_since_train", 0) >= self.n_days:
            return True, f"Time trigger: {state['days_since_train']} days elapsed"

        if state.get("new_samples", 0) >= self.n_new_samples:
            return True, f"Data trigger: {state['new_samples']} new samples"

        if state.get("live_pf_30d", 99) < self.min_oos_pf:
            return True, f"Performance trigger: live PF={state['live_pf_30d']:.3f} < {self.min_oos_pf}"

        if state.get("live_sharpe_30d", 99) < self.min_live_sharpe:
            return True, f"Sharpe trigger: live Sharpe={state['live_sharpe_30d']:.3f}"

        return False, "No trigger conditions met"

    def retrain(
        self,
        trainer_fn: callable,
        dataset: pd.DataFrame,
        registry: Optional[ModelRegistry] = None,
        model_type: str = "signal_quality",
        version_prefix: str = "v",
        **trainer_kwargs,
    ) -> Tuple[Any, str]:
        """
        Execute retraining.

        Parameters
        ----------
        trainer_fn   : callable(dataset, **kwargs) → (model, metrics)
        dataset      : latest full training dataset
        registry     : ModelRegistry to auto-register result
        version_prefix : e.g. "v" → "v20260316_0823"

        Returns
        -------
        (model_obj, model_id)
        """
        ts      = datetime.now().strftime("%Y%m%d_%H%M")
        version = f"{version_prefix}{ts}"
        log.info("Starting retraining → %s", version)

        model, metrics = trainer_fn(dataset, **trainer_kwargs)
        log.info("Retraining complete | metrics=%s", metrics)

        model_id = "local"
        if registry:
            model_id = registry.register(
                model, version=version, model_type=model_type,
                performance=metrics, dataset_version=ts,
            )
            log.info("Model registered: %s", model_id)

        self._last_train_state = {
            "version"  : version,
            "model_id" : model_id,
            "timestamp": ts,
            "metrics"  : metrics,
        }
        return model, model_id

    def get_last_train_state(self) -> dict:
        return self._last_train_state


# ─────────────────────────────────────────────────────────────────────────────
#  QUICK DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # ── Registry demo ──────────────────────────────────────────────────────
    registry = ModelRegistry("models/registry_test")
    dummy_model = {"type": "dummy", "weights": np.random.randn(10)}
    mid = registry.register(dummy_model, version="v_test",
                             model_type="signal_quality",
                             performance={"bt_pf":1.6,"bt_sharpe":6.5,
                                          "oos_pf":1.19,"wf_avg_oos_pf":0.97,
                                          "mc_ruin_rate":0.13})
    registry.promote_champion(mid, "signal_quality")
    print("\n── Model Registry ──────────────────────────────────────")
    print(registry.list_models())

    # ── Selection demo ──────────────────────────────────────────────────────
    selector = ModelSelectionEngine()
    candidates = [
        {"model_id":"A","bt_metrics":{"pf":1.6,"sharpe":6.5},
         "oos_result":{"oos_pf":1.19},"mc_result":{"ruin_rate":0.13},
         "wfv_result":{"avg_oos_pf":0.97}},
        {"model_id":"B","bt_metrics":{"pf":1.4,"sharpe":5.0},
         "oos_result":{"oos_pf":1.05},"mc_result":{"ruin_rate":5.0},
         "wfv_result":{"avg_oos_pf":0.85}},
    ]
    print("\n── Model Comparison ────────────────────────────────────")
    print(selector.comparison_table(candidates))
    winner = selector.select(candidates)
    print(f"\n🏆 Winner: {winner['model_id']}")

    # ── Retrain trigger demo ────────────────────────────────────────────────
    engine = ModelRetrainingEngine(n_new_trades=50, n_days=30)
    state  = {"trades_since_train":120, "days_since_train":10, "new_samples":1000}
    should, reason = engine.check_trigger(state)
    print(f"\nRetrain trigger: {should} — {reason}")
