"""GEPA engine for optimize_anything().

Wraps GEPA's Pareto-efficient reflective text evolution behind the
optimize_anything() API, returning an OptimizeResult.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

from RoboPhD.engines import build_val_split

if TYPE_CHECKING:
    from RoboPhD.api import GEPAConfig, OptimizeResult

logger = logging.getLogger(__name__)


def run_gepa(
    evaluator: Callable,
    dataset: List[Dict],
    seed_candidate: Optional[Dict[str, str]],
    objective: str,
    background: str,
    cfg: GEPAConfig,
    task_name: str,
) -> OptimizeResult:
    """Run GEPA optimization and return an OptimizeResult."""
    from RoboPhD.api import OptimizeResult
    from RoboPhD.config import API_KEY_ENV_VAR
    from RoboPhD.runner_utils import to_litellm_model, CostTrackingLM
    from RoboPhD.candidate_utils import materialize_candidate

    try:
        from gepa.optimize_anything import (
            optimize_anything as gepa_optimize,
            GEPAConfig as _GEPAConfig,
            EngineConfig,
            ReflectionConfig,
        )
    except ImportError:
        raise ImportError(
            "GEPA not installed. Install with: pip install gepa\n"
            "See: https://github.com/gepa-ai/gepa"
        )

    if not seed_candidate:
        raise ValueError("seed_candidate is required for GEPA")

    # API key for reflection model
    api_key = os.environ.get(API_KEY_ENV_VAR) or os.environ.get("ANTHROPIC_API_KEY")

    # Output directory
    run_dir = Path(cfg.parent_experiments_dir) if cfg.parent_experiments_dir else Path("../robophd_runs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = run_dir / "gepa" / f"{task_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build train/val split
    trainset, valset = build_val_split(
        dataset, cfg.val_dataset, cfg.val_size, cfg.seed,
    )

    # Save seed candidate
    with open(output_dir / "seed_candidate.json", "w") as f:
        json.dump(seed_candidate, f, indent=2)

    # Build GEPA config
    engine_cfg = EngineConfig(
        max_metric_calls=cfg.evaluation_budget,
        seed=cfg.seed,
        run_dir=str(output_dir / "gepa_run"),
        parallel=cfg.max_workers is None or cfg.max_workers > 1,
        max_workers=cfg.max_workers,
        track_best_outputs=True,
    )
    gepa_config = _GEPAConfig(engine=engine_cfg)

    # Reflection model
    litellm_model = to_litellm_model(cfg.reflection_model)
    reflection_lm = CostTrackingLM(
        litellm_model,
        api_key=api_key,
        debug_log_probability=cfg.debug_log_probability,
        debug_log_dir=output_dir / "debug_logs" / "reflection",
    )
    gepa_config.reflection = ReflectionConfig(reflection_lm=reflection_lm)

    logger.info(
        f"Starting GEPA optimization with evaluation_budget={cfg.evaluation_budget}, "
        f"reflection_model={cfg.reflection_model}, "
        f"train={len(trainset)}, val={len(valset)}"
    )

    # Run GEPA
    result = gepa_optimize(
        seed_candidate=seed_candidate,
        evaluator=evaluator,
        dataset=trainset,
        valset=valset,
        objective=objective,
        background=background,
        config=gepa_config,
    )

    # Extract results
    best_candidate = result.best_candidate
    val_scores = getattr(result, "val_aggregate_scores", None) or []
    candidates = getattr(result, "candidates", None) or []

    best_val = max(val_scores) if val_scores else 0.0
    reflection_cost = reflection_lm.total_cost if reflection_lm else 0.0
    eval_cost = getattr(evaluator, "total_eval_cost", 0.0)
    total_cost = eval_cost + reflection_cost
    total_evals = getattr(evaluator, "total_evaluations", 0)

    logger.info(f"Optimization complete. Best validation score: {best_val:.3f}")
    logger.info(
        f"Cost: ${total_cost:.2f} total "
        f"(eval: ${eval_cost:.2f}, reflection: ${reflection_cost:.2f})"
    )

    # Save best candidate
    with open(output_dir / "best_candidate.json", "w") as f:
        json.dump(best_candidate, f, indent=2)

    file_mapping = {key: key for key in seed_candidate}
    best_agent_dir = output_dir / "best_agent"
    materialize_candidate(best_candidate, best_agent_dir, file_mapping, name="gepa_best")

    # Save all candidates
    all_candidates_dir = output_dir / "all_candidates"
    for idx, cand in enumerate(candidates):
        cand_dir = all_candidates_dir / f"candidate_{idx}"
        score = val_scores[idx] if idx < len(val_scores) else None
        name = f"candidate_{idx}"
        if score is not None:
            name += f"_val{score:.3f}"
        materialize_candidate(cand, cand_dir, file_mapping, name=name)

    # Save summary
    summary = {
        "task": task_name,
        "engine": "gepa",
        "objective": objective,
        "evaluation_budget": cfg.evaluation_budget,
        "val_size": len(valset),
        "seed": cfg.seed,
        "reflection_model": cfg.reflection_model,
        "total_evaluations": total_evals,
        "train_size": len(trainset),
        "num_candidates_explored": len(candidates),
        "best_val_score": best_val,
        "val_scores": val_scores,
        "cost": {
            "eval_cost_usd": eval_cost,
            "reflection_cost_usd": reflection_cost,
            "total_cost_usd": total_cost,
        },
    }
    with open(output_dir / "optimization_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Build OptimizeResult
    all_candidates_list = []
    for idx, cand in enumerate(candidates):
        all_candidates_list.append({
            "name": f"candidate_{idx}",
            "candidate": cand,
            "elo": 0,
            "mean_score": val_scores[idx] if idx < len(val_scores) else 0.0,
            "test_count": 0,
        })
    all_candidates_list.sort(key=lambda x: x["mean_score"], reverse=True)

    return OptimizeResult(
        best_candidate=best_candidate,
        best_score=best_val,
        experiment_dir=output_dir,
        all_candidates=all_candidates_list,
        num_iterations_completed=len(candidates),
        total_evaluations=total_evals,
        completed_normally=True,
    )
