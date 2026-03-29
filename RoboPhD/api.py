"""
Simple programmatic API for RoboPhD optimization.

Provides an optimize_anything() interface inspired by GEPA's API, wrapping
RoboPhD's ELO evolution engine (ParallelAgentResearcher) behind a simple
function call.

Usage:
    from RoboPhD import optimize_anything

    def evaluator(candidate, example):
        score = 1.0 if candidate["prompt"] in example["expected"] else 0.0
        return score, {"output": candidate["prompt"]}

    result = optimize_anything(
        evaluator=evaluator,
        dataset=[{"input": "2+2", "expected": "4"}, ...],
        seed_candidate={"prompt": "Solve the math problem:"},
        objective="Maximize accuracy on math problems",
    )
    print(result.best_candidate, result.best_score)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class RoboPhDConfig:
    """Configuration for RoboPhD ELO evolution engine.

    All fields have sensible defaults. For most use cases, only
    ``num_iterations`` and ``evaluation_budget`` need tuning.
    """

    # Budget & scale
    num_iterations: int = 10
    evaluation_budget: Optional[int] = None
    """Max evaluator calls across all iterations. None = unlimited."""
    examples_per_iteration: int = 20

    # Evolution
    evolution_strategy: str = "use_your_judgment"
    evolution_model: str = "opus-4.6"
    evolution_timeout: int = 3600
    """Seconds per evolution session (Claude Code CLI subprocess)."""

    # Parallelism
    max_workers: Optional[int] = None
    """Thread pool size for concurrent evaluation. None = Python default."""

    # Output
    run_dir: Union[str, Path, None] = None
    """Directory for experiment output. None = ../robophd_runs."""

    # Reproducibility
    random_seed: Optional[int] = None

    # Evaluation
    eval_timeout: int = 300
    """Seconds per evaluator call before timeout (scores 0)."""

    # Advanced overrides (passed directly to ConfigManager)
    engine_overrides: Optional[Dict[str, Any]] = None
    """Extra ConfigManager parameters for power users (e.g.
    weighted_random_configs, config_schedule, new_agent_test_rounds)."""


@dataclass
class OptimizeResult:
    """Result of an optimize_anything() or optimize_task() call."""

    best_candidate: Dict[str, str]
    """Best agent's text artifacts (same keys as seed_candidate)."""
    best_score: float
    """Best agent's ELO rating."""
    experiment_dir: Path
    """Path to the full experiment directory (checkpoints, logs, agents)."""
    all_candidates: List[Dict[str, Any]]
    """All agents with their scores: [{"name": ..., "candidate": ..., "elo": ..., "mean_score": ...}, ...]."""
    num_iterations_completed: int
    total_evaluations: int


def optimize_anything(
    evaluator: Callable,
    dataset: List[Dict],
    seed_candidate: Dict[str, str],
    objective: str,
    background: str = "",
    config: Optional[RoboPhDConfig] = None,
) -> OptimizeResult:
    """Optimize text artifacts using RoboPhD's ELO evolution engine.

    This is the simple, programmatic entry point to RoboPhD. Under the hood it
    runs the full multi-agent ELO competition: each iteration, an evolution AI
    (Claude Code) proposes improved agents, which are evaluated head-to-head on
    sampled problems from your dataset.

    Args:
        evaluator: Scoring function with signature
            ``(candidate: dict, example: dict) -> (score: float, diagnostics: dict)``.
            Higher scores are better. Must be thread-safe (called concurrently).
        dataset: List of example dicts for evaluation.
        seed_candidate: Initial text artifact(s) to optimize. Dict mapping
            component names to text content, e.g. ``{"prompt": "Solve carefully"}``.
        objective: Natural-language optimization goal shown to the evolution AI.
        background: Optional domain documentation shown to the evolution AI.
        config: Engine configuration. If None, uses ``RoboPhDConfig()`` defaults.

    Returns:
        OptimizeResult with best_candidate, best_score, and experiment_dir.

    Example::

        from RoboPhD import optimize_anything, RoboPhDConfig

        result = optimize_anything(
            evaluator=my_scorer,
            dataset=my_examples,
            seed_candidate={"prompt": "Solve the problem step by step"},
            objective="Maximize accuracy on math problems",
            config=RoboPhDConfig(num_iterations=5, evaluation_budget=200),
        )
    """
    from RoboPhD.config_manager import ConfigManager, ConfigSource
    from RoboPhD.researcher import ParallelAgentResearcher
    from RoboPhD.adapters.candidate_utils import extract_candidate, materialize_candidate
    from RoboPhD.adapters.runner_utils import find_best_agent

    if not dataset:
        raise ValueError("dataset must be a non-empty list of example dicts")
    if not seed_candidate:
        raise ValueError("seed_candidate must be a non-empty dict")

    cfg = config or RoboPhDConfig()

    # 1. File mapping: each candidate key is its own filename
    file_mapping = {key: key for key in seed_candidate}

    # 2. Materialize seed agent to a temp directory
    run_dir = Path(cfg.run_dir) if cfg.run_dir else Path("../robophd_runs")
    seed_agents_dir = run_dir / "robophd" / "_optimize_anything_seeds"
    seed_agents_dir.mkdir(parents=True, exist_ok=True)

    seed_dir = Path(tempfile.mkdtemp(dir=seed_agents_dir, prefix="seed_"))
    materialize_candidate(seed_candidate, seed_dir, file_mapping, name="seed")
    seed_agent_name = seed_dir.name

    # 3. Build ConfigManager
    config_manager = ConfigManager()
    researcher_config = {
        "domain": "external",
        "evolution_strategy": cfg.evolution_strategy,
        "evolution_model": cfg.evolution_model,
        "evolution_timeout": cfg.evolution_timeout,
        "examples_per_iteration": cfg.examples_per_iteration,
        "agents_directory": str(seed_dir.parent),
        "initial_agents": [seed_agent_name],
    }
    if cfg.max_workers is not None:
        researcher_config["max_workers"] = cfg.max_workers
    if cfg.evaluation_budget is not None:
        researcher_config["evaluation_budget"] = cfg.evaluation_budget
    if cfg.engine_overrides:
        researcher_config.update(cfg.engine_overrides)

    config_manager.set_initial_config(researcher_config, ConfigSource.CLI)

    # 4. Build runtime_config (non-serializable, never checkpointed)
    runtime_config = {
        "evaluator_fn": evaluator,
        "dataset": dataset,
        "file_mapping": file_mapping,
        "task_objective": objective,
        "task_description": objective,
        "task_background": background,
        "task_name": "optimize_anything",
        "diagnostic_files": {},
        "runs_dir": str(run_dir),
        "eval_timeout": cfg.eval_timeout,
    }

    # 5. Create and run researcher
    researcher = ParallelAgentResearcher(
        config_manager=config_manager,
        num_iterations=cfg.num_iterations,
        random_seed=cfg.random_seed,
        runtime_config=runtime_config,
        task_config={},
    )

    completed_normally = researcher.run(initial_agents=[seed_agent_name])

    # 6. Extract results
    try:
        result = _build_result(researcher.experiment_dir, file_mapping)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise RuntimeError(
            f"Optimization failed: could not extract results from {researcher.experiment_dir}. "
            f"Run completed_normally={completed_normally}. Cause: {exc}"
        ) from exc

    if not completed_normally:
        logger.warning(
            "Optimization ended early (evolution failed). "
            "Returning partial results from %d completed iteration(s).",
            result.num_iterations_completed,
        )

    return result


def optimize_task(
    task_name: str,
    *,
    seed_agent: Optional[str] = None,
    task_config: Optional[Dict[str, Any]] = None,
    config: Optional[RoboPhDConfig] = None,
) -> OptimizeResult:
    """Run RoboPhD ELO evolution on a registered task.

    Equivalent to ``run_robophd.py --task <name>`` but callable from Python.

    Args:
        task_name: Registered task name (e.g. "cant_be_late_stdout", "arc_agi_1").
        seed_agent: Path to seed agent directory. Defaults to task's default_seed_agent.
        task_config: Task-specific configuration (e.g. split, model overrides).
        config: Engine configuration. If None, uses ``RoboPhDConfig()`` defaults.

    Returns:
        OptimizeResult with best_candidate, best_score, and experiment_dir.

    Example::

        from RoboPhD import optimize_task, RoboPhDConfig

        result = optimize_task(
            "cant_be_late_stdout",
            config=RoboPhDConfig(num_iterations=5, evaluation_budget=200),
        )
    """
    from RoboPhD.config_manager import ConfigManager, ConfigSource
    from RoboPhD.researcher import ParallelAgentResearcher
    from RoboPhD.adapters.candidate_utils import extract_candidate
    from RoboPhD.adapters.runner_utils import find_best_agent
    from RoboPhD.tasks import get_task

    task = get_task(task_name)
    cfg = config or RoboPhDConfig()
    tc = dict(task_config or {})

    # Build full config for evaluator/dataset factories
    full_config = {**task.config_defaults, **tc}

    # Build evaluator and dataset
    evaluator = task.evaluator_factory(full_config)
    dataset = task.dataset_builder(full_config)
    if not dataset:
        raise ValueError(f"Empty dataset for task {task_name!r}")

    # Resolve seed agent
    seed_agent_path = Path(tc.get("seed_agent", seed_agent or task.default_seed_agent))
    if not seed_agent_path.exists():
        raise FileNotFoundError(f"Seed agent not found: {seed_agent_path}")

    # Build ConfigManager
    config_manager = ConfigManager()
    researcher_config = {
        "domain": "external",
        "meta_evolution_domain": task.name,
        "evolution_strategy": cfg.evolution_strategy,
        "evolution_model": cfg.evolution_model,
        "evolution_timeout": cfg.evolution_timeout,
        "examples_per_iteration": cfg.examples_per_iteration,
        "agents_directory": str(seed_agent_path.parent),
        "initial_agents": [seed_agent_path.name],
    }
    if cfg.max_workers is not None:
        researcher_config["max_workers"] = cfg.max_workers
    if cfg.evaluation_budget is not None:
        researcher_config["evaluation_budget"] = cfg.evaluation_budget
    elif "evaluation_budget" in task.config_defaults:
        researcher_config["evaluation_budget"] = task.config_defaults["evaluation_budget"]
    if cfg.engine_overrides:
        researcher_config.update(cfg.engine_overrides)

    config_manager.set_initial_config(researcher_config, ConfigSource.CLI)

    # Build runtime_config
    run_dir = Path(cfg.run_dir) if cfg.run_dir else Path("../robophd_runs")
    runtime_config = {
        "evaluator_fn": evaluator,
        "dataset": dataset,
        "file_mapping": task.file_mapping,
        "task_objective": task.objective,
        "task_description": task.description,
        "task_background": task.background,
        "task_name": task.name,
        "diagnostic_files": task.diagnostic_files,
        "runs_dir": str(run_dir),
        "eval_timeout": cfg.eval_timeout,
    }

    # Create and run researcher
    researcher = ParallelAgentResearcher(
        config_manager=config_manager,
        num_iterations=cfg.num_iterations,
        random_seed=cfg.random_seed,
        runtime_config=runtime_config,
        task_config=tc,
    )

    completed_normally = researcher.run(initial_agents=[seed_agent_path.name])

    # Extract results
    try:
        result = _build_result(researcher.experiment_dir, task.file_mapping)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise RuntimeError(
            f"Optimization failed: could not extract results from {researcher.experiment_dir}. "
            f"Run completed_normally={completed_normally}. Cause: {exc}"
        ) from exc

    if not completed_normally:
        logger.warning(
            "Optimization ended early (evolution failed). "
            "Returning partial results from %d completed iteration(s).",
            result.num_iterations_completed,
        )

    return result


def _build_result(experiment_dir: Path, file_mapping: Dict[str, str]) -> OptimizeResult:
    """Extract OptimizeResult from a completed experiment directory."""
    from RoboPhD.adapters.candidate_utils import extract_candidate
    from RoboPhD.adapters.runner_utils import find_best_agent
    from RoboPhD.researcher import ParallelAgentResearcher

    agent_name, agent_dir = find_best_agent(experiment_dir)
    best_candidate = extract_candidate(agent_dir, file_mapping)

    checkpoint = ParallelAgentResearcher.load_checkpoint(experiment_dir)
    best_perf = checkpoint["performance_records"][agent_name]
    best_score = best_perf["elo"]

    # Build all_candidates list from agent pool + performance records
    all_candidates = []
    for aid, aperf in checkpoint["performance_records"].items():
        agent_info = checkpoint["agent_pool"].get(aid, {})
        pkg_dir = agent_info.get("package_dir")
        candidate = None
        if pkg_dir:
            agent_path = experiment_dir / pkg_dir
            if agent_path.exists():
                candidate = extract_candidate(agent_path, file_mapping)
        all_candidates.append({
            "name": aid,
            "candidate": candidate,
            "elo": aperf.get("elo", 1500),
            "mean_score": aperf.get("mean_score", 0.0),
            "test_count": aperf.get("test_count", 0),
        })
    all_candidates.sort(key=lambda x: x["elo"], reverse=True)

    num_iterations = checkpoint.get("last_completed_iteration", 0)
    total_evals = sum(checkpoint.get("iteration_fresh_evals", []))

    return OptimizeResult(
        best_candidate=best_candidate,
        best_score=best_score,
        experiment_dir=experiment_dir,
        all_candidates=all_candidates,
        num_iterations_completed=num_iterations,
        total_evaluations=total_evals,
    )
