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

import logging
import tempfile
from dataclasses import dataclass
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
    evaluation_budget: int = 1500
    """Max evaluator calls across all iterations."""
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
    parent_experiments_dir: Union[str, Path, None] = None
    """Root directory under which new experiment folders are created.
    Each run gets its own timestamped subdirectory (e.g.
    ``<parent_experiments_dir>/robophd/optimize_anything_20260329_120000/``).
    None = ``../robophd_runs``."""

    # Reproducibility
    random_seed: Optional[int] = None

    # Evaluation
    eval_timeout: int = 300
    """Seconds per evaluator call before timeout (scores 0)."""

    # Advanced overrides (passed directly to ConfigManager)
    engine_overrides: Optional[Dict[str, Any]] = None
    """Extra ConfigManager parameters for power users (e.g.
    weighted_random_configs, config_schedule, new_agent_test_rounds)."""

    # Resume / extend
    experiment_dir: Union[str, Path, None] = None
    """Path to a prior experiment directory to resume from.
    Pass the experiment_dir from a previous OptimizeResult."""
    extend_iterations: Optional[int] = None
    """Add N more iterations to a resumed run. Only valid with experiment_dir."""
    from_iteration: Optional[int] = None
    """Restart from a specific iteration (discards later work). Only valid with experiment_dir."""


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
    completed_normally: bool
    """Whether the optimization ran to completion (True) or ended early due to failure (False)."""


@dataclass
class RoboPhDEvalConfig:
    """Configuration for eval_candidate()."""

    max_workers: Optional[int] = None
    """Thread pool size for concurrent evaluation. None = Python default."""

    test_repeats: int = 1
    """Number of times to repeat the dataset (scores averaged across all repeats)."""

    eval_timeout: int = 300
    """Seconds per evaluator call before timeout (scores 0)."""


@dataclass
class EvalResult:
    """Result of an eval_candidate() call."""

    mean_score: float
    """Mean score across all evaluated examples."""
    total_score: float
    """Sum of all scores."""
    num_examples: int
    """Total examples evaluated (dataset_size * test_repeats)."""
    per_example_scores: List[float]
    """Ordered list of per-example scores."""
    per_example_diagnostics: List[Dict]
    """Ordered list of per-example diagnostics from the evaluator."""
    had_timeouts: bool
    """Whether any evaluations timed out (leaked threads may be present)."""


@dataclass
class GEPAConfig:
    """Configuration for GEPA Pareto-based reflective text evolution.

    Pass this as the ``config`` argument to ``optimize_anything()`` to use
    GEPA instead of the default RoboPhD ELO engine.

    GEPA evaluates each candidate on a minibatch then validates promising
    candidates on a held-out validation set. Use ``val_dataset`` to provide
    a separate validation pool, or let the engine split ``dataset``.
    """

    evaluation_budget: int = 1500
    """Max evaluator calls across all iterations."""

    val_dataset: Optional[List[Dict]] = None
    """Validation examples. If larger than val_size, val_size examples are
    sampled and the rest are added to training. If None, dataset is split."""

    val_size: int = 100
    """Validation set size. Paper results showed 100 outperforms 200."""

    reflection_model: str = "opus-4.6"
    """Model for GEPA reflection (mutation proposals)."""

    max_workers: Optional[int] = None
    """Thread pool size for concurrent evaluation. None = Python default."""

    seed: int = 0
    """Random seed for reproducibility."""

    eval_timeout: int = 300
    """Seconds per evaluator call before timeout."""

    test_repeats: int = 1
    """Number of test set repetitions."""

    max_test_workers: Optional[int] = None
    """Test eval thread pool size. Default: max_workers // 2."""

    debug_log_probability: float = 0.1
    """Probability (0-1.0) of logging LLM calls for debugging."""

    only_log_reflection: bool = True
    """If true, debug logging only applies to reflection model."""

    parent_experiments_dir: Union[str, Path, None] = None
    """Root directory for experiment output. None = ``../robophd_runs``."""


@dataclass
class AutoresearchConfig:
    """Configuration for Autoresearch single-session greedy hill-climbing.

    Pass this as the ``config`` argument to ``optimize_anything()`` to use
    Autoresearch instead of the default RoboPhD ELO engine.

    Autoresearch runs a single continuous Claude Code session that
    autonomously experiments with the agent code, using greedy keep/discard
    decisions based on a held-out validation set.
    """

    evaluation_budget: int = 1500
    """Max evaluator calls (train + val)."""

    val_dataset: Optional[List[Dict]] = None
    """Validation examples. If larger than val_size, val_size examples are
    sampled and the rest are added to training. If None, dataset is split."""

    val_size: int = 100
    """Validation set size. Paper results showed 100 outperforms 200."""

    model: str = "opus-4.6"
    """Claude Code model for the autonomous session."""

    max_workers: Optional[int] = None
    """Max parallel workers for evaluation. None = Python default."""

    eval_timeout: int = 300
    """Per-example evaluation timeout in seconds."""

    seed: int = 0
    """Random seed."""

    overall_timeout: Optional[int] = None
    """Max wall-clock seconds for the entire run. None = no limit."""

    parent_experiments_dir: Union[str, Path, None] = None
    """Root directory for experiment output. None = ``../robophd_runs``."""


def _validate_resume_config(cfg: RoboPhDConfig) -> None:
    """Validate resume-related config fields."""
    if cfg.extend_iterations is not None and cfg.experiment_dir is None:
        raise ValueError("extend_iterations requires experiment_dir")
    if cfg.from_iteration is not None and cfg.experiment_dir is None:
        raise ValueError("from_iteration requires experiment_dir")


def _build_resume_kwargs(
    cfg: RoboPhDConfig,
) -> Tuple[Any, int, Dict[str, Any], Dict[str, str]]:
    """Build researcher constructor args for resuming from a checkpoint.

    Mirrors the resume logic in run_robophd.py:303-383.

    Returns:
        (config_manager, num_iterations, researcher_kwargs, task_config)
        where task_config contains file_mapping, objective, background, etc.
    """
    from RoboPhD.config_manager import ConfigManager, ConfigSource
    from RoboPhD.researcher import ParallelAgentResearcher

    experiment_dir = Path(cfg.experiment_dir)
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    checkpoint = ParallelAgentResearcher.load_checkpoint(experiment_dir)

    if "config_manager" not in checkpoint:
        raise ValueError(f"Checkpoint missing ConfigManager data: {experiment_dir}")

    # Recover file_mapping from task_config
    task_config = checkpoint.get("task_config", {})
    file_mapping = task_config.get("file_mapping")
    if not file_mapping:
        raise ValueError(
            f"Checkpoint missing file_mapping in task_config. "
            f"This run was created before resume support was added — "
            f"it cannot be resumed via the API."
        )

    config_manager = ConfigManager.from_checkpoint(checkpoint["config_manager"])
    last_completed = checkpoint["last_completed_iteration"]
    checkpoint_num_iterations = checkpoint.get("num_iterations", last_completed)

    if cfg.from_iteration is not None:
        resume_from = cfg.from_iteration
        if resume_from > last_completed:
            raise ValueError(
                f"from_iteration={resume_from} exceeds last completed "
                f"iteration ({last_completed})"
            )
        config_manager.clear_from_iteration(resume_from)
        logger.info("Restarting from iteration %d", resume_from)
    else:
        resume_from = last_completed + 1
        logger.info("Auto-resuming from iteration %d", resume_from)

    if cfg.extend_iterations is not None:
        num_iterations = checkpoint_num_iterations + cfg.extend_iterations
        checkpoint["num_iterations"] = num_iterations
        logger.info(
            "Extending by %d iterations (to %d total)",
            cfg.extend_iterations, num_iterations,
        )
    else:
        num_iterations = checkpoint_num_iterations

    # Apply engine_overrides as a delta on the resume iteration
    if cfg.engine_overrides:
        config_manager.apply_delta(
            iteration=resume_from,
            delta=cfg.engine_overrides,
            source=ConfigSource.CLI,
            rationale=f"engine_overrides on resume: {cfg.engine_overrides}",
        )
        logger.info(
            "Applied engine_overrides at iteration %d: %s",
            resume_from, cfg.engine_overrides,
        )

    researcher_kwargs = dict(
        resume_mode=True,
        resume_from_iteration=resume_from,
        resume_checkpoint=checkpoint,
        resume_experiment_dir=experiment_dir,
    )

    return config_manager, num_iterations, researcher_kwargs, task_config


def optimize_anything(
    evaluator: Callable,
    dataset: List[Dict],
    seed_candidate: Optional[Dict[str, str]] = None,
    objective: str = "",
    background: str = "",
    config: Optional[Union[RoboPhDConfig, GEPAConfig, AutoresearchConfig]] = None,
    task_name: str = "optimize_anything",
) -> OptimizeResult:
    """Optimize text artifacts using evolutionary search.

    The engine is determined by the config type:

    - ``RoboPhDConfig`` (default): Multi-agent ELO competition with Deep Focus
    - ``GEPAConfig``: Pareto-based reflective text evolution
    - ``AutoresearchConfig``: Single Claude Code session with greedy hill-climbing

    Example::

        # RoboPhD (default)
        result = optimize_anything(evaluator=e, dataset=d,
                                   seed_candidate={"prompt": "..."}, objective="...")

        # GEPA
        result = optimize_anything(evaluator=e, dataset=train,
                                   seed_candidate={"prompt": "..."},
                                   config=GEPAConfig(val_dataset=val))

        # Autoresearch
        result = optimize_anything(evaluator=e, dataset=train,
                                   seed_candidate={"prompt": "..."},
                                   config=AutoresearchConfig(val_dataset=val))

    Supports resume/extend via ``config.experiment_dir`` (RoboPhD only)::

        result = optimize_anything(
            evaluator=..., dataset=..., objective="...",
            config=RoboPhDConfig(experiment_dir=result.experiment_dir, extend_iterations=5),
        )

    Args:
        evaluator: Scoring function with signature
            ``(candidate: dict, example: dict) -> (score: float, diagnostics: dict)``.
            Higher scores are better. Must be thread-safe (called concurrently).
        dataset: List of example dicts. For RoboPhD, all examples enter the ELO
            competition pool. For GEPA/Autoresearch, this is the training pool
            (validation comes from ``config.val_dataset``).
        seed_candidate: Initial text artifact(s) to optimize. Dict mapping
            component names to text content, e.g. ``{"prompt": "Solve carefully"}``.
            Required for fresh runs; optional when resuming (recovered from checkpoint).
        objective: Natural-language optimization goal shown to the evolution AI.
        background: Optional domain documentation shown to the evolution AI.
        config: Engine configuration. Type determines the engine. If None,
            uses ``RoboPhDConfig()`` defaults.
        task_name: Name for the experiment directory.

    Returns:
        OptimizeResult with best_candidate, best_score, and experiment_dir.
    """
    if not dataset:
        raise ValueError("dataset must be a non-empty list of example dicts")

    cfg = config or RoboPhDConfig()

    # Dispatch to engine based on config type
    if isinstance(cfg, GEPAConfig):
        from RoboPhD.engines.gepa import run_gepa
        return run_gepa(evaluator, dataset, seed_candidate, objective, background, cfg, task_name)
    elif isinstance(cfg, AutoresearchConfig):
        from RoboPhD.engines.autoresearch import run_autoresearch
        return run_autoresearch(evaluator, dataset, seed_candidate, objective, background, cfg, task_name)

    # --- RoboPhD ELO engine (default) ---
    from RoboPhD.config_manager import ConfigManager, ConfigSource
    from RoboPhD.researcher import ParallelAgentResearcher
    from RoboPhD.candidate_utils import materialize_candidate
    _validate_resume_config(cfg)
    run_dir = Path(cfg.parent_experiments_dir) if cfg.parent_experiments_dir else Path("../robophd_runs")

    if cfg.experiment_dir:
        # --- Resume path ---
        config_manager, num_iterations, resume_kwargs, saved_task_config = (
            _build_resume_kwargs(cfg)
        )
        file_mapping = saved_task_config["file_mapping"]

        # Recover objective/background from checkpoint; caller overrides if provided
        effective_objective = objective or saved_task_config.get("objective", "")
        effective_background = background or saved_task_config.get("background", "")

        runtime_config = {
            "evaluator_fn": evaluator,
            "dataset": dataset,
            "file_mapping": file_mapping,
            "task_objective": effective_objective,
            "task_description": effective_objective,
            "task_background": effective_background,
            "task_name": task_name,
            "diagnostic_files": {},
            "runs_dir": str(run_dir),
            "eval_timeout": cfg.eval_timeout,
        }

        researcher = ParallelAgentResearcher(
            config_manager=config_manager,
            num_iterations=num_iterations,
            runtime_config=runtime_config,
            **resume_kwargs,
        )
        initial_agents = None

    else:
        # --- Fresh start ---
        if not seed_candidate:
            raise ValueError("seed_candidate is required for fresh runs")

        file_mapping = {key: key for key in seed_candidate}

        seed_agents_dir = run_dir / "robophd" / "_optimize_anything_seeds"
        seed_agents_dir.mkdir(parents=True, exist_ok=True)
        seed_dir = Path(tempfile.mkdtemp(dir=seed_agents_dir, prefix="seed_"))
        materialize_candidate(seed_candidate, seed_dir, file_mapping, name="seed")
        seed_agent_name = seed_dir.name

        config_manager = ConfigManager()
        researcher_config = {
            "domain": "external",
            "evolution_strategy": cfg.evolution_strategy,
            "evolution_model": cfg.evolution_model,
            "evolution_timeout": cfg.evolution_timeout,
            "eval_timeout": cfg.eval_timeout,
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

        runtime_config = {
            "evaluator_fn": evaluator,
            "dataset": dataset,
            "file_mapping": file_mapping,
            "task_objective": objective,
            "task_description": objective,
            "task_background": background,
            "task_name": task_name,
            "diagnostic_files": {},
            "runs_dir": str(run_dir),
            "eval_timeout": cfg.eval_timeout,
        }

        # Persist objective/background in task_config so they survive resume
        researcher = ParallelAgentResearcher(
            config_manager=config_manager,
            num_iterations=cfg.num_iterations,
            random_seed=cfg.random_seed,
            runtime_config=runtime_config,
            task_config={
                "file_mapping": file_mapping,
                "objective": objective,
                "background": background,
            },
        )
        initial_agents = [seed_agent_name]

    completed_normally = researcher.run(initial_agents=initial_agents)

    try:
        result = _build_result(researcher.experiment_dir, file_mapping, completed_normally)
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
    Supports resume/extend via ``config.experiment_dir``.

    Args:
        task_name: Registered task name (e.g. "cant_be_late_stdout", "arc_agi_1").
        seed_agent: Path to seed agent directory. Defaults to task's default_seed_agent.
        task_config: Task-specific configuration (e.g. split, model overrides).
        config: Engine configuration. If None, uses ``RoboPhDConfig()`` defaults.

    Returns:
        OptimizeResult with best_candidate, best_score, and experiment_dir.
    """
    from RoboPhD.config_manager import ConfigManager, ConfigSource
    from RoboPhD.researcher import ParallelAgentResearcher
    from RoboPhD.tasks import get_task

    task = get_task(task_name)
    cfg = config or RoboPhDConfig()
    _validate_resume_config(cfg)
    tc = dict(task_config or {})

    # Build full config for evaluator/dataset factories
    full_config = {**task.config_defaults, **tc}

    # Build evaluator and dataset (always needed, even on resume)
    evaluator = task.evaluator_factory(full_config)
    dataset = task.dataset_builder(full_config)
    if not dataset:
        raise ValueError(f"Empty dataset for task {task_name!r}")

    run_dir = Path(cfg.parent_experiments_dir) if cfg.parent_experiments_dir else Path("../robophd_runs")
    file_mapping = task.file_mapping

    if cfg.experiment_dir:
        # --- Resume path ---
        config_manager, num_iterations, resume_kwargs, _ = (
            _build_resume_kwargs(cfg)
        )
        # Use task.file_mapping (authoritative) rather than checkpoint's copy

        runtime_config = {
            "evaluator_fn": evaluator,
            "dataset": dataset,
            "file_mapping": file_mapping,
            "task_objective": task.objective,
            "task_description": task.description,
            "task_background": task.background,
            "task_name": task.name,
            "diagnostic_files": task.diagnostic_files,
            "runs_dir": str(run_dir),
            "eval_timeout": cfg.eval_timeout,
        }

        researcher = ParallelAgentResearcher(
            config_manager=config_manager,
            num_iterations=num_iterations,
            runtime_config=runtime_config,
            task_config=tc,
            **resume_kwargs,
        )
        initial_agents = None

    else:
        # --- Fresh start ---
        seed_agent_path = Path(tc.get("seed_agent", seed_agent or task.default_seed_agent))
        if not seed_agent_path.exists():
            raise FileNotFoundError(f"Seed agent not found: {seed_agent_path}")

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

        runtime_config = {
            "evaluator_fn": evaluator,
            "dataset": dataset,
            "file_mapping": file_mapping,
            "task_objective": task.objective,
            "task_description": task.description,
            "task_background": task.background,
            "task_name": task.name,
            "diagnostic_files": task.diagnostic_files,
            "runs_dir": str(run_dir),
            "eval_timeout": cfg.eval_timeout,
        }

        researcher = ParallelAgentResearcher(
            config_manager=config_manager,
            num_iterations=cfg.num_iterations,
            random_seed=cfg.random_seed,
            runtime_config=runtime_config,
            task_config=tc,
        )
        initial_agents = [seed_agent_path.name]

    completed_normally = researcher.run(initial_agents=initial_agents)

    try:
        result = _build_result(researcher.experiment_dir, file_mapping, completed_normally)
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


def _build_result(experiment_dir: Path, file_mapping: Dict[str, str], completed_normally: bool) -> OptimizeResult:
    """Extract OptimizeResult from a completed experiment directory."""
    from RoboPhD.candidate_utils import extract_candidate
    from RoboPhD.runner_utils import find_best_agent
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
        completed_normally=completed_normally,
    )


def eval_candidate(
    evaluator: Callable,
    dataset: List[Dict],
    candidate: Dict[str, str],
    config: Optional[RoboPhDEvalConfig] = None,
) -> EvalResult:
    """Evaluate a candidate on a dataset using the given evaluator.

    This is the evaluation companion to optimize_anything(). It runs the
    same evaluator and dataset you used for optimization, but on a specific
    candidate — typically ``result.best_candidate`` from a prior optimization.

    Args:
        evaluator: Scoring function with signature
            ``(candidate: dict, example: dict) -> (score: float, diagnostics: dict)``.
            Must be thread-safe (called concurrently).
        dataset: List of example dicts for evaluation.
        candidate: Text artifact(s) to evaluate (same shape as seed_candidate).
        config: Evaluation configuration. If None, uses ``RoboPhDEvalConfig()`` defaults.

    Returns:
        EvalResult with mean_score, per_example_scores, and diagnostics.
    """
    from RoboPhD.eval_utils import run_parallel_eval

    if not dataset:
        raise ValueError("dataset must be a non-empty list of example dicts")
    if not candidate:
        raise ValueError("candidate must be a non-empty dict")

    cfg = config or RoboPhDEvalConfig()
    examples = dataset * cfg.test_repeats

    logger.info(
        "Evaluating candidate on %d examples (%d unique x %d repeats)",
        len(examples), len(dataset), cfg.test_repeats,
    )

    result = run_parallel_eval(
        evaluator, candidate, examples,
        max_workers=cfg.max_workers,
        eval_timeout=cfg.eval_timeout,
    )

    return EvalResult(
        mean_score=result["test_results"]["mean_test_score"],
        total_score=result["test_results"]["total_test_score"],
        num_examples=result["test_results"]["total_test_problems"],
        per_example_scores=result["scores"],
        per_example_diagnostics=result["diagnostics"],
        had_timeouts=result["timed_out"],
    )


def eval_run(
    evaluator: Callable,
    dataset: List[Dict],
    experiment_dir: Union[str, Path],
    config: Optional[RoboPhDEvalConfig] = None,
) -> EvalResult:
    """Evaluate the best agent from a completed optimization run.

    Extracts the highest-ELO agent from the checkpoint, then runs
    ``eval_candidate()`` on it. Typical use: test-set evaluation after
    ``optimize_anything()`` finishes.

    Args:
        evaluator: Same evaluator used during optimization.
        dataset: Test examples to evaluate on.
        experiment_dir: Path to the experiment directory (contains checkpoint.json).
        config: Evaluation configuration. If None, uses ``RoboPhDEvalConfig()`` defaults.

    Returns:
        EvalResult with mean_score, per_example_scores, and diagnostics.
    """
    from RoboPhD.runner_utils import find_best_agent
    from RoboPhD.candidate_utils import extract_candidate
    from RoboPhD.researcher import ParallelAgentResearcher

    experiment_dir = Path(experiment_dir)
    agent_name, agent_dir = find_best_agent(experiment_dir)

    # Recover file_mapping from checkpoint
    checkpoint = ParallelAgentResearcher.load_checkpoint(experiment_dir)
    task_config = checkpoint.get("task_config", {})
    file_mapping = task_config.get("file_mapping")
    if not file_mapping:
        raise ValueError(
            f"Checkpoint missing file_mapping in task_config: {experiment_dir}. "
            f"Cannot determine agent artifact structure."
        )

    best_candidate = extract_candidate(agent_dir, file_mapping)
    logger.info(f"Evaluating best agent: {agent_name}")

    return eval_candidate(evaluator, dataset, best_candidate, config)
