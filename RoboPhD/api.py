"""
Simple programmatic API for RoboPhD optimization.

Provides an optimize_anything() interface inspired by GEPA's API, wrapping
RoboPhD's Elo evolution engine (ParallelAgentResearcher) behind a simple
function call.

Usage:
    from RoboPhD import optimize_anything

    def evaluator(candidate, example):
        score = 1.0 if candidate["prompt"] in example["expected"] else 0.0
        return score, {"output": candidate["prompt"]}

    result = optimize_anything(
        evaluator=evaluator,
        dataset=[{"input": "2+2", "expected": "4"}, ...],
        seed_agents={"baseline": {"prompt": "Solve the math problem:"}},
        objective="Maximize accuracy on math problems",
    )
    print(result.best_candidate, result.best_score)
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class RoboPhDConfig:
    """Configuration for RoboPhD Elo evolution engine.

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
    evolution_model: str = "opus-5"
    evolution_timeout: int = 3600
    """Seconds per evolution session (Claude Code CLI subprocess)."""

    # Meta-evolution
    meta_evolution_strategy: Optional[str] = None
    """Meta-evolution strategy name (e.g. ``train_a_winner``); ``None`` disables meta-evolution."""
    meta_evolution_model: str = "opus-5"
    """Model for meta-evolution sessions (Claude Code CLI subprocess)."""
    meta_evolution_first_iteration: int = 4
    """First iteration at which meta-evolution fires."""
    meta_evolution_cadence: int = 3
    """Iterations between meta-evolution firings (firing at ``first``, ``first + cadence``, ...)."""

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

    # Evolution sandbox
    extra_read_paths: Optional[List[str]] = None
    """Absolute paths added to the evolution sandbox's read scope. The
    default sandbox restricts evolution Claude CLI sessions to reads
    under the experiment directory only; add paths here when a task uses
    symlinked-in resources outside that tree (e.g., text2sql adds
    ``RoboPhD/benchmark_resources``). Read-only — does not grant write
    permission."""

    session_tools: Optional[List[str]] = None
    """Files copied into ``<experiment>/session_tools/`` at researcher
    startup as helper scripts for evolution sessions (inside the
    sessions' read scope, outside their write root). Copied on fresh
    starts AND resumes — a resumed run picks up repo-side fixes — and a
    missing source file fails loudly. Not persisted in the checkpoint;
    re-supply on every invocation, like ``extra_read_paths``."""

    # Task-specific persistence
    task_config_extras: Optional[Dict[str, Any]] = None
    """Caller-owned keys merged into the run's ``task_config``, which the
    researcher persists verbatim into ``checkpoint.json`` every iteration
    and round-trips on resume. Use this for task-specific knobs that must
    survive interruption (e.g. asta_ds1000's cost-penalty parameters).
    Nest values under a single task-named key to avoid collisions. The
    reserved keys ``file_mapping``/``objective``/``background`` cannot be
    overridden.

    Extras are immutable across a run, enforced on resume: a key absent
    from the checkpoint's stored task_config is added (bootstrapping runs
    from before the task persisted it), but a key whose value differs
    from the stored one raises ``ValueError``. Resolve CLI-derived values
    against the stored ones before passing them — see
    ``runner_utils.read_task_config_extras`` /
    ``runner_utils.resolve_run_immutable`` for the standard pattern."""

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
    """Result of an optimize_anything() call."""

    best_candidate: Dict[str, str]
    """Best agent's text artifacts (same keys as the seed agents')."""
    best_score: float
    """Best agent's Elo rating."""
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
    """Iteration-level score from the evaluator's aggregator (or simple
    mean when no aggregator is defined).

    **Scale is evaluator-defined and may differ between training and
    test modes for the same evaluator.** Notably DS-1000's
    ``Ds1000Evaluator.aggregate`` returns:

    - ``SCORE_SCALE × mean_raw - penalty`` (~85 scale) in training mode
      (``apply_cost_penalty=True``), so the [0, 1] cost penalty stays
      a tiebreaker against the percentage-scaled accuracy.
    - ``mean_raw`` ([0, 1] fraction) in test mode
      (``apply_cost_penalty=False``), for leaderboard parity.

    Cross-task and cross-mode aggregators must be interpreted with this
    in mind — comparing raw ``mean_score`` numbers across tasks (or
    across test/training for the same task) is not meaningful unless
    you've checked the aggregator's scale convention.
    """
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
    aggregate_explanation: str = ""
    """Explanation string from the evaluator's `aggregate` method, if any.
    Empty when the evaluator uses the default mean aggregator. Populated
    by tasks that apply batch-level scoring (e.g. DS-1000's cost-penalty
    explanation in training mode)."""


@dataclass
class GEPAConfig:
    """Configuration for GEPA Pareto-based reflective text evolution.

    Pass this as the ``config`` argument to ``optimize_anything()`` to use
    GEPA instead of the default RoboPhD Elo engine.

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

    reflection_model: str = "opus-5"
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
    Autoresearch instead of the default RoboPhD Elo engine.

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

    model: str = "opus-5"
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

    session_tools: Optional[List[str]] = None
    """Files copied into ``<output_dir>/session_tools/`` at run startup
    as helper scripts for the autonomous session (readable/runnable from
    the workspace at ``../session_tools/``, outside its write root). A
    missing source file fails loudly."""


_TASK_CONFIG_RESERVED_KEYS = ("file_mapping", "objective", "background")


def _merge_task_config_extras(
    task_config: Dict[str, Any], extras: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge caller-owned task_config extras over the framework keys.

    Reserved keys are framework-owned (file_mapping drives candidate
    materialization; objective/background are recovered on resume) —
    letting extras shadow them would corrupt resume, so collide loudly.

    One level deep: when a key holds a dict on both sides (the documented
    nest-under-a-task-named-key convention), sub-keys are merged rather
    than the dict replaced wholesale — so extras built by older task code
    can't silently drop sub-keys a newer version had persisted.
    """
    if not extras:
        return task_config
    collisions = sorted(k for k in extras if k in _TASK_CONFIG_RESERVED_KEYS)
    if collisions:
        raise ValueError(
            f"task_config_extras cannot override reserved task_config "
            f"keys {collisions}; nest task-specific values under a "
            f"task-named key instead."
        )
    merged = dict(task_config)
    for k, v in extras.items():
        stored = merged.get(k)
        if isinstance(stored, dict) and isinstance(v, dict):
            merged[k] = {**stored, **v}
        else:
            merged[k] = v
    return merged


def _changed_extras_on_resume(
    stored: Dict[str, Any], extras: Dict[str, Any],
) -> List[str]:
    """List extras entries that would CHANGE a stored task_config value.

    Additions (key or sub-key absent from stored) are fine — that's the
    bootstrap case. Matches the merge's one-level-deep semantics: for a
    dict-valued key, sub-keys are compared individually so a task that
    grew a new knob can still resume its older runs.
    """
    changed: List[str] = []
    for k, v in extras.items():
        if k not in stored:
            continue
        sv = stored[k]
        if isinstance(sv, dict) and isinstance(v, dict):
            changed.extend(
                f"{k}.{sk}: stored={sv[sk]!r} -> extras={sval!r}"
                for sk, sval in v.items()
                if sk in sv and sv[sk] != sval
            )
        elif sv != v:
            changed.append(f"{k}: stored={sv!r} -> extras={v!r}")
    return changed


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

    # Apply engine_overrides as a delta on the resume iteration.
    #
    # !!! DANGER — caller invariant !!!
    # On resume, populate engine_overrides ONLY with values the user
    # explicitly set on the resume CLI. Every key in this dict is
    # applied as a fresh delta that OVERWRITES whatever was in effect
    # at iteration `resume_from`. A CLI default silently packed by
    # main.py will clobber the original run's setting — this is the
    # bug pattern fixed in commits 5d654f0 + b684c11 for asta_ds1000.
    #
    # The safe pattern (see examples/asta_ds1000/main.py):
    #   1. argparse default=None for any flag exposed in engine_overrides
    #   2. `if args.X is not None: engine_overrides["X"] = args.X`
    #   3. If the task has a default that differs from RoboPhD's
    #      framework default (config_manager.get_defaults), pack it
    #      only on initial runs: `elif not is_resume: engine_overrides["X"] = TASK_DEFAULT`
    #
    # Examples that only pack `--engine-config` JSON (most of them) are
    # automatically safe — argparse + JSON only carries user-set keys.
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

    # Merge task_config_extras over the checkpoint's stored task_config
    # and pass the result explicitly (the explicit param wins over
    # resume_checkpoint in the researcher). The next _save_checkpoint
    # persists the merged dict, so a bootstrap resume of a run from
    # before its extras were persisted heals itself.
    #
    # Immutability backstop: extras may ADD keys absent from the stored
    # task_config (the bootstrap case) but may not CHANGE stored ones.
    # A changed value here means the caller passed extras computed from
    # CLI flags without resolving them against the stored values first —
    # silently accepting it would mutate a "persisted" task knob mid-run
    # (the failure class task_config_extras exists to prevent). Callers
    # with a flag-level guard (runner_utils.resolve_run_immutable) error
    # before reaching this; the backstop protects callers without one.
    if cfg.task_config_extras:
        changed = _changed_extras_on_resume(task_config, cfg.task_config_extras)
        if changed:
            raise ValueError(
                f"task_config_extras would change stored task_config "
                f"value(s) on resume: {'; '.join(changed)}. Extras are "
                f"immutable across a run: they may add keys missing from "
                f"the checkpoint (bootstrap) but not change stored ones. "
                f"Resolve extras against the stored values before passing "
                f"them (see runner_utils.resolve_run_immutable), or start "
                f"a new run to change the values."
            )
    task_config = _merge_task_config_extras(task_config, cfg.task_config_extras)

    researcher_kwargs = dict(
        resume_mode=True,
        resume_from_iteration=resume_from,
        resume_checkpoint=checkpoint,
        resume_experiment_dir=experiment_dir,
        task_config=task_config,
    )

    return config_manager, num_iterations, researcher_kwargs, task_config


# Seed names become directory names under the seeds container and then agent
# IDs in the pool, so they must be safe single path components. The leading
# character is restricted separately from the rest: a name of all-allowed
# characters can still be "." or ".." (which would materialize the agent into
# the container root or its parent), ".hidden" (which read_agent_dir would
# skip if it appeared inside an agent), or "-x" (ambiguous as a CLI value to
# --eval-agent).
_SEED_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]*$")
# ...and must not look like an evolved agent. `--from-iteration` archival
# parses the `iter<N>_` prefix off any directory in <experiment_dir>/agents/
# to decide what to move out (researcher._archive_from_iteration), so a seed
# carrying its source run's name would be read as "created in iteration N of
# THIS run" and silently archived away.
_EVOLVED_NAME_RE = re.compile(r"^iter\d")


# Parameters removed from optimize_anything, and what to do instead. Dropping
# them outright leaves callers with a bare "unexpected keyword argument", which
# is loud but says nothing — and this is the project's primary public API, so
# the retirement should explain itself the way a retired model handle does.
_RETIRED_KWARGS = {
    "seed_candidate": (
        "seed_candidate was replaced by seed_agents, which is keyed by AGENT "
        'name: seed_agents={"baseline": {"agent.py": ...}} for artifacts, or '
        'seed_agents={"baseline": Path(".../seeds/baseline")} to read an '
        "agent directory. The name you choose is what appears in the Elo "
        "table and --eval-agent."
    ),
    "seed_candidates": (
        "seed_candidates was renamed to seed_agents. Same {name: candidate} "
        "shape, and values may now also be a Path to an agent directory."
    ),
}


def _reject_retired_kwargs(kwargs: Dict[str, Any]) -> None:
    """Turn a call using a removed parameter into an actionable error."""
    for name in kwargs:
        if name in _RETIRED_KWARGS:
            raise TypeError(f"optimize_anything(): {_RETIRED_KWARGS[name]}")
    # Anything else is an ordinary typo; keep Python's own wording rather than
    # letting **kwargs quietly change what that failure looks like.
    raise TypeError(
        f"optimize_anything() got an unexpected keyword argument "
        f"'{next(iter(kwargs))}'"
    )


def _ensure_seeds_dir(run_dir: Path) -> Path:
    """The staging area seed agents are materialized into.

    Purely startup scaffolding: the researcher copies each agent into
    ``<experiment_dir>/agents/`` at load, and ``agents_directory`` is never
    read again (``researcher.load_initial_agents`` is the only reader, and
    resume skips it).
    """
    seeds_dir = Path(run_dir) / "robophd" / "_optimize_anything_seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)
    return seeds_dir


def _resolve_seed_agents(
    seed_agents: Dict[str, Union[Path, Dict[str, str]]],
) -> Dict[str, Dict[str, str]]:
    """Resolve ``{name: directory-or-artifacts}`` to ``{name: artifacts}``.

    Directory values are read with ``read_agent_dir``; artifact dicts pass
    through. Names are validated here rather than at materialization so the
    failure names the offending seed instead of surfacing as a path error.

    Raises ValueError (or FileNotFoundError, from a missing directory).
    """
    from RoboPhD.candidate_utils import read_agent_dir

    if not seed_agents:
        raise ValueError(
            "seed_agents must be a non-empty {name: directory-or-artifacts} dict"
        )

    resolved: Dict[str, Dict[str, str]] = {}
    for name, source in seed_agents.items():
        if not isinstance(name, str) or not _SEED_NAME_RE.match(name):
            raise ValueError(
                f"Invalid seed name {name!r}: names become directory names and "
                f"agent IDs, so they must match {_SEED_NAME_RE.pattern} — "
                f"letters, digits, '.', '_' and '-', starting with a letter, "
                f"digit or underscore. That excludes path separators, '.' and "
                f"'..', and names beginning with '.' or '-'."
            )
        if _EVOLVED_NAME_RE.match(name):
            raise ValueError(
                f"Invalid seed name {name!r}: names starting with 'iter<N>' "
                f"collide with evolved-agent naming. --from-iteration archival "
                f"parses that prefix as an iteration number of the CURRENT run "
                f"and would move the seed out of the pool. Prefix it instead "
                f"(e.g. 'seed_{name}')."
            )

        # os.PathLike, not str: a bare string is ambiguous between "a
        # directory path" and one artifact's contents, and the second is the
        # likely mistake (passing ONE agent's artifacts where a pool of agents
        # belongs). Requiring Path makes the two forms disjoint, so that
        # mistake gets a message instead of a confusing path error.
        if isinstance(source, os.PathLike):
            resolved[name] = read_agent_dir(Path(source))
        elif isinstance(source, dict):
            if not source:
                raise ValueError(f"Seed {name!r} has no artifacts")
            wrong = [k for k, v in source.items() if not isinstance(v, str)]
            if wrong:
                raise ValueError(
                    f"Seed {name!r} maps {wrong[0]!r} to "
                    f"{type(source[wrong[0]]).__name__}, not text. Artifact "
                    f"dicts are {{relative path: file contents}}."
                )
            resolved[name] = source
        else:
            hint = (
                " If that is a directory, wrap it in Path()."
                if isinstance(source, str) else ""
            )
            raise ValueError(
                f"Seed {name!r} must be a Path to an agent directory or a "
                f"{{relative path: file contents}} dict, got "
                f"{type(source).__name__}.{hint} Note seed_agents is keyed by "
                f'AGENT name: {{"baseline": Path(...)}} or '
                f'{{"baseline": {{"agent.py": "..."}}}} — not by artifact name.'
            )

    # One file_mapping is derived from these keys and applied to every seed. A
    # seed missing a mapped file fails the researcher's agent-dir validation
    # and is dropped with only a printed warning, so catch it here.
    key_sets = {name: frozenset(c) for name, c in resolved.items()}
    reference_name, reference_keys = next(iter(key_sets.items()))
    for name, keys in key_sets.items():
        if keys != reference_keys:
            raise ValueError(
                f"All seed agents must have the same artifacts (they define a "
                f"single file_mapping). Seed {reference_name!r} has "
                f"{sorted(reference_keys)}; seed {name!r} has {sorted(keys)}."
            )
    return resolved


def optimize_anything(
    evaluator: Callable,
    dataset: List[Dict],
    seed_agents: Optional[Dict[str, Union[Path, Dict[str, str]]]] = None,
    objective: str = "",
    background: str = "",
    config: Optional[Union[RoboPhDConfig, GEPAConfig, AutoresearchConfig]] = None,
    task_name: str = "optimize_anything",
    **retired,
) -> OptimizeResult:
    """Optimize text artifacts using evolutionary search.

    The engine is determined by the config type:

    - ``RoboPhDConfig`` (default): Multi-agent Elo competition with Deep Focus
    - ``GEPAConfig``: Pareto-based reflective text evolution
    - ``AutoresearchConfig``: Single Claude Code session with greedy hill-climbing

    Example::

        # RoboPhD (default) — seed from an agent directory on disk
        result = optimize_anything(evaluator=e, dataset=d, objective="...",
                                   seed_agents={"baseline": HERE / "seeds" / "baseline"})

        # Several seeds, e.g. the winners of prior runs
        result = optimize_anything(evaluator=e, dataset=d, objective="...",
                                   seed_agents={"cheap": dir_a, "expensive": dir_b})

        # GEPA / Autoresearch optimize one agent, so the pool must hold one
        result = optimize_anything(evaluator=e, dataset=train,
                                   seed_agents={"baseline": {"prompt": "..."}},
                                   config=GEPAConfig(val_dataset=val))

    Supports resume/extend via ``config.experiment_dir`` (RoboPhD only)::

        result = optimize_anything(
            evaluator=..., dataset=..., objective="...",
            config=RoboPhDConfig(experiment_dir=result.experiment_dir, extend_iterations=5),
        )

    Args:
        evaluator: Scoring function with signature
            ``(candidate: dict, example: dict) -> (score: float, diagnostics: dict)``.
            Higher scores are better. Must be thread-safe (called concurrently).
        dataset: List of example dicts. For RoboPhD, all examples enter the Elo
            competition pool. For GEPA/Autoresearch, this is the training pool
            (validation comes from ``config.val_dataset``).
        seed_agents: The agents the run starts from, as ``{name: source}``.
            **Keys are agent names** — they become directory names, pool IDs,
            and what you pass to ``--eval-agent``. Each value is either a path
            to an agent directory (read with ``candidate_utils.read_agent_dir``,
            skipping dot-prefixed entries and ``__pycache__``) or the artifacts
            themselves as ``{relative path: file contents}``.

            Required for fresh runs; optional when resuming (recovered from the
            checkpoint). Every seed enters the pool at Elo 1500 and competes
            from iteration 1, and all of them must have the same artifacts,
            since they define one ``file_mapping``.

            Names must be safe single path components and must not start with
            ``iter<N>`` — ``--from-iteration`` archival parses that prefix as an
            iteration number of the *current* run and would move the seed out of
            the pool.

            Seeding more agents than ``agents_per_iteration`` loses none of them:
            untested agents have selection priority, so a 4th seed in a 3-slot
            run is tested at iteration 2. GEPA and Autoresearch optimize a single
            agent and require a one-entry pool.
        objective: Natural-language optimization goal shown to the evolution AI.
        background: Optional domain documentation shown to the evolution AI.
        config: Engine configuration. Type determines the engine. If None,
            uses ``RoboPhDConfig()`` defaults. RoboPhD-engine-specific
            knobs (e.g. ``extra_read_paths`` for the evolution sandbox,
            ``task_config_extras`` for checkpoint-persisted task values)
            live on ``RoboPhDConfig`` — see its field docstrings.
        task_name: Name for the experiment directory.

    Returns:
        OptimizeResult with best_candidate, best_score, and experiment_dir.
    """
    if retired:
        _reject_retired_kwargs(retired)
    if not dataset:
        raise ValueError("dataset must be a non-empty list of example dicts")

    cfg = config or RoboPhDConfig()

    # Dispatch to engine based on config type. GEPA and Autoresearch evolve a
    # single agent, so they take one candidate rather than a pool; resolve the
    # pool here and hand the sole entry down.
    if isinstance(cfg, (GEPAConfig, AutoresearchConfig)):
        single_candidate = None
        if seed_agents:
            resolved = _resolve_seed_agents(seed_agents)
            if len(resolved) > 1:
                raise ValueError(
                    f"{type(cfg).__name__} optimizes a single agent, but "
                    f"seed_agents holds {len(resolved)}: "
                    f"{', '.join(resolved)}. Pass a one-entry pool, or use "
                    f"RoboPhDConfig to evolve several agents against each other."
                )
            single_candidate = next(iter(resolved.values()))
        if isinstance(cfg, GEPAConfig):
            from RoboPhD.engines.gepa import run_gepa
            return run_gepa(evaluator, dataset, single_candidate, objective, background, cfg, task_name)
        from RoboPhD.engines.autoresearch import run_autoresearch
        return run_autoresearch(evaluator, dataset, single_candidate, objective, background, cfg, task_name)

    # --- RoboPhD Elo engine (default) ---
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
            "extra_read_paths": cfg.extra_read_paths,
            "session_tools": cfg.session_tools,
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
        if not seed_agents:
            raise ValueError("seed_agents is required for fresh runs")

        resolved_seeds = _resolve_seed_agents(seed_agents)
        # Artifacts are keyed by their path within the agent directory, and
        # _resolve_seed_agents has pinned every seed to the same set.
        file_mapping = {key: key for key in next(iter(resolved_seeds.values()))}

        # One container per run, so the seed directories inside it can carry
        # the caller's names as-is: uniqueness across runs comes from the
        # container, uniqueness within it from the dict keys. (Pointing
        # agents_directory at the shared parent instead would force every seed
        # name to be globally unique across every run and task.)
        seeds_root = Path(
            tempfile.mkdtemp(
                dir=_ensure_seeds_dir(run_dir), prefix="seeds_"
            )
        )
        for name, candidate in resolved_seeds.items():
            materialize_candidate(candidate, seeds_root / name, file_mapping, name=name)

        seed_agents_root = seeds_root
        seed_agent_names = list(resolved_seeds)
        logger.info(
            "Seeding %d agent(s): %s",
            len(seed_agent_names), ", ".join(seed_agent_names),
        )

        config_manager = ConfigManager()
        researcher_config = {
            "domain": "external",
            "meta_evolution_domain": task_name,
            "evolution_strategy": cfg.evolution_strategy,
            "evolution_model": cfg.evolution_model,
            "evolution_timeout": cfg.evolution_timeout,
            "eval_timeout": cfg.eval_timeout,
            "examples_per_iteration": cfg.examples_per_iteration,
            "meta_evolution_strategy": cfg.meta_evolution_strategy,
            "meta_evolution_model": cfg.meta_evolution_model,
            "meta_evolution_first_iteration": cfg.meta_evolution_first_iteration,
            "meta_evolution_cadence": cfg.meta_evolution_cadence,
            "agents_directory": str(seed_agents_root),
            "initial_agents": list(seed_agent_names),
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
            "extra_read_paths": cfg.extra_read_paths,
            "session_tools": cfg.session_tools,
        }

        # Persist objective/background in task_config so they survive
        # resume; task_config_extras rides along for task-specific values.
        researcher = ParallelAgentResearcher(
            config_manager=config_manager,
            num_iterations=cfg.num_iterations,
            random_seed=cfg.random_seed,
            runtime_config=runtime_config,
            task_config=_merge_task_config_extras(
                {
                    "file_mapping": file_mapping,
                    "objective": objective,
                    "background": background,
                },
                cfg.task_config_extras,
            ),
        )
        initial_agents = list(seed_agent_names)

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
        candidate: Text artifact(s) to evaluate — one agent's worth, i.e. a
            single ``seed_agents`` value in its dict form.
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
        aggregate_explanation=result["test_results"].get("aggregate_explanation", ""),
    )


def eval_run(
    evaluator: Callable,
    dataset: List[Dict],
    experiment_dir: Union[str, Path],
    config: Optional[RoboPhDEvalConfig] = None,
) -> EvalResult:
    """Evaluate the best agent from a completed optimization run.

    Engine-agnostic: GEPA and Autoresearch write ``best_candidate.json`` and
    ``best_agent/`` at the run root — those are preferred when present. The
    RoboPhD Elo path falls through to the highest-Elo agent in the checkpoint.
    Typical use: test-set evaluation after ``optimize_anything()`` finishes.

    Args:
        evaluator: Same evaluator used during optimization.
        dataset: Test examples to evaluate on.
        experiment_dir: Path to the experiment directory.
        config: Evaluation configuration. If None, uses ``RoboPhDEvalConfig()`` defaults.

    Returns:
        EvalResult with mean_score, per_example_scores, and diagnostics.
    """
    from RoboPhD.runner_utils import load_best_candidate

    experiment_dir = Path(experiment_dir)
    best_candidate, label = load_best_candidate(experiment_dir)
    logger.info(f"Evaluating {label} from {experiment_dir.name}")
    return eval_candidate(evaluator, dataset, best_candidate, config)
