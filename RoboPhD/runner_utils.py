"""
Shared utilities for runner scripts (run_gepa.py, run_robophd.py, run_autoresearch.py).
"""

import argparse
import dataclasses
import json
import logging
import os
import random
import threading
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from RoboPhD.debug_logging import maybe_debug_log
from RoboPhD.config import SUPPORTED_MODELS
from RoboPhD.candidate_utils import extract_candidate
from RoboPhD.eval_utils import run_parallel_eval


def to_litellm_model(name: str) -> str:
    """Translate RoboPhD model shorthand (e.g. 'opus-4.6') to litellm name ('claude-opus-4-6')."""
    if name in SUPPORTED_MODELS:
        full_name = SUPPORTED_MODELS[name]["name"]
        _ensure_litellm_pricing(name, full_name)
        return full_name
    return name


def register_supported_model_pricing() -> None:
    """Register litellm pricing for every SUPPORTED_MODELS entry missing
    from litellm's bundled registry (e.g. claude-fable-5 on litellm 1.82).

    For callers that price usage via litellm.cost_per_token without going
    through to_litellm_model (e.g. the asta_ds1000 evaluator's
    _estimate_cost) — keeps SUPPORTED_MODELS the single source of rates.
    """
    for shorthand, spec in SUPPORTED_MODELS.items():
        _ensure_litellm_pricing(shorthand, spec["name"])


def _ensure_litellm_pricing(shorthand: str, full_name: str) -> None:
    """Register pricing for models newer than litellm's bundled registry.

    For models missing from its pricing DB (e.g. claude-fable-5 on litellm
    1.82), litellm raises "LLM Provider NOT provided" at completion() and
    "model isn't mapped yet" at completion_cost(). Registering from
    SUPPORTED_MODELS decouples new-model adoption from litellm releases.
    """
    import litellm  # lazy: litellm import is slow; only litellm callers pay it

    if full_name in litellm.model_cost:
        return
    pricing = SUPPORTED_MODELS[shorthand]["pricing"]
    litellm.register_model({
        full_name: {
            "litellm_provider": "anthropic",  # all SUPPORTED_MODELS are Claude
            "mode": "chat",
            "input_cost_per_token": pricing["input"] / 1e6,
            "output_cost_per_token": pricing["output"] / 1e6,
            "cache_creation_input_token_cost": pricing["cache_write"] / 1e6,
            "cache_read_input_token_cost": pricing["cache_read"] / 1e6,
        }
    })


class CostTrackingLM:
    """Wraps litellm.completion to track token usage and cost for GEPA reflection calls.

    Accepts an explicit api_key so callers don't need to set ANTHROPIC_API_KEY
    in the environment (which would cause Claude Code CLI subprocesses to bill
    through the API account instead of Claude Max).
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        debug_log_probability: float = 0.0,
        debug_log_dir: Path | str | None = None,
    ):
        self.model_name = model_name
        self._api_key = api_key
        self.debug_log_probability = debug_log_probability
        self.debug_log_dir = Path(debug_log_dir) if debug_log_dir else None
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0
        self._lock = threading.Lock()

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        import litellm

        if isinstance(prompt, str):
            messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        kwargs: dict[str, Any] = {"model": self.model_name, "messages": messages}
        if self._api_key:
            kwargs["api_key"] = self._api_key

        completion = litellm.completion(**kwargs)

        usage = getattr(completion, "usage", None)
        cost = litellm.completion_cost(completion_response=completion)

        with self._lock:
            self.call_count += 1
            self.total_cost += cost
            if usage:
                self.total_input_tokens += getattr(usage, "prompt_tokens", 0)
                self.total_output_tokens += getattr(usage, "completion_tokens", 0)
            call_count = self.call_count

        response_text = completion.choices[0].message.content

        maybe_debug_log(
            debug_log_probability=self.debug_log_probability,
            debug_log_dir=self.debug_log_dir,
            call_type="reflection",
            model=self.model_name,
            messages=messages,
            response_text=response_text or "",
            metadata={
                "cost_usd": cost,
                "call_count": call_count,
                "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                "output_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
            },
        )

        return response_text


def apply_engine_config(cfg: Any, engine_config: Any) -> Any:
    """Overlay --engine-config values onto an engine config dataclass
    (e.g. GEPAConfig, AutoresearchConfig).

    Accepts the raw JSON string from the CLI flag, an already-parsed
    dict, or None/empty (no-op). Every key must be a field of the
    dataclass; unknown keys raise ValueError listing the valid names
    rather than being silently dropped. Applied after construction, so
    these values win over flag-derived constructor arguments — the JSON
    overlay is the escape hatch, same precedence as the RoboPhD
    engine's engine_overrides.
    """
    if not engine_config:
        return cfg
    overrides = (
        json.loads(engine_config)
        if isinstance(engine_config, str)
        else engine_config
    )
    valid = {f.name for f in dataclasses.fields(cfg)}
    unknown = sorted(set(overrides) - valid)
    if unknown:
        raise ValueError(
            f"Unknown --engine-config key(s) for {type(cfg).__name__}: "
            f"{', '.join(unknown)}. Valid keys: {', '.join(sorted(valid))}"
        )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def fmt_val(val: Any) -> str:
    """Format a config value for display."""
    if isinstance(val, str):
        return f'"{val}"'
    if val is None:
        return "null"
    if isinstance(val, dict) and not val:
        return "{}"
    if isinstance(val, list) and not val:
        return "[]"
    return str(val)


def print_task_params(task) -> None:
    """Print task-specific config parameters."""
    print(f"\nTask: {task.name} — {task.description}")
    print(f"  Default seed agent: {task.default_seed_agent}")
    print(f"  File mapping: {task.file_mapping}")

    if task.config_defaults:
        print(f"\n  Task defaults (overridable via --task-config):")
        for k, v in sorted(task.config_defaults.items()):
            print(f"    {k}: {fmt_val(v)}")

    if task.test_overrides:
        print(f"\n  Test overrides (applied during test evaluation):")
        for k, v in sorted(task.test_overrides.items()):
            print(f"    {k}: {fmt_val(v)}")
    print()


def parse_config_arg(value: str) -> dict:
    """Parse a config argument: either a JSON file path or an inline JSON string."""
    if value is None:
        return {}
    stripped = value.strip()
    # If it looks like JSON (starts with {), parse directly
    if stripped.startswith("{"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            raise argparse.ArgumentTypeError(
                f"Config must be a path to a JSON file or an inline JSON string, got: {value!r}"
            )
    # Otherwise treat as file path
    path = Path(value)
    if path.exists() and path.is_file():
        with open(path) as f:
            return json.load(f)
    raise argparse.ArgumentTypeError(
        f"Config must be a path to a JSON file or an inline JSON string, got: {value!r}"
    )


def split_train_val(
    dataset: List,
    val_size: int,
    val_ratio: float | None,
    user_provided_keys: Set[str],
    seed: int = 0,
    logger: logging.Logger | None = None,
) -> Tuple[List, List]:
    """Split a dataset into train and val sets.

    Priority:
        1. If val_ratio explicitly provided by user → use it
        2. Otherwise → use val_size (default 200)

    Raises:
        ValueError: If both val_ratio and val_size are explicitly provided,
            val_size is invalid, or val set is larger than train set.

    Args:
        dataset: Full dataset to split.
        val_size: Validation set size (default 200).
        val_ratio: Fraction held out for validation (None unless user set it).
        user_provided_keys: Set of config keys the user explicitly provided
            (from --engine-config and --task-config).
        seed: Random seed for shuffling.
        logger: Optional logger.

    Returns:
        (trainset, valset) tuple.
    """
    log = logger or logging.getLogger(__name__)

    if "val_ratio" in user_provided_keys and "val_size" in user_provided_keys:
        raise ValueError("Cannot specify both val_ratio and val_size")

    rng = random.Random(seed)
    shuffled = list(dataset)
    rng.shuffle(shuffled)

    if val_ratio is not None:
        split_idx = max(1, int(len(shuffled) * (1 - val_ratio)))
    else:
        if val_size < 1:
            raise ValueError(f"val_size ({val_size}) must be >= 1")
        if val_size >= len(shuffled):
            raise ValueError(f"val_size ({val_size}) must be less than dataset size ({len(shuffled)})")
        split_idx = len(shuffled) - val_size

    trainset = shuffled[:split_idx]
    valset = shuffled[split_idx:]

    if len(valset) > len(trainset):
        raise ValueError(
            f"Validation set ({len(valset)}) is larger than training set ({len(trainset)}). "
            f"This is likely not intended. Use --engine-config '{{\"val_size\": N}}' to set a smaller val size."
        )

    log.info(f"Training set: {len(trainset)}, Validation set: {len(valset)}")
    return trainset, valset


def find_best_agent(run_dir: Path) -> Tuple[str, Path]:
    """Find the best agent by Elo from a checkpoint.json.

    Returns (agent_name, agent_dir).
    """
    log = logging.getLogger(__name__)
    checkpoint_path = Path(run_dir) / "checkpoint.json"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint.json found in {run_dir}")

    with open(checkpoint_path) as f:
        ckpt = json.load(f)

    perf_records = ckpt.get("performance_records", {})
    agent_pool = ckpt.get("agent_pool", {})

    if not perf_records:
        raise ValueError(f"No performance records in {checkpoint_path}")

    best_id = max(perf_records, key=lambda k: perf_records[k]["elo"])
    best_perf = perf_records[best_id]

    log.info(
        f"Best agent: {best_id} "
        f"(Elo: {best_perf['elo']:.0f}, "
        f"score: {best_perf['mean_score']:.3f}, "
        f"tests: {best_perf['test_count']})"
    )

    agent_info = agent_pool.get(best_id)
    if not agent_info or "package_dir" not in agent_info:
        raise ValueError(f"Agent {best_id} not found in agent_pool or missing package_dir")

    agent_dir = Path(run_dir) / agent_info["package_dir"]
    if not agent_dir.exists():
        raise FileNotFoundError(f"Agent directory not found: {agent_dir}")

    return best_id, agent_dir


def find_named_agent(run_dir: Path, agent_name: str) -> Tuple[str, Path]:
    """Find a specific named agent from a run's agent_pool.

    Symmetric to find_best_agent but looks up by explicit name rather than
    Elo. Backs the examples' --eval-agent CLI surfaces so the user can
    baseline the seed, inspect a specific iteration's agent, or compare any
    two agents on the same held-out data.

    Returns (agent_name, agent_dir). Raises FileNotFoundError consistently
    for any lookup failure: missing checkpoint.json, missing agent_pool key
    (schema drift), unknown agent name, or missing agent directory on disk.
    For the unknown-name case the message includes the sorted list of
    available agent_pool keys so CLI callers can surface it directly without
    reformatting.
    """
    log = logging.getLogger(__name__)
    checkpoint_path = Path(run_dir) / "checkpoint.json"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint.json found in {run_dir}")

    with open(checkpoint_path) as f:
        ckpt = json.load(f)

    # Distinguish malformed/schema-drifted checkpoint from a bad agent name:
    # a missing agent_pool key is a checkpoint integrity issue, whereas an
    # empty or populated agent_pool that doesn't contain the name is a
    # legitimate "not found" path.
    if "agent_pool" not in ckpt:
        raise FileNotFoundError(
            f"Checkpoint at {checkpoint_path} has no agent_pool key (schema error)."
        )
    agent_pool = ckpt["agent_pool"]
    if agent_name not in agent_pool:
        available = sorted(agent_pool.keys())
        raise FileNotFoundError(
            f"Agent '{agent_name}' not found in agent_pool of {run_dir}.\n"
            f"Available ({len(available)}): {', '.join(available)}"
        )

    perf_records = ckpt.get("performance_records", {})
    agent_dir = _resolve_agent_dir(run_dir, agent_name, agent_pool, perf_records, log)
    return agent_name, agent_dir


def read_checkpoint_max_workers(resume_dir) -> "int | None":
    """Read max_workers from a resumed run's checkpoint.json, or None if
    absent / unparseable.

    Walks ``config_manager.iteration_configs`` to the highest iteration with an
    explicit ``max_workers`` value (skipping nulls, which mean "framework
    default"), recovering the value the original run actually used. Used by
    example main.py files so the EVAL paths (--eval-only / --eval-test-set,
    which take a RoboPhDEvalConfig directly and bypass ConfigManager) honor the
    resumed run's worker count. The training path doesn't need this — it routes
    max_workers through engine_overrides, where ConfigManager's delta
    inheritance carries it forward.
    """
    cp_path = Path(resume_dir) / "checkpoint.json"
    if not cp_path.is_file():
        return None
    try:
        cp = json.loads(cp_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    iter_configs = (cp.get("config_manager") or {}).get("iteration_configs") or {}
    for k in sorted(iter_configs.keys(),
                    key=lambda s: int(s) if s.isdigit() else -1, reverse=True):
        val = (iter_configs[k] or {}).get("max_workers")
        if val is not None:
            return int(val)
    return None


def read_task_config_extras(resume_dir, task_key: str,
                            legacy_sidecar: "str | None" = None) -> dict:
    """Read a task's run-immutable values from a resumed run.

    Primary source: checkpoint.json's ``task_config[task_key]`` — written
    by the framework every iteration when the run passes
    ``RoboPhDConfig.task_config_extras``. Fallback: a task-named legacy
    sidecar JSON at the experiment-dir root (runs from before the task
    adopted task_config_extras). Best-effort: missing or malformed
    sources read as ``{}`` rather than raising — the caller's
    ``resolve_run_immutable`` is where a missing value becomes an error.
    """
    resume_dir = Path(resume_dir)
    try:
        checkpoint = json.loads((resume_dir / "checkpoint.json").read_text())
        stored = (checkpoint.get("task_config") or {}).get(task_key)
        if isinstance(stored, dict):
            return stored
    except (FileNotFoundError, json.JSONDecodeError, OSError, AttributeError):
        pass
    if legacy_sidecar:
        try:
            data = json.loads((resume_dir / legacy_sidecar).read_text())
            if isinstance(data, dict):
                return data
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
    return {}


def resolve_run_immutable(cli_value, stored_value, default_value, flag: str, *,
                          on_resume: bool, fmt=str, missing_note: str = ""):
    """Resolve a per-run-immutable task knob (e.g. asta_ds1000's
    --cost-threshold, arc_agi_1's --num-train).

    Fresh run: CLI flag wins, else the task default.

    Resume + stored value present (from ``read_task_config_extras``):
      - CLI flag passed and disagrees: SystemExit (immutability).
      - CLI flag passed and matches: use stored (silent no-op).
      - No CLI flag: use stored.

    Resume + no stored value (run crashed before its first checkpoint,
    or predates the task's adoption of task_config_extras):
      - CLI flag passed: one-time bootstrap, return CLI value. Passing
        the resolved value back through task_config_extras persists it
        at the next completed iteration, locked thereafter.
      - No CLI flag: SystemExit — never a silent default fallback, which
        would change task behavior mid-run.

    ``missing_note`` is appended to the missing-value error for task-
    specific guidance (e.g. ds1000's both-flags-together requirement).
    The api.py resume merge independently backstops immutability at the
    framework level; this resolver is the friendlier flag-level guard.
    """
    if not on_resume:
        return cli_value if cli_value is not None else default_value
    if stored_value is not None:
        if cli_value is not None and cli_value != stored_value:
            raise SystemExit(
                f"--{flag} cannot be changed on --resume; the value is "
                f"locked for the lifetime of the run (stored value: "
                f"{fmt(stored_value)}). Start a new run to use a "
                f"different {flag}."
            )
        return stored_value
    if cli_value is not None:
        logging.getLogger(__name__).info(
            f"Resume: bootstrapping {flag}={fmt(cli_value)} into the "
            f"checkpoint's task_config (no prior stored value)."
        )
        return cli_value
    raise SystemExit(
        f"--resume failed: no stored {flag} in the checkpoint's "
        f"task_config (or legacy sidecar). This run predates "
        f"per-iteration persistence of the value and was interrupted "
        f"before completing. Pass --{flag} <value> to bootstrap it "
        f"(one-time — it persists at the next completed iteration, then "
        f"is locked). {missing_note}Or restart the run."
    )


def load_best_candidate(
    run_dir: Path,
    file_mapping: Dict[str, str] | None = None,
) -> Tuple[Dict[str, str], str]:
    """Load the best candidate from any engine's run directory.

    Engine-agnostic: GEPA and Autoresearch both write best_candidate.json
    and best_agent/ at the run root; RoboPhD writes checkpoint.json +
    agent_pool. Tries those in order and returns the first match.

    Returns (candidate_dict, label) where label is one of:
      - "best_candidate" — loaded from best_candidate.json
      - "best_agent"     — loaded from best_agent/<files...>
      - <agent_name>     — loaded from the highest-Elo checkpoint agent

    Callers use `label` for user-facing identification (log lines, output
    filename suffixes); `eval_run` ignores it, while the test-eval surfaces
    use it to distinguish output files.

    file_mapping controls how non-flat candidates are read from disk:
      - best_candidate.json: ignored (the file IS the candidate dict)
      - best_agent/: used by extract_candidate when provided; if None,
        auto-falls-back only when the directory contains exactly a single
        agent.py (the single-file tasks). A multi-file best_agent/ with no
        explicit mapping raises rather than silently dropping files.
      - checkpoint (Elo fallthrough): used by extract_candidate; if None,
        read from checkpoint.task_config.

    Engines writing best_candidate.json must write a flat file_mapping
    dict (keys = component names, values = text). Wrapped shapes like
    {"candidate": {...}, "score": ...} will be rejected here rather than
    failing deep in the evaluator.
    """
    run_dir = Path(run_dir)

    best_json = run_dir / "best_candidate.json"
    if best_json.exists():
        with open(best_json) as f:
            candidate = json.load(f)
        if not isinstance(candidate, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in candidate.items()
        ):
            raise ValueError(
                f"{best_json} is not a flat {{str: str}} candidate dict. "
                f"Engines writing best_candidate.json must write the candidate directly, "
                f"not wrapped in an outer envelope."
            )
        return candidate, "best_candidate"

    best_dir = run_dir / "best_agent"
    if best_dir.exists():
        if file_mapping is not None:
            return extract_candidate(best_dir, file_mapping), "best_agent"
        files = sorted(p.name for p in best_dir.iterdir() if p.is_file())
        if files == ["agent.py"]:
            return {"agent.py": (best_dir / "agent.py").read_text()}, "best_agent"
        raise ValueError(
            f"{best_dir} contains {files}; cannot infer candidate shape without "
            f"an explicit file_mapping. Pass file_mapping to load_best_candidate."
        )

    # RoboPhD Elo fallthrough: checkpoint.json + agent_pool.
    agent_name, agent_dir = find_best_agent(run_dir)
    mapping = file_mapping
    if mapping is None:
        from RoboPhD.researcher import ParallelAgentResearcher
        checkpoint = ParallelAgentResearcher.load_checkpoint(run_dir)
        mapping = checkpoint.get("task_config", {}).get("file_mapping")
        if not mapping:
            raise ValueError(
                f"Checkpoint at {run_dir} missing file_mapping in task_config; "
                f"pass file_mapping explicitly to load_best_candidate."
            )
    return extract_candidate(agent_dir, mapping), agent_name


def find_last_winner(run_dir: Path) -> Tuple[str, Path, bool]:
    """Find the agent that won the last completed iteration.

    For king-of-the-hill runs (oldest_agent_wins_ties=true), the last-round
    winner may differ from the Elo leader.

    Returns (agent_name, agent_dir, is_also_elo_leader).
    """
    log = logging.getLogger(__name__)
    checkpoint_path = Path(run_dir) / "checkpoint.json"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint.json found in {run_dir}")

    with open(checkpoint_path) as f:
        ckpt = json.load(f)

    perf_records = ckpt.get("performance_records", {})
    agent_pool = ckpt.get("agent_pool", {})
    last_iter = ckpt.get("last_completed_iteration", 0)

    if not perf_records:
        raise ValueError(f"No performance records in {checkpoint_path}")

    elo_leader = max(perf_records, key=lambda k: perf_records[k]["elo"])

    # Find agents whose last win was at the final iteration
    last_winners = [
        name for name, rec in perf_records.items()
        if rec.get("last_win_iteration") == last_iter
    ]

    if not last_winners:
        has_field = any(
            "last_win_iteration" in rec for rec in perf_records.values()
        )
        if not has_field:
            log.warning(
                f"No last_win_iteration field in performance records "
                f"(older checkpoint format); falling back to Elo leader"
            )
        else:
            log.warning(
                f"No agent won iteration {last_iter}; falling back to Elo leader"
            )
        return elo_leader, _resolve_agent_dir(run_dir, elo_leader, agent_pool, perf_records, log), True

    # Tiebreak by Elo
    winner_id = max(last_winners, key=lambda k: perf_records[k]["elo"])
    is_elo_leader = winner_id == elo_leader
    winner_perf = perf_records[winner_id]

    log.info(
        f"Last-round winner (iteration {last_iter}): {winner_id} "
        f"(Elo: {winner_perf['elo']:.0f}, "
        f"score: {winner_perf['mean_score']:.3f}, "
        f"tests: {winner_perf['test_count']}"
        f"{', also Elo leader' if is_elo_leader else ''})"
    )

    return winner_id, _resolve_agent_dir(run_dir, winner_id, agent_pool, perf_records, log), is_elo_leader


def _resolve_agent_dir(run_dir: Path, agent_id: str, agent_pool: dict,
                       perf_records: dict, log) -> Path:
    """Resolve an agent's directory from the agent pool."""
    agent_info = agent_pool.get(agent_id)
    if not agent_info or "package_dir" not in agent_info:
        raise ValueError(f"Agent {agent_id} not found in agent_pool or missing package_dir")

    agent_dir = Path(run_dir) / agent_info["package_dir"]
    if not agent_dir.exists():
        raise FileNotFoundError(f"Agent directory not found: {agent_dir}")

    return agent_dir


def run_test_eval(
    candidate: Dict[str, str],
    task,
    config: Dict[str, Any],
    output_dir: Path,
    max_workers: int | None = None,
    metadata: Dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
    output_filename: str = "test_results.json",
) -> Dict[str, Any]:
    """Evaluate a candidate on the held-out test set.

    Shared by run_robophd.py, run_gepa.py, run_autoresearch.py, and eval_test_set.py.

    Args:
        candidate: Dict of agent artifacts (from extract_candidate).
        task: TaskDefinition with test_overrides, dataset_builder, evaluator_factory.
        config: Merged config dict (training config — test_overrides applied internally).
        output_dir: Where to write the results file and test_work/.
        max_workers: Override for parallel workers.
        metadata: Optional extra fields to include in the results file
            (e.g., agent_name, agent_dir, task).
        logger: Optional logger.
        output_filename: Name of the results file (default: test_results.json).

    Returns:
        dict with mean_test_score, total_test_score, total_test_problems, test_eval_cost_usd,
        plus any metadata fields. Also writes the results file to output_dir.
    """
    log = logger or logging.getLogger(__name__)

    test_config = {**config, **task.test_overrides}
    test_examples = task.dataset_builder(test_config)
    test_repeats = test_config.get("test_repeats", 1)
    test_examples = test_examples * test_repeats
    log.info(
        f"Test set: {len(test_examples)} problems "
        f"({len(test_examples) // test_repeats} unique × {test_repeats})"
    )

    test_config["work_dir"] = str(Path(output_dir) / "test_work")
    test_evaluator = task.evaluator_factory(test_config)

    test_workers = (
        max_workers
        or test_config.get("max_test_workers")
        or max(1, min(32, (os.cpu_count() or 4) + 4) // 2)
    )
    eval_timeout = test_config.get("eval_timeout", 300)
    log.info(f"Test evaluation: {len(test_examples)} problems, {test_workers} workers")

    result = run_parallel_eval(
        test_evaluator, candidate, test_examples,
        max_workers=test_workers, eval_timeout=eval_timeout,
    )

    test_eval_cost = getattr(test_evaluator, "total_eval_cost", 0.0)
    test_results = {**result["test_results"], "test_eval_cost_usd": test_eval_cost}
    if metadata:
        test_results.update(metadata)

    output_path = Path(output_dir) / output_filename
    with open(output_path, "w") as f:
        json.dump(test_results, f, indent=2)
    log.info(f"Test results saved to {output_path}")

    return test_results
