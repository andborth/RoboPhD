"""
Shared utilities for runner scripts (run_gepa.py, run_robophd.py, run_autoresearch.py).
"""

import argparse
import json
import logging
import os
import random
import threading
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from RoboPhD.debug_logging import maybe_debug_log
from RoboPhD.config import SUPPORTED_MODELS
from RoboPhD.eval_utils import run_parallel_eval


def to_litellm_model(name: str) -> str:
    """Translate RoboPhD model shorthand (e.g. 'opus-4.6') to litellm name ('claude-opus-4-6')."""
    if name in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[name]["name"]
    return name


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
    """Find the best agent by ELO from a checkpoint.json.

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
        f"(ELO: {best_perf['elo']:.0f}, "
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


def find_last_winner(run_dir: Path) -> Tuple[str, Path, bool]:
    """Find the agent that won the last completed iteration.

    For king-of-the-hill runs (oldest_agent_wins_ties=true), the last-round
    winner may differ from the ELO leader.

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
                f"(older checkpoint format); falling back to ELO leader"
            )
        else:
            log.warning(
                f"No agent won iteration {last_iter}; falling back to ELO leader"
            )
        return elo_leader, _resolve_agent_dir(run_dir, elo_leader, agent_pool, perf_records, log), True

    # Tiebreak by ELO
    winner_id = max(last_winners, key=lambda k: perf_records[k]["elo"])
    is_elo_leader = winner_id == elo_leader
    winner_perf = perf_records[winner_id]

    log.info(
        f"Last-round winner (iteration {last_iter}): {winner_id} "
        f"(ELO: {winner_perf['elo']:.0f}, "
        f"score: {winner_perf['mean_score']:.3f}, "
        f"tests: {winner_perf['test_count']}"
        f"{', also ELO leader' if is_elo_leader else ''})"
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
