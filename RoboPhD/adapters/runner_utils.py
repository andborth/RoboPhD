"""
Shared utilities for runner scripts (run_gepa.py, run_robophd.py, run_autoresearch.py).
"""

import argparse
import json
import logging
import random
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from RoboPhD.adapters.debug_logging import maybe_debug_log
from RoboPhD.config import SUPPORTED_MODELS


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

    Crashes if both val_ratio and val_size are explicitly provided,
    or if the resulting val set is larger than the training set.

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
        log.error("Cannot specify both val_ratio and val_size")
        sys.exit(1)

    rng = random.Random(seed)
    shuffled = list(dataset)
    rng.shuffle(shuffled)

    if val_ratio is not None:
        split_idx = max(1, int(len(shuffled) * (1 - val_ratio)))
    else:
        if val_size < 1:
            log.error(f"val_size ({val_size}) must be >= 1")
            sys.exit(1)
        if val_size >= len(shuffled):
            log.error(f"val_size ({val_size}) must be less than dataset size ({len(shuffled)})")
            sys.exit(1)
        split_idx = len(shuffled) - val_size

    trainset = shuffled[:split_idx]
    valset = shuffled[split_idx:]

    if len(valset) > len(trainset):
        log.error(
            f"Validation set ({len(valset)}) is larger than training set ({len(trainset)}). "
            f"This is likely not intended. Use --engine-config '{{\"val_size\": N}}' to set a smaller val size."
        )
        sys.exit(1)

    log.info(f"Training set: {len(trainset)}, Validation set: {len(valset)}")
    return trainset, valset
