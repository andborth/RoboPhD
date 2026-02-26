"""
Shared utilities for runner scripts (run_gepa.py, run_robophd.py).
"""

import argparse
import json
import threading
from pathlib import Path
from typing import Any

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

    def __init__(self, model_name: str, api_key: str | None = None):
        self.model_name = model_name
        self._api_key = api_key
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

        return completion.choices[0].message.content


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
        print(f"\n  Task defaults (overridable via --config):")
        for k, v in sorted(task.config_defaults.items()):
            print(f"    {k}: {fmt_val(v)}")
    print()


def parse_config_arg(value: str) -> dict:
    """Parse a config argument: either a JSON file path or an inline JSON string."""
    if value is None:
        return {}
    path = Path(value)
    if path.exists() and path.is_file():
        with open(path) as f:
            return json.load(f)
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError(
            f"Config must be a path to a JSON file or an inline JSON string, got: {value!r}"
        )
