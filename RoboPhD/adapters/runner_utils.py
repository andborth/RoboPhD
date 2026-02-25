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
    """Wraps litellm.completion to track token usage and cost for GEPA reflection calls."""

    def __init__(self, model_name: str):
        self.model_name = model_name
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

        completion = litellm.completion(model=self.model_name, messages=messages)

        usage = getattr(completion, "usage", None)
        cost = litellm.completion_cost(completion_response=completion)

        with self._lock:
            self.call_count += 1
            self.total_cost += cost
            if usage:
                self.total_input_tokens += getattr(usage, "prompt_tokens", 0)
                self.total_output_tokens += getattr(usage, "completion_tokens", 0)

        return completion.choices[0].message.content


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
