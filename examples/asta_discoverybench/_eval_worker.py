#!/usr/bin/env python3
"""Subprocess worker for DiscoveryBenchEvaluator.

Process-isolation strategy: inspect-ai's `inspect.eval()` raises if two
calls are in flight in the same Python process. To get real parallelism
across RoboPhD's worker threads, the evaluator spawns one of these
workers per evaluation. Each subprocess imports inspect-ai/astabench
freshly (~5–10s) but has its own process-global state, so concurrent
workers don't fight for the eval_async singleton.

Protocol:
  python _eval_worker.py <input.json> <output.json>

Input JSON shape:
  {"candidate": {"agent.py": "..."},
   "example": {<Sample.model_dump()>},
   "apply_cost_penalty": true,
   "min_cost_threshold": 0.01,
   "cost_penalty_saturation": 1.0}

Output JSON shape:
  {"score": <float>, "diagnostics": <dict>}

Worker exit codes:
  0 = success (output.json populated)
  != 0 = failure (stderr contains the traceback; parent surfaces it)
"""

import json
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <input.json> <output.json>", file=sys.stderr)
        return 2

    input_path, output_path = sys.argv[1], sys.argv[2]

    try:
        with open(input_path) as f:
            params = json.load(f)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 3

    try:
        # Import after argv parsing so usage errors don't pay the import cost.
        from evaluator import DiscoveryBenchEvaluator

        # Use the evaluator's defaults when the parent didn't provide a key
        # (compatibility with older parents that haven't been updated to
        # send min_cost_threshold / cost_penalty_saturation).
        evaluator_kwargs = {
            # Parent already pre-flighted Docker; don't re-check per worker.
            "skip_docker_check": True,
            # We ARE the subprocess — don't recurse.
            "subprocess_isolation": False,
            "apply_cost_penalty": params.get("apply_cost_penalty", True),
        }
        if "min_cost_threshold" in params:
            evaluator_kwargs["min_cost_threshold"] = params["min_cost_threshold"]
        if "cost_penalty_saturation" in params:
            evaluator_kwargs["cost_penalty_saturation"] = params["cost_penalty_saturation"]
        evaluator = DiscoveryBenchEvaluator(**evaluator_kwargs)
        score, diagnostics = evaluator.evaluate(params["candidate"], params["example"])
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 4

    try:
        with open(output_path, "w") as f:
            json.dump({"score": score, "diagnostics": diagnostics}, f, default=str)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 5

    return 0


if __name__ == "__main__":
    sys.exit(main())
