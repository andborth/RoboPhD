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
   "model": "openai/gpt-5-mini",
   "cost_budget": 0.10}

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

        evaluator = DiscoveryBenchEvaluator(
            model=params["model"],
            cost_budget=params["cost_budget"],
            # Parent already pre-flighted Docker; don't re-check per worker.
            skip_docker_check=True,
            # We ARE the subprocess — don't recurse.
            subprocess_isolation=False,
            # Default True for missing-key compatibility with older parents.
            apply_cost_penalty=params.get("apply_cost_penalty", True),
        )
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
