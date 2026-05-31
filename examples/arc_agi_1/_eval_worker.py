#!/usr/bin/env python3
"""Subprocess worker for ArcAGI1Evaluator.

Runs one ARC evaluation in a fresh, memory-capped child process so a
pathological evolved agent can't take down the whole run:

- A memory bomb (unbounded allocation) hits the RSS watchdog / RLIMIT_AS and
  this child dies with exit code 42, scored 0 — the parent run survives.
- An agent that arms signal.alarm() works correctly here because it runs on
  this child's MAIN thread (in the parent it ran in a ThreadPoolExecutor
  worker, where signal.signal() can't install a handler, so a stray SIGALRM
  terminated the whole process). A stray signal now only kills this child.

Protocol:  python _eval_worker.py <input.json> <output.json>

Input JSON:   {"candidate": {"agent.py": "..."}, "example": {...},
               "evaluator_config": {<ArcAGI1Evaluator constructor kwargs>}}
Output JSON:  {"score": <float>, "diagnostics": <dict>}

The memory ceiling is read from the ROBOPHD_AGENT_MEMORY_BYTES env var set by
the parent's run_evaluation_in_subprocess.
"""

import json
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Mirror main.py's sys.path setup so `RoboPhD.*` (repo root), `evaluator`, and
# the vendored `utils` (example dir) all import in this fresh interpreter.
for p in (str(HERE.parent.parent), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)


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
        from RoboPhD.eval_utils import apply_agent_memory_cap
        from evaluator import ArcAGI1Evaluator

        # Cap this child's memory before running agent code. The watchdog
        # daemon thread runs in the background so the agent keeps the main
        # thread (needed for its own signal.signal(SIGALRM) to install).
        mem_bytes = int(os.environ.get("ROBOPHD_AGENT_MEMORY_BYTES", "0"))
        apply_agent_memory_cap(mem_bytes)

        evaluator = ArcAGI1Evaluator(
            # We ARE the isolated child — run in-process, don't recurse.
            agent_subprocess_isolation=False,
            **params["evaluator_config"],
        )
        score, diagnostics = evaluator._evaluate_inprocess(
            params["candidate"], params["example"]
        )
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
