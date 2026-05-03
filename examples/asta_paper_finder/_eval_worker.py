#!/usr/bin/env python3
"""Subprocess worker for PaperFinderEvaluator.

See examples/asta_discoverybench/_eval_worker.py for the rationale.
Same protocol — JSON in, JSON out — adapted to PaperFinderEvaluator's
constructor signature.
"""

import json
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
        from evaluator import PaperFinderEvaluator

        evaluator = PaperFinderEvaluator(
            model=params["model"],
            tool_source=params.get("tool_source"),
            subprocess_isolation=False,
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
