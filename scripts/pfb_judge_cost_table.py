#!/usr/bin/env python3
"""Judge-cost table for official PaperFindingBench runs, from their .eval logs.

Source for the cost figures quoted in asta_paper_finder_submit.py and in
examples/asta_paper_finder/rejudge_test.py, so they are recomputed rather than
hand-entered. For each submissions/asta_paper_finder/<name>/logs/full_test/
log, over the semantic queries:

  chars/paper  mean markdown_evidence length of the submitted papers
  judged       GPT-4o judge calls (papers already known Perfect or cached
               are scored without one, so judged can be < submitted)
  $/call       judge cost per call at list price ($2.50/M in, $10/M out)

    python scripts/pfb_judge_cost_table.py
"""
from __future__ import annotations

import glob
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "submissions" / "asta_paper_finder"
JUDGE = "gpt-4o"
PRICE_IN, PRICE_OUT = 2.5e-6, 1e-5


def run_stats(eval_path: str) -> dict:
    z = zipfile.ZipFile(eval_path)
    chars = papers = calls = tin = tout = 0
    for name in z.namelist():
        if not name.startswith("samples/semantic"):
            continue
        s = json.loads(z.read(name))
        try:
            results = json.loads(s["output"]["completion"])["output"]["results"]
        except (KeyError, TypeError, ValueError):
            results = []
        papers += len(results)
        chars += sum(len(r.get("markdown_evidence") or "") for r in results)
        for e in s.get("events", []):
            if e.get("event") == "model" and JUDGE in (e.get("model") or ""):
                u = (e.get("output") or {}).get("usage") or {}
                calls += 1
                tin += u.get("input_tokens", 0)
                tout += u.get("output_tokens", 0)
    cost = tin * PRICE_IN + tout * PRICE_OUT
    return {"papers": papers, "chars": chars / max(papers, 1), "calls": calls,
            "cost": cost, "in": tin / max(calls, 1), "out": tout / max(calls, 1)}


def main() -> int:
    logs = sorted(glob.glob(str(ROOT / "*" / "logs" / "full_test" / "*.eval")))
    if not logs:
        print(f"no official logs under {ROOT}", file=sys.stderr)
        return 1
    print(f"{'run':30} {'chars/paper':>11} {'submitted':>9} {'judged':>7} "
          f"{'judge $':>8} {'$/call':>8} {'in/call':>7} {'out/call':>8}")
    for path in logs:
        r = run_stats(path)
        name = Path(path).parents[2].name
        print(f"{name:30} {r['chars']:11.0f} {r['papers']:9d} {r['calls']:7d} "
              f"{r['cost']:8.2f} {r['cost'] / max(r['calls'], 1):8.5f} "
              f"{r['in']:7.0f} {r['out']:8.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
