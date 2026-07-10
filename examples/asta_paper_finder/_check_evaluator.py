"""Evaluator sanity gate with synthetic candidates (no LLM spend).

Three checks:

  1. Gold-ID leak scan: no numeric token in the evolution-facing
     artifacts (background.md, objective.md, seeds/baseline/agent.py)
     may be a gold corpus_id in EITHER split. A real ID in those files
     hands ground truth to the evolution AI (this actually happened —
     the original background.md's worked example used a gold ID present
     in both splits).
  2. A candidate that returns exactly the gold corpus_id for a real
     `specific_f1` sample must score > 0 (the scorer chain — tools
     setup, output parsing, F1 — works).
  3. A candidate that returns an empty result list must score 0.

Runs in-process (subprocess_isolation=False) with the test-mode
evaluator so no cost penalty machinery interferes. specific_f1 is the
right sample class here because its scoring is deterministic — no
GPT-4o judge, so the gate is free and non-flaky.

Requires all three provider keys (the evaluator preflights them) plus
HF_ACCESS_TOKEN and ASTA_TOOL_KEY (or a search-fallback key).
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

from evaluator import PaperFinderEvaluator, load_paper_finder


CANDIDATE_TEMPLATE = '''
import json
from inspect_ai.solver import Generate, TaskState, solver

@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.output.completion = json.dumps({{
            "output": {{"query_id": state.sample_id, "results": {results}}}
        }})
        return state
    return solve
'''


def _gold_ids_of(samples) -> set[str]:
    """All gold corpus_ids referenced by a split's scorer criteria."""
    out: set[str] = set()
    for s in samples:
        try:
            crit = json.loads(str(s.target))
        except (json.JSONDecodeError, TypeError):
            continue
        for key in ("corpus_ids", "known_to_be_good", "known_to_be_bad"):
            out |= {str(x) for x in (crit.get(key) or [])}
    return out


def _check_gold_id_leaks(samples) -> list[str]:
    """No numeric token in the evolution-facing artifacts may be a gold
    corpus_id in either split. Enforced here (not just documented) so a
    future doc edit can't quietly hand ground truth to the evolution AI."""
    gold = _gold_ids_of(samples) | _gold_ids_of(load_paper_finder("test"))
    failures = []
    for rel in ("background.md", "objective.md", "seeds/baseline/agent.py"):
        text = (HERE / rel).read_text()
        leaked = sorted(set(re.findall(r"\b\d{6,10}\b", text)) & gold)
        if leaked:
            failures.append(
                f"{rel} contains gold corpus_id(s) {leaked} — replace with "
                f"synthetic IDs; this leaks ground truth to evolution"
            )
    return failures


def main() -> int:
    samples = load_paper_finder("validation")

    failures = _check_gold_id_leaks(samples)
    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("gold-ID leak scan: clean")

    specific = next(
        s for s in samples if str(s.id).startswith("specific")
    )
    gold = json.loads(str(specific.target))
    gold_ids = gold.get("corpus_ids") or []
    if not gold_ids:
        print(f"FATAL: sample {specific.id} target has no corpus_ids: {gold}")
        return 1
    print(f"sample={specific.id} gold corpus_ids={gold_ids}")

    evaluator = PaperFinderEvaluator(
        subprocess_isolation=False,
        apply_cost_penalty=False,
    )
    example = specific.model_dump()

    gold_results = [
        {"paper_id": str(cid), "markdown_evidence": "gold candidate"}
        for cid in gold_ids
    ]
    gold_candidate = {
        "agent.py": CANDIDATE_TEMPLATE.format(results=json.dumps(gold_results))
    }
    empty_candidate = {
        "agent.py": CANDIDATE_TEMPLATE.format(results="[]")
    }

    failures = []

    score, diag = evaluator(gold_candidate, example)
    print(f"gold-returning candidate: score={score} "
          f"agent_cost=${diag.get('agent_cost_usd', 0):.4f} "
          f"judge_cost=${diag.get('other_cost_usd', 0):.4f} "
          f"error={diag.get('error.md')!r}")
    if score <= 0:
        failures.append(f"gold-returning candidate scored {score}, expected > 0")
    if diag.get("error.md"):
        failures.append(f"gold-returning candidate errored: {diag['error.md'][:300]}")
    if diag.get("other_cost_usd", 0):
        failures.append("specific_f1 sample billed judge cost — split is misrouting")

    score, diag = evaluator(empty_candidate, example)
    print(f"empty candidate: score={score} error={diag.get('error.md')!r}")
    if score != 0:
        failures.append(f"empty candidate scored {score}, expected 0")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
