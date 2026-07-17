"""Evaluator sanity gate with synthetic candidates (no LLM spend).

Four checks:

  1. Gold-ID leak scan: no numeric token in the evolution-facing
     artifacts (background.md, objective.md, seeds/baseline/agent.py)
     may be a gold corpus_id in EITHER split. A real ID in those files
     hands ground truth to the evolution AI (this actually happened —
     the original background.md's worked example used a gold ID present
     in both splits).
  2. A candidate that returns exactly the gold corpus_id for a real
     `specific_f1` sample must score > 0 (the scorer chain — tools
     setup, output parsing, F1 — works), and its diagnostics must
     include a live `score_calculation.md` (the diagnostic reads the
     scorer's Score.metadata by key name and silently degrades to
     absent if astabench renames a component key — unit tests fabricate
     that metadata, so only a live run catches the drift).
  3. A candidate that returns an empty result list must score 0.
  4. A `semantic_f1` candidate submitting only `known_to_be_good`
     papers (pre-seeded Perfect — the scorer makes zero judge calls)
     must produce a `score_calculation.md` with the rank/recall/K
     components, covering the semantic key names the same way.

Runs in-process (subprocess_isolation=False) with the test-mode
evaluator so no cost penalty machinery interferes. Both scored samples
are judge-free by construction, so the gate stays free and non-flaky.

Requires all three provider keys (the evaluator preflights them) plus
HF_ACCESS_TOKEN and ASTA_TOOL_KEY.
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
    # Judge-cache integrity: a torn detailed_reference.json raises
    # straight through astabench's scorer init (it catches only
    # FileNotFoundError) and zeroes every eval. The evaluator's safe
    # writer prevents new corruption; this catches a bad file arriving
    # by other means before it poisons a run.
    from astabench.evals.paper_finder.paper_finder_utils import detailed_reference_path
    if Path(detailed_reference_path).exists():
        try:
            json.loads(Path(detailed_reference_path).read_text())
            print("judge-cache integrity: OK")
        except json.JSONDecodeError as e:
            print(f"FATAL: judge cache is corrupt ({e}):\n  {detailed_reference_path}\n"
                  f"Delete it (verdicts re-judge fresh) before running.")
            return 1
    else:
        print("judge-cache integrity: no cache file (fresh)")

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
    calc = diag.get("score_calculation.md", "")
    if "precision = hits / #submitted" not in calc:
        failures.append(
            f"score_calculation.md missing/degraded for specific_f1 "
            f"(got {calc[:200]!r}) — astabench may have renamed a "
            f"Score.metadata component key"
        )
    else:
        print("specific score_calculation.md: OK")

    score, diag = evaluator(empty_candidate, example)
    print(f"empty candidate: score={score} error={diag.get('error.md')!r}")
    if score != 0:
        failures.append(f"empty candidate scored {score}, expected 0")

    # Semantic score-calculation gate: submit only known_to_be_good papers,
    # which the scorer pre-seeds Perfect without any judge call — free and
    # deterministic, but exercises the real calc_adjusted_f1 metadata keys.
    semantic = next(
        (s for s in samples
         if str(s.id).startswith("semantic")
         and (json.loads(str(s.target)).get("known_to_be_good") or [])),
        None,
    )
    if semantic is None:
        failures.append(
            "no semantic sample with known_to_be_good in the validation "
            "split — the semantic score_calculation gate cannot run"
        )
    else:
        kg = json.loads(str(semantic.target))["known_to_be_good"]
        kg_results = [
            {"paper_id": str(cid), "markdown_evidence": ""} for cid in kg
        ]
        kg_candidate = {
            "agent.py": CANDIDATE_TEMPLATE.format(results=json.dumps(kg_results))
        }
        score, diag = evaluator(kg_candidate, semantic.model_dump())
        calc = diag.get("score_calculation.md", "")
        print(f"known-good-only semantic candidate: sample={semantic.id} "
              f"score={score} judge_cost=${diag.get('other_cost_usd', 0):.4f}")
        if diag.get("other_cost_usd", 0):
            failures.append(
                "known-good-only semantic sample billed judge cost — "
                "known_to_be_good pre-seeding is broken"
            )
        if "harmonic(rank, recall)" not in calc or "K=" not in calc:
            failures.append(
                f"score_calculation.md missing/degraded for semantic_f1 "
                f"(got {calc[:200]!r}) — astabench may have renamed a "
                f"Score.metadata component key"
            )
        else:
            print("semantic score_calculation.md: OK")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
