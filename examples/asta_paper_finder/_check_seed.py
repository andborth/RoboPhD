"""Run the seed agent against one sample of each score_type via eval_candidate.

Validates the full evaluator chain (registry-handle LLM call, MCP tools,
scoring incl. the GPT-4o judge on the semantic sample, cost split)
without spending on evolution. Gates:

  - no sample reports an `error.md` diagnostic
  - every sample reports non-zero agent cost (the seed makes one rerank
    LLM call whenever paper_search returns hits)
  - the semantic sample bills judge cost to other_cost_usd; the
    specific/metadata samples bill zero judge cost
  - none of the judge spend leaks into cost_usd / cost_by_model_usd

Score-based assertions are deliberately loose — n=3 is too noisy to
threshold — but we do print scores for eyeballing. Requires all three
provider keys plus HF_ACCESS_TOKEN and ASTA_TOOL_KEY.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

from RoboPhD import eval_candidate, RoboPhDEvalConfig
from evaluator import JUDGE_MODEL_IDS, PaperFinderEvaluator, load_paper_finder


def main() -> int:
    samples = load_paper_finder("validation")
    picks = []
    for prefix in ("semantic", "specific", "metadata"):
        picks.append(next(s for s in samples if str(s.id).startswith(prefix)))
    print(f"samples: {[str(s.id) for s in picks]}")
    sample_dicts = [s.model_dump() for s in picks]

    seed_src = (HERE / "seeds" / "baseline" / "agent.py").read_text()
    candidate = {"agent.py": seed_src}

    evaluator = PaperFinderEvaluator(
        subprocess_isolation=True,
        apply_cost_penalty=False,
    )
    result = eval_candidate(
        evaluator=evaluator,
        dataset=sample_dicts,
        candidate=candidate,
        config=RoboPhDEvalConfig(eval_timeout=600, max_workers=3),
    )

    print(f"mean_score: {result.mean_score:.3f}")

    failures = []
    for i, (s, d) in enumerate(
        zip(result.per_example_scores, result.per_example_diagnostics or [])
    ):
        d = d or {}
        sid = d.get("sample_id")
        stype = d.get("score_type")
        agent_c = d.get("agent_cost_usd", 0) or 0
        judge_c = d.get("other_cost_usd", 0) or 0
        err = d.get("error.md")
        wall = d.get("eval_wall_clock_seconds")
        print(f"  [{i}] {sid} score_type={stype} score={s} "
              f"agent=${agent_c:.4f} judge=${judge_c:.4f} wall={wall}s")
        if err:
            failures.append(f"{sid} reported error.md: {err[:300]}")
        if agent_c == 0:
            failures.append(
                f"{sid} reported zero agent cost — registry-handle usage "
                f"capture or litellm pricing broken?"
            )
        if stype == "semantic" and judge_c == 0:
            failures.append(
                f"{sid} (semantic) reported zero judge cost — either the "
                f"agent returned no papers or the judge split is broken"
            )
        if stype in ("specific", "metadata") and judge_c > 0:
            failures.append(
                f"{sid} ({stype}) billed judge cost ${judge_c:.4f} — "
                f"deterministic scoring should never invoke the judge"
            )
        overlap = JUDGE_MODEL_IDS & set(d.get("cost_by_model_usd") or {})
        if overlap:
            failures.append(f"{sid} judge model leaked into agent buckets: {overlap}")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
