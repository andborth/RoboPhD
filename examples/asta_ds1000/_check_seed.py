"""Run the seed agent against a 3-sample slice via eval_candidate.

Validates the full evaluator chain (LLM call, sandbox exec, scoring,
cost reporting) without spending on evolution. Gate: every sample
must complete without an `error` diagnostic and report a non-zero
agent cost. Score-based assertions are deliberately omitted —
n=3 is too noisy to threshold meaningfully — but a broken chain
will surface as an `error` field in diagnostics, which we do gate on.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

from RoboPhD import eval_candidate
from evaluator import Ds1000Evaluator, load_ds1000


samples = load_ds1000("validation")[:3]
sample_dicts = [s.model_dump() for s in samples]

seed_src = (HERE / "seeds" / "baseline" / "agent.py").read_text()
candidate = {"agent.py": seed_src}

evaluator = Ds1000Evaluator(subprocess_isolation=True)
result = eval_candidate(evaluator=evaluator, dataset=sample_dicts, candidate=candidate)

print(f"mean_score: {result.mean_score:.3f}")
print(f"per-example scores: {result.per_example_scores}")

failures = []
for i, (s, d) in enumerate(zip(result.per_example_scores, result.per_example_diagnostics or [])):
    d = d or {}
    cost = d.get('agent_cost_usd', 0) or 0
    err = d.get('error')
    print(f"  [{i}] sample {d.get('sample_id')} library={d.get('library')!r} score={s} cost=${cost:.4f}")
    if err:
        failures.append(f"sample {d.get('sample_id')} reported error: {err[:200]}")
    if cost == 0:
        failures.append(f"sample {d.get('sample_id')} reported zero cost — litellm pricing or usage capture broken?")

if failures:
    print()
    print("== FAIL ==")
    for f in failures:
        print(f"  {f}")
    sys.exit(1)
print()
print("== PASS == 3/3 samples completed without error and reported non-zero cost")
