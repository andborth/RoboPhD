# Evolution Reflection — DS-1000 Agent (Iteration 11)

Advice for future instances working on this task.

## What worked well

- **Read the lineage history, not just the last iteration.** The most valuable
  move was building a score-over-time table across iters 5–10 from the
  `error_analysis_report.md` files. The latest batch is tiny (20 problems) and
  saturates at 100% for multiple agents, so it carries almost no signal. The
  *trajectory* (which agent held up at 85–95% on harder batches vs. which spiked)
  is what actually distinguishes a generalizer from a lucky run. Do this first.

- **Treat the codebase as an evolutionary record.** Each agent's module docstring
  documents exactly what it changed and why (e.g. iter10's problem-165 dtype story,
  iter4's "majority-vote-without-judge loses" lesson). Reading the top 2–3 agents'
  docstrings end-to-end gave me the design rationale faster than re-deriving it.
  Future instances should keep this discipline: write a clear docstring explaining
  the *delta* and the evidence for it.

- **Additive, superset changes over rewrites.** The winning pattern in this lineage
  is "keep the proven core verbatim, add ONE well-motivated, separately-validated
  lever." I combined iter10's dtype-rich diagnostics with iter7's third-family
  candidate. Low risk (every old path preserved), clear hypothesis, easy to reason
  about. Wholesale rewrites would risk regressions on an already-strong (100%) base.

- **Identifying orthogonal, already-validated levers.** Rather than inventing
  something novel, I looked for two ideas that had each independently helped but had
  never been combined (dtype diagnostics + 3-family diversity). Composition of
  proven levers is a high-expected-value, low-variance strategy here.

## What was challenging / time-consuming

- **Score saturation makes the signal weak.** With batches of 20 and several agents
  at 100%, you can't tell which design is genuinely better on the visible data. Most
  of the "decision" is therefore reasoning about generalization, not measurement.
  This is inherent, but it means confident A/B improvement is hard.

- **No held-out validation in-session.** I can't run my agent against problems
  before submitting (no labeled examples + the real models aren't importable in the
  scratch env). I could only syntax-check and unit-test the pure helper functions in
  isolation (extracting them via `ast` to dodge the `model_registry` import). That
  caught nothing broken but also couldn't validate the end-to-end behavior change.

## Tools

- `model_registry` and `inspect_ai` are injected only at eval time, so `agent.py`
  won't import in the scratch env. **Tip for future instances:** to smoke-test logic,
  parse the file with `ast` and `exec` only the pure helper functions (regex/parsing)
  — that's a fast, reliable correctness check for the non-LLM machinery, which is
  where subtle bugs (prompt parsing, code extraction, consensus comparison) actually
  live. `jq`/`grep` over the iteration reports were sufficient for analysis; no
  dedicated strategy_tools/ existed this round.

## What I'd do differently / suggestions

- **The strategy keeps producing near-clones.** The clone-detection penalty fired on
  iter4, iter7, iter9 — agents that scored identically to a predecessor on all 20
  problems. When the base is already at/near 100% on small batches, *any* additive
  change tends to score identically. The evolution process would benefit from a
  **larger or harder evaluation batch** (e.g. 50–100 problems, or a curated set of
  historically-failed problems like 165, 372, 883, 611, 284) so changes are
  distinguishable and clone penalties reflect real sameness rather than batch
  saturation.

- **A persistent "hard problem" regression set.** Several problem IDs recur as
  failures across iterations (165, 372, 883, 723, 706). Feeding a fixed pool of these
  into every iteration's eval would directly measure whether a change fixes known
  weaknesses without regressing — far more informative than a fresh random 20.

- **Cost is a non-constraint here; lean into it.** Every agent sits at $0.01–0.03
  against a $0.08 free zone. Future instances should not over-optimize cost — spend
  the budget on diversity (more candidates) and a strong judge. The marginal accuracy
  is worth far more than the marginal cents until you approach $0.06+.

## Insight about the strategy itself

The dominant failure mode is "code runs cleanly but is subtly wrong" (dtype, tie
handling, axis names, label-indexing convention, near-synonym matplotlib kwargs).
Execution alone doesn't catch these — only *grounded comparison against the expected
output shown in the prompt* does. Every winning generation invested in exactly that:
richer diagnostics (dtype/repr), a stronger judge that reads the prompt's example,
and cross-family diversity so a correct reference exists for the judge to prefer.
Future work should keep pushing on the *quality of the comparison signal* (what gets
printed and judged), not on raw generation horsepower.
