# Evolution Reflection — Iteration 013

## What worked well

**Comparing the previous iteration's three agents head-to-head was the
single highest-value activity.** The error-analysis report's per-problem
score table immediately surfaced two diagnostic patterns:

1. iter12 had **one solo win** (problem 269) — a single feature responsible.
2. iter12 had **two solo losses** (906, 860) — a different feature responsible.

Reading the agent_stdout for those four cases (iter12 won 1, iter12 lost 2,
plus iter8/iter9 winners on the lost ones) told me exactly which iter12
addition to keep and which to drop. No speculation — the stdout literally
prints "alt-approach regen adopted (NoneType fixture noise; alt smoke FAIL)"
right before the agent submits a worse answer than its candidates produced.

**Reading the cost report alongside scores caught a subtle pattern**: iter12
paid 2× per problem ($0.044 vs ~$0.020) which translated to small (0.01-0.03)
cost penalties on every problem it shared with iter8/iter9. These never flip
the ranking on their own, but they confirmed that iter12 spent more per
problem AND solved fewer — strictly dominated.

**Forking the proven base and adding one feature** was clearly the right
move. iter9 was tied at 95.0; iter12 was an experiment that mostly hurt.
Picking the winner of the two ties and grafting only the demonstrated-good
piece of the experiment is much safer than building from scratch.

## What was challenging

**Distinguishing genuinely good features from features that didn't matter
on this batch.** iter12's TF Keras 3 hint and pandas-tuple hint targeted
problems that simply weren't drawn this iteration. They might help on
future batches; they might be no-ops; they might be subtly wrong on edge
cases. With one batch of 20 problems I cannot tell. I dropped them — but
that's a judgment call, not a data-driven one. **More iterations of
held-out evaluation per agent would resolve this.** Even 3 batches of 20
would discriminate "real win" from "lucky win" much better.

**The score column for iter12 looked like 99.97, 99.98 etc. — visually
indistinguishable from 100.00.** I almost missed that those are
correct-but-cost-penalized scores while -0.030 means actually wrong. The
sign and magnitude both encode something. A column that explicitly split
correctness from cost-penalty would save a few seconds and prevent
misreading.

## Tools

- `Read`, `Grep`, `Bash` were sufficient. I never needed `Agent` for this
  task — the analysis was localized to ~5 files.
- The `agents/` and `iteration_NNN/` layout is good. Per-problem
  directories with `extracted_code.md`, `test_result.md`, `agent_stdout`,
  and `result.json` made root-cause analysis fast.
- The `error_analysis_report.md` summary is well-structured. Keep it.
- One small friction: `model_registry` isn't on the read scope, so I
  couldn't import-test the agent inline. The AST syntax check is the
  workable substitute.

## Suggestions for future iterations

1. **Always investigate solo wins AND solo losses, not just losses.** A
   solo win identifies a feature that converts a problem class. A solo
   loss often identifies a regression. Both are signal.

2. **When a previous agent has many features but a worse score than its
   parent, default to fork-the-parent + add-the-one-proven-feature.**
   Don't try to keep "most" of the regression — pick the one
   demonstrated-good piece. Resist the urge to keep version-aware
   gotcha-fix prompts that haven't fired on test data.

3. **Cost stays in the free zone (<$0.04/problem) easily with 4
   candidates.** The 5th model adds ~$0.01-0.02/problem, which lifts you
   into the cost-penalty zone for marginal correctness gain. 4 is the
   sweet spot for this task.

4. **The smoke fixture's `load_data() -> None` is a recurring noise
   source.** When all candidates fail the same NoneType-from-stub error,
   smoke is signalling nothing about correctness. The
   "code-consensus short-circuit when smoke is uninformative" pattern
   handles this safely; the "alt-approach regen" pattern handles it
   destructively. Future iterations should preserve the former and
   resist re-introducing the latter.

5. **Style assertions (`assert "for" not in tokens`) are a real and
   common DS-1000 trap.** Detecting prompt phrases like "idiomatic" /
   "vectorized" / "efficient" and tokenizing candidates for `for`/`while`
   to prefer loop-free options is a clean, generalizable feature. It
   will likely keep paying off on unseen problems.

6. **Process improvement for the prompt**: the evolution prompt could
   note explicitly that "score = 100 × correctness − cost_penalty,
   so 99.97 ≠ wrong-with-penalty, it's correct-with-tiny-penalty" — this
   distinction is easy to miss when scanning the report.

## What I'd do differently

If I had this iteration again I'd do exactly the same triage but spend
less time reading the iter9/iter12 source side-by-side and more time
exercising my draft agent against the diagnostic problems (269, 906,
860) before submitting. That would have caught any regression in the
loop-token detector or the style-retry validation against the chosen
candidate's REPR. The current implementation does include the safety
gate (only adopt style retry if its smoke REPR matches the original's),
so I'm reasonably confident — but a quick local replay would be the
cheap way to confirm.
