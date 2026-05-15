# Evolution Reflection — Iteration 4 (DS-1000)

## What worked well

- **Diagnose before designing.** The single highest-leverage move was reading
  every per-problem diagnostic for the *consensus failures* (the problems all
  agents missed), not just the score table. `agent_stdout`, `extracted_code.md`,
  `test_result.md`, `reference.md` together pinpoint root cause precisely. The
  4 consensus failures cleanly bucketed into distinct, *generalizable* failure
  modes (style-token test, invent-signature function, env-noise masking,
  idiomatic-dtype). Targeting buckets beats per-problem patching.

- **Build on the winner, change additively.** iter3 already strictly dominated
  (80%, no regressions vs the 75% agent). Keeping its pipeline verbatim and
  adding detector-gated layers means zero regression risk on the 16/20 it
  already solved — every change is opt-in behind a regex/AST detector.

- **The `agent_stdout` print logs were decisive.** Seeing `sandbox OK on
  attempt 0` for problems 269 and 420 immediately proved the sandbox
  verify-loop was *structurally blind* to style failures and
  defined-but-uncalled functions — not a model-quality issue. That reframed
  the whole strategy from "use a stronger model" to "fix the verifier's blind
  spots." Without those logs I'd have guessed wrong.

- **Unit-testing helper logic in isolation** (stripping the inspect_ai imports
  and exec-ing only the pure functions) caught nothing broken this time but
  gave high confidence cheaply (~1 min). Worth doing every iteration.

## What was challenging / time-consuming

- **Sandbox scope friction.** The conda python (`/opt/anaconda3/...`) is
  outside the read scope, so `py_compile`/unit tests failed until I found
  `/usr/bin/python3` (3.9). Future instances: use `/usr/bin/python3` directly
  for scratch validation; don't waste a turn discovering the denial. Note
  3.9 vs the sandbox's 3.11 — fine for syntax/AST checks, but don't rely on
  3.10+ syntax in test harnesses.

- **No way to actually run the agent locally.** I can validate helpers and
  syntax but not the end-to-end solver (needs inspect_ai + model handles +
  Docker sandbox). All confidence in the full pipeline is by reasoning, not
  execution. This is the biggest uncertainty in the loop.

- **Wrong-value traps are nearly unfixable from outside.** Problem 165
  (`np.column_stack` upcasts int→str, expected dtype is object) can't be
  caught by sandbox verification because there's no expected output and no
  traceback. Only a prompt nudge is possible. ~25% of consensus failures are
  this class and largely irreducible without the reference.

## Tooling notes

- The provided directory structure and `error_analysis_report.md` are good.
  The report's split-decision breakdown ("✓ A,B | ✗ C") is the fastest way to
  see which agent has unique strengths to preserve.
- `strategy_tools/` did not exist this iteration — fine, but the prompt
  mentions it conditionally, which is correctly handled.

## What I'd do differently / suggestions for the process

1. **Provide expected outputs (or a public validation split) to the agent
   at solve time is NOT possible — but for the *evolution author*, surfacing
   `expected_output.md` in diagnostics would help.** It exists in the problem
   dirs; future instances should always diff `actual_output.md` vs
   `expected_output.md` for wrong-value failures to spot dtype/shape traps.
   (I used `reference.md`; `expected_output.md` is the complementary signal.)

2. **The consensus-failure list is the highest-signal artifact.** Suggest the
   prompt explicitly tell future instances: "Start by reading all diagnostics
   for consensus failures; categorize them; only then design." This would
   shortcut the exploration phase.

3. **Cost is a non-constraint here.** Mean spend was ~$0.01 vs $0.08 free
   zone (8x headroom). Future instances should not hesitate to add
   verification/regeneration rounds or escalate to a stronger model — the
   prompt could state this more bluntly to discourage premature cost-saving.

4. **Diminishing returns on prompt-checklist tweaks.** iter2→iter3 already
   loaded a big gotcha checklist. Adding more bullet points has low marginal
   value; structural fixes (making the verifier see new failure classes) have
   high value. Future instances should prioritize *mechanism* changes over
   *prompt-wording* changes once the checklist is mature.

5. **Beware over-engineering fragile heuristics.** The invent-signature
   variable-detection (filtering config-named skeleton vars) is the riskiest
   part of my agent — regex-based, could mis-pick the principal input. I kept
   it bounded (probe only adds a traceback, never silently changes behavior)
   but future instances should keep such heuristics *fail-safe*: a bad guess
   should at worst be a no-op, never a regression.

6. **Track which split-decision wins to protect.** I verified 723's skeleton
   (`def f(...)` present) wouldn't trip the new invent-function path, avoiding
   a regression on a problem iter3 passed. Always cross-check new detectors
   against the previously-passing split-decision problems.

## One-line summary

Read consensus-failure diagnostics first, categorize root causes, then make
*structural* (verifier blind-spot) fixes on top of the dominating agent with
fail-safe detectors — not more prompt bullets.
