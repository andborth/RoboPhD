# Evolution Reflection — DS-1000 Agent (Iteration 8)

## What worked well

- **Start from the proven champion, change one thing.** iter3_safe_repair was the
  consistent winner across iterations. I kept its entire scaffold byte-for-byte
  (simplicity guide, func-body indentation normalization, objective execution probe,
  never-second-guess-a-clean-run, error-only repair, always-fall-back-to-best-clean) and
  added exactly one new layer. Result: top performer (85% vs 80/75/75), zero unique
  failures, and a unique win — at $0.0053 mean cost.

- **Objective signals over LLM arbitration.** The decisive lesson from the lineage is that
  letting an LLM *review/rewrite a clean answer* (iter5/iter6/iter7) regresses: it broke
  correct literal answers (e.g. dropped a `tf.Variable.assign` side-effect, over-engineered
  a sparse multiply) more than it fixed. My consensus layer decides by **exact execution
  output agreement** across candidates — never a model opinion — so a clean majority is
  never overwritten. This kept the floor at iter3 while adding upside.

- **Prompt-perspective diversity, not temperature.** Instead of relying on sampling
  temperature (whose support is unclear here — the docs only advertise `reasoning_effort`
  and `max_tokens`), I generated 3 candidates under distinct framings (idiom /
  general-robust / exact-output-form). This guarantees diversity even at temp 0, and the
  output-form perspective directly recovered a dtype/shape win (problem 238) and at least
  partially addressed the float-cast class of bug.

- **The free sandbox is the highest-value tool.** `python_session` is unmetered, so
  executing every candidate and grouping by a structured probe (TYPE / COLUMNS / DTYPES /
  repr) costs nothing and is the backbone of both repair and consensus. Lean on it heavily.

## What was challenging

- **20-problem batches are noisy.** A 1–2 problem swing dominates the score, so it's easy
  to over-read a single batch. iter3 scored 95% one iteration and 75% the next on a
  different draw. The right response is to optimize *expected* value with no regression
  path, not to chase the specific misses of one batch.

- **A floor of unwinnable problems.** Some failures are structurally unfixable:
  - **Environment bugs** (706: a protobuf/keras `MessageFactory.GetPrototype` crash at
    model-construction — the reference itself would fail).
  - **Quirky references** that no idiomatic code reproduces (129: a per-user positional
    `df.iloc[i:i+tot].sample()` order; 165: `np.column_stack` silently coercing int counts
    to strings, so the "correct" answer is the buggy-looking one).
  Recognizing these as out-of-reach (rather than burning effort/regression-risk on them)
  is itself a skill. Don't add aggressive output-mimicry heuristics to chase them.

## Tooling notes

- Per-problem diagnostics (`problem.md`, `reference.md`, `extracted_code.md`,
  `test_result.md`) are excellent — comparing my extracted code against the reference made
  every failure diagnosable in seconds without re-running. The `test_string` vs
  `test_execution` distinction (style/idiom vs correctness) is valuable; worth surfacing
  in the report.
- `agent_stdout` print-tracing was essential for verifying the control flow actually fired
  (consensus N/M, escalation, repair) and confirming cost paths. Future agents: print
  enough to audit which branch ran.
- Minor friction: `cd` into a relative subdir reset between Bash calls; absolute paths are
  more reliable.

## Advice for future instances

1. **Don't rebuild — inherit and add one orthogonal layer.** The codebase already encodes
   hard-won fixes (indentation normalization, error-only repair). Reuse the winner's
   helpers verbatim; spend your budget on a single well-motivated improvement.
2. **Decide with objective signals; never let an LLM overwrite a clean answer.** This is
   the most repeated regression in the lineage. Consensus/voting by execution output is
   safe; free arbitration is not.
3. **Exploit the cost headroom but don't assume more spend = more accuracy.** Free zone is
   $0.05 mean; agents win at ~$0.002–0.01. A stronger *generator* is NOT obviously better —
   the mini's simplicity bias is a feature, since over-engineering produces clean-but-wrong
   answers that error-only repair can't catch. Reserve the strong model for genuine
   uncertainty (no consensus) or errors, not for everything.
4. **Build for no-regression.** Ensure every shipped answer is a clean run or the proven
   fallback. An improvement that can only promote, never demote, beats a cleverer one that
   sometimes regresses.
5. **Triage failures before acting.** Sort misses into env-bug / unguessable-reference /
   genuinely-fixable. Only the last bucket is worth code changes; the others are traps.

## Process / prompt suggestions

- The error report's split-decision view is the single most useful artifact — keep it.
  Adding a tag per failure (correctness vs style vs env-error vs quirk-reference) would
  speed triage further.
- Consider reporting score across *both* the current and prior batches per agent, to damp
  the 20-problem noise and discourage single-batch overfitting.
- The "Use Your Judgment" open-ended strategy worked well here; the structured lineage of
  prior agents + diagnostics gave enough signal to make a disciplined, evidence-based call
  without prescriptive constraints.
