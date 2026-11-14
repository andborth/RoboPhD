# Evolution Reflection — DS-1000 Agent (Iteration 10)

Concise notes for future instances working on this task.

## What worked well

- **Starting from the most complete prior agent and making one targeted change.**
  iter9 already encoded hard-won lessons (dtype/shape matching, construct-undefined-
  objects, function-arity, format-aware module/function handling, value serializer,
  cross-model repair, escalation). Re-deriving any of that risks regressions for no
  gain. The high-EV move was to keep the proven machine intact and add a single,
  well-scoped lever.

- **Distinguishing signal from noise before acting.** All three iter9 agents tied
  at 80%. The "split" problems (each solved by exactly one agent) were model
  stochasticity, not capability gaps — the agents differed only in reasoning effort
  and escalation timing. Recognizing this stopped me from chasing noise (e.g.
  adding voters or bumping reasoning, which the data showed bought nothing: iter9
  spent 5× iter3 for identical accuracy).

- **Focusing on the *consensus* failures.** The only systematic, addressable
  pattern was problems all three agents missed (445, 883). Both shared one root
  cause: the agents gave the *cleverer / more statistically faithful* answer while
  DS-1000's reference uses the *direct literal transform*. That generalizes into a
  prompt rule; per-agent splits do not.

## Key domain insight (worth propagating)

**DS-1000 reference solutions favor the most direct, literal reading** — the
simplest expression that reproduces the shown example — *not* the most
sophisticated or statistically correct equivalent. Two recurring traps:
1. "reverse/opposite of F" → arithmetic transform of F's output (`len(a)-rankdata(a)`),
   not negated inputs (`rankdata(-a)`), which diverge at ties/boundaries.
2. Feed data in the form presented (pass the given matrix straight in); don't add
   preprocessing the problem never mentioned (squareform, reshaping, normalization).

This is a **prompt-only lever**. Voting/self-checking cannot fix it: there is no
oracle in the sandbox, and independent strong models make the *same* clever
substitution, so they agree on the wrong answer. Future instances should not waste
budget trying to ensemble their way out of this class.

## What was challenging / time-consuming

- **Small, noisy batches (20 problems).** A single problem is 5% of the score, and
  the batch resamples each iteration (these same agents scored 95% in iter8, 80%
  here). This makes it genuinely hard to tell whether a change helped or the batch
  just got harder. Treat in-batch deltas under ~2 problems as noise.

- **Unfixable-by-design failures.** Some misses (445, 883) require matching an
  idiosyncratic, arguably-wrong reference. You can nudge toward them with a prior
  but cannot guarantee them without overfitting. Accepting this early saves time.

## Tools

- The per-problem diagnostics (`problem.md`, `reference.md`, `extracted_code.md`,
  `test_result.md`, `library`) were excellent — the reference + extracted code +
  traceback side-by-side is exactly what's needed to diagnose a miss. The
  `test_string` vs `test_execution` distinction (form vs correctness) is documented
  and real.
- `error_analysis_report.md` cleanly separates consensus failures from splits —
  this is the single most useful artifact for deciding where to spend effort.
- No `strategy_tools/` dir existed this iteration (the prompt says "if it exists").
- I could not run the agent end-to-end (no eval harness in the workspace), so
  validation was limited to `py_compile` + structural grep + manual reasoning. A
  way to dry-run the agent against the visible batch would tighten the loop a lot.

## What I'd do differently / suggestions for the process

- **The biggest limitation is the inability to measure.** Future instances make
  blind prompt edits and hope. If the harness exposed a cheap "score this candidate
  agent on the last N visible problems" command, iteration quality would jump. Even
  caching prior agents' per-problem outputs so a new prompt could be A/B'd offline
  would help.

- **Diminishing returns on architecture.** Across iter3→iter7→iter9 the
  architecture converged (strong base model + format handling + value-agreement +
  repair). Further gains are almost entirely in (a) the base-instruction prompt and
  (b) variance reduction, not new control flow. Future prompts could say so
  explicitly to steer effort toward prompt content over re-plumbing.

- **Cost was never binding.** Every agent sat 2–10× under the $0.05 free zone.
  Unless an iteration's instructions emphasize cost, optimize purely for accuracy
  and ignore the penalty math.

- **Guard everything; make the loop monotone.** The proven pattern is that every
  verify/repair/escalate step is wrapped so it can only convert a 0 into a possible
  1, never the reverse (empty/crashing solutions score 0 anyway). Keep this
  invariant — it's why successive agents never regressed below their base.

## One-line summary

The architecture is mature; the remaining accuracy lives in the base-instruction
prompt. Diagnose consensus failures (not splits), encode their *general* lesson,
keep the proven monotone machine untouched, and don't ensemble your way at
problems that need a prior, not an oracle.
