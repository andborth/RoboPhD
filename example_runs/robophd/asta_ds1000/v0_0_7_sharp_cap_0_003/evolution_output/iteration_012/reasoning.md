Name: iter12_strong_toplevel

# Iteration 12 — analysis and plan

## What the data says (11 iterations, decisive and consistent)

The architecture has been settled for many iterations and re-confirmed every batch:

**Single `GPT_5_4.generate()` call, no reasoning, no verification, iter6's TINY
preamble, format-only extraction hardening.** This recipe (iter6/iter10) has led every
recent batch (iter6: 80/75/90; iter10_strong_reindent: 85% on iter-10, 70% and
leader on iter-11).

Two whole families of ideas were falsified with data and must not be revisited:
- **Machinery** (verify / self-consistency / reasoning escalation) is net-negative.
  Wrong DS-1000 answers execute cleanly — the scorer compares the exact target *value*
  (dtype/shape/order/index), so crash-verification catches nothing, and the marginal
  vote/verify loop swaps correct-simple answers for over-clever-broken ones.
- **Heavy/prescriptive preambles** are net-negative (iter5's 5-rule guide → 40%;
  iter7's contract preamble caused IndentationErrors and over-condensed one-liners).
  Only iter6's genuinely tiny output-contract preamble survives.

## iter-11 result, correctly attributed

iter-11 batch: iter10_strong_reindent 70% (14/20) > seed 65% > iter11_strong_assign
60% (12/20). iter11 = iter10 + a third format fix (target-variable assignment wrap).

I checked whether the assign-wrap *caused* iter11's two extra failures (723, 838). It
did not — both are pure model-content sampling differences:
- **723**: iter10 sampled `return sA.multiply(sB)` (correct); iter11 sampled
  `return sA.multiply(np.asarray(sB.toarray()).ravel())` (wrong). Same prompt, different
  GPT_5_4 sample.
- **838**: iter10 sampled a simple `eval_set=[(testX,testY)]` fit; iter11 sampled an
  elaborate custom `GridSearchCV` subclass (wrong). Sampling noise.

Neither difference is a post-processing artifact. The assign-wrap fired harmlessly (or
not at all) everywhere on the batch. So iter11's lower score is n=20 sampling noise
(~1σ per problem), **not** evidence against the third fix — which is genuinely
fix-or-no-op and removes a confirmed recurring loss class (451, dropped by two
GPT_5_4 agents in two separate batches).

## Decision for iter-12

Keep the settled recipe and **all three** validated pure-upside format fixes
(unescape, function-body reindent, target-variable assignment). Over many held-out
problems, carrying every fix-or-no-op format fix maximizes expected accuracy: each
removes a confirmed rare loss class and none can regress a working answer.

**One concrete improvement over iter11.** iter11's assign-wrap bailed on *any* `=` in
the emitted line, to avoid double-assigning something that already contained a
statement-level assignment. But that also blocked it whenever the bare expression
contained a keyword argument or dict literal — e.g. `np.array([1,2], dtype=int)` or
`pd.Series(d, index=idx)` — an extremely common shape for DS-1000 one-liners. Those
are still bare expressions that would `NameError` against the hidden test, yet iter11
would leave them unwrapped and lose them.

iter12 replaces the crude "any `=` → bail" guard with a bracket/quote-depth scan
(`_has_top_level_assign`) that only treats a `=` at statement level (depth 0, outside
strings, excluding `==`/`!=`/`<=`/`>=`/`:=`) as a real assignment. A `=` inside `()`,
`[]`, `{}`, or a string is a kwarg/dict/subscript and no longer blocks the fix. This
**strictly widens** the assign-wrap's coverage while keeping it fix-or-no-op: it fires
only when there is genuinely no statement-level assignment (so the line was guaranteed
to `NameError` as-is), never on function-style problems (no `NAME = ...` placeholder),
and never on an already-assigned or multi-line answer.

There is still exactly **one candidate** (the model's direct answer) — the falsified
machinery/guide failure modes cannot recur. All changes touch only the *form* of the
extracted code, never the model's chosen answer.

## Validation

`test_extract.py` — 13 end-to-end extraction cases + unit tests for the depth scanner,
all passing:
- unescape (`&lt;`→`<`), function-body reindent (fix + no-op), 451-class bare-expr wrap;
- **new**: kwarg-expr and dict-kwarg exprs now wrap correctly (iter11 missed these);
- no-op guards hold: already-assigned (incl. kwarg RHS), multi-line, statement,
  function-style, and comparison-as-answer all behave correctly.

## Why this generalizes / cost

- Model quality and idiom-preference come from the settled GPT_5_4 + tiny-preamble
  backbone that has led every batch; nothing about the read is over-fit to a batch.
- The format fixes are deterministic and problem-agnostic; each is provably
  fix-or-no-op, so they can only help unseen problems, never hurt.
- Cost: one GPT_5_4 call on DS-1000's short prompts ≈ $0.0013–0.0016/problem, well
  inside the $0.003 free zone (default reasoning "none", `max_tokens=1024`). Mini
  fallback on provider error/empty so a hiccup never hard-zeros a problem.
