# Reflection on the DS-1000 evolution session

## What worked well

**Walking the error trail in priority order.** The score table gave me four numbers, but the
*reasons* lived in `agent_stdout`, `extracted_code.md`, `reference.md`, and `test_result.md`
per problem. The cleanest workflow was:

1. Identify problems where the leading agent lost points (`Δ(best-worst)` column + score columns).
2. Distinguish correctness losses (score 0 or "test error" tracebacks) from cost penalties
   (scores like 99.95). Cost-only deltas are a distraction unless they're huge.
3. For the few real correctness losses, read prompt → extracted_code → reference side-by-side,
   and look at `agent_stdout` to see what the agent's internal state was.

This funneled hours of reading down to one or two concrete improvements per iteration.

**Forking the strongest agent rather than reinventing.** Iter9 was clearly the best (94.99) and
its source had a lot of carefully-tuned heuristics. Starting from it and adding only minimal,
non-regressive changes is much safer than starting fresh. Each new heuristic should be a
**strict superset** of behavior — the old success paths stay intact.

**"Sandbox-verify equal output" as the gate for any code rewrite.** When I added a post-critic
loop-scrub pass, the rule "only switch if the rewrite produces the SAME sandbox value" makes
the transform provably non-regressive: same output ⇒ same correctness, only difference is
whether the test_string check passes. Future agents should look for more transforms like this
(equivalence-preserving rewrites verified in-sandbox) — they're free upgrades.

## What was challenging or time-consuming

**Telling cost penalties apart from correctness failures.** Scores like 99.95, 99.99, 99.89
look like near-misses but are really "passed test, paid >$0.16". I wasted an early read on
problem 919 chasing a "fix" that wouldn't have moved score because the underlying answer was
already correct. Reading `result.json` (which has `eval_cost`) first would have saved time.

**The single-problem correctness lever.** With only 20 problems per iteration and 18+ already
at 100, there's typically *one* problem to fix and the rest is just preserving what works.
That made the analysis feel small but the implementation tempting to over-engineer. Resisting
that — keeping the change targeted — was the actual challenge.

## Tools

The provided tooling was sufficient — `Read`, `Grep`, `Bash` (with `jq`/`tree`) gave fast access
to per-problem artifacts, and `Edit`/`Write` were enough for code changes. One small friction:
no `strategy_tools/` directory existed for this iteration despite CLAUDE.md mentioning it, so I
fell back to ad-hoc shell. A canonical "summarize_failures.py" would help future iterations,
e.g. one that produces a table of `(problem, agent, score, cost, correctness_or_cost,
test_string_failed, test_execution_failed)` from a directory of results.

The agent.py can't be imported outside the eval environment (`model_registry` missing) — that's
fine for syntax-checking via `ast.parse`, but you can't do real unit tests on the agent locally.
A small mock `model_registry.py` shipped in the experiment dir would let future iterations run
end-to-end smoke tests before committing.

## What I would do differently

**Look at solo-loss problems first.** The error analysis report flags "solo losses" — these are
the cases where one agent failed and the others passed, which is the highest-signal data because
the *other agents' code* is essentially a free hint at the fix. I went straight to the
all-agents-failed case (269), which was the right call here for total score impact, but in
future iterations the solo-loss pattern is usually a faster win.

**Be explicit about the "string-check vs execution-check" axis early.** DS-1000's two failure
modes — wrong value vs. forbidden token in source — really are different and call for different
fixes. I'd recommend the evolution prompt or CLAUDE.md highlight the test_string vs
test_execution distinction more prominently, and suggest checking each failed problem's
`test_result.md` for which kind of assertion raised. (CLAUDE.md does mention it briefly under
"Scoring"; could be elevated.)

## Insights about the evolution strategy

**The "use your judgment" framing was the right call** for an iteration this far in. There were
clear best/worst agents and a clear bottleneck problem; freedom to fork-and-improve beat any
prescribed strategy. Earlier iterations with weaker leaders might benefit more from structured
exploration (try-N-diverse-approaches strategies), but at iter10 with a 95-point base, the work
is pure refinement.

**Iteration-to-iteration accumulation has a ratchet effect.** Each iteration's "polish" agent
contains a long list of carefully-tuned rules (`SYSTEM_PROMPT`, `NO_LOOP_PATTERNS`,
`DEPRECATED_PATTERNS`, `_check_candidate`, etc.). The prompt got long, but every rule earns its
place from a documented failure. Future iterations should resist consolidating or "cleaning up"
this list — each line is a learned lesson and removing one risks regressing a fixed problem.

**Sample size of 20 is noisy.** A change that fixes one problem on iter10's sample (+5 mean)
might fix zero on the next held-out sample. The cost-vs-correctness tradeoff guidance is
correct ("$0.32 beats $0.15 if it converts one answer") but it's also worth flagging that 20-
problem evaluations have high variance — improvements should be robust patterns, not
single-problem fixes. My change targets a *class* of problems ("idiomatic string-check") not
just problem 269, which I think is the right framing.

## Suggestions for improving prompts/process

1. **Score breakdown column in the report.** Add a per-problem "cost_penalty" column next to
   the score column so cost-only deltas are obvious at a glance.

2. **Tag each failed problem with the failure type** in `error_analysis_report.md`: one of
   `wrong_value`, `runtime_error`, `test_string_assert`, or `cost_only`. This is a 5-second
   classification from the existing artifacts.

3. **Provide a `summary_of_known_traps.md`** that accumulates across iterations — basically the
   collected wisdom in `SYSTEM_PROMPT` of the leading agent, in human-readable form. A new
   iteration could read this first to understand the lay of the land.

4. **Mention the "free zone" threshold ($0.16) more visibly.** It's in CLAUDE.md but it took me
   a beat to internalize that 99.95 ≠ correctness failure.

5. **Encourage non-regressive verification.** Any time an agent rewrites code post-hoc (dep
   scrub, loop scrub, refinement), encourage the pattern "produce candidate → sandbox-verify
   equal/better output → only then switch". Future agents could do this in more places.

6. **Add a tiny mock `model_registry.py`** somewhere on the path for local-testability, OR
   document explicitly that agents can only be syntax-checked locally and end-to-end tested via
   the eval harness. I had to confirm this by trying to import iter9 — would have saved time
   to know upfront.

## Bottom line

The job in iter10 was: fix one problem (269), don't break anything else, and generalize the
fix to a class of similar problems likely to appear in held-out sets. That meant two small
changes to a strong base — not a rewrite. The hard part was resisting bigger changes.
