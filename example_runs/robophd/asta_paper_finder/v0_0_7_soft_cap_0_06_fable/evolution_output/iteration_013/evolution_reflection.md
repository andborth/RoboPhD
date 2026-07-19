# Evolution Reflection — Iteration 13 (asta_paper_finder)

## What worked well

**Deriving the score's sensitivity before choosing a change.** The single most
valuable thing I did this round was compute, on a real observed grade vector,
what each lever is worth: converting grade-2 papers to grade-3 quadruples the
score (0.116 → 0.472); perfecting the ordering adds 2% (0.116 → 0.118). That
one calculation explained the 44–46 plateau (prior iterations kept polishing
ordering — tail sweeps, band keys, boosts — which the harmonic mean caps at a
couple of points) and dictated where all effort should go. Do this arithmetic
FIRST, every iteration, before reading any agent code.

**Treating "a stage reports zero, every time, on every query" as a bug
symptom, not a difficulty signal.** iter12's telemetry said `rescue promoted
0`, `head verify: 0/24 confirmed`, `predicted-perfect 0 -> 0` across queries.
Prior iterations read this as "these queries are hard." The actual cause was
mechanical: every internal grader read `_cut(evidence, 600)` — title + tldr +
the head of the abstract — so the body snippets fetched specifically to prove
weak criteria never reached any grader. Three LLM stages (~45% of spend) had
been grading a text that excluded the evidence they existed to check. The
lesson: when a subsystem's output is *uniformly* null, diff what it reads
against what it's supposed to read before theorizing about the domain.

**Verifying the data path with the actual artifacts.** I confirmed the
truncation bug by loading iteration-12 `submission.json` files and checking
what sits at character 600 (still mid-abstract, all four papers inspected).
Five minutes of jq/python against real submissions beat an hour of code
reading.

**Checking one failure against ground truth via the public S2 API.**
metadata_4 scored 0.000 for all three agents. `curl` to
api.semanticscholar.org (fine during evolution — the Standard Tools
constraint binds the *agent*, not the analysis) showed a gold paper was
"A metric for odorant comparison" in *Nature Methods* — i.e. retrieved in the
452-paper author set and then dropped. That pinpointed the filter, and the
filter's bug was `sorted(distinct)[:120]`: an alphabetical truncation that
cut every N-initial Nature venue. A regression vs an *earlier, cruder* agent
(iter2 scored 0.500 on the same query) was the tell that a filter, not
retrieval, was at fault.

**An offline smoke test with stubbed imports.** Stubbing `inspect_ai` and
`model_registry`, exec'ing agent.py, and unit-testing the changed helpers
(33 checks: grade view contains snippets, evidence stays ≤8 verbatim
passages, coverage ordering, degenerate docs, constants consistency, plus
the scoring-arithmetic checks) caught a wrong assumption of mine within
minutes. There is no other way to test before the expensive eval round.

## What was challenging / time-consuming

- **The agent is 2100 lines and growing.** Reading it end to end costs most
  of the context budget. The per-iteration docstring changelog at the top is
  what makes this tractable — keep maintaining it religiously.
- **Diagnosing across 12 iterations of per-problem directories.** The shell
  loops I wrote to extract K values, grade histograms, and gold sizes across
  all iterations should really be standing scripts. If you write such a loop,
  save it in your iteration dir so successors can reuse it.
- **Truncated views lie.** The 25k-token Read cap on agent.py means you can
  miss the exact line that matters. Grep for the specific constant/callsite
  rather than paging.

## Insights about the domain (worth preserving in the prompt/CLAUDE.md)

1. **Recall in grade-3s is the only lever that matters on semantic queries.**
   rank ≈ 0.55–0.78 essentially for free; recall ≈ 0.05–0.23 is the binding
   term of the harmonic mean. Grade 2 earns literally nothing.
2. **K distribution** (46 observed queries): median 52, p25 20, min 6,
   max 304. Positions 0–20 are judged on every query; position 100+ on ~25%
   of them. Concentrate spend at the top.
3. **Grade 3 requires every weighted criterion judged Perfectly Relevant**
   (`weighted > 0.99`). So evidence must explicitly state *all* criteria,
   including qualifier criteria the abstract satisfies only implicitly —
   fetch body passages for every criterion in the top ~35, not just the ones
   a cheap triage flags as weak.
4. **`references` is dead server-side** under every field variant tried
   (`'NoneType' object is not iterable`), on both `get_paper` and
   `get_paper_batch`. Three iterations were wasted rediscovering this.
   Cites-X verification needs a different mechanism entirely (candidate:
   `snippet_search` scoped to the candidate with the cited work's name as
   query — bibliographies and related-work sections are body text).
5. **Multiple corpus records and split author profiles are routine.** The
   specific path's alias/duplicate handling (title_sim ≥ 0.88 + same first
   author) has scored 1.000 consistently — don't touch it.
6. **`snippet_search` is free and sees body text; keyword search sees only
   title/abstract surface.** Conjunction queries ("X *used in* Y") live in
   method sections. Budget snippet queries their own share of the pool.

## What I would do differently / advice to successors

- **Audit data flow before adding strategy.** Ask, for each LLM stage: what
  exact string does it read, and does that string contain the information the
  stage is supposed to judge? The biggest win in 13 iterations was closing an
  open loop, not a new idea.
- **Look for regressions vs older agents, not just deltas vs the current
  best.** iter2 beating iter12 on metadata_4 by 0.5 was the loudest bug
  signal in the data and nobody had noticed it for ten iterations.
- **Don't over-index on 14-query batches.** Solo wins/losses tables churn;
  a 1–2 point mean difference is ~1 query's noise. Prefer changes justified
  by mechanism (arithmetic, fixed bug, closed loop) over changes justified
  by "it won this batch."
- **Cost is a soft wall, not a target.** All three agents sat at $0.043–0.053
  vs a $0.06 threshold. There was ~$0.01/query of unused headroom for three
  iterations while scores plateaued — spend it on the top-K where recall
  lives (deeper GPT_5_4 verification), funded by cutting depth the judge
  rarely reads (sim beyond position ~55, pool beyond ~320).

## Process/prompt suggestions

- Add to CLAUDE.md the empirical facts above (K distribution, grade-3
  threshold behavior, dead `references` field) so successors stop
  re-deriving them. The grade-3/`weighted > 0.99` rule IS documented but its
  consequence — grade 2 is worth zero, so optimize conversions not ordering —
  deserves a bold call-out.
- The error_analysis_report is score-only. A per-problem one-liner of
  *stage telemetry deltas* (predicted-perfect by stage, rescue/verify
  confirmations) would surface "a stage always outputs zero" bugs in one
  glance instead of requiring stdout spelunking.
- Encourage a standing `analysis_tools/` convention: each iteration's shell
  loops and extraction scripts saved next to reasoning.md so the next
  instance runs them instead of rewriting them.
- The smoke-test pattern (stub imports, exec, assert on changed helpers)
  should be suggested in the evolution prompt — it is cheap and has caught
  real bugs in at least two iterations.
