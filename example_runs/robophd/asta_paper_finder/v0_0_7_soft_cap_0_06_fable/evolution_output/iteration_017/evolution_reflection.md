# Evolution Reflection — Iteration 17 (asta_paper_finder)

## What worked well

**Reading the last reflection FIRST, then verifying its predictions.**
Iteration 16's reflection explicitly asked the next instance to check
whether its pool-widening mechanism fired. It did fire (640 vs 320 in
every stdout) — and the agent still lost. That single check turned "pool
breadth failed" from a guess into a mechanism: the loss came from the
*bundle* (cheap triage + oversized pool + gutted sim), not from breadth
itself. Prior reflections are the highest-density artifact in the whole
directory; ten minutes there saved hours of re-derivation.

**Recomputing the failure attribution with the REAL scoring geometry.**
My first attribution table (copied from iteration 16's analyze7.py) used
"max judged position" as the depth cutoff and mislabeled categories. The
fix — parse K, rank, recall from each `score_calculation.md` and count
grade-3s inside/outside top-K — produced this round's central discovery:
**the judge only grades the first ~K positions, and virtually every judged
grade-3 already sits inside K.** That inverted the naive reading of the
attribution table (which suggested "stranded" papers were an ordering
problem) and redirected the design toward triage precision at the
submission cut. Lesson: before trusting any analysis script inherited from
a prior round, re-derive one row by hand against the raw diagnostics.

**Isolating confounded changes instead of accepting bundle verdicts.**
iter16 shipped ~8 changes in one agent and lost. Its diff was separable:
the compact output format is parse-identical (verified by direct test),
the metadata fixes were root-caused against specific zero-scoring
problems, the evidence lengthening is free by scoring rules. I kept those
four and reverted the rest. Evolution history should be mined at the
*change* granularity, not the *agent* granularity.

**Splice-porting between sibling agents by function boundary.** iter13's
metadata solver was byte-identical to iter12's, and iter16's was iter12's
plus fixes — so `_solve_metadata` swapped in cleanly as a line-splice
between `async def _solve_metadata` and `def make_solver`, with an AST
check for undefined names afterward. Diffing sibling regions (`diff <(sed
-n '/pattern/,/pattern/p' a) <(...)`) is the fast way to learn what
actually changed across a 2000-line lineage; whole-file reads are not.

**Costing every change against measured telemetry.** The `llm-usage`
stdout lines × the price table showed iter13 was already *over* the free
zone at the nominal 73%-semantic test mix ($0.062) despite being safely
inside on this batch's 64% mix. That reframed the whole round: not "spend
the margin" but "widen retrieval while getting cheaper." Batch mix
sensitivity is invisible if you only look at the cost report's batch mean.

**A smoke test that exercises the changed code paths directly.** Stubbing
inspect_ai/model_registry took ~40 lines (copied from iteration 16) and
let me run `_grade_chunk`, `_plan_semantic`, and `_evidence` against fake
LLM replies. One check was initially wrong (grepping for `[:120]` matched
benign probe caps) — grep-the-source checks need to target the exact bug
pattern, not a substring.

## What was challenging / time-consuming

- **Category label drift across analysis scripts.** "not_retrieved" in the
  inherited script meant "not in the submission", which conflates
  pool-miss with triage-cut — exactly the distinction that mattered this
  round. Naming buckets by their operational definition (not_in_pool /
  cut_at_submission / judged_below_K / graded_2) would have saved an hour.
- **Confirming pool membership.** stdout logs candidate-pool *sizes* but
  not member ids, so "was this missed paper in the pool before the cut?"
  is unanswerable post-hoc. One `print` of the pool's corpus ids (or just
  the cut ids) per query would make the retrieval-vs-triage attribution
  exact instead of inferential.
- **The 2350-line agent.** Manageable only via the docstring changelog +
  function outline + targeted sed ranges. The changelog convention (each
  iteration's docstring lists numbered changes with the evidence for each)
  is genuinely load-bearing — keep maintaining it.

## Tools

Filesystem + grep + jq + python heredocs covered everything; no strategy
tools existed this round. The per-problem diagnostics (`judge_verdicts.md`,
`score_calculation.md`, `submission.json`, stdout) are excellent — the
cross-agent natural experiment they enable is the single most valuable
data source. `attrib16.py` in this directory is the reusable version of
the attribution analysis (run it against iteration_017 next round).

## What I would do differently

- **Check the intended mechanisms fire, then stop trusting projections.**
  Next round, grep iteration-17 stdout for: `candidate pool:` (~400 of
  more uniques than before? per-source list should show ~19 lists),
  keyword-query lists (are the 14 actually category-diverse or
  paraphrases?), `head verify:` / `stage2 judge-sim` (did the trims starve
  ordering on any query?), `ambiguous shortlist:` (did a SPIKE-type query
  appear and submit ~12?), and `citation intersection` / `cap_hit` on any
  citing-metadata query. Also verify measured semantic cost ≈ $0.076.
- **Quantify grade-2→3 conversion headroom before another retrieval
  round.** The top-K is full of grade-2s (47 vs 2 on semantic_104). Two
  prior refutations exist for *lexical evidence heuristics*, but nobody
  has yet measured how often the grade-2 verdicts correspond to papers
  where a proving passage EXISTS but wasn't fetched (vs papers that
  genuinely fail a criterion). Sampling ~20 grade-2 papers and manually
  checking their full text via snippet_search would settle whether
  enrichment or retrieval is the next lever.

## Insights about the task itself

1. **The judge reads only the top ~K positions (K=6–114, hidden).**
   Everything below is dead weight except as insurance for large K.
   Ordering *within* the judged prefix is nearly solved; what binds is
   how many true grade-3s make the prefix at all.
2. **Breadth and triage quality are coupled through the 250-slot cut.**
   Widening the pool without keeping per-candidate triage quality
   *increases* lost grade-3s. Any future pool increase must hold
   T1_TITLE/T1_BODY/GRADE_CHUNK fixed or improve them.
3. **Batch composition drives both score variance (±5 pts on identical
   code) and cost-zone membership.** Always project cost at 73% and 86%
   semantic, not at the observed batch mix.
4. **Exact-match zeros are bugs or structural gaps, not hard queries.**
   Every one so far (metadata_4, metadata_25, metadata_26, specific_39)
   had a specific mechanical cause visible in stdout within minutes.
5. **The judge is deterministic given evidence text** (established
   iteration 16, 104/104 agreement). Cross-agent grade differences on
   shared papers are signal about evidence, not noise.

## Suggestions for the process / prompts

- **Add to CLAUDE.md the two hard-won scoring facts**: the judge grades
  only ~K positions (so recall is capped by grade-3s in the prefix), and
  it is deterministic given evidence. Both were re-derived across multiple
  iterations at real cost.
- **Ask each round's builder to log the submission-cut ids** (or pool ids)
  so retrieval-vs-triage attribution becomes exact. One print line.
- **Standing instruction: "read the last two evolution_reflection.md files
  before any analysis."** It happened to work here because iteration 16's
  reflection was excellent, but the prompt doesn't require it.
- **Standing instruction: verify predictions the previous round made about
  its own changes** (each reflection should end with 3–5 grep-able
  predictions; each new round should start by grading them). This converts
  batch-noise-contaminated scores into mechanism-level learning.
- **Report raw F1 and cost penalty as separate columns** in the score
  summary (repeated from iteration 16 — still worth doing).
- **Warn against multi-change agents.** iter16 bundled a good idea with
  three funding cuts and the round's verdict on the good idea was nearly
  lost. When a change needs offsetting cuts, prefer cuts already measured
  as no-ops (gap-fill) over cuts to unmeasured machinery (triage view).
