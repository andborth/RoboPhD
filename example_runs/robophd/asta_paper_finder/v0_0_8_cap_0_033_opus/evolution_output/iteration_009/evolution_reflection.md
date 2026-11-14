# Evolution Reflection — PaperFindingBench (Iteration 9)

Read alongside CLAUDE.md and the iter-2..8 reflections (still accurate). This round I
stopped tweaking the grade-3 *evidence* lever and attacked the *rank* lever instead.
Below is what I'd tell the next instance.

## The single most important thing I learned this round

**"Rank is healthy, recall is the sole binding term" was over-generalized — verify it
per query, don't inherit it.** Every reflection since iter-7 repeated that claim, and
it steered three rounds of effort exclusively onto grade-3 evidence conversion (iter7
regressed, iter8 was flat). But reading iter6's iteration-8 `score_calculation.md`
across all 11 semantic queries, **rank is 0.34–0.55 on four of them** (semantic_70
0.34, semantic_203 0.51, semantic_108 0.52, semantic_104 0.55). On those, I checked
`judge_verdicts.md`: the grade-3 papers are *already retrieved and inside the top-K
window*, just scattered (semantic_104: "Perfectly Relevant" at positions 2, 4, 17, 18,
19, 20, 23, 40, 46, 52 for K=56). That is a pure **ordering** loss, not a retrieval or
evidence loss — and `score = harmonic(rank, recall)`, so lifting rank 0.55→0.85 moves
the score directly, with zero risk to the delicate evidence assembly. Lesson: **an
inherited diagnosis becomes dogma across rounds; re-derive it from the current batch's
per-problem files before committing your one change.**

## The specific under-exploited weakness (my change)

iter6's reranker feeds the LLM only `title[:140] + one snippet[:200]` per candidate and
rates 0–3. Two flaws I fixed (rerank only; everything else byte-identical to iter6):
1. **The abstract — the best all-aspect discriminator — is enriched onto every
   candidate but never fed to the reranker.** I now feed `title + abstract[:320]`.
2. **A 0–3 scale collapses every strong paper into "3"**, forcing noisy retrieval-order
   tie-breaks. I switched to a finer 0–10 scale anchored to the judge's rule (top score
   only if EVERY aspect holds), so all-aspect papers sort above topic-only ones — the
   grade-3 vs grade-2 distinction recall rewards.
Output length is unchanged (one int per candidate), so the existing empty/partial-output
fallbacks are untouched: worst case it matches iter6. **Whether it beat iter6 is for the
next instance to check** — diff rank on semantic_104/108/70/203 vs iter6.

## What worked well (confirming prior rounds)

- **Build on the aggregate winner (iter6), change ONE thing, keep the rest byte-for-byte.**
  Fourth round this is the right call. It's the only way attribution survives the noise.
- **Diagnostics are the whole game — but read `judge_verdicts.md` position-by-position,
  not just the rank/recall summary.** The scatter of grade-3 papers *within* the top-K
  is invisible in `score_calculation.md` (which only gives the two scalars) but obvious
  in the verdict list, and it's what revealed the ordering headroom.
- **Smoke-testing pure helpers with stubbed inspect_ai/model_registry** (`_smoke.py`,
  ~150 lines, 4 tests incl. one end-to-end `_solve_semantic` run). Confirmed the abstract
  reaches the reranker, 0–10 reorders correctly, and empty/partial output falls back.
  Gotcha still true: stub `sys.modules["model_registry"]` *before* importing agent, and
  the agent binds `GPT_5_4_MINI` at import — patch the handler object the agent holds.

## A retrieval fact worth banking (checked this round)

Most semantic queries have **empty `known_to_be_good`/`known_to_be_bad`** — the judge
scores purely on the weighted criteria, so `known_good` is only a sparse diagnostic
hint, not the gold set. But when it IS populated it's gold: semantic_7's single
`known_good` (254685611) was **never retrieved** (a genuine retrieval gap), and 2/3 of
semantic_125's were missed. So there are three distinct failure modes and they need
different fixes: never-retrieved (retrieval breadth), retrieved-but-scattered (rank —
my target), retrieved-but-grade-2 (evidence). Don't apply the evidence hammer to a
retrieval or ordering problem. Print `known_good` positions per query to classify.

## What was challenging / time-consuming (unchanged perennials)

- **No local tool execution.** The reranker's live ordering quality is unverified until
  eval; the smoke test only proves the plumbing. Mitigation as always: the change only
  *reorders* candidates (cannot lower recall, evidence assembly untouched) + fallbacks.
- **Cross-iteration attribution is still fully manual.** There is STILL no
  `strategy_tools/` dir despite the prompt implying one may exist, and every reflection
  since iter-6 has asked for a script that diffs per-shared-problem {rank, recall,
  grade-3 count} between two agent dirs. It is the single highest-value missing tool and
  I re-request it. Samples are also redrawn each round, so only a handful of sample_ids
  overlap the prior iteration — the batch relabels shared problems only by luck.

## Suggestions for the prompt / process (ranked)

1. **Ship the `strategy_tools/` diff script.** Requested four rounds running. Without it,
   every instance re-derives the rank/recall breakdown by hand and can't tell if last
   round's feature actually worked.
2. **A/B two agent slots per iteration.** Still the biggest process win on the table.
   Bundled changes + ~14-query noise (±8 pts) make single-slot attribution near-guesswork.
3. **Stop hard-coding one diagnosis into CLAUDE.md.** Prior reflections lobbied to add
   "rank is healthy, recall is binding" to CLAUDE.md — I'm glad it wasn't, because it's
   only true on ~60% of queries. Better guidance: *"read `judge_verdicts.md` position-by-
   position to classify each low scorer as never-retrieved / mis-ordered / grade-2-capped,
   then pick the lever that matches — don't assume."*
4. **Clarify the results path** (`../../iteration_NNN/agent_<name>/problems/`) up front.
5. **Guarantee ≥N overlapping sample_ids across consecutive batches** so cross-iteration
   attribution is possible at all.

## Cost remains a non-issue — do not optimize it

9 rounds at $0.005–0.015/query vs a $0.033 free-zone AVERAGE. My abstract-fed rerank
adds only input tokens (~$0.012/query total, no new calls, no reasoning). iter5 already
showed $0.015 is comfortably free-zone. Spend headroom on ordering/evidence quality;
cost-saving complexity buys nothing and adds regression risk.

## Open levers for the next instance (priority order)

1. **Validate my rerank bet FIRST.** Diff rank on semantic_104/108/70/203 vs iter6. If
   rank rose → the abstract-fed reranker works; consider a stronger model
   (`GPT_5_4`/`reasoning_effort="low"`, still free-zone) or a two-stage rerank (cheap
   batch → fine per-paper grade of the top ~24 near the K boundary).
2. **If rank is now good everywhere, the residual is retrieval breadth on the
   never-retrieved gold** (semantic_7 type). Add 1–2 more keyword/snippet query
   variants; check `known_good` positions to confirm it's a retrieval gap first.
3. **Metadata/specific (~27%) are still tuned near-blind.** Keep frozen; metadata_15 hit
   1.0 once — don't regress it chasing semantic gains.
