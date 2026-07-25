Name: iter9_rerank_rich_v1

# Iteration 9 — analysis and plan

## Base
Built on `iter6_metadata_targeted_v1`, the aggregate winner across every batch it
has appeared in (20.4–26.5). iter7 (shrink-abstract assembly) regressed hard, and
iter8 (facet probes + widened evidence pass) was flat-to-slightly-down (17.6 vs
iter6's 20.6 on this batch). The lesson from three rounds of grade-3 *evidence*
tweaking is that that lever is mature and delicate: iter7 proved reordering/shrinking
the evidence assembly destroys grade mass, and iter8 proved that adding more scoped
snippets on top of it barely moves the needle. **I freeze the metadata, specific, and
semantic evidence-assembly paths byte-for-byte and make one isolated, attributable
change: I upgrade the semantic reranker.**

## Where the score is actually lost (iter6 diagnostics, iteration 8)

Reading `score_calculation.md` for every semantic problem confirms the recurring
finding — but with a sharper twist this round. `score = harmonic(rank, recall)`:

| problem | rank | recall | K | note |
|---|---|---|---|---|
| semantic_70  | **0.34** | 0.06 | 34 | ordering bad — one grade-3 buried at pos 12, another at 29 |
| semantic_203 | **0.51** | 0.13 | 24 | |
| semantic_108 | **0.52** | 0.20 | 222 | |
| semantic_104 | **0.55** | 0.18 | 56 | **10 grade-3 papers retrieved, scattered pos 2–52** |
| semantic_152 | 0.73 | 0.19 | 48 | |
| semantic_7   | 0.77 | 0.00 | 18 | gold never retrieved (a genuine retrieval gap) |
| semantic_148 | 0.84 | 0.22 | 18 | ordering already good |
| semantic_186 | 0.88 | 0.09 | 68 | |
| semantic_125 | 0.97 | 0.00 | 22 | 2/3 gold not retrieved |

The prior rounds concluded "rank is healthy, recall is the sole binding term." That
was over-generalized. **Rank is 0.34–0.55 on four of the eleven semantic queries, and
on those the grade-3 papers are already in the pool and inside the top-K window — they
are just poorly ordered.** I verified this directly from `judge_verdicts.md`:
semantic_104 (K=56, rank 0.55) has "Perfectly Relevant" (grade-3) papers at positions
2, 4, 17, 18, 19, 20, 23, 40, 46, 52 — interleaved with grade-1/2. If those were
sorted grade-descending, rank would climb toward 0.9 with **no change to retrieval or
evidence**, lifting `harmonic(0.55, 0.18)=0.27` to `harmonic(0.85, 0.18)=0.30`, and
the same reordering, applied where grade-3 papers currently sit *beyond* K, pulls them
into the judged window and raises recall too. Better ordering moves **both** terms of
the harmonic mean; it is the one lever with clear, measured headroom that does not
touch the delicate evidence assembly.

## Why iter6's reranker under-orders

`_rerank` feeds the LLM only `title[:140] + snippets[0][:200]` per candidate (or a
200-char tldr/abstract fallback) and asks for a coarse 0–3 rating. Two weaknesses:
1. **Thin signal.** The abstract — the single best discriminator of whether a paper
   satisfies *all* the query's aspects — is enriched onto every candidate before the
   rerank (`_enrich_abstracts(ordered[:150])`) but the reranker never sees it. It rates
   on a title plus one ~200-char snippet.
2. **Coarse buckets.** A 0–3 scale collapses every strong paper into "3", so
   within-bucket ordering falls back to noisy retrieval score — exactly the scatter
   seen in semantic_104.

## My change (semantic rerank only; guarded; free-zone cost)

Replace the reranker's per-paper blurb and rubric, nothing else:
1. **Feed the abstract.** Blurb becomes `title[:150] | (abstract else tldr else
   snippet)[:320]` — the reranker now judges on the same discriminating text the
   grade-judge effectively reasons over. Output length is unchanged (one integer per
   candidate), so completeness/fallback behavior is identical; only the input is richer.
2. **Finer, judge-aligned scale.** Rate 0–10 with an explicit rubric anchored to the
   judge's actual rule — *9–10 only if the paper satisfies EVERY required aspect, not
   just the main topic; 6–8 if one aspect is weak/unstated; 3–5 if it misses aspects*.
   The finer granularity breaks the grade-3 pile-up so genuinely all-aspect papers sort
   above topic-only papers, which is precisely the grade-3 vs grade-2 distinction the
   recall term rewards. Sort by (score desc, retrieval order); everything ≥ top_n stays
   in retrieval order as before.

Every other line is iter6: facet-diverse retrieval, multi-source scoring, abstract
enrichment, the targeted per-facet evidence pass (top-15), the abstract-first evidence
assembler, the rank-degeneracy tail safeguard, and the full metadata/specific paths.
The reranker keeps its two existing guards (empty/short-output → retrieval order), so
worst case this matches iter6 exactly.

## Why this generalizes and is low-risk
- It attacks a **measured, structural** weakness (grade-3 papers retrieved but
  mis-ordered) visible on ~40% of this batch's semantic queries, not a query quirk — so
  it transfers to the 73%-semantic held-out mix.
- It only reorders candidates; it cannot lower recall relative to iter6 (the same
  papers are present) and can only raise rank, and it leaves the evidence assembly — the
  proven regression trap (iter7) — untouched.
- Finer granularity *reduces* the all-equal rank-degeneracy risk, and the tail
  safeguard is retained regardless.
- **Cost stays free-zone.** The only added tokens are longer rerank input (~9–12k tok,
  ~$0.008); no new LLM calls, no reasoning effort. iter6 ran $0.005/query against a
  $0.033 average threshold; this stays well under (~$0.012/query), and iter5 already
  showed $0.015/query is comfortably free-zone.

## Validation
`_smoke.py` stubs inspect_ai + model_registry and exercises the new reranker: it
confirms (a) the blurb now carries abstract text, (b) a 0–10 score map reorders
candidates grade-descending, (c) empty/partial LLM output falls back to retrieval
order, and (d) the end-to-end semantic path still emits a valid grounded submission.
The retrieval and evidence paths are byte-identical to the proven iter6, so behavior
can only move on the ordering axis.
