# Iteration 5 — Analysis and Plan

Name: iter5_cite_verify_deep_evidence

## What the iteration-4 data says

Best agent: `iter4_judge_sim_ranker` (21.4), all agents in the cost free zone
($0.013–$0.032 vs $0.06 threshold). iter4 won or tied 9/14 problems and had
zero solo losses — its architecture (route on score_type; semantic = predict
criteria → broad retrieval → per-criterion triage → snippet enrichment →
judge-simulation rerank) is the right base. The losses decompose cleanly:

### Semantic (73% of the mix): recall-bound, and grade-2 papers are the loss

`score_calculation.md` for every semantic problem shows rank 0.40–0.86 but
recall 0.05–0.38 — recall is the binding term everywhere. Tabulating
`judge_verdicts.md`:

| problem | judged (=K) | Perfect | Highly | Somewhat/Not |
|---|---|---|---|---|
| semantic_104 | 56 | 3 | **45** | 8 |
| semantic_160 | 30 | 4 | **21** | 5 |
| semantic_2 | 114 | 18 | 13 | 83 |
| semantic_100 | 162 | 21 | 21 | 120 |

Two distinct loss modes:

1. **Grade-2 saturation** (104, 160, 70): dozens of papers judged "Highly
   Relevant" — worth *zero* recall. One weighted criterion (usually a
   qualifier like "must discuss *common/widely-used* architectures" or
   "must *explicitly connect* concept A to B") wasn't demonstrated by the
   submitted evidence. This is partly evidence engineering (the judge reads
   only `markdown_evidence`) and partly retrieval targeting (for
   "common approaches" queries, *surveys* satisfy the qualifier naturally;
   iter4's pool was dominated by application papers).
2. **Pool starvation** (2, 100, 7): the judged head contains many
   Somewhat/Not — not enough genuinely-relevant papers retrieved to fill
   K=114–162 positions.

iter4 enriched only the top 80 with ONE combined snippet query per paper.
K reaches 162 — positions 80–162 went in with abstract-only evidence.

### Metadata: "cites X" queries are structural zeros

- metadata_25 ("citing DistilBERT after 2022, >50 citations", gold 172) →
  0.010: `get_citations` caps at 1000 of DistilBERT's ~10k citations
  (recency-skewed, so almost none have >50 citations yet). 26 submitted,
  1 hit.
- metadata_42 (NeurIPS 22–23 citing RoBERTa, ≥30 cites, >3 authors, gold
  70) → **submitted 0 papers** (filters emptied the 1000-cap subset; the
  relax fallback sat after the wrong stage).
- metadata_31 (Harel journal papers ≥10 cites citing papers by Gera Weiss,
  not self-citations, gold 16) → iter4 ignored the cites-author constraint
  entirely (submitted all 211 Harel papers: recall 1.0, precision 0.076).
  iter2's LLM guess got 0.524. The true filter is mechanical: candidates'
  `references` ∩ Weiss's paper ids.

### Specific: duplicate corpus records + unicode

- specific_20 ("the cnn paper"): gold = BOTH AlexNet records (NeurIPS 2012
  + CACM 2017 re-publication). iter4 confidently submitted LeNet only → 0.
  iter2 submitted 3 interpretations → 0.4. Lesson: gold can contain multiple
  corpus records of one work, and ambiguous aliases warrant multiple
  interpretations.
- specific_9 ("the MS^2 DeYong2021 paper"): iter4 searched for the literal
  "MS²" (superscript) and hallucinated titles → 0. Needs unicode/superscript
  normalization and a raw-alias search channel.

## The new agent: iter5_cite_verify_deep_evidence

Keep iter4's skeleton and robustness (it had zero crashes); make targeted
upgrades per route.

### Semantic path

1. **Wider pool** (10 keyword variants + 3 snippet queries, cap 340), with
   the planner instructed to add survey/review-phrased variants when the
   request asks about approaches/solutions/landscape (surveys convert
   breadth-qualifier criteria to grade 3).
2. **Gap-fill retrieval round**: if stage-1 triage predicts < 25 all-perfect
   candidates, ask GPT_5_4 for 5 fresh queries targeting all criteria
   jointly (alternate terminology / method names), search, triage the new
   candidates, merge. Attacks pool starvation on hard queries only.
3. **Deeper, per-criterion enrichment**: enrich the top 150 (was 80); the
   top 70 get one scoped `snippet_search` per weak criterion (up to 2,
   weighted-weakest first) instead of one diluted combined query. Tool calls
   are free; concurrency 8 respects the rate limit.
4. **Judge simulation over the whole enriched head 150** (was 80) on the
   exact evidence to be submitted, with stricter wording about qualifier
   criteria ("common/widely-used", "explicitly connects"). Rank
   predicted-all-perfect first, then by weighted grade.
5. Evidence assembly unchanged (title + tldr + abstract + snippets, verbatim
   `_cut` at whitespace — zero grounding failures observed across 42 runs).

### Metadata path

1. **Plan schema extended**: `cites_author`, `exclude_coauthor` (self-citation
   exclusions), `require_journal`; "after YYYY" interpreted inclusively
   (metadata_25's gold includes year-2022 papers for "after 2022" —
   preprint/publication year drift makes inclusive strictly better on F1).
2. **Cites-paper queries get three channels**: (A) `get_citations` (1000);
   (B) relevance searches on the cited paper's short name (+ venue filter
   when constrained); (C) `snippet_search` for the short name (body mentions
   ≈ citations). Channels B/C are **verified by fetching candidates'
   `references`** (batched, free) and checking the target's
   paperId/corpusId. This recovers the highly-cited citing papers the
   1000-cap can never return.
3. **Cites-author**: build the cited author's paper-id set
   (`get_author_papers` across split profiles), keep candidates whose
   references intersect it; drop candidates the cited author co-wrote when
   the query excludes self-citations. Turns metadata_31 from 0.14/0.52
   toward ~0.9.
4. **Relax ladder that can never submit 0**: progressively drop reference
   verification → venue LLM strictness → min-citations, and finally submit
   a small best-effort set (metadata_42's 0-submission is a guaranteed 0).

### Specific path

1. **Unicode/superscript normalization** ("MS²"→"MS2") on all search inputs;
   extra channels: relevance search on the raw de-templated query and on the
   top-2 candidate titles (this is what surfaces duplicate records).
2. **Always verify with GPT_5_4** (fast path removed — one ~$0.005 call),
   with a schema that returns **all indices that ARE the exact paper**
   (duplicate/re-published records) plus alternate interpretations; submit
   alternates scaled by confidence (0/1/2 extra), total cap 5. Submitting
   both records of a duplicated work protects against the pick-the-wrong-
   record zero (0.667 ≥ 0) and captures multi-record golds (specific_20 →
   potential 0.8–1.0).

### Cost

Semantic worst case ≈ $0.055 (stage-1 triage of ~340 + stage-2 sim of 150 on
GPT_5_4_MINI + one or two GPT_5_4 planning calls); metadata/specific ≈ $0.01.
Weighted mean ≈ $0.04 — inside the free zone with margin for variance.

### Why this should beat all three parents

- It is a superset of iter4 (the best agent) with each observed zero/near-zero
  addressed by a mechanism, not a prompt tweak: reference-verification fixes
  two structural metadata zeros; record-set submission + normalization fixes
  both specific zeros; deeper per-criterion enrichment + survey targeting +
  gap-fill attack the two dominant semantic loss modes (grade-2 saturation
  and pool starvation) that account for most of the remaining headroom.
- All changes generalize: they key on query *structure* (cites-X, alias
  references, qualifier criteria), not on memorized problems.
