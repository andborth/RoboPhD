# Iteration 18 Reasoning

Name: iter18-cocite-largegold-v1

Parent: `iter17_ambigunion_landmark_v1`, byte-identical outside the metadata
path (one new channel + its integration, ~130 lines).

## What iteration 17's data shows

**0. Zero tracebacks in the parent** (standing first command). iter12/iter9
show tracebacks on 3 problems but those are the older agents; iter17's batch
is clean — losses are genuine.

**1. Per-type decomposition confirms iter17 as parent with no caveats.**
specific: 0.815 vs 0.333 (iter12) / 0.444 (iter9); metadata 0.377 vs 0.401 /
0.269; semantic 0.374 vs 0.380 / 0.355. The semantic gap to iter12 (0.006)
is far below the measured ~0.02 noise floor; the metadata gap (0.024) is one
query's padding-order noise. The specific gap is huge and traced (below).

**2. Iteration 17's own fixes are now validated in the wild.**
- specific_9 = 1.000: the iter16 author-year channel fired for the first
  time (`author-year channel: 2 names -> 24 identities -> 12 papers near
  2021`) and submitted gold 233231380 solo. Previously "shipped but
  batch-unvalidated" — now validated.
- specific_39 "the SPIKE paper" = 0.444 (solo win): the ambiguous branch
  submitted 4 cross-domain referents, 2 of 5 gold. The landmark/ambiguity
  prompt work is paying.
- semantic_155 (solo loss, 0.244 vs iter12 0.504) is the recorded small-K
  variance shape: same K=14, iter17 found 2 grade-3s vs iter12's 5 on
  differently-planned pools. Not actionable (iteration-11 conclusion).

**3. The one big traced, actionable loss: metadata_25 (large-gold citation
shape), and this iteration's diagnostics finally expose WHY.** The
`score_calculation.md` lists all 159 missed gold ids. Resolving 60 via the
public S2 API:
- Gold = prominent 2022–2024 papers (LLM surveys, OpenVLA, TinyStories,
  RRHF, LightGaussian...) that cite DistilBERT **once in passing**.
- **0/60 mention DistilBERT in title, 1/60 in abstract.** Topical keyword
  search can structurally never reach them.
- Probe: 8 snippet_search variants for "DistilBERT" surfaced only 4/172
  gold in 343 unique papers — snippet relevance favors papers *about*
  DistilBERT, anti-correlated with the prominent-passing-citer gold. The
  mention channel is also a dead end for this shape.
- The only reliable signature is a **citation-graph** property: cites the
  seed + citationCount>50 + year>=2022.

**4. Co-citation probe — the new channel works.** References of a random
100 of the seed's 1000-newest window citers (fetched via the S2 batch API)
contain **26/172 gold** among 1087 distinct 2022+ referenced papers.
Frequency>=2 collapses coverage (4/172) — gold are co-cited rarely, each by
the few window citers sharing its niche — so the channel must keep >=1x
candidates and get precision from ref-verify (existing, validated
machinery). Scaling to ~500 mined citers should reach a large multiple of
26. Mechanics probe-verified: `get_paper_batch` accepts the raw sha
`paperId`s that reference entries carry (references have no corpusId), so
sha -> metadata -> filter -> ref-verify is implementable with existing
plumbing. Wall-clock headroom exists (metadata_25 used 668s of the 1080s
stage deadline).

## The change: a co-citation channel for large-gold citation queries

Gate (identical to the existing large-gold padding gate):
`expected_result_count=="many"` AND seed citationCount>=3000 AND no venue
filter AND seeds resolved AND no author constraints. metadata_42-style
venue-bearing queries and everything else are untouched.

Channel (`_cocite_candidates`, run concurrently with the existing reverse
channel):
1. Mine references of up to 500 in-snapshot window citers (oldest-first —
   2024 citers co-cite the 2022–2023 gold era) via the existing per-paper
   `_fetch_references`.
2. Count frequency per referenced sha (seed excluded); keep the top 1000.
3. Batch-fetch their metadata via `_batch_fetch` (new `raw_ids` flag — ids
   passed without the `CorpusId:` prefix), apply the query's filters.
4. Ref-verify up to 400 filter-passers (by co-cite frequency desc) with the
   existing `_ref_verify`; verified citers join the main submission pool.
5. Unverified filter-passers join the padding pool ahead of mention-source
   pads: their measured gold prior (~26 gold in ~150 filter-passers on the
   probe) is ~10x the mention pads' realized ~2%.

F1 math: gold G=172, N capped at 250, F1=2H/(N+G). iter17 hit 13 → 0.062.
Every verified co-citer is near-certain gold; even +40 hits → ~0.25, +70 →
~0.36. The shape has appeared in 2 of the last 6 batches (metadata_25
twice), both times scored <0.09 by every agent — this is the largest
addressable single-query pool left in the lineage.

## Why nothing else changed

- **Semantic path untouched** (the lineage's best; remaining losses are the
  recorded near-saturated shapes: small-K judge variance, terminology
  starvation, g2-conversion blocked on absent per-criterion verdicts).
- **Specific path untouched** — iteration 17's fixes just validated; churn
  risks regressions (the iter15-gate lesson).
- Cost: the channel is pure tool calls ($0 metered). Parent mean $0.232,
  threshold $0.355 — untouched.

## Validation

`smoke_test.py` (all passing): whole-file `ast.parse` + import; `_batch_fetch`
raw_ids flag exercised against a stub tool (prefix vs raw); the
exec-extracted co-cite counting/capping block over fixtures (frequency
ranking, seed-sha exclusion, null-paperId refs); gate fixtures (venue-bearing
and "one" queries never fire the channel); `_cocite_candidates` end-to-end
against stub state (verified/unverified split, deadline short-circuit).
