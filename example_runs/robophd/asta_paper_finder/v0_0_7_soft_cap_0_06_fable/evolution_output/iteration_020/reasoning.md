# Iteration 20 — Analysis and Plan

Name: iter20_cite_proof

Base: `iter18_triage_first` (winner of the last two batch scores), with its
semantic pipeline **byte-for-byte unchanged**. Every change this round is a
deterministic fix on the exact-match (metadata/specific) paths or a
transport-robustness fix — zero new LLM calls, zero new cost.

## 1. What the iteration-19 data says

### Standings

| Agent | Score | Raw F1 | Mean cost |
| --- | --- | --- | --- |
| iter18_triage_first | 29.94 | 0.299 | $0.0404 |
| iter12_body_conjunction | 29.12 | 0.291 | $0.0546 |
| iter19_breadth_restored | 26.60 | 0.266 | $0.0398 |

All in the free zone; ranking = raw F1 ranking.

### The pre-registered diagnostic: iteration 19's revert did NOT deliver

iter19 = iter18 + three retrieval reverts toward iter13 (10-query planner,
gap-fill, SNIP_INIT 50), predicted to cut `not_retrieved` to ≤35%. I re-ran
the grade-3 attribution (`attrib19.py`, this directory):

| agent | got_it | not_retrieved | evidence_lost | stranded |
| --- | --- | --- | --- | --- |
| iter12_body_conjunction | 54.1% | 24.9% | 12.6% | 8.4% |
| iter18_triage_first | 53.8% | 28.6% | 8.1% | 9.4% |
| iter19_breadth_restored | 53.8% | 27.6% | 10.5% | 8.1% |

`not_retrieved` moved 28.6→27.6 (nothing like the predicted ≥3.3-point drop),
`got_it` is a three-way tie, and iter19 *lost* the batch by 3.3 points. The
entire iter18-vs-iter19 semantic gap traces to ONE query (semantic_57,
0.796 vs 0.284: iter19's head ordering left 7 of 10 known-3s below K=12 —
plan/verify stochasticity, not stack structure). Without it the two semantic
stacks are a dead tie (0.270 vs 0.268). Conclusion: the iter13-vs-iter18
retrieval-stack question is *resolved as noise*; stop relitigating it. The
semantic side stays exactly iter18's — the configuration that won the batch
score two rounds running and is the cheapest ($0.067/semantic → ~$0.052
projected at the 73%-semantic test mix, safely inside the free zone).

### Where the deterministic losses are: the exact-match paths

Batch 19 had 4 exact-match queries and every agent bled on the same three:

**metadata_42 (0.053): a broken verification instrument dropped 92% of the
candidates.** "NeurIPS 2022-23 citing RoBERTa, ≥30 citations, >3 authors" —
gold has **70** papers. The pipeline built a good 72-candidate set
(post venue/year/citation filters), then "reference verification: 72 → 6".
The refs check is the broken part: `get_paper_batch(references)` returned
reference lists for 67 of 72, but only ~1 contained a RoBERTa match —
i.e. the returned reference lists are truncated or id-less, so the check
produces false negatives at scale (a NeurIPS-2022/23 paper surfaced by a
"RoBERTa <topic>" keyword search almost certainly cites RoBERTa). Six
survived only via the title/abstract-mention fail-open. Submitting the 72
would have scored roughly F1≈0.5 instead of 0.05.

**metadata_26 (0.000, all agents): capped citation intersection + gold-era
drift.** "Papers citing the T5 paper and the spider paper". Both citer lists
hit the 1000 cap; `get_citations` is recency-ordered, so at eval time
(cutoff 2025-06-01) the visible window is Mar–Jun 2025. The gold's 10 papers
are all corpus_id 272M–276M (≈Oct 2024–Feb 2025): gold was computed from an
*earlier* snapshot's recency window that has since scrolled out of reach.
The pure get_citations intersection can never see it. The only channel that
can reach that era is content search (keyword/snippet), verified by body
mentions.

**meta-batch chunk poisoning.** On metadata_42 the metadata backfill
(`get_paper_batch` × 50-id chunks) failed on *both* attempts with "Paper …
is newer than the date cutoff" — one poison id kills all 50 ids' metadata,
and docs with `citationCount=None`/`authors=None` are then silently dropped
by the cheap filters. `_fetch_references` already bisects on chunk failure;
the other three batch sites (`meta-batch`, `_fill_abstracts`, `expand-meta`)
do not.

## 2. Changes vs iter18_triage_first

1. **Body-mention verification via scoped `snippet_search`** (free tool
   calls). New helper: scope `snippet_search(query=<short name>,
   paper_ids=<25 candidates>, limit=100)` and accept a candidate as a
   verified citer iff a returned passage literally contains the cited work's
   short name (normalized, word-bounded). Wired into the reference
   verification step as a third acceptance channel:
   `_cites_target OR refs-verified OR body-mention OR title/abstract-mention`.
   This rescues the 66 candidates metadata_42 threw away. Body mention of
   "RoBERTa" in a paper already retrieved by a RoBERTa keyword search is a
   far more reliable citation signal than S2's truncated reference lists.

2. **Conjunction augmentation when the citer cap binds** (metadata_26-type).
   When a multi-target citing query hits the 1000-cap and the intersection is
   small (<40), add a mention-conjunction channel: keyword searches on the
   joined short names (+topic) and a global snippet search "both A and B";
   then keep only candidates whose body passages mention **every** target
   (per-target scoped snippet verification, intersected). Verified extras
   (capped to 40 minus the intersection size) join the intersection with
   `_cites_target=True`. Downside is bounded: these queries currently score
   0.000, and extras only dilute when the intersection itself has hits — the
   <40 gate plus the cap keeps that dilution small.

3. **Bisect-on-failure for every `get_paper_batch` site.** New
   `_batch_bisect` helper (the `_fetch_references` pattern, factored out)
   used by the metadata backfill, `_fill_abstracts`, and the citation-
   expansion metadata fetch: on chunk error, split recursively, so one
   date-cutoff-violating id costs one id, not fifty.

Everything else — planner, pool, triage, sim, rescue, verify, evidence,
tail sweep, specific path, venue filter, cost telemetry — is untouched.

## 3. Why this should score higher

- **The gains are deterministic, not statistical.** Unlike semantic-side
  knob-turning (three consecutive iterations of ±noise), these are repairs
  of observed instrument failures with known counterfactual scores:
  metadata_42-type queries go from "92% of a correct candidate set discarded"
  to "kept", worth ~+0.4–0.5 F1 on that query alone; chunk poisoning stops
  silently deleting metadata mid-pipeline.
- **Exact-match queries are ~27% of the test mix** (train: 10 specific + 8
  metadata of 66) and citation-constrained metadata queries recur across
  batches (metadata_25/26/42 in recent iterations all lost to exactly these
  two failure modes: capped/recency-drifted citer lists and false-negative
  reference verification).
- **Cost is unchanged**: all new machinery is tool calls (free) — measured
  batch mean stays ~$0.040, far inside the free zone.
- **The semantic side keeps the two-time batch-score winner unchanged**, so
  the downside vs iter18 is bounded at ~0 while the metadata upside is real.

## 4. Verification

- `smoke_test.py` (this directory) — harness-stubbed offline suite: asserts
  the semantic stack's constants and wiring are byte-identical to iter18's
  values, plus scenario tests for the three fixes: (a) a metadata_42-style
  run where refs verification fails but scoped-snippet body mentions rescue
  the candidate set; (b) a metadata_26-style conjunction run where cap_hit
  triggers the mention-conjunction channel and only both-target-verified
  extras are added; (c) `_batch_bisect` salvaging a chunk with one poison id.
- Pre-registered prediction for next round: on any batch containing a
  citation-constrained metadata query ("papers citing X …"), this agent's
  submitted-candidate count after verification should be ≥50% of its
  post-venue-filter candidate count (iter18: 8%), and the query's F1 should
  beat iter18's on the same query. Semantic-side per-query deltas vs iter18
  should be pure noise (mean |Δ| within batch variance) — if the semantic
  mean drops >2 points below iter18's on the same batch, something in the
  shared code was accidentally touched: diff against
  `../../agents/iter18_triage_first/agent.py` (the diff should contain only
  the three fixes).
- `attrib19.py` (this directory) — attribution script repointed at
  iteration_019; re-run against iteration_020 next round.
