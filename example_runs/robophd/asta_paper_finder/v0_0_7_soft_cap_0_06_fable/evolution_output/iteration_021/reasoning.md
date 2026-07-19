# Iteration 21 — Analysis and Plan

Name: iter21_reverse_cite

Base: `iter20_cite_proof` (iteration-20 winner, 32.79), with its semantic
pipeline **byte-for-byte unchanged**. Like iteration 20, every change is a
deterministic repair on the exact-match paths — zero new LLM calls, zero new
cost (measured mean $0.0346, deep inside the free zone).

## 1. What the iteration-20 data says

### Standings

| Agent | Score | Raw F1 | Mean cost |
| --- | --- | --- | --- |
| iter20_cite_proof | 32.79 | 0.328 | $0.0346 |
| iter13_grade_view | 30.28 | 0.303 | $0.0498 |
| iter18_triage_first | 25.71 | 0.257 | $0.0364 |

iter20 won the batch. Its semantic side (= iter18's, unchanged) took 6 solo
wins; per-query semantic deltas vs iter13 are the usual ±noise (three
consecutive iterations have shown semantic knob-turning does not replicate).
The semantic stack stays frozen again this round.

### The exact-match paths are still where the deterministic losses live

Four of 14 queries were exact-match; iter20 scored 0.000 / 0.043 / 0.052 /
0.222 on them, and the diagnostics attribute each loss to an identifiable
instrument failure, not to noise:

**(a) `get_paper[_batch]` with a `references` field is now dead server-side.**
Every refs fetch in the batch — both metadata problems and the semantic
expand-ref channel — failed with `ToolError: 'NoneType' object is not
iterable` (metadata_31 log: "references unavailable under all field
variants"). Iteration 20's fixes still ran the pipeline *through* this dead
instrument and only fail-opened after the damage:

- **metadata_31 (0.043, gold 16)** — "Journal articles by David Harel ≥10
  citations, citing papers by Gera Weiss, not self-citations". A good
  74-candidate base (Harel's journal papers) was built, then "reference
  verification: 74 -> 0" and the relax ladder blindly submitted 30 pre-ref
  papers: 1 hit. The verification the query needs is *reconstructable
  exactly from a live instrument*: Gera Weiss's ~99 papers each have far
  fewer than 1000 citers, so `get_citations` over his papers yields the
  **complete** set of papers-citing-Weiss; intersecting Harel's filtered
  papers with it answers the query deterministically. ~100 free tool calls.

- **metadata_42 (0.052, gold 70)** — "NeurIPS 2022-23 citing RoBERTa, ≥30
  citations, >3 authors". The 1000-cap on `get_citations` is recency-ordered,
  so channel A cannot reach 2022-23 citers at all; the mention channels
  correctly rebuilt a 72-candidate filtered set — and then "reference
  verification: 72 -> 7". The refs check false-negatived (dead API) and
  iteration 20's new body-mention rescue verified just **1** of ~60 (scoped
  snippet coverage is too sparse to serve as a required gate). Submitting the
  72 would have scored several times higher: with gold 70, even a 40% hit
  rate on 72 submissions is F1≈0.4 vs the actual 0.052.

The lesson generalizes: **verification signals should gate only when the
underlying instrument is complete; when it is provably truncated or dead,
candidates that already passed every observable filter (venue, year,
citations, authors) must be kept, ranked behind the verified ones.**

**(b) specific-path ambiguity hedging picks the wrong hedges.**
- **specific_20 "the cnn paper" (0.222, gold 2 = LeCun-1998 + AlexNet)** —
  the planner's own `candidate_titles` named *both* gold papers, the title
  searches retrieved both, and the submission still missed AlexNet: the
  alternates/hedge slots were filled with alias-*titled* works ("CNN Features
  Off-the-Shelf…") because famous interpretations whose real titles don't
  start with the alias are invisible to the `_alias_titled` hedge.
- **specific_39 "the SPIKE paper" (0.000, gold 5)** — the shortlist held 9
  alias-titled works of 48 candidates, but hedges were taken in *search-rank*
  order; the gold SPIKEs are old classics (4 of 5 gold corpus ids < 14M,
  i.e. old papers), which relevance rank buries. Citation count — not
  retrieved at all on this path (`spec_fields` lacked `citationCount`) — is
  the obvious prior for "which works are known as *the* X paper".

## 2. Changes vs iter20_cite_proof

All in the exact-match paths; the semantic solver, planner, pool, triage,
evidence, tail sweep, venue filter, conjunction machinery, and cost telemetry
are untouched.

1. **Reverse-citation membership verification** (new, free). A shared
   `_citers_of_cid` helper feeds three uses:
   - `cites_author`: fetch citers of up to 150 of the cited author's papers;
     the union is the exact "cites any paper by X" set (complete when every
     list is under the 1000 cap). Candidates in it are marked verified.
   - `author_base + cites_paper` (papers *by* A citing paper P): fetch each
     target's citer list once and mark candidates by membership — iter20 had
     literally no citer channel on this route (channel A only ran when the
     author base was absent).
   - Channel A (unchanged behavior) now also feeds the completeness flag.

2. **Completeness-gated verification, tiered fail-open.** The reference-
   verification step now splits survivors into: tier-0 (citation membership
   proven), tier-1 (refs-verified / body-mention / title-abstract-mention),
   tier-2 (passed every metadata filter but citation status unknowable).
   - If every required citer set was fetched *complete* → keep tiers 0-1
     (exact behavior, as before).
   - If any set is truncated (cap-hit) or unfetchable (dead refs) → keep
     tier-2 too, sorted by citation count, capped (total 160 when
     discriminating filters exist, 40 otherwise). This converts metadata_42's
     "72 built, 7 submitted" into "72 built, 72 submitted".

3. **Wider cap-era probes.** The extra mention probes for capped citer lists
   now also trigger when a *year* constraint puts gold below the recency
   window (metadata_42 had no `min_citations`-only trigger… it did, but the
   probe set gains "fine-tuned X" / "X baseline" and the keyword budget rises
   8→10) — free searches.

4. **Specific-path ambiguity fixes** (free):
   - `citationCount` added to `spec_fields`.
   - On ambiguous aliases the submission is now: verified primary (+
     duplicate records) → **best title-search record of each planner
     `candidate_title`** (the distinct famous interpretations — fixes
     specific_20) → LLM alternates → alias-titled hedge sorted by
     **citation count descending** (fixes specific_39's rank-ordered hedge)
     — capped at 7 (was 8).

## 3. Why this should score higher

- The gains are deterministic repairs with known counterfactuals:
  metadata_31-type → exact reverse-citation intersection (0.04 → plausibly
  0.6+); metadata_42-type → submit the filtered set instead of 10% of it
  (0.05 → ~0.3-0.5); specific_20-type → the second gold interpretation is
  now submitted (0.22 → ~0.4); specific_39-type → citation-ranked hedges.
- Exact-match queries are ~27% of the held-out mix and these same three
  failure shapes (citing-author queries, capped/recency-drifted citer lists,
  ambiguous aliases) recur across batches (metadata_25/26/31/42,
  specific_20/39).
- Cost is byte-identical (no new LLM calls); the semantic side is the
  batch-winning configuration three rounds running.
- Downside is bounded: every change strictly widens or re-orders candidate
  sets that currently score near 0, behind ranking that submits proven
  papers first.

## 4. Verification

- `python -m py_compile agent.py` and an offline stub-harness smoke test
  (`smoke_test.py`, this directory) covering: (a) a metadata_31-style run
  where the reverse-citer set is complete and prunes exactly the
  non-citers; (b) a metadata_42-style run where cap-hit + dead refs keeps
  the tier-2 filtered set instead of dropping it; (c) an ambiguous specific
  run asserting planner-title records and citation-ranked hedges enter the
  submission; (d) semantic-path constants unchanged vs iter20.
- Pre-registered predictions for the next batch: on any "citing papers by
  <author>" query, submitted count ≈ post-filter ∩ reverse-citer set and F1
  beats iter20's on the same query; on any capped "citing <paper>" query
  with year filters, submitted count ≥ 50% of the post-venue-filter count
  (iter20: 10%). Semantic per-query deltas vs iter20 should be pure noise —
  if the semantic mean drops >2 points on the same batch, diff against
  `../../agents/iter20_cite_proof/agent.py` (the diff must touch only
  `_solve_specific`, `_solve_metadata`, and the new `_citers_of_cid`
  helper).
