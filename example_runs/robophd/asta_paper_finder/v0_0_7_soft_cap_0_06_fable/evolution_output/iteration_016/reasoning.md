Name: iter16_pool_breadth

# Iteration 16 — analysis and plan

## Headline finding: the lineage has been optimizing the two smallest terms

I decomposed every grade-3 paper the iteration-15 batch produced. For each
semantic query I took `known3` = the set of papers that *some* agent got
graded Perfectly Relevant (a strict lower bound on the truly-perfect set),
then asked, for the winning agent iter12, why each one did or didn't count:

| cause | count | share |
|---|---|---|
| got it (graded 3 inside K) | 219 | 49.5% |
| **never retrieved at all** | **178** | **40.3%** |
| retrieved + judged, but evidence lost the grade | 30 | 6.8% |
| retrieved + would-be-3, stranded below position K | 15 | 3.4% |

Retrieval misses outnumber evidence losses 6:1 and ordering losses 12:1.
Iterations 13, 14 and 15 all shipped evidence/ordering work (grade_view,
crit_coverage, cite_inverse) — the 6.8% and 3.4% columns — and the score
drifted down across those rounds. The 40% column has not been touched.

`analyze7.py` in this directory reproduces the table.

## Why the retrieval gap exists: the pool cap throws away 70% of retrieval

Every semantic query's `agent_stdout` says the same thing:

```
candidate pool: 360 of 1143 uniques (per-source: [100 x10, ~47 x5])
```

Retrieval already surfaces ~1000–1150 distinct papers. `POOL_CAP = 360`
discards two-thirds of them *before anything looks at them*, purely because
stage-1 LLM triage is priced per candidate. The 10 keyword lists barely
overlap (1150 uniques from 1225 slots), so the round-robin merge keeps only
the top ~24 of each list. The `not_retrieved` papers above were found by
sibling agents whose planner happened to emit different query phrasings —
i.e. the pool is a lottery over a much larger relevant set, and the fix is
breadth, not cleverness.

## Two supporting findings from the diagnostics

**The judge is deterministic.** 104 papers were submitted by two or more
agents with byte-identical `markdown_evidence`; all 104 got identical
grades. Where evidence differed, 112 of 479 shared papers disagreed (72 of
them 3-vs-not-3). So evidence *is* a real lever — but a paired feature
comparison over those 72 flips found no usable handle: passage count,
average length, truncation rate and criterion-overlap all sit at 45–60%
agreement, i.e. noise. This independently re-confirms iteration 14's
`calibrate.py` result, so I am not building another lexical evidence
heuristic. I take only the free, unambiguously-non-harmful evidence wins
(stop truncating passages; dedupe passages against each other).

**Ordering is genuinely near-exhausted.** Controlling for query, grade-3
rate falls monotonically across deciles of the judged prefix (41% → 19%),
and only 15 papers total were stranded below K. My first cut of this
analysis showed an apparent *inversion* (head worse than tail); that was
entirely cross-query confounding — large-K queries are also easy queries.
Worth recording so the next instance doesn't chase it.

## Plan

Reallocate spend from the exhausted ordering stages into the triage field,
while landing *under* the cost cliff rather than on it.

### Semantic path (headline)

1. **Wider, more diverse retrieval.** Planner emits 16 keyword queries (was
   10) and 6 snippet queries (was 5), and the prompt now demands specific
   *categories* of query — synonym, named method/system, survey, task or
   application, adjacent subfield — so the extra six are new angles rather
   than paraphrases. `SNIP_INIT_LIMIT` 50 → 100 (the tool max; free).
2. **`POOL_CAP` 360 → 640, `POOL_CAP_TOTAL` 420 → 760.** With 22 source
   lists this keeps per-list depth about where it is (24 → 29) and spends
   the increase on new angles, which is what the cross-agent evidence
   supports.
3. **Cheaper per-candidate triage** so the wider field is affordable:
   `T1_TITLE` 110 → 85, `T1_BODY` 170 → 105, `GRADE_CHUNK` 32 → 48 (better
   amortization of the criteria block). ~70 → ~48 input tokens per
   candidate.
4. **Drop the separate gap-fill round.** Measured value across five
   observed firings: predicted-perfect went 1→1, 5→5, 12→12, 19→19, 8→10.
   Its query budget moves into the upfront 16.
5. **Trim the ordering stages** whose marginal product is measured near
   zero: stage-2 sim depth capped at 48 head papers (was ~100),
   `RESCUE_MAX` 24 → 14, verify chunk 6 → 8, and — the significant one —
   the GPT_5_4 head verify now fires only on thin pools (trigger
   `n_perfect <= 10`, was `<= 32`), so it runs on roughly 4 of 9 queries
   instead of all of them.

   The justification for narrowing verify is the score algebra, not just
   cost. Verify is a pure *reordering* pass: it never adds a paper, and
   recall counts grade-3 papers anywhere inside the first K regardless of
   order, so it moves the rank term alone. Since
   `harmonic(rank, recall) ≈ 2·recall` whenever `rank >> recall`, and the
   observed regime is rank 0.35–0.90 against recall 0.05–0.26, the marginal
   value of rank here is near zero. It stays on thin pools, where too few
   grade-3s exist for that approximation to hold.

6. **Compact triage output.** Candidates are numbered locally (1..N per
   chunk) instead of by global pool index, and grades are requested
   unspaced (`7:313`, not `412: 3 1 3`). Triage output is billed at 6× the
   input rate, so on a 740-candidate pool this is worth more than any input
   trim; the parser extracts digits individually, so both forms still
   parse (covered by tests).

### Cost outcome

Modelled from iteration-15's `llm-usage` telemetry × the price table,
calibrated so the model reproduces iteration 15's reported $0.077/semantic
query:

| batch semantic mix | projected mean | free zone? |
|---|---|---|
| 64% (iteration 15's mix) | $0.045 | yes |
| 73% (nominal) | $0.051 | yes |
| 86% (iteration 14's mix) | $0.059 | yes |
| 100% | $0.068 | no |

**$0.068 per semantic query vs iteration 15's $0.077 — 12% cheaper while
triaging 37% more candidates.** That matters because iteration 14 saw *all
three* agents penalized on an 86%-semantic batch; this agent stays in the
free zone there.

### Evidence (free, no LLM cost)

6. Abstract passages 1300 → 2000 chars, snippet passages 600 → 900. The
   scorer imposes no length limit and `_cut` still returns a verbatim
   substring, so this is free; truncated passages were visible in the
   grade-2 side of several flips.
7. Global passage dedup by normalized containment, including against the
   title/tldr/abstract already emitted, so none of the 8 slots is spent
   restating text the judge has already read.
8. Tail evidence sweep extended to position 250 (was 235).

### Metadata path (2 of 14 queries scored 0.000 and 0.010)

9. **Multi-target citation intersection.** metadata_26 ("paper citing the
   T5 paper *and* the spider paper") failed because the planner emitted a
   single `cites_paper_title` holding both titles joined by "; ", so the
   conjunction was never expressed. The planner now emits
   `cites_paper_titles` as a list; with more than one target the agent
   fetches `get_citations` per target and submits the **intersection** of
   the citer sets (falling back to papers citing the most targets if the
   intersection is empty). `get_citations` is the one citation surface that
   works — `references` is dead server-side, confirmed again in this
   round's stdout.
10. **Stop starving high-gold citing queries.** metadata_25 submitted 31
    papers against a gold set of 172 (F1 0.0099). Every one of the 12 I
    checked is a 2025 paper; all sampled gold is 2022–2024. Diagnosis:
    `get_citations` is recency-ordered and hard-capped at 1000, so on a
    paper as cited as DistilBERT the entire sample is the newest tail and
    misses the established papers gold is made of. I can't page past the
    cap, so instead: add citation-count-bearing topical keyword channels
    around the target's short name and title, rank candidates by
    `citationCount` descending, and widen the submission (up to 200) when
    the citer channel hit the 1000-cap and a `min_citations` filter implies
    a large gold set. With gold this large, F1 rewards volume — 15 hits in
    150 submissions scores ~0.09 against the current 0.0099.

## Why I expect this to beat iter12

The change targets the one failure column that accounts for 40% of the
missed score and that no agent in the lineage has addressed, it is funded
by stages whose marginal product I measured at ~zero rather than by new
spend, and it lands further inside the free zone than the incumbent. The
evidence and ordering work I am *not* doing is the work that the last three
iterations did while the score went sideways.

Risks I accept: a wider pool with a slightly thinner per-candidate view
could add noise to stage-1 triage. I keep the full evidence view for the
head, the lexical prescreen for the deep tail, and the existing band
ordering, so a noisier stage 1 degrades ranking (the 3.4% column) rather
than recall (the 40% column) — the right direction to trade.

## Verification

`smoke_test.py` stubs out `inspect_ai` and `model_registry` and runs 52
checks over the changed surfaces: constants, `_cut` verbatim-substring
guarantees at the new evidence caps, `_redundant` containment in both
directions, `_evidence` dedup and grounding (every emitted passage asserted
to be a substring of the doc's retrieved text), `cites_paper_titles`
parsing including the exact semicolon-joined shape that broke metadata_26,
the intersection/fallback semantics, keyword-query dedup, and
`_grade_chunk` parsing of both the new compact and the legacy spaced
formats plus its failure modes. All pass.

Two real bugs surfaced during that pass and were fixed rather than
tolerated:

- `_redundant` missed near-duplicates differing only by punctuation.
  `_norm` rewrites each punctuation run as a space, so "deep learning, for
  NLP" and "deep learning for NLP" differ by a double space and the
  containment test failed. Fixed by collapsing whitespace before comparing.
- The keyword-query dedup key was built from `_content_words`, which drops
  tokens of 2 characters or fewer. "T5 finetuning" and "RL finetuning" both
  reduce to `{finetuning}` and would have collapsed into one query —
  silently *narrowing* retrieval in the change whose whole purpose is to
  widen it. The key now keeps short tokens.

The analysis scripts (`analyze.py` … `analyze7.py`) are left in place;
`analyze7.py` produces the root-cause table above and is the one worth
re-running against the next round's diagnostics.

## Note for the next instance

Two traps in this data, both of which I walked into before checking:

1. **Rank-decile analysis without controlling for query shows the head
   performing *worse* than the tail.** It is entirely cross-query
   confounding — large-K queries are also the easy ones. Within query, the
   ordering is monotone and fine.
2. **Lexical criterion "coverage" of the evidence is inversely correlated
   with grade** (45% grade-3 at zero coverage, 24% at full coverage). This
   is confounded too — papers that need coverage machinery are the weak
   ones — but combined with iteration 14's `calibrate.py` it is now two
   independent refutations of the same idea. Do not build a third lexical
   evidence heuristic without new evidence.
