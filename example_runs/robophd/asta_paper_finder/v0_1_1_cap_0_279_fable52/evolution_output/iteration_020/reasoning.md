# Iteration 20 — reasoning

Name: iter20_year_steered_expansion

Base: `iter19_backdated_citation_counts`. One change, on the metadata path only. The semantic and
specific solvers are byte-identical to iter18/iter19.

## What the data shows

**1. Iteration 19 by query type (iter19 agent, 14 queries).**

| Type | n | Mean score | Mean cost | Notes |
| --- | --- | --- | --- | --- |
| specific | 3 | 1.00 | $0.16 | solved on this batch |
| metadata | 1 | 0.64 | $0.04 | precision 0.875, recall 0.50 |
| semantic | 10 | 0.46 | $0.29 | recall is the loss; rank term is 0.86–0.99 on most |

**2. iter18 vs iter19 on semantic queries is run-to-run noise.** The two agents share the semantic
code, yet per-query scores differ by up to ±0.17 (`semantic_152` 0.64 vs 0.47, `semantic_120`
0.26 vs 0.12, `semantic_140` 0.28 vs 0.40). The 0.36-point gap between them on the batch is not
a signal. A small semantic prompt tweak cannot be judged on a 10-query batch, so I made none.

**3. iter19's back-dated citation filter held up.** On `metadata_42` precision went 0.67 → 0.875
and hits 32 → 35 (score 0.542 → 0.636), in line with the replay that session reported. That is
why iter19 is the base.

**4. Metadata recall is a candidate problem, and the cause is visible.** On `metadata_42`
("NeurIPS papers 2022-2023 that cite RoBERTa, ≥ 30 citations, > 3 authors"), 35 of 70 gold were
missed. I probed 20 of the missed ids: every one is a NeurIPS 2022/2023 paper with 46–6,397
citations and 4–22 authors, so each would pass every filter. They were never candidates. The log
shows why: the venue-filtered expansion found 3,633 NeurIPS papers but only 417 passed the
year/citation/author filters. The search tool has no year parameter, so about 88 % of every
100-hit result page was spent on the wrong years.

**5. A year token in the keyword steers the ranking to that year.** Probe:
`search_papers_by_relevance keyword="neural networks 2023"` with the NeurIPS venue filter returned
82 of 100 hits from 2023 (plus 6 from 2024, 2 from 2022). A second probe with a longer keyword
(`"masked language model representation learning 2022"`) was weaker: 34 of 100 from 2022 and 15
from 2023, and it took over two minutes to return. For comparison the unsuffixed expansion kept
about 11 % of its hits after all numeric filters. So the effect is real but varies with query
length, and whether the year token itself slows the search is unknown (a third probe,
`"graph neural networks 2022"`, ran 125 s and returned no hits — an error or an empty result; stderr
was discarded, so I cannot tell which).

**6. `semantic_145` scored 0 for every agent** ("earliest paper comparing two dialogue systems
…"). K = 6, all six judged papers were "Highly"; gold carries an "Earliest instance" criterion
(weight 0.2) that no evidence text can demonstrate, and the two known-good papers sat at
positions 59 and 63. I found no general fix and left it.

## Change

`year_variants` in `solve_metadata`: when the request bounds the year to an explicit list or to a
range of at most four years, each expansion query is issued once per requested year with the
year appended (`"<topic> 2022"`, `"<topic> 2023"`), instead of once without a year.

- Citing-paper expansion: all ~67 plain topic queries are kept and launched first, so the worst
  case is iter19's candidate set. Up to 100 year-suffixed variants follow, with a 45 s timeout
  each (plain queries keep 60 s). Years rotate across queries so that, when the cap bites, every
  topic still gets at least one year.
- Keyword path (no author / citation structure): the plain queries are kept and up to 40
  year-suffixed variants are added.
- One new log line: `expansion queries: N -> M (years [...])`.

Nothing else changed: filters, back-dated counts, verification order, the 700-candidate
verification cap and the time budgets are as in iter19. No model call was added, so cost is
unchanged.

## Expected effect

On year-bounded metadata queries the share of search hits that survive the year filter should
rise several-fold, so more of the gold reaches reference-list verification. I have no measured
estimate of the score gain. Metadata is about 12 % of queries, and only those with a year bound
and a search-based candidate stage are affected.

## Risks

- **Verification time.** Reference-list verification ran at under one list per second on
  `metadata_42` (417 lists in 460 s). More candidates means the 700 cap and the 21-minute soft
  budget will bind more often. Candidates are verified highest back-dated citation count first, and
  calls return empty after the soft budget, so the run ends with a partial list rather than a
  timeout — but late candidates may go unchecked.
- **More search calls** (about 167 instead of 67) on a path that already logged 25 search
  timeouts. If every year-suffixed call ran to its 45 s timeout, that would add roughly 190 s
  before verification starts.
- **Two of three year-suffixed probes took over two minutes**, and one of those returned nothing.
  In the agent such calls hit the 45 s timeout and contribute no candidates. I did not time the
  same keywords without the year, so I cannot say whether the year token or backend load caused it.
- **The steering effect is measured on two probes** (82 % and 34 % in the requested year). Where
  it is weak, the result is iter19's candidates plus a few.

## Verification

| Check | Result |
| --- | --- |
| pyflakes, `ast.parse` | clean |
| Year steering in the search tool | two probes: 82 and 34 of 100 hits in the requested year |
| Missed gold of `metadata_42` pass all filters | 20 of 20 probed |
| Any agent run with the new code | **none** |

## Not addressed

- Semantic recall (73 % of queries, mean 0.46), including the gap between the agent's own
  "all criteria met" verdict and the judge's.
- Superlative requests such as `semantic_145` ("earliest …").
- The `get_paper_batch` `'NoneType' object is not iterable` errors (154 failures on
  `metadata_42`) that force bisection and slow reference-list verification.
