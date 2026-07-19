# Iteration 14 — Analysis and Plan

Name: iter14_crit_coverage

Base: `iter13_grade_view` (iteration-13 winner, 41.76, mean cost $0.0594).

## 1. What the data says

### The batch was hard, and the spread is tiny

| Agent | Score | Cost |
| --- | --- | --- |
| iter13_grade_view | 41.76 | $0.0594 |
| iter12_body_conjunction | 39.92 | $0.0603 |
| iter11_tail_saturate | 39.03 | $0.0621 |

All three agents landed within 2.7 points, and two of the three lost points
to the cost penalty. Two `specific` queries scored 1.000 for everyone and one
(`specific_39`) scored 0.000 for everyone. Everything that separates agents
lives in the eleven semantic queries — matching the 73%-semantic test mix, so
that is where I worked.

`specific_39` deserves one note and no code: the query is "the SPIKE paper"
and the gold is **five different papers that happen to share the name SPIKE**
(a syntactic-search tool, a signaling-pathway database, a spike-train
synchrony method, a protocol-analysis fuzzer, and a banded linear solver — I
confirmed all five titles via the public S2 API). The agent returned seven
plausible SPIKE-ish papers and hit none. This is a genuinely ambiguous
name-collision query, and tuning the specific path toward "return every paper
matching the acronym" would wreck the path that scores 1.000 on every
well-posed specific query. I left it alone deliberately.

### The score arithmetic, re-derived on this batch

`score = harmonic(rank, recall)`, `recall = |grade-3 in first K| / K`. Only
grade 3 earns recall, and grade 3 requires `weighted > 0.99` — i.e.
essentially *every* weighted criterion judged Perfectly Relevant.

Actual iteration-13 numbers, pulled from `score_calculation.md`:

| query | rank | recall | score |
| --- | --- | --- | --- |
| semantic_43 | 0.940 | 0.063 | 0.117 |
| semantic_98 | 0.874 | 0.227 | 0.361 |
| semantic_22 | 0.616 | 0.220 | 0.325 |
| semantic_192 | 0.530 | 0.244 | 0.334 |
| semantic_7 | 0.512 | 0.111 | 0.183 |

Rank is 0.51–0.94; recall is 0.06–0.24. Recall remains the binding term of
the harmonic mean by a factor of 3–15x. Ordering work stays nearly exhausted.

### The pools are already right; the evidence is what fails

This is the finding that shaped the whole iteration. Grade histograms from
`judge_verdicts.md`:

| query | Perfect | Highly | Somewhat | Not |
| --- | --- | --- | --- | --- |
| semantic_7 | 3 | **15** | 1 | 0 |
| semantic_43 | 1 | **7** | 8 | 0 |
| semantic_192 | 20 | **16** | 40 | 6 |
| semantic_98 | 5 | **6** | 9 | 2 |

semantic_7 is the cleanest case in twelve iterations of data: of 19 judged
papers, **15 are Highly Relevant** — topically correct, retrieved, ranked into
the judged region, and one criterion short of the only grade that scores.
Converting those 15 would move recall from 2/18 to ~17/18 and the query score
from 0.18 to ~0.66. Retrieval is not the problem on this query. Evidence is.

Its four criteria are LLMs (0.30), Text Summarization (0.30), Reference-Based
Human Evaluation (0.20), Reference-Free Human Evaluation (0.20). Missing
exactly one 0.20 criterion gives `weighted = 0.80` → grade 2, zero recall. So
the two niche criteria are the entire game, and I checked whether the
submitted evidence even mentions them: **12 of the 15 Highly papers have no
passage containing either "reference-based" or "reference-free" phrasing.**

### Two mechanical defects in evidence assembly

**(a) Duplicate passages burn slots.** `_dedup_snips` dedupes snippets
against each other, but `_evidence` never dedupes them against the title,
tldr, and abstract it already emitted. `snippet_search` returns
title-section and abstract-section passages freely, so these collide. Paper
256416014 on semantic_7 spent 8 slots on 5 distinct texts: its abstract
appeared twice and its title twice. Measured across all 1193 papers with
evidence in this iteration's semantic submissions:

```
total passages: 6308, duplicate/contained: 536 (8.5%)
papers with >=1 wasted slot: 404 (33.9%)
```

A third of all papers — including the top-ranked ones that dominate both DCG
and recall — throw away evidence capacity on text the judge has already read.

**(b) The criterion matcher cannot tell criteria apart.** `_cover_snippets`
scores a snippet against criterion *j* by raw content-word overlap
`len(cw & sw) / len(cw)` with a 0.12 floor. Criterion descriptions share
massive boilerplate — for semantic_7, "reference-based human evaluation" and
"reference-free human evaluation" share *every* content word except one
(`based` vs `free`), all of it swamped by `human evaluation quality text
summarization outputs llms analysis discussion`. Any generic human-eval
passage clears 0.12 for **both** criteria. The matcher therefore reports a
niche criterion as covered while the distinguishing word is absent, and the
slot that should have carried real proof goes to a near-duplicate of the
abstract.

This also explains a long-standing telemetry oddity: internal graders report
`predicted-perfect: 0` and `head verify: 0-3/30 confirmed` while the real
judge returns Perfects on those same papers. The graders aren't merely
strict — they're reading views assembled by a matcher that fills criterion
slots with off-criterion text.

## 2. Changes

Every change below is **free** — pure lexical work plus tool calls. No new
LLM spend.

**1. Global passage dedup (`_dedup_against`).** Snippets are now deduped
against the title/tldr/abstract already emitted and against each other, by
normalized containment in both directions, not just an 80-char prefix key.
Directly recovers the 34% of papers measured above. Applied in both
`_evidence` and `_grade_view`, so internal graders benefit too.

**2. Distinctiveness-weighted criterion vocabulary (`_crit_vocab`).** Each
criterion's words are weighted by how *unique* they are across the query's
criteria: a word in one criterion scores 1.0, a word shared by *k* criteria
scores 1/k, and words appearing in every criterion score near zero. Snippet
scoring switches to this weighted measure, so `free` and `based` decide the
reference-free/reference-based assignment instead of the shared boilerplate.

**3. Coverage requires a distinctive hit.** A snippet is only credited to
criterion *j* if it contains at least one word that is genuinely distinctive
to *j* (weight ≥ 0.5), not merely 0.12 of its generic vocabulary. A criterion
with no qualifying passage is now reported *uncovered* rather than falsely
satisfied — which is what makes change 4 meaningful.

**4. NOT SHIPPED — and the negative result is the most useful thing here.**
The plan was to put `_cov_score` (weighted fraction of criteria whose
distinctive vocabulary appears in the submitted evidence) into `_key2`, on
the theory that it mirrors the judge's grade-3 rule and that the LLM graders
are miscalibrated. Before wiring it in I tested it against ground truth —
`calibrate.py` scores all **1208 judged papers** of iteration 13 and groups
mean coverage by the judge's actual grade:

```
thr=0.34   Not=0.287  Some=0.332  High=0.492  Perf=0.405
thr=0.25   Not=0.466  Some=0.546  High=0.653  Perf=0.600
thr=0.45   Not=0.124  Some=0.138  High=0.256  Perf=0.195
```

**Highly scores above Perfect at every threshold from 0.20 to 0.50.** Ordering
on this would have systematically promoted grade-2 papers above the grade-3s
that are the only papers earning recall — an actively negative change, and
one whose story ("measures what the judge measures, on the judge's own
input") was clean enough that I would have shipped it on reasoning alone.

I checked the obvious alternatives too, and none separate the grades either:
passage count (Somewhat 5.24 vs Perfect 5.33), evidence length (2802 vs 2882
chars), submitted position (Somewhat 75 vs Perfect 85 — mildly *inverted*).
The Perfect-vs-Highly distinction is genuinely semantic, and no cheap lexical
proxy over the submitted evidence captures it. `_cov_score` survives as
telemetry only, `calibrate.py` ships alongside the agent, and both the
docstring and `_key2` carry the refutation so a successor doesn't rebuild it.

This also recalibrates how much changes 1–3 can claim. They make evidence
*denser and better targeted*, which is defensible on the documented grade-3
rule and on the one feature that does move monotonically with grade (distinct
passage count: 4.23 Not → 4.90 Perfect). They are not a ranking improvement,
and I am not claiming one.

**5. Criterion-conjunction retrieval.** Three extra pool-building snippet
queries pair the lowest-weight criteria with the heaviest one. Low-weight
qualifier criteria are exactly what gate grade 3, and a pool retrieved by
main-topic queries systematically under-samples papers that satisfy them —
which is why semantic_7's pool came back 15 Highly and 3 Perfect. These
phrases live in method and evaluation sections, which is `snippet_search`'s
territory; keyword search sees only title/abstract surface. Free, and with
recall binding the harmonic mean, pool composition is the lever with the most
room left.

**6. Cost trim for margin.** iter13 sat $0.0006 under the threshold; iter12
and iter11 both crossed it and paid. Stage-1 triage is $0.030/query, over
half of all spend. `T1_BODY` 170→150 and `SIM_DEPTH` 55→48 recover roughly
$0.004/query without touching any lever above. Target ≈ $0.055, a real
buffer rather than a coin flip.

## 3. Why this should score higher

I want to be straight about the size of the claim, because the calibration
step cut it down midway through.

**What is solid.** The dedup defect is measured, not inferred: 536 of 6308
passages and 33.9% of papers, including the top-ranked ones that dominate
both DCG and recall. Fixing it strictly increases the distinct text the judge
sees per paper, and distinct passage count is the one feature that does rise
monotonically with grade (4.23 Not → 4.69 Somewhat → 4.76 Highly → 4.90
Perfect). The cost trim is also solid and directly worth score: two of three
agents last round paid a penalty, iter13 cleared the threshold by $0.0006,
and landing near $0.055 turns a recurring coin flip into headroom.

**What is reasoned but unproven.** Changes 2, 3 and 5 rest on the documented
grade-3 rule — evidence must demonstrate *every* weighted criterion — plus
the observation that the matcher provably cannot separate semantic_7's two
deciding criteria. I believe change 5 (conjunction retrieval) is the largest
of these, because recall binds the harmonic mean by 3–15x on every query in
the batch and pool composition is the least-exhausted lever left. But I have
no ground-truth measurement that these convert grade 2 into grade 3, and the
refutation of change 4 is a warning about exactly that kind of confidence.

**What I removed.** The change I expected to be the headline was refuted by
its own calibration against 1208 judged papers, and shipping it would have
cost score. Finding that before the eval round is the main result of this
iteration, and it is why `calibrate.py` ships with the agent.

So my expectation is a modest gain — dedup and better-targeted evidence
compounding with a conjunction-retrieval channel aimed at the criteria that
gate grade 3, plus roughly half a point recovered from cost — rather than the
step change that change 4 would have promised on paper.

Residual risks, in order of concern:

- **Changes 2/3/5 may simply not convert.** The Perfect-vs-Highly boundary is
  semantic, and I have shown that lexical proxies don't track it. These
  changes improve what the judge *reads*; whether that flips its verdict is
  untested. They are, at least, free and structurally hard to make things
  worse with.
- **Stricter coverage crediting could shrink evidence.** It cannot: an
  uncovered criterion falls through to the fill-with-the-rest pass, so the
  same passage count ships — the passages are just no longer mislabeled as
  proof of a criterion they don't state.
- **Three extra snippet queries dilute the round-robin merge**, taking ~16%
  of pool slots (3 of 19 source lists). That is the intended trade — targeted
  breadth against main-topic depth — but if the conjunction queries return
  noise on some query shapes it costs pool quality. `_cut(q, 300)` and
  `attempts=1` bound the wall-clock exposure.
- **The T1_BODY trim could coarsen triage.** 150 chars still captures the
  abstract head a 0/1/3 per-criterion triage actually uses, and rank (which
  triage drives) is the term with the least to lose.

## 4. Artifacts

`smoke_test.py` — 41 offline checks, all passing, with `inspect_ai` and
`model_registry` stubbed: dedup against the title/abstract, the
reference-based/reference-free pair being decidable on the real semantic_7
criteria (each passage beats the other by ~2.5x on its own criterion), the
≤8-passage verbatim-grounding contract, `_cover_snippets` slot accounting,
degenerate inputs, and a regression guard asserting `_cov_score` stays out of
`_key2`. It caught two real bugs during this iteration: a distinctiveness
threshold that admitted words shared by two of four criteria, and a coverage
floor set high enough (0.34) to reject a passage scoring 0.258 on its own
criterion.

`calibrate.py` — scores every judged paper's submitted evidence against its
query's real criteria and reports mean coverage per judge grade over a
threshold sweep. This is what refuted change 4. **Run it before trusting any
lexical proxy for the judge's grade.**
