# Iteration 015 — analysis and plan

Name: iter15_verdict_repair

Base: `iter13_balanced_digest_wide_vetting` (best of the draw, 49.888, $0.0471).
Diff: ~140 lines, one new pipeline step, zero new LLM calls.

---

## 1. What the data says

### 1.1 The score is a recall problem, and recall is a *grade-2* problem

Per-query decomposition of iter13's nine semantic queries (`score_meta.json`
joined with the label histogram from `judge_verdicts.json`):

| query | K | grade-3 | grade-2 | grade-1 | grade-0 | rank | recall | score |
|---|---|---|---|---|---|---|---|---|
| semantic_91 | 100 | 31 | **64** | 5 | 0 | 0.644 | 0.310 | 0.418 |
| semantic_110 | 228 | 112 | **57** | 15 | 44 | 0.946 | 0.491 | 0.646 |
| semantic_170 | 204 | 78 | **34** | 67 | 25 | 0.801 | 0.382 | 0.517 |
| semantic_101 | 94 | 23 | **30** | 34 | 7 | 0.721 | 0.244 | 0.365 |
| semantic_193 | 180 | 27 | 22 | 50 | 81 | 0.829 | 0.150 | 0.254 |
| semantic_57 | 12 | 3 | **7** | 2 | 0 | 0.938 | 0.250 | 0.394 |
| semantic_7 | 18 | 2 | **7** | 8 | 2 | 0.937 | 0.055 | 0.104 |
| semantic_148 | 18 | 6 | 4 | 7 | 1 | 0.891 | 0.333 | 0.485 |
| semantic_172 | 198 | 137 | 4 | 50 | 7 | 0.939 | 0.691 | 0.796 |

`rank` averages 0.85, `recall` averages 0.34. For `H = harmonic(r, c)`,
`∂H/∂c = 2r²/(r+c)² = 1.02` against `∂H/∂r = 2c²/(r+c)² = 0.16` — **recall is
worth ~6× rank**, and recall counts grade 3 and nothing else.

The new observation is the grade-2 column. Across the batch there are **229
grade-2 papers inside judged prefixes against 419 grade-3**. Grade 3 requires
`weighted > 0.99`, i.e. essentially *every* weighted criterion judged Perfectly
Relevant; grade 2 is the band immediately below it. **A grade-2 paper is a paper
the judge found on-topic but could not see one criterion demonstrated in the
evidence text.** It earns exactly zero recall.

semantic_91 is the pure case: 100 judged, **0 not-relevant**, 31 grade-3, 64
grade-2. Retrieval was already right. The entire 0.69 of missing recall was
evidence that failed to show a criterion. Converting one grade-2 in four across
the batch is worth ~+0.05 mean recall ≈ **+3 points**.

I read the actual strings for semantic_91 (criteria: *dependency-based models*
0.4 / *relation extraction* 0.3 / *non-local syntactic relations* 0.3):

- grade-2 `85502320` — "shortest dependency path based LSTM". SDP is *literally*
  the mechanism for capturing non-adjacent relations; the evidence never says so.
- grade-2 `44767125` — "Dependency Tree Kernels". Evidence is title + tldr only,
  ~230 chars.
- grade-3 `49544037` — same topic, but its abstract contains the sentence
  *"Dependency trees help relation extraction models capture long-range relations
  between words"*, which demonstrates criterion 3 outright.

The difference between grade 2 and grade 3 here is one retrieved sentence.

### 1.2 The selector five iterations have used for this is nearly uninformative

Iterations 9–14 all chase the same quantity through `_covers`, a stem-overlap
test, and iter13 prints `evidence: N/250 papers cover every criterion lexically`
calling it "the number to move next iteration: it upper-bounds grade-3 count".
Iteration 13's own reflection warned this proxy had never been validated. I
validated it, against 1047 recorded judge verdicts:

```
uncovered criteria = 3  ->  P(perfectly_relevant) = 0.37   (n=794)
uncovered criteria = 2  ->  P(perfectly_relevant) = 0.48   (n=203)
uncovered criteria = 1  ->  P(perfectly_relevant) = 0.53   (n= 45)
uncovered criteria = 0  ->  P(perfectly_relevant) = 0.33   (n=  3)
```

A 0.16 spread, and **76% of judged papers fall in one bucket**, so it barely
partitions anything. The mechanism is obvious in hindsight: *"long-range
dependencies between words"* shares no stem with *"non-local syntactic
relations"*, so the test misses real gaps and invents fake ones. On semantic_91
iter13 spent **355 s of wall clock** (32 of 163 probes timing out at 100 s each)
buying passages chosen by this signal.

I also checked the cheaper hypothesis that short evidence is the culprit. It is a
minor factor only: 13% of grade-2 papers have <600-char evidence against 4% of
grade-3, but median lengths are 2124 vs 2224. Content per criterion, not length.

### 1.3 The right selector is already being computed and thrown away

`_judge_evidence` prompts its graders for **one digit per criterion** — the
prompt says "9 = a sentence in the quoted text explicitly states it … 4-6 =
partial: right area, but the specific requirement is not shown". `_parse_judge`
returns that vector. `_weighted` immediately collapses it to a scalar and the
components are discarded.

Those digits are the opinion of a model that read **the same evidence text the
benchmark judge will read**, expressed on the axis that decides grade 3. That is
a far better answer to "which criterion is this paper short on" than stem
overlap, it costs nothing extra (the grade pass already ran), and acting on it is
`snippet_search` calls, which are **free**.

## 2. The change

**Step 8b — verdict-driven criterion repair**, inserted between the pool grade
and the deep/head re-grades:

1. `_judge_evidence` gains an optional `vec_out` dict, filled with
   `cid -> per-criterion digits`. Return type and all four call sites unchanged.
2. For the top 170 papers by pool vote, take the digit vector. Weak criteria are
   those scoring ≤ 7 ("the text does not show it"). Papers with **1–2** weak
   criteria are near-misses and get probed; papers weak on everything are
   off-topic, and a passage there lifts grade 0 to grade 1, which pays zero.
3. Probe the heaviest weak criterion first (under `weighted = Σ w_c·r_c/3` the
   w=0.4 criterion is what actually blocks the 0.99 gate), one scoped
   `snippet_search` per (paper, criterion), skipping pairs the pre-grade pass
   already bought. ≤260 calls, concurrency 32, 60 s timeout, 1240 s deadline.
4. Rebuild evidence **only** for papers that gained a passage, through the
   existing `_build_evidence` — the same code path the pre-grade repair already
   feeds, so the evidence *format* is untouched and only the raw material grew.
   Repaired papers then get their deep/head second opinion on the *repaired*
   text, so ordering improves alongside evidence.

**Funding it.** The lexical pre-grade repair shrinks (head 170→110, calls
420→200, timeout 100→60 s, deadline 1000→600 s) on the §1.2 measurement. It is
kept as a cheap prior — its passages feed the pool grade that now selects the
real targets — but it no longer owns 355 s of the budget. Projected wall clock
~1000–1100 s against the 1740 s timeout, in line with iter13's 745–1158 s.

**Secondary: metadata never submits an empty list.** iter14's `metadata_33`
("A SPLASH 2019 and beyond paper that cites any NeurIPS") scored a *guaranteed*
0.000 on all three agents with `pool=0` — the planner put both the paper's venue
and the *cited* venue into `venues`, the intersection was empty, and all three
existing fallbacks re-slice a pool that was never populated. The last resort has
to be a fresh retrieval. Added: one keyword search on the query's content words,
venue-filtered then unfiltered, top 20 by citations. One tool call, no LLM, fires
only where the score is already provably 0.

## 3. Why I expect this to score higher

- **It targets the largest measured pool of loss.** 229 grade-2 papers in judged
  prefixes, each one criterion from counting, each currently worth zero recall.
  No other single bucket in the artifacts is that large or that mechanically
  addressable.
- **The selector is measured, not assumed.** The one it replaces separates
  P(perfect) by 0.16 over 1047 real verdicts; the replacement is a grader reading
  the judge's own input on the judge's own axis. Iteration 13's reflection asked
  that a proxy be correlated against recorded labels before being optimised —
  this is that correlation, and it says stop optimising the old one.
- **Cost is unchanged.** No new LLM calls; the digits come from a pass that
  already runs. Projected mean stays at iter13's **$0.0471** against the $0.063
  free-zone threshold. Iteration 14's reflection §6 called cost the binding
  constraint and free levers the highest-value class of change — this is one.
- **The blast radius is bounded and asserted.** `smoke_test.py` loads iter13 and
  iter15 side by side and checks: (1) a *control* that the base cannot express
  the new signal; (2) `_build_evidence` byte-identical on **108 real submitted
  evidence strings** from the recorded artifacts, plus 8 shared helpers textually
  identical — the frozen-evidence rule iteration 12 paid 3.6 points for; (3) the
  near-miss selector's decision table (`[9,9,9]`→no probe, `[2,1,0]`→no probe,
  `[9,9,4]`→probe c3, `[4,9,5]`→heaviest gap first); (4) deadline ordering leaves
  ≥500 s before the 29-minute timeout. All pass.
- **I probed the corpus rather than assuming it.** A live scoped
  `snippet_search` on grade-2 paper `9225214` with the criterion-3 probe returns
  four body passages from *Extended dependency graph*, *Method* and *Syntax based
  arg0 and arg1* sections — real, verbatim, grounding-checkable text that the
  title+abstract-only evidence never had.

**Where it can fail.** If a grade-2 paper genuinely lacks the criterion, the
probe returns weak text and nothing changes (passages are additive; the existing
evidence survives inside the same assembly). The real risk is wall clock, which
is why the pass is deadline-bounded, fails fast at 60 s, and is funded by cutting
a pass measured to be near-useless rather than added on top.

## 4. Left on the table

- **`metadata_26`-class queries** ("cites A and B" on a heavily-cited A) are
  structurally unreachable through `get_citations`' 1000-entry newest-first
  window — iteration 14 proved this by probe. The fix it suggests (intersect from
  the *less*-cited paper and check each candidate's `references`) is real work I
  could not fit in the hour.
- **K > 250 queries** are capped by the scorer's submission limit regardless.
- **`semantic_193`** is the one query whose loss is genuinely retrieval (81
  grade-0 of 180 judged). Different problem, different fix.
