# Iteration 021 — analysis and plan

Name: iter21_gold_rubric_and_hard_predicates

Parent: `iter19_deep_screen_bulk_evidence` (iteration 020's winner, 39.426).
Two surgical changes, each aimed at a gap I measured rather than guessed at.
The semantic retrieval/mining/ranking machinery is untouched.

---

## 1. What the data actually says

### 1a. On semantic queries, `rank` is saturated and `recall` is the whole game

From `score_meta.json` across iter19's 9 semantic problems:

| | min | mean | max |
|---|---|---|---|
| `rank` (nDCG term) | 0.786 | **0.864** | 0.938 |
| `recall` term | 0.164 | **0.376** | 0.786 |

`score = harmonic(rank, recall)`. With rank already at 0.86, a *perfect*
reranker buys at most ~+0.06 on the harmonic mean. Recall at 0.38 is where the
score is. This independently reproduces iteration 020's oracle-replay finding
(ranking ceiling +0.042, coverage ceiling +0.144) from a different direction,
and it is the reason I did not touch the reranker.

### 1b. …and most of the missing recall is sitting one criterion away

`recall = |{i ≤ K : gᵢ = 3}| / K` — **only grade 3 counts**. Pulling the judge's
own labels out of `judge_verdicts.json` (K = `scored_depth_cap`):

| problem | K | perfectly (g=3) | **highly (g=2)** | somewhat | not |
|---|---|---|---|---|---|
| semantic_104 | 56 | 28 | **22** | 6 | 0 |
| semantic_110 | 228 | 144 | **65** | 13 | 6 |
| semantic_222 | 134 | 22 | **43** | 69 | 0 |
| semantic_189 | 40 | 12 | **11** | 15 | 2 |
| semantic_77 | 12 | 3 | **8** | 1 | 0 |
| semantic_205 | 52 | 12 | 7 | 28 | 5 |
| semantic_174 | 14 | 11 | 2 | 1 | 0 |
| semantic_186 | 68 | 24 | 0 | 43 | 1 |
| semantic_22 | 118 | 19 | 0 | 56 | 42 |

There are **158 grade-2 papers** inside K across the batch, versus 275 grade-3s.
Every one earns exactly zero recall. And grade 2 is not a judgement about the
paper — it is a threshold band. Per the scoring rules,
`weighted = min(1, Σ_c w_c·r_c/3)` with `r_c ∈ {0,1,3}`, and grade 3 needs
`weighted > 0.99`. With the usual gold weights 0.4/0.3/0.3, **one criterion at
"somewhat" instead of "perfectly" gives 0.80 → grade 2 → zero recall.** A
grade-2 paper is a paper whose evidence proved all but one criterion.

Converting even half the grade-2s would be worth, per problem:
semantic_77 0.385 → ~0.69, semantic_222 0.272 → ~0.46, semantic_104
0.625 → ~0.76. That dwarfs anything available on the ranking side.

### 1c. The mechanism: our criteria under-split relative to the gold rubric

`gold_criteria.md` shows the rubric the judge actually scores against. Diffing
it against the `criteria=` line the agent printed:

| problem | gold criteria | agent's criteria | the missing one |
|---|---|---|---|
| semantic_104 | Retrieval-Augmented LMs 0.5 / Model Architectures 0.3 / **Commonality of Architectures 0.2** | RALMs 0.6 / common model architectures 0.4 | the qualifier "common" was folded into criterion 2 instead of standing alone |
| semantic_222 | Multimodal Foundation Models 0.4 / Pre-training on Large-Scale Datasets 0.3 / **Exclusion of Survey Papers 0.3** | MFMs 0.4 / Visual and Audio Inputs 0.3 / Large Scale Pre Training Data 0.3 | the explicit "Please exclude survey papers" clause became no criterion at all |
| semantic_77 | 3 criteria, 0.4/0.3/0.3 | 3 criteria, same split | (matched — and this is the one where the split was right) |

This matters twice over, because `Candidate.evidence()` **spends one evidence
slot per criterion in `plan["criteria"]`**. A gold criterion we never generated
is a gold criterion we never mined a passage for and never quoted — so the
judge reads evidence that cannot prove it, rates it below "perfectly", and the
paper caps at grade 2. semantic_222 having 43 grade-2s and an un-modelled
"is not a survey" criterion is exactly that signature.

The gold rubrics are strikingly regular: **almost always exactly 3 criteria**,
weights from {0.4/0.3/0.3, 0.5/0.3/0.2}, one per atomic concept, with qualifier
adjectives and exclusion clauses promoted to full criteria.

### 1d. On metadata queries the agent is losing on a completely different axis

iter19's metadata mean is **0.209** against a semantic mean of 0.497.
Reading `score_calculation.md` for each:

| problem | query | gold | submitted | hits | score |
|---|---|---|---|---|---|
| metadata_15 | Claire Cardie ACL 2014 or 2017 | 6 | 10 | 6 | 0.750 |
| metadata_26 | citing T5 **and** Spider | 10 | 30 | 5 | 0.250 |
| metadata_42 | NeurIPS 2022-23 citing RoBERTa, ≥30 citations, >3 authors | 70 | **19** | 2 | 0.045 |
| metadata_25 | citing DistilBERT after 2022 with >50 citations | 172 | **30** | **0** | 0.000 |
| metadata_33 | SPLASH 2019+ citing any NeurIPS | 1 | 20 | 0 | 0.000 |

Three concrete, separable defects:

**(i) The numeric predicates don't exist in the plan.** "more than 50
citations", "cited by at least 30 other papers", "written by more than 3
authors" — none of these have a slot in the analysis schema. So they get turned
into *LLM relevance criteria* ("More than 50 citations", weight 0.25) and
handed to a reranker that reads titles and abstracts. **An abstract cannot
reveal a citationCount.** On metadata_25 the agent had 995 in-snapshot
DistilBERT citers in hand, LLM-graded 900 of them, submitted its top 30 — and
hit **0 of 172 gold**. Blind selection would have expected ~5. The LLM screen
did worse than chance because it was scoring topical similarity against a
predicate that is not topical.

**(ii) `CITE_SUBMIT = 30` caps recall below the gold size.** That constant was
tuned on metadata_26 (gold 10), where truncation is genuinely right. But
metadata_25's gold is 172 and metadata_42's is 70: even *perfect* precision
over 30 ids caps F1 at 0.30 and 0.44 respectively. The agent submitted 30 and
19.

**(iii) One un-retried tool error zeroed a whole route.** metadata_42's
`agent_stdout`:

```
[warn] get_citations(198953378) failed: RuntimeError: ... ConnectionRefusedError
citers of 'RoBERTa: A Robustly Optimized BERT Pretr' -> 0
citation path empty -> falling back to keyword/author route
```

A refused connection isn't in the tool's own retry class, and `_citers` gave up
after one attempt. The citation route — the only route that can answer that
query — never ran.

I also confirmed iteration 020's open worry: `grep -rl "pipeline failed"` over
all three agents' problems returns nothing, so iter20's PRF round did not crash
silently. It simply lost on merit (34.7 vs 39.4), consistent with 1a — it spent
itself on the coverage-of-marginal-candidates side and diluted the head.

---

## 2. What I changed

### Change A — generate the rubric the judge actually uses (semantic path)

Rewrote the `"criteria"` section of `ANALYSIS_PROMPT` to reproduce the gold
rubric's construction rules rather than a generic decomposition:

- **Default to exactly 3 criteria**, weights from {0.4/0.3/0.3, 0.5/0.3/0.2}.
- **Qualifiers are their own criterion.** "common", "widely used",
  "large-scale", "recent", "state-of-the-art", "various", "real-world" must
  stand alone, never be folded into a neighbouring label. This is the
  semantic_104 failure, stated as a rule.
- **Exclusion clauses are always their own criterion**, and must be phrased in
  the *positive* vocabulary a qualifying paper's own prose would use ("must
  present original research — proposing a new model, method or dataset and
  reporting experiments — rather than being a survey"). Phrasing matters
  mechanically here: `_window()` selects the evidence passage by term overlap
  with the criterion text, so a criterion worded as a negation retrieves
  nothing, while one worded with "propose / method / experiments" retrieves the
  contribution sentence that proves it.
- Three worked examples, taken verbatim from the observed gold rubrics of
  semantic_104, semantic_222 and semantic_77.

Nothing downstream changed: `Candidate.evidence()` already spends one slot per
criterion heaviest-first and then fills the least-covered gap. This change feeds
it the right criteria, so those slots are spent on what the judge will actually
ask about. It also aligns the stage-2 judge replica with the real judge, which
should help ranking as a side effect.

Cost impact: zero material — it is a longer prompt on one analysis call per
query, ~500 extra input tokens on GPT_5_4 ≈ $0.001.

### Change B — hard predicates decide membership on the metadata path

1. **New plan keys** `min_citations`, `min_authors`, `max_authors`, parsed
   through a `_posint` guard; plus an explicit prompt note that "after 2022"
   means `year_min = 2023` (whereas "2022 and beyond" means 2022).
2. **`_citers` now requests `citationCount` and `publicationDate`**, so the
   predicates are checkable at all, and **retries 3× with backoff** on any
   exception (defect iii).
3. **`_predicate_fail(doc, plan)`** — one deterministic checker for
   year / venue / citationCount / author-count, run over the raw tool docs. A
   field the corpus never returned counts as unknown and *passes*: dropping a
   paper for a missing field costs recall, which is the binding term. It
   replaces the ad-hoc year/venue loop inside `citation_path` and reports a
   per-reason drop histogram to stdout.
4. **`_hard_filters(plan)` gates the truncation.** If the request states at
   least one predicate that was verified against real corpus fields, the
   survivor list *is* the answer set — precision comes from the filter — so the
   agent submits all survivors up to 250 and skips the LLM re-screen entirely.
   If the request states none (a bare "cites A and B", where gold is some
   unseen subset of everyone who cites both), the existing newest-first
   `CITE_SUBMIT = 30` behaviour is kept unchanged, because that is exactly the
   case it was tuned on and it wins there.
5. The same numeric predicates are applied on the keyword/author metadata
   route, via new `Candidate.ncites` / `.nauthors` fields populated in
   `absorb()`, with the same "unknown passes" rule and a guard against
   `authors` being clipped at 12.

metadata_26 is deliberately untouched (no hard predicate → same code path as
before).

### Change B, checked against the corpus — and one honest negative

I probed the DistilBERT citer window directly
(`tool_probe.py get_citations paper_id=CorpusId:203626972 limit=1000`) and
scored my own filter against metadata_25's 172 published gold ids:

```
1000 citers returned; years: 2024 → 389, 2025 → 611, nothing older
corpusId range 235,294,021 … 288,669,284   (995 inside the snapshot ceiling)
gold ids present in the window: 1 of 172
```

**The window is entirely 2024–2025.** metadata_25's gold is the 2023 cohort
(ids 245M–273M), and `get_citations` caps at 1000 with no paging and returns
newest-first, so on a ~10k-citation landmark those papers are structurally
unreachable — exactly the failure mode Domain Background warns about. So
metadata_25 was **not** losable-to-winnable: no filtering strategy recovers it,
and my change moves it from 0.000 to ~0.01, not to 0.6. I had assumed
otherwise when I started the change; the probe corrected it.

What survives that correction, and why I kept Change B anyway:

- Defects (i)–(iii) are real and independent of this one query. Reading
  `citationCount` instead of asking a model to infer it from an abstract is
  unconditionally correct, and the 30-id truncation genuinely caps F1 below
  0.44 whenever gold is large and the window *does* cover it (metadata_15's
  route, and any "author X at venue Y in years Z" query, are covered fine).
- The submit-all rule cannot hurt the unreachable case: hits are ~0 whether the
  agent submits 30 ids or 250.
- metadata_42 now behaves better for a second-order reason. With the retry in
  place its RoBERTa citer window resolves, the year predicate (2022–2023) drops
  a window that is entirely 2024–2025, `citation_path` returns empty, and the
  agent falls through to the venue/keyword route — now *with* the NeurIPS venue
  filter, the year range, `min_citations=30` and `min_authors=4` applied
  deterministically. Previously it reached that route via a crash, with none of
  those predicates checkable.

The generalisable lesson for the next session is in §4.

---

## 3. Why I expect this to generalise

- Change A targets a **structural property of the scorer**, not a property of
  these 14 queries: grade 3 requires every gold criterion proven, gold rubrics
  are consistently 3-way splits, and our evidence budget is allocated per
  criterion. Any semantic query whose phrasing carries a qualifier or an
  exclusion clause is affected, and the training mix is 73% semantic.
- Change B is **deterministic**. It replaces a stochastic LLM judgement about
  an unobservable field with a comparison against the field itself. There is no
  version of "does this paper have >50 citations" where reading `citationCount`
  is worse than asking a model to infer it from an abstract.
- Both changes are **strictly additive to a known-good parent** and confined to
  branches that fire only when their trigger is present. Queries with no
  numeric predicate take byte-identical code paths to iter19; the criteria
  prompt is the only thing every query sees, and it moves in the direction the
  gold rubrics demonstrably take.
- Cost stays in the free zone. Change B *removes* LLM work (the re-screen of
  filtered candidates) and Change A adds ~$0.001/query, so mean spend should
  land at or slightly below iter19's $0.2375 against the $0.355 threshold.

## 4. Risk I did not retire

No live `model_registry` is reachable from this session (only stub modules from
earlier iterations), so `analyze()` could not be exercised against a real model
and I could not verify the new plan keys come back populated. The parsing is
defensive — every new key defaults to `None` and every new branch is gated on
truthiness, so a model that ignores the additions degrades to exactly iter19's
behaviour rather than erroring. Verification was `py_compile` plus an import
audit (`inspect_ai`, `model_registry`, stdlib only — no forbidden backends).

The probe in §2 also reframes how much metadata is worth at all. Of iter19's
five metadata problems, metadata_25 and metadata_42 are gated by the
1000-citer window rather than by agent logic, and metadata_33's gold is a
single paper. The reachable headroom on this path is smaller than its 0.209
mean suggests — which is why Change A (73% of the query mix, 158 grade-2
papers one criterion from counting) is the primary bet and Change B the
secondary one, not the reverse.

**For iteration 022:** the numbers to read first are the new stdout lines
`hard-predicate filter -> N of M (dropped {...})` and `hard filters [...]
verified -> submitting all N survivors`. If those appear on metadata problems
and the scores are still near zero, the predicates are being extracted but the
1000-citer window doesn't contain the gold, and the fix is a second enumeration
route (venue-scoped search + reference verification) rather than more
filtering. If the lines never appear, the analysis model isn't populating the
new keys and the fix is in the prompt. On the semantic side, re-run the
grade-2 count in §1b: if the 158 grade-2s drop materially, Change A worked and
should be pushed further (4-way splits, per-criterion evidence repair); if it
doesn't move, criteria alignment is exhausted and the remaining recall is a
pure retrieval-coverage problem.
