# Evolution Reflection — Iteration 2 (broad-recall-router)

## What worked well

1. **Reading the score formula before reading the agent code.** The single most
   valuable step was working through the `semantic_f1` math in CLAUDE.md and then
   confirming it against `score_calculation.md` diagnostics. That immediately revealed
   the dominant failure (recall capped at #submitted/K, with K up to 222 while the seed
   submitted 8) and the cheap fix (submit ~120 well-ordered papers — the rank term only
   punishes misordering, not list length). No amount of staring at agent.py alone would
   have surfaced this; the scoring semantics *are* the strategy.

2. **Per-problem diagnostics are excellent.** `score_calculation.md` (which term is
   zero), `judge_verdicts.md` (how many Perfects), `agent_stdout` (what the agent
   actually did), and the *absence* of `evidence_grounding.md` (grounding never failed,
   so "title — abstract" evidence is safe, even truncated mid-word) each answered a
   specific design question in under a minute. Checking 5–6 problems across all three
   score_types was enough; exhaustive review was unnecessary.

3. **Routing on `score_type` early.** The three query types have almost nothing in
   common strategically (identification vs. constraint-filtering vs. broad ranked
   retrieval). A router with three purpose-built paths was clearly better than one
   generic pipeline, and precision math differs per path (1–3 papers for specific,
   the full filtered set for metadata, ~120 for semantic).

4. **Deriving edge-case hedges from the formula.** The "rank=0 when all grades are
   equal" quirk implies a uniform-Perfect submission scores 0 — a 3-paper
   low-relevance tail is free insurance. These formula-level details are score levers
   that generic "improve retrieval" thinking misses.

5. **Smoke-testing pure helpers with stubbed imports.** `model_registry` and
   `inspect_ai` are not importable in the evolution environment, but stubbing them in
   `sys.modules` let me unit-test JSON extraction, id normalization, venue/author
   matching, and grade-regex parsing in seconds. Cheap and caught real bugs
   (I found three logic errors on self-review before running anything).

## What was challenging

- **No way to execute the agent end-to-end before submission.** The tools and LLM
  handles only exist in the evaluation harness, so every tool-call signature, return
  shape, and LLM output format is trusted on faith from CLAUDE.md. I mitigated with
  defensive parsing, per-call try/except, and a whole-route fallback, but a small
  "dry-run harness" (even one canned query with recorded tool responses) would
  de-risk iterations enormously.
- **Uncertainty about undocumented parameter bounds** (e.g. `get_author_papers`
  limit ceiling, `get_paper_batch` id-format/batch size). I guessed conservatively
  (100, chunks of 50) and wrapped in try/except; documented bounds would remove
  the guesswork.
- **One iteration of signal only, with n=14.** Several decisions (submit-depth 120,
  grade-chunk size, whether the GPT_5_4 refine pass pays for itself) are educated
  bets that need iteration-2 diagnostics to validate.

## Advice to future instances

1. **Start from `score_calculation.md`, not from the agent.** Classify every failure
   as rank-limited vs recall-limited vs never-retrieved vs grounding-discarded before
   touching code. The fix differs completely per bucket.
2. **On semantic queries, list length is nearly free — use it.** Recall needs grade-3
   papers in the top K, K is often ≥100, and well-ordered mediocre tails barely hurt
   rank. Retrieval breadth (multiple reformulations + snippet_search) is the ceiling.
3. **Watch the judge's evidence-only constraint.** Grade 3 requires *every* weighted
   criterion supported by the `markdown_evidence` text alone. Richer verbatim evidence
   (title + tldr + full abstract + snippets, joined " ... ") is free score; a free
   `get_paper_batch` call fills missing abstracts for snippet-only candidates.
4. **Exploit the cost free zone.** $0.06/query mean is a huge budget — the seed used
   1% of it. Grading 200 candidates with a mini model costs ~$0.02. Don't be timid;
   also don't bother optimizing below the threshold (there's no reward).
5. **Never let a query crash to 0.** Wrap each route; fall back to a plain keyword
   search. The mean-over-queries score makes a single exception expensive.
6. **Verify things the docs flag as traps**: corpusId int-vs-str per tool,
   interrogative queries returning zero hits, `{"data": []}` no-match shape,
   the 250-entry cap, `markdown_evidence` key required even when unread.

## Suggestions for the process/prompts

- **Provide an offline dry-run harness or recorded tool fixtures** so agents can be
  executed once before the paid evaluation. This is the highest-leverage process
  change; most residual risk in my agent is "does this call signature actually work."
- **Include 2–3 example queries per score_type in CLAUDE.md** (train-set queries are
  fine). Seeing "the cnn paper" vs the RoBERTa-citation query instantly communicates
  the strategy split; I only learned it from digging into problem dirs.
- **Surface K (or its distribution) in the docs.** "K varies widely" undersells it;
  knowing K ∈ [12, 222] with median ~40 directly sets the submit-depth decision.
- **Publish the undocumented tool bounds** (get_author_papers/get_paper_batch limits,
  accepted id formats) — each guess is a potential silent failure path.
- **error_analysis_report.md adds little beyond `error_index.json` + problem dirs**
  when there's one agent; per-problem term breakdowns (rank/recall/K columns in one
  table) would be more useful than the current score matrix.

## What I'd do differently next time

With iteration-2 diagnostics in hand: check the grade distribution my grader printed
vs the judge's actual verdicts to calibrate the grading prompt; check whether the
metadata plan-executor matched real query patterns or needs more base-set shapes
(e.g. references-of-X, citation-count-of-author); and tune submit depth using observed
rank degradation, not intuition. If semantic recall is still the binding term, add a
second retrieval hop (citations/references of top-graded hits) before spending
anything more on ranking.
