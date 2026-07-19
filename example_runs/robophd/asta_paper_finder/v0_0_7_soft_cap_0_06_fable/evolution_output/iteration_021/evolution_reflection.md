# Evolution Reflection — Iteration 21 (iter21_reverse_cite)

## What worked well

**Diagnostics-first, deterministic-fixes-only.** The single most effective
workflow: read `agent_stdout` + `score_calculation.md` + `query.md` for the
worst-scoring problems before touching any code. Every point of improvement I
targeted this round came from a specific, attributable instrument failure
(dead `references` API, capped/recency-ordered `get_citations`, hedge slots
filled in search-rank order) — not from intuition about "better retrieval."
The exact-match paths (specific/metadata, ~27% of the mix) reward this
strongly because their failures are deterministic and their counterfactual
scores are computable from the gold in `score_calculation.md` ("submitting
the 72 pre-verification candidates ≈ F1 0.4-0.5 vs actual 0.05").

**Freezing the semantic stack.** Twenty iterations of history show semantic
knob-turning does not replicate across batches — iter18 vs iter19 vs iter13
per-query deltas were noise, repeatedly relitigated at real cost. Declaring
the semantic side frozen (byte-identical to the multi-round batch winner) and
spending the whole iteration on exact-match repairs is what iter20 did to win
and what I repeated. The diff-discipline framing helps: "the diff vs the base
agent must contain only these named functions" is checkable and prevents
accidental regressions.

**Stub-harness smoke tests.** Copying iteration 20's pattern — stub
`inspect_ai`/`model_registry` in `sys.modules`, monkeypatch `_gen` with
canned JSON, drive `_solve_metadata`/`_solve_specific` end-to-end with fake
tools — caught wiring mistakes cheaply and let me assert behavior ("tier-2
fail-open keeps 15/15") rather than just compile. Worth the ~30 minutes every
round. Make the fake exercise the *specific* failure being fixed (my first
test 2 accidentally routed all docs through tier-1; I had to adjust the fake
data to prove tier-2 worked).

**Free-tool-only changes.** Tool calls cost nothing; LLM calls are metered.
Structuring every fix as pure tool orchestration (reverse citations, tiered
keeps, citation-sorted hedges) means zero cost-regression risk and no
tradeoff analysis needed. The cost budget was never binding for any recent
agent (~$0.035-0.05 vs $0.06 free zone) — don't spend analysis time on it.

## What was challenging / time-consuming

**Server-side API drift.** The biggest surprise in the data: the `references`
field on `get_paper`/`get_paper_batch` died server-side between iterations
('NoneType' object is not iterable, every call). Iteration 20 built fixes
*around* this instrument (probing field variants, bisecting batches) instead
of *replacing* it, and lost most of the intended gains. Lesson: when an
instrument fails at 90%+ rates in the logs, don't harden it — find an
equivalent live instrument (here: `get_citations` run in reverse over the
cited author's papers reconstructs "cites any paper by X" exactly).

**Verification-signal weakness is empirical, not theoretical.** The scoped
`snippet_search` body-mention check sounded decisive on paper; in the actual
run it verified 1 of ~60 candidates that near-certainly cite the target
(numeric citation styles, sparse snippet coverage, passage-ranking
starvation). Reading the actual verified-counts in stdout, not the design
intent, is what exposed this. Any verification gate should be judged by its
measured accept-rate on candidates that are near-certain positives.

**Large agent files.** agent.py is ~2,600 lines. Grep for the function map
first, read only the routes being changed, and edit surgically. Reading the
whole file is a waste of context.

## Tooling notes

- The per-problem diagnostics (`agent_stdout`, `score_calculation.md`,
  `judge_verdicts.md`, `gold_criteria.md`) are excellent — the stdout
  logging discipline of prior agents (printing plan JSON, channel counts,
  filter transitions, tool-failure messages) is what makes iteration
  possible. Preserve and extend it: every new mechanism should print its
  input size, output size, and completeness flags.
- `error_analysis_report.md`'s solo-wins/solo-losses table is the fastest
  triage: solo losses on exact-match queries are almost always deterministic
  bugs; solo wins/losses on semantic queries are usually noise.
- No strategy_tools/ directory existed this run; the interim report's Elo
  table was useful for confirming which agents' ideas had survived.

## What I'd do differently / advice to future instances

1. **Attribute before you build.** For each 0-to-0.1 exact-match query,
   write one sentence: which pipeline stage discarded the gold, and what is
   the counterfactual F1 if it hadn't. Fix only stages with large
   counterfactuals.
2. **Fail open when instruments are incomplete, gate hard when complete.**
   This round's core structural idea generalizes: every filter should know
   whether its evidence source was complete (citer list under the 1000 cap,
   refs fetched, metadata backfilled) and only hard-drop on complete
   evidence. Silent drops from failed fetches were the #1 score killer
   across metadata_25/26/31/42.
3. **Check per-agent stdout for the SAME query across agents** before
   concluding one approach is better — metadata_15's 0.545-vs-0.750 gap
   turned out to be identical code with LLM-classification variance.
4. **The semantic ceiling problem is recall of grade-3s in top-K.** Judge
   verdicts show many grade-2 "Highly Relevant" near-misses (one criterion
   judged Somewhat instead of Perfectly). If anyone re-opens the semantic
   side, the lever is evidence that explicitly covers EVERY weighted
   criterion for head papers — not more retrieval breadth. But bring a
   multi-batch evaluation plan; single-batch semantic deltas are noise.
5. **Re-verify environment facts each round.** CLAUDE.md's tool docs were
   accurate at write time, but the references-API death shows the backend
   drifts. Trust the latest iteration's logs over documentation.

## Suggestions for the prompt / process

- Add to CLAUDE.md: "the `references`/`citations` nested fields on
  get_paper/get_paper_batch currently fail server-side; the only live
  citation instrument is `get_citations`" — with a dated note, since it may
  come back.
- The evolution prompt could ask explicitly for a per-query *attribution
  table* (query → failing stage → counterfactual F1) in reasoning.md; it is
  the highest-value artifact for the next round and forces the
  diagnostics-first discipline.
- Batch-to-batch variance dominates single-iteration comparisons (mean F1
  swung 0.26-0.42 for the same agents across batches). If budget allows,
  evaluating the top 2 agents on a union of two batches would sharply reduce
  false ranking signals that caused several wasted relitigation rounds.
- Pre-registered predictions (started in iter20's reasoning.md) are worth
  keeping as a convention: they let the next instance falsify a fix in
  minutes ("submitted count should be ≥50% of post-venue-filter count")
  instead of re-deriving intent from code.
