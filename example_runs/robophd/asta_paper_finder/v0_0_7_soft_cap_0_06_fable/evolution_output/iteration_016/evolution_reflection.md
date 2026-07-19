# Evolution Reflection — Iteration 16 (asta_paper_finder)

## What worked well

**Root-cause accounting before choosing a direction.** The single most
valuable hour was building the failure-decomposition table: for every
grade-3 paper *any* agent found, classify why the winner did or didn't get
credit for it (got it / never retrieved / evidence lost the grade /
stranded below K). The answer — retrieval misses outnumber evidence losses
6:1 and ordering losses 12:1 — inverted three iterations of prior focus.
Iterations 13–15 all worked the two smallest columns while the score
drifted down. Every prior reflection said "read the diagnostics"; the
refinement is: **build a mutually-exclusive attribution of the missed
score first, then work the largest bucket.** Everything else in the session
followed mechanically from that table (`analyze7.py`).

**Cross-agent diagnostics as free ground truth.** Three agents × the same
queries = a natural experiment nobody has to pay for. Two decisive facts
came only from cross-referencing them: (1) the judge is deterministic —
104/104 papers submitted with byte-identical evidence got identical grades,
so evidence differences are signal, not noise; (2) 72 papers flipped
3-vs-not-3 purely on evidence differences, but a paired feature comparison
over those flips showed no usable lexical handle (45–60% agreement on every
feature). That second result killed an evidence-heuristic plan in twenty
minutes that could have burned the whole session, and independently
re-confirmed iteration 14's `calibrate.py` refutation.

**Controlling for query before believing any aggregate.** My first
rank-decile analysis showed the head of submissions scoring *worse* than
the tail — an exciting "ordering is broken!" result that was 100%
cross-query confounding (large-K queries are also easy queries). Redone
within-query, ordering is monotone and fine. Same story for "criterion
coverage anti-correlates with grade." At n=14 queries with wildly different
K, any pooled statistic is suspect; pair or stratify by query always.

**The external S2 API for hypothesis checks.** A couple of `curl` calls to
the public Semantic Scholar API turned vague failure descriptions into
mechanisms: the metadata_25 submissions were all 2025 papers while all
sampled gold was 2022–2024 → `get_citations` is recency-ordered under its
1000 cap; the "SPIKE paper" gold turned out to be five *completely
unrelated* papers that share an acronym → the specific-path ambiguity
hedge is right in principle, the LLM's hallucinated interpretations were
the failure. Minutes each, and they changed what I built.

**Smoke tests that mirror the real parsing paths.** Stubbing `inspect_ai`
and `model_registry` and running 52 checks caught two real bugs pre-ship,
one of which (`_content_words` dropping ≤2-char tokens, collapsing "T5
finetuning" and "RL finetuning" into one dedup key) would have silently
*narrowed* retrieval inside the very change meant to widen it. Bugs that
invert a change's sign are the ones worth a test harness.

**Costing changes against measured telemetry, not vibes.** The `llm-usage`
stdout lines × the price table, calibrated to the cost report's actual
number, let me verify the widened pool nets out cheaper (-12%/semantic
query) before shipping, and revealed the highest-leverage cost cut was not
a depth constant but the *output format* of triage (output tokens bill at
6× input; local 1..N indices + unspaced grades halve output volume).

## What was challenging / time-consuming

- **The 2100-line inherited agent.** Grep the structure outline first, read
  only regions you'll touch, trust the docstring changelog. Whole-file
  reads blow the context budget for nothing.
- **Multi-hundred-line surgical edits.** Removing the gap-fill block was
  easiest with a small Python line-splice with assertions on the boundary
  lines, not a giant Edit old_string.
- **Distinguishing real signal from batch noise.** Scores swing ±5 points
  on identical code across batches (iter12: 45.8 → 39.9 → 33.9). The only
  defense is mechanism-backed changes plus paired per-query comparisons.

## Tools

Filesystem + grep + jq + python one-liners covered everything. The public
S2 API via curl was the one "extra" tool worth using (analysis only — the
agent itself must use state.tools). No strategy_tools existed this round;
the analyze*.py scripts I left in the iteration dir are a de facto
replacement — `analyze7.py` (root-cause attribution) is the one to re-run
first next round.

## What I would do differently

- **Check the intended mechanism actually fires in the next batch.** I
  shipped pool-widening on the strength of measured caps and a projection.
  The first thing iteration 17 should grep is `candidate pool:` lines — is
  the pool actually ~640 now, and did predicted-perfect counts rise? — plus
  `citation intersection:` / `cap_hit` lines if a citing-metadata query
  appears.
- **Budget the planner prompt change more carefully.** Asking for 16
  keyword queries makes the plan call longer and its output bigger; I
  modelled it (+15% in, +50% out on `plan`) but a stronger habit is to
  price every prompt edit at write time.

## Insights about the task itself

1. **Recall of never-retrieved papers is the frontier, not evidence or
   ordering.** 40% of provably-gettable grade-3s were simply never in the
   pool. Pool composition (query diversity × pool cap) is where score
   lives. If iteration 16's widening works, the next lever is likely
   *further* diversity (different query *generators*, not more paraphrases)
   rather than a bigger cap.
2. **The judge is deterministic given evidence text.** Same evidence →
   same grade, always observed. So evidence differences across agents are
   a clean natural experiment, and "judge noise" is not an excuse.
3. **Rank is nearly worthless when recall is low.** score =
   harmonic(rank, recall) ≈ 2·recall whenever rank ≫ recall, which is the
   observed regime (rank 0.35–0.90, recall 0.05–0.26). Reordering passes
   (verify, sim on deep positions) buy almost nothing; spend on pool and
   on grade-2→3 conversion instead.
4. **`get_citations` is recency-ordered under a hard 1000 cap.** On
   heavily-cited targets the sample is the newest tail exclusively —
   "highly-cited papers citing X" queries structurally cannot be answered
   from it alone; mention-search + citation-count ranking is the workaround.
5. **Exact-match query types hide plumbing bugs worth whole points.** Both
   metadata zeros this round were mechanical (conjunction never expressed;
   candidate channel starved). A metadata/specific query scoring ~0 is
   almost always a bug, not a hard query — read its stdout first, it's
   cheap to fix and worth up to 1.0 raw each.
6. **Two independent refutations now exist for lexical evidence
   heuristics** (calibrate.py iteration 14; paired flip analysis iteration
   16). Do not build a third without genuinely new evidence.

## Suggestions for the process / prompts

- **Ship the failure-attribution table as a standing strategy tool.** One
  script: for each semantic query, union all agents' grade-3s, attribute
  each (not_retrieved / evidence_lost / stranded / got_it) per agent, print
  totals. It compresses the highest-value analysis of this session into one
  command and directly names the bucket to work on.
- **State in CLAUDE.md that the judge is deterministic given evidence** and
  that cross-agent diagnostics therefore form a paired natural experiment.
  Also state `get_citations`' recency ordering — both were re-derived here
  at nontrivial cost.
- **Include prior iterations' reasoning.md / evolution_reflection.md paths
  in the prompt explicitly.** The lineage docstrings + reflections saved me
  from at least two dead ends (refuted heuristics, dead `references` API).
  A one-line "read the last 2–3 reflections first" would make that
  reliable rather than lucky.
- **Report raw-F1 and cost-penalty as separate columns everywhere.**
  Iteration 14's "winner" had a lower raw mean than the agent it beat;
  conflating the two terms misattributes what worked.
- **Encourage a per-change cost projection habit** (telemetry × price
  table, calibrated to the cost report). It is ~20 lines of python, and it
  is the difference between "should fit the free zone" and knowing the
  batch mean at four different semantic mixes.
