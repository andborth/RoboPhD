# Evolution Reflection — Iteration 7 (iter7_simview_breadth)

## What worked well

1. **Cross-agent differential analysis of the judge verdicts, not just scores.**
   The decisive finding this round came from joining `judge_verdicts.md` +
   `submission.json` across *two* agents on the same problem: 41 papers the
   judge graded Perfectly Relevant for iter5 on semantic_87 were completely
   absent from iter6's 250-entry submission. A per-agent view says "iter6
   recall 0.64"; only the cross-agent join says *why* (retrieval-pool variance,
   not evidence quality). Ten lines of Python over two submissions beat any
   amount of staring at score tables.

2. **Auditing the *data flow* of the inherited code against its own logs.**
   iter6's stdout said "rescue round: 30 near-miss papers … rescue promoted 1"
   on problem after problem. Tracing why led to a confirmed mechanical fault:
   the sim graded `_cut(evidence, 700)`, a prefix that ends inside the
   abstract, so the sim (and the rescue re-sim!) never saw a single snippet —
   including the ones rescue had just fetched. The fix (a structured sim-view
   mirroring what the judge reads) is the highest-confidence change I made.
   Lesson: when a mechanism's conversion rate looks absurdly low, suspect a
   plumbing bug before a strategy flaw.

3. **Funding improvements by re-allocating spend, not adding it.** The cost
   penalty is a cliff at $0.06 mean; iter5 already died at $0.0613. Stage-1
   triage (~55% of spend) is the least precision-critical stage — moving it
   to the cheaper Gemini flash-lite handle pays for a 35% bigger pool and a
   richer sim. Every new spend (narrow-query GPT_5_4 verify) is gated on a
   measurable trigger (stage-1 predicted-perfect ≤ 8) so it fires on ~20% of
   queries, and free levers (tool calls) were preferred everywhere: tail
   snippet enrichment costs nothing but upgrades judge-visible evidence.

4. **Inheriting the best agent's file wholesale and editing surgically.**
   The metadata path, relax ladders, unicode handling, and fallback routes
   encode five iterations of failure fixes. Copy + targeted Edits + offline
   smoke test preserved all of it; a rewrite would have silently dropped some.

5. **Offline smoke tests with stubbed `inspect_ai`/`model_registry`.** The
   containment-similarity guard had a real leak ("Deep Learning" ⊂ "Deep
   Learning for X" → 1.0) that the smoke test caught in seconds; I added a
   min-token gate before it ever cost an eval. Also added an import-failure
   fallback for the new model handle — an ImportError at module load would
   zero the whole batch.

## What was challenging / time-consuming

- **Cost projection is still hand arithmetic.** Estimating (pool size ×
  chars/doc × chunk overhead × price) per stage, calibrated against the
  previous iteration's observed totals, took real effort and is error-prone.
  The single most useful infrastructure addition: per-call token/cost logging
  into `agent_stdout` (model, input toks, output toks) so the next instance
  calibrates from data instead of char-count guesses. I added stage-count
  prints but could not add true token counts.
- **Distinguishing architecture from luck.** iter5 vs iter6 differences on
  single problems (±0.1) are mostly stochastic (LLM-planned keyword queries
  differ per run, and pool composition swings with them). I only trusted
  deltas with a mechanistic explanation (absent-from-pool counts, truncation
  bug). The pool-variance finding itself argues most per-problem deltas
  between sibling agents are noise — resist per-problem tuning.
- **Untestable model swap.** Moving stage-1 to GEMINI_3_1_FLASH_LITE could
  not be validated offline (no live handles in the evolution sandbox). I
  bounded the risk (per-chunk parse-failure fallback to mini, import
  fallback), but a tiny "run one real LLM call per handle" harness would
  convert this from calculated risk to verified choice.

## Tools & data: assessment

- The diagnostics are excellent and sufficient: `judge_verdicts.md` +
  `score_calculation.md` + `submission.json` + `agent_stdout` let me separate
  "never retrieved" / "retrieved but buried" / "judged grade-2" precisely.
- No `strategy_tools/` existed. The two scripts worth providing:
  (a) per-problem table of (K, rank, recall, grade histogram, cost) across
  all agents — I rebuilt this by hand for the third iteration running;
  (b) a cross-agent verdict join: for agents A and B, papers judge-Perfect
  for A that are absent/low-ranked in B's list. Both are ~30 lines and would
  save every future instance an hour.
- Beware `evolution_output/iteration_NNN/` vs `iteration_NNN/` at the
  experiment root — the relative paths in the prompt resolve from the
  working directory, and the first `ls` attempt at absolute guesses failed.

## What I'd do differently / advice to future instances

1. **Read the parent's stdout for anomalies first** (conversion rates,
   "promoted 1", zero-counts). Mechanical faults hide behind plausible-looking
   aggregate scores, and one fixed fault is worth more than three new ideas.
2. **Then do the cross-agent Perfect-paper join** before designing anything.
   Where do the other agents' grade-3 papers sit in your best agent's list?
   Absent → work on retrieval breadth. Present-but-low → ranking. Judged
   grade-2 → evidence/rescue.
3. **Treat the sim/judge alignment as the core asset.** The internal sim must
   see exactly what the judge sees (same passages, proportional trims). Any
   asymmetry (prefix cuts, different snippet selection) silently corrupts
   ranking, rescue, and band decisions.
4. **Keep the free levers maxed.** Tool calls cost nothing: bigger pools,
   scoped snippet enrichment (head *and* judged tail), duplicate-record
   scans. The only real budget is LLM grading volume — spend it where the
   score formula concentrates (top-K for small K, band membership for large).
5. **Budget to ~$0.050–0.055 mean, not $0.06**, and re-check the semantic
   share of each batch: the mix shifts, and an 86%-semantic batch raises the
   mean ~$0.01 over a 73% one at fixed per-type costs.
6. **Hedge specific queries when verifier confidence is low.** Gold sets with
   two corpus records of one work have now appeared twice; F1 math makes a
   4-entry hedge (two interpretations × their records) strictly better than
   a confident single guess at conf < ~0.6.

## Suggestions for the process/prompts

- Provide the two analysis scripts above as `strategy_tools/`.
- Ask agents to print per-LLM-call token counts; fold them into a
  `cost_report.md` per stage (planner / triage / sim / rescue), not just per
  model — stage-level attribution is what design decisions need.
- Surface each problem's K in a machine-readable index (it's already implied
  in `score_calculation.md`); K-stratified analysis (small vs large K need
  different mechanisms) is central and currently requires regex scraping.
- A one-shot live smoke harness (one real call per model handle, one call per
  MCP tool) would de-risk model swaps and tool-signature drift cheaply.
