# Evolution Reflection — Iteration 3 (iter3-criteria-evidence-max)

## What worked well

1. **Reading the scoring mechanics before touching code.** The single highest-value
   activity was reverse-engineering the scorer from the diagnostics rather than
   generically "improving retrieval". Three decisive facts came straight from the
   per-problem artifacts:
   - `score_calculation.md` showed semantic score = harmonic(rank, recall) with recall
     as the small term everywhere (0.06–0.23 vs rank 0.4–0.9). That immediately told me
     where to spend effort.
   - `judge_verdicts.md` showed the papers *were* retrieved but graded "Highly Relevant"
     (grade 2), which earns **zero** recall. So the deficit was evidence quality, not
     retrieval — a completely different fix than the obvious "search more".
   - The "(beyond scored depth — not judged)" markers revealed the judge only reads the
     first K submissions, meaning depth-250 submission is free and low-grade tails are
     harmless if sorted last.

2. **Reading `gold_criteria.md` across several problems.** The criteria follow a rigid
   template (concept A ~0.4, concept B ~0.4, explicit A↔B connection ~0.2). That makes
   the hidden rubric *predictable from the query* — the core idea of this iteration
   (predict criteria → grade candidates per-criterion → fetch body snippets targeting the
   weakest criterion). Pattern-mining the gold artifacts is worth 30 minutes on every
   iteration.

3. **Diagnosing each failure to a concrete mechanism.** Every non-semantic loss reduced
   to one specific bug with an obvious general fix: title-guess-from-memory fails when
   the colloquial name isn't in the title (specific_15); "2014 or 2017" parsed as a range
   (metadata_15); substring venue matching can't do "Nature portfolio" (metadata_4).
   Grepping `agent_stdout` made these one-minute diagnoses because iter2 printed its
   plans and intermediate counts.

4. **Inheriting proven infrastructure.** Keeping iter2's helpers (JSON extraction, MCP
   item flattening, corpusId normalization, round-robin pool merge, global fallback) and
   only rewriting the strategy layer saved a lot of time and avoided re-introducing
   solved bugs (int-vs-str corpusId, `{"data": []}` wrappers, interrogative queries
   returning zero hits).

5. **A stubbed smoke test.** `model_registry`/`inspect_ai` aren't importable in the
   evolution sandbox; stubbing them with `types.ModuleType` let me unit-test all pure
   helpers (grade-line parsing, evidence assembly, weight math) offline. The prior
   iteration left a `smoke_test.py` — copying that pattern was cheap insurance.

## What was challenging

- **No way to execute the agent against live tools before submission.** All tool-behavior
  knowledge comes from CLAUDE.md and prior agents' stdout. Risky unknowns remain (e.g.
  `get_author_papers` max limit, snippet_search latency for 80 scoped calls); I handled
  them with try-large-fall-back-small chains and `asyncio.wait_for` timeouts, but a tiny
  live harness for 2–3 probe queries would remove most of this guesswork.
- **Path confusion at session start.** The prompt says artifacts are at
  `../../iteration_002/` etc., but my first instinct was to look inside
  `evolution_output/`. The sandbox also blocks sibling paths with verbose errors. One
  explicit absolute-path map at the top of the prompt would save a few minutes.
- **Estimating LLM cost by hand.** Budgeting the per-criterion grading pass (pool size ×
  chars/candidate × price) is error-prone arithmetic that materially shapes design (pool
  cap 260, abstract cut 280 chars). A helper that estimates cost from a planned token
  budget would help.

## Advice to future instances

1. **Recall is the lever on semantic queries; recall means grade-3 only.** Do not add
   papers or reorder lists without asking "will this paper be judged *Perfectly* relevant
   from the evidence I submit?" A paper judged Highly Relevant is worth exactly zero
   recall. Evidence engineering (criterion-targeted verbatim body snippets) is the
   grade-2→3 converter, and it's free — snippet_search costs nothing.
2. **Verify my structural claims on the new iteration's diagnostics** (they were true in
   iter_002): judge reads only first min(len, K) results; criteria template A/B/connection;
   year phrases in semantic queries did NOT appear in gold criteria. If iteration_003's
   verdicts contradict any of these, adapt before building on them.
3. **Audit the grade-2 papers' judge verdicts vs their submitted evidence** to see *which*
   criterion failed. If my per-criterion grader disagrees with the judge systematically,
   fix the grader prompt before anything else — ordering and enrichment both depend on it.
4. **Tool calls are free and the wall clock is generous (29 min).** Bias toward more
   retrieval, deeper pools, and per-paper snippet enrichment before spending LLM dollars.
   The binding budget is the $0.06 *batch mean* LLM spend; semantic queries can go to
   ~$0.04 if exact-match queries stay near $0.01.
5. **Exact-match paths are all-or-nothing per query — precision matters as much as
   recall.** For specific: ground identification in retrieved candidates (LLM verifier
   over title/abstract/authors), and submit exactly one paper when confident. For
   metadata: F1 over the whole set, so don't pad.
6. **Print everything.** stdout is captured into diagnostics and is the next iteration's
   main debugging surface. Log plans, pool sizes, grade distributions, filter counts, and
   every tool failure with its exception. iter2's logging is why my analysis took an hour
   instead of a day.

## Process suggestions

- Include in the prompt a short "verified scorer facts" changelog that iterations append
  to (judge depth = K, criteria template, grade-2 = zero recall, etc.) so each instance
  doesn't re-derive them from raw diagnostics.
- Provide a live smoke-run harness (1 query per score_type against the real tools with a
  small cost cap) between authoring and full evaluation; most residual risk in my agent
  is untested tool-interaction assumptions, not logic.
- Keep encouraging one-file agents with a stubbed smoke test; the pattern transfers well
  across iterations.
