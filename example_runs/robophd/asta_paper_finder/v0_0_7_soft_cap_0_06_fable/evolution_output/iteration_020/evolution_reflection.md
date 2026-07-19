# Evolution Reflection — Iteration 20 (iter20_cite_proof)

## What worked well

1. **Reading the prior iteration's working directory FIRST.** The single
   highest-leverage ten minutes of the session was reading
   `../iteration_019/reasoning.md` and `evolution_reflection.md`, then
   re-running the inherited attribution script (`attrib18.py`, repointed via
   one `sed`) against the new batch. It answered the central open question
   (did the retrieval revert help?) with a number (`got_it` three-way tie at
   54.1/53.8/53.8%) before I had formed any opinion of my own. The
   "leave a falsifiable prediction + the script that tests it" convention
   established by prior iterations is the backbone of this whole experiment;
   preserve it religiously.

2. **Treating a resolved question as resolved.** Iterations 16–19 spent four
   rounds relitigating the semantic retrieval stack (pool width, planner
   query count, gap-fill), and the accumulated verdict is: all differences
   are batch noise; every widening attempt lost. Accepting that and freezing
   the semantic side byte-for-byte freed the entire session for the paths
   where losses were *deterministic and mechanistically explained* — the
   exact-match metadata queries. Distinguishing "noisy 2-point semantic
   delta" from "0.053 on a query with a known ~0.5 counterfactual" is the
   core skill of this task.

3. **Root-causing from per-problem diagnostics, not summary scores.** The
   decisive evidence was in `agent_stdout` + `score_calculation.md`:
   - metadata_42's stdout line "reference verification: 72 -> 6" against a
     gold of 70 is a complete diagnosis in one line.
   - Comparing gold corpus_ids (272M–276M ≈ Oct 2024–Feb 2025) with the
     agent's submitted ids (277M–280M ≈ Mar–Jun 2025) on metadata_26 proved
     the gold-era drift of the recency-ordered 1000-citer cap — an insight
     no score table could give.
   - Counting "(beyond scored depth — not judged)" lines in
     judge_verdicts.md confirmed judged depth == K exactly (12 to 228),
     which bounds what ordering/tail work can ever pay.
   Corpus_id ≈ ingestion date is a genuinely useful forensic trick; use it
   whenever citation-window questions arise.

4. **Deterministic fixes over statistical tweaks at n=14.** With ±5-point
   batch noise, only two kinds of change are defensible: (a) repairs of
   observed instrument failures (a verification step discarding 92% of a
   correct candidate set; one poison id killing a 50-id batch chunk), and
   (b) provably free improvements (tool calls cost nothing). Everything in
   this iteration is one of those two, which also makes next round's
   attribution trivial: semantic deltas vs iter18 = pure noise baseline;
   metadata deltas = my fixes.

5. **Surgical edits + AST-level diff verification.** Building by copying
   the incumbent 2,400-line agent and applying minimal edits, then
   verifying with an AST function-level diff (`added / removed / changed`)
   that exactly the intended functions changed, is fast and safe. The
   harness-stubbed smoke test pattern (fake `inspect_ai` + `model_registry`
   modules, stub tools returning `Item(json)` objects) let me run full
   `_solve_metadata` scenarios offline that reproduced both observed
   failures and confirmed the fixes rescue them.

## What was challenging / time-consuming

1. **The temptation to keep tuning the 73%.** Semantic queries are 73% of
   the score, so it always *feels* wrong to spend an iteration on the 27%.
   But four rounds of semantic knob-turning produced nothing separable from
   noise, while the metadata path had two never-fixed structural bugs worth
   ~+0.5 F1 each on affected queries. Expected value per unit of change
   risk was clearly on the exact-match side. The prompt could say this
   directly: "check whether the small query classes have deterministic
   failures before tuning the large one."

2. **Verification instruments that lie.** The S2 `references` field returns
   truncated/id-less lists, so "does this paper cite X?" via references
   false-negatives at scale — and the pipeline trusted it as a hard filter.
   The general lesson: before letting any check *drop* candidates, measure
   its pass rate on candidates that are known-positive by construction
   (channel-A citers). A filter that fails known-positives is an instrument
   failure, not information. This cost prior iterations at least two
   queries (metadata_25, metadata_42) before being diagnosed.

3. **Smoke-test stub subtleties.** `_safe_tool(attempts=0)` silently
   disables the call; stubbed citer lists that are identical across targets
   make an intersection test vacuous. Scenario tests need the stub to
   reproduce the *shape* of the failure (disjoint capped citer windows,
   refs that return but don't match), which takes more care than unit
   checks.

## Tooling notes

- The provided artifacts (per-problem `agent_stdout`, `judge_verdicts.md`,
  `score_calculation.md`, `submission.json`, gold ids in
  `score_calculation.md`) are excellent — everything needed for root cause
  was there.
- `error_index.json` still lacks per-problem per-agent scores; I re-globbed
  `problems/*/result.json` like every predecessor. Add
  `by_agent[agent][problem] = score` and a `cached: true/false` flag.
- The headline results table still omits batch query-type mix. One line
  ("10 semantic / 3 metadata / 1 specific") prevents misreading cost
  penalties and win/loss patterns.
- `result.json`'s `raw_query` was null; the query lives only in `query.md`.
  Minor, but it broke my first jq pass.

## What I'd do differently / advice to successors

1. **Keep the two-track discipline:** semantic stack frozen unless a batch
   shows a >5-point *attributed* regression (check the AST diff first —
   if the semantic mean drops, suspect accidental code drift, not fate);
   exact-match paths are where deterministic wins remain.
2. **Remaining known-open items, in priority order:**
   - metadata_26-type conjunction queries: my augmentation is a bounded
     bet (0.000 floor). If it doesn't convert, the honest conclusion may be
     that capped-citation-intersection golds are structurally unreachable;
     stop spending on them.
   - `stranded_below_K` (8–9% of lost credit) is real ranking headroom on
     large-K queries: positions 100–250 are ordered by stage-1 grades only.
     Any improvement must not touch head machinery; consider free signals
     (source multiplicity, lexical criteria overlap) as tie-breakers.
   - `evidence_lost` on huge-K queries (semantic_110: ~24 papers for every
     agent) — the deep tail's evidence is thin; the tail sweep already
     helps, but per-criterion coverage below position 100 is unexplored.
3. **Pre-registered prediction for iteration 20's results** (test it):
   on any citation-constrained metadata query, post-verification retention
   should be ≥50% of post-venue-filter candidates (iter18: 8%), and F1
   should beat iter18's on the same query. If retention is high but F1
   isn't, the candidate *set* is wrong, not the verification — attack
   candidate generation next.
4. **Don't bundle.** The one-retrieval-change-per-iteration rule from
   iteration 19's reflection remains right; this iteration's corollary:
   exact-match-path changes are separable by construction (different
   solver), so they can be bundled freely with a frozen semantic side.

## Process/prompt suggestions

1. Add to the standing prompt: "Judged depth equals K exactly (observed
   12–228). Papers below position K earn nothing — neither rank nor
   recall. On exact-match paths order never matters; F1 =
   harmonic(hits/#submitted, hits/#gold), so with a large gold set,
   discarding plausible candidates is the costliest error."
2. Add: "Before trusting any hard filter, verify it passes candidates that
   are positive by construction. S2 reference lists are truncated —
   body-mention via scoped snippet_search is the reliable citation check."
3. Add: "get_citations is recency-ordered and capped at 1000 with no
   paging; gold for citation queries was built from an earlier snapshot's
   window. Content search + verification is the only channel to older
   citers."
4. Consider running the incumbent + challenger on the same batch every
   round (already done — keep it; the three-agent comparisons are the only
   reason attribution works) and adding one deliberate ablation slot when
   a change is separable.
5. The reflection chain (each iteration reading the last) is working:
   conventions like attribution scripts, smoke-test stubs, and per-stage
   `llm-usage` telemetry survived four generations and compounded. Make
   "read `../iteration_N-1/reasoning.md` and `evolution_reflection.md`
   before anything else" the first line of the task prompt.
