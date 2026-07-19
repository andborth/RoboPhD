# Evolution Reflection — Iteration 9 (PaperFindingBench)

## What worked well

1. **Reading per-problem diagnostics before touching code.** The single most
   valuable artifact was `judge_verdicts.md` + `score_calculation.md` side by
   side across agents for the *same* problem. That is how the decisive scorer
   mechanic was confirmed: the judge grades exactly the first K submitted
   entries (K = the hidden recall denominator, observed 6–232), and only
   grade-3 papers earn recall. Everything below position K is inert. This one
   fact reframes the whole task as "get true grade-3s into the top ~15 on
   narrow queries" and should be stated plainly to future instances (it now is
   in CLAUDE.md, but earlier iterations spent effort discovering it).

2. **Treating recurring all-zero problems as structural, not noise.** Three
   queries (specific_39, metadata_33, semantic_77) zeroed *every* agent in
   *both* iterations 7 and 8. Cross-referencing two iterations of results
   distinguished "hard query, noisy judge" from "my architecture has no code
   path for this query shape" (ambiguous aliases with multi-paper gold sets;
   cites-a-venue constraints; transport 502s starving the pool). Fixing shapes
   generalizes; chasing single-iteration deltas often does not.

3. **Grafting mechanisms rather than picking a whole winner.** The leaderboard
   said iter6 > iter5 > iter8, but the per-problem table showed each agent had
   solo wins with identifiable causes. Basing on the incumbent and grafting
   only the *mechanism behind* each rival's wins (iter8's narrow verify and
   retry wrapper) is lower-variance than either rewriting or wholesale
   adoption.

4. **Bounded-downside design.** Every change was gated (only triggers on
   conditions that were losses before) or promotion/boost-only (cannot demote
   an already-good ordering). With 14-sample iterations and heavy judge noise,
   un-gated changes are nearly impossible to attribute; gated ones at least
   can't regress the cases you were already winning.

5. **Cheap verification of the artifact.** `py_compile` plus a stubbed
   `model_registry` import and a handful of asserts on pure helpers (alias
   matching, venue umbrella matching) caught nothing this time but costs ~2
   minutes and has high expected value — a syntax error or bad closure would
   zero the entire batch.

## What was challenging

1. **Noise dominates small deltas.** The same paper with near-identical
   evidence graded "Perfectly" for one agent and "Highly" for another
   (semantic_57, corpus 276408140). Mean F1 swung 0.30↔0.48 across iterations
   purely from query sampling. Implication: per-problem score differences
   under ~0.1 between agents are mostly unattributable; only mechanism-level
   explanations (verified in stdout traces) are worth acting on. Evidence
   micro-tuning is likely a dead end past a point already reached.

2. **Context budget for reading agent code.** By iteration 9 the agents are
   1300–1800 lines each and mostly identical. Reading two full agents plus
   diagnostics consumed a lot of context. A per-agent CHANGELOG (each agent's
   docstring did this well — keep that convention) plus a diff against its
   parent would be much cheaper than re-reading whole files.

3. **The write-scope sandbox.** `/tmp` is unwritable; a scratch-file habit
   from normal work fails once before you adjust. Minor, but the prompt could
   state it more prominently (it does mention it — I still hit it once).

4. **stdout truncation/`head` mishaps with heredocs.** Multi-command bash with
   `echo ====` separators occasionally collided with zsh globbing (`(eval):1:
   ==== not found`). Quote separators or use printf.

## Tooling notes

- `jq` over `submission.json`/`error_index.json` was essential and fast.
- No `strategy_tools/` existed this round; the manual workflow (score table →
  solo wins/losses → per-problem verdict diff → stdout trace) was fine but is
  exactly what a helper script should automate. A script that emits, per
  problem: query, score_type, K, per-agent (rank, recall, first-5 verdicts),
  and gold-vs-submitted diff would cut analysis time by half.
- The evaluator's diagnostics are excellent. `judge_verdicts.md`'s
  "(beyond scored depth — not judged)" marker is the load-bearing detail.

## What I would do differently

1. **Budget more of the free-zone cost headroom deliberately.** The best agent
   ran at $0.0425 against a $0.06 free threshold. That is ~30% unused budget
   that could buy a GPT_5_4 verify on *every* semantic query or a second
   independent triage vote to average out grader noise. I widened one gate;
   an explicit "spend up to $0.058" design stance would be better.

2. **Attack recall breadth harder.** On many semantic queries recall, not
   rank, is the binding term (recall 0.08–0.29 with decent rank). Known-good
   papers auto-grade Perfect, and several losses traced to the gold/known-good
   paper simply never entering the 340-doc pool. More diverse retrieval
   (citation-graph expansion from confirmed-perfect papers: their references
   and citers are prime grade-3 candidates) is untried and free.

3. **Log richer stdout.** Iterations that printed plans, per-stage counts, and
   failure reprs were far easier to learn from than early ones. Print the
   final top-10 with predicted grades so the next instance can compare
   predicted vs actual judge verdicts directly.

## Suggestions for the process/prompt

1. **Carry forward a cumulative "confirmed facts" file.** Each iteration
   rediscovers scorer mechanics from diagnostics. A `facts.md` at the
   experiment root — appended by each instance, with evidence pointers —
   (judge depth = K; known-good auto-Perfect; 502 not auto-retried; gold for
   ambiguous "the X paper" can contain 5 ids; venue strings for umbrella
   conferences) would save most of the analysis phase. CLAUDE.md now encodes
   much of this, but agent-discovered facts should have a designated home too.

2. **Persist a regression list.** The three all-zero queries recurred across
   samples. A shared file listing "query shapes we currently score 0 on, and
   why" would focus each iteration on marginal structural coverage instead of
   re-deriving it from two iterations of tables.

3. **Report K and score_type in the cross-agent table.** The error analysis
   report compares raw scores but omits K; a 0.15 on a K=232 query and a 0.15
   on a K=6 query need opposite responses (breadth vs head precision).

4. **Encourage per-mechanism attribution in reasoning.md.** "Agent A beat
   agent B on problem P because mechanism M (see stdout line …)" is the only
   analysis that survives sampling noise. The evolution prompt could ask for
   this explicitly.

5. **Two-batch evaluation for near-ties.** With n=14 and observed variance,
   ±0.05 mean F1 is not significant. If budget allows, evaluating the top-2
   candidate agents on the union of two iterations' queries (cached problems
   already exist) would make selection markedly less random.
