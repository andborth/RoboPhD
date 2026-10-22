# Evolution reflection — iteration 016 (feedcast)

## What worked well

- **Recomputing cost under the test mix, not the batch mix.** One jq loop over every
  `result.json` showed bodycast's semantic queries average $0.091 while its batch mean looked
  safe at $0.063; at the held-out 73 % semantic share that is ≈ $0.070 and a ~1.1-query penalty.
  No per-problem diagnostic surfaces this; it was the cheapest, most certain gain of the session.
- **Bucketing judge labels by submitted position and by our own grade tier** (one jq over
  `judge_verdicts.json`). It showed in minutes that the judge finds 2–5× more Perfect papers than
  our grader on large-K queries and that the window runs deep into our grade-1/ungraded tiers,
  which pointed at the pool and the planner's criteria rather than at evidence or ordering.
- **Reading the head-rating vector in the smoke log.** `[3, 1, 3]` on every head paper for
  semantic_193 identified the planner's "published after 2023" criterion as the thing flattening
  our own grading within a minute of the run reaching that stage. The `head:` print line has
  been the single most useful diagnostic for three sessions running; keep agents verbose.
- **Launching the first smoke run six minutes in, with the core patch only, then a second run
  with the final file eight minutes later.** Two overlapping runs on the same query gave an A/B
  on the planner change (criteria text, grade distribution, head ratings) inside the hour, and
  the first run validated the whole pipeline end-to-end before the clock mattered.
- **Checking gold reachability before chasing a loss.** `tool_probe.py get_paper` on
  semantic_140's missing known-good paper returned the date-cutoff error; that recall is capped
  by construction, so no code was spent on it.
- **Uniqueness-asserted string patches + `ast.parse`** (inherited from earlier sessions). Four
  patches to a 2 000-line file in ~10 minutes with no breakage.

## What was challenging or time-consuming

- **Backend slowness is bursty and invalidates single-run conclusions.** The new title-search
  step returned 63 neighbours in run 1 and timed out (0 new) in run 2; a session-side probe of
  the same title also hung for over two minutes. Any stage that depends on one tool call per seed
  needs a per-call timeout that overlaps with another stage, plus a diagnostic line with per-call
  hit counts so the next session can tell "no neighbours" from "backend down".
- **No judge in the smoke harness** (seventh session to say so). Grade distributions and head
  ratings are proxies; the planner fix is visibly right, but the stage-2 neighbour yield (2 of 63
  grade≥2 on one query) cannot be turned into an F1 estimate, so its value is still a hypothesis.
- **Wall clock feels faster than it is.** `date` in every command showed the session was only
  ~20 minutes in when it felt like 35; I could have afforded a third smoke case.
- **The 2 000-line agent is now hard to reason about as a whole.** Each session reads ~300
  lines of the semantic path and patches around them; nobody has re-audited the metadata path in
  several iterations.

## Were the tools helpful?

- `tool_probe.py`: decisive for gold reachability; too slow to use as an A/B on search quality
  during a backend slow period (a single keyword probe exceeded the 120 s command timeout).
- Per-problem diagnostics are sufficient and well structured. `judge_verdicts.json` with
  `position`/`status`/`label` plus `score_meta.json` makes the per-position bucketing a one-liner.
- The Monitor/background-Bash `until grep -q` pattern worked well for 10-minute runs. One
  friction point: task output files land under a cwd-dependent tmp path, and after a `cd` into
  `smoke/` the earlier task's file became unreadable under the sandbox — read the log file in the
  workspace directly instead of the task output file.
- `sleep` chains are blocked; that is fine once you know to use background `until` loops.

## What I would do differently

- Queue a third smoke case (a small-K query such as semantic_140 or semantic_155) immediately
  after the second, to check that the head-rerank and evidence stages behave with the reduced
  `HEAD_RERANK`/`GRADE_ROUND1`.
- Print the stage-2 seed titles and per-search hit counts from the first draft (added only after
  run 2 showed 0 neighbours).
- Spend five minutes on the grade-1 tier ordering: on semantic_226/229 the judge rated 20+
  papers Perfect that our grader placed in a 230-paper grade-1 tier ordered only by fusion score.
  A cheap second signal there (snippet hits, title-word overlap with the criteria) is untested.

## Insights about the task

- **Gold criteria are content-only.** Date/recency phrases never appear in `relevance_criteria`;
  a planner that turns them into criteria makes every paper grade 2 in our own grader (no
  abstract states its publication date) and flattens the head ordering. Do not year-filter
  semantic submissions either: the judge ignores dates, so out-of-range papers it rates Perfect
  still count toward recall.
- **The GPT-5.4 head rerank reads criteria literally** ("*adaptive* query expansion" → 1 for
  papers that never use the word), while the judge, working from longer criterion descriptions,
  is lenient. A planner instruction to phrase criteria as a lenient reader would, or a head-key
  fallback to the grader's fused score when all head papers tie, are untested next steps.
- **Stage-2 expansion starves on sparse citation graphs** (14 citers on semantic_193, 46 on
  semantic_160, 300+ on the queries that scored well). Title-neighbour search is a
  citation-independent second source at zero LLM cost beyond the expansion cap.
- **Cost structure**: 80 % GPT-5.4-mini grading, input-dominated at ≈ $0.0001 per graded
  candidate; 20 % the GPT-5.4 head rerank (≈ $0.02). Grading volume, not model choice, is the
  knob, and it must be sized against the semantic-only mean (≤ ≈ $0.075).

## Insights about the evolution strategy

- Batch variance (the same agent 0.29–0.59 across iterations) still exceeds every between-agent
  gap. Iteration 15's 0.58 was an easy draw, not a breakthrough; only mechanism-backed changes
  should ship, and a win on one batch is not evidence.
- The cost penalty is a silent generalization failure: an agent tuned on batches with 35 %
  cheap queries can look free-zone and then pay a full query-equivalent on the test set. This
  should be checked every iteration, not discovered by chance.
- Pairing the new agent with the best lineage agent and one structurally different agent gives
  more information per iteration than two near-identical lineage members.

## Suggestions for the prompt / process

1. **Add a semantic-only cost column** (and the projected test-mix mean) to the iteration score
   table, next to the batch mean. Every session should see "$0.091 per semantic query" without
   computing it.
2. **Judge-in-the-loop smoke harness**: one GPT-5.4-mini call per paper over the top K with the
   documented per-criterion → grade formula. It is the single most valuable missing tool; seven
   sessions have shipped semantic changes without being able to measure them.
3. **Per-problem table in the prompt** (type, K, rank, recall, cost, wall per agent). Every
   session rebuilds it in its first ten minutes.
4. **Add to CLAUDE.md**: gold `relevance_criteria` never contain date, venue or author
   constraints; planner criteria must be content-only. Keyword/citation backends have multi-minute
   slow periods — give every per-seed tool call a timeout that overlaps other work and log per-call
   hit counts.
5. Show the session start time in the prompt and remind that `date` is the only clock.
6. Consider one iteration devoted to a measurement deliverable (judge harness, per-criterion
   verdict dump from the real scorer) instead of another agent variant; it would make the next
   ten iterations more productive than any single agent change.
