# Evolution Reflection — Iteration 18 (iter18-cocite-largegold-v1)

## What worked well

**1. The missed-gold list in `score_calculation.md` is a goldmine on
exact-match queries — read it before theorizing.** metadata_25's diagnostics
enumerate all 159 missed gold ids. Resolving 60 of them through the public
S2 API turned two iterations of "the funnel starves" hand-waving into a
proof in ~10 minutes: 0/60 name the seed in the title, 1/60 in the
abstract — so text search structurally cannot reach this gold, and every
keyword/mention-channel tweak previous sessions considered was doomed
before it was written. On any exact-match loss, resolve the missed gold
FIRST; it converts channel design from guesswork into constraint
satisfaction.

**2. Probe-before-build, twice in one session.** (a) 8 snippet_search
variants surfaced only 4/172 gold — the "expand the mention channel" idea I
would otherwise have shipped was measured dead in ~5 minutes. (b)
References of 100 random window citers covered 26/172 gold → the
co-citation channel was built on a measured prior, and the frequency>=2
collapse (26 → 4 gold) was discovered before it could become a shipped bug.
Also probe-verified the load-bearing mechanical fact (get_paper_batch
accepts raw sha ids) before writing any code that depends on it. The
pattern that generalizes: for each candidate channel, measure gold coverage
with a 5-minute probe before implementing; most candidate channels die at
this step, and the survivors ship with calibrated caps instead of guesses.

**3. The standing reading order held up again**: traceback grep → per-type
split → validate the previous session's shipped-but-unvalidated fixes via
their stdout markers → trace one or two losses end-to-end → only then read
prior narratives. The per-type split made parent choice trivial in two
minutes (iter17 best specific by 0.37, semantic within 0.006 = noise). The
stdout-marker check validated iter16's author-year channel and iter17's
ambiguous-union for free.

**4. The memory ledger is the compounding asset.** This session's key
decisions leaned on ~8 recorded facts (noise floor ~0.02 per type at n≈10,
"semantic_155 is small-K variance — don't chase it", references-field
shape, the 1000-window era gap, gold-id ceiling, mention-pad hit rates).
Sessions that read the ledger before the data are anchored; sessions that
read it after use it as a cross-check. The latter is correct.

**5. Small gated diff on a validated parent.** +153/−20 lines, all behind
an existing gate that only fires on the large-gold citation shape;
specific and semantic paths byte-identical to the lineage's best. At this
maturity (clean batches, solved plumbing, variance-dominated endgame), the
marginal value of a second speculative change is negative — it mostly adds
regression surface and attribution ambiguity.

**6. Exec-extracted-block smoke tests.** Testing the SHIPPED block text
(sliced out of agent.py by marker, exec'd with stub fixtures) caught one
real harness-vs-production mismatch (deadline is always a float in
production) and verified the gate logic, crash handling, and the raw-id
flag without needing the full import chain.

## What was challenging / time-consuming

- **The references field carries only {paperId sha, title}** — no corpusId,
  no year — so the co-citation channel needs an extra batch hop (sha →
  metadata) before it can filter anything, and cannot pre-filter refs by
  year at mining time. Frequency ranking has to carry the selection until
  metadata arrives; this forced the sha cap and is the channel's main
  fragility.
- **Public S2 API rate limits**: unauthenticated batch calls 429 readily;
  every script needed retry/backoff loops. Budget ~2x the naive time for
  any S2-resolution step, and batch aggressively.
- **tool_probe.py output format**: concatenated pretty-printed JSON objects
  (not a JSON array, not JSONL) — my first parse attempt failed; a
  raw_decode loop is needed. ~30s startup and noisy stderr per invocation
  also mean you cannot loop it per-paper; design probes as one-call-per-
  question.
- **Wall-clock reasoning is the real risk of free tool calls.** Tool calls
  cost $0 but launch at a shared 8/s; the new channel's ~900 extra calls
  ride on the existing ~2700. I shipped deadline-awareness everywhere, but
  couldn't measure the true combined wall-clock session-side. Next session
  must check `eval_wall_clock_seconds` before concluding anything about the
  channel's yield.

## Tools

- **tool_probe.py**: highest-value tool again; both decisive probes ran
  through it. Quirks above.
- **Public S2 API**: indispensable for resolving gold ids — the ONLY way to
  learn what the misses have in common. Rate limits are the tax.
- **jq over score_meta.json / result.json**: per-type means and rank/recall
  decomposition in one command each. (judge_verdicts.json has a non-uniform
  tail element — naive `.[] | .status` jq errors; slice or guard.)
- The diagnostics filename is `agent_stdout` (no .md) — still costs every
  session one wasted grep.

## What I'd do differently

- I spent ~35 minutes on analysis+probes and ~20 on implementation, which
  left the smoke tests slightly rushed (one fixture bug cost a cycle). The
  probes were worth it, but I should have written the smoke-test skeleton
  while the slow citation-window probe ran in the background.
- I did not probe the co-citation channel at full scale (500 citers) — only
  100 — so the shipped caps are extrapolations. A second 15-minute probe
  run in parallel would have calibrated COCITE_SHA_CAP against the real
  frequency distribution instead of a Poisson guess.

## Insights about the evolution strategy

- **The lineage's returns now come from shape-level channels, not
  parameter tuning.** Every recent real gain (author-year channel,
  ambiguous union, large-gold padding, now co-citation) is a new retrieval
  channel gated to a recurring query shape, validated by a probe, and
  measured by a stdout marker in the next batch. Tuning existing constants
  has been noise-level for ~6 iterations.
- **Fix-validation is asynchronous**: batch composition varies (0–3
  metadata per batch), so a shipped channel may wait 1–3 iterations for its
  shape to appear. The "shipped but batch-unvalidated" list in the ledger,
  plus a unique stdout marker per channel, is what makes this tractable.
  iter16's author-year channel validated two iterations after shipping —
  exactly through this mechanism.
- **Solo losses need triage against recorded variance shapes before
  engineering.** semantic_155 (0.244 vs 0.504) looks alarming but is the
  recorded small-K judge-variance shape; chasing it would have burned the
  session on noise. The ledger's list of "known variance / known saturated"
  shapes is as valuable as its list of open problems.

## Suggestions for the process/prompt (ranked)

1. **Add the missed-gold id list to exact-match diagnostics prominently**
   (it exists in score_calculation.md — surface it as a machine-readable
   sibling, e.g. `missed_gold.json`). This session's entire result came
   from noticing it; earlier sessions apparently did not, and spent two
   iterations designing channels blind.
2. **Expose per-criterion judge verdicts** (5th session asking). The
   grade-2 conversion pool remains the largest unaddressable semantic
   lever; evidence repair is blind without knowing WHICH criterion failed.
3. **Per-type mean rows and a K column in error_analysis_report.md.**
   Every session recomputes both by hand as its first act.
4. **Session-side model_registry proxy** for one real LLM call — prompt
   changes still ship logic-validated but roll-unvalidated.
5. **Fix tool_probe.py output to JSONL** and document the `agent_stdout`
   filename in CLAUDE.md; both cost each session a few minutes of
   rediscovery.
6. Consider persisting the memory-ledger's confirmed environment facts
   (references-field shape, raw-sha batch fetch, gold-id ceiling, window
   era gap) into CLAUDE.md — they are load-bearing and currently live only
   in one session-private file.

## Standing follow-ups for next session

1. Grep stdout for `co-cite:` on metadata queries; compare hits vs
   iteration 17's 13/172 on the metadata_25 shape; check
   `eval_wall_clock_seconds` (metadata_25 was 668s of the 1080s stage
   deadline). If the channel starved on time, lower COCITE_CITER_CAP
   before concluding the idea failed.
2. specific_39 got 2/5 SPIKE gold; the missed three are cross-domain. If
   ambiguous nicknames recur, consider one title guess PER research area in
   the plan prompt — but only with a trace showing the pick had them and
   dropped them.
3. semantic small-K variance and g2 conversion remain recorded-blocked;
   don't churn the semantic path without a new diagnostic.
