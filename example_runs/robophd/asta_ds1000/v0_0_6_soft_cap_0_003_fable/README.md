# asta_ds1000 / v0_0_6_soft_cap_0_003_fable

RoboPhD evolution run on AstaBench's DS-1000 task. Headline submitted agent: **`iter8_expected_diff_cascade`**. Run id: `robophd-asta_ds1000-036`. Evolved by **Claude Fable 5** (the evolution model; not a solver) — our first fable-5-evolved submission.

**This is the ultra-low-cap experiment.** Same run configuration family as v0_0_5 (budget 750, examples-per-iteration 20, Deep Focus off) with two deliberate changes: the evolution model (**fable-5** vs opus-4.8) and a **17x lower cost cap** (`cost_threshold=0.003`, `cost_per_error=0.001` — vs the $0.05/$0.01 default). The motivation was twofold: (a) probe what evolution produces at the extreme low end of the cost axis, and (b) take the official leaderboard's **ReAct/GPT-5-Mini** frontier point (0.710 displayed @ $0.00 band; true cost $0.003), one of the two entries our submissions do not yet Pareto-dominate. Outcome: (a) yes, (b) no — the official cost came in at $0.004280, above the target; see "Post-run cost postmortem" for the two-part cause. The entry still lands on the frontier as a new point between ReAct/GPT-5-Mini and our deep_focus entry.

What's distinctive about this submission:

- **A budget-tier cascade with no strong generator.** Generation is entirely gpt-5.4-mini + gemini-3.1-flash-lite (with claude-haiku-4-5 escalation); the strong gpt-5.4 appears **only as a rare, 450-token-capped tie-break arbiter** (two call sites: failed majority vote, and the no-sandbox disagreement fallback), so its cost amortizes to noise under the batch-average free zone. No dependence on any single relatively strong model — relevant to deployment scenarios where that dependence is itself the thing to avoid.
- **Cap-hugging at 98% — but against a mis-measured basis.** Internal mean agent cost was $0.00294/problem against the $0.003 cap, extending the fable-5 pattern (its $0.03-cap sibling `robophd-asta_ds1000-035` used 84% of its cap; opus-4.8 runs typically leave 60%+ unused). The post-run cost postmortem (below) found the internal number understated: on the leaderboard's billing basis the agent actually spends **$0.00428/problem — 143% of the intended cap**. Evolution hugged a cap it was mis-measuring.
- **Mechanical detection instead of LLM triage.** The iter8 delta over the iter7 champion replaces flash-lite MATCH/MISMATCH judgment with expected-output diffing: a flash-lite call quotes the desired-output block the asker printed in the problem text (mechanically validated), rendered candidate values are scored by canonicalized order-aware containment (difflib), and the score drives candidate selection, audit mismatch detection, and fix adoption behind a no-hardcoded-literals AST guard. Per the iter8 docstring: "Detection — not audit coverage — was the weak link."

The `soft_cap_0_003` tail names the per-iteration mean-spend free-zone the run was trained under (`cost_threshold=0.003`, `cost_per_error=0.001`); the `fable` tail marks the evolution model (Claude Fable 5).

## Leaderboard score

**To be submitted as `RoboPhD`.** Official `astabench score` run completed 2026-07-05:

| | Official (`astabench score`) | Internal dev eval |
|---|---|---|
| Accuracy (DS-1000 test, 900 samples) | **0.7322** (659/900, stderr 0.0148) | 0.7322 (659/900) |
| Per-problem cost | **$0.004280** | $0.00294 (pre-fix basis; see postmortem) |
| Wrapper fallbacks | 0 / 900 | 0 / 900 |
| Submission name | `v0_0_6_soft_cap_0_003_fable` / form: `RoboPhD` | — |
| Pareto outcome | ReAct/GPT-5-Mini (0.710 @ $0.003 true cost) is **NOT dominated** — we are +2.2pp on accuracy but 43% above it on true cost. The entry is instead a **new frontier point in its own right**: nothing on the board is both cheaper and more accurate, and it slots between ReAct/GPT-5-Mini and our deep_focus entry (0.751 @ $0.0069). Displays as 0.732 @ $0.00 (the rounded band covers ≤$0.0049). | — |
| Leaderboard | [AstaBench DS-1000 leaderboard](https://allenai-asta-bench-leaderboard.hf.space/code-execution) | — |

Margin note: the score replicated internally→officially EXACTLY (659/900 both — the tightest replication of any of our submissions; prior deltas ±0.8-1.7pp). The cost did not: see the postmortem below for the two-part cause.

## Post-run cost postmortem (why the $0.003 target was missed)

The internal $0.00294 was never achievable officially — it was measured on a flawed basis, in two independent ways:

1. **Gemini reasoning tokens were dropped** (our bug). Gemini reports thinking tokens separately from `output_tokens` (total = input + output + reasoning) but bills them at the output rate; the internal evaluator collected only input/output and priced 233k flash-lite thinking tokens at $0. `astabench score` bills them (agenteval folds reasoning into completion tokens when the token arithmetic proves output excludes it). Fixed in commit `dde6356`.
2. **Price-map version skew** (leaderboard-side artifact). The official scorer pins litellm 1.88.1's bundled price map (`LITELLM_LOCAL_MODEL_COST_MAP=True`), which bills gemini-3.1-flash-lite at $0.45/$2.70 per M — 1.8x Google's current $0.25/$1.50 that the live map (and our internal tracking) used. Repricing this run's logged usage on the bundled map reproduces the official $0.004280 to six decimal places.

Consequence for this run: evolution optimized under prices that made flash-lite thinking look ~2.6x cheaper than the board bills it, converged on a flash-lite-heavy cascade, and "hugged" a cap it was actually 43% over. Under correct billing, this architecture doesn't fit $0.003 even with thinking removed (~$0.0036), so the miss was structural, not marginal.

**Token-usage stability — the delta is accounting, not behavior.** The agent essentially reproduced its internal behavior in the official run: repricing the official run's logged tokens on the internal (pre-fix) basis gives $2.73 vs the internal run's recorded $2.645 (within 3.2%), and the score was identical (659/900 both; equal counts don't strictly prove the same 659 problems — overlap unchecked). Same tokens, different price tag. The same logged usage priced three ways:

| Basis | Per problem | vs $0.003 cap |
|---|---|---|
| Pre-fix (reasoning dropped + live prices) | $0.00294 | 98% |
| Reasoning billed + true live prices | $0.00343 | 114% |
| Reasoning billed + leaderboard bundled prices (= official) | $0.00428 | 143% |

The middle row shows the reasoning bug ALONE breaks the cap, independent of the price-map skew — the miss did not require Ai2's stale prices.

Policy going forward (decided 2026-07-05): internal costs track **Ai2's billing basis** — litellm's bundled snapshot, reasoning tokens folded per agenteval's rule — for comparability with other leaderboard systems, even where the bundled map lags true provider prices; we alert Ai2 to stale entries but move only when they move. Implemented in commits `b74f8e7` (bundled-map basis + live-map fallback warnings) and `ae1e410` (drift warnings + unit tests incl. a golden regression pinning this run's official cost). Cost-capped runs recorded before these commits understate Gemini-using agents' costs (v0_0_4 official +36%, this run +46% vs internal).

## Submission metadata

| Field | Value |
|---|---|
| Agent name (form) | RoboPhD |
| Openness | Open source & closed weights |
| Tools tier | Standard (uses `python_session` provided by the task) |
| Models (solver, from logs) | gpt-5.4-mini + gemini-3.1-flash-lite (generators), claude-haiku-4-5-20251001 (escalation), gpt-5.4-2026-03-05 (rare arbiter) |
| URL (form) | https://github.com/andborth/RoboPhD |
| Leaderboard | [AstaBench DS-1000 leaderboard](https://allenai-asta-bench-leaderboard.hf.space/code-execution) |

Suggested agent description for the form:

> Evolved by RoboPhD with Claude Fable 5 as the evolution (but not a solver) model, under an ultra-low soft cost cap of $0.003/problem — a probe of the extreme low end of the cost axis. The evolved agent is a budget-tier cascade: GPT-5.4-mini and Gemini-3.1-Flash-Lite generate candidates, each executed in the task sandbox; a mechanical expected-output diff (quoting the desired-output block printed in the problem text, scored by canonicalized containment) replaces LLM triage for catching agreed-but-wrong answers; Claude-Haiku-4.5 joins on escalation, and a stronger GPT-5.4 is consulted only as a rare, token-capped tie-break arbiter whose cost amortizes to noise across the batch. Same RoboPhD code base as our previous submissions; this run differs in the evolution model (Fable 5 vs Opus 4.8) and the 17x-lower cost cap.

## Approach (iter8_expected_diff_cascade)

1,419-line solver. Selection confidence is led by **mean training accuracy: 86.8 across 11 Elo test rounds** (~220 training-problem evaluations) — the absolute metric; Elo (1650, +104 over next-best iter7 at 1546) is reported secondarily since Elo is relative to the run's own agent pool. Champion for 12 consecutive iterations (iterations 9-20; no later agent unseated it). Pipeline:

1. **Guided dual generation (cheap).** `GPT_5_4_MINI` and `GEMINI_3_1_FLASH_LITE` generate candidates in parallel, with anti-hardcode AST stripping.
2. **Sandbox execution + traceback repair.** Candidates run in the free sandbox; crashes get one repair round with the traceback fed back.
3. **Expected-output containment scoring.** Where the problem text prints the desired output, candidates are scored mechanically against it (canonicalized token streams, order-aware containment); a ≥0.98 score with ≥0.08 margin selects outright — overriding wrong majorities and skipping escalation spend.
4. **Majority vote with Haiku escalation.** Disagreement without a containment winner brings in `CLAUDE_HAIKU_4_5` as a third voice.
5. **Rare strong arbiter.** Only when voting fails does `GPT_5_4` adjudicate (450-token cap); a no-sandbox fallback path uses the same arbiter on compile-checked disagreement.
6. **Audit + guarded fix adoption.** Containment mismatch (<0.97) triggers an audit; fixes are adopted only on a clean run + ≥0.08 containment gain + a no-hardcoded-expected-literals AST guard. Graceful degradation on any sandbox failure.

`iteration_007/` is the prior-iter context the iter8-producing evolution session read; `iteration_008/` is iter8's own first scoring; `evolution_output/iteration_008/` is the Claude Code session that produced it.

## Submission resilience wrapper

The submitted `agent.py` (inside the tarball) is the same auto-generated two-tier wrapper used by prior submissions: `make_solver` from this directory's iter8 source (renamed `agent_inner.py`) with a bundled `seed_agent.py` fallback, both bounded by `asyncio.wait_for(timeout=1200)`. Internal eval recorded `n_fallback_used: 0` across all 900 samples. Wrapper template: [`scripts/asta_ds1000_submit.py`](../../../../scripts/asta_ds1000_submit.py) (`WRAPPER_TEMPLATE`).

## Internal development scoring (pre-submission)

| | Value |
|---|---|
| Score (RoboPhD-internal eval, full test) | **0.7322** (659 / 900) |
| Per-problem inference cost | $0.00294 as measured pre-fix (98% of the $0.003 cap); $0.004280 on the leaderboard billing basis — see postmortem |
| Test eval total cost | $2.65 |
| Train mean (11 ELO rounds, ~220 problems) | **0.868** |
| Best-agent ELO | 1650 (next-best iter7 at 1546) |
| Distinct winning agents | 9 (of 20 iters) |
| Wrapper-level fallbacks during internal eval | 0 / 900 |
| Evolution model / cost | Claude Fable 5 / $156.46 ($2.48 eval + $153.97 evo) — our costliest DS-1000 evolution (fable-5 is ~2x opus-4.8's per-token price and writes long) |
| Config | budget 750, cost cap $0.003/$0.001, new_agent_test_rounds=0, max_workers 5, seed 201138 |

Train 0.868 → test 0.732 is a real gap; with only 100 training examples some train-pool fit at the high end is expected and is part of what this run series studies.

## Lineage (agents/)

20 agents in `agents/`: `seed_vjhvs6o3` → `iter2_exec_verify_ensemble` … `iter6_audited_cascade` → `iter7_lean_audited_cascade` → **`iter8_expected_diff_cascade`** (submitted) → `iter9_shape_aware_cascade` … `iter20_exact_inplace_cascade` (none unseated iter8's 1650 ELO). The cascade family iter2 → 6 → 7 → 8 is a strict improvement lineage; iter8's delta over iter7 is the mechanical expected-output diffing described above.

## Lightweight inclusion

Two result subdirs (`iteration_007/`, `iteration_008/`), one evolution-output subdir (`evolution_output/iteration_008/`), plus `evolution_output/CLAUDE.md`. The other result subdirs and the rest of `evolution_output/` live in the local `robophd_runs/` mirror.

## Reproducing the submission

```
cd /path/to/repo
pip install litellm==1.88.1      # submission-scoring price map (see examples/asta_ds1000/README.md)
python scripts/asta_ds1000_submit.py --only v0_0_6_soft_cap_0_003_fable
```

The script copies `agents/iter8_expected_diff_cascade/agent.py` and the canonical seed into a working dir, wraps them in the two-tier `WRAPPER_TEMPLATE`, runs `astabench eval --solver agent.py --model none --split test --task DS_1000_test`, scores, and tarballs for upload.
