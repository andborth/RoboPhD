# asta_ds1000 / v0_1_0_cap_0_127_fable51

RoboPhD evolution run on AstaBench's DS-1000 task. Headline submitted agent: **`iter6_ensemble_exec_vote_escalate`**. Run id: `robophd-asta_ds1000-054`. Evolved by **Claude Fable 5.1** (the evolution model; not a solver) — the first Fable-5.1-evolved DS-1000 submission.

**This run aimed at the top of the board.** The cost threshold was set to the per-problem cost of the board's highest-accuracy entry — our own v0_0_1_soft_cap_0_16 (0.8622 @ $0.1273) — so that any agent evolution produced inside the free zone would be at most as expensive as that point. Standard run configuration otherwise (budget 750, examples-per-iteration 20, Deep Focus off, standard seed, `use_your_judgment`), with `cost_threshold=0.127` and the default-scaled `cost_per_error=0.0127`.

Outcome: **the target was hit** — internal 900-test **0.8711 (784/900) @ $0.0561/problem**, which strictly Pareto-dominates v0_0_1_soft_cap_0_16 (0.8622 @ $0.1273): +0.9pp accuracy at 44% of the cost. That would be a new headline DS-1000 score.

What's distinctive about this submission:

- **Saturation, not cost pressure, set the price.** The agent spends 44% of the cap. It appeared at iteration 6 and was beaten head-to-head only twice afterwards (iterations 10 and 13), scoring 20/20 in 9 of its 17 Elo rounds; 14 of the 19 later agents were flagged as exact clones because, at near-perfect accuracy on 20-problem batches, most debuts tied it at 20/20. RoboPhD saturated the training signal early, so nothing pushed spend toward the cap.
- **Three-provider execution vote with escalation.** Claude Opus 4.8, GPT-5.4 (medium reasoning) and Gemini 3.1 Pro write candidates in parallel; each is executed in a fresh sandbox subprocess and its target variables dumped with type/shape/dtype/index, and a crashing candidate gets one Sonnet 4.6 repair round. Matching executed outputs win outright with no judge call (loop-free code preferred). Only on a three-way split do Sonnet 4.6 (medium) and GPT-5.5 add two more voices (3 of 5 wins), else an Opus 4.8 low-reasoning judge picks among one representative per distinct output; a separate Opus judge handles value-agree/dtype-split cases.
- **A generality check voting can't provide.** When a majority agrees on the example but their code differs, Sonnet 4.6 writes an alternative input and the agreeing candidates are re-run on it; if they diverge, the judge decides with both executions in view.
- **Cost-aware staging.** Its own token-cost estimate, a $0.16 soft cap on optional steps and a $0.30 judge cap, per-stage wall-clock cutoffs, and best-so-far always emitted. 1,226 lines.
- **Convergent with the agent it dominates.** v0_0_1's iter10 (Opus-4.7-evolved, months earlier) reached the same core design — multi-family candidates executed in the sandbox, a judge on disagreement, an output-preserving loop rewrite — but always runs four strong models plus Opus critic calls (5–8 calls/problem). This agent lets agreement short-circuit the judge and escalates only on disagreement, which is where the 56% cost saving comes from.

The `cap_0_127` tail names the training free zone, `cost_threshold=0.127`, at the default-scaled `cost_per_error` (10% of the threshold); `fable51` names the evolution model. The version moves to 0.1.0 because the submission infrastructure (resilience wrapper, per-sample module isolation, billing-basis parity) is now considered settled.

## Comparison with the dominated entry

Paired per-problem against v0_0_1_soft_cap_0_16's official eval log (recomputed with litellm's bundled price map, reproducing 0.8622 @ $0.1273 exactly):

| | v0_0_1_soft_cap_0_16 (official) | this agent (internal) |
|---|---|---|
| Accuracy | 0.8622 (776/900) | 0.8711 (784/900) |
| Cost / problem (mean) | $0.1273 | $0.0561 |
| Cost / problem (median, p90, max) | $0.112, $0.203, $0.361 | $0.044, $0.095, $0.371 |
| Wrapper fallbacks | 34 / 900 (1200 s timeouts) | 0 / 900 |

- Paired accuracy: 37 problems solved only by this agent, 29 only by v0_0_1, 87 by neither (McNemar exact p = 0.39). **Statistically tied on accuracy; the domination is decisively on cost** — cheaper on 873/900 problems and in every library.
- The accuracy margin (+0.9pp) is smaller than the run-to-run spread seen on a sibling agent (0.821 vs 0.792 on two internal evals of -053's winner), so the official number decides whether the headline claim stands.

## Wrapper-timeout check

The submission wrapper bounds each sample at 1200 s (`PRIMARY_TIMEOUT_S`), below this agent's own 1260 s `HARD_LIMIT` (it was written for the internal 1770 s budget). The internal 900-test shows the limit is not in play: max 336 s per problem, p99 230 s.

## Leaderboard score

**To be submitted as `RoboPhD`.** Official `astabench score` run: **pending**.

| | Official (`astabench score`) | Internal dev eval |
|---|---|---|
| Accuracy (DS-1000 test, 900 samples) | pending | 0.8711 (784/900) |
| Per-problem cost | pending | $0.0561 (leaderboard billing basis) |
| Wrapper fallbacks | pending | 0 / 900 |
| Submission name | `v0_1_0_cap_0_127_fable51` / form: `RoboPhD` | — |
| Pareto target | v0_0_1_soft_cap_0_16 (0.8622 @ $0.1273): internally **strictly dominated** (+0.9pp at 44% of the cost). Not dominated by our other ~0.853 points ($0.037 Opus-4.8-evolved, $0.052 Fable-5-evolved): higher accuracy. | — |
| Leaderboard | [AstaBench DS-1000 leaderboard](https://allenai-asta-bench-leaderboard.hf.space/code-execution) | — |

## Submission metadata

| Field | Value |
|---|---|
| Agent name (form) | RoboPhD |
| Openness | Open source & closed weights |
| Tools tier | Standard (uses `python_session` provided by the task) |
| Models (solver) | claude-opus-4-8, gpt-5.4-2026-03-05, gemini-3.1-pro-preview (first-round candidates; Opus also judges), claude-sonnet-4-6 (repair, generality input, escalation; ~58% of problems), gpt-5.5 (escalation; ~12% of problems), gpt-5.4-mini (last-resort one-shot) |
| URL (form) | https://github.com/andborth/RoboPhD |
| Leaderboard | [AstaBench DS-1000 leaderboard](https://allenai-asta-bench-leaderboard.hf.space/code-execution) |

Key stats (not required by the AstaBench form; recorded for our own tracking):

| Stat | Value |
|---|---|
| Evolved agent name | `iter6_ensemble_exec_vote_escalate` |
| Lines of code | 1,226 (`agent.py`, single file) |
| Evolution model | claude-fable-5-1 |
| Cost target (`cost_threshold`) | $0.127/problem (training free zone; batch mean) |
| Cost-breach penalty (`cost_per_error`) | $0.0127 of mean overage per error-equivalent (default-scaled, 10% of threshold) |
| Elo / training rounds | 1610 / 17 (mean train score 97.35; train–test gap 10.2pp) |
| Run cost | $200.94 (evolution $167.10, training evals $33.84) + $50.45 internal test |
| Model calls / problem | ~4.1 mean |
