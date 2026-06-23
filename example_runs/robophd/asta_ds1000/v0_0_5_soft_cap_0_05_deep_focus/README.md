# asta_ds1000 / v0_0_5_soft_cap_0_05_deep_focus

RoboPhD evolution run on AstaBench's DS-1000 task. Headline submitted agent: **`iter8_perspective_consensus_ds1000`**. Run id: `robophd-asta_ds1000-032`. Evolved by **Claude Opus 4.8** (the evolution model; not a solver).

**This is the Deep-Focus variant of `v0_0_5_soft_cap_0_05`.** Same run configuration as v0_0_5 (opus-4.8 evolution, budget 750, cost cap $0.05/$0.01, examples-per-iteration 20) with a single deliberate change: **Deep Focus Round 2 ON** (`new_agent_test_rounds=1` vs v0_0_5's 0) — each new agent gets a second fresh-batch evaluation within its iteration, under the Round-2-aware objective framing. As expected from stochastic evolution, this did NOT refine v0_0_5's agent — it produced a different one: where v0_0_5 converged on a GPT-5.4 + Sonnet cross-agreement solver at ~0.83 / $0.017, this run converged on a cheaper GPT-5.4-mini ensemble at a lower tier (official 0.751 / $0.0069). So it reads as a distinct, cheaper frontier point rather than a stronger v0_0_5.

What's distinctive about this submission:

- **Our cheapest and shortest submission.** The primary solver is **GPT-5.4-mini** (a 3-perspective ensemble), with GPT-5.4 invoked only to escalate no-consensus cases. At **$0.0069/problem** it's the cheapest of our six submissions (next: v0_0_2 $0.01), and at **354 lines** the shortest agent we've submitted (next: v0_0_4 500, v0_0_5 525, v0_0_2 561, v0_0_1 1214, v0_0_3 1614) — far under its $0.05 cap.
- **First submission with Deep Focus on — still under evaluation for DS-1000.** Every prior submission ran with Deep Focus off. This is one run, and we draw no conclusion from it: the immediately prior (unsubmitted) Deep-Focus run `robophd-asta_ds1000-031` produced a *typical* agent (0.84 @ $0.031, 1026 lines), so the cheap/short character here is not attributable to Deep Focus — it's run-to-run variance pending a controlled comparison.
- **A new low-cost frontier point.** At its scored **0.751 @ $0.0069** it occupies the cheap end of the Pareto curve that the prior RoboPhD entries (all ~0.83+) did not. It is **marginally better on accuracy** than ReAct o3 (0.749 @ $0.01) and Asta v0 (0.748 @ $0.01) and **significantly improved on cost** (~31% cheaper, $0.0069 vs $0.01), so it **Pareto-dominates** both and takes their place on the frontier. (The accuracy edge is within run-to-run noise, so cost is the decisive axis.) It does NOT dominate the cheaper ReAct GPT-5 Mini (0.710 @ $0.003 — cheaper but lower) nor our own sonnet entry (0.809 @ ~$0.01 — higher tier); it slots between them.

The `soft_cap_0_05` tail names the per-iteration mean-spend free-zone the run was trained under (`cost_threshold=0.05`, `cost_per_error=0.01`); the `deep_focus` tail marks `new_agent_test_rounds=1`.

## Leaderboard score

**Submitted as `RoboPhD`.** Ai2 leaderboard-verified row pending. The official `astabench score` and the internal-dev numbers:

| | Official (`astabench score`) | Internal dev eval |
|---|---|---|
| Accuracy (DS-1000 test, 900 samples) | **0.751** (676/900) | 0.7678 (691/900) |
| Per-problem cost | **$0.0069** | $0.0069 |
| Submission name | `v0_0_5_soft_cap_0_05_deep_focus` / form: `RoboPhD` | — |
| Pareto position | New cheap-tier frontier point: marginally more accurate than ReAct o3 (0.749 @ $0.01) and Asta v0 (0.748 @ $0.01) and ~31% cheaper, so it **Pareto-dominates** both. Cheaper points still exist below it (ReAct GPT-5 Mini, 0.710 @ $0.003) | — |
| Leaderboard | [AstaBench DS-1000 leaderboard](https://allenai-asta-bench-leaderboard.hf.space/code-execution) | — |

Margin note: the accuracy edge over o3/Asta-v0 is marginal — 0.751 vs 0.748–0.749, ~0.2–0.3pp (2–3 problems), within the ±1.4pp run-to-run stderr. So the domination is carried mainly by the cost axis (~31% cheaper); accuracy sits at-or-just-above the o3/Asta-v0 cluster rather than clearly ahead. Cost held vs internal ($0.0069 both); accuracy dipped 1.7pp (0.7678→0.751), with **0 wrapper fallbacks** in the official run (the internal eval's 4 were local Docker-infra blips, absent here).

## Submission metadata

| Field | Value |
|---|---|
| Agent name (form) | RoboPhD |
| Openness | Open source & closed weights |
| Tools tier | Standard (uses `python_session` provided by the task) |
| Models (solver, from logs) | gpt-5.4-mini (primary ensemble), gpt-5.4-2026-03-05 (no-consensus escalation) |
| URL (form) | https://github.com/andborth/RoboPhD |
| Leaderboard | [AstaBench DS-1000 leaderboard](https://allenai-asta-bench-leaderboard.hf.space/code-execution) |

Suggested agent description for the form:

> Evolved by RoboPhD with Claude Opus 4.8 as the evolution (but not a solver) model. The very short (354 line) evolved agent is a low-cost solver: GPT-5.4-mini generates three candidates from different framing perspectives, all executed in the sandbox with dtype/structure-aware value comparison; on consensus it submits the agreed answer, and only no-consensus cases escalate to a stronger GPT-5.4 arbiter (with verify/repair). Evolution ran under a soft cost cap of $0.05, and the discovered agent comes in far below it at well under a cent per problem. Note that this is essentially the same RoboPhD code base, but a different config relative to our previous submission. We turned on "deep focus evolution" in which newly evolved agents are tested and revised, a contribution of our paper. We are studying its usefulness in DS1000 where its effect is uncertain due to the limited training data size.

## Approach (iter8_perspective_consensus_ds1000)

354-line solver, ELO 1573.2 (clear pool leader; next-best iter11 at 1543). Pipeline:

1. **Three-perspective generation (cheap).** `GPT_5_4_MINI` generates three candidates, each prompted from a different framing perspective, in parallel.
2. **Sandbox execution + structure-aware comparison.** All candidates run in the free sandbox; executed values are compared with dtype/shape awareness, not just printed output.
3. **Consensus short-circuit.** When the cheap candidates agree, that answer is submitted — keeping the common case entirely on GPT-5.4-mini.
4. **Escalate only on disagreement.** No-consensus cases escalate to a stronger `GPT_5_4` arbiter that decides against the execution evidence, with verify-and-repair.

This "cheap-by-default, escalate-only-when-needed" structure is what holds the per-problem cost under a cent while still reaching ~0.77.

`iteration_007/` is the prior-iter context the iter8-producing evolution session read; `iteration_008/` is iter8's own first scoring; `evolution_output/iteration_008/` is the Claude Code session that produced it.

## Submission resilience wrapper

The submitted `agent.py` (inside the tarball) is the same auto-generated two-tier wrapper used by prior submissions: `make_solver` from this directory's iter8 source (renamed `agent_inner.py`) with a bundled `seed_agent.py` fallback, both bounded by `asyncio.wait_for(timeout=1200)`. Internal eval recorded `n_fallback_used: 4` — but all four were **Docker-infrastructure failures** (the sandbox failed to pull the `python:3.11` image: `auth.docker.io` 404), not agent errors; the seed recovered 3 of the 4 (sample 643 lost). On a clean eval these run normally, so the internal 0.7678 is if anything ~1 problem low. The **official run confirmed this: 0 wrapper fallbacks across all 900 samples** — the Docker-infra blips were local to the internal run. Wrapper template: [`scripts/asta_ds1000_submit.py`](../../../../scripts/asta_ds1000_submit.py) (`WRAPPER_TEMPLATE`).

## Internal development scoring (pre-submission)

| | Value |
|---|---|
| Score (RoboPhD-internal eval, full test) | **0.7678** (691 / 900) |
| Per-problem inference cost | $0.0069 |
| Test eval total cost | $6.20 |
| Best-agent ELO | 1573.2 |
| Train mean (5 ELO rounds) | 0.900 |
| Distinct winning agents | 7 (of 14 iters; 17 wins) |
| Wrapper-level fallbacks during internal eval | 4 / 900 (all Docker-infra, not agent; 3 recovered) |
| Evolution model / cost | Claude Opus 4.8 / $38.22 ($5.69 eval + $32.52 evo) |
| Config | budget 750, new_agent_test_rounds=1, max_workers 10, seed 147063 |

## Lineage (agents/)

14 agents in `agents/`: `seed_fw18t_as` → `iter2_exec_verify_ds1000` … `iter7_grounded_reconcile_ds1000` → **`iter8_perspective_consensus_ds1000`** (submitted) → `iter9_crossmodel_consensus_ds1000` … `iter14_champion_consensus_ds1000` (none unseated iter8's 1573 ELO).

## Lightweight inclusion

Two result subdirs (`iteration_007/`, `iteration_008/`), one evolution-output subdir (`evolution_output/iteration_008/`), plus `evolution_output/CLAUDE.md`. The other result subdirs and the rest of `evolution_output/` live in the local `robophd_runs/` mirror.

## Reproducing the submission

```
cd /path/to/repo
pip install litellm==1.88.1      # submission-scoring price map (see examples/asta_ds1000/README.md)
python scripts/asta_ds1000_submit.py --only v0_0_5_soft_cap_0_05_deep_focus
```

The script copies `agents/iter8_perspective_consensus_ds1000/agent.py` and the canonical seed into a working dir, wraps them in the two-tier `WRAPPER_TEMPLATE`, runs `astabench eval --solver agent.py --model none --split test --task DS_1000_test`, scores, and tarballs for upload.
