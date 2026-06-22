# asta_ds1000 / v0_0_6_soft_cap_0_05

RoboPhD evolution run on AstaBench's DS-1000 task. Headline submitted agent: **`iter8_perspective_consensus_ds1000`**. Run id: `robophd-asta_ds1000-032`. Evolved by **Claude Opus 4.8** (the evolution model; not a solver).

What's distinctive about this submission:

- **A cheap-tier agent, by design.** The primary solver is **GPT-5.4-mini** (a 3-perspective ensemble), with GPT-5.4 invoked only to escalate no-consensus cases. It lands at **$0.0069/problem** — the cheapest RoboPhD agent at any tier, and far under its $0.05 cap.
- **A new low-cost frontier point.** At 0.7678 @ $0.007 it occupies the cheap end of the Pareto curve that the prior RoboPhD entries (all ~0.83+) did not: it dominates the ReAct o3 (0.749 @ $0.01) and Asta v0 (0.748 @ $0.01) entries — higher accuracy *and* cheaper. It does NOT dominate the cheapest existing point (ReAct GPT-5 Mini, 0.710 @ $0.003 — cheaper but lower) nor our own sonnet entry (0.809 @ ~$0.01 — higher tier); it slots between them as a distinct frontier point.
- **Deep Focus Round 2 ON** (`new_agent_test_rounds=1`), unlike v0_0_5's 0 — each new agent gets a second fresh-batch evaluation within its iteration, with the Round-2-aware objective framing.

The `soft_cap_0_05` tail names the per-iteration mean-spend free-zone the run was trained under (`cost_threshold=0.05`, `cost_per_error=0.01`).

## Leaderboard score

**Submission prepared — to be uploaded as `RoboPhD`.** Leaderboard-verified numbers will be filled once processed. As-submitted (`astabench score`) and internal-dev numbers:

| | As scored (`astabench score`) | Internal dev eval |
|---|---|---|
| Accuracy (DS-1000 test, 900 samples) | _pending upload_ | 0.7678 (691/900) |
| Per-problem cost | _pending upload_ | $0.0069 |
| Submission name | `v0_0_6_soft_cap_0_05` / form: `RoboPhD` | — |
| Pareto position | New cheap-tier frontier point; dominates ReAct o3 (0.749 @ $0.01) and Asta v0 (0.748 @ $0.01). Cheaper points exist below it (ReAct GPT-5 Mini, 0.710 @ $0.003) | — |
| Leaderboard | [AstaBench DS-1000 leaderboard](https://allenai-asta-bench-leaderboard.hf.space/code-execution) | — |

Margin note: the o3/Asta-v0 domination rides on internal 0.7678 vs their 0.748–0.749 (a ~1.8–2pp edge). Prior dev→official deltas were small and signed both ways (v0_0_4 +0.9pp, v0_0_5 −0.8pp), so the edge is likely to hold but is not wide.

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

> Evolved by RoboPhD with Claude Opus 4.8 as the evolution (but not a solver) model. The evolved agent is a low-cost solver: GPT-5.4-mini generates three candidates from different framing perspectives, all executed in the sandbox with dtype/structure-aware value comparison; on consensus it submits the agreed answer, and only no-consensus cases escalate to a stronger GPT-5.4 arbiter (with verify/repair). Evolution ran under a soft cost cap of $0.05, and the discovered agent comes in far below it at well under a cent per problem.

## Approach (iter8_perspective_consensus_ds1000)

354-line solver, ELO 1573.2 (clear pool leader; next-best iter11 at 1543). Pipeline:

1. **Three-perspective generation (cheap).** `GPT_5_4_MINI` generates three candidates, each prompted from a different framing perspective, in parallel.
2. **Sandbox execution + structure-aware comparison.** All candidates run in the free sandbox; executed values are compared with dtype/shape awareness, not just printed output.
3. **Consensus short-circuit.** When the cheap candidates agree, that answer is submitted — keeping the common case entirely on GPT-5.4-mini.
4. **Escalate only on disagreement.** No-consensus cases escalate to a stronger `GPT_5_4` arbiter that decides against the execution evidence, with verify-and-repair.

This "cheap-by-default, escalate-only-when-needed" structure is what holds the per-problem cost under a cent while still reaching ~0.77.

`iteration_007/` is the prior-iter context the iter8-producing evolution session read; `iteration_008/` is iter8's own first scoring; `evolution_output/iteration_008/` is the Claude Code session that produced it.

## Submission resilience wrapper

The submitted `agent.py` (inside the tarball) is the same auto-generated two-tier wrapper used by prior submissions: `make_solver` from this directory's iter8 source (renamed `agent_inner.py`) with a bundled `seed_agent.py` fallback, both bounded by `asyncio.wait_for(timeout=1200)`. Internal eval recorded `n_fallback_used: 4` — but all four were **Docker-infrastructure failures** (the sandbox failed to pull the `python:3.11` image: `auth.docker.io` 404), not agent errors; the seed recovered 3 of the 4 (sample 643 lost). On a clean eval these run normally, so the internal 0.7678 is if anything ~1 problem low. Wrapper template: [`scripts/asta_ds1000_submit.py`](../../../../scripts/asta_ds1000_submit.py) (`WRAPPER_TEMPLATE`).

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
python scripts/asta_ds1000_submit.py --only v0_0_6_soft_cap_0_05
```

The script copies `agents/iter8_perspective_consensus_ds1000/agent.py` and the canonical seed into a working dir, wraps them in the two-tier `WRAPPER_TEMPLATE`, runs `astabench eval --solver agent.py --model none --split test --task DS_1000_test`, scores, and tarballs for upload.
