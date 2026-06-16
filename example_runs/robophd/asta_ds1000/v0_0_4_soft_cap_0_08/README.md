# asta_ds1000 / v0_0_4_soft_cap_0_08

RoboPhD evolution run on AstaBench's DS-1000 task with cost-penalty free-zone threshold `--cost-threshold 0.08`. Headline submitted agent: **`iter11_ds1000_tridtype_judge`**. Run id: `robophd-asta_ds1000-028`. Evolved by **Claude Opus 4.8** (the evolution model; not a solver).

The `0_0_4` bump from `v0_0_3_soft_cap_0_06` is a new submission from a distinct run. Two things matter most versus 0.0.3:

- **Solver model set is fully priceable on the leaderboard.** This agent calls `gpt-5.4-2026-03-05`, `claude-sonnet-4-6`, and `gemini-3.1-pro-preview` — all three appear with costs on existing leaderboard rows. 0.0.3 called `gemini-3.5-flash`, which is absent from the litellm bundled price map the leaderboard recomputes cost from, so 0.0.3 shipped with **Missing** cost. This run is expected to report a real cost figure.
- **Evolution model is Opus 4.8** (0.0.3 was the first Fable-5-evolved run).

The `soft_cap_0_08` tail names the per-iteration mean-spend free-zone the run was trained under (`cost_threshold=0.08`, `cost_per_error=0.01`).

## Leaderboard score

**Submitted 2026-06-16 as `RoboPhD (Opus 4.8-evolved)`; leaderboard validation pending (~5–7 business days).** The as-submitted `astabench score` numbers (what's in the uploaded tarball) and the development-internal numbers:

| | As submitted (`astabench score`) | Internal dev eval |
|---|---|---|
| Accuracy (DS-1000 test, 900 samples) | **0.8533** (768/900) | 0.8444 (760/900) |
| Per-problem cost | **$0.0368** | $0.0295 |
| Submission name | `v0_0_4_soft_cap_0_08` / form: `RoboPhD (Opus 4.8-evolved)` | — |
| Pareto position | Dominates ReAct gpt-5.5 (0.847 @ $0.05) and gpt-5.4 (0.838 @ $0.04); cheapest agent at ~0.853 | — |
| Leaderboard | [AstaBench DS-1000 leaderboard](https://allenai-asta-bench-leaderboard.hf.space/code-execution) | — |

Supersedes `v0_0_3_soft_cap_0_06` (RoboPhD 0.0.3): same 0.8533 accuracy, but a real (non-Missing) cost that is also lower than 0.0.3's locally-rescored $0.0519. Calibration from prior submissions: `v0_0_1_soft_cap_0_16` measured +2.3pp on the leaderboard vs internal (0.862 vs 0.8389); `v0_0_2_soft_cap_0_08` at parity (0.809 vs 0.8044). The leaderboard-displayed cost may differ slightly from $0.0368 if Ai2's time-invariant price map values `gemini-3.1-pro-preview` differently — but it will not be Missing, since all three solver models are priced.

## Submission metadata

| Field | Value |
|---|---|
| Agent name (form) | RoboPhD (Opus 4.8-evolved) |
| Openness | Open source & closed weights |
| Tools tier | Standard (uses `python_session` provided by the task) |
| Models (solver, from logs) | gpt-5.4-2026-03-05, claude-sonnet-4-6, gemini-3.1-pro-preview |
| URL (form) | https://github.com/andborth/RoboPhD |
| Leaderboard | [AstaBench DS-1000 leaderboard](https://allenai-asta-bench-leaderboard.hf.space/code-execution) |

Agent description as submitted on the form (verbatim):

> Evolved by RoboPhD with Claude Opus 4.8 as the evolution (but not a solver) model. The evolved agent generates three candidate solutions in parallel from different model families, executes them in the sandbox with dtype/structure-aware diagnostics, and adjudicates disagreements against the execution evidence with an output-grounded judge, followed by a repair pass. Evolution worked under a soft cost cap of $0.08 set deliberately above our desired target because Opus typically greatly undershoots its cost cap (unlike Fable, which, in a handful of trials, seemed to hug the target with an appropriate safety margin).

## Approach (iter11_ds1000_tridtype_judge)

Pipeline (ELO 1563.3 — clear pool leader; next-best iter6 at 1512):

1. **Three-candidate, three-family generation.** `GPT_5_4` (reasoning=low), `CLAUDE_SONNET_4_6` (reasoning=low), and `GEMINI_3_1_PRO_PREVIEW` generate candidates in parallel (`_CANDIDATE_SPECS`, agent.py:388).
2. **Sandbox execution with STRUCTURE-rich diagnostics.** The run diagnostic surfaces dtypes/shape/structure of executed values, not just printed output — the iter10→iter11 innovation that grounds the judge.
3. **Output-grounded judge.** On ambiguous cases an `GPT_5_4` judge (reasoning=high, agent.py:464) decides against the executed-value evidence; the third model family (Gemini 3.1 Pro) raises diversity on the hard cases.
4. **One repair pass** (GPT-5.4, reasoning=low) on candidates that fail to run.

Cost mix on the iteration-11 train batch: GPT-5.4 **82%**, Sonnet 4.6 **11%**, Gemini 3.1 Pro **7%** — ~$0.045/problem on the train batch, **$0.0295/problem** on the full 900-sample test.

The `iteration_010/` subdir captures the prior-iter result the iter11-producing evolution session read as context; `iteration_011/` is iter11's own first scoring; `evolution_output/iteration_011/` is the Claude Code session that produced it.

## Submission resilience wrapper

The submitted `agent.py` (inside the leaderboard tarball) is **not** the literal evolved iter11 source. It's the same auto-generated two-tier wrapper used by the prior submissions: `make_solver` imported from this directory's iter11 source (renamed `agent_inner.py`) with a bundled `seed_agent.py` fallback (the canonical seed), both tiers bounded by `asyncio.wait_for(timeout=1200)`. The internal eval recorded **2 fallbacks across the 900 samples** (`n_fallback_used: 2` in `test_results_final.json`) — i.e. the wrapper's seed tier recovered the score on 2 problems where the primary errored, exactly the crash-tolerance the internal score was produced under.

The wrapper template lives in [`scripts/asta_ds1000_submit.py`](../../../../scripts/asta_ds1000_submit.py) (`WRAPPER_TEMPLATE`) and is materialized at submission stage time.

## Lineage (agents/)

16 agents in `agents/`, in chronological order:

1. `seed_dmmf_ba5/agent.py` — the seed for this run
2. `iter2_ds1000_verify_escalate` … `iter10_ds1000_dtypeverify` — the ensemble/judge lineage: verify-escalate, ensemble judging, universal verify, verified judge, tri-verified, strong judge, exec-verify, dtype-verify
3. `iter11_ds1000_tridtype_judge/agent.py` — **the submitted candidate**: three-family ensemble + dtype-rich structure diagnostics + output-grounded judge
4. `iter12_ds1000_judgepanel` … `iter16_ds1000_richarbiter` — later iters; none unseated iter11 (all settled well below its 1563 ELO)

## Internal development scoring (pre-submission)

These are the numbers RoboPhD's internal subprocess-isolated evaluator measured during development. They guided the decision to submit but are NOT the canonical leaderboard score.

| | Value |
|---|---|
| Score (RoboPhD-internal eval, full test) | **0.8444** (760 / 900) |
| Per-problem inference cost | $0.0295 |
| Test eval total cost | $26.53 |
| Best-agent ELO | 1563.3 |
| Experiment-phase (90-sample) score | 0.8222 ($0.0314/problem) |
| Wrapper-level fallbacks during internal eval | 2 / 900 (0 timeouts) |
| Evolution model | Claude Opus 4.8 |

## Files

| File | What it is |
|---|---|
| `checkpoint.json` | Run config (engine, models, schedule, cost penalty params) + ELO/performance records |
| `final_report.md` | Evolution narrative across all iters |
| `test_results_final.json` | The 0.8444 number + cost breakdown |
| `test_results_final.per_problem.json` | Per-sample scores and costs (900 entries) |
| `agents/<name>/agent.py` | Each evolved agent's source |
| `iteration_010/` | Prior-iter result that iter11's evolution session read as context |
| `iteration_011/` | The iter where iter11 first scored (includes per-model cost report) |
| `evolution_output/iteration_011/` | Claude Code session log that produced iter11 |
| `evolution_output/CLAUDE.md` | Project-level memory the evolution sessions ran under |

## Lightweight inclusion

Same asymmetric inclusion as v0_0_3: two result subdirs (`iteration_010/`, `iteration_011/`), one evolution-output subdir (`evolution_output/iteration_011/`), plus `evolution_output/CLAUDE.md`. The other result subdirs and the rest of `evolution_output/` live in the local `robophd_runs/` mirror.

## Reproducing the submission

```
cd /path/to/repo
pip install litellm==1.88.1      # submission-scoring price map (see examples/asta_ds1000/README.md)
python scripts/asta_ds1000_submit.py --only v0_0_4_soft_cap_0_08
```

The script copies `agents/iter11_ds1000_tridtype_judge/agent.py` and the canonical seed into a working dir, wraps them in the two-tier `WRAPPER_TEMPLATE`, runs `astabench eval --solver agent.py --model none --split test --task DS_1000_test`, scores, and tarballs for upload.
