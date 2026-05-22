# asta_ds1000 / v0_0_1_soft_cap_0_16

RoboPhD evolution run on AstaBench's DS-1000 task with cost-penalty free-zone threshold `MIN_COST_THRESHOLD = $0.16` (~4× the prior `v0_soft_cap_0_04` threshold — paired with `--allow-stronger-models` to unlock Opus 4.7 and Gemini 3.1 Pro Preview as solver handles). Headline submitted agent: **`iter10_idiomatic_loop_guard_v1`**. Run id: `robophd-asta_ds1000-007`.

## Leaderboard verified score

| | Value |
|---|---|
| Accuracy (AstaBench leaderboard) | **86.2%** (0.8622) |
| Per-problem inference cost | **$0.13** ($0.1273) |
| Submission name | `v0_0_1_soft_cap_0_16` |
| Pareto position | **#1 accuracy** on the entire DS-1000 leaderboard |
| Leaderboard | [AstaBench DS-1000 leaderboard](https://allenai-asta-bench-leaderboard.hf.space/code-execution) |

This is the canonical, externally-verified number from a single batched `astabench eval` run on the 900-sample test split. The submitted `agent.py` is a thin two-tier wrapper (see [Submission resilience wrapper](#submission-resilience-wrapper) below) around the evolved `iter10_idiomatic_loop_guard_v1` source plus a bundled seed-fallback agent. The development-time internal evaluator measured a slightly different number — see [Internal development scoring (pre-submission)](#internal-development-scoring-pre-submission) near the bottom of this page for the development numbers and the timeout investigation that prompted the resubmission.

## Submission metadata

| Field | Value |
|---|---|
| Agent name | RoboPhD evolved DS-1000 quad-diverse ensemble + Opus critic |
| Openness | Open source, closed weights |
| Tools tier | Standard (uses `python_session` provided by the task) |
| Models | claude-sonnet-4-6, claude-opus-4-7, gpt-5.4, gemini-3.1-pro-preview |
| Leaderboard | [AstaBench DS-1000 leaderboard](https://allenai-asta-bench-leaderboard.hf.space/code-execution) |

## Approach (iter10_idiomatic_loop_guard_v1)

~1,400-line solver. **Quad-diverse candidate ensemble** generated in parallel (Sonnet 4.6, Opus 4.7, GPT-5.4, Gemini 3.1 Pro Preview), each smoke-tested in the `python_session` Docker sandbox to capture computed `repr()` or matplotlib axis properties. An **Opus 4.7 critic** picks among them on disagreement.

Two fast-paths bypass the critic:
- prompt-extracted expected-output match,
- unanimous-repr consensus across candidates.

iter10's distinguishing addition over its `iter9_iter7_polish_v1` ancestor:

- **LOOP-SCRUB rewrite path** — when the prompt asks for an idiomatic / vectorized solution but a candidate produced a winner with explicit `for`/`while` loops, iter10 asks Opus to rewrite the loop-using winner into an idiomatic library call and sandbox-verifies the rewrite against the original's `repr()`. If equivalent, the rewrite ships; if not, the original (loop-using) winner ships.

The `iteration_009/` subdir captures the prior-iter result that iter10's evolution session was reading as context; `iteration_010/` is iter10's own first scoring; `evolution_output/iteration_010/` is the Claude Code session that produced it.

## Submission resilience wrapper

The submitted `agent.py` (inside the leaderboard tarball) is **not** the literal evolved iter10 source. It's a small auto-generated wrapper that imports `make_solver` from both this directory's iter10 source (renamed `agent_inner.py`) and from a bundled `seed_agent.py` (a copy of the canonical GPT-5.4-mini seed), providing a **two-tier** safety net:

```python
try:
    return await inner(state, generate)        # primary: iter10
except Exception as primary:
    try:
        return await seed(state, generate)     # tier 2: GPT-5.4-mini seed
    except Exception as fallback:
        state.output.completion = ""           # scorer marks "I" → 0
        return state
```

The seed fallback was **specifically motivated by this run**: 25 of the 900 samples timed out at the 1200s wrapper-level cap during the original eval, and the post-mortem traced the wedge to orphaned grandchild processes holding the parent's stdout/stderr pipes open after a `subprocess.kill()`. Two fixes ship with the resubmission:

- Subprocess hardening (commit `438249b`): `start_new_session=True` + `os.killpg(...SIGKILL)` on timeout, plus a bounded 30s post-kill drain, kill the entire process group so grandchildren can't wedge the parent.
- Two-tier wrapper (commit `0370bad`): on any solver exception (including the iter10 LOOP-SCRUB path's Opus-call failure modes), the seed gets a shot at recovering the score before emitting an empty completion.

The wrapper template lives in [`scripts/asta_ds1000_submit.py`](../../../../scripts/asta_ds1000_submit.py) (`WRAPPER_TEMPLATE`) and is materialized into the working dir as `agent.py` at submission stage time, with the iter10 source renamed to `agent_inner.py` and the seed copied in as `seed_agent.py`.

`Inspect-AI 0.3.220` introduced `--score-on-error`, a framework-level equivalent of the empty-completion fallback. AstaBench currently pins to `inspect_ai==0.3.203`; we'll switch to the native flag when AstaBench bumps. Until then the wrapper is the cleanest path and gives the seed-fallback semantics besides.

## Pareto positioning

At the leaderboard-verified **86.2% (0.8622) / $0.13**, this submission achieves the **highest accuracy on the entire AstaBench DS-1000 leaderboard** and strictly outperforms the previous accuracy leader, `ReAct / gemini-3.1-pro-preview` (84.9% / $0.25), at roughly half the cost.

## Lineage (agents/)

15 agents in `agents/`, in chronological order:

1. `seed_40_esiah/agent.py` — the seed for this run (the canonical GPT-5.4-mini one-shot at HEAD)
2. `iter2_careful_verifier_v1/agent.py` — added `python_session` sanity check + retry-on-failure
3. `iter3_ensemble_judge_v1/agent.py` — first ensemble shape (3 candidates + judge)
4. `iter4_robust_consensus_v1/agent.py` — consensus path with explicit `repr()` comparison
5. `iter5_triple_consensus_v1/agent.py` — three-candidate consensus
6. `iter6_robust_critic_v1/agent.py` — added LLM critic for tiebreaks
7. `iter7_quad_diverse_critic_v1/agent.py` — quad-diverse ensemble (Sonnet/Opus/GPT/Gemini)
8. `iter8_simpler_idiomatic_v1/agent.py` — first idiomatic-library-call awareness
9. `iter9_iter7_polish_v1/agent.py` — backed off iter8's experiments, polished iter7's quad-diverse critic
10. `iter10_idiomatic_loop_guard_v1/agent.py` — **the submitted candidate**: added LOOP-SCRUB rewrite path
11. `iter11_iter10_consensus_shortcut_v1/agent.py` through `iter15_iter14_cluster_default_v1/agent.py` — later iters that didn't unseat iter10 as best

iter10 won the train rounds 6 times. Later iters explored consensus shortcuts, load-data stubs, cluster hints, and timeout-guard tweaks but didn't outscore iter10 across the train set.

## Internal development scoring (pre-submission)

These are the numbers RoboPhD's internal subprocess-isolated evaluator measured during development. They guided the resubmission decision but are NOT the canonical leaderboard score — see [Leaderboard verified score](#leaderboard-verified-score) at the top of this page.

| | Value |
|---|---|
| Score (RoboPhD-internal eval, full test) | **0.8389** (755 / 900) |
| Per-problem inference cost | $0.0980 |
| Test eval total cost | $88.22 |
| Best-agent ELO | 1657 |
| Mean train score | 0.9832 (90 train problems × 6 rounds) |
| Wrapper-level timeouts (pre-fix) | 25 / 900 — at the 1200s subprocess wall-clock cap |

> **Caveat.** RoboPhD's internal scoring tooling uses the same `inspect_evals.ds1000.ds1000_scorer` as the official AstaBench leaderboard but runs each sample in a subprocess-isolated `inspect.eval()` call (vs the leaderboard's single batched call across all 900 samples). 25 of the 900 samples hit the wrapper-level 1200-second timeout and were scored 0 in this internal run, dragging the recorded 0.8389 below the timeout-corrected projection of 0.8629. The killpg subprocess hardening (commit `438249b`) and the two-tier seed-fallback wrapper (commit `0370bad`, see [Submission resilience wrapper](#submission-resilience-wrapper)) shipped with the leaderboard submission and produced the leaderboard-verified 0.8622 — slightly above the timeout-corrected projection.

## Files

| File | What it is |
|---|---|
| `checkpoint.json` | Run config (engine, models, schedule, cost penalty params) |
| `final_report.md` | Evolution narrative across all 15 iters |
| `test_results_final.json` | The 0.8389 number + cost breakdown |
| `test_results_final.per_problem.json` | Per-sample scores and costs (900 entries) |
| `agents/<name>/agent.py` | Each evolved agent's source |
| `iteration_009/` | Prior-iter result that iter10's evolution session read as context |
| `iteration_010/` | The iter where iter10 first scored |
| `evolution_output/iteration_010/` | Claude Code session log that produced iter10 |

## Lightweight inclusion

Asymmetric inclusion to keep the diff modest while preserving the iter10 provenance trail:

- Result subdirs: **two committed** (`iteration_009/` and `iteration_010/`). 009 because the iter10-producing evolution session read it as context; 010 because that's where iter10 itself first scored.
- Evolution-output subdirs: **one committed** (`evolution_output/iteration_010/`). The 010 session itself is what's interesting; it reads 009's result file directly during the run, so 009's evolution_output isn't needed in this snapshot.

The other 14 result subdirs (`iteration_001/`, `iteration_002/`, …, `iteration_015/` minus 009/010) and the rest of `evolution_output/` (~25-30MB total) live in the local `robophd_runs/` mirror and may move to a HuggingFace dataset later:

> `huggingface.co/datasets/<TBD>` *(coming soon)*

## Reproducing the submission

```
cd /path/to/repo
python scripts/asta_ds1000_submit.py
```

The script copies `agents/iter10_idiomatic_loop_guard_v1/agent.py` and the canonical seed into a working dir, wraps them in the two-tier `WRAPPER_TEMPLATE`, runs `astabench eval --solver agent.py --model none --split test --task DS_1000_test`, scores, and tarballs for upload. See `scripts/asta_ds1000_submit.py` for the exact incantation.
