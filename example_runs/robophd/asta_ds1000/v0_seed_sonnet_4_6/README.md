# asta_ds1000 / v0_seed_sonnet_4_6

One-shot Claude Sonnet 4.6 baseline submitted alongside the evolved iter13 agent. Establishes the meaningful lower bound for evolved agents that have access to Sonnet 4.6.

> **This seed is structurally identical to [`v0_seed_gpt54_mini`](../v0_seed_gpt54_mini/agent.py) — and to the canonical seed at [`examples/asta_ds1000/seeds/baseline/agent.py`](../../../../examples/asta_ds1000/seeds/baseline/agent.py) — except the solver model is `CLAUDE_SONNET_4_6` instead of `GPT_5_4_MINI`.** All other differences are comments and docstring text. The two submitted seeds therefore isolate the model-choice axis on the Pareto curve: same one-shot pattern, same `temperature=0`, same `<code>...</code>` wrapping, just a different model handle.

## Recorded score

| | Value |
|---|---|
| Score (RoboPhD-internal eval) | **0.7378** (664 / 900) |
| Per-problem inference cost | $0.00243 |
| Test eval total cost | $2.18 |
| Solver model | `anthropic/claude-sonnet-4-6` |
| Agent size | 52 lines |

> **Caveat.** This score was produced by RoboPhD's internal scoring tooling, which uses the same `inspect_evals.ds1000.ds1000_scorer` as the official AstaBench leaderboard but runs each sample in a subprocess-isolated `inspect.eval()` call (vs the leaderboard's single batched call across all 900 samples). We track the official scoring as closely as possible, but small variation is possible. The leaderboard's verified score after `astabench eval` re-run is the canonical number.

## Submission metadata

| Field | Value |
|---|---|
| Agent name | Sonnet 4.6 one-shot DS-1000 baseline |
| Openness | Open source, closed weights |
| Tools tier | Standard |
| Leaderboard URL | *(filled in after submission)* |

## Approach

52-line one-shot solver. Sends the problem prompt to Claude Sonnet 4.6 at `temperature=0`, wraps the response in `<code>...</code>` tags, and returns. No self-check, no `python_session` calls, no library-specific scaffolding, no per-problem detectors. See `agent.py`. (As noted at the top, code-equivalent to the GPT-5.4-mini seed except for the model handle.)

## Pareto positioning

At 0.7378 / $0.00243 this seed sits on the leaderboard's Pareto frontier: ~2× cheaper than ReAct/o3 (0.749 / $0.01) with -1.1pp on score (both on frontier); +2.8pp over ReAct/GPT-5-Mini (0.710 / $0.00 displayed). Cost is well under the $0.01 cost-penalty free-zone threshold used in the v0_soft_cap_0_04 evolution.

## Provenance

This `agent.py` is copied verbatim from:

```
robophd_runs/robophd/asta_ds1000_20260507_032920/agents/seed_sonnet_4_6/agent.py
```

That source run was a standalone seed-baseline evaluation (not a multi-iter evolution) — created to establish the Sonnet 4.6 baseline number that runs from 2026-05-06 onward could compare evolved-agent lifts against.
