# asta_ds1000 / v0_seed_gpt54_mini

One-shot GPT-5.4-mini baseline submitted alongside the evolved iter13 agent. The cheapest of our three submitted points.

## Recorded score

| | Value |
|---|---|
| Score (RoboPhD-internal eval) | **0.6933** (624 / 900) |
| Per-problem inference cost | $0.00052 |
| Test eval total cost | $0.47 |
| Solver model | `openai/gpt-5.4-mini` |
| Agent size | 58 lines |

> **Caveat.** This score was produced by RoboPhD's internal scoring tooling, which uses the same `inspect_evals.ds1000.ds1000_scorer` as the official AstaBench leaderboard but runs each sample in a subprocess-isolated `inspect.eval()` call (vs the leaderboard's single batched call across all 900 samples). We track the official scoring as closely as possible, but small variation is possible. The leaderboard's verified score after `astabench eval` re-run is the canonical number.

## Submission metadata

| Field | Value |
|---|---|
| Agent name | GPT-5.4-mini one-shot DS-1000 baseline |
| Openness | Open source, closed weights |
| Tools tier | Standard |
| Leaderboard URL | *(filled in after submission)* |

## Approach

58-line one-shot solver. Sends the problem prompt to GPT-5.4-mini at `temperature=0`, wraps the response in `<code>...</code>` tags, and returns. No self-check, no `python_session` calls, no library-specific scaffolding, no per-problem detectors. See `agent.py`.

## Pareto positioning

At 0.6933 / $0.00052 this is the cheapest of our three submitted points. Below ReAct/GPT-5-Mini in score (0.710 at $0.00 displayed cost); whether on the leaderboard's Pareto frontier depends on whether ReAct/GPT-5-Mini's true (un-displayed) cost is greater or less than $0.00052.

## Provenance

This `agent.py` is copied verbatim from:

```
robophd_runs/robophd/asta_ds1000_20260505_145358/agents/seed_b13o9jud/agent.py
```

The source run's RoboPhD-internal seed name was `seed_b13o9jud` (b13o9jud is a random per-run hash — the v0_seed_gpt54_mini name in this snapshot is more descriptive).

**Code-equivalent to the current canonical seed** at `examples/asta_ds1000/seeds/baseline/agent.py`. The two files differ only in comments and docstring (the canonical seed's docstring was tightened over time to remove evolution-strategy hints, and its model-registry comment lists all six current handles). Executable code is identical: a one-shot `GPT_5_4_MINI.generate()` call at `temperature=0`, response wrapped in `<code>...</code>` tags.
