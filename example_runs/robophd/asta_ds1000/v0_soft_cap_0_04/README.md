# asta_ds1000 / v0_soft_cap_0_04

RoboPhD evolution run on AstaBench's DS-1000 task with cost-penalty free-zone threshold `MIN_COST_THRESHOLD = $0.04`. Headline submitted agent: **`iter13_style_aware_lean`**.

## Recorded score

| | Value |
|---|---|
| Score (RoboPhD-internal eval) | **0.8089** (728 / 900) |
| Per-problem inference cost | $0.0256 |
| Test eval total cost | $23.03 |
| Best-agent ELO | 1595 |
| Distinct iter-leaders during evolution | 5 |

> **Caveat.** This score was produced by RoboPhD's internal scoring tooling, which uses the same `inspect_evals.ds1000.ds1000_scorer` as the official AstaBench leaderboard but runs each sample in a subprocess-isolated `inspect.eval()` call (vs the leaderboard's single batched call across all 900 samples). We track the official scoring as closely as possible, but small variation is possible (sample ordering, concurrency, sandbox state). The leaderboard's verified score after `astabench eval` re-run is the canonical number.

## Submission metadata

| Field | Value |
|---|---|
| Agent name | RoboPhD evolved DS-1000 jury (4-candidate, Sonnet judge) |
| Openness | Open source, closed weights |
| Tools tier | Standard (uses `python_session` provided by the task) |
| Models | claude-sonnet-4-6, openai/gpt-5.4-2026-03-05, openai/gpt-5.4-mini, google/gemini-3-flash-preview |
| Leaderboard URL | *(filled in after submission)* |

## Approach (iter13_style_aware_lean)

1,570-line solver. **4 candidates** generated in parallel at `temperature=0` (Sonnet 4.6, GPT-5.4, GPT-5.4-mini, Gemini 3 Flash Preview), each smoke-tested in the `python_session` Docker sandbox to capture computed `repr()` or matplotlib axis properties. A **Sonnet 4.6 judge** picks among them on disagreement.

Three fast-paths bypass the judge:
- prompt-extracted expected-output match,
- matplotlib-keyword satisfaction (hatch / minor-tick / log-scale / legend / twin),
- unanimous-repr consensus across candidates.

iter13's distinguishing additions over its `iter9_mpl_aware_judge` ancestor:
- **Tokenizer-based loop-style detection** — catches `for`/`while` keywords inside list comprehensions when the prompt asks for an idiomatic/vectorized solution; judge prefers loop-free candidates and a one-shot style-retry rewrites loop-using winners.
- **Code-consensus short-circuit** — skips the judge when all candidates produced essentially the same code but smokes failed due to fixture noise.

iter13 also dropped three iter12 experiments that had hurt performance (alt-approach regeneration, 5th Haiku candidate, unproven TF/pandas system-prompt blocks) — productive backtracking by evolution.

## Lineage (agents/)

15 agents in `agents/`, in chronological order:

1. `seed_d8it1tf0/agent.py` — the seed for this run (52-line one-shot Sonnet baseline)
2. `iter2_sonnet_verify_retry/agent.py` through `iter15_generic_melt_lean/agent.py` — iters 2–15 of the evolution

iter13 won iters 13, 14, and 15 outright. iter14_dtype_aware_lean and iter15_generic_melt_lean each came in at #2 behind iter13 — they didn't displace it, indicating the iter13 architecture was stable.

## Files

| File | What it is |
|---|---|
| `checkpoint.json` | Run config (engine, models, schedule, cost penalty params) |
| `final_report.md` | Evolution narrative across all 15 iters |
| `test_results_final.json` | The 0.8089 number + cost breakdown |
| `test_results_final.per_problem.json` | Per-sample scores and costs (900 entries) |
| `agents/<name>/agent.py` | Each evolved agent's source |
| `iteration_013/` | Representative iteration (the one where iter13 emerged); see "Lightweight inclusion" below |
| `evolution_output/CLAUDE.md` | Meta-prompt that drove all evolution iterations |
| `evolution_output/iteration_013/` | Claude Code session log that produced iter13 |

## Lightweight inclusion

Only `iteration_013/` and `evolution_output/iteration_013/` are committed to git. The other 14 iterations (`iteration_001/` through `iteration_015/` minus `iteration_013/`) and the rest of `evolution_output/` (~30MB total) live on HuggingFace:

> `huggingface.co/datasets/<TBD>` *(coming soon)*

iteration_013 was picked because it's the iter where iter13_style_aware_lean first emerged and won outright — most informative single iter.

## Reproducing the submission

The agent.py files in `agents/<name>/` are runnable as Inspect-AI solvers. To reproduce the leaderboard score for iter13:

```
cd /path/to/repo
python scripts/asta_ds1000_submit.py
```

The script copies `agents/iter13_style_aware_lean/agent.py` and `examples/asta_ds1000/model_registry.py` into a working dir, runs `astabench eval --solver agent.py --model none --split test --task DS_1000_test`, scores, and tarballs for upload. See `scripts/asta_ds1000_submit.py` for the exact incantation.
