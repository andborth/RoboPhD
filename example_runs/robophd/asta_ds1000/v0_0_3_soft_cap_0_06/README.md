# asta_ds1000 / v0_0_3_soft_cap_0_06

RoboPhD evolution run on AstaBench's DS-1000 task with cost-penalty free-zone threshold `--cost-threshold 0.06`. Headline submitted agent: **`iter14_filemock_adjudicate`**. Run id: `robophd-asta_ds1000-020`. First run evolved by **Claude Fable 5** (all prior asta_ds1000 runs used Opus 4.6/4.7/4.8 as the evolution model).

The `0_0_3` patch bump from `v0_0_2_soft_cap_0_08` reflects code-state changes since the prior submission (commit `2927916`), most notably Fable 5 as the evolution model (commit `939fbf6`) and the removal of the stronger-models gate (commit `7494ff7` — all registry handles now unconditional, with the cost penalty as the only discipline). The `soft_cap_0_06` tail names the per-iteration mean-spend free-zone the run was trained under.

## Leaderboard verified score

**Posted 2026-06-10; board-confirmed 2026-08-05.** Displayed as **0.853 @ $0.05**.

| | Official | Internal dev eval |
|---|---|---|
| Accuracy (DS-1000 test, 900 samples) | **0.8533** ± 0.0118 (768/900) | 0.8511 (766/900) |
| Per-problem cost | **$0.05194** | $0.0462 |
| At the board's 3-decimal comparison basis | **0.853 @ $0.052** | — |
| Submission name | `v0_0_3_soft_cap_0_06` / form: `RoboPhD` | — |
| Pareto position | **On the frontier.** Ties `v0_0_4_soft_cap_0_08` exactly on score — at 1/900 granularity only 768/900 rounds to 0.853, so both are 768/900. The leaderboard treats a 3-decimal tie as a tie for domination purposes, so v0_0_4 being cheaper ($0.037) does not displace this entry; both carry frontier trophies. | — |
| Leaderboard | [AstaBench DS-1000 leaderboard](https://allenai-asta-bench-leaderboard.hf.space/code-execution) | — |

Transfer: **+0.0022** (0.8511 → 0.8533), within noise.

Calibration: for `v0_0_1_soft_cap_0_16` the leaderboard's single batched `astabench eval` measured +2.3pp above the internal number (0.862 vs 0.8389); `v0_0_2_soft_cap_0_08` landed at parity (0.809 vs 0.8044).

## Submission metadata

| Field | Value |
|---|---|
| Agent name (form) | RoboPhD 0.0.3 |
| Openness | Open source & closed weights |
| Tools tier | Standard (uses `python_session` provided by the task) |
| Models (from logs) | claude-sonnet-4-6, gpt-5.4, gpt-5.5, gemini-3.5-flash |
| URL (form) | https://github.com/andborth/RoboPhD |
| Leaderboard | [AstaBench DS-1000 leaderboard](https://allenai-asta-bench-leaderboard.hf.space/code-execution) |

Agent description as submitted on the form:

> Evolved by RoboPhD with Claude Fable 5 as the evolution (but not a solver) model. The evolved agent generates two candidate solutions in parallel, executes both in the sandbox, and adjudicates disagreements against the execution evidence.

## Approach (iter14_filemock_adjudicate)

1614-line solver evolved from `iter13_fnsig_adjudicate`. Pipeline:

1. **Dual-model generation.** Claude Sonnet 4.6 and GPT-5.4 generate candidates in parallel under a large merged rule set distilled across 13 prior iterations.
2. **Sandbox execution** of both candidates with type/tolerance-aware value comparison (matplotlib problems compared via structured figure-introspection dumps rather than image diffs).
3. **Expected-output validation.** A cheap GPT-5.4 call checks executed values against the expected output displayed in the problem text.
4. **On consensus:** the more direct candidate is chosen, gated by a checklist critique.
5. **On disagreement:** a third Gemini Flash candidate is generated and GPT-5.5 adjudicates with full execution/expectation/third-vote evidence.
6. **Final execution + one repair pass**, with a loop-token guard and a loop-rewrite stage on all submit paths (DS-1000's "without a loop" idiom constraints).

The iteration-14 innovation over its parent is the **file-mock stage**: problems whose code calls `pd.read_csv` (or similar) on files that don't exist get those files reconstructed verbatim from data shown in the problem text before sandbox execution, converting previously un-runnable problems into executable ones.

Cost mix during training evaluation (iteration 14): GPT-5.5 (adjudicator) 54%, Sonnet 4.6 23%, GPT-5.4 17%, Gemini Flash 6% — ~$0.054/problem on the train batch, $0.046/problem on the full 900-sample test.

The `iteration_013/` subdir captures the prior-iter result the iter14-producing evolution session read as context; `iteration_014/` is iter14's own first scoring; `evolution_output/iteration_014/` is the Claude Code session that produced it.

## Model serving note (Gemini)

The agent imports `GEMINI_3_FLASH_PREVIEW`, and the registry at run time requested `google/gemini-3-flash-preview` — but Google had already routed that preview endpoint to **gemini-3.5-flash**, which is what the run's usage logs record (and what litellm billed at, $1.50/$9.00 per MTok). Between the run and this submission, the registry migrated to request `gemini-3.5-flash` explicitly (`GEMINI_3_FLASH_PREVIEW` remains as a code-only alias, which keeps this agent's import working). Net effect: the official eval hits the same served models the internal score was measured on. The run's `evolution_output/CLAUDE.md` preserves the exact (pre-migration) model table the evolution sessions saw, including the then-stale $0.50/$3.00 advertised price for the Gemini Flash handle; measured costs everywhere in this snapshot used the true 3.5-flash rates.

## Submission resilience wrapper

The submitted `agent.py` (inside the leaderboard tarball) is **not** the literal evolved iter14 source. It's the same auto-generated two-tier wrapper used by both prior submissions: `make_solver` imported from this directory's iter14 source (renamed `agent_inner.py`) with a bundled `seed_agent.py` fallback (the canonical GPT-5.4-mini seed), both tiers bounded by `asyncio.wait_for(timeout=1200)`. Defensive insurance — the internal eval finished with 0 errors, 0 timeouts, and 0 fallbacks on all 900 samples.

The wrapper template lives in [`scripts/asta_ds1000_submit.py`](../../../../scripts/asta_ds1000_submit.py) (`WRAPPER_TEMPLATE`) and is materialized at submission stage time.

## Patches relative to v0_0_2_soft_cap_0_08

Notable changes between submission SHAs that affected this run:

- `939fbf6` — **fable-5 registered as an evolution model** (and made the meta-evolution default). This run is the first asta_ds1000 evolution driven by Claude Fable 5; it spent ~2.5x per evolution session vs the prior opus-4.8 run ($4.58 vs $1.80 mean) and produced the new best agent.
- `7494ff7` — **stronger-models gate dropped** from `model_registry.py`: all handles (including GPT-5.5, used heavily by this agent as adjudicator) are unconditionally importable; the iteration-aggregate cost penalty is the only overuse discipline. Cost knobs also lock on `--resume`.
- `b2b9949` / `bca06e7` — cost-knob sidecar persistence across `--resume`.
- Post-run, pre-submission: `d5de9dc` — registry migration `GEMINI_3_FLASH_PREVIEW` → `GEMINI_3_5_FLASH` (see Model serving note above) and `CLAUDE_FABLE_5` added as a solver handle (not used by this agent).

## Lineage (agents/)

15 agents in `agents/`, in chronological order:

1. `seed_lj690bsd/agent.py` — the seed for this run (canonical GPT-5.4-mini one-shot at HEAD)
2. `iter2_crosscheck_adjudicate` … `iter13_fnsig_adjudicate` — the adjudication lineage: cross-checking, loop-token guards, tri-vote, expected-output verification, in-place mutation handling, matplotlib probing, label conversion, third-vote evidence, function-signature grounding
3. `iter14_filemock_adjudicate/agent.py` — **the submitted candidate**: adds file-mock CSV reconstruction
4. `iter15_mockverify_adjudicate/agent.py` — final iter; didn't unseat iter14

iter14 entered at iteration 14 with a perfect 40/40 across its two train rounds and held the ELO lead (1531) through the end of the run. 8 distinct agents won at least one round across the run — an unusually contested pool.

## Internal development scoring (pre-submission)

These are the numbers RoboPhD's internal subprocess-isolated evaluator measured during development. They guided the decision to submit but are NOT the canonical leaderboard score.

| | Value |
|---|---|
| Score (RoboPhD-internal eval, full test) | **0.8511** (766 / 900) |
| Per-problem inference cost | $0.0462 |
| Test eval total cost | $41.58 |
| Best-agent ELO | 1531 |
| Mean train score | 1.000 (20 train problems × 2 rounds) |
| Experiment-phase (90-sample) score | 0.8222 |
| Wrapper-level timeouts / fallbacks during internal eval | 0 / 900 |
| Evolution cost (15 iters, fable-5) | $64.09 |
| Total run cost (evolution + training evals) | $85.40 |

## Files

| File | What it is |
|---|---|
| `checkpoint.json` | Run config (engine, models, schedule, cost penalty params) |
| `final_report.md` | Evolution narrative across all 15 iters |
| `test_results_final.json` | The 0.8511 number + cost breakdown |
| `test_results_final.per_problem.json` | Per-sample scores and costs (900 entries) |
| `agents/<name>/agent.py` | Each evolved agent's source |
| `iteration_013/` | Prior-iter result that iter14's evolution session read as context |
| `iteration_014/` | The iter where iter14 first scored (includes per-model cost report) |
| `evolution_output/iteration_014/` | Claude Code session log that produced iter14 |
| `evolution_output/CLAUDE.md` | Project-level Claude Code memory file the evolution sessions ran under — documents the scoring mechanism, the $0.06 free-zone, and the exact model table evolution saw |

## Lightweight inclusion

Same asymmetric inclusion as v0_0_2: two result subdirs (`iteration_013/`, `iteration_014/`), one evolution-output subdir (`evolution_output/iteration_014/`), plus `evolution_output/CLAUDE.md`. The other 13 result subdirs and the rest of `evolution_output/` live in the local `robophd_runs/` mirror.

## Reproducing the submission

```
cd /path/to/repo
python scripts/asta_ds1000_submit.py --only v0_0_3_soft_cap_0_06
```

The script copies `agents/iter14_filemock_adjudicate/agent.py` and the canonical seed into a working dir, wraps them in the two-tier `WRAPPER_TEMPLATE`, runs `astabench eval --solver agent.py --model none --split test --task DS_1000_test`, scores, and tarballs for upload.
