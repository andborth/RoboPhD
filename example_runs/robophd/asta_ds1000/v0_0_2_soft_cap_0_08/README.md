# asta_ds1000 / v0_0_2_soft_cap_0_08

RoboPhD evolution run on AstaBench's DS-1000 task with cost-penalty free-zone threshold `MIN_COST_THRESHOLD = $0.08` (the default; paired with `--allow-stronger-models` to unlock Opus 4.7 as a fallback handle). Headline submitted agent: **`iter4_ds1000_idiom_probe`**. Run id: `robophd-asta_ds1000-009`.

The `0_0_2` patch bump from `v0_0_1_soft_cap_0_16` reflects code-state changes since the prior submission (commit `eb7dd11`), most notably the iteration-aggregate cost-penalty mechanism (commit `b453ee0`). The `soft_cap_0_08` tail names the per-iteration mean-spend free-zone the run was trained under.

## Leaderboard verified score

| | Value |
|---|---|
| Accuracy (AstaBench leaderboard) | **80.9%** (0.8089) |
| Per-problem inference cost | **$0.01** ($0.01181) |
| Submission name | `v0_0_2_soft_cap_0_08` |
| Pareto position | Lowest-cost submission above 80% accuracy; strictly outperforms five higher-cost submissions |
| Leaderboard | [AstaBench DS-1000 leaderboard](https://allenai-asta-bench-leaderboard.hf.space/code-execution) |

This is the canonical, externally-verified number from a single batched `astabench eval` run on the 900-sample test split. The submitted `agent.py` is a thin two-tier wrapper (see [Submission resilience wrapper](#submission-resilience-wrapper) below) around the evolved `iter4_ds1000_idiom_probe` source plus a bundled seed-fallback agent. The development-time internal evaluator measured a slightly lower number — see [Internal development scoring (pre-submission)](#internal-development-scoring-pre-submission) near the bottom of this page.

## Submission metadata

| Field | Value |
|---|---|
| Agent name | RoboPhD evolved DS-1000 idiom probe (Sonnet primary + Opus fallback) |
| Openness | Open source, closed weights |
| Tools tier | Standard (uses `python_session` provided by the task) |
| Models | claude-sonnet-4-6, claude-opus-4-7 |
| Leaderboard | [AstaBench DS-1000 leaderboard](https://allenai-asta-bench-leaderboard.hf.space/code-execution) |

## Approach (iter4_ds1000_idiom_probe)

561-line solver evolved from `iter3_ds1000_format_aware`. Single primary model (**Sonnet 4.6 with `reasoning_effort="high"`**) generates the candidate solution; the `python_session` Docker sandbox runs it under the problem's hidden-test invocation pattern; up to **3 sandbox-verify-and-repair** iterations correct actionable errors. Two specialized layers run before/inside this loop:

1. **Idiom-constraint detection.** Detects DS-1000's "without a loop", "vectorized", "most idiomatic", "elegant", "pythonic" phrasings → adds a stronger system message, AST-scans the candidate for `for`/`while` tokens (including comprehensions and generator expressions), and triggers up to 2 loop-free regenerations if found.
2. **Invent-signature function detection.** When the prompt asks the agent to define a function the skeleton doesn't declare, prompts for a single-argument signature (matching the example input pattern the hidden test will use) and actively probes the function in the sandbox the way the grader will, rather than just running for tracebacks.

A **CLAUDE_OPUS_4_7 escalation** fires only when the Sonnet repair loop exits with the sandbox still reporting an actionable (non-environmental) error — this is a narrow path that almost never triggers in practice (0/200 fallback fires across the run's training evaluations).

A final **idiom-guard regeneration** sweeps the chosen solution one more time if it still contains a loop on a problem with idiom-constraint signal.

The architecture is meaningfully simpler than the v0_0_1_soft_cap_0_16 submission's quad-diverse ensemble + Opus critic: one primary model, narrow conditional escalation, no repr-based consensus voting. This gives a much lower per-problem inference cost (~$0.012 vs ~$0.13) at the cost of ~6pp lower score.

The `iteration_003/` subdir captures the prior-iter result that iter4's evolution session was reading as context; `iteration_004/` is iter4's own first scoring; `evolution_output/iteration_004/` is the Claude Code session that produced it.

## Submission resilience wrapper

The submitted `agent.py` (inside the leaderboard tarball) is **not** the literal evolved iter4 source. It's a small auto-generated wrapper that imports `make_solver` from both this directory's iter4 source (renamed `agent_inner.py`) and from a bundled `seed_agent.py` (a copy of the canonical GPT-5.4-mini seed), providing a **two-tier** safety net:

```python
try:
    return await inner(state, generate)        # primary: iter4
except Exception as primary:
    try:
        return await seed(state, generate)     # tier 2: GPT-5.4-mini seed
    except Exception as fallback:
        state.output.completion = ""           # scorer marks "I" → 0
        return state
```

Both tiers are bounded by `asyncio.wait_for(timeout=1200)` so neither can wedge the eval indefinitely. For this submission the wrapper is essentially defensive insurance — the internal eval finished with 0 errors and 0 timeouts on all 900 samples — but using the same wrapper as `v0_0_1_soft_cap_0_16` keeps the two submissions methodologically comparable.

The wrapper template lives in [`scripts/asta_ds1000_submit.py`](../../../../scripts/asta_ds1000_submit.py) (`WRAPPER_TEMPLATE`) and is materialized into the working dir as `agent.py` at submission stage time, with the iter4 source renamed to `agent_inner.py` and the seed copied in as `seed_agent.py`.

## Patches relative to v0_0_1_soft_cap_0_16

13 commits between submission SHAs. The `scripts/asta_ds1000_submit.py` submission script and `WRAPPER_TEMPLATE` are unchanged. Notable changes that affected the run:

- `b453ee0` — **iteration-level cost penalty** via evaluator aggregator hook. Replaces v0_0_1's per-example penalty with a mean-across-the-iteration's-batch penalty. The free-zone threshold a single problem must clear shifted from "your per-example spend < $0.16" to "your batch's mean spend < $0.08", giving evolved agents budget headroom on individual hard problems while keeping average spend honest. This is the dominant scoring-mechanism change between the two runs and motivates the new `soft_cap_0_08` tag.
- `a374e54` — Gemini 3.1 flash-lite **preview → GA migration** in `examples/asta_ds1000/model_registry.py`. Doesn't affect this submission (iter4 doesn't use Gemini) but the registry shape changed.
- `5d654f0` — `--resume` no longer clobbers original-run settings.
- `0d37586`, `7a52ad0` — post-eval error-analysis tooling fixes.
- Other commits: README math fixes, defaults tweaks, no submission-path impact.

## Pareto positioning

At the leaderboard-verified **80.9% (0.8089) / $0.01**, this submission strictly Pareto-dominates **five existing AstaBench leaderboard entries** — every higher-cost submission scoring below 80.9%, including all four GPT-5-based agents on the board:

| Dominated entry | Cost | Accuracy |
|---|---|---|
| `ReAct / claude-opus-4-7` | $0.06 | 78.6% |
| `EvoScientist-Code (GPT-5)` | $0.03 | 78.4% |
| `ReAct / GPT-5` | $0.02 | 78.0% |
| `Smolagents Coder / GPT-5` | $0.02 | 75.7% |
| `ReAct / Claude Sonnet 4` | $0.04 | 75.6% |

Together with the companion `v0_0_1_soft_cap_0_16` submission (86.2% / $0.13) at the high-accuracy end of the leaderboard, the two RoboPhD agents form a two-rung Pareto curve spanning roughly an order of magnitude in cost on DS-1000.

## Lineage (agents/)

15 agents in `agents/`, in chronological order:

1. `seed_yyg6m9ud/agent.py` — the seed for this run (the canonical GPT-5.4-mini one-shot at HEAD)
2. `iter2_ds1000_verify_repair/agent.py` — switched the primary model from `GPT_5_4_MINI` to `CLAUDE_SONNET_4_6` (`reasoning_effort="medium"`); added `python_session` sanity check + verify-and-repair pipeline
3. `iter3_ds1000_format_aware/agent.py` — bumped Sonnet's `reasoning_effort` to `"high"`; added deterministic format/indent preservation
4. `iter4_ds1000_idiom_probe/agent.py` — **the submitted candidate**: added idiom-constraint detection, invent-signature handling, and the narrow Opus escalation
5. `iter5_ds1000_consensus/agent.py` through `iter15_ds1000_audit_split/agent.py` — later iters (consensus voting, dtype anchoring, trap audits, ground-truth best-of, opus-literal, audit splits, etc.) that didn't unseat iter4 as best

iter4 won the train rounds 8 times (the most of any agent) and held the ELO lead for 9 of the 15 iterations. Later iters explored consensus and verification variants but didn't outscore iter4 across the train set.

## Internal development scoring (pre-submission)

These are the numbers RoboPhD's internal subprocess-isolated evaluator measured during development. They guided the decision to submit but are NOT the canonical leaderboard score — see [Leaderboard verified score](#leaderboard-verified-score) at the top of this page.

| | Value |
|---|---|
| Score (RoboPhD-internal eval, full test) | **0.8044** (724 / 900) |
| Per-problem inference cost | $0.0123 |
| Test eval total cost | $11.09 |
| Best-agent ELO | 1552 |
| Mean train score | 0.9300 (90 train problems × 10 rounds) |
| Wrapper-level timeouts | 0 / 900 |
| Per-problem errors / fallbacks during internal eval | 0 |

> **Caveat.** RoboPhD's internal scoring tooling uses the same `inspect_evals.ds1000.ds1000_scorer` as the official AstaBench leaderboard but runs each sample in a subprocess-isolated `inspect.eval()` call (vs the leaderboard's single batched call across all 900 samples). The internal eval was clean — 0 timeouts, 0 fallbacks — so the AstaBench-measured 0.8089 tracked within +0.45pp of this 0.8044 internal number, well within sampling noise.

## Files

| File | What it is |
|---|---|
| `checkpoint.json` | Run config (engine, models, schedule, cost penalty params) |
| `final_report.md` | Evolution narrative across all 15 iters |
| `test_results_final.json` | The 0.8044 number + cost breakdown |
| `test_results_final.per_problem.json` | Per-sample scores and costs (900 entries) |
| `agents/<name>/agent.py` | Each evolved agent's source |
| `iteration_003/` | Prior-iter result that iter4's evolution session read as context |
| `iteration_004/` | The iter where iter4 first scored |
| `evolution_output/iteration_004/` | Claude Code session log that produced iter4 |
| `evolution_output/CLAUDE.md` | Project-level Claude Code memory file the evolution sessions ran under — describes the iteration-aggregate scoring + $0.08 free-zone the run was trained under |

## Lightweight inclusion

Asymmetric inclusion to keep the diff modest while preserving the iter4 provenance trail:

- Result subdirs: **two committed** (`iteration_003/` and `iteration_004/`). 003 because the iter4-producing evolution session read it as context; 004 because that's where iter4 itself first scored.
- Evolution-output subdirs: **one committed** (`evolution_output/iteration_004/`). The 004 session itself is what's interesting; it reads 003's result file directly during the run, so 003's evolution_output isn't needed in this snapshot.
- New in v0_0_2 vs v0_0_1: `evolution_output/CLAUDE.md` is committed too. It's the project-level memory file Claude saw during every evolution session and documents the run's domain background, scoring mechanism, and cost penalty — small (11 KB) and useful for understanding what the evolutionary AI was optimizing for.

The other 13 result subdirs (`iteration_001/`, `iteration_002/`, `iteration_005/`, …, `iteration_015/` minus 003/004) and the rest of `evolution_output/` (~25-30MB total) live in the local `robophd_runs/` mirror and may move to a HuggingFace dataset later:

> `huggingface.co/datasets/<TBD>` *(coming soon)*

## Reproducing the submission

```
cd /path/to/repo
python scripts/asta_ds1000_submit.py
```

The script copies `agents/iter4_ds1000_idiom_probe/agent.py` and the canonical seed into a working dir, wraps them in the two-tier `WRAPPER_TEMPLATE`, runs `astabench eval --solver agent.py --model none --split test --task DS_1000_test`, scores, and tarballs for upload. See `scripts/asta_ds1000_submit.py` for the exact incantation.
