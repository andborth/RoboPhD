# asta_ds1000 / v0_0_5_soft_cap_0_05

RoboPhD evolution run on AstaBench's DS-1000 task. Headline submitted agent: **`iter10_literal_consensus`**. Run id: `robophd-asta_ds1000-029`. Evolved by **Claude Opus 4.8** (the evolution model; not a solver).

What's distinctive about this submission versus the prior ones (v0_0_3/v0_0_4):

- **Leanest of the high-accuracy agents.** It calls only two solver models — `gpt-5.4-2026-03-05` and `claude-sonnet-4-6` (no Gemini) — at **$0.0173/problem**: cheaper than every other RoboPhD agent scoring ~0.84+ (v0_0_4 $0.04, v0_0_3 $0.05, v0_0_1 $0.13). (The sonnet-only v0_0_2 is cheaper still at $0.01, but scores only 0.809 — a separate, lower-accuracy frontier point this does not dominate.) Both models are priced on the leaderboard, so cost computes cleanly.
- **A PROMPT-level lever, not a new mechanism.** Where v0_0_3/v0_0_4 advanced via architecture (file-mock reconstruction; dtype-rich diagnostics + 3-family diversity), this agent's edge is a "literal-reference" reading rule on a 2-model cross-agreement scaffold — see Approach.
- **First run on the new 750 evaluation budget** (up from ~620), 19 iterations; the longer budget did not overfit.

The `soft_cap_0_05` tail names the per-iteration mean-spend free-zone the run was trained under (`cost_threshold=0.05`, `cost_per_error=0.01` — now the framework default).

## Leaderboard score

**Posted 2026-06-20; board-confirmed 2026-08-05**, displayed as **0.830 @ $0.02** ($0.017 at the board's 3-decimal comparison basis). The posted row matches the `astabench score` below exactly on both axes.

| | Official (posted) | Internal dev eval |
|---|---|---|
| Accuracy (DS-1000 test, 900 samples) | **0.83** (747/900) | 0.8378 (754/900) |
| Per-problem cost | **$0.017** | $0.0173 |
| Submission name | `v0_0_5_soft_cap_0_05` / form: `RoboPhD` | — |
| Pareto position | Distinct cheap frontier point — cheapest agent reaching ~0.83, ~40% cheaper than Button. Does NOT dominate Button (0.83 < 0.836); it's the cheaper-but-lower-accuracy neighbor on the frontier (sonnet-only v0_0_2 at $0.01/0.809 sits below it) | — |
| Leaderboard | [AstaBench DS-1000 leaderboard](https://allenai-asta-bench-leaderboard.hf.space/code-execution) | — |

## Submission metadata

| Field | Value |
|---|---|
| Agent name (form) | RoboPhD |
| Openness | Open source & closed weights |
| Tools tier | Standard (uses `python_session` provided by the task) |
| Models (solver, from logs) | gpt-5.4-2026-03-05, claude-sonnet-4-6 (+ gpt-5.4-mini seed-fallback, fired 1/900) |
| URL (form) | https://github.com/andborth/RoboPhD |
| Leaderboard | [AstaBench DS-1000 leaderboard](https://allenai-asta-bench-leaderboard.hf.space/code-execution) |

Suggested agent description for the form:

> Evolved by RoboPhD with Claude Opus 4.8 as the evolution (but not a solver) model. Evolution ran under a soft cost cap of $0.05, set deliberately above our desired target because Opus evolvers typically undershoot their cap by a wide margin. The evolved agent generates two independent candidates — from GPT-5.4 and Claude Sonnet 4.6 — executes both in the sandbox with dtype/structure-aware value comparison, short-circuits on agreement, and escalates disagreements to a high-reasoning GPT-5.4 tiebreaker (with verify/repair). A "literal-reference" prompting rule steers every stage toward the most direct, literal reading of the problem and the simplest call reproducing the shown example, addressing a class of errors that cross-model agreement alone could not fix. (A GPT-5.4-mini seed agent runs only as a crash-fallback — it fired on just 1 of the 900 problems here.)

## Approach (iter10_literal_consensus)

525-line solver, ELO 1560.3 (clear pool leader). Pipeline:

1. **Primary generation.** `GPT_5_4` (reasoning=medium) produces the primary solution, with iter3's format-aware path for function-body problems.
2. **Independent cross-check.** `CLAUDE_SONNET_4_6` generates a second candidate; both run in the free sandbox with format/indent preservation, dtype/shape handling, and value-signature comparison of executed results.
3. **Agree-or-escalate.** Agreement short-circuits; disagreement escalates to a high-reasoning `GPT_5_4` tiebreaker. A verify-and-repair loop guards every submit path.
4. **Literal-reference steer.** Base generator and tiebreaker alike are told that DS-1000 references favor the most DIRECT, literal reading and the simplest call reproducing the shown example (e.g. `len(a)-rankdata(a)` over `rankdata(-a)`) — targeting the "consensus-miss" class where strong models otherwise converge on the same clever-but-wrong substitution.

`iteration_009/` is the prior-iter context the iter10-producing evolution session read; `iteration_010/` is iter10's own first scoring; `evolution_output/iteration_010/` is the Claude Code session that produced it.

## Submission resilience wrapper

The submitted `agent.py` (inside the tarball) is the same auto-generated two-tier wrapper used by prior submissions: `make_solver` from this directory's iter10 source (renamed `agent_inner.py`) with a bundled `seed_agent.py` fallback, both bounded by `asyncio.wait_for(timeout=1200)`. Internal eval recorded `n_fallback_used: 1` across the 900 samples. Wrapper template: [`scripts/asta_ds1000_submit.py`](../../../../scripts/asta_ds1000_submit.py) (`WRAPPER_TEMPLATE`).

## Internal development scoring (pre-submission)

| | Value |
|---|---|
| Score (RoboPhD-internal eval, full test) | **0.8378** (754 / 900) |
| Per-problem inference cost | $0.0173 |
| Test eval total cost | $15.58 |
| Best-agent ELO | 1560.3 |
| Train mean (6 ELO rounds) | 0.9667 |
| Distinct winning agents | 9 (of 19 iters; 27 wins — unusually contested) |
| Wrapper-level fallbacks during internal eval | 1 / 900 (0 timeouts) |
| Evolution model / cost | Claude Opus 4.8 / $40.39 ($8.72 eval + $31.67 evo) |

## Lineage (agents/)

19 agents in `agents/`: `seed_pl47_s10` → `iter2_selfcheck_cascade` … `iter9_reason_agree` → **`iter10_literal_consensus`** (submitted) → `iter11_consensus_plus` … `iter19_literal_struct_consensus` (none unseated iter10's 1560 ELO).

## Lightweight inclusion

Two result subdirs (`iteration_009/`, `iteration_010/`), one evolution-output subdir (`evolution_output/iteration_010/`), plus `evolution_output/CLAUDE.md`. The other result subdirs and the rest of `evolution_output/` live in the local `robophd_runs/` mirror.

## Reproducing the submission

```
cd /path/to/repo
pip install litellm==1.88.1      # submission-scoring price map (see examples/asta_ds1000/README.md)
python scripts/asta_ds1000_submit.py --only v0_0_5_soft_cap_0_05
```

The script copies `agents/iter10_literal_consensus/agent.py` and the canonical seed into a working dir, wraps them in the two-tier `WRAPPER_TEMPLATE`, runs `astabench eval --solver agent.py --model none --split test --task DS_1000_test`, scores, and tarballs for upload.
