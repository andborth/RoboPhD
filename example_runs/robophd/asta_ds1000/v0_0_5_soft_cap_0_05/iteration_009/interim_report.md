# Parallel Agent Research - Interim Report (Iteration 9)

## Experiment Configuration

**Run ID**: asta_ds1000_20260618_155040
**Date**: 2026-06-18 19:06:44
**Iterations**: 9/999

### System Defaults (Iteration 0)

Default values when no configuration is specified:

- agents_directory: None
- agents_per_iteration: 3
- dataset: train-filtered
- debug_log_probability: 0.02
- dollar_budget: None
- domain: external
- eval_result_cache: True
- eval_timeout: 300
- evaluation_budget: None
- evolution_model: opus-4.8 (claude-opus-4-8)
- evolution_strategy: use_your_judgment
- evolution_timeout: 3600
- examples_per_iteration: 20
- include_evolution_rankings: False
- initial_agents: ['naive']
- llm_call_timeout: 120
- max_workers: None
- meta_evolution_cadence: 3
- meta_evolution_domain: None
- meta_evolution_first_iteration: 4
- meta_evolution_model: opus-4.8 (claude-opus-4-8)
- new_agent_test_round_offset: -2
- new_agent_test_rounds: 1
- oldest_agent_wins_ties: False
- random_agent_wins_ties: True
- runs_directory: None
- strategies_directory: None
- use_weighted_random: False

### Task Configuration

Task-specific parameters (from task defaults + --task-config overrides):

- ds1000_runtime: {'cost_threshold': 0.05, 'cost_per_error': 0.01}
- file_mapping: {'agent.py': 'agent.py'}

### User Configuration (Iteration 1)

The following parameters were explicitly set, overriding defaults:

- agents_directory: ../robophd_runs/robophd/_optimize_anything_seeds
- domain: external
- eval_timeout: 1800
- evaluation_budget: 750
- evolution_model: opus-4.8 (claude-opus-4-8)
- evolution_strategy: use_your_judgment
- evolution_timeout: 3600
- examples_per_iteration: 20
- initial_agents: ['seed_pl47_s10']
- max_workers: 10
- meta_evolution_cadence: 3
- meta_evolution_domain: asta_ds1000
- meta_evolution_first_iteration: 4
- meta_evolution_model: opus-4.8 (claude-opus-4-8)
- new_agent_test_rounds: 0

### Configuration Changes by Iteration

(No configuration changes across iterations)


## Winning Agents by Evolution Strategy

#### **Use Your Judgment** (12 wins - 92.3% of all wins)
- **iter3_fmt_strong_cascade** (5 wins, created iteration 3)
- **iter2_selfcheck_cascade** (3 wins, created iteration 2)
- **iter7_agree_escalate** (2 wins, created iteration 7)
- **iter5_diverse_vote_mpl** (1 win, created iteration 5)
- **iter9_reason_agree** (1 win, created iteration 9)

#### **Initial** (1 wins - 7.7% of all wins)
- **seed_pl47_s10** (1 win, created iteration 0)

#### Summary Statistics
- **Total iterations with winners**: 9
- **Total wins counted**: 13 (includes ties)
- **Unique winning agents**: 6
- **Evolution strategies that produced winners**: 2


## Complete Performance Ranking Table

| Agent | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 | Iter 6 | Iter 7 | Iter 8 | Iter 9 | Final Elo | Mean Rank |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|--------|-----------|----------|
| iter3_fmt_strong_cascade | - | - | **#1** 75.000 | **#1** 90.000 | #2 85.000 | **#1** 80.000 | **#1** 90.000 | #3 90.000 | **#1** 80.000 | **1534** | **1.43** |
| iter7_agree_escalate | - | - | - | - | - | - | #2 85.000 | **#1** 95.000 | **#1** 80.000 | 1520 | **1.33** |
| iter9_reason_agree | - | - | - | - | - | - | - | - | **#1** 80.000 | 1503 | **1.00** |
| iter5_diverse_vote_mpl | - | - | - | - | **#1** 95.000 | #3 70.000 | - | - | - | 1500 | 2.00 |
| iter2_selfcheck_cascade | - | **#1** 75.000 | **#1** 75.000 | **#1** 90.000 | #3 60.000 | - | - | - | - | 1495 | **1.50** |
| iter6_wide_vote | - | - | - | - | - | #2 75.000 | #3 80.000 | - | - | 1473 | 2.50 |
| seed_pl47_s10 | **#1** 40.000 | #2 60.000 | #3 70.000 | - | - | - | - | - | - | 1455 | 2.50 |
| iter8_reason_cascade | - | - | - | - | - | - | - | **#1** 95.000 | - | 1319* | **1.00** |
| iter4_exec_vote_ensemble | - | - | - | **#1** 90.000 | - | - | - | - | - | 1302* | **1.00** |

### Legend:
- **#1** = 1st place (winner of iteration)
- #2, #3, etc. = 2nd, 3rd place, etc.
- Score = Average score (0-1) on that iteration's problems
- **Bold Elo/Rank** = Top performer
- `-` = Agent not tested in that iteration
- \* *Exact clone: identical per-problem scores to an existing agent on debut. Elo penalized by 200.*



### Quick Summary

| Agent | Elo | Mean Score | Tests |
|-------|-----|------------|-------|
| iter3_fmt_strong_cascade | 1534 | 84.286 | 7 |
| iter7_agree_escalate | 1520 | 86.667 | 3 |
| iter9_reason_agree | 1503 | 80.000 | 1 |
| iter5_diverse_vote_mpl | 1500 | 82.500 | 2 |
| iter2_selfcheck_cascade | 1495 | 75.000 | 4 |
| iter6_wide_vote | 1473 | 77.500 | 2 |
| seed_pl47_s10 | 1455 | 56.667 | 3 |
| iter8_reason_cascade | 1319 | 95.000 | 1 |
| iter4_exec_vote_ensemble | 1302 | 90.000 | 1 |

### Iteration Progression

| Iter | Winner(s) | Score | Evo Time | Test Time | Total Time |
|------|-----------|-------|----------|-----------|------------|
| 1 | seed_pl47_s10 | 40.000 | - | 24.6m | 24.6m |
| 2 | iter2_selfcheck_cascade | 75.000 | 6.1m | 7.1m | 13.3m |
| 3 | iter2_selfcheck_cascade, iter3_fmt_strong_cascade | 75.000 | 6.3m | 11.5m | 17.9m |
| 4 | iter2_selfcheck_cascade, iter3_fmt_strong_cascade (iter4_exec_vote_ensemble* excluded) | 90.000 | 6.9m | 14.7m | 21.6m |
| 5 | iter5_diverse_vote_mpl | 95.000 | 5.9m | 12.2m | 18.1m |
| 6 | iter3_fmt_strong_cascade | 80.000 | 3.4m | 15.9m | 19.3m |
| 7 | iter3_fmt_strong_cascade | 90.000 | 7.2m | 34.0m | 41.2m |
| 8 | iter7_agree_escalate (iter8_reason_cascade* excluded) | 95.000 | 5.2m | 18.4m | 23.5m |
| 9 | iter7_agree_escalate +2 | 80.000 | 3.6m | 12.8m | 16.4m |

\* *Exact clone: identical per-problem scores to an existing agent on debut. Elo penalized by 200; excluded from winner selection.*

## Best Agent: iter3_fmt_strong_cascade
- Elo Score: 1534
- Mean Score: 84.286


## Elo Leadership Progression

Iteration-by-iteration Elo leaders:

| Iteration | Leader | Elo | Evolution Strategy |
|-----------|--------|-----|-------------------|
| 1 | seed_pl47_s10 | 1500 | Initial |
| 2 | iter2_selfcheck_cascade | 1516 | use_your_judgment |
| 3 | iter2_selfcheck_cascade | 1530 | use_your_judgment |
| 4 | iter2_selfcheck_cascade | 1528 | use_your_judgment |
| 5 | iter5_diverse_vote_mpl | 1533 | use_your_judgment |
| 6 | iter3_fmt_strong_cascade | 1547 | use_your_judgment |
| 7 | iter3_fmt_strong_cascade | 1574 | use_your_judgment |
| 8 | iter3_fmt_strong_cascade | 1536 | use_your_judgment |
| 9 | iter3_fmt_strong_cascade | 1534 | use_your_judgment |


## Deep Focus Evolution Timing Analysis
- **Iterations with evolution**: 8/9
- **Total evolution time**: 44.7 minutes
- **Average evolution time**: 5.6 minutes

### Time Breakdown by Evolution Phase

| Phase | Avg Time | % of Evolution |
|-------|----------|----------------|
| First Draft (Round 1) | 5.0m | 89.7% |
| Reflection | 0.6m | 10.3% |
| **Total** | **5.6m** | **100.0%** |

### Evolution vs Test vs Meta-Evolution Time

| Metric | Time | Percentage | Avg/Iter | Avg/Occurrence |
|--------|------|------------|----------|----------------|
| Total Evolution Time | 44.7m | 22.8% | 5.0m | 5.6m |
| Total Test Time | 151.3m | 77.2% | 16.8m | 16.8m |
| Total Meta-Evolution Time | 0.0m | 0.0% | 0.0m | 0.0m |
| **Total** | **196.0m** | **100.0%** | **21.8m** | **-** |

- **Evolution/Test ratio**: 0.30x
  - For every minute of testing, 0.30 minutes spent on evolution

## Claude CLI Usage & Cost Analysis
- **Total Cost**: $16.05
  - **Evaluation Cost**: $2.64
  - **Evolution CLI Cost**: $13.42
- **Total CLI Calls**: 16
- **Total CLI Input Tokens**: 39,711
- **Total CLI Output Tokens**: 199,303
- **Total Cache Created**: 485,517
- **Total Cache Read**: 6,726,563

### Cost Breakdown

| Operation | Cost | % of Total | Calls | Tokens In | Tokens Out |
|-----------|------|------------|-------|-----------|------------|
| Evaluation | $2.64 | 16.4% | - | - | - |
| Evolution (CLI) | $13.42 | 83.6% | 16 | 39,711 | 199,303 |
| **Total** | **$16.05** | **100%** | **16** | **39,711** | **199,303** |

**Answer: Evolution drives 83.6% of total costs**

### Detailed Per-Iteration Costs

| Iter | Total | **Eval** | **Evo** | Iter Eval | 1st Draft | Refl | Meta | Fresh | Strategy | Meta-strategy |
|------|------|------|------|------|------|------|------|------|------|------|
| **1** | $0.01 | **$0.01** | **-** | $0.01 | - | - | - | 20 | - | - |
| **2** | $1.72 | **$0.05** | **$1.67** | $0.05 | $1.52 | $0.15 | - | 37 | use_your_judgment | - |
| **3** | $2.42 | **$0.14** | **$2.28** | $0.14 | $1.62 | $0.66 | - | 54 | use_your_judgment | - |
| **4** | $2.06 | **$0.27** | **$1.79** | $0.27 | $1.61 | $0.18 | - | 55 | use_your_judgment | - |
| **5** | $2.12 | **$0.26** | **$1.86** | $0.26 | $1.69 | $0.17 | - | 44 | use_your_judgment | - |
| **6** | $1.83 | **$0.55** | **$1.28** | $0.55 | $1.14 | $0.14 | - | 49 | use_your_judgment | - |
| **7** | $2.33 | **$0.47** | **$1.85** | $0.47 | $1.68 | $0.18 | - | 41 | use_your_judgment | - |
| **8** | $1.66 | **$0.30** | **$1.35** | $0.30 | $1.20 | $0.16 | - | 41 | use_your_judgment | - |
| **9** | $1.91 | **$0.59** | **$1.33** | $0.59 | $1.18 | $0.14 | - | 37 | use_your_judgment | - |
|------|------|------|------|------|------|------|------|------|------|------|
| **TOTAL** | **$16.05** | **$2.64** | **$13.42** | $2.64 | $11.65 | $1.76 | $0.00 | **378** | - | - |

### Evolution Cost Breakdown (Avg per evolution)

| Phase | Avg Cost | % of Evolution | Avg Tokens In | Avg Tokens Out |
|-------|----------|----------------|---------------|----------------|
| First Draft (Round 1) | $1.46 | 86.8% | 4528 | 22926 |
| Reflection | $0.22 | 13.2% | 436 | 1986 |
| **Total** | **$1.68** | **100%** | **4964** | **24913** |

### Cache Performance

- **Cache Creation**: 485,517 tokens
- **Cache Hits**: 6,726,563 tokens (93.3% hit rate)

## Zero Accuracy Cases
No zero accuracy cases encountered ✅

## Exception Failures
No exception failures encountered ✅

## Evolution Health

No artifact validation failures ✅