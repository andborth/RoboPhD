# Parallel Agent Research - Final Report

## Experiment Configuration

**Run ID**: asta_ds1000_20260511_162205
**Date**: 2026-05-11 20:46:37
**Iterations**: 15

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
- evolution_model: opus-4.7 (claude-opus-4-7)
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
- meta_evolution_model: opus-4.7 (claude-opus-4-7)
- new_agent_test_round_offset: -2
- new_agent_test_rounds: 1
- oldest_agent_wins_ties: False
- random_agent_wins_ties: True
- runs_directory: None
- strategies_directory: None
- use_weighted_random: False

### Task Configuration

Task-specific parameters (from task defaults + --task-config overrides):

- file_mapping: {'agent.py': 'agent.py'}

### User Configuration (Iteration 1)

The following parameters were explicitly set, overriding defaults:

- agents_directory: ../robophd_runs/robophd/_optimize_anything_seeds
- domain: external
- eval_timeout: 1200
- evaluation_budget: 999999
- evolution_model: opus-4.7 (claude-opus-4-7)
- evolution_strategy: use_your_judgment
- evolution_timeout: 3600
- examples_per_iteration: 20
- initial_agents: ['seed_40_esiah']
- max_workers: 12
- meta_evolution_cadence: 3
- meta_evolution_domain: asta_ds1000
- meta_evolution_first_iteration: 4
- new_agent_test_rounds: 0

### Configuration Changes by Iteration

(No configuration changes across iterations)


## Winning Agents by Evolution Strategy

#### **Use Your Judgment** (16 wins - 88.9% of all wins)
- **iter10_idiomatic_loop_guard_v1** (6 wins, created iteration 10)
- **iter3_ensemble_judge_v1** (3 wins, created iteration 3)
- **iter2_careful_verifier_v1** (2 wins, created iteration 2)
- **iter4_robust_consensus_v1** (2 wins, created iteration 4)
- **iter7_quad_diverse_critic_v1** (2 wins, created iteration 7)
- **iter9_iter7_polish_v1** (1 win, created iteration 9)

#### **Initial** (2 wins - 11.1% of all wins)
- **seed_40_esiah** (2 wins, created iteration 0)

#### Summary Statistics
- **Total iterations with winners**: 15
- **Total wins counted**: 18 (includes ties)
- **Unique winning agents**: 7
- **Evolution strategies that produced winners**: 2


## Complete Performance Ranking Table

| Agent | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 | Iter 6 | Iter 7 | Iter 8 | Iter 9 | Iter 10 | Iter 11 | Iter 12 | Iter 13 | Iter 14 | Iter 15 | Final ELO | Mean Rank |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|-----------|----------|
| iter10_idiomatic_loop_guard_v1 | - | - | - | - | - | - | - | - | - | **#1** 99.996 | **#1** 99.987 | **#1** 94.998 | **#1** 99.996 | **#1** 94.972 | **#1** 99.981 | **1657** | **1.00** |
| iter12_iter10_load_data_stub_v1 | - | - | - | - | - | - | - | - | - | - | - | #2 94.998 | - | - | #2 99.956 | 1511 | 2.00 |
| iter14_iter13_timeout_guard_v1 | - | - | - | - | - | - | - | - | - | - | - | - | - | #2 94.902 | - | 1505 | 2.00 |
| iter4_robust_consensus_v1 | - | - | - | #2 90.000 | **#1** 95.000 | **#1** 95.000 | #2 85.000 | - | - | #2 95.000 | - | - | - | - | - | 1504 | **1.60** |
| iter9_iter7_polish_v1 | - | - | - | - | - | - | - | - | **#1** 94.992 | #3 94.996 | - | - | - | - | - | 1501 | 2.00 |
| iter2_careful_verifier_v1 | - | **#1** 75.000 | **#1** 90.000 | #3 70.000 | - | - | - | - | - | - | - | - | - | - | - | 1497 | **1.67** |
| iter3_ensemble_judge_v1 | - | - | #2 85.000 | **#1** 95.000 | **#1** 95.000 | **#1** 95.000 | #2 85.000 | #2 90.000 | #3 85.000 | - | - | - | - | - | - | 1493 | **1.71** |
| iter7_quad_diverse_critic_v1 | - | - | - | - | - | - | **#1** 89.992 | **#1** 94.995 | #2 89.997 | - | #2 94.995 | #3 84.997 | #3 94.999 | - | - | 1492 | 2.00 |
| iter13_iter12_cluster_hint_v1 | - | - | - | - | - | - | - | - | - | - | - | - | #2 99.995 | #3 89.934 | - | 1478 | 2.50 |
| iter15_iter14_cluster_default_v1 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | #3 99.917 | 1475 | 3.00 |
| iter11_iter10_consensus_shortcut_v1 | - | - | - | - | - | - | - | - | - | - | #3 84.993 | - | - | - | - | 1473 | 3.00 |
| iter8_simpler_idiomatic_v1 | - | - | - | - | - | - | - | #3 84.997 | - | - | - | - | - | - | - | 1471 | 3.00 |
| iter5_triple_consensus_v1 | - | - | - | - | #3 94.996 | - | - | - | - | - | - | - | - | - | - | 1470 | 3.00 |
| seed_40_esiah | **#1** 75.000 | **#1** 75.000 | #3 60.000 | - | - | - | - | - | - | - | - | - | - | - | - | 1469 | 2.00 |
| iter6_robust_critic_v1 | - | - | - | - | - | **#1** 95.000 | - | - | - | - | - | - | - | - | - | 1303* | **1.00** |

### Legend:
- **#1** = 1st place (winner of iteration)
- #2, #3, etc. = 2nd, 3rd place, etc.
- Score = Average score (0-1) on that iteration's problems
- **Bold ELO/Rank** = Top performer
- `-` = Agent not tested in that iteration
- \* *Exact clone: identical per-problem scores to an existing agent on debut. ELO penalized by 200.*



### Quick Summary

| Agent | ELO | Mean Score | Tests |
|-------|-----|------------|-------|
| iter10_idiomatic_loop_guard_v1 | 1657 | 98.322 | 6 |
| iter12_iter10_load_data_stub_v1 | 1511 | 97.477 | 2 |
| iter14_iter13_timeout_guard_v1 | 1505 | 94.902 | 1 |
| iter4_robust_consensus_v1 | 1504 | 92.000 | 5 |
| iter9_iter7_polish_v1 | 1501 | 94.994 | 2 |
| iter2_careful_verifier_v1 | 1497 | 78.333 | 3 |
| iter3_ensemble_judge_v1 | 1493 | 90.000 | 7 |
| iter7_quad_diverse_critic_v1 | 1492 | 91.662 | 6 |
| iter13_iter12_cluster_hint_v1 | 1478 | 94.965 | 2 |
| iter15_iter14_cluster_default_v1 | 1475 | 99.917 | 1 |
| iter11_iter10_consensus_shortcut_v1 | 1473 | 84.993 | 1 |
| iter8_simpler_idiomatic_v1 | 1471 | 84.997 | 1 |
| iter5_triple_consensus_v1 | 1470 | 94.996 | 1 |
| seed_40_esiah | 1469 | 70.000 | 3 |
| iter6_robust_critic_v1 | 1303 | 95.000 | 1 |

### Iteration Progression

| Iter | Winner(s) | Score | Evo Time | Test Time | Total Time |
|------|-----------|-------|----------|-----------|------------|
| 1 | seed_40_esiah | 75.000 | - | 1.1m | 1.1m |
| 2 | seed_40_esiah, iter2_careful_verifier_v1 | 75.000 | 9.8m | 2.6m | 12.4m |
| 3 | iter2_careful_verifier_v1 | 90.000 | 10.8m | 6.1m | 16.9m |
| 4 | iter3_ensemble_judge_v1 | 95.000 | 9.9m | 5.1m | 15.0m |
| 5 | iter3_ensemble_judge_v1, iter4_robust_consensus_v1 | 95.000 | 7.4m | 5.7m | 13.1m |
| 6 | iter4_robust_consensus_v1, iter3_ensemble_judge_v1 (iter6_robust_critic_v1* excluded) | 95.000 | 7.4m | 6.4m | 13.9m |
| 7 | iter7_quad_diverse_critic_v1 | 89.992 | 7.4m | 6.2m | 13.6m |
| 8 | iter7_quad_diverse_critic_v1 | 94.995 | 6.9m | 5.8m | 12.7m |
| 9 | iter9_iter7_polish_v1 | 94.992 | 6.6m | 7.3m | 14.0m |
| 10 | iter10_idiomatic_loop_guard_v1 | 99.996 | 7.0m | 7.2m | 14.2m |
| 11 | iter10_idiomatic_loop_guard_v1 | 99.987 | 6.9m | 8.2m | 15.1m |
| 12 | iter10_idiomatic_loop_guard_v1 | 94.998 | 6.3m | 33.6m | 39.9m |
| 13 | iter10_idiomatic_loop_guard_v1 | 99.996 | 9.2m | 6.1m | 15.2m |
| 14 | iter10_idiomatic_loop_guard_v1 | 94.972 | 10.9m | 6.4m | 17.4m |
| 15 | iter10_idiomatic_loop_guard_v1 | 99.981 | 5.4m | 7.9m | 13.3m |

\* *Exact clone: identical per-problem scores to an existing agent on debut. ELO penalized by 200; excluded from winner selection.*

## Best Agent: iter10_idiomatic_loop_guard_v1
- ELO Score: 1657
- Mean Score: 98.322


## ELO Leadership Progression

Iteration-by-iteration ELO leaders:

| Iteration | Leader | ELO | Evolution Strategy |
|-----------|--------|-----|-------------------|
| 1 | seed_40_esiah | 1500 | Initial |
| 2 | seed_40_esiah | 1500 | Initial |
| 3 | iter2_careful_verifier_v1 | 1531 | use_your_judgment |
| 4 | iter3_ensemble_judge_v1 | 1533 | use_your_judgment |
| 5 | iter3_ensemble_judge_v1 | 1546 | use_your_judgment |
| 6 | iter3_ensemble_judge_v1 | 1543 | use_your_judgment |
| 7 | iter7_quad_diverse_critic_v1 | 1534 | use_your_judgment |
| 8 | iter7_quad_diverse_critic_v1 | 1563 | use_your_judgment |
| 9 | iter7_quad_diverse_critic_v1 | 1559 | use_your_judgment |
| 10 | iter7_quad_diverse_critic_v1 | 1559 | use_your_judgment |
| 11 | iter10_idiomatic_loop_guard_v1 | 1564 | use_your_judgment |
| 12 | iter10_idiomatic_loop_guard_v1 | 1592 | use_your_judgment |
| 13 | iter10_idiomatic_loop_guard_v1 | 1616 | use_your_judgment |
| 14 | iter10_idiomatic_loop_guard_v1 | 1638 | use_your_judgment |
| 15 | iter10_idiomatic_loop_guard_v1 | 1657 | use_your_judgment |


## Deep Focus Evolution Timing Analysis
- **Iterations with evolution**: 14/15
- **Total evolution time**: 111.8 minutes
- **Average evolution time**: 8.0 minutes

### Time Breakdown by Evolution Phase

| Phase | Avg Time | % of Evolution |
|-------|----------|----------------|
| First Draft (Round 1) | 7.3m | 91.4% |
| Reflection | 0.7m | 8.6% |
| **Total** | **8.0m** | **100.0%** |

### Evolution vs Test vs Meta-Evolution Time

| Metric | Time | Percentage | Avg/Iter | Avg/Occurrence |
|--------|------|------------|----------|----------------|
| Total Evolution Time | 111.8m | 49.1% | 7.5m | 8.0m |
| Total Test Time | 115.8m | 50.9% | 7.7m | 7.7m |
| Total Meta-Evolution Time | 0.0m | 0.0% | 0.0m | 0.0m |
| **Total** | **227.7m** | **100.0%** | **15.2m** | **-** |

- **Evolution/Test ratio**: 0.97x
  - For every minute of testing, 0.97 minutes spent on evolution

## Claude CLI Usage & Cost Analysis
- **Total Cost**: $86.47
  - **Evaluation Cost**: $41.03
  - **Evolution CLI Cost**: $45.44
- **Total CLI Calls**: 28
- **Total CLI Input Tokens**: 3,324
- **Total CLI Output Tokens**: 587,280
- **Total Cache Created**: 1,686,120
- **Total Cache Read**: 40,357,841

### Cost Breakdown

| Operation | Cost | % of Total | Calls | Tokens In | Tokens Out |
|-----------|------|------------|-------|-----------|------------|
| Evaluation | $41.03 | 47.5% | - | - | - |
| Evolution (CLI) | $45.44 | 52.5% | 28 | 3,324 | 587,280 |
| **Total** | **$86.47** | **100%** | **28** | **3,324** | **587,280** |

**Answer: Evolution drives 52.5% of total costs**

### Detailed Per-Iteration Costs

| Iter | Total | **Eval** | **Evo** | Iter Eval | 1st Draft | Refl | Meta | Fresh | Strategy | Meta-strategy |
|------|------|------|------|------|------|------|------|------|------|------|
| **1** | $0.01 | **$0.01** | **-** | $0.01 | - | - | - | 20 | - | - |
| **2** | $3.13 | **$0.21** | **$2.92** | $0.21 | $2.33 | $0.59 | - | 34 | use_your_judgment | - |
| **3** | $5.77 | **$1.05** | **$4.72** | $1.05 | $4.51 | $0.21 | - | 52 | use_your_judgment | - |
| **4** | $5.87 | **$1.37** | **$4.50** | $1.37 | $4.29 | $0.21 | - | 48 | use_your_judgment | - |
| **5** | $4.84 | **$2.08** | **$2.76** | $2.08 | $2.57 | $0.19 | - | 51 | use_your_judgment | - |
| **6** | $4.76 | **$2.05** | **$2.71** | $2.05 | $2.50 | $0.21 | - | 51 | use_your_judgment | - |
| **7** | $6.25 | **$3.12** | **$3.14** | $3.12 | $2.93 | $0.21 | - | 43 | use_your_judgment | - |
| **8** | $5.34 | **$2.76** | **$2.58** | $2.76 | $2.37 | $0.21 | - | 37 | use_your_judgment | - |
| **9** | $6.29 | **$3.42** | **$2.87** | $3.42 | $2.65 | $0.23 | - | 37 | use_your_judgment | - |
| **10** | $6.20 | **$3.76** | **$2.44** | $3.76 | $2.24 | $0.20 | - | 43 | use_your_judgment | - |
| **11** | $6.93 | **$4.65** | **$2.28** | $4.65 | $2.06 | $0.23 | - | 48 | use_your_judgment | - |
| **12** | $7.18 | **$3.69** | **$3.49** | $3.69 | $3.26 | $0.23 | - | 40 | use_your_judgment | - |
| **13** | $7.62 | **$3.33** | **$4.29** | $3.33 | $4.05 | $0.24 | - | 32 | use_your_judgment | - |
| **14** | $9.21 | **$5.15** | **$4.06** | $5.15 | $3.83 | $0.23 | - | 43 | use_your_judgment | - |
| **15** | $7.07 | **$4.38** | **$2.69** | $4.38 | $2.50 | $0.19 | - | 41 | use_your_judgment | - |
|------|------|------|------|------|------|------|------|------|------|------|
| **TOTAL** | **$86.47** | **$41.03** | **$45.44** | $41.03 | $42.09 | $3.35 | $0.00 | **620** | - | - |

### Evolution Cost Breakdown (Avg per evolution)

| Phase | Avg Cost | % of Evolution | Avg Tokens In | Avg Tokens Out |
|-------|----------|----------------|---------------|----------------|
| First Draft (Round 1) | $3.01 | 92.6% | 231 | 39283 |
| Reflection | $0.24 | 7.4% | 6 | 2666 |
| **Total** | **$3.25** | **100%** | **237** | **41949** |

### Cache Performance

- **Cache Creation**: 1,686,120 tokens
- **Cache Hits**: 40,357,841 tokens (96.0% hit rate)

## Zero Accuracy Cases
No zero accuracy cases encountered ✅

## Exception Failures
No exception failures encountered ✅

## Evolution Health

No artifact validation failures ✅