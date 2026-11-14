# Parallel Agent Research - Interim Report (Iteration 9)

## Experiment Configuration

**Run ID**: asta_ds1000_20260511_162205
**Date**: 2026-05-11 18:14:38
**Iterations**: 9/15

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

#### **Use Your Judgment** (10 wins - 83.3% of all wins)
- **iter3_ensemble_judge_v1** (3 wins, created iteration 3)
- **iter2_careful_verifier_v1** (2 wins, created iteration 2)
- **iter4_robust_consensus_v1** (2 wins, created iteration 4)
- **iter7_quad_diverse_critic_v1** (2 wins, created iteration 7)
- **iter9_iter7_polish_v1** (1 win, created iteration 9)

#### **Initial** (2 wins - 16.7% of all wins)
- **seed_40_esiah** (2 wins, created iteration 0)

#### Summary Statistics
- **Total iterations with winners**: 9
- **Total wins counted**: 12 (includes ties)
- **Unique winning agents**: 6
- **Evolution strategies that produced winners**: 2


## Complete Performance Ranking Table

| Agent | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 | Iter 6 | Iter 7 | Iter 8 | Iter 9 | Final ELO | Mean Rank |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|--------|-----------|----------|
| iter7_quad_diverse_critic_v1 | - | - | - | - | - | - | **#1** 89.992 | **#1** 94.995 | #2 89.997 | **1559** | **1.33** |
| iter9_iter7_polish_v1 | - | - | - | - | - | - | - | - | **#1** 94.992 | 1535 | **1.00** |
| iter4_robust_consensus_v1 | - | - | - | #2 90.000 | **#1** 95.000 | **#1** 95.000 | #2 85.000 | - | - | 1503 | **1.50** |
| iter2_careful_verifier_v1 | - | **#1** 75.000 | **#1** 90.000 | #3 70.000 | - | - | - | - | - | 1497 | **1.67** |
| iter3_ensemble_judge_v1 | - | - | #2 85.000 | **#1** 95.000 | **#1** 95.000 | **#1** 95.000 | #2 85.000 | #2 90.000 | #3 85.000 | 1493 | **1.71** |
| iter8_simpler_idiomatic_v1 | - | - | - | - | - | - | - | #3 84.997 | - | 1471 | 3.00 |
| iter5_triple_consensus_v1 | - | - | - | - | #3 94.996 | - | - | - | - | 1470 | 3.00 |
| seed_40_esiah | **#1** 75.000 | **#1** 75.000 | #3 60.000 | - | - | - | - | - | - | 1469 | 2.00 |
| iter6_robust_critic_v1 | - | - | - | - | - | **#1** 95.000 | - | - | - | 1303* | **1.00** |

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
| iter7_quad_diverse_critic_v1 | 1559 | 91.661 | 3 |
| iter9_iter7_polish_v1 | 1535 | 94.992 | 1 |
| iter4_robust_consensus_v1 | 1503 | 91.250 | 4 |
| iter2_careful_verifier_v1 | 1497 | 78.333 | 3 |
| iter3_ensemble_judge_v1 | 1493 | 90.000 | 7 |
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

\* *Exact clone: identical per-problem scores to an existing agent on debut. ELO penalized by 200; excluded from winner selection.*

## Best Agent: iter7_quad_diverse_critic_v1
- ELO Score: 1559
- Mean Score: 91.661


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


## Deep Focus Evolution Timing Analysis
- **Iterations with evolution**: 8/9
- **Total evolution time**: 66.1 minutes
- **Average evolution time**: 8.3 minutes

### Time Breakdown by Evolution Phase

| Phase | Avg Time | % of Evolution |
|-------|----------|----------------|
| First Draft (Round 1) | 7.6m | 91.7% |
| Reflection | 0.7m | 8.3% |
| **Total** | **8.3m** | **100.0%** |

### Evolution vs Test vs Meta-Evolution Time

| Metric | Time | Percentage | Avg/Iter | Avg/Occurrence |
|--------|------|------------|----------|----------------|
| Total Evolution Time | 66.1m | 58.8% | 7.3m | 8.3m |
| Total Test Time | 46.4m | 41.2% | 5.2m | 5.2m |
| Total Meta-Evolution Time | 0.0m | 0.0% | 0.0m | 0.0m |
| **Total** | **112.5m** | **100.0%** | **12.5m** | **-** |

- **Evolution/Test ratio**: 1.43x
  - For every minute of testing, 1.43 minutes spent on evolution

## Claude CLI Usage & Cost Analysis
- **Total Cost**: $42.26
  - **Evaluation Cost**: $16.07
  - **Evolution CLI Cost**: $26.19
- **Total CLI Calls**: 16
- **Total CLI Input Tokens**: 2,103
- **Total CLI Output Tokens**: 341,728
- **Total Cache Created**: 933,598
- **Total Cache Read**: 23,586,073

### Cost Breakdown

| Operation | Cost | % of Total | Calls | Tokens In | Tokens Out |
|-----------|------|------------|-------|-----------|------------|
| Evaluation | $16.07 | 38.0% | - | - | - |
| Evolution (CLI) | $26.19 | 62.0% | 16 | 2,103 | 341,728 |
| **Total** | **$42.26** | **100%** | **16** | **2,103** | **341,728** |

**Answer: Evolution drives 62.0% of total costs**

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
|------|------|------|------|------|------|------|------|------|------|------|
| **TOTAL** | **$42.26** | **$16.07** | **$26.19** | $16.07 | $24.15 | $2.05 | $0.00 | **373** | - | - |

### Evolution Cost Breakdown (Avg per evolution)

| Phase | Avg Cost | % of Evolution | Avg Tokens In | Avg Tokens Out |
|-------|----------|----------------|---------------|----------------|
| First Draft (Round 1) | $3.02 | 92.2% | 257 | 40108 |
| Reflection | $0.26 | 7.8% | 6 | 2608 |
| **Total** | **$3.27** | **100%** | **263** | **42716** |

### Cache Performance

- **Cache Creation**: 933,598 tokens
- **Cache Hits**: 23,586,073 tokens (96.2% hit rate)

## Zero Accuracy Cases
No zero accuracy cases encountered ✅

## Exception Failures
No exception failures encountered ✅

## Evolution Health

No artifact validation failures ✅