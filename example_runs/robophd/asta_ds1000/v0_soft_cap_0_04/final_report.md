# Parallel Agent Research - Final Report

## Experiment Configuration

**Run ID**: asta_ds1000_20260506_232005
**Date**: 2026-05-07 04:31:10
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

- agents_directory: ../alt_robophd_runs/robophd/_optimize_anything_seeds
- domain: external
- eval_timeout: 600
- evaluation_budget: 999999
- evolution_model: opus-4.7 (claude-opus-4-7)
- evolution_strategy: use_your_judgment
- evolution_timeout: 3600
- examples_per_iteration: 20
- initial_agents: ['seed_d8it1tf0']
- max_workers: 12
- meta_evolution_cadence: 3
- meta_evolution_domain: asta_ds1000
- meta_evolution_first_iteration: 4
- new_agent_test_rounds: 0

### Configuration Changes by Iteration

(No configuration changes across iterations)


## Winning Agents by Evolution Strategy

#### **Use Your Judgment** (14 wins - 82.4% of all wins)
- **iter9_mpl_aware_judge** (4 wins, created iteration 9)
- **iter4_judge_consensus** (3 wins, created iteration 4)
- **iter13_style_aware_lean** (3 wins, created iteration 13)
- **iter8_robust_judge** (2 wins, created iteration 8)
- **iter3_multi_candidate_consensus** (1 win, created iteration 3)
- **iter7_timeout_safe_judge** (1 win, created iteration 7)

#### **Initial** (3 wins - 17.6% of all wins)
- **seed_d8it1tf0** (3 wins, created iteration 0)

#### Summary Statistics
- **Total iterations with winners**: 15
- **Total wins counted**: 17 (includes ties)
- **Unique winning agents**: 7
- **Evolution strategies that produced winners**: 2


## Complete Performance Ranking Table

| Agent | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 | Iter 6 | Iter 7 | Iter 8 | Iter 9 | Iter 10 | Iter 11 | Iter 12 | Iter 13 | Iter 14 | Iter 15 | Final ELO | Mean Rank |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|-----------|----------|
| iter13_style_aware_lean | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 89.999 | **#1** 95.000 | **#1** 100.000 | **1595** | **1.00** |
| iter9_mpl_aware_judge | - | - | - | - | - | - | - | - | **#1** 95.000 | **#1** 95.000 | **#1** 90.000 | **#1** 95.000 | #3 85.000 | - | #3 90.000 | 1517 | **1.67** |
| iter8_robust_judge | - | - | - | - | - | - | - | **#1** 95.000 | #2 85.000 | - | - | **#1** 95.000 | #2 85.000 | #3 85.000 | - | 1514 | **1.80** |
| iter15_generic_melt_lean | - | - | - | - | - | - | - | - | - | - | - | - | - | - | #2 95.000 | 1505 | 2.00 |
| iter14_dtype_aware_lean | - | - | - | - | - | - | - | - | - | - | - | - | - | #2 95.000 | - | 1504 | 2.00 |
| iter4_judge_consensus | - | - | - | **#1** 70.000 | **#1** 90.000 | **#1** 80.000 | #2 80.000 | #2 85.000 | - | #2 90.000 | #3 80.000 | - | - | - | - | 1502 | **1.71** |
| seed_d8it1tf0 | **#1** 80.000 | **#1** 70.000 | #2 75.000 | **#1** 70.000 | #2 75.000 | #2 75.000 | #2 80.000 | - | - | - | - | - | - | - | - | 1499 | **1.67** |
| iter3_multi_candidate_consensus | - | - | **#1** 85.000 | #3 65.000 | - | - | - | - | - | - | - | - | - | - | - | 1498 | 2.00 |
| iter10_style_aware_judge | - | - | - | - | - | - | - | - | - | #2 90.000 | - | - | - | - | - | 1488 | 2.00 |
| iter6_expected_match_judge | - | - | - | - | - | #2 75.000 | - | - | - | - | - | - | - | - | - | 1487 | 2.00 |
| iter12_version_aware_judge | - | - | - | - | - | - | - | - | - | - | - | #3 89.990 | - | - | - | 1473 | 3.00 |
| iter7_timeout_safe_judge | - | - | - | - | - | - | **#1** 85.000 | #3 75.000 | #3 75.000 | - | - | - | - | - | - | 1473 | 2.33 |
| iter5_repr_aware_judge | - | - | - | - | #3 70.000 | - | - | - | - | - | - | - | - | - | - | 1471 | 3.00 |
| iter2_sonnet_verify_retry | - | #2 65.000 | #3 65.000 | - | - | - | - | - | - | - | - | - | - | - | - | 1455 | 2.50 |
| iter11_consensus_aware_judge | - | - | - | - | - | - | - | - | - | - | **#1** 90.000 | - | - | - | - | 1320* | **1.00** |

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
| iter13_style_aware_lean | 1595 | 95.000 | 3 |
| iter9_mpl_aware_judge | 1517 | 91.667 | 6 |
| iter8_robust_judge | 1514 | 89.000 | 5 |
| iter15_generic_melt_lean | 1505 | 95.000 | 1 |
| iter14_dtype_aware_lean | 1504 | 95.000 | 1 |
| iter4_judge_consensus | 1502 | 82.143 | 7 |
| seed_d8it1tf0 | 1499 | 75.000 | 7 |
| iter3_multi_candidate_consensus | 1498 | 75.000 | 2 |
| iter10_style_aware_judge | 1488 | 90.000 | 1 |
| iter6_expected_match_judge | 1487 | 75.000 | 1 |
| iter12_version_aware_judge | 1473 | 89.990 | 1 |
| iter7_timeout_safe_judge | 1473 | 78.333 | 3 |
| iter5_repr_aware_judge | 1471 | 70.000 | 1 |
| iter2_sonnet_verify_retry | 1455 | 65.000 | 2 |
| iter11_consensus_aware_judge | 1320 | 90.000 | 1 |

### Iteration Progression

| Iter | Winner(s) | Score | Evo Time | Test Time | Total Time |
|------|-----------|-------|----------|-----------|------------|
| 1 | seed_d8it1tf0 | 80.000 | - | 0.8m | 0.8m |
| 2 | seed_d8it1tf0 | 70.000 | 6.1m | 2.6m | 8.7m |
| 3 | iter3_multi_candidate_consensus | 85.000 | 18.2m | 3.1m | 21.3m |
| 4 | iter4_judge_consensus, seed_d8it1tf0 | 70.000 | 8.0m | 4.3m | 12.3m |
| 5 | iter4_judge_consensus | 90.000 | 11.8m | 4.4m | 16.2m |
| 6 | iter4_judge_consensus | 80.000 | 15.5m | 15.8m | 31.4m |
| 7 | iter7_timeout_safe_judge | 85.000 | 9.1m | 12.5m | 21.6m |
| 8 | iter8_robust_judge | 95.000 | 8.1m | 15.9m | 24.0m |
| 9 | iter9_mpl_aware_judge | 95.000 | 9.7m | 19.4m | 29.0m |
| 10 | iter9_mpl_aware_judge | 95.000 | 10.0m | 9.5m | 19.5m |
| 11 | iter9_mpl_aware_judge (iter11_consensus_aware_judge* excluded) | 90.000 | 12.1m | 7.2m | 19.2m |
| 12 | iter9_mpl_aware_judge, iter8_robust_judge | 95.000 | 12.2m | 10.1m | 22.4m |
| 13 | iter13_style_aware_lean | 89.999 | 10.5m | 10.0m | 20.5m |
| 14 | iter13_style_aware_lean | 95.000 | 14.3m | 13.6m | 27.9m |
| 15 | iter13_style_aware_lean | 100.000 | 12.0m | 24.3m | 36.3m |

\* *Exact clone: identical per-problem scores to an existing agent on debut. ELO penalized by 200; excluded from winner selection.*

## Best Agent: iter13_style_aware_lean
- ELO Score: 1595
- Mean Score: 95.000


## ELO Leadership Progression

Iteration-by-iteration ELO leaders:

| Iteration | Leader | ELO | Evolution Strategy |
|-----------|--------|-----|-------------------|
| 1 | seed_d8it1tf0 | 1500 | Initial |
| 2 | seed_d8it1tf0 | 1516 | Initial |
| 3 | iter3_multi_candidate_consensus | 1531 | use_your_judgment |
| 4 | seed_d8it1tf0 | 1529 | Initial |
| 5 | iter4_judge_consensus | 1549 | use_your_judgment |
| 6 | iter4_judge_consensus | 1577 | use_your_judgment |
| 7 | iter4_judge_consensus | 1555 | use_your_judgment |
| 8 | iter4_judge_consensus | 1551 | use_your_judgment |
| 9 | iter4_judge_consensus | 1551 | use_your_judgment |
| 10 | iter9_mpl_aware_judge | 1564 | use_your_judgment |
| 11 | iter9_mpl_aware_judge | 1575 | use_your_judgment |
| 12 | iter9_mpl_aware_judge | 1586 | use_your_judgment |
| 13 | iter9_mpl_aware_judge | 1549 | use_your_judgment |
| 14 | iter13_style_aware_lean | 1567 | use_your_judgment |
| 15 | iter13_style_aware_lean | 1595 | use_your_judgment |


## Deep Focus Evolution Timing Analysis
- **Iterations with evolution**: 14/15
- **Total evolution time**: 157.6 minutes
- **Average evolution time**: 11.3 minutes

### Time Breakdown by Evolution Phase

| Phase | Avg Time | % of Evolution |
|-------|----------|----------------|
| First Draft (Round 1) | 10.3m | 91.8% |
| Reflection | 0.9m | 8.2% |
| **Total** | **11.3m** | **100.0%** |

### Evolution vs Test vs Meta-Evolution Time

| Metric | Time | Percentage | Avg/Iter | Avg/Occurrence |
|--------|------|------------|----------|----------------|
| Total Evolution Time | 157.6m | 50.7% | 10.5m | 11.3m |
| Total Test Time | 153.4m | 49.3% | 10.2m | 10.2m |
| Total Meta-Evolution Time | 0.0m | 0.0% | 0.0m | 0.0m |
| **Total** | **311.1m** | **100.0%** | **20.7m** | **-** |

- **Evolution/Test ratio**: 1.03x
  - For every minute of testing, 1.03 minutes spent on evolution

## Claude CLI Usage & Cost Analysis
- **Total Cost**: $66.63
  - **Evaluation Cost**: $10.27
  - **Evolution CLI Cost**: $56.35
- **Total CLI Calls**: 28
- **Total CLI Input Tokens**: 1,171
- **Total CLI Output Tokens**: 619,544
- **Total Cache Created**: 2,219,957
- **Total Cache Read**: 53,675,387

### Cost Breakdown

| Operation | Cost | % of Total | Calls | Tokens In | Tokens Out |
|-----------|------|------------|-------|-----------|------------|
| Evaluation | $10.27 | 15.4% | - | - | - |
| Evolution (CLI) | $56.35 | 84.6% | 28 | 1,171 | 619,544 |
| **Total** | **$66.63** | **100%** | **28** | **1,171** | **619,544** |

**Answer: Evolution drives 84.6% of total costs**

### Detailed Per-Iteration Costs

| Iter | Total | **Eval** | **Evo** | Iter Eval | 1st Draft | Refl | Meta | Fresh | Strategy | Meta-strategy |
|------|------|------|------|------|------|------|------|------|------|------|
| **1** | $0.01 | **$0.01** | **-** | $0.01 | - | - | - | 20 | - | - |
| **2** | $1.93 | **$0.09** | **$1.84** | $0.09 | $1.43 | $0.41 | - | 38 | use_your_judgment | - |
| **3** | $8.63 | **$0.25** | **$8.38** | $0.25 | $7.26 | $1.12 | - | 36 | use_your_judgment | - |
| **4** | $3.50 | **$0.47** | **$3.02** | $0.47 | $2.85 | $0.17 | - | 49 | use_your_judgment | - |
| **5** | $5.20 | **$0.76** | **$4.44** | $0.76 | $4.24 | $0.20 | - | 49 | use_your_judgment | - |
| **6** | $5.68 | **$0.55** | **$5.13** | $0.55 | $4.86 | $0.27 | - | 41 | use_your_judgment | - |
| **7** | $4.11 | **$0.58** | **$3.54** | $0.58 | $3.35 | $0.19 | - | 36 | use_your_judgment | - |
| **8** | $2.86 | **$0.61** | **$2.24** | $0.61 | $2.07 | $0.17 | - | 42 | use_your_judgment | - |
| **9** | $3.24 | **$0.79** | **$2.45** | $0.79 | $2.26 | $0.19 | - | 46 | use_your_judgment | - |
| **10** | $3.44 | **$1.03** | **$2.41** | $1.03 | $2.21 | $0.20 | - | 43 | use_your_judgment | - |
| **11** | $4.47 | **$0.61** | **$3.86** | $0.61 | $3.64 | $0.22 | - | 33 | use_your_judgment | - |
| **12** | $6.81 | **$1.28** | **$5.54** | $1.28 | $4.58 | $0.95 | - | 40 | use_your_judgment | - |
| **13** | $4.61 | **$0.82** | **$3.79** | $0.82 | $3.55 | $0.25 | - | 39 | use_your_judgment | - |
| **14** | $5.53 | **$1.15** | **$4.38** | $1.15 | $4.17 | $0.21 | - | 44 | use_your_judgment | - |
| **15** | $6.60 | **$1.28** | **$5.32** | $1.28 | $4.43 | $0.89 | - | 46 | use_your_judgment | - |
|------|------|------|------|------|------|------|------|------|------|------|
| **TOTAL** | **$66.63** | **$10.27** | **$56.35** | $10.27 | $50.90 | $5.45 | $0.00 | **602** | - | - |

### Evolution Cost Breakdown (Avg per evolution)

| Phase | Avg Cost | % of Evolution | Avg Tokens In | Avg Tokens Out |
|-------|----------|----------------|---------------|----------------|
| First Draft (Round 1) | $3.64 | 90.3% | 78 | 41691 |
| Reflection | $0.39 | 9.7% | 6 | 2562 |
| **Total** | **$4.03** | **100%** | **84** | **44253** |

### Cache Performance

- **Cache Creation**: 2,219,957 tokens
- **Cache Hits**: 53,675,387 tokens (96.0% hit rate)

## Zero Accuracy Cases
No zero accuracy cases encountered ✅

## Exception Failures
No exception failures encountered ✅

## Evolution Health

No artifact validation failures ✅