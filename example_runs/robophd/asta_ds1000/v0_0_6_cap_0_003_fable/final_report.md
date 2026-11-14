# Parallel Agent Research - Final Report

## Experiment Configuration

**Run ID**: asta_ds1000_20260704_112728
**Date**: 2026-07-04 19:46:47
**Iterations**: 999

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

- ds1000_runtime: {'cost_threshold': 0.003, 'cost_per_error': 0.001}
- file_mapping: {'agent.py': 'agent.py'}

### User Configuration (Iteration 1)

The following parameters were explicitly set, overriding defaults:

- agents_directory: ../robophd_runs/robophd/_optimize_anything_seeds
- domain: external
- eval_timeout: 1800
- evaluation_budget: 750
- evolution_model: fable-5 (claude-fable-5)
- evolution_strategy: use_your_judgment
- evolution_timeout: 3600
- examples_per_iteration: 20
- initial_agents: ['seed_vjhvs6o3']
- max_workers: 5
- meta_evolution_cadence: 3
- meta_evolution_domain: asta_ds1000
- meta_evolution_first_iteration: 4
- meta_evolution_model: opus-4.8 (claude-opus-4-8)
- new_agent_test_rounds: 0

### Configuration Changes by Iteration

(No configuration changes across iterations)


## Winning Agents by Evolution Strategy

#### **Use Your Judgment** (21 wins - 91.3% of all wins)
- **iter8_expected_diff_cascade** (8 wins, created iteration 8)
- **iter6_audited_cascade** (4 wins, created iteration 6)
- **iter2_exec_verify_ensemble** (3 wins, created iteration 2)
- **iter7_lean_audited_cascade** (3 wins, created iteration 7)
- **iter18_refit_audit_cascade** (2 wins, created iteration 18)
- **iter20_exact_inplace_cascade** (1 win, created iteration 20)

#### **Initial** (2 wins - 8.7% of all wins)
- **seed_vjhvs6o3** (2 wins, created iteration 0)

#### Summary Statistics
- **Total iterations with winners**: 20
- **Total wins counted**: 23 (includes ties)
- **Unique winning agents**: 7
- **Evolution strategies that produced winners**: 2


## Complete Performance Ranking Table

| Agent | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 | Iter 6 | Iter 7 | Iter 8 | Iter 9 | Iter 10 | Iter 11 | Iter 12 | Iter 13 | Iter 14 | Iter 15 | Iter 16 | Iter 17 | Iter 18 | Iter 19 | Iter 20 | Final Elo | Mean Rank |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|-----------|----------|
| iter8_expected_diff_cascade | - | - | - | - | - | - | - | **#1** 93.892 | **#1** 83.263 | **#1** 80.000 | #2 93.966 | #2 78.777 | **#1** 90.000 | #2 85.000 | **#1** 100.000 | **#1** 88.537 | **#1** 81.211 | **#1** 80.000 | - | - | **1650** | **1.27** |
| iter7_lean_audited_cascade | - | - | - | - | - | - | **#1** 80.000 | #2 90.000 | - | **#1** 80.000 | - | - | - | - | - | - | - | **#1** 80.000 | #2 78.547 | #2 85.000 | 1546 | **1.50** |
| iter20_exact_inplace_cascade | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 90.000 | 1535 | **1.00** |
| iter6_audited_cascade | - | - | - | - | - | **#1** 85.606 | #3 71.646 | - | - | - | **#1** 97.033 | **#1** 83.439 | #2 87.352 | **#1** 90.000 | #2 95.000 | #3 73.674 | #3 66.759 | - | - | - | 1513 | **1.89** |
| iter18_refit_audit_cascade | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 80.000 | **#1** 83.488 | #3 75.000 | 1509 | **1.67** |
| iter16_exact_gate_cascade | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | #2 82.069 | - | - | - | - | 1508 | 2.00 |
| iter17_thrift_guarded_cascade | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | #2 67.335 | - | - | - | 1508 | 2.00 |
| iter9_shape_aware_cascade | - | - | - | - | - | - | - | - | #2 80.000 | - | - | - | - | - | - | - | - | - | - | - | 1503 | 2.00 |
| iter2_exec_verify_ensemble | - | #2 83.681 | **#1** 90.000 | **#1** 89.104 | **#1** 73.886 | #2 83.964 | #2 72.283 | #3 87.052 | #3 69.331 | - | - | - | - | - | - | - | - | - | - | - | 1496 | **1.88** |
| seed_vjhvs6o3 | **#1** 60.000 | **#1** 90.000 | #3 80.000 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1483 | 2.00 |
| iter14_variant_sweep_cascade | - | - | - | - | - | - | - | - | - | - | - | - | - | #3 78.608 | - | - | - | - | - | - | 1476 | 3.00 |
| iter13_canonical_idiom_cascade | - | - | - | - | - | - | - | - | - | - | - | - | #3 80.455 | - | - | - | - | - | - | - | 1475 | 3.00 |
| iter5_majority_cascade | - | - | - | - | #2 72.841 | #3 76.259 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1474 | 2.50 |
| iter12_thrifty_audited_cascade | - | - | - | - | - | - | - | - | - | - | - | #3 76.081 | - | - | - | - | - | - | - | - | 1474 | 3.00 |
| iter10_robust_expected_cascade | - | - | - | - | - | - | - | - | - | #3 77.293 | - | - | - | - | - | - | - | - | - | - | 1473 | 3.00 |
| iter11_escalated_expected_cascade | - | - | - | - | - | - | - | - | - | - | #3 77.946 | - | - | - | - | - | - | - | - | - | 1472 | 3.00 |
| iter19_shapeaware_dual_cascade | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | #3 69.559 | - | 1472 | 3.00 |
| iter4_skeptic_ensemble | - | - | - | #2 87.569 | #3 67.127 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1472 | 2.50 |
| iter3_verified_mini | - | - | #2 85.000 | #3 80.000 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1469 | 2.50 |
| iter15_postpass_guarded | - | - | - | - | - | - | - | - | - | - | - | - | - | - | #2 95.000 | - | - | - | - | - | 1293* | 2.00 |

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
| iter8_expected_diff_cascade | 1650 | 86.786 | 11 |
| iter7_lean_audited_cascade | 1546 | 82.258 | 6 |
| iter20_exact_inplace_cascade | 1535 | 90.000 | 1 |
| iter6_audited_cascade | 1513 | 83.390 | 9 |
| iter18_refit_audit_cascade | 1509 | 79.496 | 3 |
| iter16_exact_gate_cascade | 1508 | 82.069 | 1 |
| iter17_thrift_guarded_cascade | 1508 | 67.335 | 1 |
| iter9_shape_aware_cascade | 1503 | 80.000 | 1 |
| iter2_exec_verify_ensemble | 1496 | 81.163 | 8 |
| seed_vjhvs6o3 | 1483 | 76.667 | 3 |
| iter14_variant_sweep_cascade | 1476 | 78.608 | 1 |
| iter13_canonical_idiom_cascade | 1475 | 80.455 | 1 |
| iter5_majority_cascade | 1474 | 74.550 | 2 |
| iter12_thrifty_audited_cascade | 1474 | 76.081 | 1 |
| iter10_robust_expected_cascade | 1473 | 77.293 | 1 |
| iter11_escalated_expected_cascade | 1472 | 77.946 | 1 |
| iter19_shapeaware_dual_cascade | 1472 | 69.559 | 1 |
| iter4_skeptic_ensemble | 1472 | 77.348 | 2 |
| iter3_verified_mini | 1469 | 82.500 | 2 |
| iter15_postpass_guarded | 1293 | 95.000 | 1 |

### Iteration Progression

| Iter | Winner(s) | Score | Evo Time | Test Time | Total Time |
|------|-----------|-------|----------|-----------|------------|
| 1 | seed_vjhvs6o3 | 60.000 | - | 1.8m | 1.8m |
| 2 | seed_vjhvs6o3 | 90.000 | 16.0m | 7.1m | 23.1m |
| 3 | iter2_exec_verify_ensemble | 90.000 | 8.9m | 10.6m | 19.5m |
| 4 | iter2_exec_verify_ensemble | 89.104 | 10.8m | 14.4m | 25.2m |
| 5 | iter2_exec_verify_ensemble | 73.886 | 10.3m | 16.9m | 27.2m |
| 6 | iter6_audited_cascade | 85.606 | 24.5m | 13.4m | 37.9m |
| 7 | iter7_lean_audited_cascade | 80.000 | 12.7m | 13.2m | 25.9m |
| 8 | iter8_expected_diff_cascade | 93.892 | 15.2m | 16.3m | 31.5m |
| 9 | iter8_expected_diff_cascade | 83.263 | 9.9m | 14.7m | 24.6m |
| 10 | iter8_expected_diff_cascade, iter7_lean_audited_cascade | 80.000 | 15.7m | 11.7m | 27.4m |
| 11 | iter6_audited_cascade | 97.033 | 14.2m | 22.2m | 36.4m |
| 12 | iter6_audited_cascade | 83.439 | 10.3m | 12.4m | 22.6m |
| 13 | iter8_expected_diff_cascade | 90.000 | 9.4m | 12.7m | 22.2m |
| 14 | iter6_audited_cascade | 90.000 | 16.1m | 11.1m | 27.3m |
| 15 | iter8_expected_diff_cascade (iter15_postpass_guarded* excluded) | 100.000 | 10.3m | 13.5m | 23.7m |
| 16 | iter8_expected_diff_cascade | 88.537 | 7.5m | 9.9m | 17.4m |
| 17 | iter8_expected_diff_cascade | 81.211 | 19.7m | 9.9m | 29.6m |
| 18 | iter8_expected_diff_cascade +2 | 80.000 | 8.5m | 11.5m | 20.0m |
| 19 | iter18_refit_audit_cascade | 83.488 | 10.4m | 17.6m | 28.0m |
| 20 | iter20_exact_inplace_cascade | 90.000 | 14.3m | 13.7m | 28.0m |

\* *Exact clone: identical per-problem scores to an existing agent on debut. Elo penalized by 200; excluded from winner selection.*

## Best Agent: iter8_expected_diff_cascade
- Elo Score: 1650
- Mean Score: 86.786


## Elo Leadership Progression

Iteration-by-iteration Elo leaders:

| Iteration | Leader | Elo | Evolution Strategy |
|-----------|--------|-----|-------------------|
| 1 | seed_vjhvs6o3 | 1500 | Initial |
| 2 | seed_vjhvs6o3 | 1516 | Initial |
| 3 | iter2_exec_verify_ensemble | 1517 | use_your_judgment |
| 4 | iter2_exec_verify_ensemble | 1547 | use_your_judgment |
| 5 | iter2_exec_verify_ensemble | 1574 | use_your_judgment |
| 6 | iter2_exec_verify_ensemble | 1568 | use_your_judgment |
| 7 | iter2_exec_verify_ensemble | 1563 | use_your_judgment |
| 8 | iter8_expected_diff_cascade | 1536 | use_your_judgment |
| 9 | iter8_expected_diff_cascade | 1565 | use_your_judgment |
| 10 | iter8_expected_diff_cascade | 1577 | use_your_judgment |
| 11 | iter8_expected_diff_cascade | 1570 | use_your_judgment |
| 12 | iter6_audited_cascade | 1569 | use_your_judgment |
| 13 | iter8_expected_diff_cascade | 1594 | use_your_judgment |
| 14 | iter6_audited_cascade | 1595 | use_your_judgment |
| 15 | iter8_expected_diff_cascade | 1616 | use_your_judgment |
| 16 | iter8_expected_diff_cascade | 1641 | use_your_judgment |
| 17 | iter8_expected_diff_cascade | 1662 | use_your_judgment |
| 18 | iter8_expected_diff_cascade | 1650 | use_your_judgment |
| 19 | iter8_expected_diff_cascade | 1650 | use_your_judgment |
| 20 | iter8_expected_diff_cascade | 1650 | use_your_judgment |


## Deep Focus Evolution Timing Analysis
- **Iterations with evolution**: 19/20
- **Total evolution time**: 244.6 minutes
- **Average evolution time**: 12.9 minutes

### Time Breakdown by Evolution Phase

| Phase | Avg Time | % of Evolution |
|-------|----------|----------------|
| First Draft (Round 1) | 12.0m | 93.5% |
| Reflection | 0.8m | 6.5% |
| **Total** | **12.9m** | **100.0%** |

### Evolution vs Test vs Meta-Evolution Time

| Metric | Time | Percentage | Avg/Iter | Avg/Occurrence |
|--------|------|------------|----------|----------------|
| Total Evolution Time | 244.6m | 49.0% | 12.2m | 12.9m |
| Total Test Time | 254.7m | 51.0% | 12.7m | 12.7m |
| Total Meta-Evolution Time | 0.0m | 0.0% | 0.0m | 0.0m |
| **Total** | **499.3m** | **100.0%** | **25.0m** | **-** |

- **Evolution/Test ratio**: 0.96x
  - For every minute of testing, 0.96 minutes spent on evolution

## Claude CLI Usage & Cost Analysis
- **Total Cost**: $156.46
  - **Evaluation Cost**: $2.48
  - **Evolution CLI Cost**: $153.97
- **Total CLI Calls**: 38
- **Total CLI Input Tokens**: 90,991
- **Total CLI Output Tokens**: 969,765
- **Total Cache Created**: 2,825,065
- **Total Cache Read**: 48,034,952

### Cost Breakdown

| Operation | Cost | % of Total | Calls | Tokens In | Tokens Out |
|-----------|------|------------|-------|-----------|------------|
| Evaluation | $2.48 | 1.6% | - | - | - |
| Evolution (CLI) | $153.97 | 98.4% | 38 | 90,991 | 969,765 |
| **Total** | **$156.46** | **100%** | **38** | **90,991** | **969,765** |

**Answer: Evolution drives 98.4% of total costs**

### Detailed Per-Iteration Costs

| Iter | Total | **Eval** | **Evo** | Iter Eval | 1st Draft | Refl | Meta | Fresh | Strategy | Meta-strategy |
|------|------|------|------|------|------|------|------|------|------|------|
| **1** | $0.01 | **$0.01** | **-** | $0.01 | - | - | - | 20 | - | - |
| **2** | $7.22 | **$0.10** | **$7.12** | $0.10 | $6.71 | $0.41 | - | 38 | use_your_judgment | - |
| **3** | $4.59 | **$0.08** | **$4.52** | $0.08 | $4.11 | $0.40 | - | 46 | use_your_judgment | - |
| **4** | $5.36 | **$0.18** | **$5.18** | $0.18 | $4.75 | $0.43 | - | 52 | use_your_judgment | - |
| **5** | $7.82 | **$0.25** | **$7.58** | $0.25 | $5.32 | $2.26 | - | 49 | use_your_judgment | - |
| **6** | $7.88 | **$0.15** | **$7.73** | $0.15 | $7.24 | $0.49 | - | 43 | use_your_judgment | - |
| **7** | $6.53 | **$0.11** | **$6.41** | $0.11 | $5.97 | $0.45 | - | 39 | use_your_judgment | - |
| **8** | $7.54 | **$0.17** | **$7.37** | $0.17 | $6.88 | $0.49 | - | 49 | use_your_judgment | - |
| **9** | $5.66 | **$0.13** | **$5.53** | $0.13 | $5.07 | $0.46 | - | 41 | use_your_judgment | - |
| **10** | $9.56 | **$0.12** | **$9.44** | $0.12 | $8.93 | $0.51 | - | 44 | use_your_judgment | - |
| **11** | $9.84 | **$0.18** | **$9.66** | $0.18 | $9.17 | $0.49 | - | 42 | use_your_judgment | - |
| **12** | $6.40 | **$0.14** | **$6.26** | $0.14 | $5.80 | $0.46 | - | 40 | use_your_judgment | - |
| **13** | $6.14 | **$0.11** | **$6.02** | $0.11 | $5.60 | $0.43 | - | 36 | use_your_judgment | - |
| **14** | $12.78 | **$0.09** | **$12.69** | $0.09 | $9.85 | $2.83 | - | 34 | use_your_judgment | - |
| **15** | $8.36 | **$0.07** | **$8.29** | $0.07 | $6.26 | $2.03 | - | 29 | use_your_judgment | - |
| **16** | $6.35 | **$0.14** | **$6.21** | $0.14 | $4.52 | $1.69 | - | 28 | use_your_judgment | - |
| **17** | $17.34 | **$0.10** | **$17.24** | $0.10 | $13.85 | $3.39 | - | 23 | use_your_judgment | - |
| **18** | $7.81 | **$0.07** | **$7.74** | $0.07 | $5.73 | $2.01 | - | 32 | use_your_judgment | - |
| **19** | $8.55 | **$0.19** | **$8.36** | $0.19 | $6.18 | $2.18 | - | 48 | use_your_judgment | - |
| **20** | $10.71 | **$0.10** | **$10.62** | $0.10 | $7.86 | $2.76 | - | 38 | use_your_judgment | - |
|------|------|------|------|------|------|------|------|------|------|------|
| **TOTAL** | **$156.46** | **$2.48** | **$153.97** | $2.48 | $129.80 | $24.17 | $0.00 | **771** | - | - |

### Evolution Cost Breakdown (Avg per evolution)

| Phase | Avg Cost | % of Evolution | Avg Tokens In | Avg Tokens Out |
|-------|----------|----------------|---------------|----------------|
| First Draft (Round 1) | $6.83 | 84.3% | 4363 | 48406 |
| Reflection | $1.27 | 15.7% | 426 | 2634 |
| **Total** | **$8.10** | **100%** | **4789** | **51040** |

### Cache Performance

- **Cache Creation**: 2,825,065 tokens
- **Cache Hits**: 48,034,952 tokens (94.4% hit rate)

## Zero Accuracy Cases
No zero accuracy cases encountered ✅

## Exception Failures
No exception failures encountered ✅

## Evolution Health

No artifact validation failures ✅