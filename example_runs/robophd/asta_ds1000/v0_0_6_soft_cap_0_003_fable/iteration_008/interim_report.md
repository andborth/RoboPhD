# Parallel Agent Research - Interim Report (Iteration 8)

## Experiment Configuration

**Run ID**: asta_ds1000_20260704_112728
**Date**: 2026-07-04 14:39:29
**Iterations**: 8/999

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

#### **Use Your Judgment** (6 wins - 75.0% of all wins)
- **iter2_exec_verify_ensemble** (3 wins, created iteration 2)
- **iter6_audited_cascade** (1 win, created iteration 6)
- **iter7_lean_audited_cascade** (1 win, created iteration 7)
- **iter8_expected_diff_cascade** (1 win, created iteration 8)

#### **Initial** (2 wins - 25.0% of all wins)
- **seed_vjhvs6o3** (2 wins, created iteration 0)

#### Summary Statistics
- **Total iterations with winners**: 8
- **Total wins counted**: 8 (includes ties)
- **Unique winning agents**: 5
- **Evolution strategies that produced winners**: 2


## Complete Performance Ranking Table

| Agent | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 | Iter 6 | Iter 7 | Iter 8 | Final Elo | Mean Rank |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|-----------|----------|
| iter8_expected_diff_cascade | - | - | - | - | - | - | - | **#1** 93.892 | **1536** | **1.00** |
| iter7_lean_audited_cascade | - | - | - | - | - | - | **#1** 80.000 | #2 90.000 | 1535 | **1.50** |
| iter2_exec_verify_ensemble | - | #2 83.681 | **#1** 90.000 | **#1** 89.104 | **#1** 73.886 | #2 83.964 | #2 72.283 | #3 87.052 | 1528 | **1.71** |
| iter6_audited_cascade | - | - | - | - | - | **#1** 85.606 | #3 71.646 | - | 1503 | 2.00 |
| seed_vjhvs6o3 | **#1** 60.000 | **#1** 90.000 | #3 80.000 | - | - | - | - | - | 1483 | 2.00 |
| iter5_majority_cascade | - | - | - | - | #2 72.841 | #3 76.259 | - | - | 1474 | 2.50 |
| iter4_skeptic_ensemble | - | - | - | #2 87.569 | #3 67.127 | - | - | - | 1472 | 2.50 |
| iter3_verified_mini | - | - | #2 85.000 | #3 80.000 | - | - | - | - | 1469 | 2.50 |

### Legend:
- **#1** = 1st place (winner of iteration)
- #2, #3, etc. = 2nd, 3rd place, etc.
- Score = Average score (0-1) on that iteration's problems
- **Bold Elo/Rank** = Top performer
- `-` = Agent not tested in that iteration



### Quick Summary

| Agent | Elo | Mean Score | Tests |
|-------|-----|------------|-------|
| iter8_expected_diff_cascade | 1536 | 93.892 | 1 |
| iter7_lean_audited_cascade | 1535 | 85.000 | 2 |
| iter2_exec_verify_ensemble | 1528 | 82.853 | 7 |
| iter6_audited_cascade | 1503 | 78.626 | 2 |
| seed_vjhvs6o3 | 1483 | 76.667 | 3 |
| iter5_majority_cascade | 1474 | 74.550 | 2 |
| iter4_skeptic_ensemble | 1472 | 77.348 | 2 |
| iter3_verified_mini | 1469 | 82.500 | 2 |

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

## Best Agent: iter8_expected_diff_cascade
- Elo Score: 1536
- Mean Score: 93.892


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


## Deep Focus Evolution Timing Analysis
- **Iterations with evolution**: 7/8
- **Total evolution time**: 98.3 minutes
- **Average evolution time**: 14.0 minutes

### Time Breakdown by Evolution Phase

| Phase | Avg Time | % of Evolution |
|-------|----------|----------------|
| First Draft (Round 1) | 13.2m | 94.1% |
| Reflection | 0.8m | 5.9% |
| **Total** | **14.0m** | **100.0%** |

### Evolution vs Test vs Meta-Evolution Time

| Metric | Time | Percentage | Avg/Iter | Avg/Occurrence |
|--------|------|------------|----------|----------------|
| Total Evolution Time | 98.3m | 51.2% | 12.3m | 14.0m |
| Total Test Time | 93.7m | 48.8% | 11.7m | 11.7m |
| Total Meta-Evolution Time | 0.0m | 0.0% | 0.0m | 0.0m |
| **Total** | **192.0m** | **100.0%** | **24.0m** | **-** |

- **Evolution/Test ratio**: 1.05x
  - For every minute of testing, 1.05 minutes spent on evolution

## Claude CLI Usage & Cost Analysis
- **Total Cost**: $46.95
  - **Evaluation Cost**: $1.04
  - **Evolution CLI Cost**: $45.91
- **Total CLI Calls**: 14
- **Total CLI Input Tokens**: 29,906
- **Total CLI Output Tokens**: 393,172
- **Total Cache Created**: 798,759
- **Total Cache Read**: 9,959,028

### Cost Breakdown

| Operation | Cost | % of Total | Calls | Tokens In | Tokens Out |
|-----------|------|------------|-------|-----------|------------|
| Evaluation | $1.04 | 2.2% | - | - | - |
| Evolution (CLI) | $45.91 | 97.8% | 14 | 29,906 | 393,172 |
| **Total** | **$46.95** | **100%** | **14** | **29,906** | **393,172** |

**Answer: Evolution drives 97.8% of total costs**

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
|------|------|------|------|------|------|------|------|------|------|------|
| **TOTAL** | **$46.95** | **$1.04** | **$45.91** | $1.04 | $40.98 | $4.93 | $0.00 | **336** | - | - |

### Evolution Cost Breakdown (Avg per evolution)

| Phase | Avg Cost | % of Evolution | Avg Tokens In | Avg Tokens Out |
|-------|----------|----------------|---------------|----------------|
| First Draft (Round 1) | $5.85 | 89.3% | 3990 | 53504 |
| Reflection | $0.70 | 10.7% | 282 | 2664 |
| **Total** | **$6.56** | **100%** | **4272** | **56167** |

### Cache Performance

- **Cache Creation**: 798,759 tokens
- **Cache Hits**: 9,959,028 tokens (92.6% hit rate)

## Zero Accuracy Cases
No zero accuracy cases encountered ✅

## Exception Failures
No exception failures encountered ✅

## Evolution Health

No artifact validation failures ✅