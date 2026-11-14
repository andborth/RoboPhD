# Parallel Agent Research - Interim Report (Iteration 7)

## Experiment Configuration

**Run ID**: asta_ds1000_20260704_112728
**Date**: 2026-07-04 14:08:02
**Iterations**: 7/999

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

#### **Use Your Judgment** (5 wins - 71.4% of all wins)
- **iter2_exec_verify_ensemble** (3 wins, created iteration 2)
- **iter6_audited_cascade** (1 win, created iteration 6)
- **iter7_lean_audited_cascade** (1 win, created iteration 7)

#### **Initial** (2 wins - 28.6% of all wins)
- **seed_vjhvs6o3** (2 wins, created iteration 0)

#### Summary Statistics
- **Total iterations with winners**: 7
- **Total wins counted**: 7 (includes ties)
- **Unique winning agents**: 4
- **Evolution strategies that produced winners**: 2


## Complete Performance Ranking Table

| Agent | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 | Iter 6 | Iter 7 | Final Elo | Mean Rank |
|-------|--------|--------|--------|--------|--------|--------|--------|-----------|----------|
| iter2_exec_verify_ensemble | - | #2 83.681 | **#1** 90.000 | **#1** 89.104 | **#1** 73.886 | #2 83.964 | #2 72.283 | **1563** | **1.50** |
| iter7_lean_audited_cascade | - | - | - | - | - | - | **#1** 80.000 | 1536 | **1.00** |
| iter6_audited_cascade | - | - | - | - | - | **#1** 85.606 | #3 71.646 | 1503 | 2.00 |
| seed_vjhvs6o3 | **#1** 60.000 | **#1** 90.000 | #3 80.000 | - | - | - | - | 1483 | 2.00 |
| iter5_majority_cascade | - | - | - | - | #2 72.841 | #3 76.259 | - | 1474 | 2.50 |
| iter4_skeptic_ensemble | - | - | - | #2 87.569 | #3 67.127 | - | - | 1472 | 2.50 |
| iter3_verified_mini | - | - | #2 85.000 | #3 80.000 | - | - | - | 1469 | 2.50 |

### Legend:
- **#1** = 1st place (winner of iteration)
- #2, #3, etc. = 2nd, 3rd place, etc.
- Score = Average score (0-1) on that iteration's problems
- **Bold Elo/Rank** = Top performer
- `-` = Agent not tested in that iteration



### Quick Summary

| Agent | Elo | Mean Score | Tests |
|-------|-----|------------|-------|
| iter2_exec_verify_ensemble | 1563 | 82.153 | 6 |
| iter7_lean_audited_cascade | 1536 | 80.000 | 1 |
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

## Best Agent: iter2_exec_verify_ensemble
- Elo Score: 1563
- Mean Score: 82.153


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


## Deep Focus Evolution Timing Analysis
- **Iterations with evolution**: 6/7
- **Total evolution time**: 83.1 minutes
- **Average evolution time**: 13.9 minutes

### Time Breakdown by Evolution Phase

| Phase | Avg Time | % of Evolution |
|-------|----------|----------------|
| First Draft (Round 1) | 13.0m | 94.0% |
| Reflection | 0.8m | 6.0% |
| **Total** | **13.9m** | **100.0%** |

### Evolution vs Test vs Meta-Evolution Time

| Metric | Time | Percentage | Avg/Iter | Avg/Occurrence |
|--------|------|------------|----------|----------------|
| Total Evolution Time | 83.1m | 51.8% | 11.9m | 13.9m |
| Total Test Time | 77.4m | 48.2% | 11.1m | 11.1m |
| Total Meta-Evolution Time | 0.0m | 0.0% | 0.0m | 0.0m |
| **Total** | **160.6m** | **100.0%** | **22.9m** | **-** |

- **Evolution/Test ratio**: 1.07x
  - For every minute of testing, 1.07 minutes spent on evolution

## Claude CLI Usage & Cost Analysis
- **Total Cost**: $39.41
  - **Evaluation Cost**: $0.88
  - **Evolution CLI Cost**: $38.54
- **Total CLI Calls**: 12
- **Total CLI Input Tokens**: 25,940
- **Total CLI Output Tokens**: 321,105
- **Total Cache Created**: 681,143
- **Total Cache Read**: 8,588,748

### Cost Breakdown

| Operation | Cost | % of Total | Calls | Tokens In | Tokens Out |
|-----------|------|------------|-------|-----------|------------|
| Evaluation | $0.88 | 2.2% | - | - | - |
| Evolution (CLI) | $38.54 | 97.8% | 12 | 25,940 | 321,105 |
| **Total** | **$39.41** | **100%** | **12** | **25,940** | **321,105** |

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
|------|------|------|------|------|------|------|------|------|------|------|
| **TOTAL** | **$39.41** | **$0.88** | **$38.54** | $0.88 | $34.10 | $4.44 | $0.00 | **287** | - | - |

### Evolution Cost Breakdown (Avg per evolution)

| Phase | Avg Cost | % of Evolution | Avg Tokens In | Avg Tokens Out |
|-------|----------|----------------|---------------|----------------|
| First Draft (Round 1) | $5.68 | 88.5% | 4038 | 50865 |
| Reflection | $0.74 | 11.5% | 285 | 2653 |
| **Total** | **$6.42** | **100%** | **4323** | **53518** |

### Cache Performance

- **Cache Creation**: 681,143 tokens
- **Cache Hits**: 8,588,748 tokens (92.7% hit rate)

## Zero Accuracy Cases
No zero accuracy cases encountered ✅

## Exception Failures
No exception failures encountered ✅

## Evolution Health

No artifact validation failures ✅