# Parallel Agent Research - Interim Report (Iteration 7)

## Experiment Configuration

**Run ID**: asta_ds1000_20260621_181159
**Date**: 2026-06-21 20:52:10
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
- initial_agents: ['seed_fw18t_as']
- max_workers: 10
- meta_evolution_cadence: 3
- meta_evolution_domain: asta_ds1000
- meta_evolution_first_iteration: 4
- meta_evolution_model: opus-4.8 (claude-opus-4-8)
- new_agent_test_rounds: 1

### Configuration Changes by Iteration

(No configuration changes across iterations)


## Winning Agents by Evolution Strategy

#### **Use Your Judgment** (6 wins - 75.0% of all wins)
- **iter3_safe_repair_ds1000** (3 wins, created iteration 3)
- **iter2_exec_verify_ds1000** (2 wins, created iteration 2)
- **iter6_grounded_repair_ds1000** (1 win, created iteration 6)

#### **Initial** (2 wins - 25.0% of all wins)
- **seed_fw18t_as** (2 wins, created iteration 0)

#### Summary Statistics
- **Total iterations with winners**: 7
- **Total wins counted**: 8 (includes ties)
- **Unique winning agents**: 4
- **Evolution strategies that produced winners**: 2


## Complete Performance Ranking Table

| Agent | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 | Iter 6 | Iter 7 | Final Elo | Mean Rank |
|-------|--------|--------|--------|--------|--------|--------|--------|-----------|----------|
| iter3_safe_repair_ds1000 | - | - | **#1** 85.000 | #2 75.000 | **#1** 90.000 | #2 75.000 | **#1** 95.000 | **1545** | **1.40** |
| iter7_grounded_reconcile_ds1000 | - | - | - | - | - | - | #2 90.000 | 1502 | 2.00 |
| iter6_grounded_repair_ds1000 | - | - | - | - | - | **#1** 80.000 | #3 85.000 | 1499 | 2.00 |
| iter2_exec_verify_ds1000 | - | #2 70.000 | **#1** 85.000 | **#1** 80.000 | #3 80.000 | - | - | 1499 | **1.75** |
| iter5_strong_dualcheck_ds1000 | - | - | - | - | #2 85.000 | #2 75.000 | - | 1488 | 2.00 |
| seed_fw18t_as | **#1** 70.000 | **#1** 80.000 | #3 75.000 | - | - | - | - | 1483 | 2.00 |
| iter4_consensus_arbiter_ds1000 | - | - | - | #2 75.000 | - | - | - | 1285* | 2.00 |

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
| iter3_safe_repair_ds1000 | 1545 | 84.000 | 5 |
| iter7_grounded_reconcile_ds1000 | 1502 | 90.000 | 1 |
| iter6_grounded_repair_ds1000 | 1499 | 82.500 | 2 |
| iter2_exec_verify_ds1000 | 1499 | 78.750 | 4 |
| iter5_strong_dualcheck_ds1000 | 1488 | 80.000 | 2 |
| seed_fw18t_as | 1483 | 75.000 | 3 |
| iter4_consensus_arbiter_ds1000 | 1285 | 75.000 | 1 |

### Iteration Progression

| Iter | Winner(s) | Score | Evo Time | Test Time | Total Time |
|------|-----------|-------|----------|-----------|------------|
| 1 | seed_fw18t_as | 70.000 | - | 3.5m | 3.5m |
| 2 | seed_fw18t_as | 80.000 | 6.5m | 8.0m | 14.5m |
| 3 | iter3_safe_repair_ds1000, iter2_exec_verify_ds1000 | 85.000 | 12.2m | 15.0m | 27.1m |
| 4 | iter2_exec_verify_ds1000 (iter4_consensus_arbiter_ds1000* excluded) | 80.000 | 20.2m | 16.0m | 36.1m |
| 5 | iter3_safe_repair_ds1000 | 90.000 | 14.8m | 11.8m | 26.6m |
| 6 | iter6_grounded_repair_ds1000 | 80.000 | 12.5m | 13.5m | 26.1m |
| 7 | iter3_safe_repair_ds1000 | 95.000 | 14.0m | 12.3m | 26.2m |

\* *Exact clone: identical per-problem scores to an existing agent on debut. Elo penalized by 200; excluded from winner selection.*

## Best Agent: iter3_safe_repair_ds1000
- Elo Score: 1545
- Mean Score: 84.000


## Elo Leadership Progression

Iteration-by-iteration Elo leaders:

| Iteration | Leader | Elo | Evolution Strategy |
|-----------|--------|-----|-------------------|
| 1 | seed_fw18t_as | 1500 | Initial |
| 2 | seed_fw18t_as | 1516 | Initial |
| 3 | iter3_safe_repair_ds1000 | 1516 | use_your_judgment |
| 4 | iter2_exec_verify_ds1000 | 1533 | use_your_judgment |
| 5 | iter3_safe_repair_ds1000 | 1532 | use_your_judgment |
| 6 | iter6_grounded_repair_ds1000 | 1533 | use_your_judgment |
| 7 | iter3_safe_repair_ds1000 | 1545 | use_your_judgment |


## Deep Focus Evolution Timing Analysis
- **Iterations with evolution**: 6/7
- **Total evolution time**: 80.2 minutes
- **Average evolution time**: 14.9 minutes

### Time Breakdown by Evolution Phase

| Phase | Avg Time | % of Evolution |
|-------|----------|----------------|
| First Draft (Round 1) | 5.2m | 34.8% |
| Test & Refine 1 (Round 2)* | 9.1m | 61.0% |
| Reflection | 0.6m | 4.1% |
| **Total** | **14.9m** | **100.0%** |

*Average excludes iterations where test round was not used

### Evolution vs Test vs Meta-Evolution Time

| Metric | Time | Percentage | Avg/Iter | Avg/Occurrence |
|--------|------|------------|----------|----------------|
| Total Evolution Time | 80.2m | 50.0% | 11.5m | 13.4m |
| Total Test Time | 80.0m | 50.0% | 11.4m | 11.4m |
| Total Meta-Evolution Time | 0.0m | 0.0% | 0.0m | 0.0m |
| **Total** | **160.2m** | **100.0%** | **22.9m** | **-** |

- **Evolution/Test ratio**: 1.00x
  - For every minute of testing, 1.00 minutes spent on evolution

## Claude CLI Usage & Cost Analysis
- **Total Cost**: $17.09
  - **Evaluation Cost**: $1.92
  - **Evolution CLI Cost**: $15.17
- **Total CLI Calls**: 17
- **Total CLI Input Tokens**: 31,136
- **Total CLI Output Tokens**: 231,631
- **Total Cache Created**: 472,145
- **Total Cache Read**: 8,989,642

### Cost Breakdown

| Operation | Cost | % of Total | Calls | Tokens In | Tokens Out |
|-----------|------|------------|-------|-----------|------------|
| Evaluation | $1.92 | 11.2% | - | - | - |
| Evolution (CLI) | $15.17 | 88.8% | 17 | 31,136 | 231,631 |
| **Total** | **$17.09** | **100%** | **17** | **31,136** | **231,631** |

**Answer: Evolution drives 88.8% of total costs**

### Detailed Per-Iteration Costs

| Iter | Total | **Eval** | **Evo** | Iter Eval | 1st Draft | DF Eval | Draft 2 | Refl | Meta | Fresh | Strategy | Meta-strategy |
|------|------|------|------|------|------|------|------|------|------|------|------|------|
| **1** | $0.01 | **$0.01** | **-** | $0.01 | - | - | - | - | - | 20 | - | - |
| **2** | $1.68 | **$0.10** | **$1.58** | $0.10 | $1.42 | - | - | $0.16 | - | 36 | use_your_judgment | - |
| **3** | $2.26 | **$0.15** | **$2.11** | $0.10 | $1.11 | $0.05 | $0.83 | $0.16 | - | 70 | use_your_judgment | - |
| **4** | $4.72 | **$0.40** | **$4.32** | $0.23 | $1.68 | $0.17 | $2.43 | $0.21 | - | 66 | use_your_judgment | - |
| **5** | $3.40 | **$0.56** | **$2.84** | $0.32 | $1.32 | $0.24 | $1.35 | $0.18 | - | 63 | use_your_judgment | - |
| **6** | $2.25 | **$0.34** | **$1.91** | $0.25 | $1.11 | $0.10 | $0.63 | $0.17 | - | 69 | use_your_judgment | - |
| **7** | $2.78 | **$0.36** | **$2.43** | $0.25 | $1.58 | $0.11 | $0.66 | $0.18 | - | 67 | use_your_judgment | - |
|------|------|------|------|------|------|------|------|------|------|------|------|------|
| **TOTAL** | **$17.09** | **$1.92** | **$15.17** | $1.25 | $8.21 | $0.67 | $5.91 | $1.06 | $0.00 | **391** | - | - |

### Evolution Cost Breakdown (Avg per evolution)

| Phase | Avg Cost | % of Evolution | Avg Tokens In | Avg Tokens Out |
|-------|----------|----------------|---------------|----------------|
| First Draft (Round 1) | $1.37 | 54.1% | 4233 | 23308 |
| **Test & Refine 1 (Round 2)** | **$1.10** | **43.3%** | **533** | **13088** |
|   ├─ Evaluation | $0.11 | 10.1% | 0 | 0 |
|   └─ Evolution | $0.98 | 89.9% | 533 | 13088 |
| Reflection | $0.18 | 7.0% | 423 | 2210 |
| **Total** | **$2.53** | **100%** | **5189** | **38605** |

### Cache Performance

- **Cache Creation**: 472,145 tokens
- **Cache Hits**: 8,989,642 tokens (95.0% hit rate)

## Zero Accuracy Cases
No zero accuracy cases encountered ✅

## Exception Failures
No exception failures encountered ✅

## Evolution Health

No artifact validation failures ✅