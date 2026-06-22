# Parallel Agent Research - Final Report

## Experiment Configuration

**Run ID**: asta_ds1000_20260621_181159
**Date**: 2026-06-22 00:11:22
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

#### **Use Your Judgment** (15 wins - 88.2% of all wins)
- **iter3_safe_repair_ds1000** (5 wins, created iteration 3)
- **iter8_perspective_consensus_ds1000** (4 wins, created iteration 8)
- **iter2_exec_verify_ds1000** (2 wins, created iteration 2)
- **iter11_diverse_consensus_ds1000** (2 wins, created iteration 11)
- **iter6_grounded_repair_ds1000** (1 win, created iteration 6)
- **iter13_champion_consensus_ds1000** (1 win, created iteration 13)

#### **Initial** (2 wins - 11.8% of all wins)
- **seed_fw18t_as** (2 wins, created iteration 0)

#### Summary Statistics
- **Total iterations with winners**: 14
- **Total wins counted**: 17 (includes ties)
- **Unique winning agents**: 7
- **Evolution strategies that produced winners**: 2


## Complete Performance Ranking Table

| Agent | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 | Iter 6 | Iter 7 | Iter 8 | Iter 9 | Iter 10 | Iter 11 | Iter 12 | Iter 13 | Iter 14 | Final Elo | Mean Rank |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|-----------|----------|
| iter8_perspective_consensus_ds1000 | - | - | - | - | - | - | - | **#1** 95.000 | **#1** 95.000 | **#1** 80.000 | #2 85.000 | - | - | **#1** 95.000 | **1573** | **1.20** |
| iter11_diverse_consensus_ds1000 | - | - | - | - | - | - | - | - | - | - | **#1** 90.000 | **#1** 85.000 | #2 95.000 | - | 1543 | **1.33** |
| iter13_champion_consensus_ds1000 | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 100.000 | #2 90.000 | 1533 | **1.50** |
| iter6_grounded_repair_ds1000 | - | - | - | - | - | **#1** 80.000 | #3 85.000 | - | - | - | - | - | - | - | 1499 | 2.00 |
| iter2_exec_verify_ds1000 | - | #2 70.000 | **#1** 85.000 | **#1** 80.000 | #3 80.000 | - | - | - | - | - | - | - | - | - | 1499 | **1.75** |
| iter3_safe_repair_ds1000 | - | - | **#1** 85.000 | #2 75.000 | **#1** 90.000 | #2 75.000 | **#1** 95.000 | #2 90.000 | **#1** 95.000 | **#1** 80.000 | #3 80.000 | #2 80.000 | #2 95.000 | - | 1490 | **1.64** |
| iter5_strong_dualcheck_ds1000 | - | - | - | - | #2 85.000 | #2 75.000 | - | - | - | - | - | - | - | - | 1488 | 2.00 |
| iter12_grounded_consensus_ds1000 | - | - | - | - | - | - | - | - | - | - | - | #2 80.000 | - | - | 1486 | 2.00 |
| seed_fw18t_as | **#1** 70.000 | **#1** 80.000 | #3 75.000 | - | - | - | - | - | - | - | - | - | - | - | 1483 | 2.00 |
| iter7_grounded_reconcile_ds1000 | - | - | - | - | - | - | #2 90.000 | #3 75.000 | - | - | - | - | - | - | 1473 | 2.50 |
| iter14_champion_consensus_ds1000 | - | - | - | - | - | - | - | - | - | - | - | - | - | #3 80.000 | 1472 | 3.00 |
| iter9_crossmodel_consensus_ds1000 | - | - | - | - | - | - | - | - | #3 90.000 | - | - | - | - | - | 1472 | 3.00 |
| iter10_grounded_verify_ds1000 | - | - | - | - | - | - | - | - | - | **#1** 80.000 | - | - | - | - | 1305* | **1.00** |
| iter4_consensus_arbiter_ds1000 | - | - | - | #2 75.000 | - | - | - | - | - | - | - | - | - | - | 1285* | 2.00 |

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
| iter8_perspective_consensus_ds1000 | 1573 | 90.000 | 5 |
| iter11_diverse_consensus_ds1000 | 1543 | 90.000 | 3 |
| iter13_champion_consensus_ds1000 | 1533 | 95.000 | 2 |
| iter6_grounded_repair_ds1000 | 1499 | 82.500 | 2 |
| iter2_exec_verify_ds1000 | 1499 | 78.750 | 4 |
| iter3_safe_repair_ds1000 | 1490 | 85.455 | 11 |
| iter5_strong_dualcheck_ds1000 | 1488 | 80.000 | 2 |
| iter12_grounded_consensus_ds1000 | 1486 | 80.000 | 1 |
| seed_fw18t_as | 1483 | 75.000 | 3 |
| iter7_grounded_reconcile_ds1000 | 1473 | 82.500 | 2 |
| iter14_champion_consensus_ds1000 | 1472 | 80.000 | 1 |
| iter9_crossmodel_consensus_ds1000 | 1472 | 90.000 | 1 |
| iter10_grounded_verify_ds1000 | 1305 | 80.000 | 1 |
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
| 8 | iter8_perspective_consensus_ds1000 | 95.000 | 13.3m | 13.6m | 26.9m |
| 9 | iter8_perspective_consensus_ds1000, iter3_safe_repair_ds1000 | 95.000 | 16.8m | 17.5m | 34.2m |
| 10 | iter3_safe_repair_ds1000, iter8_perspective_consensus_ds1000 (iter10_grounded_verify_ds1000* excluded) | 80.000 | 18.8m | 10.9m | 29.7m |
| 11 | iter11_diverse_consensus_ds1000 | 90.000 | 17.6m | 10.2m | 27.8m |
| 12 | iter11_diverse_consensus_ds1000 | 85.000 | 12.9m | 18.1m | 31.0m |
| 13 | iter13_champion_consensus_ds1000 | 100.000 | 9.3m | 11.0m | 20.3m |
| 14 | iter8_perspective_consensus_ds1000 | 95.000 | 15.4m | 13.9m | 29.3m |

\* *Exact clone: identical per-problem scores to an existing agent on debut. Elo penalized by 200; excluded from winner selection.*

## Best Agent: iter8_perspective_consensus_ds1000
- Elo Score: 1573
- Mean Score: 90.000


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
| 8 | iter3_safe_repair_ds1000 | 1541 | use_your_judgment |
| 9 | iter3_safe_repair_ds1000 | 1554 | use_your_judgment |
| 10 | iter3_safe_repair_ds1000 | 1552 | use_your_judgment |
| 11 | iter8_perspective_consensus_ds1000 | 1544 | use_your_judgment |
| 12 | iter11_diverse_consensus_ds1000 | 1565 | use_your_judgment |
| 13 | iter8_perspective_consensus_ds1000 | 1544 | use_your_judgment |
| 14 | iter8_perspective_consensus_ds1000 | 1573 | use_your_judgment |


## Deep Focus Evolution Timing Analysis
- **Iterations with evolution**: 13/14
- **Total evolution time**: 184.1 minutes
- **Average evolution time**: 14.9 minutes

### Time Breakdown by Evolution Phase

| Phase | Avg Time | % of Evolution |
|-------|----------|----------------|
| First Draft (Round 1) | 4.5m | 30.2% |
| Test & Refine 1 (Round 2)* | 9.8m | 65.5% |
| Reflection | 0.6m | 4.3% |
| **Total** | **14.9m** | **100.0%** |

*Average excludes iterations where test round was not used

### Evolution vs Test vs Meta-Evolution Time

| Metric | Time | Percentage | Avg/Iter | Avg/Occurrence |
|--------|------|------------|----------|----------------|
| Total Evolution Time | 184.1m | 51.2% | 13.2m | 14.2m |
| Total Test Time | 175.2m | 48.8% | 12.5m | 12.5m |
| Total Meta-Evolution Time | 0.0m | 0.0% | 0.0m | 0.0m |
| **Total** | **359.4m** | **100.0%** | **25.7m** | **-** |

- **Evolution/Test ratio**: 1.05x
  - For every minute of testing, 1.05 minutes spent on evolution

## Claude CLI Usage & Cost Analysis
- **Total Cost**: $38.22
  - **Evaluation Cost**: $5.69
  - **Evolution CLI Cost**: $32.52
- **Total CLI Calls**: 38
- **Total CLI Input Tokens**: 61,978
- **Total CLI Output Tokens**: 460,753
- **Total Cache Created**: 1,134,965
- **Total Cache Read**: 18,462,108

### Cost Breakdown

| Operation | Cost | % of Total | Calls | Tokens In | Tokens Out |
|-----------|------|------------|-------|-----------|------------|
| Evaluation | $5.69 | 14.9% | - | - | - |
| Evolution (CLI) | $32.52 | 85.1% | 38 | 61,978 | 460,753 |
| **Total** | **$38.22** | **100%** | **38** | **61,978** | **460,753** |

**Answer: Evolution drives 85.1% of total costs**

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
| **8** | $2.15 | **$0.35** | **$1.80** | $0.25 | $1.22 | $0.11 | $0.41 | $0.17 | - | 68 | use_your_judgment | - |
| **9** | $3.01 | **$0.55** | **$2.46** | $0.35 | $1.16 | $0.21 | $1.13 | $0.17 | - | 61 | use_your_judgment | - |
| **10** | $4.64 | **$0.30** | **$4.34** | $0.25 | $1.31 | $0.05 | $2.00 | $1.04 | - | 56 | use_your_judgment | - |
| **11** | $3.96 | **$0.57** | **$3.38** | $0.33 | $2.30 | $0.25 | $0.86 | $0.22 | - | 52 | use_your_judgment | - |
| **12** | $3.07 | **$0.72** | **$2.35** | $0.42 | $0.86 | $0.31 | $0.81 | $0.68 | - | 59 | use_your_judgment | - |
| **13** | $1.66 | **$0.60** | **$1.06** | $0.36 | $0.62 | $0.25 | $0.29 | $0.15 | - | 52 | use_your_judgment | - |
| **14** | $2.63 | **$0.67** | **$1.96** | $0.43 | $0.64 | $0.24 | $0.70 | $0.62 | - | 64 | use_your_judgment | - |
|------|------|------|------|------|------|------|------|------|------|------|------|------|
| **TOTAL** | **$38.22** | **$5.69** | **$32.52** | $3.62 | $16.31 | $2.07 | $12.11 | $4.10 | $0.00 | **803** | - | - |

### Evolution Cost Breakdown (Avg per evolution)

| Phase | Avg Cost | % of Evolution | Avg Tokens In | Avg Tokens Out |
|-------|----------|----------------|---------------|----------------|
| First Draft (Round 1) | $1.25 | 50.2% | 3738 | 19963 |
| **Test & Refine 1 (Round 2)** | **$1.09** | **43.6%** | **597** | **13237** |
|   ├─ Evaluation | $0.16 | 14.6% | 0 | 0 |
|   └─ Evolution | $0.93 | 85.4% | 597 | 13237 |
| Reflection | $0.32 | 12.6% | 433 | 2242 |
| **Total** | **$2.50** | **100%** | **4768** | **35443** |

### Cache Performance

- **Cache Creation**: 1,134,965 tokens
- **Cache Hits**: 18,462,108 tokens (94.2% hit rate)

## Zero Accuracy Cases
No zero accuracy cases encountered ✅

## Exception Failures
No exception failures encountered ✅

## Evolution Health

No artifact validation failures ✅