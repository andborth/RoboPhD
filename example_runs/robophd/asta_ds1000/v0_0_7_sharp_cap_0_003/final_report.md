# Parallel Agent Research - Final Report

## Experiment Configuration

**Run ID**: asta_ds1000_20260721_033200
**Date**: 2026-07-21 06:45:13
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

- ds1000_runtime: {'cost_threshold': 0.003, 'cost_per_error': 0.0003}
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
- initial_agents: ['seed__mxgdywk']
- max_workers: 5
- meta_evolution_cadence: 3
- meta_evolution_domain: asta_ds1000
- meta_evolution_first_iteration: 4
- meta_evolution_model: opus-4.8 (claude-opus-4-8)
- new_agent_test_rounds: 0

### Configuration Changes by Iteration

(No configuration changes across iterations)


## Winning Agents by Evolution Strategy

#### **Use Your Judgment** (14 wins - 66.7% of all wins)
- **iter12_strong_toplevel** (4 wins, created iteration 12)
- **iter6_strong_oneshot** (3 wins, created iteration 6)
- **iter16_strong_safestrip** (3 wins, created iteration 16)
- **iter10_strong_reindent** (2 wins, created iteration 10)
- **iter13_strong_multiwrap** (1 win, created iteration 13)
- **iter18_strong_truncsafe** (1 win, created iteration 18)

#### **Initial** (7 wins - 33.3% of all wins)
- **seed__mxgdywk** (7 wins, created iteration 0)

#### Summary Statistics
- **Total iterations with winners**: 20
- **Total wins counted**: 21 (includes ties)
- **Unique winning agents**: 7
- **Evolution strategies that produced winners**: 2


## Complete Performance Ranking Table

| Agent | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 | Iter 6 | Iter 7 | Iter 8 | Iter 9 | Iter 10 | Iter 11 | Iter 12 | Iter 13 | Iter 14 | Iter 15 | Iter 16 | Iter 17 | Iter 18 | Iter 19 | Iter 20 | Final Elo | Mean Rank |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|-----------|----------|
| iter12_strong_toplevel | - | - | - | - | - | - | - | - | - | - | - | **#1** 90.000 | #2 85.000 | **#1** 95.000 | **#1** 75.000 | #2 70.000 | - | - | **#1** 70.000 | #2 65.000 | **1602** | **1.43** |
| iter16_strong_safestrip | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 75.000 | **#1** 100.000 | #2 80.000 | - | **#1** 70.000 | 1566 | **1.25** |
| iter10_strong_reindent | - | - | - | - | - | - | - | - | - | **#1** 85.000 | **#1** 70.000 | #3 80.000 | - | - | - | - | - | #2 80.000 | - | - | 1517 | **1.75** |
| iter6_strong_oneshot | - | - | - | - | - | **#1** 70.000 | #2 80.000 | **#1** 75.000 | **#1** 90.000 | #3 65.000 | - | #2 85.000 | #3 75.000 | - | - | - | - | - | - | - | 1515 | **1.86** |
| seed__mxgdywk | **#1** 70.000 | **#1** 70.000 | **#1** 75.000 | **#1** 80.000 | **#1** 75.000 | **#1** 70.000 | **#1** 85.000 | #3 65.000 | #3 80.000 | #2 75.000 | #2 65.000 | - | - | - | - | - | #3 90.000 | - | - | - | 1512 | **1.73** |
| iter8_raw_strong | - | - | - | - | - | - | - | #2 70.000 | - | - | - | - | - | - | - | - | - | - | - | - | 1506 | 2.00 |
| iter9_strong_oneshot_unescape | - | - | - | - | - | - | - | - | #2 85.000 | - | - | - | - | - | - | - | - | - | - | - | 1506 | 2.00 |
| iter19_strong_readfirst | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | #2 65.000 | - | 1505 | 2.00 |
| iter18_strong_truncsafe | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 85.000 | #3 60.000 | - | 1504 | 2.00 |
| iter13_strong_multiwrap | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 90.000 | #2 90.000 | #2 70.000 | #3 65.000 | - | - | - | - | 1504 | 2.00 |
| iter4_verify_reason_escalate | - | - | - | #2 70.000 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1486 | 2.00 |
| iter20_converged_frozen | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | #3 60.000 | 1475 | 3.00 |
| iter7_strong_contract | - | - | - | - | - | - | #3 70.000 | - | - | - | - | - | - | - | - | - | - | - | - | - | 1474 | 3.00 |
| iter15_strong_driverstrip | - | - | - | - | - | - | - | - | - | - | - | - | - | - | #3 65.000 | - | - | - | - | - | 1473 | 3.00 |
| iter11_strong_assign | - | - | - | - | - | - | - | - | - | - | #3 60.000 | - | - | - | - | - | - | - | - | - | 1473 | 3.00 |
| iter14_strong_truncsafe | - | - | - | - | - | - | - | - | - | - | - | - | - | #3 85.000 | - | - | - | - | - | - | 1472 | 3.00 |
| iter5_guided_oneshot | - | - | - | - | #3 40.000 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1471 | 3.00 |
| iter3_selfconsistency_exec | - | - | #2 65.000 | #2 70.000 | #2 70.000 | #3 55.000 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1465 | 2.25 |
| iter2_verify_retry_mini | - | #2 60.000 | #3 60.000 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1455 | 2.50 |
| iter17_strong_validretry | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 100.000 | - | - | - | 1319* | **1.00** |

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
| iter12_strong_toplevel | 1602 | 78.571 | 7 |
| iter16_strong_safestrip | 1566 | 81.250 | 4 |
| iter10_strong_reindent | 1517 | 78.750 | 4 |
| iter6_strong_oneshot | 1515 | 77.143 | 7 |
| seed__mxgdywk | 1512 | 75.000 | 12 |
| iter8_raw_strong | 1506 | 70.000 | 1 |
| iter9_strong_oneshot_unescape | 1506 | 85.000 | 1 |
| iter19_strong_readfirst | 1505 | 65.000 | 1 |
| iter18_strong_truncsafe | 1504 | 72.500 | 2 |
| iter13_strong_multiwrap | 1504 | 78.750 | 4 |
| iter4_verify_reason_escalate | 1486 | 70.000 | 1 |
| iter20_converged_frozen | 1475 | 60.000 | 1 |
| iter7_strong_contract | 1474 | 70.000 | 1 |
| iter15_strong_driverstrip | 1473 | 65.000 | 1 |
| iter11_strong_assign | 1473 | 60.000 | 1 |
| iter14_strong_truncsafe | 1472 | 85.000 | 1 |
| iter5_guided_oneshot | 1471 | 40.000 | 1 |
| iter3_selfconsistency_exec | 1465 | 65.000 | 4 |
| iter2_verify_retry_mini | 1455 | 60.000 | 2 |
| iter17_strong_validretry | 1319 | 100.000 | 1 |

### Iteration Progression

| Iter | Winner(s) | Score | Evo Time | Test Time | Total Time |
|------|-----------|-------|----------|-----------|------------|
| 1 | seed__mxgdywk | 70.000 | - | 1.5m | 1.5m |
| 2 | seed__mxgdywk | 70.000 | 7.5m | 3.9m | 11.4m |
| 3 | seed__mxgdywk | 75.000 | 10.6m | 5.8m | 16.4m |
| 4 | seed__mxgdywk | 80.000 | 10.0m | 8.3m | 18.3m |
| 5 | seed__mxgdywk | 75.000 | 5.6m | 4.5m | 10.1m |
| 6 | seed__mxgdywk, iter6_strong_oneshot | 70.000 | 4.5m | 4.1m | 8.6m |
| 7 | seed__mxgdywk | 85.000 | 5.8m | 4.2m | 10.0m |
| 8 | iter6_strong_oneshot | 75.000 | 7.1m | 3.2m | 10.3m |
| 9 | iter6_strong_oneshot | 90.000 | 4.5m | 2.6m | 7.1m |
| 10 | iter10_strong_reindent | 85.000 | 5.5m | 2.5m | 8.0m |
| 11 | iter10_strong_reindent | 70.000 | 4.8m | 2.5m | 7.3m |
| 12 | iter12_strong_toplevel | 90.000 | 4.5m | 3.0m | 7.5m |
| 13 | iter13_strong_multiwrap | 90.000 | 5.6m | 3.3m | 8.9m |
| 14 | iter12_strong_toplevel | 95.000 | 5.0m | 4.4m | 9.3m |
| 15 | iter12_strong_toplevel | 75.000 | 7.0m | 3.9m | 10.9m |
| 16 | iter16_strong_safestrip | 75.000 | 8.3m | 3.3m | 11.6m |
| 17 | iter16_strong_safestrip (iter17_strong_validretry* excluded) | 100.000 | 6.6m | 3.5m | 10.1m |
| 18 | iter18_strong_truncsafe | 85.000 | 4.3m | 3.8m | 8.1m |
| 19 | iter12_strong_toplevel | 70.000 | 5.8m | 3.7m | 9.5m |
| 20 | iter16_strong_safestrip | 70.000 | 4.4m | 3.8m | 8.2m |

\* *Exact clone: identical per-problem scores to an existing agent on debut. Elo penalized by 200; excluded from winner selection.*

## Best Agent: iter12_strong_toplevel
- Elo Score: 1602
- Mean Score: 78.571


## Elo Leadership Progression

Iteration-by-iteration Elo leaders:

| Iteration | Leader | Elo | Evolution Strategy |
|-----------|--------|-----|-------------------|
| 1 | seed__mxgdywk | 1500 | Initial |
| 2 | seed__mxgdywk | 1516 | Initial |
| 3 | seed__mxgdywk | 1545 | Initial |
| 4 | seed__mxgdywk | 1572 | Initial |
| 5 | seed__mxgdywk | 1597 | Initial |
| 6 | seed__mxgdywk | 1604 | Initial |
| 7 | seed__mxgdywk | 1627 | Initial |
| 8 | seed__mxgdywk | 1585 | Initial |
| 9 | iter6_strong_oneshot | 1587 | use_your_judgment |
| 10 | iter6_strong_oneshot | 1550 | use_your_judgment |
| 11 | iter10_strong_reindent | 1567 | use_your_judgment |
| 12 | iter6_strong_oneshot | 1549 | use_your_judgment |
| 13 | seed__mxgdywk | 1546 | Initial |
| 14 | iter12_strong_toplevel | 1565 | use_your_judgment |
| 15 | iter12_strong_toplevel | 1592 | use_your_judgment |
| 16 | iter12_strong_toplevel | 1585 | use_your_judgment |
| 17 | iter12_strong_toplevel | 1585 | use_your_judgment |
| 18 | iter12_strong_toplevel | 1585 | use_your_judgment |
| 19 | iter12_strong_toplevel | 1611 | use_your_judgment |
| 20 | iter12_strong_toplevel | 1602 | use_your_judgment |


## Deep Focus Evolution Timing Analysis
- **Iterations with evolution**: 19/20
- **Total evolution time**: 117.4 minutes
- **Average evolution time**: 6.2 minutes

### Time Breakdown by Evolution Phase

| Phase | Avg Time | % of Evolution |
|-------|----------|----------------|
| First Draft (Round 1) | 5.5m | 89.4% |
| Reflection | 0.7m | 10.6% |
| **Total** | **6.2m** | **100.0%** |

### Evolution vs Test vs Meta-Evolution Time

| Metric | Time | Percentage | Avg/Iter | Avg/Occurrence |
|--------|------|------------|----------|----------------|
| Total Evolution Time | 117.4m | 60.8% | 5.9m | 6.2m |
| Total Test Time | 75.8m | 39.2% | 3.8m | 3.8m |
| Total Meta-Evolution Time | 0.0m | 0.0% | 0.0m | 0.0m |
| **Total** | **193.2m** | **100.0%** | **9.7m** | **-** |

- **Evolution/Test ratio**: 1.55x
  - For every minute of testing, 1.55 minutes spent on evolution

## Claude CLI Usage & Cost Analysis
- **Total Cost**: $34.21
  - **Evaluation Cost**: $1.14
  - **Evolution CLI Cost**: $33.07
- **Total CLI Calls**: 38
- **Total CLI Input Tokens**: 35,469
- **Total CLI Output Tokens**: 489,246
- **Total Cache Created**: 1,296,611
- **Total Cache Read**: 15,311,228

### Cost Breakdown

| Operation | Cost | % of Total | Calls | Tokens In | Tokens Out |
|-----------|------|------------|-------|-----------|------------|
| Evaluation | $1.14 | 3.3% | - | - | - |
| Evolution (CLI) | $33.07 | 96.7% | 38 | 35,469 | 489,246 |
| **Total** | **$34.21** | **100%** | **38** | **35,469** | **489,246** |

**Answer: Evolution drives 96.7% of total costs**

### Detailed Per-Iteration Costs

| Iter | Total | **Eval** | **Evo** | Iter Eval | 1st Draft | Refl | Meta | Fresh | Strategy | Meta-strategy |
|------|------|------|------|------|------|------|------|------|------|------|
| **1** | $0.01 | **$0.01** | **-** | $0.01 | - | - | - | 20 | - | - |
| **2** | $1.72 | **$0.04** | **$1.68** | $0.04 | $1.52 | $0.15 | - | 35 | use_your_judgment | - |
| **3** | $3.49 | **$0.07** | **$3.42** | $0.07 | $2.50 | $0.92 | - | 53 | use_your_judgment | - |
| **4** | $2.45 | **$0.05** | **$2.39** | $0.05 | $2.22 | $0.18 | - | 47 | use_your_judgment | - |
| **5** | $1.43 | **$0.04** | **$1.40** | $0.04 | $1.25 | $0.15 | - | 38 | use_your_judgment | - |
| **6** | $1.21 | **$0.06** | **$1.15** | $0.06 | $1.02 | $0.13 | - | 39 | use_your_judgment | - |
| **7** | $1.58 | **$0.07** | **$1.50** | $0.07 | $1.35 | $0.15 | - | 45 | use_your_judgment | - |
| **8** | $2.25 | **$0.07** | **$2.18** | $0.07 | $1.51 | $0.67 | - | 42 | use_your_judgment | - |
| **9** | $1.21 | **$0.04** | **$1.17** | $0.04 | $1.01 | $0.16 | - | 30 | use_your_judgment | - |
| **10** | $1.51 | **$0.04** | **$1.47** | $0.04 | $1.32 | $0.15 | - | 31 | use_your_judgment | - |
| **11** | $1.44 | **$0.05** | **$1.39** | $0.05 | $1.24 | $0.16 | - | 32 | use_your_judgment | - |
| **12** | $1.42 | **$0.07** | **$1.35** | $0.07 | $1.20 | $0.15 | - | 40 | use_your_judgment | - |
| **13** | $1.63 | **$0.06** | **$1.56** | $0.06 | $1.41 | $0.15 | - | 38 | use_your_judgment | - |
| **14** | $1.42 | **$0.06** | **$1.36** | $0.06 | $1.19 | $0.16 | - | 48 | use_your_judgment | - |
| **15** | $1.95 | **$0.08** | **$1.87** | $0.08 | $1.71 | $0.17 | - | 44 | use_your_judgment | - |
| **16** | $2.39 | **$0.06** | **$2.33** | $0.06 | $2.15 | $0.18 | - | 38 | use_your_judgment | - |
| **17** | $2.29 | **$0.05** | **$2.24** | $0.05 | $1.53 | $0.71 | - | 37 | use_your_judgment | - |
| **18** | $1.40 | **$0.08** | **$1.33** | $0.08 | $1.16 | $0.16 | - | 44 | use_your_judgment | - |
| **19** | $1.90 | **$0.07** | **$1.84** | $0.07 | $1.67 | $0.17 | - | 43 | use_your_judgment | - |
| **20** | $1.50 | **$0.07** | **$1.43** | $0.07 | $1.26 | $0.17 | - | 39 | use_your_judgment | - |
|------|------|------|------|------|------|------|------|------|------|------|
| **TOTAL** | **$34.21** | **$1.14** | **$33.07** | $1.14 | $28.24 | $4.83 | $0.00 | **783** | - | - |

### Evolution Cost Breakdown (Avg per evolution)

| Phase | Avg Cost | % of Evolution | Avg Tokens In | Avg Tokens Out |
|-------|----------|----------------|---------------|----------------|
| First Draft (Round 1) | $1.49 | 85.4% | 1863 | 23562 |
| Reflection | $0.25 | 14.6% | 4 | 2188 |
| **Total** | **$1.74** | **100%** | **1867** | **25750** |

### Cache Performance

- **Cache Creation**: 1,296,611 tokens
- **Cache Hits**: 15,311,228 tokens (92.2% hit rate)

## Zero Accuracy Cases
No zero accuracy cases encountered ✅

## Exception Failures
No exception failures encountered ✅

## Evolution Health

No artifact validation failures ✅