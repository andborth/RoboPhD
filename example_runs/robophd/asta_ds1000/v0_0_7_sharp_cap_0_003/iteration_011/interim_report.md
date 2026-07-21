# Parallel Agent Research - Interim Report (Iteration 11)

## Experiment Configuration

**Run ID**: asta_ds1000_20260721_033200
**Date**: 2026-07-21 05:21:03
**Iterations**: 11/999

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

#### **Initial** (7 wins - 58.3% of all wins)
- **seed__mxgdywk** (7 wins, created iteration 0)

#### **Use Your Judgment** (5 wins - 41.7% of all wins)
- **iter6_strong_oneshot** (3 wins, created iteration 6)
- **iter10_strong_reindent** (2 wins, created iteration 10)

#### Summary Statistics
- **Total iterations with winners**: 11
- **Total wins counted**: 12 (includes ties)
- **Unique winning agents**: 3
- **Evolution strategies that produced winners**: 2


## Complete Performance Ranking Table

| Agent | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 | Iter 6 | Iter 7 | Iter 8 | Iter 9 | Iter 10 | Iter 11 | Final Elo | Mean Rank |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|-----------|----------|
| iter10_strong_reindent | - | - | - | - | - | - | - | - | - | **#1** 85.000 | **#1** 70.000 | **1567** | **1.00** |
| iter6_strong_oneshot | - | - | - | - | - | **#1** 70.000 | #2 80.000 | **#1** 75.000 | **#1** 90.000 | #3 65.000 | - | 1550 | **1.60** |
| seed__mxgdywk | **#1** 70.000 | **#1** 70.000 | **#1** 75.000 | **#1** 80.000 | **#1** 75.000 | **#1** 70.000 | **#1** 85.000 | #3 65.000 | #3 80.000 | #2 75.000 | #2 65.000 | 1546 | **1.60** |
| iter8_raw_strong | - | - | - | - | - | - | - | #2 70.000 | - | - | - | 1506 | 2.00 |
| iter9_strong_oneshot_unescape | - | - | - | - | - | - | - | - | #2 85.000 | - | - | 1506 | 2.00 |
| iter4_verify_reason_escalate | - | - | - | #2 70.000 | - | - | - | - | - | - | - | 1486 | 2.00 |
| iter7_strong_contract | - | - | - | - | - | - | #3 70.000 | - | - | - | - | 1474 | 3.00 |
| iter11_strong_assign | - | - | - | - | - | - | - | - | - | - | #3 60.000 | 1473 | 3.00 |
| iter5_guided_oneshot | - | - | - | - | #3 40.000 | - | - | - | - | - | - | 1471 | 3.00 |
| iter3_selfconsistency_exec | - | - | #2 65.000 | #2 70.000 | #2 70.000 | #3 55.000 | - | - | - | - | - | 1465 | 2.25 |
| iter2_verify_retry_mini | - | #2 60.000 | #3 60.000 | - | - | - | - | - | - | - | - | 1455 | 2.50 |

### Legend:
- **#1** = 1st place (winner of iteration)
- #2, #3, etc. = 2nd, 3rd place, etc.
- Score = Average score (0-1) on that iteration's problems
- **Bold Elo/Rank** = Top performer
- `-` = Agent not tested in that iteration



### Quick Summary

| Agent | Elo | Mean Score | Tests |
|-------|-----|------------|-------|
| iter10_strong_reindent | 1567 | 77.500 | 2 |
| iter6_strong_oneshot | 1550 | 76.000 | 5 |
| seed__mxgdywk | 1546 | 73.636 | 11 |
| iter8_raw_strong | 1506 | 70.000 | 1 |
| iter9_strong_oneshot_unescape | 1506 | 85.000 | 1 |
| iter4_verify_reason_escalate | 1486 | 70.000 | 1 |
| iter7_strong_contract | 1474 | 70.000 | 1 |
| iter11_strong_assign | 1473 | 60.000 | 1 |
| iter5_guided_oneshot | 1471 | 40.000 | 1 |
| iter3_selfconsistency_exec | 1465 | 65.000 | 4 |
| iter2_verify_retry_mini | 1455 | 60.000 | 2 |

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

## Best Agent: iter10_strong_reindent
- Elo Score: 1567
- Mean Score: 77.500


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


## Deep Focus Evolution Timing Analysis
- **Iterations with evolution**: 10/11
- **Total evolution time**: 65.8 minutes
- **Average evolution time**: 6.6 minutes

### Time Breakdown by Evolution Phase

| Phase | Avg Time | % of Evolution |
|-------|----------|----------------|
| First Draft (Round 1) | 6.0m | 90.5% |
| Reflection | 0.6m | 9.5% |
| **Total** | **6.6m** | **100.0%** |

### Evolution vs Test vs Meta-Evolution Time

| Metric | Time | Percentage | Avg/Iter | Avg/Occurrence |
|--------|------|------------|----------|----------------|
| Total Evolution Time | 65.8m | 60.4% | 6.0m | 6.6m |
| Total Test Time | 43.2m | 39.6% | 3.9m | 3.9m |
| Total Meta-Evolution Time | 0.0m | 0.0% | 0.0m | 0.0m |
| **Total** | **109.0m** | **100.0%** | **9.9m** | **-** |

- **Evolution/Test ratio**: 1.52x
  - For every minute of testing, 1.52 minutes spent on evolution

## Claude CLI Usage & Cost Analysis
- **Total Cost**: $18.30
  - **Evaluation Cost**: $0.54
  - **Evolution CLI Cost**: $17.76
- **Total CLI Calls**: 20
- **Total CLI Input Tokens**: 5,537
- **Total CLI Output Tokens**: 270,901
- **Total Cache Created**: 693,495
- **Total Cache Read**: 8,003,908

### Cost Breakdown

| Operation | Cost | % of Total | Calls | Tokens In | Tokens Out |
|-----------|------|------------|-------|-----------|------------|
| Evaluation | $0.54 | 3.0% | - | - | - |
| Evolution (CLI) | $17.76 | 97.0% | 20 | 5,537 | 270,901 |
| **Total** | **$18.30** | **100%** | **20** | **5,537** | **270,901** |

**Answer: Evolution drives 97.0% of total costs**

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
|------|------|------|------|------|------|------|------|------|------|------|
| **TOTAL** | **$18.30** | **$0.54** | **$17.76** | $0.54 | $14.94 | $2.81 | $0.00 | **412** | - | - |

### Evolution Cost Breakdown (Avg per evolution)

| Phase | Avg Cost | % of Evolution | Avg Tokens In | Avg Tokens Out |
|-------|----------|----------------|---------------|----------------|
| First Draft (Round 1) | $1.49 | 84.2% | 550 | 24970 |
| Reflection | $0.28 | 15.8% | 4 | 2120 |
| **Total** | **$1.78** | **100%** | **554** | **27090** |

### Cache Performance

- **Cache Creation**: 693,495 tokens
- **Cache Hits**: 8,003,908 tokens (92.0% hit rate)

## Zero Accuracy Cases
No zero accuracy cases encountered ✅

## Exception Failures
No exception failures encountered ✅

## Evolution Health

No artifact validation failures ✅