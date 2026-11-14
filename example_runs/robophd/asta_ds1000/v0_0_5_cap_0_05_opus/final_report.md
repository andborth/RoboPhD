# Parallel Agent Research - Final Report

## Experiment Configuration

**Run ID**: asta_ds1000_20260618_155040
**Date**: 2026-06-18 22:06:43
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
- initial_agents: ['seed_pl47_s10']
- max_workers: 10
- meta_evolution_cadence: 3
- meta_evolution_domain: asta_ds1000
- meta_evolution_first_iteration: 4
- meta_evolution_model: opus-4.8 (claude-opus-4-8)
- new_agent_test_rounds: 0

### Configuration Changes by Iteration

(No configuration changes across iterations)


## Winning Agents by Evolution Strategy

#### **Use Your Judgment** (26 wins - 96.3% of all wins)
- **iter3_fmt_strong_cascade** (5 wins, created iteration 3)
- **iter7_agree_escalate** (5 wins, created iteration 7)
- **iter9_reason_agree** (5 wins, created iteration 9)
- **iter10_literal_consensus** (5 wins, created iteration 10)
- **iter2_selfcheck_cascade** (3 wins, created iteration 2)
- **iter5_diverse_vote_mpl** (1 win, created iteration 5)
- **iter14_bounded_reconcile** (1 win, created iteration 14)
- **iter19_literal_struct_consensus** (1 win, created iteration 19)

#### **Initial** (1 wins - 3.7% of all wins)
- **seed_pl47_s10** (1 win, created iteration 0)

#### Summary Statistics
- **Total iterations with winners**: 19
- **Total wins counted**: 27 (includes ties)
- **Unique winning agents**: 9
- **Evolution strategies that produced winners**: 2


## Complete Performance Ranking Table

| Agent | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 | Iter 6 | Iter 7 | Iter 8 | Iter 9 | Iter 10 | Iter 11 | Iter 12 | Iter 13 | Iter 14 | Iter 15 | Iter 16 | Iter 17 | Iter 18 | Iter 19 | Final Elo | Mean Rank |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|-----------|----------|
| iter10_literal_consensus | - | - | - | - | - | - | - | - | - | **#1** 100.000 | **#1** 95.000 | **#1** 95.000 | **#1** 100.000 | - | - | - | - | **#1** 95.000 | #2 95.000 | **1560** | **1.17** |
| iter19_literal_struct_consensus | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 100.000 | 1535 | **1.00** |
| iter7_agree_escalate | - | - | - | - | - | - | #2 85.000 | **#1** 95.000 | **#1** 80.000 | #2 95.000 | **#1** 95.000 | #2 90.000 | - | **#1** 80.000 | - | **#1** 95.000 | #2 85.000 | - | - | 1516 | **1.44** |
| iter9_reason_agree | - | - | - | - | - | - | - | - | **#1** 80.000 | - | - | - | **#1** 100.000 | **#1** 80.000 | **#1** 95.000 | #2 90.000 | **#1** 90.000 | #3 85.000 | - | 1503 | **1.43** |
| iter3_fmt_strong_cascade | - | - | **#1** 75.000 | **#1** 90.000 | #2 85.000 | **#1** 80.000 | **#1** 90.000 | #3 90.000 | **#1** 80.000 | #3 90.000 | - | - | - | - | - | - | - | - | - | 1500 | **1.62** |
| iter5_diverse_vote_mpl | - | - | - | - | **#1** 95.000 | #3 70.000 | - | - | - | - | - | - | - | - | - | - | - | - | - | 1500 | 2.00 |
| iter2_selfcheck_cascade | - | **#1** 75.000 | **#1** 75.000 | **#1** 90.000 | #3 60.000 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1495 | **1.50** |
| iter18_reason_agree_struct | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | #2 90.000 | #2 95.000 | 1492 | 2.00 |
| iter17_agree_escalate_v2 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | #2 85.000 | - | - | 1486 | 2.00 |
| iter6_wide_vote | - | - | - | - | - | #2 75.000 | #3 80.000 | - | - | - | - | - | - | - | - | - | - | - | - | 1473 | 2.50 |
| iter14_bounded_reconcile | - | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 80.000 | #3 90.000 | - | - | - | - | 1469 | 2.00 |
| seed_pl47_s10 | **#1** 40.000 | #2 60.000 | #3 70.000 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1455 | 2.50 |
| iter8_reason_cascade | - | - | - | - | - | - | - | **#1** 95.000 | - | - | - | - | - | - | - | - | - | - | - | 1319* | **1.00** |
| iter15_agree_sharpen | - | - | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 95.000 | - | - | - | - | 1316* | **1.00** |
| iter13_output_reconcile | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 100.000 | - | - | - | - | - | - | 1303* | **1.00** |
| iter11_consensus_plus | - | - | - | - | - | - | - | - | - | - | **#1** 95.000 | - | - | - | - | - | - | - | - | 1302* | **1.00** |
| iter4_exec_vote_ensemble | - | - | - | **#1** 90.000 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1302* | **1.00** |
| iter12_func_agree | - | - | - | - | - | - | - | - | - | - | - | #2 90.000 | - | - | - | - | - | - | - | 1287* | 2.00 |
| iter16_triad_vote | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | #2 90.000 | - | - | - | 1286* | 2.00 |

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
| iter10_literal_consensus | 1560 | 96.667 | 6 |
| iter19_literal_struct_consensus | 1535 | 100.000 | 1 |
| iter7_agree_escalate | 1516 | 88.889 | 9 |
| iter9_reason_agree | 1503 | 88.571 | 7 |
| iter3_fmt_strong_cascade | 1500 | 85.000 | 8 |
| iter5_diverse_vote_mpl | 1500 | 82.500 | 2 |
| iter2_selfcheck_cascade | 1495 | 75.000 | 4 |
| iter18_reason_agree_struct | 1492 | 92.500 | 2 |
| iter17_agree_escalate_v2 | 1486 | 85.000 | 1 |
| iter6_wide_vote | 1473 | 77.500 | 2 |
| iter14_bounded_reconcile | 1469 | 85.000 | 2 |
| seed_pl47_s10 | 1455 | 56.667 | 3 |
| iter8_reason_cascade | 1319 | 95.000 | 1 |
| iter15_agree_sharpen | 1316 | 95.000 | 1 |
| iter13_output_reconcile | 1303 | 100.000 | 1 |
| iter11_consensus_plus | 1302 | 95.000 | 1 |
| iter4_exec_vote_ensemble | 1302 | 90.000 | 1 |
| iter12_func_agree | 1287 | 90.000 | 1 |
| iter16_triad_vote | 1286 | 90.000 | 1 |

### Iteration Progression

| Iter | Winner(s) | Score | Evo Time | Test Time | Total Time |
|------|-----------|-------|----------|-----------|------------|
| 1 | seed_pl47_s10 | 40.000 | - | 24.6m | 24.6m |
| 2 | iter2_selfcheck_cascade | 75.000 | 6.1m | 7.1m | 13.3m |
| 3 | iter2_selfcheck_cascade, iter3_fmt_strong_cascade | 75.000 | 6.3m | 11.5m | 17.9m |
| 4 | iter2_selfcheck_cascade, iter3_fmt_strong_cascade (iter4_exec_vote_ensemble* excluded) | 90.000 | 6.9m | 14.7m | 21.6m |
| 5 | iter5_diverse_vote_mpl | 95.000 | 5.9m | 12.2m | 18.1m |
| 6 | iter3_fmt_strong_cascade | 80.000 | 3.4m | 15.9m | 19.3m |
| 7 | iter3_fmt_strong_cascade | 90.000 | 7.2m | 34.0m | 41.2m |
| 8 | iter7_agree_escalate (iter8_reason_cascade* excluded) | 95.000 | 5.2m | 18.4m | 23.5m |
| 9 | iter7_agree_escalate +2 | 80.000 | 3.6m | 12.8m | 16.4m |
| 10 | iter10_literal_consensus | 100.000 | 4.8m | 10.4m | 15.1m |
| 11 | iter10_literal_consensus, iter7_agree_escalate (iter11_consensus_plus* excluded) | 95.000 | 5.4m | 18.6m | 24.0m |
| 12 | iter10_literal_consensus (iter12_func_agree* excluded) | 95.000 | 6.6m | 14.8m | 21.3m |
| 13 | iter10_literal_consensus, iter9_reason_agree (iter13_output_reconcile* excluded) | 100.000 | 5.2m | 15.9m | 21.1m |
| 14 | iter9_reason_agree +2 | 80.000 | 6.1m | 11.3m | 17.4m |
| 15 | iter9_reason_agree (iter15_agree_sharpen* excluded) | 95.000 | 5.2m | 14.8m | 19.9m |
| 16 | iter7_agree_escalate (iter16_triad_vote* excluded) | 95.000 | 5.2m | 10.2m | 15.5m |
| 17 | iter9_reason_agree | 90.000 | 3.7m | 8.4m | 12.0m |
| 18 | iter10_literal_consensus | 95.000 | 5.7m | 9.5m | 15.1m |
| 19 | iter19_literal_struct_consensus | 100.000 | 4.3m | 14.1m | 18.5m |

\* *Exact clone: identical per-problem scores to an existing agent on debut. Elo penalized by 200; excluded from winner selection.*

## Best Agent: iter10_literal_consensus
- Elo Score: 1560
- Mean Score: 96.667


## Elo Leadership Progression

Iteration-by-iteration Elo leaders:

| Iteration | Leader | Elo | Evolution Strategy |
|-----------|--------|-----|-------------------|
| 1 | seed_pl47_s10 | 1500 | Initial |
| 2 | iter2_selfcheck_cascade | 1516 | use_your_judgment |
| 3 | iter2_selfcheck_cascade | 1530 | use_your_judgment |
| 4 | iter2_selfcheck_cascade | 1528 | use_your_judgment |
| 5 | iter5_diverse_vote_mpl | 1533 | use_your_judgment |
| 6 | iter3_fmt_strong_cascade | 1547 | use_your_judgment |
| 7 | iter3_fmt_strong_cascade | 1574 | use_your_judgment |
| 8 | iter3_fmt_strong_cascade | 1536 | use_your_judgment |
| 9 | iter3_fmt_strong_cascade | 1534 | use_your_judgment |
| 10 | iter10_literal_consensus | 1534 | use_your_judgment |
| 11 | iter10_literal_consensus | 1532 | use_your_judgment |
| 12 | iter10_literal_consensus | 1561 | use_your_judgment |
| 13 | iter10_literal_consensus | 1556 | use_your_judgment |
| 14 | iter10_literal_consensus | 1556 | use_your_judgment |
| 15 | iter10_literal_consensus | 1556 | use_your_judgment |
| 16 | iter10_literal_consensus | 1556 | use_your_judgment |
| 17 | iter10_literal_consensus | 1556 | use_your_judgment |
| 18 | iter10_literal_consensus | 1583 | use_your_judgment |
| 19 | iter10_literal_consensus | 1560 | use_your_judgment |


## Deep Focus Evolution Timing Analysis
- **Iterations with evolution**: 18/19
- **Total evolution time**: 96.7 minutes
- **Average evolution time**: 5.4 minutes

### Time Breakdown by Evolution Phase

| Phase | Avg Time | % of Evolution |
|-------|----------|----------------|
| First Draft (Round 1) | 4.8m | 88.9% |
| Reflection | 0.6m | 11.1% |
| **Total** | **5.4m** | **100.0%** |

### Evolution vs Test vs Meta-Evolution Time

| Metric | Time | Percentage | Avg/Iter | Avg/Occurrence |
|--------|------|------------|----------|----------------|
| Total Evolution Time | 96.7m | 25.7% | 5.1m | 5.4m |
| Total Test Time | 279.3m | 74.3% | 14.7m | 14.7m |
| Total Meta-Evolution Time | 0.0m | 0.0% | 0.0m | 0.0m |
| **Total** | **376.0m** | **100.0%** | **19.8m** | **-** |

- **Evolution/Test ratio**: 0.35x
  - For every minute of testing, 0.35 minutes spent on evolution

## Claude CLI Usage & Cost Analysis
- **Total Cost**: $40.39
  - **Evaluation Cost**: $8.72
  - **Evolution CLI Cost**: $31.67
- **Total CLI Calls**: 36
- **Total CLI Input Tokens**: 89,026
- **Total CLI Output Tokens**: 417,093
- **Total Cache Created**: 1,120,700
- **Total Cache Read**: 19,113,333

### Cost Breakdown

| Operation | Cost | % of Total | Calls | Tokens In | Tokens Out |
|-----------|------|------------|-------|-----------|------------|
| Evaluation | $8.72 | 21.6% | - | - | - |
| Evolution (CLI) | $31.67 | 78.4% | 36 | 89,026 | 417,093 |
| **Total** | **$40.39** | **100%** | **36** | **89,026** | **417,093** |

**Answer: Evolution drives 78.4% of total costs**

### Detailed Per-Iteration Costs

| Iter | Total | **Eval** | **Evo** | Iter Eval | 1st Draft | Refl | Meta | Fresh | Strategy | Meta-strategy |
|------|------|------|------|------|------|------|------|------|------|------|
| **1** | $0.01 | **$0.01** | **-** | $0.01 | - | - | - | 20 | - | - |
| **2** | $1.72 | **$0.05** | **$1.67** | $0.05 | $1.52 | $0.15 | - | 37 | use_your_judgment | - |
| **3** | $2.42 | **$0.14** | **$2.28** | $0.14 | $1.62 | $0.66 | - | 54 | use_your_judgment | - |
| **4** | $2.06 | **$0.27** | **$1.79** | $0.27 | $1.61 | $0.18 | - | 55 | use_your_judgment | - |
| **5** | $2.12 | **$0.26** | **$1.86** | $0.26 | $1.69 | $0.17 | - | 44 | use_your_judgment | - |
| **6** | $1.83 | **$0.55** | **$1.28** | $0.55 | $1.14 | $0.14 | - | 49 | use_your_judgment | - |
| **7** | $2.33 | **$0.47** | **$1.85** | $0.47 | $1.68 | $0.18 | - | 41 | use_your_judgment | - |
| **8** | $1.66 | **$0.30** | **$1.35** | $0.30 | $1.20 | $0.16 | - | 41 | use_your_judgment | - |
| **9** | $1.91 | **$0.59** | **$1.33** | $0.59 | $1.18 | $0.14 | - | 37 | use_your_judgment | - |
| **10** | $2.10 | **$0.38** | **$1.72** | $0.38 | $1.56 | $0.16 | - | 37 | use_your_judgment | - |
| **11** | $2.45 | **$0.68** | **$1.77** | $0.68 | $1.61 | $0.16 | - | 49 | use_your_judgment | - |
| **12** | $3.03 | **$0.42** | **$2.61** | $0.42 | $2.42 | $0.19 | - | 31 | use_your_judgment | - |
| **13** | $2.91 | **$0.94** | **$1.96** | $0.94 | $1.80 | $0.17 | - | 49 | use_your_judgment | - |
| **14** | $2.83 | **$0.76** | **$2.07** | $0.76 | $1.88 | $0.19 | - | 39 | use_your_judgment | - |
| **15** | $2.36 | **$0.73** | **$1.63** | $0.73 | $1.47 | $0.16 | - | 50 | use_your_judgment | - |
| **16** | $1.99 | **$0.52** | **$1.47** | $0.52 | $1.31 | $0.15 | - | 33 | use_your_judgment | - |
| **17** | $1.73 | **$0.31** | **$1.42** | $0.31 | $1.27 | $0.16 | - | 29 | use_your_judgment | - |
| **18** | $2.31 | **$0.45** | **$1.86** | $0.45 | $1.70 | $0.16 | - | 31 | use_your_judgment | - |
| **19** | $2.65 | **$0.90** | **$1.75** | $0.90 | $1.58 | $0.17 | - | 42 | use_your_judgment | - |
|------|------|------|------|------|------|------|------|------|------|------|
| **TOTAL** | **$40.39** | **$8.72** | **$31.67** | $8.72 | $28.23 | $3.44 | $0.00 | **768** | - | - |

### Evolution Cost Breakdown (Avg per evolution)

| Phase | Avg Cost | % of Evolution | Avg Tokens In | Avg Tokens Out |
|-------|----------|----------------|---------------|----------------|
| First Draft (Round 1) | $1.57 | 89.1% | 4509 | 21107 |
| Reflection | $0.19 | 10.9% | 437 | 2064 |
| **Total** | **$1.76** | **100%** | **4946** | **23172** |

### Cache Performance

- **Cache Creation**: 1,120,700 tokens
- **Cache Hits**: 19,113,333 tokens (94.5% hit rate)

## Zero Accuracy Cases
No zero accuracy cases encountered ✅

## Exception Failures
No exception failures encountered ✅

## Evolution Health

No artifact validation failures ✅