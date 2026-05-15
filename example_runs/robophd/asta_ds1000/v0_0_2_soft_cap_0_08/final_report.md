# Parallel Agent Research - Final Report

## Experiment Configuration

**Run ID**: asta_ds1000_20260514_231614
**Date**: 2026-05-15 04:55:51
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

- agents_directory: ../robophd_runs/robophd/_optimize_anything_seeds
- domain: external
- eval_timeout: 1200
- evaluation_budget: 999999
- evolution_model: opus-4.7 (claude-opus-4-7)
- evolution_strategy: use_your_judgment
- evolution_timeout: 3600
- examples_per_iteration: 20
- initial_agents: ['seed_yyg6m9ud']
- max_workers: 6
- meta_evolution_cadence: 3
- meta_evolution_domain: asta_ds1000
- meta_evolution_first_iteration: 4
- new_agent_test_rounds: 0

### Configuration Changes by Iteration

(No configuration changes across iterations)


## Winning Agents by Evolution Strategy

#### **Use Your Judgment** (18 wins - 90.0% of all wins)
- **iter4_ds1000_idiom_probe** (8 wins, created iteration 4)
- **iter7_ds1000_example_verify** (4 wins, created iteration 7)
- **iter3_ds1000_format_aware** (3 wins, created iteration 3)
- **iter9_ds1000_anchor_robust** (2 wins, created iteration 9)
- **iter14_ds1000_opus_literal** (1 win, created iteration 14)

#### **Initial** (2 wins - 10.0% of all wins)
- **seed_yyg6m9ud** (2 wins, created iteration 0)

#### Summary Statistics
- **Total iterations with winners**: 15
- **Total wins counted**: 20 (includes ties)
- **Unique winning agents**: 6
- **Evolution strategies that produced winners**: 2


## Complete Performance Ranking Table

| Agent | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 | Iter 6 | Iter 7 | Iter 8 | Iter 9 | Iter 10 | Iter 11 | Iter 12 | Iter 13 | Iter 14 | Iter 15 | Final ELO | Mean Rank |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|-----------|----------|
| iter4_ds1000_idiom_probe | - | - | - | **#1** 100.000 | **#1** 95.000 | **#1** 95.000 | #2 85.000 | **#1** 90.000 | #2 95.000 | - | - | **#1** 95.000 | **#1** 90.000 | **#1** 95.000 | **#1** 90.000 | **1552** | **1.20** |
| iter14_ds1000_opus_literal | - | - | - | - | - | - | - | - | - | - | - | - | - | #2 90.000 | **#1** 90.000 | 1504 | **1.50** |
| iter9_ds1000_anchor_robust | - | - | - | - | - | - | - | - | **#1** 100.000 | **#1** 95.000 | #3 90.000 | - | - | - | - | 1499 | **1.67** |
| iter7_ds1000_example_verify | - | - | - | - | - | - | **#1** 90.000 | **#1** 90.000 | #2 95.000 | **#1** 95.000 | **#1** 95.000 | #3 90.000 | - | - | - | 1498 | **1.50** |
| seed_yyg6m9ud | **#1** 70.000 | **#1** 80.000 | #3 65.000 | - | - | - | - | - | - | - | - | - | - | - | - | 1483 | 2.00 |
| iter2_ds1000_verify_repair | - | #2 75.000 | #2 75.000 | #2 95.000 | - | - | - | - | - | - | - | - | - | - | - | 1474 | 2.00 |
| iter3_ds1000_format_aware | - | - | **#1** 80.000 | #2 95.000 | #2 85.000 | **#1** 95.000 | #2 85.000 | - | - | - | - | - | **#1** 90.000 | #3 85.000 | - | 1471 | **1.71** |
| iter5_ds1000_consensus | - | - | - | - | #3 60.000 | - | - | - | - | - | - | - | - | - | - | 1471 | 3.00 |
| iter11_ds1000_genstress_probe | - | - | - | - | - | - | - | - | - | - | **#1** 95.000 | - | - | - | - | 1318* | **1.00** |
| iter12_ds1000_echoproof_verify | - | - | - | - | - | - | - | - | - | - | - | **#1** 95.000 | - | - | - | 1318* | **1.00** |
| iter6_ds1000_trap_audit | - | - | - | - | - | **#1** 95.000 | - | - | - | - | - | - | - | - | - | 1303* | **1.00** |
| iter8_ds1000_dtype_anchor | - | - | - | - | - | - | - | **#1** 90.000 | - | - | - | - | - | - | - | 1303* | **1.00** |
| iter15_ds1000_audit_split | - | - | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 90.000 | 1303* | **1.00** |
| iter10_ds1000_gt_bestof | - | - | - | - | - | - | - | - | - | **#1** 95.000 | - | - | - | - | - | 1302* | **1.00** |
| iter13_ds1000_gt_echoproof | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 90.000 | - | - | 1301* | **1.00** |

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
| iter4_ds1000_idiom_probe | 1552 | 93.000 | 10 |
| iter14_ds1000_opus_literal | 1504 | 90.000 | 2 |
| iter9_ds1000_anchor_robust | 1499 | 95.000 | 3 |
| iter7_ds1000_example_verify | 1498 | 92.500 | 6 |
| seed_yyg6m9ud | 1483 | 71.667 | 3 |
| iter2_ds1000_verify_repair | 1474 | 81.667 | 3 |
| iter3_ds1000_format_aware | 1471 | 87.857 | 7 |
| iter5_ds1000_consensus | 1471 | 60.000 | 1 |
| iter11_ds1000_genstress_probe | 1318 | 95.000 | 1 |
| iter12_ds1000_echoproof_verify | 1318 | 95.000 | 1 |
| iter6_ds1000_trap_audit | 1303 | 95.000 | 1 |
| iter8_ds1000_dtype_anchor | 1303 | 90.000 | 1 |
| iter15_ds1000_audit_split | 1303 | 90.000 | 1 |
| iter10_ds1000_gt_bestof | 1302 | 95.000 | 1 |
| iter13_ds1000_gt_echoproof | 1301 | 90.000 | 1 |

### Iteration Progression

| Iter | Winner(s) | Score | Evo Time | Test Time | Total Time |
|------|-----------|-------|----------|-----------|------------|
| 1 | seed_yyg6m9ud | 70.000 | - | 15.2m | 15.2m |
| 2 | seed_yyg6m9ud | 80.000 | 4.3m | 33.0m | 37.3m |
| 3 | iter3_ds1000_format_aware | 80.000 | 12.4m | 53.4m | 65.9m |
| 4 | iter4_ds1000_idiom_probe | 100.000 | 8.2m | 9.2m | 17.4m |
| 5 | iter4_ds1000_idiom_probe | 95.000 | 7.8m | 43.0m | 50.7m |
| 6 | iter4_ds1000_idiom_probe, iter3_ds1000_format_aware (iter6_ds1000_trap_audit* excluded) | 95.000 | 6.4m | 6.6m | 13.0m |
| 7 | iter7_ds1000_example_verify | 90.000 | 7.5m | 5.7m | 13.2m |
| 8 | iter7_ds1000_example_verify, iter4_ds1000_idiom_probe (iter8_ds1000_dtype_anchor* excluded) | 90.000 | 8.7m | 7.7m | 16.4m |
| 9 | iter9_ds1000_anchor_robust | 100.000 | 9.5m | 6.4m | 15.9m |
| 10 | iter9_ds1000_anchor_robust, iter7_ds1000_example_verify (iter10_ds1000_gt_bestof* excluded) | 95.000 | 7.5m | 9.8m | 17.4m |
| 11 | iter7_ds1000_example_verify (iter11_ds1000_genstress_probe* excluded) | 95.000 | 11.9m | 7.2m | 19.1m |
| 12 | iter4_ds1000_idiom_probe (iter12_ds1000_echoproof_verify* excluded) | 95.000 | 10.4m | 4.8m | 15.1m |
| 13 | iter4_ds1000_idiom_probe, iter3_ds1000_format_aware (iter13_ds1000_gt_echoproof* excluded) | 90.000 | 7.8m | 5.5m | 13.2m |
| 14 | iter4_ds1000_idiom_probe | 95.000 | 5.9m | 11.0m | 16.8m |
| 15 | iter4_ds1000_idiom_probe, iter14_ds1000_opus_literal (iter15_ds1000_audit_split* excluded) | 90.000 | 6.7m | 6.1m | 12.9m |

\* *Exact clone: identical per-problem scores to an existing agent on debut. ELO penalized by 200; excluded from winner selection.*

## Best Agent: iter4_ds1000_idiom_probe
- ELO Score: 1552
- Mean Score: 93.000


## ELO Leadership Progression

Iteration-by-iteration ELO leaders:

| Iteration | Leader | ELO | Evolution Strategy |
|-----------|--------|-----|-------------------|
| 1 | seed_yyg6m9ud | 1500 | Initial |
| 2 | seed_yyg6m9ud | 1516 | Initial |
| 3 | iter3_ds1000_format_aware | 1531 | use_your_judgment |
| 4 | iter4_ds1000_idiom_probe | 1532 | use_your_judgment |
| 5 | iter4_ds1000_idiom_probe | 1561 | use_your_judgment |
| 6 | iter4_ds1000_idiom_probe | 1556 | use_your_judgment |
| 7 | iter4_ds1000_idiom_probe | 1536 | use_your_judgment |
| 8 | iter4_ds1000_idiom_probe | 1534 | use_your_judgment |
| 9 | iter9_ds1000_anchor_robust | 1534 | use_your_judgment |
| 10 | iter9_ds1000_anchor_robust | 1532 | use_your_judgment |
| 11 | iter7_ds1000_example_verify | 1531 | use_your_judgment |
| 12 | iter4_ds1000_idiom_probe | 1532 | use_your_judgment |
| 13 | iter4_ds1000_idiom_probe | 1529 | use_your_judgment |
| 14 | iter4_ds1000_idiom_probe | 1557 | use_your_judgment |
| 15 | iter4_ds1000_idiom_probe | 1552 | use_your_judgment |


## Deep Focus Evolution Timing Analysis
- **Iterations with evolution**: 14/15
- **Total evolution time**: 114.9 minutes
- **Average evolution time**: 8.2 minutes

### Time Breakdown by Evolution Phase

| Phase | Avg Time | % of Evolution |
|-------|----------|----------------|
| First Draft (Round 1) | 7.5m | 91.9% |
| Reflection | 0.7m | 8.1% |
| **Total** | **8.2m** | **100.0%** |

### Evolution vs Test vs Meta-Evolution Time

| Metric | Time | Percentage | Avg/Iter | Avg/Occurrence |
|--------|------|------------|----------|----------------|
| Total Evolution Time | 114.9m | 33.8% | 7.7m | 8.2m |
| Total Test Time | 224.7m | 66.2% | 15.0m | 15.0m |
| Total Meta-Evolution Time | 0.0m | 0.0% | 0.0m | 0.0m |
| **Total** | **339.6m** | **100.0%** | **22.6m** | **-** |

- **Evolution/Test ratio**: 0.51x
  - For every minute of testing, 0.51 minutes spent on evolution

## Claude CLI Usage & Cost Analysis
- **Total Cost**: $33.91
  - **Evaluation Cost**: $9.73
  - **Evolution CLI Cost**: $24.17
- **Total CLI Calls**: 28
- **Total CLI Input Tokens**: 6,781
- **Total CLI Output Tokens**: 450,955
- **Total Cache Created**: 1,076,489
- **Total Cache Read**: 12,218,521

### Cost Breakdown

| Operation | Cost | % of Total | Calls | Tokens In | Tokens Out |
|-----------|------|------------|-------|-----------|------------|
| Evaluation | $9.73 | 28.7% | - | - | - |
| Evolution (CLI) | $24.17 | 71.3% | 28 | 6,781 | 450,955 |
| **Total** | **$33.91** | **100%** | **28** | **6,781** | **450,955** |

**Answer: Evolution drives 71.3% of total costs**

### Detailed Per-Iteration Costs

| Iter | Total | **Eval** | **Evo** | Iter Eval | 1st Draft | Refl | Meta | Fresh | Strategy | Meta-strategy |
|------|------|------|------|------|------|------|------|------|------|------|
| **1** | $0.01 | **$0.01** | **-** | $0.01 | - | - | - | 20 | - | - |
| **2** | $1.00 | **$0.18** | **$0.81** | $0.18 | $0.70 | $0.11 | - | 36 | use_your_judgment | - |
| **3** | $3.55 | **$0.26** | **$3.28** | $0.26 | $2.65 | $0.64 | - | 45 | use_your_judgment | - |
| **4** | $2.07 | **$0.51** | **$1.57** | $0.51 | $1.42 | $0.14 | - | 52 | use_your_judgment | - |
| **5** | $2.10 | **$0.36** | **$1.74** | $0.36 | $1.60 | $0.13 | - | 44 | use_your_judgment | - |
| **6** | $2.07 | **$0.68** | **$1.39** | $0.68 | $1.25 | $0.14 | - | 43 | use_your_judgment | - |
| **7** | $1.95 | **$0.38** | **$1.57** | $0.38 | $1.40 | $0.16 | - | 37 | use_your_judgment | - |
| **8** | $2.69 | **$1.02** | **$1.66** | $1.02 | $1.51 | $0.15 | - | 45 | use_your_judgment | - |
| **9** | $2.46 | **$0.65** | **$1.81** | $0.65 | $1.66 | $0.15 | - | 37 | use_your_judgment | - |
| **10** | $3.41 | **$1.90** | **$1.51** | $1.90 | $1.37 | $0.15 | - | 48 | use_your_judgment | - |
| **11** | $3.43 | **$1.02** | **$2.41** | $1.02 | $2.23 | $0.18 | - | 39 | use_your_judgment | - |
| **12** | $2.35 | **$0.37** | **$1.98** | $0.37 | $1.82 | $0.15 | - | 33 | use_your_judgment | - |
| **13** | $2.85 | **$0.97** | **$1.88** | $0.97 | $1.71 | $0.17 | - | 33 | use_your_judgment | - |
| **14** | $1.77 | **$0.56** | **$1.21** | $0.56 | $1.08 | $0.13 | - | 31 | use_your_judgment | - |
| **15** | $2.21 | **$0.85** | **$1.36** | $0.85 | $1.23 | $0.13 | - | 34 | use_your_judgment | - |
|------|------|------|------|------|------|------|------|------|------|------|
| **TOTAL** | **$33.91** | **$9.73** | **$24.17** | $9.73 | $21.63 | $2.54 | $0.00 | **577** | - | - |

### Evolution Cost Breakdown (Avg per evolution)

| Phase | Avg Cost | % of Evolution | Avg Tokens In | Avg Tokens Out |
|-------|----------|----------------|---------------|----------------|
| First Draft (Round 1) | $1.55 | 89.5% | 481 | 30260 |
| Reflection | $0.18 | 10.5% | 3 | 1952 |
| **Total** | **$1.73** | **100%** | **484** | **32211** |

### Cache Performance

- **Cache Creation**: 1,076,489 tokens
- **Cache Hits**: 12,218,521 tokens (91.9% hit rate)

## Zero Accuracy Cases
No zero accuracy cases encountered ✅

## Exception Failures
No exception failures encountered ✅

## Evolution Health

No artifact validation failures ✅