# Parallel Agent Research - Final Report

## Experiment Configuration

**Run ID**: asta_ds1000_20260609_144857
**Date**: 2026-06-09 21:23:09
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

- file_mapping: {'agent.py': 'agent.py'}

### User Configuration (Iteration 1)

The following parameters were explicitly set, overriding defaults:

- agents_directory: ../robophd_runs/robophd/_optimize_anything_seeds
- domain: external
- eval_timeout: 1800
- evaluation_budget: 999999
- evolution_model: fable-5 (claude-fable-5)
- evolution_strategy: use_your_judgment
- evolution_timeout: 3600
- examples_per_iteration: 20
- initial_agents: ['seed_lj690bsd']
- max_workers: 5
- meta_evolution_cadence: 3
- meta_evolution_domain: asta_ds1000
- meta_evolution_first_iteration: 4
- meta_evolution_model: fable-5 (claude-fable-5)
- new_agent_test_rounds: 0

### Configuration Changes by Iteration

(No configuration changes across iterations)


## Winning Agents by Evolution Strategy

#### **Use Your Judgment** (19 wins - 95.0% of all wins)
- **iter2_crosscheck_adjudicate** (5 wins, created iteration 2)
- **iter6_inplace_expect_adjudicate** (4 wins, created iteration 6)
- **iter9_mplprobe_adjudicate** (4 wins, created iteration 9)
- **iter8_refquirk_adjudicate** (2 wins, created iteration 8)
- **iter14_filemock_adjudicate** (2 wins, created iteration 14)
- **iter3_tokenguard_adjudicate** (1 win, created iteration 3)
- **iter12_thirdvote_adjudicate** (1 win, created iteration 12)

#### **Initial** (1 wins - 5.0% of all wins)
- **seed_lj690bsd** (1 win, created iteration 0)

#### Summary Statistics
- **Total iterations with winners**: 15
- **Total wins counted**: 20 (includes ties)
- **Unique winning agents**: 8
- **Evolution strategies that produced winners**: 2


## Complete Performance Ranking Table

| Agent | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 | Iter 6 | Iter 7 | Iter 8 | Iter 9 | Iter 10 | Iter 11 | Iter 12 | Iter 13 | Iter 14 | Iter 15 | Final Elo | Mean Rank |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|-----------|----------|
| iter14_filemock_adjudicate | - | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 100.000 | **#1** 100.000 | **1531** | **1.00** |
| iter8_refquirk_adjudicate | - | - | - | - | - | - | - | **#1** 100.000 | #2 90.000 | - | - | - | **#1** 100.000 | #2 90.000 | - | 1530 | **1.50** |
| iter9_mplprobe_adjudicate | - | - | - | - | - | - | - | - | **#1** 95.000 | **#1** 95.000 | **#1** 100.000 | #2 85.000 | - | - | **#1** 100.000 | 1514 | **1.20** |
| iter2_crosscheck_adjudicate | - | **#1** 95.000 | **#1** 100.000 | **#1** 85.000 | **#1** 95.000 | **#1** 95.000 | #2 90.000 | #2 90.000 | #2 90.000 | - | - | - | - | - | - | 1508 | **1.38** |
| iter12_thirdvote_adjudicate | - | - | - | - | - | - | - | - | - | - | - | **#1** 95.000 | #3 95.000 | - | - | 1500 | 2.00 |
| iter7_directform_adjudicate | - | - | - | - | - | - | #2 90.000 | - | - | - | - | - | - | - | - | 1488 | 2.00 |
| iter6_inplace_expect_adjudicate | - | - | - | - | - | **#1** 95.000 | **#1** 95.000 | #2 90.000 | - | **#1** 95.000 | **#1** 100.000 | #2 85.000 | - | #3 85.000 | - | 1480 | **1.57** |
| seed_lj690bsd | **#1** 75.000 | #2 75.000 | #3 75.000 | - | - | - | - | - | - | - | - | - | - | - | - | 1455 | 2.50 |
| iter3_tokenguard_adjudicate | - | - | #2 95.000 | #3 80.000 | **#1** 95.000 | #3 90.000 | - | - | - | - | - | - | - | - | - | 1449 | 2.25 |
| iter13_fnsig_adjudicate | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 100.000 | - | - | 1318* | **1.00** |
| iter4_trivote_adjudicate | - | - | - | **#1** 85.000 | - | - | - | - | - | - | - | - | - | - | - | 1317* | **1.00** |
| iter10_loopfree_adjudicate | - | - | - | - | - | - | - | - | - | **#1** 95.000 | - | - | - | - | - | 1303* | **1.00** |
| iter11_labelconv_adjudicate | - | - | - | - | - | - | - | - | - | - | **#1** 100.000 | - | - | - | - | 1303* | **1.00** |
| iter15_mockverify_adjudicate | - | - | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 100.000 | 1302* | **1.00** |
| iter5_expectverify_adjudicate | - | - | - | - | **#1** 95.000 | - | - | - | - | - | - | - | - | - | - | 1301* | **1.00** |

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
| iter14_filemock_adjudicate | 1531 | 100.000 | 2 |
| iter8_refquirk_adjudicate | 1530 | 95.000 | 4 |
| iter9_mplprobe_adjudicate | 1514 | 95.000 | 5 |
| iter2_crosscheck_adjudicate | 1508 | 92.500 | 8 |
| iter12_thirdvote_adjudicate | 1500 | 95.000 | 2 |
| iter7_directform_adjudicate | 1488 | 90.000 | 1 |
| iter6_inplace_expect_adjudicate | 1480 | 92.143 | 7 |
| seed_lj690bsd | 1455 | 75.000 | 3 |
| iter3_tokenguard_adjudicate | 1449 | 90.000 | 4 |
| iter13_fnsig_adjudicate | 1318 | 100.000 | 1 |
| iter4_trivote_adjudicate | 1317 | 85.000 | 1 |
| iter10_loopfree_adjudicate | 1303 | 95.000 | 1 |
| iter11_labelconv_adjudicate | 1303 | 100.000 | 1 |
| iter15_mockverify_adjudicate | 1302 | 100.000 | 1 |
| iter5_expectverify_adjudicate | 1301 | 95.000 | 1 |

### Iteration Progression

| Iter | Winner(s) | Score | Evo Time | Test Time | Total Time |
|------|-----------|-------|----------|-----------|------------|
| 1 | seed_lj690bsd | 75.000 | - | 3.5m | 3.5m |
| 2 | iter2_crosscheck_adjudicate | 95.000 | 11.5m | 10.8m | 22.3m |
| 3 | iter2_crosscheck_adjudicate | 100.000 | 4.7m | 13.1m | 17.8m |
| 4 | iter2_crosscheck_adjudicate (iter4_trivote_adjudicate* excluded) | 85.000 | 6.6m | 15.8m | 22.4m |
| 5 | iter2_crosscheck_adjudicate, iter3_tokenguard_adjudicate (iter5_expectverify_adjudicate* excluded) | 95.000 | 8.3m | 18.7m | 27.0m |
| 6 | iter6_inplace_expect_adjudicate, iter2_crosscheck_adjudicate | 95.000 | 7.3m | 18.9m | 26.2m |
| 7 | iter6_inplace_expect_adjudicate | 95.000 | 6.4m | 24.8m | 31.2m |
| 8 | iter8_refquirk_adjudicate | 100.000 | 8.1m | 16.5m | 24.6m |
| 9 | iter9_mplprobe_adjudicate | 95.000 | 11.4m | 21.7m | 33.2m |
| 10 | iter9_mplprobe_adjudicate, iter6_inplace_expect_adjudicate (iter10_loopfree_adjudicate* excluded) | 95.000 | 8.6m | 21.3m | 29.9m |
| 11 | iter9_mplprobe_adjudicate, iter6_inplace_expect_adjudicate (iter11_labelconv_adjudicate* excluded) | 100.000 | 5.2m | 20.1m | 25.3m |
| 12 | iter12_thirdvote_adjudicate | 95.000 | 9.4m | 21.6m | 31.0m |
| 13 | iter8_refquirk_adjudicate (iter13_fnsig_adjudicate* excluded) | 100.000 | 7.0m | 31.6m | 38.6m |
| 14 | iter14_filemock_adjudicate | 100.000 | 11.2m | 18.0m | 29.2m |
| 15 | iter14_filemock_adjudicate, iter9_mplprobe_adjudicate (iter15_mockverify_adjudicate* excluded) | 100.000 | 10.9m | 21.0m | 31.9m |

\* *Exact clone: identical per-problem scores to an existing agent on debut. Elo penalized by 200; excluded from winner selection.*

## Best Agent: iter14_filemock_adjudicate
- Elo Score: 1531
- Mean Score: 100.000


## Elo Leadership Progression

Iteration-by-iteration Elo leaders:

| Iteration | Leader | Elo | Evolution Strategy |
|-----------|--------|-----|-------------------|
| 1 | seed_lj690bsd | 1500 | Initial |
| 2 | iter2_crosscheck_adjudicate | 1516 | use_your_judgment |
| 3 | iter2_crosscheck_adjudicate | 1545 | use_your_judgment |
| 4 | iter2_crosscheck_adjudicate | 1557 | use_your_judgment |
| 5 | iter2_crosscheck_adjudicate | 1551 | use_your_judgment |
| 6 | iter2_crosscheck_adjudicate | 1560 | use_your_judgment |
| 7 | iter6_inplace_expect_adjudicate | 1549 | use_your_judgment |
| 8 | iter8_refquirk_adjudicate | 1535 | use_your_judgment |
| 9 | iter9_mplprobe_adjudicate | 1534 | use_your_judgment |
| 10 | iter9_mplprobe_adjudicate | 1532 | use_your_judgment |
| 11 | iter9_mplprobe_adjudicate | 1531 | use_your_judgment |
| 12 | iter12_thirdvote_adjudicate | 1534 | use_your_judgment |
| 13 | iter8_refquirk_adjudicate | 1532 | use_your_judgment |
| 14 | iter14_filemock_adjudicate | 1533 | use_your_judgment |
| 15 | iter14_filemock_adjudicate | 1531 | use_your_judgment |


## Deep Focus Evolution Timing Analysis
- **Iterations with evolution**: 14/15
- **Total evolution time**: 116.9 minutes
- **Average evolution time**: 8.4 minutes

### Time Breakdown by Evolution Phase

| Phase | Avg Time | % of Evolution |
|-------|----------|----------------|
| First Draft (Round 1) | 7.6m | 90.9% |
| Reflection | 0.8m | 9.1% |
| **Total** | **8.4m** | **100.0%** |

### Evolution vs Test vs Meta-Evolution Time

| Metric | Time | Percentage | Avg/Iter | Avg/Occurrence |
|--------|------|------------|----------|----------------|
| Total Evolution Time | 116.9m | 29.7% | 7.8m | 8.4m |
| Total Test Time | 277.3m | 70.3% | 18.5m | 18.5m |
| Total Meta-Evolution Time | 0.0m | 0.0% | 0.0m | 0.0m |
| **Total** | **394.2m** | **100.0%** | **26.3m** | **-** |

- **Evolution/Test ratio**: 0.42x
  - For every minute of testing, 0.42 minutes spent on evolution

## Claude CLI Usage & Cost Analysis
- **Total Cost**: $85.40
  - **Evaluation Cost**: $21.32
  - **Evolution CLI Cost**: $64.09
- **Total CLI Calls**: 28
- **Total CLI Input Tokens**: 53,009
- **Total CLI Output Tokens**: 474,121
- **Total Cache Created**: 1,179,940
- **Total Cache Read**: 25,073,832

### Cost Breakdown

| Operation | Cost | % of Total | Calls | Tokens In | Tokens Out |
|-----------|------|------------|-------|-----------|------------|
| Evaluation | $21.32 | 25.0% | - | - | - |
| Evolution (CLI) | $64.09 | 75.0% | 28 | 53,009 | 474,121 |
| **Total** | **$85.40** | **100%** | **28** | **53,009** | **474,121** |

**Answer: Evolution drives 75.0% of total costs**

### Detailed Per-Iteration Costs

| Iter | Total | **Eval** | **Evo** | Iter Eval | 1st Draft | Refl | Meta | Fresh | Strategy | Meta-strategy |
|------|------|------|------|------|------|------|------|------|------|------|
| **1** | $0.01 | **$0.01** | **-** | $0.01 | - | - | - | 20 | - | - |
| **2** | $5.94 | **$0.65** | **$5.29** | $0.65 | $4.95 | $0.35 | - | 38 | use_your_judgment | - |
| **3** | $3.24 | **$0.89** | **$2.34** | $0.89 | $2.07 | $0.28 | - | 47 | use_your_judgment | - |
| **4** | $4.32 | **$1.26** | **$3.07** | $1.26 | $2.74 | $0.33 | - | 49 | use_your_judgment | - |
| **5** | $5.77 | **$1.76** | **$4.00** | $1.76 | $3.64 | $0.37 | - | 44 | use_your_judgment | - |
| **6** | $6.22 | **$1.40** | **$4.81** | $1.40 | $3.54 | $1.28 | - | 43 | use_your_judgment | - |
| **7** | $5.49 | **$1.68** | **$3.82** | $1.68 | $3.44 | $0.37 | - | 42 | use_your_judgment | - |
| **8** | $5.89 | **$1.38** | **$4.51** | $1.38 | $4.14 | $0.36 | - | 34 | use_your_judgment | - |
| **9** | $7.21 | **$1.58** | **$5.63** | $1.58 | $5.22 | $0.41 | - | 46 | use_your_judgment | - |
| **10** | $5.89 | **$1.51** | **$4.38** | $1.51 | $4.00 | $0.38 | - | 46 | use_your_judgment | - |
| **11** | $4.24 | **$1.33** | **$2.91** | $1.33 | $2.58 | $0.33 | - | 42 | use_your_judgment | - |
| **12** | $6.99 | **$1.82** | **$5.17** | $1.82 | $4.78 | $0.39 | - | 38 | use_your_judgment | - |
| **13** | $6.36 | **$2.19** | **$4.17** | $2.19 | $3.83 | $0.35 | - | 45 | use_your_judgment | - |
| **14** | $8.86 | **$1.85** | **$7.00** | $1.85 | $6.59 | $0.41 | - | 33 | use_your_judgment | - |
| **15** | $8.96 | **$1.99** | **$6.97** | $1.99 | $6.57 | $0.40 | - | 45 | use_your_judgment | - |
|------|------|------|------|------|------|------|------|------|------|------|
| **TOTAL** | **$85.40** | **$21.32** | **$64.09** | $21.32 | $58.08 | $6.01 | $0.00 | **612** | - | - |

### Evolution Cost Breakdown (Avg per evolution)

| Phase | Avg Cost | % of Evolution | Avg Tokens In | Avg Tokens Out |
|-------|----------|----------------|---------------|----------------|
| First Draft (Round 1) | $4.15 | 90.6% | 3387 | 31321 |
| Reflection | $0.43 | 9.4% | 399 | 2545 |
| **Total** | **$4.58** | **100%** | **3786** | **33866** |

### Cache Performance

- **Cache Creation**: 1,179,940 tokens
- **Cache Hits**: 25,073,832 tokens (95.5% hit rate)

## Zero Accuracy Cases
No zero accuracy cases encountered ✅

## Exception Failures
No exception failures encountered ✅

## Evolution Health

No artifact validation failures ✅