# Parallel Agent Research - Final Report

## Experiment Configuration

**Run ID**: asta_ds1000_20261014_081142
**Date**: 2026-10-14 16:14:13
**Iterations**: 999

### System Defaults (Iteration 0)

Default values when no configuration is specified:

- agents_directory: None
- agents_per_iteration: 3
- dataset: train-filtered
- debug_log_probability: 0.02
- dollar_budget: None
- domain: external
- elo_reachability_guard: True
- elo_reachability_min_history: 5
- eval_result_cache: True
- eval_timeout: 300
- evaluation_budget: None
- evolution_model: opus-5.5 (claude-opus-5-5)
- evolution_strategy: use_your_judgment
- evolution_timeout: 3600
- examples_per_iteration: 20
- include_evolution_rankings: False
- include_prior_evolution: True
- initial_agents: ['naive']
- llm_call_timeout: 120
- max_workers: None
- meta_evolution_cadence: 3
- meta_evolution_domain: None
- meta_evolution_first_iteration: 4
- meta_evolution_model: opus-5.5 (claude-opus-5-5)
- new_agent_test_round_offset: -2
- new_agent_test_rounds: 1
- oldest_agent_wins_ties: False
- random_agent_wins_ties: True
- runs_directory: None
- strategies_directory: None
- use_weighted_random: False

### Task Configuration

Task-specific parameters (from task defaults + --task-config overrides):

- ds1000_runtime: {'cost_threshold': 0.127, 'cost_per_error': 0.0127}
- file_mapping: {'agent.py': 'agent.py'}

### User Configuration (Iteration 1)

The following parameters were explicitly set, overriding defaults:

- agents_directory: ../robophd_runs/robophd/_optimize_anything_seeds/seeds_0hf_cch3
- domain: external
- eval_timeout: 1800
- evaluation_budget: 750
- evolution_model: fable-5.1 (claude-fable-5-1)
- evolution_strategy: use_your_judgment
- evolution_timeout: 3600
- examples_per_iteration: 20
- initial_agents: ['baseline']
- max_workers: 10
- meta_evolution_cadence: 3
- meta_evolution_domain: asta_ds1000
- meta_evolution_first_iteration: 4
- meta_evolution_model: opus-5.5 (claude-opus-5-5)
- new_agent_test_rounds: 0

### Configuration Changes by Iteration

- **Iteration 21**: evolution_strategy=greedy


## Winning Agents by Evolution Strategy

#### **Use Your Judgment** (27 wins - 96.4% of all wins)
- **iter6_ensemble_exec_vote_escalate** (15 wins, created iteration 6)
- **iter2_exec_verified_sonnet** (4 wins, created iteration 2)
- **iter3_ensemble_exec_vote** (4 wins, created iteration 3)
- **iter10_ensemble_exec_vote_figure_state** (3 wins, created iteration 10)
- **iter11_ensemble_exec_vote_seeded_exec** (1 win, created iteration 11)

#### **Initial** (1 wins - 3.6% of all wins)
- **baseline** (1 win, created iteration 0)

#### Summary Statistics
- **Total iterations with winners**: 22
- **Total wins counted**: 28 (includes ties)
- **Unique winning agents**: 6
- **Evolution strategies that produced winners**: 2


## Complete Performance Ranking Table

| Agent | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 | Iter 6 | Iter 7 | Iter 8 | Iter 9 | Iter 10 | Iter 11 | Iter 12 | Iter 13 | Iter 14 | Iter 15 | Iter 16 | Iter 17 | Iter 18 | Iter 19 | Iter 20 | Iter 21 | Iter 22 | Final Elo | Mean Rank |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|-----------|----------|
| iter6_ensemble_exec_vote_escalate | - | - | - | - | - | **#1** 90.000 | **#1** 95.000 | **#1** 100.000 | **#1** 100.000 | #2 95.000 | **#1** 95.000 | **#1** 95.000 | #3 95.000 | **#1** 100.000 | **#1** 100.000 | **#1** 100.000 | **#1** 95.000 | **#1** 100.000 | **#1** 95.000 | **#1** 100.000 | **#1** 100.000 | **#1** 100.000 | **1610** | **1.18** |
| iter3_ensemble_exec_vote | - | - | #2 85.000 | **#1** 90.000 | **#1** 100.000 | #3 80.000 | - | - | **#1** 100.000 | #2 95.000 | - | - | - | - | - | - | - | - | - | - | **#1** 100.000 | #2 95.000 | 1474 | **1.62** |
| iter11_ensemble_exec_vote_seeded... | - | - | - | - | - | - | - | - | - | - | #3 85.000 | - | - | - | - | - | - | - | **#1** 95.000 | #3 95.000 | - | - | 1452 | 2.33 |
| iter2_exec_verified_sonnet | - | **#1** 75.000 | **#1** 90.000 | **#1** 90.000 | **#1** 100.000 | #2 85.000 | #3 85.000 | #3 95.000 | - | - | - | - | - | - | - | - | - | #3 95.000 | - | - | - | - | 1447 | **1.88** |
| iter10_ensemble_exec_vote_figure... | - | - | - | - | - | - | - | - | - | **#1** 100.000 | #2 90.000 | #3 90.000 | **#1** 100.000 | #3 95.000 | #2 95.000 | **#1** 100.000 | #3 90.000 | - | - | - | - | #2 95.000 | 1446 | 2.00 |
| baseline | **#1** 60.000 | #2 65.000 | #3 80.000 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | #3 85.000 | - | 1430 | 2.67 |
| iter12_ensemble_exec_vote_dump_fit | - | - | - | - | - | - | - | - | - | - | - | **#1** 95.000 | - | - | - | - | - | - | - | - | - | - | 1319* | **1.00** |
| iter13_ensemble_exec_vote_struct... | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 100.000 | - | - | - | - | - | - | - | - | - | 1319* | **1.00** |
| iter14_ensemble_exec_vote_probe_... | - | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 100.000 | - | - | - | - | - | - | - | - | 1318* | **1.00** |
| iter7_ensemble_exec_vote_synthdata | - | - | - | - | - | - | **#1** 95.000 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1318* | **1.00** |
| iter20_ensemble_exec_vote_reliab... | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 100.000 | - | - | 1318* | **1.00** |
| iter17_ensemble_exec_vote_fallba... | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 95.000 | - | - | - | - | - | 1318* | **1.00** |
| iter8_ensemble_exec_vote_recall_... | - | - | - | - | - | - | - | **#1** 100.000 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1318* | **1.00** |
| iter18_ensemble_exec_vote_conven... | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 100.000 | - | - | - | - | 1317* | **1.00** |
| iter19_ensemble_exec_vote_reliab... | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 95.000 | - | - | - | 1303* | **1.00** |
| iter16_ensemble_exec_vote_corrob... | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 100.000 | - | - | - | - | - | - | 1302* | **1.00** |
| iter4_ensemble_exec_vote_hardened | - | - | - | **#1** 90.000 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1302* | **1.00** |
| iter5_ensemble_exec_vote_dtype_j... | - | - | - | - | **#1** 100.000 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1302* | **1.00** |
| iter9_ensemble_exec_vote_api_guard | - | - | - | - | - | - | - | - | **#1** 100.000 | - | - | - | - | - | - | - | - | - | - | - | - | - | 1302* | **1.00** |
| iter15_ensemble_exec_vote_probe_... | - | - | - | - | - | - | - | - | - | - | - | - | - | - | #2 95.000 | - | - | - | - | - | - | - | 1286* | 2.00 |

### Legend:
- **#1** = 1st place (winner of iteration)
- #2, #3, etc. = 2nd, 3rd place, etc.
- Score = Average score (0-1) on that iteration's problems
- **Bold Elo/Rank** = Top performer
- `-` = Agent not tested in that iteration
- \* *Exact clone: identical per-problem scores to an existing agent on debut. Elo penalized by 200.*



### Quick Summary

| Agent | Elo | Mean Score | Mean Cost | Tests |
|-------|-----|------------|-----------|-------|
| iter6_ensemble_exec_vote_escalate | 1610 | 97.353 | $0.054 | 17 |
| iter3_ensemble_exec_vote | 1474 | 93.125 | $0.039 | 8 |
| iter11_ensemble_exec_vote_seeded_exec | 1452 | 91.667 | $0.052 | 3 |
| iter2_exec_verified_sonnet | 1447 | 89.375 | $0.013 | 8 |
| iter10_ensemble_exec_vote_figure_state | 1446 | 95.000 | $0.063 | 9 |
| baseline | 1430 | 72.500 | $0.001 | 4 |
| iter12_ensemble_exec_vote_dump_fit | 1319 | 95.000 | $0.060 | 1 |
| iter13_ensemble_exec_vote_structural_probe | 1319 | 100.000 | $0.053 | 1 |
| iter14_ensemble_exec_vote_probe_inplace_targets | 1318 | 100.000 | $0.053 | 1 |
| iter7_ensemble_exec_vote_synthdata | 1318 | 95.000 | $0.055 | 1 |
| iter20_ensemble_exec_vote_reliability_emergency | 1318 | 100.000 | $0.062 | 1 |
| iter17_ensemble_exec_vote_fallback_archivist | 1318 | 95.000 | $0.065 | 1 |
| iter8_ensemble_exec_vote_recall_dtype | 1318 | 100.000 | $0.051 | 1 |
| iter18_ensemble_exec_vote_convention_judge | 1317 | 100.000 | $0.049 | 1 |
| iter19_ensemble_exec_vote_reliability_fallback | 1303 | 95.000 | $0.049 | 1 |
| iter16_ensemble_exec_vote_corroborated_recall | 1302 | 100.000 | $0.051 | 1 |
| iter4_ensemble_exec_vote_hardened | 1302 | 90.000 | $0.042 | 1 |
| iter5_ensemble_exec_vote_dtype_judge | 1302 | 100.000 | $0.048 | 1 |
| iter9_ensemble_exec_vote_api_guard | 1302 | 100.000 | $0.051 | 1 |
| iter15_ensemble_exec_vote_probe_function_form | 1286 | 95.000 | $0.052 | 1 |

### Iteration Progression

| Iter | Winner(s) | Score | Evo Time | Test Time | Total Time |
|------|-----------|-------|----------|-----------|------------|
| 1 | baseline | 60.000 | - | 1.5m | 1.5m |
| 2 | iter2_exec_verified_sonnet | 75.000 | 10.1m | 3.9m | 13.9m |
| 3 | iter2_exec_verified_sonnet | 90.000 | 9.3m | 48.0m | 57.3m |
| 4 | iter2_exec_verified_sonnet, iter3_ensemble_exec_vote (iter4_ensemble_exec_vote_hardened* excluded) | 90.000 | 16.8m | 22.2m | 39.0m |
| 5 | iter2_exec_verified_sonnet, iter3_ensemble_exec_vote (iter5_ensemble_exec_vote_dtype_judge* excluded) | 100.000 | 9.8m | 18.4m | 28.2m |
| 6 | iter6_ensemble_exec_vote_escalate | 90.000 | 13.0m | 19.1m | 32.0m |
| 7 | iter6_ensemble_exec_vote_escalate (iter7_ensemble_exec_vote_synthdata* excluded) | 95.000 | 9.7m | 13.0m | 22.8m |
| 8 | iter6_ensemble_exec_vote_escalate (iter8_ensemble_exec_vote_recall_dtype* excluded) | 100.000 | 8.5m | 12.7m | 21.2m |
| 9 | iter6_ensemble_exec_vote_escalate, iter3_ensemble_exec_vote (iter9_ensemble_exec_vote_api_guard* excluded) | 100.000 | 8.9m | 14.2m | 23.0m |
| 10 | iter10_ensemble_exec_vote_figure_state | 100.000 | 10.2m | 15.7m | 26.0m |
| 11 | iter6_ensemble_exec_vote_escalate | 95.000 | 9.0m | 12.3m | 21.4m |
| 12 | iter6_ensemble_exec_vote_escalate (iter12_ensemble_exec_vote_dump_fit* excluded) | 95.000 | 8.5m | 11.3m | 19.8m |
| 13 | iter10_ensemble_exec_vote_figure_state (iter13_ensemble_exec_vote_structural_probe* excluded) | 100.000 | 11.0m | 12.5m | 23.5m |
| 14 | iter6_ensemble_exec_vote_escalate (iter14_ensemble_exec_vote_probe_inplace_targets* excluded) | 100.000 | 9.8m | 11.6m | 21.4m |
| 15 | iter6_ensemble_exec_vote_escalate (iter15_ensemble_exec_vote_probe_function_form* excluded) | 100.000 | 10.0m | 9.8m | 19.8m |
| 16 | iter6_ensemble_exec_vote_escalate, iter10_ensemble_exec_vote_figure_state (iter16_ensemble_exec_vote_corroborated_recall* excluded) | 100.000 | 13.6m | 10.1m | 23.7m |
| 17 | iter6_ensemble_exec_vote_escalate (iter17_ensemble_exec_vote_fallback_archivist* excluded) | 95.000 | 12.4m | 11.0m | 23.4m |
| 18 | iter6_ensemble_exec_vote_escalate (iter18_ensemble_exec_vote_convention_judge* excluded) | 100.000 | 14.8m | 6.9m | 21.7m |
| 19 | iter6_ensemble_exec_vote_escalate, iter11_ensemble_exec_vote_seeded_exec (iter19_ensemble_exec_vote_reliability_fallback* excluded) | 95.000 | 8.7m | 11.2m | 19.9m |
| 20 | iter6_ensemble_exec_vote_escalate (iter20_ensemble_exec_vote_reliability_emergency* excluded) | 100.000 | 5.3m | 11.0m | 16.3m |
| 21 | iter6_ensemble_exec_vote_escalate, iter3_ensemble_exec_vote | 100.000 | - | 2.0m | 2.0m |
| 22 | iter6_ensemble_exec_vote_escalate | 100.000 | - | 4.7m | 4.7m |

\* *Exact clone: identical per-problem scores to an existing agent on debut. Elo penalized by 200; excluded from winner selection.*

## Best Agent: iter6_ensemble_exec_vote_escalate
- Elo Score: 1610
- Mean Score: 97.353


## Elo Leadership Progression

Iteration-by-iteration Elo leaders:

| Iteration | Leader | Elo | Evolution Strategy |
|-----------|--------|-----|-------------------|
| 1 | baseline | 1500 | Initial |
| 2 | iter2_exec_verified_sonnet | 1516 | use_your_judgment |
| 3 | iter2_exec_verified_sonnet | 1545 | use_your_judgment |
| 4 | iter2_exec_verified_sonnet | 1541 | use_your_judgment |
| 5 | iter2_exec_verified_sonnet | 1537 | use_your_judgment |
| 6 | iter2_exec_verified_sonnet | 1534 | use_your_judgment |
| 7 | iter6_ensemble_exec_vote_escalate | 1548 | use_your_judgment |
| 8 | iter6_ensemble_exec_vote_escalate | 1559 | use_your_judgment |
| 9 | iter6_ensemble_exec_vote_escalate | 1553 | use_your_judgment |
| 10 | iter10_ensemble_exec_vote_figure_state | 1533 | use_your_judgment |
| 11 | iter6_ensemble_exec_vote_escalate | 1562 | use_your_judgment |
| 12 | iter6_ensemble_exec_vote_escalate | 1574 | use_your_judgment |
| 13 | iter6_ensemble_exec_vote_escalate | 1536 | use_your_judgment |
| 14 | iter6_ensemble_exec_vote_escalate | 1549 | use_your_judgment |
| 15 | iter6_ensemble_exec_vote_escalate | 1575 | use_your_judgment |
| 16 | iter6_ensemble_exec_vote_escalate | 1567 | use_your_judgment |
| 17 | iter6_ensemble_exec_vote_escalate | 1576 | use_your_judgment |
| 18 | iter6_ensemble_exec_vote_escalate | 1584 | use_your_judgment |
| 19 | iter6_ensemble_exec_vote_escalate | 1575 | use_your_judgment |
| 20 | iter6_ensemble_exec_vote_escalate | 1583 | use_your_judgment |
| 21 | iter6_ensemble_exec_vote_escalate | 1588 | use_your_judgment |
| 22 | iter6_ensemble_exec_vote_escalate | 1610 | use_your_judgment |


## Deep Focus Evolution Timing Analysis
- **Iterations with evolution**: 19/22
- **Total evolution time**: 199.3 minutes
- **Average evolution time**: 10.5 minutes

### Time Breakdown by Evolution Phase

| Phase | Avg Time | % of Evolution |
|-------|----------|----------------|
| First Draft (Round 1) | 9.7m | 92.7% |
| Reflection | 0.8m | 7.3% |
| **Total** | **10.5m** | **100.0%** |

### Evolution vs Test vs Meta-Evolution Time

| Metric | Time | Percentage | Avg/Iter | Avg/Occurrence |
|--------|------|------------|----------|----------------|
| Total Evolution Time | 199.3m | 41.3% | 9.1m | 10.5m |
| Total Test Time | 283.2m | 58.7% | 12.9m | 12.9m |
| Total Meta-Evolution Time | 0.0m | 0.0% | 0.0m | 0.0m |
| **Total** | **482.5m** | **100.0%** | **21.9m** | **-** |

- **Evolution/Test ratio**: 0.70x
  - For every minute of testing, 0.70 minutes spent on evolution

## Claude CLI Usage & Cost Analysis
- **Total Cost**: $200.94
  - **Evaluation Cost**: $33.84
  - **Evolution CLI Cost**: $167.10
- **Total CLI Calls**: 38
- **Total CLI Input Tokens**: 15,602
- **Total CLI Output Tokens**: 707,802
- **Total Cache Created**: 2,005,491
- **Total Cache Read**: 41,598,448

### Cost Breakdown

| Operation | Cost | % of Total | Calls | Tokens In | Tokens Out |
|-----------|------|------------|-------|-----------|------------|
| Evaluation | $33.84 | 16.8% | - | - | - |
| Evolution (CLI) | $167.10 | 83.2% | 38 | 15,602 | 707,802 |
| **Total** | **$200.94** | **100%** | **38** | **15,602** | **707,802** |

**Answer: Evolution drives 83.2% of total costs**

### Detailed Per-Iteration Costs

| Iter | Total | **Eval** | **Evo** | Iter Eval | 1st Draft | Refl | Meta | Fresh | Strategy | Meta-strategy |
|------|------|------|------|------|------|------|------|------|------|------|
| **1** | $0.01 | **$0.01** | **-** | $0.01 | - | - | - | 20 | - | - |
| **2** | $9.05 | **$0.27** | **$8.77** | $0.27 | $4.27 | $4.50 | - | 36 | use_your_judgment | - |
| **3** | $9.35 | **$1.07** | **$8.28** | $1.07 | $4.01 | $4.27 | - | 49 | use_your_judgment | - |
| **4** | $12.60 | **$1.48** | **$11.11** | $1.48 | $5.41 | $5.70 | - | 43 | use_your_judgment | - |
| **5** | $11.26 | **$1.88** | **$9.38** | $1.88 | $4.55 | $4.83 | - | 50 | use_your_judgment | - |
| **6** | $10.56 | **$1.99** | **$8.57** | $1.99 | $4.16 | $4.41 | - | 42 | use_your_judgment | - |
| **7** | $12.08 | **$2.19** | **$9.89** | $2.19 | $4.81 | $5.08 | - | 39 | use_your_judgment | - |
| **8** | $9.22 | **$2.00** | **$7.23** | $2.00 | $3.48 | $3.75 | - | 44 | use_your_judgment | - |
| **9** | $10.66 | **$1.90** | **$8.76** | $1.90 | $4.25 | $4.51 | - | 44 | use_your_judgment | - |
| **10** | $11.33 | **$1.70** | **$9.63** | $1.70 | $4.67 | $4.96 | - | 39 | use_your_judgment | - |
| **11** | $10.84 | **$2.57** | **$8.26** | $2.57 | $3.99 | $4.27 | - | 42 | use_your_judgment | - |
| **12** | $11.65 | **$2.60** | **$9.05** | $2.60 | $4.39 | $4.66 | - | 39 | use_your_judgment | - |
| **13** | $12.86 | **$1.87** | **$10.99** | $1.87 | $5.37 | $5.62 | - | 36 | use_your_judgment | - |
| **14** | $7.95 | **$1.85** | **$6.10** | $1.85 | $2.93 | $3.17 | - | 34 | use_your_judgment | - |
| **15** | $8.81 | **$1.64** | **$7.17** | $1.64 | $3.47 | $3.71 | - | 30 | use_your_judgment | - |
| **16** | $11.53 | **$1.45** | **$10.08** | $1.45 | $4.90 | $5.18 | - | 25 | use_your_judgment | - |
| **17** | $10.51 | **$1.75** | **$8.76** | $1.75 | $4.23 | $4.52 | - | 26 | use_your_judgment | - |
| **18** | $12.63 | **$1.02** | **$11.61** | $1.02 | $5.66 | $5.95 | - | 24 | use_your_judgment | - |
| **19** | $9.28 | **$1.79** | **$7.48** | $1.79 | $3.62 | $3.87 | - | 39 | use_your_judgment | - |
| **20** | $8.03 | **$2.07** | **$5.97** | $2.07 | $2.87 | $3.10 | - | 33 | use_your_judgment | - |
| **21** | $0.21 | **$0.21** | **-** | $0.21 | - | - | - | 12 | - | - |
| **22** | $0.51 | **$0.51** | **-** | $0.51 | - | - | - | 8 | - | - |
|------|------|------|------|------|------|------|------|------|------|------|
| **TOTAL** | **$200.94** | **$33.84** | **$167.10** | $33.84 | $81.04 | $86.06 | $0.00 | **754** | - | - |

### Evolution Cost Breakdown (Avg per evolution)

| Phase | Avg Cost | % of Evolution | Avg Tokens In | Avg Tokens Out |
|-------|----------|----------------|---------------|----------------|
| First Draft (Round 1) | $4.27 | 48.5% | 787 | 34750 |
| Reflection | $4.53 | 51.5% | 34 | 2502 |
| **Total** | **$8.79** | **100%** | **821** | **37253** |

### Cache Performance

- **Cache Creation**: 2,005,491 tokens
- **Cache Hits**: 41,598,448 tokens (95.4% hit rate)

## Zero Accuracy Cases
No zero accuracy cases encountered ✅

## Exception Failures
No exception failures encountered ✅

## Evolution Health

No artifact validation failures ✅