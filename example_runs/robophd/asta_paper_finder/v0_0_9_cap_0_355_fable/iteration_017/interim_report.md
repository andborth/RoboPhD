# Parallel Agent Research - Interim Report (Iteration 17)

## Experiment Configuration

**Run ID**: asta_paper_finder_20260802_103017
**Date**: 2026-08-02 23:30:32
**Iterations**: 17/999

### System Defaults (Iteration 0)

Default values when no configuration is specified:

- agents_directory: None
- agents_per_iteration: 3
- dataset: train-filtered
- debug_log_probability: 0.02
- dollar_budget: None
- domain: external
- elo_reachability_guard: False
- elo_reachability_min_history: 5
- eval_result_cache: True
- eval_timeout: 300
- evaluation_budget: None
- evolution_model: opus-5 (claude-opus-5)
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
- meta_evolution_model: opus-5 (claude-opus-5)
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
- paper_finder_runtime: {'cost_threshold': 0.355, 'cost_per_error': 0.0355, 'cap_judge_to_estimate': True, 'training_judge': 'openai/gpt-5.6-luna', 'judge_prompt': 'no-prose', 'evidence_char_cap': 2500}

### User Configuration (Iteration 1)

The following parameters were explicitly set, overriding defaults:

- agents_directory: ../robophd_runs/robophd/_optimize_anything_seeds
- domain: external
- elo_reachability_guard: True
- eval_timeout: 1800
- evaluation_budget: 600
- evolution_model: fable-5 (claude-fable-5)
- evolution_strategy: use_your_judgment
- evolution_timeout: 3600
- examples_per_iteration: 14
- initial_agents: ['seed_pkvc5eb0']
- max_workers: 8
- meta_evolution_cadence: 3
- meta_evolution_domain: asta_paper_finder
- meta_evolution_first_iteration: 4
- meta_evolution_model: opus-5 (claude-opus-5)
- new_agent_test_rounds: 0

### Configuration Changes by Iteration

(No configuration changes across iterations)


## Winning Agents by Evolution Strategy

#### **Use Your Judgment** (16 wins - 94.1% of all wins)
- **iter9_metafix_poolboost_v1** (4 wins, created iteration 9)
- **iter3_recallforge_v1** (2 wins, created iteration 3)
- **iter4_citegraph_evidence_v1** (2 wins, created iteration 4)
- **iter12_citerevive_multiref_v1** (2 wins, created iteration 12)
- **iter2_router_deeplist_v1** (1 win, created iteration 2)
- **iter6_evrepair_cite_v1** (1 win, created iteration 6)
- **iter8_nickname_depth_v1** (1 win, created iteration 8)
- **iter13_largegold_termnames_v1** (1 win, created iteration 13)
- **iter15_authorsfix_goldceiling_v1** (1 win, created iteration 15)
- **iter17_ambigunion_landmark_v1** (1 win, created iteration 17)

#### **Initial** (1 wins - 5.9% of all wins)
- **seed_pkvc5eb0** (1 win, created iteration 0)

#### Summary Statistics
- **Total iterations with winners**: 17
- **Total wins counted**: 17 (includes ties)
- **Unique winning agents**: 11
- **Evolution strategies that produced winners**: 2


## Complete Performance Ranking Table

| Agent | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 | Iter 6 | Iter 7 | Iter 8 | Iter 9 | Iter 10 | Iter 11 | Iter 12 | Iter 13 | Iter 14 | Iter 15 | Iter 16 | Iter 17 | Final Elo | Mean Rank |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|-----------|----------|
| iter12_citerevive_multiref_v1 | - | - | - | - | - | - | - | - | - | - | - | **#1** 53.255 | #2 51.078 | **#1** 39.586 | #2 46.157 | - | #2 37.452 | **1559** | **1.60** |
| iter9_metafix_poolboost_v1 | - | - | - | - | - | - | - | - | **#1** 51.016 | **#1** 50.280 | **#1** 51.299 | #2 52.762 | - | - | #3 45.570 | **#1** 47.356 | #3 35.587 | 1540 | **1.71** |
| iter17_ambigunion_landmark_v1 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 46.900 | 1537 | **1.00** |
| iter15_authorsfix_goldceiling_v1 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 48.266 | #3 46.088 | - | 1505 | 2.00 |
| iter16_specplan_authorchan_v1 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | #2 46.095 | - | 1504 | 2.00 |
| iter14_overflow_minigrade_v1 | - | - | - | - | - | - | - | - | - | - | - | - | - | #2 33.038 | - | - | - | 1503 | 2.00 |
| iter10_refsolo_hedgefix_v1 | - | - | - | - | - | - | - | - | - | #2 47.850 | - | - | - | - | - | - | - | 1503 | 2.00 |
| iter6_evrepair_cite_v1 | - | - | - | - | - | **#1** 38.900 | #3 32.654 | - | - | - | - | - | - | - | - | - | - | 1502 | 2.00 |
| iter8_nickname_depth_v1 | - | - | - | - | - | - | - | **#1** 50.572 | #2 48.217 | - | - | #3 51.878 | - | - | - | - | - | 1502 | 2.00 |
| iter13_largegold_termnames_v1 | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 51.562 | #3 32.278 | - | - | - | 1500 | 2.00 |
| iter4_citegraph_evidence_v1 | - | - | - | **#1** 40.998 | #2 35.679 | #2 33.590 | **#1** 35.500 | #3 39.823 | #3 38.331 | - | - | - | - | - | - | - | - | 1493 | 2.00 |
| iter3_recallforge_v1 | - | - | **#1** 33.918 | #2 35.017 | **#1** 40.223 | #3 33.495 | - | - | - | #3 43.158 | - | - | - | - | - | - | - | 1492 | 2.00 |
| iter2_router_deeplist_v1 | - | **#1** 32.602 | #2 33.726 | #3 29.860 | - | - | - | - | - | - | - | - | - | - | - | - | - | 1483 | 2.00 |
| iter7_robustharvest_v1 | - | - | - | - | - | - | #2 34.242 | #2 49.657 | - | - | #3 50.356 | - | - | - | - | - | - | 1476 | 2.33 |
| iter11_secondwave_fullk_v1 | - | - | - | - | - | - | - | - | - | - | #2 51.167 | - | #3 49.409 | - | - | - | - | 1473 | 2.50 |
| iter5_robustcite_blend_v1 | - | - | - | - | #3 34.731 | - | - | - | - | - | - | - | - | - | - | - | - | 1471 | 3.00 |
| seed_pkvc5eb0 | **#1** 2.100 | #2 3.370 | #3 3.331 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1455 | 2.50 |

### Legend:
- **#1** = 1st place (winner of iteration)
- #2, #3, etc. = 2nd, 3rd place, etc.
- Score = Average score (0-1) on that iteration's problems
- **Bold Elo/Rank** = Top performer
- `-` = Agent not tested in that iteration



### Quick Summary

| Agent | Elo | Mean Score | Mean Cost | Tests |
|-------|-----|------------|-----------|-------|
| iter12_citerevive_multiref_v1 | 1559 | 45.505 | $0.240 | 5 |
| iter9_metafix_poolboost_v1 | 1540 | 47.696 | $0.187 | 7 |
| iter17_ambigunion_landmark_v1 | 1537 | 46.900 | $0.232 | 1 |
| iter15_authorsfix_goldceiling_v1 | 1505 | 47.177 | $0.257 | 2 |
| iter16_specplan_authorchan_v1 | 1504 | 46.095 | $0.250 | 1 |
| iter14_overflow_minigrade_v1 | 1503 | 33.038 | $0.309 | 1 |
| iter10_refsolo_hedgefix_v1 | 1503 | 47.850 | $0.249 | 1 |
| iter6_evrepair_cite_v1 | 1502 | 35.777 | $0.159 | 2 |
| iter8_nickname_depth_v1 | 1502 | 50.222 | $0.188 | 3 |
| iter13_largegold_termnames_v1 | 1500 | 41.920 | $0.289 | 2 |
| iter4_citegraph_evidence_v1 | 1493 | 37.320 | $0.154 | 6 |
| iter3_recallforge_v1 | 1492 | 37.162 | $0.087 | 5 |
| iter2_router_deeplist_v1 | 1483 | 32.063 | $0.050 | 3 |
| iter7_robustharvest_v1 | 1476 | 44.752 | $0.158 | 3 |
| iter11_secondwave_fullk_v1 | 1473 | 50.288 | $0.251 | 2 |
| iter5_robustcite_blend_v1 | 1471 | 34.731 | $0.199 | 1 |
| seed_pkvc5eb0 | 1455 | 2.934 | $0.001 | 3 |

### Iteration Progression

| Iter | Winner(s) | Score | Evo Time | Test Time | Total Time |
|------|-----------|-------|----------|-----------|------------|
| 1 | seed_pkvc5eb0 | 2.100 | - | 2.4m | 2.4m |
| 2 | iter2_router_deeplist_v1 | 32.602 | 12.8m | 5.7m | 18.6m |
| 3 | iter3_recallforge_v1 | 33.918 | 8.5m | 12.4m | 20.9m |
| 4 | iter4_citegraph_evidence_v1 | 40.998 | 12.1m | 18.6m | 30.6m |
| 5 | iter3_recallforge_v1 | 40.223 | 11.3m | 20.5m | 31.7m |
| 6 | iter6_evrepair_cite_v1 | 38.900 | 11.8m | 34.4m | 46.3m |
| 7 | iter4_citegraph_evidence_v1 | 35.500 | 14.0m | 38.3m | 52.3m |
| 8 | iter8_nickname_depth_v1 | 50.572 | 11.7m | 37.7m | 49.4m |
| 9 | iter9_metafix_poolboost_v1 | 51.016 | 12.5m | 36.3m | 48.8m |
| 10 | iter9_metafix_poolboost_v1 | 50.280 | 8.9m | 40.3m | 49.3m |
| 11 | iter9_metafix_poolboost_v1 | 51.299 | 12.2m | 44.9m | 57.0m |
| 12 | iter12_citerevive_multiref_v1 | 53.255 | 14.8m | 57.6m | 72.4m |
| 13 | iter13_largegold_termnames_v1 | 51.562 | 9.4m | 64.1m | 73.5m |
| 14 | iter12_citerevive_multiref_v1 | 39.586 | 8.9m | 62.8m | 71.7m |
| 15 | iter15_authorsfix_goldceiling_v1 | 48.266 | 12.0m | 37.4m | 49.4m |
| 16 | iter9_metafix_poolboost_v1 | 47.356 | 10.8m | 48.4m | 59.1m |
| 17 | iter17_ambigunion_landmark_v1 | 46.900 | 8.1m | 38.6m | 46.7m |

## Best Agent: iter12_citerevive_multiref_v1
- Elo Score: 1559
- Mean Score: 45.505


## Elo Leadership Progression

Iteration-by-iteration Elo leaders:

| Iteration | Leader | Elo | Evolution Strategy |
|-----------|--------|-----|-------------------|
| 1 | seed_pkvc5eb0 | 1500 | Initial |
| 2 | iter2_router_deeplist_v1 | 1516 | use_your_judgment |
| 3 | iter3_recallforge_v1 | 1531 | use_your_judgment |
| 4 | iter4_citegraph_evidence_v1 | 1533 | use_your_judgment |
| 5 | iter3_recallforge_v1 | 1559 | use_your_judgment |
| 6 | iter6_evrepair_cite_v1 | 1535 | use_your_judgment |
| 7 | iter4_citegraph_evidence_v1 | 1562 | use_your_judgment |
| 8 | iter8_nickname_depth_v1 | 1534 | use_your_judgment |
| 9 | iter9_metafix_poolboost_v1 | 1534 | use_your_judgment |
| 10 | iter9_metafix_poolboost_v1 | 1563 | use_your_judgment |
| 11 | iter9_metafix_poolboost_v1 | 1589 | use_your_judgment |
| 12 | iter9_metafix_poolboost_v1 | 1583 | use_your_judgment |
| 13 | iter9_metafix_poolboost_v1 | 1583 | use_your_judgment |
| 14 | iter9_metafix_poolboost_v1 | 1583 | use_your_judgment |
| 15 | iter12_citerevive_multiref_v1 | 1561 | use_your_judgment |
| 16 | iter9_metafix_poolboost_v1 | 1576 | use_your_judgment |
| 17 | iter12_citerevive_multiref_v1 | 1559 | use_your_judgment |


## Deep Focus Evolution Timing Analysis
- **Iterations with evolution**: 16/17
- **Total evolution time**: 179.8 minutes
- **Average evolution time**: 11.2 minutes

### Time Breakdown by Evolution Phase

| Phase | Avg Time | % of Evolution |
|-------|----------|----------------|
| First Draft (Round 1) | 10.4m | 92.5% |
| Reflection | 0.8m | 7.5% |
| **Total** | **11.2m** | **100.0%** |

### Evolution vs Test vs Meta-Evolution Time

| Metric | Time | Percentage | Avg/Iter | Avg/Occurrence |
|--------|------|------------|----------|----------------|
| Total Evolution Time | 179.8m | 23.0% | 10.6m | 11.2m |
| Total Test Time | 600.5m | 77.0% | 35.3m | 35.3m |
| Total Meta-Evolution Time | 0.0m | 0.0% | 0.0m | 0.0m |
| **Total** | **780.2m** | **100.0%** | **45.9m** | **-** |

- **Evolution/Test ratio**: 0.30x
  - For every minute of testing, 0.30 minutes spent on evolution

## Claude CLI Usage & Cost Analysis
- **Total Cost**: $214.82
  - **Evaluation Cost**: $88.14
  - **Evolution CLI Cost**: $119.89
  - **Other (evaluator overhead) Cost**: $6.79
- **Total CLI Calls**: 32
- **Total CLI Input Tokens**: 1,160
- **Total CLI Output Tokens**: 707,208
- **Total Cache Created**: 2,212,341
- **Total Cache Read**: 49,346,455

### Cost Breakdown

| Operation | Cost | % of Total | Calls | Tokens In | Tokens Out |
|-----------|------|------------|-------|-----------|------------|
| Evaluation | $88.14 | 41.0% | - | - | - |
| Evolution (CLI) | $119.89 | 55.8% | 32 | 1,160 | 707,208 |
| Other (evaluator overhead) | $6.79 | 3.2% | - | - | - |
| **Total** | **$214.82** | **100%** | **32** | **1,160** | **707,208** |

**Answer: Evolution drives 55.8% of total costs**

### Detailed Per-Iteration Costs

| Iter | Total | **Eval** | **Evo** | Iter Eval | 1st Draft | Refl | Meta | Other | Fresh | Strategy | Meta-strategy |
|------|------|------|------|------|------|------|------|------|------|------|------|
| **1** | $0.01 | **$0.01** | **-** | $0.01 | - | - | - | $0.00 | 14 | - | - |
| **2** | $7.80 | **$0.78** | **$6.87** | $0.78 | $5.09 | $1.79 | - | $0.15 | 24 | use_your_judgment | - |
| **3** | $8.54 | **$1.21** | **$7.10** | $1.21 | $5.13 | $1.97 | - | $0.23 | 31 | use_your_judgment | - |
| **4** | $11.50 | **$3.21** | **$7.90** | $3.21 | $5.63 | $2.27 | - | $0.39 | 31 | use_your_judgment | - |
| **5** | $12.98 | **$5.54** | **$6.55** | $5.54 | $6.08 | $0.47 | - | $0.90 | 32 | use_your_judgment | - |
| **6** | $13.98 | **$4.78** | **$8.72** | $4.78 | $8.24 | $0.47 | - | $0.49 | 31 | use_your_judgment | - |
| **7** | $13.89 | **$4.30** | **$9.10** | $4.30 | $8.62 | $0.48 | - | $0.49 | 31 | use_your_judgment | - |
| **8** | $14.31 | **$6.08** | **$7.63** | $6.08 | $5.48 | $2.16 | - | $0.59 | 33 | use_your_judgment | - |
| **9** | $13.07 | **$4.17** | **$8.77** | $4.17 | $8.27 | $0.50 | - | $0.13 | 26 | use_your_judgment | - |
| **10** | $14.15 | **$6.97** | **$6.64** | $6.97 | $6.18 | $0.46 | - | $0.54 | 32 | use_your_judgment | - |
| **11** | $13.73 | **$5.93** | **$7.44** | $5.93 | $6.95 | $0.49 | - | $0.36 | 31 | use_your_judgment | - |
| **12** | $12.84 | **$5.35** | **$7.13** | $5.35 | $6.38 | $0.75 | - | $0.36 | 27 | use_your_judgment | - |
| **13** | $17.75 | **$10.23** | **$6.95** | $10.23 | $6.47 | $0.48 | - | $0.58 | 35 | use_your_judgment | - |
| **14** | $17.31 | **$9.91** | **$6.85** | $9.91 | $6.38 | $0.47 | - | $0.56 | 36 | use_your_judgment | - |
| **15** | $15.76 | **$6.74** | **$8.82** | $6.74 | $8.36 | $0.47 | - | $0.20 | 26 | use_your_judgment | - |
| **16** | $16.60 | **$8.19** | **$7.92** | $8.19 | $7.47 | $0.45 | - | $0.49 | 35 | use_your_judgment | - |
| **17** | $10.59 | **$4.76** | **$5.51** | $4.76 | $5.06 | $0.45 | - | $0.32 | 21 | use_your_judgment | - |
|------|------|------|------|------|------|------|------|------|------|------|------|
| **TOTAL** | **$214.82** | **$88.14** | **$119.89** | $88.14 | $105.78 | $14.12 | $0.00 | **$6.79** | **496** | - | - |

### Evolution Cost Breakdown (Avg per evolution)

| Phase | Avg Cost | % of Evolution | Avg Tokens In | Avg Tokens Out |
|-------|----------|----------------|---------------|----------------|
| First Draft (Round 1) | $6.61 | 88.2% | 69 | 41280 |
| Reflection | $0.88 | 11.8% | 4 | 2921 |
| **Total** | **$7.49** | **100%** | **72** | **44200** |

### Cache Performance

- **Cache Creation**: 2,212,341 tokens
- **Cache Hits**: 49,346,455 tokens (95.7% hit rate)

## Zero Accuracy Cases
No zero accuracy cases encountered ✅

## Exception Failures
No exception failures encountered ✅

## Evolution Health

No artifact validation failures ✅