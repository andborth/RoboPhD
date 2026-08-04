# Parallel Agent Research - Final Report

## Experiment Configuration

**Run ID**: asta_paper_finder_20260802_103017
**Date**: 2026-08-03 04:00:24
**Iterations**: 999

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

- **Iteration 21**: evolution_strategy=greedy


## Winning Agents by Evolution Strategy

#### **Use Your Judgment** (21 wins - 95.5% of all wins)
- **iter9_metafix_poolboost_v1** (4 wins, created iteration 9)
- **iter12_citerevive_multiref_v1** (3 wins, created iteration 12)
- **iter18_cocite_largegold_v1** (3 wins, created iteration 18)
- **iter3_recallforge_v1** (2 wins, created iteration 3)
- **iter4_citegraph_evidence_v1** (2 wins, created iteration 4)
- **iter17_ambigunion_landmark_v1** (2 wins, created iteration 17)
- **iter2_router_deeplist_v1** (1 win, created iteration 2)
- **iter6_evrepair_cite_v1** (1 win, created iteration 6)
- **iter8_nickname_depth_v1** (1 win, created iteration 8)
- **iter13_largegold_termnames_v1** (1 win, created iteration 13)
- **iter15_authorsfix_goldceiling_v1** (1 win, created iteration 15)

#### **Initial** (1 wins - 4.5% of all wins)
- **seed_pkvc5eb0** (1 win, created iteration 0)

#### Summary Statistics
- **Total iterations with winners**: 22
- **Total wins counted**: 22 (includes ties)
- **Unique winning agents**: 12
- **Evolution strategies that produced winners**: 2


## Complete Performance Ranking Table

| Agent | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 | Iter 6 | Iter 7 | Iter 8 | Iter 9 | Iter 10 | Iter 11 | Iter 12 | Iter 13 | Iter 14 | Iter 15 | Iter 16 | Iter 17 | Iter 18 | Iter 19 | Iter 20 | Iter 21 | Iter 22 | Final Elo | Mean Rank |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|-----------|----------|
| iter18_cocite_largegold_v1 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 41.532 | #2 45.046 | **#1** 34.137 | #2 60.350 | **#1** 59.668 | **1594** | **1.40** |
| iter17_ambigunion_landmark_v1 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 46.900 | #2 40.142 | - | - | **#1** 63.057 | #2 58.497 | 1567 | **1.50** |
| iter9_metafix_poolboost_v1 | - | - | - | - | - | - | - | - | **#1** 51.016 | **#1** 50.280 | **#1** 51.299 | #2 52.762 | - | - | #3 45.570 | **#1** 47.356 | #3 35.587 | #3 36.795 | - | - | - | - | 1507 | **1.88** |
| iter20_referent_audit_v1 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | #2 33.493 | - | - | 1505 | 2.00 |
| iter15_authorsfix_goldceiling_v1 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 48.266 | #3 46.088 | - | - | - | - | - | - | 1505 | 2.00 |
| iter16_specplan_authorchan_v1 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | #2 46.095 | - | - | - | - | - | - | 1504 | 2.00 |
| iter14_overflow_minigrade_v1 | - | - | - | - | - | - | - | - | - | - | - | - | - | #2 33.038 | - | - | - | - | - | - | - | - | 1503 | 2.00 |
| iter10_refsolo_hedgefix_v1 | - | - | - | - | - | - | - | - | - | #2 47.850 | - | - | - | - | - | - | - | - | - | - | - | - | 1503 | 2.00 |
| iter6_evrepair_cite_v1 | - | - | - | - | - | **#1** 38.900 | #3 32.654 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1502 | 2.00 |
| iter8_nickname_depth_v1 | - | - | - | - | - | - | - | **#1** 50.572 | #2 48.217 | - | - | #3 51.878 | - | - | - | - | - | - | - | - | - | - | 1502 | 2.00 |
| iter13_largegold_termnames_v1 | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 51.562 | #3 32.278 | - | - | - | - | - | - | - | - | 1500 | 2.00 |
| iter4_citegraph_evidence_v1 | - | - | - | **#1** 40.998 | #2 35.679 | #2 33.590 | **#1** 35.500 | #3 39.823 | #3 38.331 | - | - | - | - | - | - | - | - | - | - | - | - | - | 1493 | 2.00 |
| iter3_recallforge_v1 | - | - | **#1** 33.918 | #2 35.017 | **#1** 40.223 | #3 33.495 | - | - | - | #3 43.158 | - | - | - | - | - | - | - | - | - | - | - | - | 1492 | 2.00 |
| iter12_citerevive_multiref_v1 | - | - | - | - | - | - | - | - | - | - | - | **#1** 53.255 | #2 51.078 | **#1** 39.586 | #2 46.157 | - | #2 37.452 | - | **#1** 46.989 | #3 32.617 | #3 58.639 | #3 52.250 | 1491 | 2.00 |
| iter2_router_deeplist_v1 | - | **#1** 32.602 | #2 33.726 | #3 29.860 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1483 | 2.00 |
| iter7_robustharvest_v1 | - | - | - | - | - | - | #2 34.242 | #2 49.657 | - | - | #3 50.356 | - | - | - | - | - | - | - | - | - | - | - | 1476 | 2.33 |
| iter11_secondwave_fullk_v1 | - | - | - | - | - | - | - | - | - | - | #2 51.167 | - | #3 49.409 | - | - | - | - | - | - | - | - | - | 1473 | 2.50 |
| iter19_metatime_winddown_v1 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | #3 44.677 | - | - | - | 1473 | 3.00 |
| iter5_robustcite_blend_v1 | - | - | - | - | #3 34.731 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1471 | 3.00 |
| seed_pkvc5eb0 | **#1** 2.100 | #2 3.370 | #3 3.331 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1455 | 2.50 |

### Legend:
- **#1** = 1st place (winner of iteration)
- #2, #3, etc. = 2nd, 3rd place, etc.
- Score = Average score (0-1) on that iteration's problems
- **Bold Elo/Rank** = Top performer
- `-` = Agent not tested in that iteration



### Quick Summary

| Agent | Elo | Mean Score | Mean Cost | Tests |
|-------|-----|------------|-----------|-------|
| iter18_cocite_largegold_v1 | 1594 | 48.147 | $0.303 | 5 |
| iter17_ambigunion_landmark_v1 | 1567 | 52.149 | $0.279 | 4 |
| iter9_metafix_poolboost_v1 | 1507 | 46.333 | $0.194 | 8 |
| iter20_referent_audit_v1 | 1505 | 33.493 | $0.326 | 1 |
| iter15_authorsfix_goldceiling_v1 | 1505 | 47.177 | $0.257 | 2 |
| iter16_specplan_authorchan_v1 | 1504 | 46.095 | $0.250 | 1 |
| iter14_overflow_minigrade_v1 | 1503 | 33.038 | $0.309 | 1 |
| iter10_refsolo_hedgefix_v1 | 1503 | 47.850 | $0.249 | 1 |
| iter6_evrepair_cite_v1 | 1502 | 35.777 | $0.159 | 2 |
| iter8_nickname_depth_v1 | 1502 | 50.222 | $0.188 | 3 |
| iter13_largegold_termnames_v1 | 1500 | 41.920 | $0.289 | 2 |
| iter4_citegraph_evidence_v1 | 1493 | 37.320 | $0.154 | 6 |
| iter3_recallforge_v1 | 1492 | 37.162 | $0.087 | 5 |
| iter12_citerevive_multiref_v1 | 1491 | 46.447 | $0.245 | 9 |
| iter2_router_deeplist_v1 | 1483 | 32.063 | $0.050 | 3 |
| iter7_robustharvest_v1 | 1476 | 44.752 | $0.158 | 3 |
| iter11_secondwave_fullk_v1 | 1473 | 50.288 | $0.251 | 2 |
| iter19_metatime_winddown_v1 | 1473 | 44.677 | $0.328 | 1 |
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
| 18 | iter18_cocite_largegold_v1 | 41.532 | 17.8m | 50.1m | 68.0m |
| 19 | iter12_citerevive_multiref_v1 | 46.989 | 9.0m | 52.8m | 61.8m |
| 20 | iter18_cocite_largegold_v1 | 34.137 | 10.9m | 49.1m | 59.9m |
| 21 | iter17_ambigunion_landmark_v1 | 63.057 | - | 28.2m | 28.2m |
| 22 | iter18_cocite_largegold_v1 | 59.668 | - | 51.9m | 51.9m |

## Best Agent: iter18_cocite_largegold_v1
- Elo Score: 1594
- Mean Score: 48.147


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
| 18 | iter12_citerevive_multiref_v1 | 1559 | use_your_judgment |
| 19 | iter12_citerevive_multiref_v1 | 1587 | use_your_judgment |
| 20 | iter18_cocite_largegold_v1 | 1567 | use_your_judgment |
| 21 | iter17_ambigunion_landmark_v1 | 1569 | use_your_judgment |
| 22 | iter18_cocite_largegold_v1 | 1594 | use_your_judgment |


## Deep Focus Evolution Timing Analysis
- **Iterations with evolution**: 19/22
- **Total evolution time**: 217.5 minutes
- **Average evolution time**: 11.4 minutes

### Time Breakdown by Evolution Phase

| Phase | Avg Time | % of Evolution |
|-------|----------|----------------|
| First Draft (Round 1) | 10.6m | 92.5% |
| Reflection | 0.9m | 7.5% |
| **Total** | **11.4m** | **100.0%** |

### Evolution vs Test vs Meta-Evolution Time

| Metric | Time | Percentage | Avg/Iter | Avg/Occurrence |
|--------|------|------------|----------|----------------|
| Total Evolution Time | 217.5m | 20.7% | 9.9m | 11.4m |
| Total Test Time | 832.6m | 79.3% | 37.8m | 37.8m |
| Total Meta-Evolution Time | 0.0m | 0.0% | 0.0m | 0.0m |
| **Total** | **1050.1m** | **100.0%** | **47.7m** | **-** |

- **Evolution/Test ratio**: 0.26x
  - For every minute of testing, 0.26 minutes spent on evolution

## Claude CLI Usage & Cost Analysis
- **Total Cost**: $275.38
  - **Evaluation Cost**: $124.51
  - **Evolution CLI Cost**: $142.71
  - **Other (evaluator overhead) Cost**: $8.16
- **Total CLI Calls**: 38
- **Total CLI Input Tokens**: 1,359
- **Total CLI Output Tokens**: 834,886
- **Total Cache Created**: 2,550,321
- **Total Cache Read**: 59,010,534

### Cost Breakdown

| Operation | Cost | % of Total | Calls | Tokens In | Tokens Out |
|-----------|------|------------|-------|-----------|------------|
| Evaluation | $124.51 | 45.2% | - | - | - |
| Evolution (CLI) | $142.71 | 51.8% | 38 | 1,359 | 834,886 |
| Other (evaluator overhead) | $8.16 | 3.0% | - | - | - |
| **Total** | **$275.38** | **100%** | **38** | **1,359** | **834,886** |

**Answer: Evolution drives 51.8% of total costs**

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
| **18** | $19.52 | **$9.52** | **$9.57** | $9.52 | $9.01 | $0.55 | - | $0.44 | 26 | use_your_judgment | - |
| **19** | $16.88 | **$10.03** | **$6.50** | $10.03 | $5.96 | $0.55 | - | $0.34 | 31 | use_your_judgment | - |
| **20** | $15.31 | **$8.26** | **$6.75** | $8.26 | $6.28 | $0.46 | - | $0.30 | 27 | use_your_judgment | - |
| **21** | $3.26 | **$3.19** | **-** | $3.19 | - | - | - | $0.06 | 18 | - | - |
| **22** | $5.59 | **$5.37** | **-** | $5.37 | - | - | - | $0.22 | 15 | - | - |
|------|------|------|------|------|------|------|------|------|------|------|------|
| **TOTAL** | **$275.38** | **$124.51** | **$142.71** | $124.51 | $127.03 | $15.68 | $0.00 | **$8.16** | **613** | - | - |

### Evolution Cost Breakdown (Avg per evolution)

| Phase | Avg Cost | % of Evolution | Avg Tokens In | Avg Tokens Out |
|-------|----------|----------------|---------------|----------------|
| First Draft (Round 1) | $6.69 | 89.0% | 68 | 40964 |
| Reflection | $0.83 | 11.0% | 4 | 2977 |
| **Total** | **$7.51** | **100%** | **72** | **43941** |

### Cache Performance

- **Cache Creation**: 2,550,321 tokens
- **Cache Hits**: 59,010,534 tokens (95.9% hit rate)

## Zero Accuracy Cases
No zero accuracy cases encountered ✅

## Exception Failures
No exception failures encountered ✅

## Evolution Health

No artifact validation failures ✅