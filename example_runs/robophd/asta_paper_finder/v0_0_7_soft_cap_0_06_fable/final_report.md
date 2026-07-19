# Parallel Agent Research - Final Report

## Experiment Configuration

**Run ID**: asta_paper_finder_20260717_170858
**Date**: 2026-07-18 10:33:45
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

- file_mapping: {'agent.py': 'agent.py'}
- paper_finder_runtime: {'cost_threshold': 0.06, 'cost_per_error': 0.02, 'cap_judge_to_estimate': True}

### User Configuration (Iteration 1)

The following parameters were explicitly set, overriding defaults:

- agents_directory: ../robophd_runs/robophd/_optimize_anything_seeds
- domain: external
- eval_timeout: 1800
- evaluation_budget: 600
- evolution_model: fable-5 (claude-fable-5)
- evolution_strategy: use_your_judgment
- evolution_timeout: 3600
- examples_per_iteration: 14
- initial_agents: ['seed_swslfmxl']
- max_workers: 8
- meta_evolution_cadence: 3
- meta_evolution_domain: asta_paper_finder
- meta_evolution_first_iteration: 4
- meta_evolution_model: opus-4.8 (claude-opus-4-8)
- new_agent_test_rounds: 0

### Configuration Changes by Iteration

(No configuration changes across iterations)


## Winning Agents by Evolution Strategy

#### **Use Your Judgment** (20 wins - 95.2% of all wins)
- **iter4_judge_sim_ranker** (3 wins, created iteration 4)
- **iter6_grade3_rescue** (3 wins, created iteration 6)
- **iter12_body_conjunction** (3 wins, created iteration 12)
- **iter13_grade_view** (3 wins, created iteration 13)
- **iter2_broad_recall_router** (2 wins, created iteration 2)
- **iter18_triage_first** (2 wins, created iteration 18)
- **iter10_cite_expand** (1 win, created iteration 10)
- **iter11_tail_saturate** (1 win, created iteration 11)
- **iter14_crit_coverage** (1 win, created iteration 14)
- **iter20_cite_proof** (1 win, created iteration 20)

#### **Initial** (1 wins - 4.8% of all wins)
- **seed_swslfmxl** (1 win, created iteration 0)

#### Summary Statistics
- **Total iterations with winners**: 21
- **Total wins counted**: 21 (includes ties)
- **Unique winning agents**: 11
- **Evolution strategies that produced winners**: 2


## Complete Performance Ranking Table

| Agent | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 | Iter 6 | Iter 7 | Iter 8 | Iter 9 | Iter 10 | Iter 11 | Iter 12 | Iter 13 | Iter 14 | Iter 15 | Iter 16 | Iter 17 | Iter 18 | Iter 19 | Iter 20 | Iter 21 | Final Elo | Mean Rank |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|-----------|----------|
| iter12_body_conjunction | - | - | - | - | - | - | - | - | - | - | - | **#1** 45.830 | #2 39.919 | - | **#1** 33.947 | #2 41.598 | #2 39.131 | #2 33.407 | #2 29.120 | - | **#1** 39.660 | **1581** | **1.62** |
| iter13_grade_view | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 41.760 | #2 38.120 | - | **#1** 41.737 | **#1** 41.426 | #3 32.260 | - | #2 30.276 | - | 1553 | **1.67** |
| iter20_cite_proof | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 32.787 | #2 37.976 | 1536 | **1.50** |
| iter18_triage_first | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 34.035 | **#1** 29.937 | #3 25.708 | - | 1533 | **1.67** |
| iter14_crit_coverage | - | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 38.931 | #2 31.340 | - | - | - | - | - | - | 1533 | **1.50** |
| iter6_grade3_rescue | - | - | - | - | - | **#1** 48.013 | **#1** 30.033 | **#1** 32.220 | #3 33.062 | - | - | #3 43.280 | - | - | - | - | - | - | - | - | - | 1516 | **1.80** |
| iter5_cite_verify_deep_evidence | - | - | - | - | #2 46.363 | #2 44.900 | - | #2 30.327 | - | #2 46.455 | - | - | - | - | - | - | - | - | - | - | - | 1510 | 2.00 |
| iter9_verified_head | - | - | - | - | - | - | - | - | #2 34.632 | - | - | - | - | - | - | - | - | - | - | - | - | 1505 | 2.00 |
| iter7_simview_breadth | - | - | - | - | - | - | #2 29.495 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1502 | 2.00 |
| iter11_tail_saturate | - | - | - | - | - | - | - | - | - | - | **#1** 44.202 | #2 44.044 | #3 39.030 | - | - | - | - | - | - | - | - | 1500 | 2.00 |
| iter10_cite_expand | - | - | - | - | - | - | - | - | - | **#1** 47.928 | #2 42.340 | - | - | #3 36.084 | - | - | - | - | - | - | - | 1500 | 2.00 |
| iter4_judge_sim_ranker | - | - | - | **#1** 21.388 | **#1** 46.586 | #3 43.347 | - | - | **#1** 35.588 | #3 39.101 | #3 34.914 | - | - | - | - | - | - | - | - | - | - | 1491 | 2.00 |
| iter2_broad_recall_router | - | **#1** 40.643 | **#1** 35.210 | #2 20.926 | #3 37.350 | - | #3 18.056 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1477 | 2.00 |
| iter17_recall_stack | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | #3 37.052 | - | - | - | - | 1474 | 3.00 |
| iter16_pool_breadth | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | #3 37.044 | - | - | - | - | - | 1473 | 3.00 |
| iter19_breadth_restored | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | #3 26.603 | - | - | 1473 | 3.00 |
| iter21_reverse_cite | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | #3 32.643 | 1473 | 3.00 |
| iter15_cite_inverse | - | - | - | - | - | - | - | - | - | - | - | - | - | - | #3 31.117 | - | - | - | - | - | - | 1472 | 3.00 |
| iter8_resilient_lean | - | - | - | - | - | - | - | #3 27.476 | - | - | - | - | - | - | - | - | - | - | - | - | - | 1472 | 3.00 |
| iter3_criteria_evidence_max | - | - | #2 31.953 | #3 17.500 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1471 | 2.50 |
| seed_swslfmxl | **#1** 2.291 | #2 6.651 | #3 4.507 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1455 | 2.50 |

### Legend:
- **#1** = 1st place (winner of iteration)
- #2, #3, etc. = 2nd, 3rd place, etc.
- Score = Average score (0-1) on that iteration's problems
- **Bold Elo/Rank** = Top performer
- `-` = Agent not tested in that iteration



### Quick Summary

| Agent | Elo | Mean Score | Tests |
|-------|-----|------------|-------|
| iter12_body_conjunction | 1581 | 37.827 | 8 |
| iter13_grade_view | 1553 | 37.597 | 6 |
| iter20_cite_proof | 1536 | 35.382 | 2 |
| iter18_triage_first | 1533 | 29.893 | 3 |
| iter14_crit_coverage | 1533 | 35.136 | 2 |
| iter6_grade3_rescue | 1516 | 37.322 | 5 |
| iter5_cite_verify_deep_evidence | 1510 | 42.011 | 4 |
| iter9_verified_head | 1505 | 34.632 | 1 |
| iter7_simview_breadth | 1502 | 29.495 | 1 |
| iter11_tail_saturate | 1500 | 42.426 | 3 |
| iter10_cite_expand | 1500 | 42.117 | 3 |
| iter4_judge_sim_ranker | 1491 | 36.821 | 6 |
| iter2_broad_recall_router | 1477 | 30.437 | 5 |
| iter17_recall_stack | 1474 | 37.052 | 1 |
| iter16_pool_breadth | 1473 | 37.044 | 1 |
| iter19_breadth_restored | 1473 | 26.603 | 1 |
| iter21_reverse_cite | 1473 | 32.643 | 1 |
| iter15_cite_inverse | 1472 | 31.117 | 1 |
| iter8_resilient_lean | 1472 | 27.476 | 1 |
| iter3_criteria_evidence_max | 1471 | 24.727 | 2 |
| seed_swslfmxl | 1455 | 4.483 | 3 |

### Iteration Progression

| Iter | Winner(s) | Score | Evo Time | Test Time | Total Time |
|------|-----------|-------|----------|-----------|------------|
| 1 | seed_swslfmxl | 2.291 | - | 0.7m | 0.7m |
| 2 | iter2_broad_recall_router | 40.643 | 8.7m | 1.4m | 10.1m |
| 3 | iter2_broad_recall_router | 35.210 | 11.2m | 9.5m | 20.8m |
| 4 | iter4_judge_sim_ranker | 21.388 | 10.9m | 11.0m | 21.9m |
| 5 | iter4_judge_sim_ranker | 46.586 | 16.9m | 25.4m | 42.3m |
| 6 | iter6_grade3_rescue | 48.013 | 12.2m | 38.0m | 50.2m |
| 7 | iter6_grade3_rescue | 30.033 | 16.7m | 37.6m | 54.3m |
| 8 | iter6_grade3_rescue | 32.220 | 13.4m | 81.1m | 94.5m |
| 9 | iter4_judge_sim_ranker | 35.588 | 14.2m | 52.4m | 66.6m |
| 10 | iter10_cite_expand | 47.928 | 12.1m | 25.9m | 38.0m |
| 11 | iter11_tail_saturate | 44.202 | 16.5m | 27.4m | 43.8m |
| 12 | iter12_body_conjunction | 45.830 | 14.0m | 36.4m | 50.4m |
| 13 | iter13_grade_view | 41.760 | 13.9m | 41.7m | 55.5m |
| 14 | iter14_crit_coverage | 38.931 | 14.5m | 55.4m | 70.0m |
| 15 | iter12_body_conjunction | 33.947 | 12.0m | 49.7m | 61.7m |
| 16 | iter13_grade_view | 41.737 | 21.7m | 55.6m | 77.3m |
| 17 | iter13_grade_view | 41.426 | 16.4m | 42.8m | 59.2m |
| 18 | iter18_triage_first | 34.035 | 15.9m | 45.0m | 60.8m |
| 19 | iter18_triage_first | 29.937 | 11.5m | 48.8m | 60.3m |
| 20 | iter20_cite_proof | 32.787 | 18.4m | 44.9m | 63.3m |
| 21 | iter12_body_conjunction | 39.660 | 15.0m | 28.1m | 43.1m |

## Best Agent: iter12_body_conjunction
- Elo Score: 1581
- Mean Score: 37.827


## Elo Leadership Progression

Iteration-by-iteration Elo leaders:

| Iteration | Leader | Elo | Evolution Strategy |
|-----------|--------|-----|-------------------|
| 1 | seed_swslfmxl | 1500 | Initial |
| 2 | iter2_broad_recall_router | 1516 | use_your_judgment |
| 3 | iter2_broad_recall_router | 1545 | use_your_judgment |
| 4 | iter2_broad_recall_router | 1541 | use_your_judgment |
| 5 | iter4_judge_sim_ranker | 1563 | use_your_judgment |
| 6 | iter6_grade3_rescue | 1534 | use_your_judgment |
| 7 | iter6_grade3_rescue | 1563 | use_your_judgment |
| 8 | iter6_grade3_rescue | 1589 | use_your_judgment |
| 9 | iter4_judge_sim_ranker | 1560 | use_your_judgment |
| 10 | iter6_grade3_rescue | 1551 | use_your_judgment |
| 11 | iter6_grade3_rescue | 1551 | use_your_judgment |
| 12 | iter12_body_conjunction | 1535 | use_your_judgment |
| 13 | iter13_grade_view | 1534 | use_your_judgment |
| 14 | iter14_crit_coverage | 1534 | use_your_judgment |
| 15 | iter12_body_conjunction | 1563 | use_your_judgment |
| 16 | iter13_grade_view | 1564 | use_your_judgment |
| 17 | iter13_grade_view | 1592 | use_your_judgment |
| 18 | iter12_body_conjunction | 1556 | use_your_judgment |
| 19 | iter18_triage_first | 1568 | use_your_judgment |
| 20 | iter13_grade_view | 1553 | use_your_judgment |
| 21 | iter12_body_conjunction | 1581 | use_your_judgment |


## Deep Focus Evolution Timing Analysis
- **Iterations with evolution**: 20/21
- **Total evolution time**: 286.2 minutes
- **Average evolution time**: 14.3 minutes

### Time Breakdown by Evolution Phase

| Phase | Avg Time | % of Evolution |
|-------|----------|----------------|
| First Draft (Round 1) | 13.4m | 93.9% |
| Reflection | 0.9m | 6.1% |
| **Total** | **14.3m** | **100.0%** |

### Evolution vs Test vs Meta-Evolution Time

| Metric | Time | Percentage | Avg/Iter | Avg/Occurrence |
|--------|------|------------|----------|----------------|
| Total Evolution Time | 286.2m | 27.4% | 13.6m | 14.3m |
| Total Test Time | 758.6m | 72.6% | 36.1m | 36.1m |
| Total Meta-Evolution Time | 0.0m | 0.0% | 0.0m | 0.0m |
| **Total** | **1044.8m** | **100.0%** | **49.8m** | **-** |

- **Evolution/Test ratio**: 0.38x
  - For every minute of testing, 0.38 minutes spent on evolution

## Claude CLI Usage & Cost Analysis
- **Total Cost**: $368.33
  - **Evaluation Cost**: $26.55
  - **Evolution CLI Cost**: $204.10
  - **Other (evaluator overhead) Cost**: $137.68
- **Total CLI Calls**: 40
- **Total CLI Input Tokens**: 5,016
- **Total CLI Output Tokens**: 1,220,875
- **Total Cache Created**: 3,759,110
- **Total Cache Read**: 92,821,928

### Cost Breakdown

| Operation | Cost | % of Total | Calls | Tokens In | Tokens Out |
|-----------|------|------------|-------|-----------|------------|
| Evaluation | $26.55 | 7.2% | - | - | - |
| Evolution (CLI) | $204.10 | 55.4% | 40 | 5,016 | 1,220,875 |
| Other (evaluator overhead) | $137.68 | 37.4% | - | - | - |
| **Total** | **$368.33** | **100%** | **40** | **5,016** | **1,220,875** |

**Answer: Evolution drives 55.4% of total costs**

### Detailed Per-Iteration Costs

| Iter | Total | **Eval** | **Evo** | Iter Eval | 1st Draft | Refl | Meta | Other | Fresh | Strategy | Meta-strategy |
|------|------|------|------|------|------|------|------|------|------|------|------|
| **1** | $0.13 | **$0.01** | **-** | $0.01 | - | - | - | $0.12 | 14 | - | - |
| **2** | $6.39 | **$0.21** | **$4.15** | $0.21 | $3.79 | $0.36 | - | $2.03 | 27 | use_your_judgment | - |
| **3** | $15.31 | **$0.56** | **$7.90** | $0.56 | $5.72 | $2.19 | - | $6.85 | 38 | use_your_judgment | - |
| **4** | $11.95 | **$0.76** | **$7.31** | $0.76 | $6.85 | $0.46 | - | $3.88 | 34 | use_your_judgment | - |
| **5** | $19.87 | **$1.59** | **$9.67** | $1.59 | $9.09 | $0.57 | - | $8.62 | 36 | use_your_judgment | - |
| **6** | $20.78 | **$1.66** | **$7.23** | $1.66 | $6.73 | $0.50 | - | $11.88 | 31 | use_your_judgment | - |
| **7** | $17.90 | **$1.48** | **$12.06** | $1.48 | $11.50 | $0.56 | - | $4.36 | 31 | use_your_judgment | - |
| **8** | $16.50 | **$1.57** | **$10.15** | $1.57 | $9.59 | $0.56 | - | $4.78 | 32 | use_your_judgment | - |
| **9** | $18.67 | **$1.55** | **$8.87** | $1.55 | $8.33 | $0.53 | - | $8.25 | 29 | use_your_judgment | - |
| **10** | $15.38 | **$1.33** | **$8.86** | $1.33 | $8.32 | $0.54 | - | $5.19 | 23 | use_your_judgment | - |
| **11** | $24.84 | **$1.49** | **$14.41** | $1.49 | $11.29 | $3.12 | - | $8.94 | 28 | use_your_judgment | - |
| **12** | $16.16 | **$1.61** | **$9.94** | $1.61 | $9.36 | $0.58 | - | $4.61 | 31 | use_your_judgment | - |
| **13** | $24.25 | **$1.78** | **$10.73** | $1.78 | $7.72 | $3.01 | - | $11.75 | 30 | use_your_judgment | - |
| **14** | $20.46 | **$2.20** | **$9.05** | $2.20 | $6.50 | $2.55 | - | $9.20 | 34 | use_your_judgment | - |
| **15** | $18.98 | **$1.69** | **$10.08** | $1.69 | $9.58 | $0.49 | - | $7.21 | 35 | use_your_judgment | - |
| **16** | $22.49 | **$1.55** | **$16.51** | $1.55 | $12.70 | $3.80 | - | $4.43 | 30 | use_your_judgment | - |
| **17** | $21.76 | **$1.26** | **$13.58** | $1.26 | $13.00 | $0.59 | - | $6.92 | 26 | use_your_judgment | - |
| **18** | $22.50 | **$1.16** | **$11.00** | $1.16 | $8.21 | $2.80 | - | $10.34 | 21 | use_your_judgment | - |
| **19** | $17.60 | **$1.23** | **$8.68** | $1.23 | $8.20 | $0.48 | - | $7.69 | 31 | use_your_judgment | - |
| **20** | $21.04 | **$1.10** | **$13.34** | $1.10 | $12.72 | $0.61 | - | $6.59 | 29 | use_your_judgment | - |
| **21** | $15.36 | **$0.74** | **$10.58** | $0.74 | $10.05 | $0.53 | - | $4.04 | 25 | use_your_judgment | - |
|------|------|------|------|------|------|------|------|------|------|------|------|
| **TOTAL** | **$368.33** | **$26.55** | **$204.10** | $26.55 | $179.25 | $24.85 | $0.00 | **$137.68** | **615** | - | - |

### Evolution Cost Breakdown (Avg per evolution)

| Phase | Avg Cost | % of Evolution | Avg Tokens In | Avg Tokens Out |
|-------|----------|----------------|---------------|----------------|
| First Draft (Round 1) | $8.96 | 87.8% | 247 | 58062 |
| Reflection | $1.24 | 12.2% | 4 | 2981 |
| **Total** | **$10.21** | **100%** | **251** | **61044** |

### Cache Performance

- **Cache Creation**: 3,759,110 tokens
- **Cache Hits**: 92,821,928 tokens (96.1% hit rate)

## Zero Accuracy Cases
No zero accuracy cases encountered ✅

## Exception Failures
No exception failures encountered ✅

## Evolution Health

No artifact validation failures ✅