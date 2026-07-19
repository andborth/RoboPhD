# Parallel Agent Research - Interim Report (Iteration 12)

## Experiment Configuration

**Run ID**: asta_paper_finder_20260717_170858
**Date**: 2026-07-18 01:22:34
**Iterations**: 12/999

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

#### **Use Your Judgment** (11 wins - 91.7% of all wins)
- **iter4_judge_sim_ranker** (3 wins, created iteration 4)
- **iter6_grade3_rescue** (3 wins, created iteration 6)
- **iter2_broad_recall_router** (2 wins, created iteration 2)
- **iter10_cite_expand** (1 win, created iteration 10)
- **iter11_tail_saturate** (1 win, created iteration 11)
- **iter12_body_conjunction** (1 win, created iteration 12)

#### **Initial** (1 wins - 8.3% of all wins)
- **seed_swslfmxl** (1 win, created iteration 0)

#### Summary Statistics
- **Total iterations with winners**: 12
- **Total wins counted**: 12 (includes ties)
- **Unique winning agents**: 7
- **Evolution strategies that produced winners**: 2


## Complete Performance Ranking Table

| Agent | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 | Iter 6 | Iter 7 | Iter 8 | Iter 9 | Iter 10 | Iter 11 | Iter 12 | Final Elo | Mean Rank |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|-----------|----------|
| iter12_body_conjunction | - | - | - | - | - | - | - | - | - | - | - | **#1** 45.830 | **1535** | **1.00** |
| iter11_tail_saturate | - | - | - | - | - | - | - | - | - | - | **#1** 44.202 | #2 44.044 | 1533 | **1.50** |
| iter10_cite_expand | - | - | - | - | - | - | - | - | - | **#1** 47.928 | #2 42.340 | - | 1532 | **1.50** |
| iter6_grade3_rescue | - | - | - | - | - | **#1** 48.013 | **#1** 30.033 | **#1** 32.220 | #3 33.062 | - | - | #3 43.280 | 1516 | **1.80** |
| iter5_cite_verify_deep_evidence | - | - | - | - | #2 46.363 | #2 44.900 | - | #2 30.327 | - | #2 46.455 | - | - | 1510 | 2.00 |
| iter9_verified_head | - | - | - | - | - | - | - | - | #2 34.632 | - | - | - | 1505 | 2.00 |
| iter7_simview_breadth | - | - | - | - | - | - | #2 29.495 | - | - | - | - | - | 1502 | 2.00 |
| iter4_judge_sim_ranker | - | - | - | **#1** 21.388 | **#1** 46.586 | #3 43.347 | - | - | **#1** 35.588 | #3 39.101 | #3 34.914 | - | 1491 | 2.00 |
| iter2_broad_recall_router | - | **#1** 40.643 | **#1** 35.210 | #2 20.926 | #3 37.350 | - | #3 18.056 | - | - | - | - | - | 1477 | 2.00 |
| iter8_resilient_lean | - | - | - | - | - | - | - | #3 27.476 | - | - | - | - | 1472 | 3.00 |
| iter3_criteria_evidence_max | - | - | #2 31.953 | #3 17.500 | - | - | - | - | - | - | - | - | 1471 | 2.50 |
| seed_swslfmxl | **#1** 2.291 | #2 6.651 | #3 4.507 | - | - | - | - | - | - | - | - | - | 1455 | 2.50 |

### Legend:
- **#1** = 1st place (winner of iteration)
- #2, #3, etc. = 2nd, 3rd place, etc.
- Score = Average score (0-1) on that iteration's problems
- **Bold Elo/Rank** = Top performer
- `-` = Agent not tested in that iteration



### Quick Summary

| Agent | Elo | Mean Score | Tests |
|-------|-----|------------|-------|
| iter12_body_conjunction | 1535 | 45.830 | 1 |
| iter11_tail_saturate | 1533 | 44.123 | 2 |
| iter10_cite_expand | 1532 | 45.134 | 2 |
| iter6_grade3_rescue | 1516 | 37.322 | 5 |
| iter5_cite_verify_deep_evidence | 1510 | 42.011 | 4 |
| iter9_verified_head | 1505 | 34.632 | 1 |
| iter7_simview_breadth | 1502 | 29.495 | 1 |
| iter4_judge_sim_ranker | 1491 | 36.821 | 6 |
| iter2_broad_recall_router | 1477 | 30.437 | 5 |
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

## Best Agent: iter12_body_conjunction
- Elo Score: 1535
- Mean Score: 45.830


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


## Deep Focus Evolution Timing Analysis
- **Iterations with evolution**: 11/12
- **Total evolution time**: 146.9 minutes
- **Average evolution time**: 13.4 minutes

### Time Breakdown by Evolution Phase

| Phase | Avg Time | % of Evolution |
|-------|----------|----------------|
| First Draft (Round 1) | 12.5m | 93.6% |
| Reflection | 0.9m | 6.4% |
| **Total** | **13.4m** | **100.0%** |

### Evolution vs Test vs Meta-Evolution Time

| Metric | Time | Percentage | Avg/Iter | Avg/Occurrence |
|--------|------|------------|----------|----------------|
| Total Evolution Time | 146.9m | 29.8% | 12.2m | 13.4m |
| Total Test Time | 346.7m | 70.2% | 28.9m | 28.9m |
| Total Meta-Evolution Time | 0.0m | 0.0% | 0.0m | 0.0m |
| **Total** | **493.6m** | **100.0%** | **41.1m** | **-** |

- **Evolution/Test ratio**: 0.42x
  - For every minute of testing, 0.42 minutes spent on evolution

## Claude CLI Usage & Cost Analysis
- **Total Cost**: $183.89
  - **Evaluation Cost**: $13.82
  - **Evolution CLI Cost**: $100.55
  - **Other (evaluator overhead) Cost**: $69.51
- **Total CLI Calls**: 22
- **Total CLI Input Tokens**: 2,300
- **Total CLI Output Tokens**: 653,555
- **Total Cache Created**: 1,595,470
- **Total Cache Read**: 35,923,164

### Cost Breakdown

| Operation | Cost | % of Total | Calls | Tokens In | Tokens Out |
|-----------|------|------------|-------|-----------|------------|
| Evaluation | $13.82 | 7.5% | - | - | - |
| Evolution (CLI) | $100.55 | 54.7% | 22 | 2,300 | 653,555 |
| Other (evaluator overhead) | $69.51 | 37.8% | - | - | - |
| **Total** | **$183.89** | **100%** | **22** | **2,300** | **653,555** |

**Answer: Evolution drives 54.7% of total costs**

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
|------|------|------|------|------|------|------|------|------|------|------|------|
| **TOTAL** | **$183.89** | **$13.82** | **$100.55** | $13.82 | $90.57 | $9.98 | $0.00 | **$69.51** | **354** | - | - |

### Evolution Cost Breakdown (Avg per evolution)

| Phase | Avg Cost | % of Evolution | Avg Tokens In | Avg Tokens Out |
|-------|----------|----------------|---------------|----------------|
| First Draft (Round 1) | $8.23 | 90.1% | 206 | 56492 |
| Reflection | $0.91 | 9.9% | 4 | 2922 |
| **Total** | **$9.14** | **100%** | **209** | **59414** |

### Cache Performance

- **Cache Creation**: 1,595,470 tokens
- **Cache Hits**: 35,923,164 tokens (95.7% hit rate)

## Zero Accuracy Cases
No zero accuracy cases encountered ✅

## Exception Failures
No exception failures encountered ✅

## Evolution Health

No artifact validation failures ✅