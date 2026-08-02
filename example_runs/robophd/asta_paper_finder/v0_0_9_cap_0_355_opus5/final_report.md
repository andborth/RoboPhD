# Parallel Agent Research - Final Report

## Experiment Configuration

**Run ID**: asta_paper_finder_20260731_023655
**Date**: 2026-08-01 10:52:03
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
- evolution_model: opus-5 (claude-opus-5)
- evolution_strategy: use_your_judgment
- evolution_timeout: 3600
- examples_per_iteration: 14
- initial_agents: ['seed_xst5jrr0']
- max_workers: 8
- meta_evolution_cadence: 3
- meta_evolution_domain: asta_paper_finder
- meta_evolution_first_iteration: 4
- meta_evolution_model: opus-5 (claude-opus-5)
- new_agent_test_rounds: 0

### Configuration Changes by Iteration

- **Iteration 22**: evolution_strategy=greedy


## Winning Agents by Evolution Strategy

#### **Use Your Judgment** (22 wins - 95.7% of all wins)
- **iter9_bulk_passage_harvest** (5 wins, created iteration 9)
- **iter8_criterion_window_evidence** (4 wins, created iteration 8)
- **iter21_gold_rubric_and_hard_predicates** (3 wins, created iteration 21)
- **iter3_criterion_evidence_miner** (2 wins, created iteration 3)
- **iter4_facet_conjunction_recall** (2 wins, created iteration 4)
- **iter7_solo_mining_coverage** (2 wins, created iteration 7)
- **iter19_deep_screen_bulk_evidence** (2 wins, created iteration 19)
- **iter2_deep_recall_router** (1 win, created iteration 2)
- **iter15_citation_neighborhood_recall** (1 win, created iteration 15)

#### **Initial** (1 wins - 4.3% of all wins)
- **seed_xst5jrr0** (1 win, created iteration 0)

#### Summary Statistics
- **Total iterations with winners**: 23
- **Total wins counted**: 23 (includes ties)
- **Unique winning agents**: 10
- **Evolution strategies that produced winners**: 2


## Complete Performance Ranking Table

| Agent | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 | Iter 6 | Iter 7 | Iter 8 | Iter 9 | Iter 10 | Iter 11 | Iter 12 | Iter 13 | Iter 14 | Iter 15 | Iter 16 | Iter 17 | Iter 18 | Iter 19 | Iter 20 | Iter 21 | Iter 22 | Iter 23 | Final Elo | Mean Rank |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|-----------|----------|
| iter21_gold_rubric_and_hard_pred... | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 58.092 | **#1** 43.463 | **#1** 46.730 | **1602** | **1.00** |
| iter8_criterion_window_evidence | - | - | - | - | - | - | - | **#1** 45.937 | #2 44.089 | - | **#1** 46.722 | #3 37.558 | - | **#1** 44.214 | #2 43.225 | - | - | **#1** 55.838 | #2 43.200 | - | #2 55.207 | #2 39.824 | #3 42.542 | 1542 | **1.82** |
| iter9_bulk_passage_harvest | - | - | - | - | - | - | - | - | **#1** 46.177 | **#1** 55.011 | #3 43.755 | - | **#1** 43.095 | #2 44.046 | - | **#1** 51.648 | **#1** 61.279 | #3 51.857 | - | #2 36.780 | - | #3 39.692 | #2 43.692 | 1538 | **1.82** |
| iter19_deep_screen_bulk_evidence | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 45.214 | **#1** 39.426 | #3 54.383 | - | - | 1534 | **1.67** |
| iter7_solo_mining_coverage | - | - | - | - | - | - | **#1** 41.712 | #2 42.981 | - | #2 53.601 | - | **#1** 40.183 | #2 41.461 | - | #3 37.979 | - | - | - | - | - | - | - | - | 1526 | **1.83** |
| iter18_gold_criteria_venue_anchor | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | #2 52.783 | - | - | - | - | - | 1507 | 2.00 |
| iter17_full_pool_screen | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | #2 59.460 | - | - | - | - | - | - | 1505 | 2.00 |
| iter15_citation_neighborhood_recall | - | - | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 46.618 | #2 48.495 | #3 59.126 | - | - | - | - | - | - | 1505 | 2.00 |
| iter11_tldr_abstract_density | - | - | - | - | - | - | - | - | - | - | #2 45.209 | - | - | - | - | - | - | - | - | - | - | - | - | 1504 | 2.00 |
| iter12_atomic_criteria_coverage | - | - | - | - | - | - | - | - | - | - | - | #2 38.997 | - | - | - | - | - | - | - | - | - | - | - | 1504 | 2.00 |
| iter4_facet_conjunction_recall | - | - | - | #2 41.189 | **#1** 44.059 | **#1** 49.745 | #2 39.661 | - | #3 40.942 | - | - | - | - | - | - | - | - | - | #3 34.700 | - | - | - | - | 1498 | 2.00 |
| iter3_criterion_evidence_miner | - | - | **#1** 41.141 | **#1** 44.406 | #2 37.533 | #2 49.255 | #3 34.622 | #3 32.628 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1488 | 2.00 |
| iter2_deep_recall_router | - | **#1** 47.179 | #2 41.004 | #3 33.916 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1483 | 2.00 |
| iter20_prf_coverage_expansion | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | #3 34.746 | - | - | - | 1474 | 3.00 |
| iter13_gold_shaped_criteria_repair | - | - | - | - | - | - | - | - | - | - | - | - | #3 41.172 | - | - | - | - | - | - | - | - | - | - | 1473 | 3.00 |
| iter16_live_citation_graph | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | #3 48.065 | - | - | - | - | - | - | - | 1473 | 3.00 |
| iter6_conjunction_evidence_fusion | - | - | - | - | - | #3 44.416 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1473 | 3.00 |
| iter14_metadata_predicate_exactness | - | - | - | - | - | - | - | - | - | - | - | - | - | #3 43.467 | - | - | - | - | - | - | - | - | - | 1473 | 3.00 |
| iter10_criterion_gap_closure | - | - | - | - | - | - | - | - | - | #3 53.552 | - | - | - | - | - | - | - | - | - | - | - | - | - | 1472 | 3.00 |
| iter5_depth_evidence_miner | - | - | - | - | #3 36.687 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1472 | 3.00 |
| seed_xst5jrr0 | **#1** 3.194 | #2 4.527 | #3 6.385 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1455 | 2.50 |

### Legend:
- **#1** = 1st place (winner of iteration)
- #2, #3, etc. = 2nd, 3rd place, etc.
- Score = Average score (0-1) on that iteration's problems
- **Bold Elo/Rank** = Top performer
- `-` = Agent not tested in that iteration



### Quick Summary

| Agent | Elo | Mean Score | Mean Cost | Tests |
|-------|-----|------------|-----------|-------|
| iter21_gold_rubric_and_hard_predicates | 1602 | 49.428 | $0.270 | 3 |
| iter8_criterion_window_evidence | 1542 | 45.305 | $0.178 | 11 |
| iter9_bulk_passage_harvest | 1538 | 47.003 | $0.210 | 11 |
| iter19_deep_screen_bulk_evidence | 1534 | 46.341 | $0.247 | 3 |
| iter7_solo_mining_coverage | 1526 | 42.986 | $0.164 | 6 |
| iter18_gold_criteria_venue_anchor | 1507 | 52.783 | $0.230 | 1 |
| iter17_full_pool_screen | 1505 | 59.460 | $0.259 | 1 |
| iter15_citation_neighborhood_recall | 1505 | 51.413 | $0.230 | 3 |
| iter11_tldr_abstract_density | 1504 | 45.209 | $0.168 | 1 |
| iter12_atomic_criteria_coverage | 1504 | 38.997 | $0.164 | 1 |
| iter4_facet_conjunction_recall | 1498 | 41.716 | $0.127 | 6 |
| iter3_criterion_evidence_miner | 1488 | 39.931 | $0.117 | 6 |
| iter2_deep_recall_router | 1483 | 40.700 | $0.063 | 3 |
| iter20_prf_coverage_expansion | 1474 | 34.746 | $0.292 | 1 |
| iter13_gold_shaped_criteria_repair | 1473 | 41.172 | $0.162 | 1 |
| iter16_live_citation_graph | 1473 | 48.065 | $0.198 | 1 |
| iter6_conjunction_evidence_fusion | 1473 | 44.416 | $0.129 | 1 |
| iter14_metadata_predicate_exactness | 1473 | 43.467 | $0.177 | 1 |
| iter10_criterion_gap_closure | 1472 | 53.552 | $0.221 | 1 |
| iter5_depth_evidence_miner | 1472 | 36.687 | $0.170 | 1 |
| seed_xst5jrr0 | 1455 | 4.702 | $0.001 | 3 |

### Iteration Progression

| Iter | Winner(s) | Score | Evo Time | Test Time | Total Time |
|------|-----------|-------|----------|-----------|------------|
| 1 | seed_xst5jrr0 | 3.194 | - | 2.7m | 2.7m |
| 2 | iter2_deep_recall_router | 47.179 | 12.9m | 4.7m | 17.6m |
| 3 | iter3_criterion_evidence_miner | 41.141 | 25.6m | 21.9m | 47.5m |
| 4 | iter3_criterion_evidence_miner | 44.406 | 32.2m | 46.5m | 78.7m |
| 5 | iter4_facet_conjunction_recall | 44.059 | 13.4m | 53.9m | 67.2m |
| 6 | iter4_facet_conjunction_recall | 49.745 | 11.4m | 37.1m | 48.5m |
| 7 | iter7_solo_mining_coverage | 41.712 | 8.3m | 57.3m | 65.6m |
| 8 | iter8_criterion_window_evidence | 45.937 | 11.4m | 74.5m | 85.9m |
| 9 | iter9_bulk_passage_harvest | 46.177 | 12.9m | 56.2m | 69.1m |
| 10 | iter9_bulk_passage_harvest | 55.011 | 8.5m | 80.6m | 89.1m |
| 11 | iter8_criterion_window_evidence | 46.722 | 9.6m | 47.5m | 57.1m |
| 12 | iter7_solo_mining_coverage | 40.183 | 10.8m | 67.1m | 77.9m |
| 13 | iter9_bulk_passage_harvest | 43.095 | 8.8m | 75.1m | 83.9m |
| 14 | iter8_criterion_window_evidence | 44.214 | 9.3m | 56.6m | 65.9m |
| 15 | iter15_citation_neighborhood_recall | 46.618 | 11.9m | 69.8m | 81.7m |
| 16 | iter9_bulk_passage_harvest | 51.648 | 8.6m | 48.6m | 57.2m |
| 17 | iter9_bulk_passage_harvest | 61.279 | 10.9m | 60.2m | 71.1m |
| 18 | iter8_criterion_window_evidence | 55.838 | 8.6m | 59.4m | 68.0m |
| 19 | iter19_deep_screen_bulk_evidence | 45.214 | 9.3m | 41.8m | 51.1m |
| 20 | iter19_deep_screen_bulk_evidence | 39.426 | 7.7m | 71.7m | 79.4m |
| 21 | iter21_gold_rubric_and_hard_predicates | 58.092 | 11.6m | 61.6m | 73.2m |
| 22 | iter21_gold_rubric_and_hard_predicates | 43.463 | - | 38.5m | 38.5m |
| 23 | iter21_gold_rubric_and_hard_predicates | 46.730 | - | 27.5m | 27.5m |

## Best Agent: iter21_gold_rubric_and_hard_predicates
- Elo Score: 1602
- Mean Score: 49.428


## Elo Leadership Progression

Iteration-by-iteration Elo leaders:

| Iteration | Leader | Elo | Evolution Strategy |
|-----------|--------|-----|-------------------|
| 1 | seed_xst5jrr0 | 1500 | Initial |
| 2 | iter2_deep_recall_router | 1516 | use_your_judgment |
| 3 | iter3_criterion_evidence_miner | 1531 | use_your_judgment |
| 4 | iter3_criterion_evidence_miner | 1560 | use_your_judgment |
| 5 | iter3_criterion_evidence_miner | 1555 | use_your_judgment |
| 6 | iter4_facet_conjunction_recall | 1566 | use_your_judgment |
| 7 | iter4_facet_conjunction_recall | 1563 | use_your_judgment |
| 8 | iter4_facet_conjunction_recall | 1563 | use_your_judgment |
| 9 | iter9_bulk_passage_harvest | 1536 | use_your_judgment |
| 10 | iter9_bulk_passage_harvest | 1565 | use_your_judgment |
| 11 | iter8_criterion_window_evidence | 1565 | use_your_judgment |
| 12 | iter7_solo_mining_coverage | 1564 | use_your_judgment |
| 13 | iter9_bulk_passage_harvest | 1561 | use_your_judgment |
| 14 | iter8_criterion_window_evidence | 1560 | use_your_judgment |
| 15 | iter8_criterion_window_evidence | 1558 | use_your_judgment |
| 16 | iter9_bulk_passage_harvest | 1585 | use_your_judgment |
| 17 | iter9_bulk_passage_harvest | 1610 | use_your_judgment |
| 18 | iter8_criterion_window_evidence | 1589 | use_your_judgment |
| 19 | iter8_criterion_window_evidence | 1582 | use_your_judgment |
| 20 | iter8_criterion_window_evidence | 1582 | use_your_judgment |
| 21 | iter8_criterion_window_evidence | 1578 | use_your_judgment |
| 22 | iter8_criterion_window_evidence | 1576 | use_your_judgment |
| 23 | iter21_gold_rubric_and_hard_predicates | 1602 | use_your_judgment |


## Deep Focus Evolution Timing Analysis
- **Iterations with evolution**: 20/23
- **Total evolution time**: 243.6 minutes
- **Average evolution time**: 12.2 minutes

### Time Breakdown by Evolution Phase

| Phase | Avg Time | % of Evolution |
|-------|----------|----------------|
| First Draft (Round 1) | 10.9m | 89.7% |
| Reflection | 1.3m | 10.3% |
| **Total** | **12.2m** | **100.0%** |

### Evolution vs Test vs Meta-Evolution Time

| Metric | Time | Percentage | Avg/Iter | Avg/Occurrence |
|--------|------|------------|----------|----------------|
| Total Evolution Time | 243.6m | 17.3% | 10.6m | 12.2m |
| Total Test Time | 1160.7m | 82.7% | 50.5m | 50.5m |
| Total Meta-Evolution Time | 0.0m | 0.0% | 0.0m | 0.0m |
| **Total** | **1404.3m** | **100.0%** | **61.1m** | **-** |

- **Evolution/Test ratio**: 0.21x
  - For every minute of testing, 0.21 minutes spent on evolution

## Claude CLI Usage & Cost Analysis
- **Total Cost**: $199.94
  - **Evaluation Cost**: $104.26
  - **Evolution CLI Cost**: $73.00
  - **Other (evaluator overhead) Cost**: $22.67
- **Total CLI Calls**: 40
- **Total CLI Input Tokens**: 12,216
- **Total CLI Output Tokens**: 858,144
- **Total Cache Created**: 2,282,998
- **Total Cache Read**: 57,228,653

### Cost Breakdown

| Operation | Cost | % of Total | Calls | Tokens In | Tokens Out |
|-----------|------|------------|-------|-----------|------------|
| Evaluation | $104.26 | 52.1% | - | - | - |
| Evolution (CLI) | $73.00 | 36.5% | 40 | 12,216 | 858,144 |
| Other (evaluator overhead) | $22.67 | 11.3% | - | - | - |
| **Total** | **$199.94** | **100%** | **40** | **12,216** | **858,144** |

**Answer: Evaluation drives 52.1% of total costs**

### Detailed Per-Iteration Costs

| Iter | Total | **Eval** | **Evo** | Iter Eval | 1st Draft | Refl | Meta | Other | Fresh | Strategy | Meta-strategy |
|------|------|------|------|------|------|------|------|------|------|------|------|
| **1** | $0.05 | **$0.01** | **-** | $0.01 | - | - | - | $0.04 | 14 | - | - |
| **2** | $4.45 | **$0.75** | **$3.14** | $0.75 | $2.20 | $0.94 | - | $0.56 | 26 | use_your_judgment | - |
| **3** | $8.34 | **$2.45** | **$3.95** | $2.45 | $3.60 | $0.35 | - | $1.94 | 36 | use_your_judgment | - |
| **4** | $11.33 | **$4.05** | **$3.86** | $4.05 | $3.57 | $0.28 | - | $3.42 | 37 | use_your_judgment | - |
| **5** | $14.17 | **$4.75** | **$7.07** | $4.75 | $6.66 | $0.41 | - | $2.35 | 35 | use_your_judgment | - |
| **6** | $9.03 | **$3.31** | **$3.47** | $3.31 | $3.11 | $0.36 | - | $2.26 | 28 | use_your_judgment | - |
| **7** | $10.65 | **$4.61** | **$2.87** | $4.61 | $2.57 | $0.29 | - | $3.18 | 27 | use_your_judgment | - |
| **8** | $11.72 | **$5.52** | **$3.58** | $5.52 | $3.28 | $0.30 | - | $2.62 | 30 | use_your_judgment | - |
| **9** | $12.00 | **$6.38** | **$4.88** | $6.38 | $3.65 | $1.23 | - | $0.74 | 32 | use_your_judgment | - |
| **10** | $10.50 | **$6.94** | **$3.01** | $6.94 | $2.76 | $0.25 | - | $0.55 | 35 | use_your_judgment | - |
| **11** | $8.09 | **$4.79** | **$2.99** | $4.79 | $2.68 | $0.31 | - | $0.31 | 32 | use_your_judgment | - |
| **12** | $8.17 | **$4.60** | **$3.35** | $4.60 | $3.08 | $0.27 | - | $0.23 | 30 | use_your_judgment | - |
| **13** | $7.78 | **$4.31** | **$3.09** | $4.31 | $2.81 | $0.27 | - | $0.38 | 23 | use_your_judgment | - |
| **14** | $7.72 | **$4.17** | **$2.97** | $4.17 | $2.69 | $0.28 | - | $0.58 | 24 | use_your_judgment | - |
| **15** | $11.02 | **$5.54** | **$5.00** | $5.54 | $3.74 | $1.26 | - | $0.48 | 24 | use_your_judgment | - |
| **16** | $8.13 | **$4.80** | **$3.02** | $4.80 | $2.76 | $0.26 | - | $0.31 | 27 | use_your_judgment | - |
| **17** | $11.73 | **$6.95** | **$4.21** | $6.95 | $3.89 | $0.32 | - | $0.57 | 27 | use_your_judgment | - |
| **18** | $8.61 | **$5.45** | **$2.78** | $5.45 | $2.52 | $0.26 | - | $0.38 | 22 | use_your_judgment | - |
| **19** | $7.82 | **$4.26** | **$3.25** | $4.26 | $2.97 | $0.28 | - | $0.31 | 20 | use_your_judgment | - |
| **20** | $10.75 | **$7.68** | **$2.55** | $7.68 | $2.26 | $0.29 | - | $0.51 | 29 | use_your_judgment | - |
| **21** | $10.99 | **$6.56** | **$3.97** | $6.56 | $3.70 | $0.27 | - | $0.47 | 25 | use_your_judgment | - |
| **22** | $4.33 | **$4.05** | **-** | $4.05 | - | - | - | $0.28 | 16 | - | - |
| **23** | $2.55 | **$2.34** | **-** | $2.34 | - | - | - | $0.21 | 10 | - | - |
|------|------|------|------|------|------|------|------|------|------|------|------|
| **TOTAL** | **$199.94** | **$104.26** | **$73.00** | $104.26 | $64.51 | $8.49 | $0.00 | **$22.67** | **609** | - | - |

### Evolution Cost Breakdown (Avg per evolution)

| Phase | Avg Cost | % of Evolution | Avg Tokens In | Avg Tokens Out |
|-------|----------|----------------|---------------|----------------|
| First Draft (Round 1) | $3.23 | 88.4% | 607 | 37972 |
| Reflection | $0.42 | 11.6% | 4 | 4936 |
| **Total** | **$3.65** | **100%** | **611** | **42907** |

### Cache Performance

- **Cache Creation**: 2,282,998 tokens
- **Cache Hits**: 57,228,653 tokens (96.2% hit rate)

## Zero Accuracy Cases
No zero accuracy cases encountered ✅

## Exception Failures
No exception failures encountered ✅

## Evolution Health

No artifact validation failures ✅