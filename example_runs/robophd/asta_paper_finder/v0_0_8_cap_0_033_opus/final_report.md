# Parallel Agent Research - Final Report

## Experiment Configuration

**Run ID**: asta_paper_finder_20260724_004103
**Date**: 2026-07-24 11:02:45
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
- include_prior_evolution: False
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
- paper_finder_runtime: {'cost_threshold': 0.033, 'cost_per_error': 0.003, 'cap_judge_to_estimate': True, 'training_judge': 'openai/gpt-5.6-luna', 'judge_prompt': 'no-prose', 'evidence_char_cap': 2500}

### User Configuration (Iteration 1)

The following parameters were explicitly set, overriding defaults:

- agents_directory: ../robophd_runs/robophd/_optimize_anything_seeds
- domain: external
- eval_timeout: 1800
- evaluation_budget: 600
- evolution_model: opus-4.8 (claude-opus-4-8)
- evolution_strategy: use_your_judgment
- evolution_timeout: 3600
- examples_per_iteration: 14
- include_prior_evolution: True
- initial_agents: ['seed_k_udbj7b']
- max_workers: 8
- meta_evolution_cadence: 3
- meta_evolution_domain: asta_paper_finder
- meta_evolution_first_iteration: 4
- meta_evolution_model: opus-4.8 (claude-opus-4-8)
- new_agent_test_rounds: 0

### Configuration Changes by Iteration

(No configuration changes across iterations)


## Winning Agents by Evolution Strategy

#### **Use Your Judgment** (21 wins - 95.5% of all wins)
- **iter6_metadata_targeted_v1** (5 wins, created iteration 6)
- **iter14_richev_grade3rerank_v1** (4 wins, created iteration 14)
- **iter3_evidence_rerank_v1** (3 wins, created iteration 3)
- **iter9_rerank_rich_v1** (3 wins, created iteration 9)
- **iter15_facetsnip_richev_v1** (2 wins, created iteration 15)
- **iter2_recall_max_v1** (1 win, created iteration 2)
- **iter4_facet_evidence_v1** (1 win, created iteration 4)
- **iter5_grade3_rerank_v1** (1 win, created iteration 5)
- **iter20_robust_rerank_rich_v1** (1 win, created iteration 20)

#### **Initial** (1 wins - 4.5% of all wins)
- **seed_k_udbj7b** (1 win, created iteration 0)

#### Summary Statistics
- **Total iterations with winners**: 22
- **Total wins counted**: 22 (includes ties)
- **Unique winning agents**: 10
- **Evolution strategies that produced winners**: 2


## Complete Performance Ranking Table

| Agent | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 | Iter 6 | Iter 7 | Iter 8 | Iter 9 | Iter 10 | Iter 11 | Iter 12 | Iter 13 | Iter 14 | Iter 15 | Iter 16 | Iter 17 | Iter 18 | Iter 19 | Iter 20 | Iter 21 | Iter 22 | Final Elo | Mean Rank |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|-----------|----------|
| iter9_rerank_rich_v1 | - | - | - | - | - | - | - | - | #2 12.064 | - | - | - | **#1** 16.085 | #2 34.057 | - | - | **#1** 22.772 | #2 40.108 | **#1** 23.102 | #2 21.311 | - | - | **1589** | **1.57** |
| iter14_richev_grade3rerank_v1 | - | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 34.330 | #2 31.040 | **#1** 33.016 | #3 22.068 | - | - | - | **#1** 32.317 | **#1** 30.484 | 1588 | **1.50** |
| iter6_metadata_targeted_v1 | - | - | - | - | - | **#1** 20.362 | **#1** 26.484 | **#1** 20.566 | #3 11.958 | - | **#1** 31.560 | **#1** 20.774 | #2 15.280 | #3 33.357 | #3 27.583 | - | - | - | - | - | - | - | 1522 | **1.78** |
| iter3_evidence_rerank_v1 | - | - | **#1** 31.614 | **#1** 18.589 | #3 15.393 | #2 13.364 | #2 21.860 | - | **#1** 12.227 | #3 23.403 | - | - | - | - | - | - | - | - | - | - | - | - | 1520 | **1.86** |
| iter20_robust_rerank_rich_v1 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 22.992 | #3 31.595 | - | 1505 | 2.00 |
| iter17_richev_abstractguard_v1 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | #2 22.268 | - | - | - | - | - | 1505 | 2.00 |
| iter22_richdump_robuststack_v1 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | #2 30.332 | 1504 | 2.00 |
| iter11_grade3_metadata_v1 | - | - | - | - | - | - | - | - | - | - | #2 26.667 | - | - | - | - | - | - | - | - | - | - | - | 1504 | 2.00 |
| iter12_robust_specific_v1 | - | - | - | - | - | - | - | - | - | - | - | #2 19.750 | - | - | - | - | - | - | - | - | - | - | 1504 | 2.00 |
| iter21_metacite_author_v1 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | #2 32.160 | - | 1503 | 2.00 |
| iter10_richev_metadata_v1 | - | - | - | - | - | - | - | - | - | #2 25.106 | - | - | - | - | - | - | - | - | - | - | - | - | 1503 | 2.00 |
| iter4_facet_evidence_v1 | - | - | - | #2 15.901 | **#1** 23.767 | #3 12.590 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1502 | 2.00 |
| iter15_facetsnip_richev_v1 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 32.849 | #2 22.971 | - | **#1** 40.248 | #2 14.343 | #3 21.037 | - | #3 26.345 | 1499 | 2.00 |
| iter2_recall_max_v1 | - | **#1** 13.275 | #2 23.740 | #3 13.929 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1483 | 2.00 |
| iter5_grade3_rerank_v1 | - | - | - | - | #2 18.859 | - | - | #2 17.801 | - | **#1** 33.243 | #3 23.911 | #3 18.737 | - | - | - | - | - | - | - | - | - | - | 1478 | 2.20 |
| iter19_abstractfacet_richev_v1 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | #3 9.176 | - | - | - | 1475 | 3.00 |
| iter13_robust_metadata_v1 | - | - | - | - | - | - | - | - | - | - | - | - | #3 14.685 | - | - | - | - | - | - | - | - | - | 1474 | 3.00 |
| iter18_citeanchor_abstractguard_v1 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | #3 40.090 | - | - | - | - | 1473 | 3.00 |
| iter16_facetrerank_richev_v1 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | #3 15.616 | - | - | - | - | - | - | 1472 | 3.00 |
| iter8_grade3_probe_v1 | - | - | - | - | - | - | - | #3 17.567 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1472 | 3.00 |
| iter7_grade3_evidence_v1 | - | - | - | - | - | - | #3 16.146 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1471 | 3.00 |
| seed_k_udbj7b | **#1** 7.622 | #2 7.259 | #3 4.680 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1455 | 2.50 |

### Legend:
- **#1** = 1st place (winner of iteration)
- #2, #3, etc. = 2nd, 3rd place, etc.
- Score = Average score (0-1) on that iteration's problems
- **Bold Elo/Rank** = Top performer
- `-` = Agent not tested in that iteration



### Quick Summary

| Agent | Elo | Mean Score | Mean Cost | Tests |
|-------|-----|------------|-----------|-------|
| iter9_rerank_rich_v1 | 1589 | 24.214 | $0.006 | 7 |
| iter14_richev_grade3rerank_v1 | 1588 | 30.542 | $0.007 | 6 |
| iter6_metadata_targeted_v1 | 1522 | 23.103 | $0.006 | 9 |
| iter3_evidence_rerank_v1 | 1520 | 19.493 | $0.003 | 7 |
| iter20_robust_rerank_rich_v1 | 1505 | 27.293 | $0.006 | 2 |
| iter17_richev_abstractguard_v1 | 1505 | 22.268 | $0.006 | 1 |
| iter22_richdump_robuststack_v1 | 1504 | 30.332 | $0.007 | 1 |
| iter11_grade3_metadata_v1 | 1504 | 26.667 | $0.019 | 1 |
| iter12_robust_specific_v1 | 1504 | 19.750 | $0.019 | 1 |
| iter21_metacite_author_v1 | 1503 | 32.160 | $0.006 | 1 |
| iter10_richev_metadata_v1 | 1503 | 25.106 | $0.003 | 1 |
| iter4_facet_evidence_v1 | 1502 | 17.419 | $0.005 | 3 |
| iter15_facetsnip_richev_v1 | 1499 | 26.299 | $0.006 | 6 |
| iter2_recall_max_v1 | 1483 | 16.981 | $0.000 | 3 |
| iter5_grade3_rerank_v1 | 1478 | 22.510 | $0.015 | 5 |
| iter19_abstractfacet_richev_v1 | 1475 | 9.176 | $0.006 | 1 |
| iter13_robust_metadata_v1 | 1474 | 14.685 | $0.006 | 1 |
| iter18_citeanchor_abstractguard_v1 | 1473 | 40.090 | $0.005 | 1 |
| iter16_facetrerank_richev_v1 | 1472 | 15.616 | $0.006 | 1 |
| iter8_grade3_probe_v1 | 1472 | 17.567 | $0.005 | 1 |
| iter7_grade3_evidence_v1 | 1471 | 16.146 | $0.003 | 1 |
| seed_k_udbj7b | 1455 | 6.521 | $0.001 | 3 |

### Iteration Progression

| Iter | Winner(s) | Score | Evo Time | Test Time | Total Time |
|------|-----------|-------|----------|-----------|------------|
| 1 | seed_k_udbj7b | 7.622 | - | 2.3m | 2.3m |
| 2 | iter2_recall_max_v1 | 13.275 | 8.2m | 15.1m | 23.3m |
| 3 | iter3_evidence_rerank_v1 | 31.614 | 7.3m | 19.6m | 26.9m |
| 4 | iter3_evidence_rerank_v1 | 18.589 | 10.4m | 25.3m | 35.6m |
| 5 | iter4_facet_evidence_v1 | 23.767 | 9.9m | 28.2m | 38.1m |
| 6 | iter6_metadata_targeted_v1 | 20.362 | 8.7m | 34.9m | 43.6m |
| 7 | iter6_metadata_targeted_v1 | 26.484 | 8.6m | 62.5m | 71.1m |
| 8 | iter6_metadata_targeted_v1 | 20.566 | 7.3m | 40.6m | 47.9m |
| 9 | iter3_evidence_rerank_v1 | 12.227 | 7.6m | 27.9m | 35.5m |
| 10 | iter5_grade3_rerank_v1 | 33.243 | 8.9m | 15.0m | 23.9m |
| 11 | iter6_metadata_targeted_v1 | 31.560 | 8.4m | 24.8m | 33.2m |
| 12 | iter6_metadata_targeted_v1 | 20.774 | 8.7m | 11.8m | 20.5m |
| 13 | iter9_rerank_rich_v1 | 16.085 | 7.7m | 29.3m | 37.0m |
| 14 | iter14_richev_grade3rerank_v1 | 34.330 | 7.6m | 16.5m | 24.0m |
| 15 | iter15_facetsnip_richev_v1 | 32.849 | 7.5m | 14.3m | 21.7m |
| 16 | iter14_richev_grade3rerank_v1 | 33.016 | 7.8m | 11.4m | 19.2m |
| 17 | iter9_rerank_rich_v1 | 22.772 | 6.6m | 9.8m | 16.4m |
| 18 | iter15_facetsnip_richev_v1 | 40.248 | 8.0m | 11.8m | 19.8m |
| 19 | iter9_rerank_rich_v1 | 23.102 | 8.9m | 6.6m | 15.5m |
| 20 | iter20_robust_rerank_rich_v1 | 22.992 | 8.0m | 12.4m | 20.4m |
| 21 | iter14_richev_grade3rerank_v1 | 32.317 | 5.5m | 14.6m | 20.1m |
| 22 | iter14_richev_grade3rerank_v1 | 30.484 | 7.2m | 18.2m | 25.4m |

## Best Agent: iter9_rerank_rich_v1
- Elo Score: 1589
- Mean Score: 24.214


## Elo Leadership Progression

Iteration-by-iteration Elo leaders:

| Iteration | Leader | Elo | Evolution Strategy |
|-----------|--------|-----|-------------------|
| 1 | seed_k_udbj7b | 1500 | Initial |
| 2 | iter2_recall_max_v1 | 1516 | use_your_judgment |
| 3 | iter3_evidence_rerank_v1 | 1531 | use_your_judgment |
| 4 | iter3_evidence_rerank_v1 | 1560 | use_your_judgment |
| 5 | iter4_facet_evidence_v1 | 1536 | use_your_judgment |
| 6 | iter6_metadata_targeted_v1 | 1534 | use_your_judgment |
| 7 | iter6_metadata_targeted_v1 | 1563 | use_your_judgment |
| 8 | iter6_metadata_targeted_v1 | 1589 | use_your_judgment |
| 9 | iter3_evidence_rerank_v1 | 1556 | use_your_judgment |
| 10 | iter6_metadata_targeted_v1 | 1551 | use_your_judgment |
| 11 | iter6_metadata_targeted_v1 | 1579 | use_your_judgment |
| 12 | iter6_metadata_targeted_v1 | 1604 | use_your_judgment |
| 13 | iter6_metadata_targeted_v1 | 1595 | use_your_judgment |
| 14 | iter6_metadata_targeted_v1 | 1557 | use_your_judgment |
| 15 | iter9_rerank_rich_v1 | 1541 | use_your_judgment |
| 16 | iter14_richev_grade3rerank_v1 | 1566 | use_your_judgment |
| 17 | iter9_rerank_rich_v1 | 1571 | use_your_judgment |
| 18 | iter9_rerank_rich_v1 | 1567 | use_your_judgment |
| 19 | iter9_rerank_rich_v1 | 1595 | use_your_judgment |
| 20 | iter9_rerank_rich_v1 | 1589 | use_your_judgment |
| 21 | iter9_rerank_rich_v1 | 1589 | use_your_judgment |
| 22 | iter9_rerank_rich_v1 | 1589 | use_your_judgment |


## Deep Focus Evolution Timing Analysis
- **Iterations with evolution**: 21/22
- **Total evolution time**: 168.8 minutes
- **Average evolution time**: 8.0 minutes

### Time Breakdown by Evolution Phase

| Phase | Avg Time | % of Evolution |
|-------|----------|----------------|
| First Draft (Round 1) | 7.3m | 90.2% |
| Reflection | 0.8m | 9.8% |
| **Total** | **8.0m** | **100.0%** |

### Evolution vs Test vs Meta-Evolution Time

| Metric | Time | Percentage | Avg/Iter | Avg/Occurrence |
|--------|------|------------|----------|----------------|
| Total Evolution Time | 168.8m | 27.1% | 7.7m | 8.0m |
| Total Test Time | 452.9m | 72.9% | 20.6m | 20.6m |
| Total Meta-Evolution Time | 0.0m | 0.0% | 0.0m | 0.0m |
| **Total** | **621.7m** | **100.0%** | **28.3m** | **-** |

- **Evolution/Test ratio**: 0.37x
  - For every minute of testing, 0.37 minutes spent on evolution

## Claude CLI Usage & Cost Analysis
- **Total Cost**: $98.65
  - **Evaluation Cost**: $3.68
  - **Evolution CLI Cost**: $64.89
  - **Other (evaluator overhead) Cost**: $30.08
- **Total CLI Calls**: 42
- **Total CLI Input Tokens**: 22,376
- **Total CLI Output Tokens**: 720,148
- **Total Cache Created**: 2,958,822
- **Total Cache Read**: 34,280,021

### Cost Breakdown

| Operation | Cost | % of Total | Calls | Tokens In | Tokens Out |
|-----------|------|------------|-------|-----------|------------|
| Evaluation | $3.68 | 3.7% | - | - | - |
| Evolution (CLI) | $64.89 | 65.8% | 42 | 22,376 | 720,148 |
| Other (evaluator overhead) | $30.08 | 30.5% | - | - | - |
| **Total** | **$98.65** | **100%** | **42** | **22,376** | **720,148** |

**Answer: Evolution drives 65.8% of total costs**

### Detailed Per-Iteration Costs

| Iter | Total | **Eval** | **Evo** | Iter Eval | 1st Draft | Refl | Meta | Other | Fresh | Strategy | Meta-strategy |
|------|------|------|------|------|------|------|------|------|------|------|------|
| **1** | $0.17 | **$0.01** | **-** | $0.01 | - | - | - | $0.16 | 14 | - | - |
| **2** | $3.10 | **$0.01** | **$1.92** | $0.01 | $1.76 | $0.16 | - | $1.18 | 26 | use_your_judgment | - |
| **3** | $4.77 | **$0.05** | **$1.86** | $0.05 | $1.68 | $0.17 | - | $2.87 | 33 | use_your_judgment | - |
| **4** | $7.47 | **$0.12** | **$3.71** | $0.12 | $2.73 | $0.98 | - | $3.64 | 38 | use_your_judgment | - |
| **5** | $5.26 | **$0.27** | **$2.78** | $0.27 | $2.39 | $0.39 | - | $2.21 | 35 | use_your_judgment | - |
| **6** | $5.49 | **$0.12** | **$3.22** | $0.12 | $2.29 | $0.93 | - | $2.15 | 26 | use_your_judgment | - |
| **7** | $5.15 | **$0.12** | **$2.97** | $0.12 | $2.75 | $0.22 | - | $2.06 | 28 | use_your_judgment | - |
| **8** | $4.86 | **$0.26** | **$3.20** | $0.26 | $2.29 | $0.92 | - | $1.39 | 32 | use_your_judgment | - |
| **9** | $3.76 | **$0.16** | **$2.45** | $0.16 | $2.22 | $0.23 | - | $1.15 | 28 | use_your_judgment | - |
| **10** | $5.61 | **$0.20** | **$3.90** | $0.20 | $2.74 | $1.16 | - | $1.51 | 27 | use_your_judgment | - |
| **11** | $5.61 | **$0.44** | **$3.46** | $0.44 | $2.42 | $1.04 | - | $1.71 | 29 | use_your_judgment | - |
| **12** | $4.64 | **$0.31** | **$3.46** | $0.31 | $2.63 | $0.83 | - | $0.87 | 18 | use_your_judgment | - |
| **13** | $4.63 | **$0.16** | **$3.35** | $0.16 | $2.53 | $0.82 | - | $1.12 | 27 | use_your_judgment | - |
| **14** | $5.25 | **$0.18** | **$3.86** | $0.18 | $2.72 | $1.14 | - | $1.21 | 28 | use_your_judgment | - |
| **15** | $3.85 | **$0.19** | **$3.08** | $0.19 | $2.13 | $0.95 | - | $0.59 | 27 | use_your_judgment | - |
| **16** | $4.58 | **$0.23** | **$2.99** | $0.23 | $2.04 | $0.94 | - | $1.36 | 38 | use_your_judgment | - |
| **17** | $4.22 | **$0.14** | **$2.70** | $0.14 | $1.86 | $0.84 | - | $1.37 | 26 | use_your_judgment | - |
| **18** | $4.65 | **$0.17** | **$3.04** | $0.17 | $2.20 | $0.84 | - | $1.44 | 31 | use_your_judgment | - |
| **19** | $4.27 | **$0.11** | **$3.59** | $0.11 | $2.64 | $0.95 | - | $0.56 | 21 | use_your_judgment | - |
| **20** | $4.38 | **$0.12** | **$3.90** | $0.12 | $2.85 | $1.05 | - | $0.36 | 23 | use_your_judgment | - |
| **21** | $3.06 | **$0.17** | **$2.46** | $0.17 | $1.74 | $0.73 | - | $0.43 | 28 | use_your_judgment | - |
| **22** | $3.86 | **$0.14** | **$2.98** | $0.14 | $2.15 | $0.83 | - | $0.75 | 19 | use_your_judgment | - |
|------|------|------|------|------|------|------|------|------|------|------|------|
| **TOTAL** | **$98.65** | **$3.68** | **$64.89** | $3.68 | $48.77 | $16.12 | $0.00 | **$30.08** | **602** | - | - |

### Evolution Cost Breakdown (Avg per evolution)

| Phase | Avg Cost | % of Evolution | Avg Tokens In | Avg Tokens Out |
|-------|----------|----------------|---------------|----------------|
| First Draft (Round 1) | $2.32 | 75.2% | 1062 | 31561 |
| Reflection | $0.77 | 24.8% | 4 | 2732 |
| **Total** | **$3.09** | **100%** | **1066** | **34293** |

### Cache Performance

- **Cache Creation**: 2,958,822 tokens
- **Cache Hits**: 34,280,021 tokens (92.1% hit rate)

## Zero Accuracy Cases
No zero accuracy cases encountered ✅

## Exception Failures
No exception failures encountered ✅

## Evolution Health

No artifact validation failures ✅