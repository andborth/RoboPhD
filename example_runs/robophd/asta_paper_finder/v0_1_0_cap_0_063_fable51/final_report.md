# Parallel Agent Research - Final Report

## Experiment Configuration

**Run ID**: asta_paper_finder_20261019_184607
**Date**: 2026-10-20 19:16:29
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

- file_mapping: {'agent.py': 'agent.py'}
- paper_finder_runtime: {'cost_threshold': 0.063, 'cost_per_error': 0.0063, 'cap_judge_to_estimate': True, 'training_judge': 'openai/gpt-5.6-luna', 'judge_prompt': 'no-prose', 'evidence_char_cap': 2500}

### User Configuration (Iteration 1)

The following parameters were explicitly set, overriding defaults:

- agents_directory: ../robophd_runs/robophd/_optimize_anything_seeds/seeds_89d6o4_2
- domain: external
- eval_timeout: 1800
- evaluation_budget: 600
- evolution_model: fable-5.1 (claude-fable-5-1)
- evolution_strategy: use_your_judgment
- evolution_timeout: 3600
- examples_per_iteration: 14
- initial_agents: ['baseline']
- max_workers: 8
- meta_evolution_cadence: 3
- meta_evolution_domain: asta_paper_finder
- meta_evolution_first_iteration: 4
- meta_evolution_model: opus-5.5 (claude-opus-5-5)
- new_agent_test_rounds: 0

### Configuration Changes by Iteration

- **Iteration 21**: evolution_strategy=greedy


## Winning Agents by Evolution Strategy

#### **Use Your Judgment** (20 wins - 95.2% of all wins)
- **iter7_headcast** (4 wins, created iteration 7)
- **iter3_widecast** (3 wins, created iteration 3)
- **iter16_feedcast** (3 wins, created iteration 16)
- **iter18_oldcast** (3 wins, created iteration 18)
- **iter11_citecast** (2 wins, created iteration 11)
- **iter13_crosscast** (2 wins, created iteration 13)
- **iter2_deepcast** (1 win, created iteration 2)
- **iter6_surecast** (1 win, created iteration 6)
- **iter15_bodycast** (1 win, created iteration 15)

#### **Initial** (1 wins - 4.8% of all wins)
- **baseline** (1 win, created iteration 0)

#### Summary Statistics
- **Total iterations with winners**: 21
- **Total wins counted**: 21 (includes ties)
- **Unique winning agents**: 10
- **Evolution strategies that produced winners**: 2


## Complete Performance Ranking Table

| Agent | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 | Iter 6 | Iter 7 | Iter 8 | Iter 9 | Iter 10 | Iter 11 | Iter 12 | Iter 13 | Iter 14 | Iter 15 | Iter 16 | Iter 17 | Iter 18 | Iter 19 | Iter 20 | Iter 21 | Final Elo | Mean Rank |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|-----------|----------|
| iter16_feedcast | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 48.402 | **#1** 44.195 | #2 27.807 | #2 48.820 | - | **#1** 47.891 | **1590** | **1.40** |
| iter18_oldcast | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 30.479 | **#1** 50.619 | **#1** 48.263 | #2 46.369 | 1589 | **1.25** |
| iter12_provecast | - | - | - | - | - | - | - | - | - | - | - | #2 28.570 | - | - | - | - | - | - | - | - | - | 1506 | 2.00 |
| iter6_surecast | - | - | - | - | - | **#1** 51.928 | #3 32.394 | - | #2 48.734 | - | - | - | - | - | - | - | - | - | - | - | - | 1506 | 2.00 |
| iter14_refcast | - | - | - | - | - | - | - | - | - | - | - | - | - | #2 36.011 | - | - | - | - | - | - | - | 1504 | 2.00 |
| iter17_loopcast | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | #2 43.856 | - | - | - | - | 1504 | 2.00 |
| iter20_origcast | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | #2 47.509 | - | 1504 | 2.00 |
| iter15_bodycast | - | - | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 58.862 | #2 45.215 | - | #3 17.009 | - | - | - | 1504 | 2.00 |
| iter8_critcast | - | - | - | - | - | - | - | #2 41.452 | - | - | - | - | - | - | - | - | - | - | - | - | - | 1502 | 2.00 |
| iter13_crosscast | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 36.293 | **#1** 38.512 | #2 58.413 | - | #3 40.450 | - | - | - | #3 39.592 | 1499 | 2.00 |
| iter7_headcast | - | - | - | - | - | - | **#1** 34.514 | **#1** 42.867 | **#1** 50.968 | **#1** 46.418 | #2 32.907 | #3 27.698 | - | #3 29.717 | #3 51.810 | - | - | - | - | - | - | 1499 | **1.88** |
| iter11_citecast | - | - | - | - | - | - | - | - | - | - | **#1** 34.990 | **#1** 29.318 | #2 34.104 | - | - | #3 43.367 | - | - | - | #3 38.617 | - | 1497 | 2.00 |
| iter3_widecast | - | - | **#1** 28.559 | **#1** 25.742 | **#1** 33.424 | #3 39.927 | #2 33.015 | - | - | #3 45.377 | - | - | #3 28.977 | - | - | - | - | - | - | - | - | 1484 | 2.00 |
| iter2_deepcast | - | **#1** 36.296 | #2 28.092 | #2 24.536 | #3 31.623 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1484 | 2.00 |
| iter10_clearcast | - | - | - | - | - | - | - | - | - | #2 45.492 | #3 29.837 | - | - | - | - | - | - | - | - | - | - | 1479 | 2.50 |
| iter5_broadcast | - | - | - | - | #2 31.681 | #2 40.736 | - | #3 37.893 | - | - | - | - | - | - | - | - | - | - | - | - | - | 1476 | 2.33 |
| iter19_extcast | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | #3 48.126 | - | - | 1473 | 3.00 |
| iter9_steadycast | - | - | - | - | - | - | - | - | #3 45.072 | - | - | - | - | - | - | - | - | - | - | - | - | 1472 | 3.00 |
| iter4_judgecast | - | - | - | #3 18.103 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1471 | 3.00 |
| baseline | **#1** 7.859 | #2 1.823 | #3 1.812 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1455 | 2.50 |

### Legend:
- **#1** = 1st place (winner of iteration)
- #2, #3, etc. = 2nd, 3rd place, etc.
- Score = Average score (0-1) on that iteration's problems
- **Bold Elo/Rank** = Top performer
- `-` = Agent not tested in that iteration



### Quick Summary

| Agent | Elo | Mean Score | Mean Cost | Tests |
|-------|-----|------------|-----------|-------|
| iter16_feedcast | 1590 | 43.423 | $0.060 | 5 |
| iter18_oldcast | 1589 | 43.932 | $0.062 | 4 |
| iter12_provecast | 1506 | 28.570 | $0.047 | 1 |
| iter6_surecast | 1506 | 44.352 | $0.049 | 3 |
| iter14_refcast | 1504 | 36.011 | $0.056 | 1 |
| iter17_loopcast | 1504 | 43.856 | $0.056 | 1 |
| iter20_origcast | 1504 | 47.509 | $0.051 | 1 |
| iter15_bodycast | 1504 | 40.362 | $0.068 | 3 |
| iter8_critcast | 1502 | 41.452 | $0.058 | 1 |
| iter13_crosscast | 1499 | 42.652 | $0.052 | 5 |
| iter7_headcast | 1499 | 39.612 | $0.052 | 8 |
| iter11_citecast | 1497 | 36.079 | $0.056 | 5 |
| iter3_widecast | 1484 | 33.574 | $0.033 | 7 |
| iter2_deepcast | 1484 | 30.137 | $0.025 | 4 |
| iter10_clearcast | 1479 | 37.664 | $0.059 | 2 |
| iter5_broadcast | 1476 | 36.770 | $0.045 | 3 |
| iter19_extcast | 1473 | 48.126 | $0.048 | 1 |
| iter9_steadycast | 1472 | 45.072 | $0.056 | 1 |
| iter4_judgecast | 1471 | 18.103 | $0.044 | 1 |
| baseline | 1455 | 3.832 | $0.001 | 3 |

### Iteration Progression

| Iter | Winner(s) | Score | Evo Time | Test Time | Total Time |
|------|-----------|-------|----------|-----------|------------|
| 1 | baseline | 7.859 | - | 1.0m | 1.0m |
| 2 | iter2_deepcast | 36.296 | 21.1m | 9.5m | 30.6m |
| 3 | iter3_widecast | 28.559 | 24.0m | 35.8m | 59.7m |
| 4 | iter3_widecast | 25.742 | 60.9m | 47.9m | 108.9m |
| 5 | iter3_widecast | 33.424 | 20.4m | 46.6m | 67.1m |
| 6 | iter6_surecast | 51.928 | 13.3m | 45.2m | 58.5m |
| 7 | iter7_headcast | 34.514 | 16.6m | 51.8m | 68.4m |
| 8 | iter7_headcast | 42.867 | 24.1m | 51.3m | 75.4m |
| 9 | iter7_headcast | 50.968 | 13.8m | 38.2m | 52.0m |
| 10 | iter7_headcast | 46.418 | 9.1m | 34.0m | 43.0m |
| 11 | iter11_citecast | 34.990 | 23.2m | 51.3m | 74.5m |
| 12 | iter11_citecast | 29.318 | 24.3m | 61.5m | 85.8m |
| 13 | iter13_crosscast | 36.293 | 51.7m | 38.2m | 89.9m |
| 14 | iter13_crosscast | 38.512 | 50.0m | 61.2m | 111.2m |
| 15 | iter15_bodycast | 58.862 | 21.9m | 47.0m | 68.9m |
| 16 | iter16_feedcast | 48.402 | 20.6m | 65.2m | 85.8m |
| 17 | iter16_feedcast | 44.195 | 29.0m | 52.2m | 81.2m |
| 18 | iter18_oldcast | 30.479 | 42.3m | 54.1m | 96.4m |
| 19 | iter18_oldcast | 50.619 | 28.6m | 62.6m | 91.2m |
| 20 | iter18_oldcast | 48.263 | 14.8m | 55.6m | 70.3m |
| 21 | iter16_feedcast | 47.891 | - | 50.4m | 50.4m |

## Best Agent: iter16_feedcast
- Elo Score: 1590
- Mean Score: 43.423


## Elo Leadership Progression

Iteration-by-iteration Elo leaders:

| Iteration | Leader | Elo | Evolution Strategy |
|-----------|--------|-----|-------------------|
| 1 | baseline | 1500 | Initial |
| 2 | iter2_deepcast | 1516 | use_your_judgment |
| 3 | iter3_widecast | 1531 | use_your_judgment |
| 4 | iter3_widecast | 1560 | use_your_judgment |
| 5 | iter3_widecast | 1587 | use_your_judgment |
| 6 | iter3_widecast | 1548 | use_your_judgment |
| 7 | iter3_widecast | 1545 | use_your_judgment |
| 8 | iter7_headcast | 1563 | use_your_judgment |
| 9 | iter7_headcast | 1589 | use_your_judgment |
| 10 | iter7_headcast | 1615 | use_your_judgment |
| 11 | iter7_headcast | 1605 | use_your_judgment |
| 12 | iter11_citecast | 1569 | use_your_judgment |
| 13 | iter7_headcast | 1566 | use_your_judgment |
| 14 | iter13_crosscast | 1566 | use_your_judgment |
| 15 | iter11_citecast | 1564 | use_your_judgment |
| 16 | iter13_crosscast | 1562 | use_your_judgment |
| 17 | iter16_feedcast | 1567 | use_your_judgment |
| 18 | iter16_feedcast | 1562 | use_your_judgment |
| 19 | iter18_oldcast | 1567 | use_your_judgment |
| 20 | iter18_oldcast | 1593 | use_your_judgment |
| 21 | iter16_feedcast | 1590 | use_your_judgment |


## Deep Focus Evolution Timing Analysis
- **Iterations with evolution**: 19/21
- **Total evolution time**: 509.7 minutes
- **Average evolution time**: 26.8 minutes

### Time Breakdown by Evolution Phase

| Phase | Avg Time | % of Evolution |
|-------|----------|----------------|
| First Draft (Round 1) | 26.0m | 97.0% |
| Reflection | 0.8m | 3.0% |
| **Total** | **26.8m** | **100.0%** |

### Evolution vs Test vs Meta-Evolution Time

| Metric | Time | Percentage | Avg/Iter | Avg/Occurrence |
|--------|------|------------|----------|----------------|
| Total Evolution Time | 509.7m | 34.7% | 24.3m | 26.8m |
| Total Test Time | 960.6m | 65.3% | 45.7m | 45.7m |
| Total Meta-Evolution Time | 0.0m | 0.0% | 0.0m | 0.0m |
| **Total** | **1470.3m** | **100.0%** | **70.0m** | **-** |

- **Evolution/Test ratio**: 0.53x
  - For every minute of testing, 0.53 minutes spent on evolution

## Claude CLI Usage & Cost Analysis
- **Total Cost**: $240.29
  - **Evaluation Cost**: $29.11
  - **Evolution CLI Cost**: $203.74
  - **Other (evaluator overhead) Cost**: $7.43
- **Total CLI Calls**: 37
- **Total CLI Input Tokens**: 9,518
- **Total CLI Output Tokens**: 380,390
- **Total Cache Created**: 1,015,700
- **Total Cache Read**: 33,218,414

### Cost Breakdown

| Operation | Cost | % of Total | Calls | Tokens In | Tokens Out |
|-----------|------|------------|-------|-----------|------------|
| Evaluation | $29.11 | 12.1% | - | - | - |
| Evolution (CLI) | $203.74 | 84.8% | 37 | 9,518 | 380,390 |
| Other (evaluator overhead) | $7.43 | 3.1% | - | - | - |
| **Total** | **$240.29** | **100%** | **37** | **9,518** | **380,390** |

**Answer: Evolution drives 84.8% of total costs**

### Detailed Per-Iteration Costs

| Iter | Total | **Eval** | **Evo** | Iter Eval | 1st Draft | Refl | Meta | Other | Fresh | Strategy | Meta-strategy |
|------|------|------|------|------|------|------|------|------|------|------|------|
| **1** | $0.02 | **$0.01** | **-** | $0.01 | - | - | - | $0.01 | 14 | - | - |
| **2** | $10.36 | **$0.33** | **$9.86** | $0.33 | $4.81 | $5.05 | - | $0.18 | 24 | use_your_judgment | - |
| **3** | $11.66 | **$0.68** | **$10.72** | $0.68 | $5.23 | $5.48 | - | $0.27 | 30 | use_your_judgment | - |
| **4** | $2.28 | **$1.29** | **$0.35** | $1.29 | - | $0.35 | - | $0.63 | 34 | use_your_judgment | - |
| **5** | $11.11 | **$1.08** | **$9.78** | $1.08 | $4.78 | $5.00 | - | $0.25 | 33 | use_your_judgment | - |
| **6** | $10.62 | **$1.40** | **$8.76** | $1.40 | $4.25 | $4.51 | - | $0.46 | 30 | use_your_judgment | - |
| **7** | $10.11 | **$1.63** | **$8.20** | $1.63 | $3.96 | $4.24 | - | $0.28 | 33 | use_your_judgment | - |
| **8** | $17.37 | **$1.89** | **$15.17** | $1.89 | $7.43 | $7.74 | - | $0.31 | 35 | use_your_judgment | - |
| **9** | $12.43 | **$1.69** | **$10.42** | $1.69 | $5.08 | $5.34 | - | $0.32 | 31 | use_your_judgment | - |
| **10** | $9.40 | **$1.18** | **$7.97** | $1.18 | $3.85 | $4.13 | - | $0.25 | 22 | use_your_judgment | - |
| **11** | $11.95 | **$2.07** | **$9.52** | $2.07 | $4.60 | $4.92 | - | $0.35 | 32 | use_your_judgment | - |
| **12** | $15.12 | **$1.43** | **$13.32** | $1.43 | $6.50 | $6.82 | - | $0.37 | 31 | use_your_judgment | - |
| **13** | $17.19 | **$1.17** | **$15.70** | $1.17 | $7.70 | $8.00 | - | $0.32 | 23 | use_your_judgment | - |
| **14** | $15.77 | **$1.51** | **$13.86** | $1.51 | $6.78 | $7.08 | - | $0.41 | 30 | use_your_judgment | - |
| **15** | $16.47 | **$1.49** | **$14.55** | $1.49 | $7.13 | $7.42 | - | $0.43 | 26 | use_your_judgment | - |
| **16** | $15.07 | **$2.10** | **$12.33** | $2.10 | $5.99 | $6.34 | - | $0.64 | 34 | use_your_judgment | - |
| **17** | $16.35 | **$1.56** | **$14.36** | $1.56 | $7.00 | $7.36 | - | $0.43 | 29 | use_your_judgment | - |
| **18** | $16.45 | **$2.23** | **$13.62** | $2.23 | $6.65 | $6.97 | - | $0.60 | 30 | use_your_judgment | - |
| **19** | $10.53 | **$1.98** | **$8.19** | $1.98 | $3.95 | $4.24 | - | $0.36 | 36 | use_your_judgment | - |
| **20** | $8.64 | **$1.28** | **$7.08** | $1.28 | $3.40 | $3.67 | - | $0.28 | 27 | use_your_judgment | - |
| **21** | $1.40 | **$1.11** | **-** | $1.11 | - | - | - | $0.29 | 20 | - | - |
|------|------|------|------|------|------|------|------|------|------|------|------|
| **TOTAL** | **$240.29** | **$29.11** | **$203.74** | $29.11 | $99.08 | $104.67 | $0.00 | **$7.43** | **604** | - | - |

### Evolution Cost Breakdown (Avg per evolution)

| Phase | Avg Cost | % of Evolution | Avg Tokens In | Avg Tokens Out |
|-------|----------|----------------|---------------|----------------|
| First Draft (Round 1) | $5.21 | 48.6% | 464 | 17369 |
| Reflection | $5.51 | 51.4% | 37 | 2652 |
| **Total** | **$10.72** | **100%** | **501** | **20021** |

### Cache Performance

- **Cache Creation**: 1,015,700 tokens
- **Cache Hits**: 33,218,414 tokens (97.0% hit rate)

## Zero Accuracy Cases
No zero accuracy cases encountered ✅

## Exception Failures
No exception failures encountered ✅

## Evolution Health

No artifact validation failures ✅