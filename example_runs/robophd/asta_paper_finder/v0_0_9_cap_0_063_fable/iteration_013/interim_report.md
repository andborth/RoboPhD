# Parallel Agent Research - Interim Report (Iteration 13)

## Experiment Configuration

**Run ID**: asta_paper_finder_20260804_143045
**Date**: 2026-08-05 03:01:27
**Iterations**: 13/999

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
- paper_finder_runtime: {'cost_threshold': 0.063, 'cost_per_error': 0.0063, 'cap_judge_to_estimate': True, 'training_judge': 'openai/gpt-5.6-luna', 'judge_prompt': 'no-prose', 'evidence_char_cap': 2500}

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
- initial_agents: ['seed_1rbu9viz']
- max_workers: 8
- meta_evolution_cadence: 3
- meta_evolution_domain: asta_paper_finder
- meta_evolution_first_iteration: 4
- meta_evolution_model: opus-5 (claude-opus-5)
- new_agent_test_rounds: 0

### Configuration Changes by Iteration

(No configuration changes across iterations)


## Winning Agents by Evolution Strategy

#### **Use Your Judgment** (12 wins - 92.3% of all wins)
- **iter3_criterion_evidence** (2 wins, created iteration 3)
- **iter6_graph_recall** (2 wins, created iteration 6)
- **iter10_deadline_guard** (2 wins, created iteration 10)
- **iter2_routed_deep_recall** (1 win, created iteration 2)
- **iter5_evidence_saturation** (1 win, created iteration 5)
- **iter8_criterion_saturation** (1 win, created iteration 8)
- **iter7_topk_fidelity** (1 win, created iteration 7)
- **iter12_salvage_rank** (1 win, created iteration 12)
- **iter13_any_author_gate** (1 win, created iteration 13)

#### **Initial** (1 wins - 7.7% of all wins)
- **seed_1rbu9viz** (1 win, created iteration 0)

#### Summary Statistics
- **Total iterations with winners**: 13
- **Total wins counted**: 13 (includes ties)
- **Unique winning agents**: 10
- **Evolution strategies that produced winners**: 2


## Complete Performance Ranking Table

| Agent | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 | Iter 6 | Iter 7 | Iter 8 | Iter 9 | Iter 10 | Iter 11 | Iter 12 | Iter 13 | Final Elo | Mean Rank |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|-----------|----------|
| iter10_deadline_guard | - | - | - | - | - | - | - | - | - | **#1** 42.653 | **#1** 43.533 | #2 32.339 | #2 41.839 | **1556** | **1.50** |
| iter13_any_author_gate | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 42.398 | 1535 | **1.00** |
| iter6_graph_recall | - | - | - | - | - | **#1** 31.910 | **#1** 47.677 | #2 39.081 | - | - | #3 38.379 | - | - | 1522 | **1.75** |
| iter11_mining_first | - | - | - | - | - | - | - | - | - | - | #2 38.794 | - | - | 1504 | 2.00 |
| iter12_salvage_rank | - | - | - | - | - | - | - | - | - | - | - | **#1** 33.786 | #3 41.741 | 1504 | 2.00 |
| iter7_topk_fidelity | - | - | - | - | - | - | #2 44.565 | - | **#1** 38.582 | #2 38.200 | - | #3 31.338 | - | 1501 | 2.00 |
| iter8_criterion_saturation | - | - | - | - | - | - | - | **#1** 45.751 | #3 23.255 | - | - | - | - | 1500 | 2.00 |
| iter5_evidence_saturation | - | - | - | - | **#1** 38.219 | #3 29.277 | - | - | - | - | - | - | - | 1500 | 2.00 |
| iter3_criterion_evidence | - | - | **#1** 36.829 | **#1** 31.985 | #3 35.740 | - | #3 34.645 | - | - | - | - | - | - | 1492 | 2.00 |
| iter2_routed_deep_recall | - | **#1** 27.119 | #2 32.422 | #3 24.485 | - | - | - | - | - | - | - | - | - | 1483 | 2.00 |
| iter4_deep_window | - | - | - | #2 31.360 | #2 37.831 | #2 30.457 | - | #3 38.607 | - | - | - | - | - | 1477 | 2.25 |
| iter9_robust_recall | - | - | - | - | - | - | - | - | #2 37.055 | #3 30.615 | - | - | - | 1472 | 2.50 |
| seed_1rbu9viz | **#1** 5.201 | #2 4.246 | #3 3.761 | - | - | - | - | - | - | - | - | - | - | 1455 | 2.50 |

### Legend:
- **#1** = 1st place (winner of iteration)
- #2, #3, etc. = 2nd, 3rd place, etc.
- Score = Average score (0-1) on that iteration's problems
- **Bold Elo/Rank** = Top performer
- `-` = Agent not tested in that iteration



### Quick Summary

| Agent | Elo | Mean Score | Mean Cost | Tests |
|-------|-----|------------|-----------|-------|
| iter10_deadline_guard | 1556 | 40.091 | $0.062 | 4 |
| iter13_any_author_gate | 1535 | 42.398 | $0.060 | 1 |
| iter6_graph_recall | 1522 | 39.262 | $0.043 | 4 |
| iter11_mining_first | 1504 | 38.794 | $0.057 | 1 |
| iter12_salvage_rank | 1504 | 37.763 | $0.057 | 2 |
| iter7_topk_fidelity | 1501 | 38.171 | $0.049 | 4 |
| iter8_criterion_saturation | 1500 | 34.503 | $0.038 | 2 |
| iter5_evidence_saturation | 1500 | 33.748 | $0.041 | 2 |
| iter3_criterion_evidence | 1492 | 34.800 | $0.034 | 4 |
| iter2_routed_deep_recall | 1483 | 28.009 | $0.028 | 3 |
| iter4_deep_window | 1477 | 34.564 | $0.038 | 4 |
| iter9_robust_recall | 1472 | 33.835 | $0.023 | 2 |
| seed_1rbu9viz | 1455 | 4.402 | $0.001 | 3 |

### Iteration Progression

| Iter | Winner(s) | Score | Evo Time | Test Time | Total Time |
|------|-----------|-------|----------|-----------|------------|
| 1 | seed_1rbu9viz | 5.201 | - | 1.9m | 1.9m |
| 2 | iter2_routed_deep_recall | 27.119 | 11.0m | 6.4m | 17.4m |
| 3 | iter3_criterion_evidence | 36.829 | 11.9m | 17.0m | 28.9m |
| 4 | iter3_criterion_evidence | 31.985 | 15.4m | 24.1m | 39.5m |
| 5 | iter5_evidence_saturation | 38.219 | 10.6m | 24.6m | 35.2m |
| 6 | iter6_graph_recall | 31.910 | 12.9m | 51.3m | 64.2m |
| 7 | iter6_graph_recall | 47.677 | 15.1m | 24.3m | 39.4m |
| 8 | iter8_criterion_saturation | 45.751 | 10.7m | 33.2m | 43.8m |
| 9 | iter7_topk_fidelity | 38.582 | 18.1m | 67.3m | 85.4m |
| 10 | iter10_deadline_guard | 42.653 | 11.1m | 81.9m | 93.0m |
| 11 | iter10_deadline_guard | 43.533 | 17.1m | 75.5m | 92.5m |
| 12 | iter12_salvage_rank | 33.786 | 10.8m | 75.3m | 86.0m |
| 13 | iter13_any_author_gate | 42.398 | 12.9m | 110.4m | 123.4m |

## Best Agent: iter10_deadline_guard
- Elo Score: 1556
- Mean Score: 40.091


## Elo Leadership Progression

Iteration-by-iteration Elo leaders:

| Iteration | Leader | Elo | Evolution Strategy |
|-----------|--------|-----|-------------------|
| 1 | seed_1rbu9viz | 1500 | Initial |
| 2 | iter2_routed_deep_recall | 1516 | use_your_judgment |
| 3 | iter3_criterion_evidence | 1531 | use_your_judgment |
| 4 | iter3_criterion_evidence | 1560 | use_your_judgment |
| 5 | iter5_evidence_saturation | 1534 | use_your_judgment |
| 6 | iter6_graph_recall | 1533 | use_your_judgment |
| 7 | iter6_graph_recall | 1562 | use_your_judgment |
| 8 | iter6_graph_recall | 1557 | use_your_judgment |
| 9 | iter6_graph_recall | 1557 | use_your_judgment |
| 10 | iter6_graph_recall | 1557 | use_your_judgment |
| 11 | iter10_deadline_guard | 1564 | use_your_judgment |
| 12 | iter10_deadline_guard | 1560 | use_your_judgment |
| 13 | iter10_deadline_guard | 1556 | use_your_judgment |


## Deep Focus Evolution Timing Analysis
- **Iterations with evolution**: 12/13
- **Total evolution time**: 157.5 minutes
- **Average evolution time**: 13.1 minutes

### Time Breakdown by Evolution Phase

| Phase | Avg Time | % of Evolution |
|-------|----------|----------------|
| First Draft (Round 1) | 12.2m | 93.3% |
| Reflection | 0.9m | 6.7% |
| **Total** | **13.1m** | **100.0%** |

### Evolution vs Test vs Meta-Evolution Time

| Metric | Time | Percentage | Avg/Iter | Avg/Occurrence |
|--------|------|------------|----------|----------------|
| Total Evolution Time | 157.5m | 21.0% | 12.1m | 13.1m |
| Total Test Time | 593.2m | 79.0% | 45.6m | 45.6m |
| Total Meta-Evolution Time | 0.0m | 0.0% | 0.0m | 0.0m |
| **Total** | **750.7m** | **100.0%** | **57.7m** | **-** |

- **Evolution/Test ratio**: 0.27x
  - For every minute of testing, 0.27 minutes spent on evolution

## Claude CLI Usage & Cost Analysis
- **Total Cost**: $119.42
  - **Evaluation Cost**: $16.10
  - **Evolution CLI Cost**: $97.07
  - **Other (evaluator overhead) Cost**: $6.25
- **Total CLI Calls**: 24
- **Total CLI Input Tokens**: 6,924
- **Total CLI Output Tokens**: 570,630
- **Total Cache Created**: 1,818,242
- **Total Cache Read**: 40,949,441

### Cost Breakdown

| Operation | Cost | % of Total | Calls | Tokens In | Tokens Out |
|-----------|------|------------|-------|-----------|------------|
| Evaluation | $16.10 | 13.5% | - | - | - |
| Evolution (CLI) | $97.07 | 81.3% | 24 | 6,924 | 570,630 |
| Other (evaluator overhead) | $6.25 | 5.2% | - | - | - |
| **Total** | **$119.42** | **100%** | **24** | **6,924** | **570,630** |

**Answer: Evolution drives 81.3% of total costs**

### Detailed Per-Iteration Costs

| Iter | Total | **Eval** | **Evo** | Iter Eval | 1st Draft | Refl | Meta | Other | Fresh | Strategy | Meta-strategy |
|------|------|------|------|------|------|------|------|------|------|------|------|
| **1** | $0.02 | **$0.01** | **-** | $0.01 | - | - | - | $0.01 | 14 | - | - |
| **2** | $6.78 | **$0.45** | **$6.14** | $0.45 | $4.44 | $1.70 | - | $0.19 | 24 | use_your_judgment | - |
| **3** | $7.06 | **$0.80** | **$5.96** | $0.80 | $5.50 | $0.46 | - | $0.31 | 36 | use_your_judgment | - |
| **4** | $9.57 | **$1.16** | **$7.83** | $1.16 | $6.20 | $1.62 | - | $0.58 | 34 | use_your_judgment | - |
| **5** | $8.32 | **$1.24** | **$6.45** | $1.24 | $4.32 | $2.13 | - | $0.63 | 32 | use_your_judgment | - |
| **6** | $10.35 | **$1.51** | **$8.05** | $1.51 | $7.56 | $0.49 | - | $0.79 | 38 | use_your_judgment | - |
| **7** | $12.56 | **$1.40** | **$10.55** | $1.40 | $10.00 | $0.55 | - | $0.61 | 34 | use_your_judgment | - |
| **8** | $10.42 | **$1.08** | **$8.71** | $1.08 | $8.20 | $0.51 | - | $0.62 | 26 | use_your_judgment | - |
| **9** | $11.46 | **$1.13** | **$9.96** | $1.13 | $9.45 | $0.51 | - | $0.36 | 36 | use_your_judgment | - |
| **10** | $11.90 | **$1.62** | **$9.93** | $1.62 | $9.41 | $0.52 | - | $0.35 | 29 | use_your_judgment | - |
| **11** | $9.84 | **$1.90** | **$7.56** | $1.90 | $7.05 | $0.51 | - | $0.38 | 34 | use_your_judgment | - |
| **12** | $9.52 | **$1.54** | **$7.33** | $1.54 | $6.89 | $0.44 | - | $0.66 | 27 | use_your_judgment | - |
| **13** | $11.62 | **$2.27** | **$8.60** | $2.27 | $8.08 | $0.52 | - | $0.76 | 38 | use_your_judgment | - |
|------|------|------|------|------|------|------|------|------|------|------|------|
| **TOTAL** | **$119.42** | **$16.10** | **$97.07** | $16.10 | $87.11 | $9.96 | $0.00 | **$6.25** | **402** | - | - |

### Evolution Cost Breakdown (Avg per evolution)

| Phase | Avg Cost | % of Evolution | Avg Tokens In | Avg Tokens Out |
|-------|----------|----------------|---------------|----------------|
| First Draft (Round 1) | $7.26 | 89.7% | 573 | 44437 |
| Reflection | $0.83 | 10.3% | 4 | 3115 |
| **Total** | **$8.09** | **100%** | **577** | **47552** |

### Cache Performance

- **Cache Creation**: 1,818,242 tokens
- **Cache Hits**: 40,949,441 tokens (95.7% hit rate)

## Zero Accuracy Cases
No zero accuracy cases encountered ✅

## Exception Failures
No exception failures encountered ✅

## Evolution Health

No artifact validation failures ✅