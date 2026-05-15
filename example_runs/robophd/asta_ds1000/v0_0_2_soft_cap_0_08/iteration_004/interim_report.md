# Parallel Agent Research - Interim Report (Iteration 4)

## Experiment Configuration

**Run ID**: asta_ds1000_20260514_231614
**Date**: 2026-05-15 01:32:02
**Iterations**: 4/15

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
- evolution_model: opus-4.7 (claude-opus-4-7)
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
- meta_evolution_model: opus-4.7 (claude-opus-4-7)
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
- eval_timeout: 1200
- evaluation_budget: 999999
- evolution_model: opus-4.7 (claude-opus-4-7)
- evolution_strategy: use_your_judgment
- evolution_timeout: 3600
- examples_per_iteration: 20
- initial_agents: ['seed_yyg6m9ud']
- max_workers: 6
- meta_evolution_cadence: 3
- meta_evolution_domain: asta_ds1000
- meta_evolution_first_iteration: 4
- new_agent_test_rounds: 0

### Configuration Changes by Iteration

(No configuration changes across iterations)


## Winning Agents by Evolution Strategy

#### **Initial** (2 wins - 50.0% of all wins)
- **seed_yyg6m9ud** (2 wins, created iteration 0)

#### **Use Your Judgment** (2 wins - 50.0% of all wins)
- **iter3_ds1000_format_aware** (1 win, created iteration 3)
- **iter4_ds1000_idiom_probe** (1 win, created iteration 4)

#### Summary Statistics
- **Total iterations with winners**: 4
- **Total wins counted**: 4 (includes ties)
- **Unique winning agents**: 3
- **Evolution strategies that produced winners**: 2


## Complete Performance Ranking Table

| Agent | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Final ELO | Mean Rank |
|-------|--------|--------|--------|--------|-----------|----------|
| iter4_ds1000_idiom_probe | - | - | - | **#1** 100.000 | **1532** | **1.00** |
| iter3_ds1000_format_aware | - | - | **#1** 80.000 | #2 95.000 | 1512 | **1.50** |
| seed_yyg6m9ud | **#1** 70.000 | **#1** 80.000 | #3 65.000 | - | 1483 | 2.00 |
| iter2_ds1000_verify_repair | - | #2 75.000 | #2 75.000 | #2 95.000 | 1474 | 2.00 |

### Legend:
- **#1** = 1st place (winner of iteration)
- #2, #3, etc. = 2nd, 3rd place, etc.
- Score = Average score (0-1) on that iteration's problems
- **Bold ELO/Rank** = Top performer
- `-` = Agent not tested in that iteration



### Quick Summary

| Agent | ELO | Mean Score | Tests |
|-------|-----|------------|-------|
| iter4_ds1000_idiom_probe | 1532 | 100.000 | 1 |
| iter3_ds1000_format_aware | 1512 | 87.500 | 2 |
| seed_yyg6m9ud | 1483 | 71.667 | 3 |
| iter2_ds1000_verify_repair | 1474 | 81.667 | 3 |

### Iteration Progression

| Iter | Winner(s) | Score | Evo Time | Test Time | Total Time |
|------|-----------|-------|----------|-----------|------------|
| 1 | seed_yyg6m9ud | 70.000 | - | 15.2m | 15.2m |
| 2 | seed_yyg6m9ud | 80.000 | 4.3m | 33.0m | 37.3m |
| 3 | iter3_ds1000_format_aware | 80.000 | 12.4m | 53.4m | 65.9m |
| 4 | iter4_ds1000_idiom_probe | 100.000 | 8.2m | 9.2m | 17.4m |

## Best Agent: iter4_ds1000_idiom_probe
- ELO Score: 1532
- Mean Score: 100.000


## ELO Leadership Progression

Iteration-by-iteration ELO leaders:

| Iteration | Leader | ELO | Evolution Strategy |
|-----------|--------|-----|-------------------|
| 1 | seed_yyg6m9ud | 1500 | Initial |
| 2 | seed_yyg6m9ud | 1516 | Initial |
| 3 | iter3_ds1000_format_aware | 1531 | use_your_judgment |
| 4 | iter4_ds1000_idiom_probe | 1532 | use_your_judgment |


## Deep Focus Evolution Timing Analysis
- **Iterations with evolution**: 3/4
- **Total evolution time**: 25.0 minutes
- **Average evolution time**: 8.3 minutes

### Time Breakdown by Evolution Phase

| Phase | Avg Time | % of Evolution |
|-------|----------|----------------|
| First Draft (Round 1) | 7.7m | 92.2% |
| Reflection | 0.7m | 7.8% |
| **Total** | **8.3m** | **100.0%** |

### Evolution vs Test vs Meta-Evolution Time

| Metric | Time | Percentage | Avg/Iter | Avg/Occurrence |
|--------|------|------------|----------|----------------|
| Total Evolution Time | 25.0m | 18.4% | 6.2m | 8.3m |
| Total Test Time | 110.8m | 81.6% | 27.7m | 27.7m |
| Total Meta-Evolution Time | 0.0m | 0.0% | 0.0m | 0.0m |
| **Total** | **135.8m** | **100.0%** | **34.0m** | **-** |

- **Evolution/Test ratio**: 0.23x
  - For every minute of testing, 0.23 minutes spent on evolution

## Claude CLI Usage & Cost Analysis
- **Total Cost**: $6.63
  - **Evaluation Cost**: $0.96
  - **Evolution CLI Cost**: $5.66
- **Total CLI Calls**: 6
- **Total CLI Input Tokens**: 1,756
- **Total CLI Output Tokens**: 92,205
- **Total Cache Created**: 267,970
- **Total Cache Read**: 3,340,446

### Cost Breakdown

| Operation | Cost | % of Total | Calls | Tokens In | Tokens Out |
|-----------|------|------------|-------|-----------|------------|
| Evaluation | $0.96 | 14.5% | - | - | - |
| Evolution (CLI) | $5.66 | 85.5% | 6 | 1,756 | 92,205 |
| **Total** | **$6.63** | **100%** | **6** | **1,756** | **92,205** |

**Answer: Evolution drives 85.5% of total costs**

### Detailed Per-Iteration Costs

| Iter | Total | **Eval** | **Evo** | Iter Eval | 1st Draft | Refl | Meta | Fresh | Strategy | Meta-strategy |
|------|------|------|------|------|------|------|------|------|------|------|
| **1** | $0.01 | **$0.01** | **-** | $0.01 | - | - | - | 20 | - | - |
| **2** | $1.00 | **$0.18** | **$0.81** | $0.18 | $0.70 | $0.11 | - | 36 | use_your_judgment | - |
| **3** | $3.55 | **$0.26** | **$3.28** | $0.26 | $2.65 | $0.64 | - | 45 | use_your_judgment | - |
| **4** | $2.07 | **$0.51** | **$1.57** | $0.51 | $1.42 | $0.14 | - | 52 | use_your_judgment | - |
|------|------|------|------|------|------|------|------|------|------|------|
| **TOTAL** | **$6.63** | **$0.96** | **$5.66** | $0.96 | $4.77 | $0.89 | $0.00 | **153** | - | - |

### Evolution Cost Breakdown (Avg per evolution)

| Phase | Avg Cost | % of Evolution | Avg Tokens In | Avg Tokens Out |
|-------|----------|----------------|---------------|----------------|
| First Draft (Round 1) | $1.59 | 84.3% | 582 | 28697 |
| Reflection | $0.30 | 15.7% | 3 | 2038 |
| **Total** | **$1.89** | **100%** | **585** | **30735** |

### Cache Performance

- **Cache Creation**: 267,970 tokens
- **Cache Hits**: 3,340,446 tokens (92.6% hit rate)

## Zero Accuracy Cases
No zero accuracy cases encountered ✅

## Exception Failures
No exception failures encountered ✅

## Evolution Health

No artifact validation failures ✅