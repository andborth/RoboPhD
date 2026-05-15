# Parallel Agent Research - Interim Report (Iteration 3)

## Experiment Configuration

**Run ID**: asta_ds1000_20260514_231614
**Date**: 2026-05-15 01:14:39
**Iterations**: 3/15

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

#### **Initial** (2 wins - 66.7% of all wins)
- **seed_yyg6m9ud** (2 wins, created iteration 0)

#### **Use Your Judgment** (1 wins - 33.3% of all wins)
- **iter3_ds1000_format_aware** (1 win, created iteration 3)

#### Summary Statistics
- **Total iterations with winners**: 3
- **Total wins counted**: 3 (includes ties)
- **Unique winning agents**: 2
- **Evolution strategies that produced winners**: 2


## Complete Performance Ranking Table

| Agent | Iter 1 | Iter 2 | Iter 3 | Final ELO | Mean Rank |
|-------|--------|--------|--------|-----------|----------|
| iter3_ds1000_format_aware | - | - | **#1** 80.000 | **1531** | **1.00** |
| iter2_ds1000_verify_repair | - | #2 75.000 | #2 75.000 | 1486 | 2.00 |
| seed_yyg6m9ud | **#1** 70.000 | **#1** 80.000 | #3 65.000 | 1483 | 2.00 |

### Legend:
- **#1** = 1st place (winner of iteration)
- #2, #3, etc. = 2nd, 3rd place, etc.
- Score = Average score (0-1) on that iteration's problems
- **Bold ELO/Rank** = Top performer
- `-` = Agent not tested in that iteration



### Quick Summary

| Agent | ELO | Mean Score | Tests |
|-------|-----|------------|-------|
| iter3_ds1000_format_aware | 1531 | 80.000 | 1 |
| iter2_ds1000_verify_repair | 1486 | 75.000 | 2 |
| seed_yyg6m9ud | 1483 | 71.667 | 3 |

### Iteration Progression

| Iter | Winner(s) | Score | Evo Time | Test Time | Total Time |
|------|-----------|-------|----------|-----------|------------|
| 1 | seed_yyg6m9ud | 70.000 | - | 15.2m | 15.2m |
| 2 | seed_yyg6m9ud | 80.000 | 4.3m | 33.0m | 37.3m |
| 3 | iter3_ds1000_format_aware | 80.000 | 12.4m | 53.4m | 65.9m |

## Best Agent: iter3_ds1000_format_aware
- ELO Score: 1531
- Mean Score: 80.000


## ELO Leadership Progression

Iteration-by-iteration ELO leaders:

| Iteration | Leader | ELO | Evolution Strategy |
|-----------|--------|-----|-------------------|
| 1 | seed_yyg6m9ud | 1500 | Initial |
| 2 | seed_yyg6m9ud | 1516 | Initial |
| 3 | iter3_ds1000_format_aware | 1531 | use_your_judgment |


## Deep Focus Evolution Timing Analysis
- **Iterations with evolution**: 2/3
- **Total evolution time**: 16.7 minutes
- **Average evolution time**: 8.4 minutes

### Time Breakdown by Evolution Phase

| Phase | Avg Time | % of Evolution |
|-------|----------|----------------|
| First Draft (Round 1) | 7.7m | 92.3% |
| Reflection | 0.6m | 7.7% |
| **Total** | **8.4m** | **100.0%** |

### Evolution vs Test vs Meta-Evolution Time

| Metric | Time | Percentage | Avg/Iter | Avg/Occurrence |
|--------|------|------------|----------|----------------|
| Total Evolution Time | 16.7m | 14.1% | 5.6m | 8.4m |
| Total Test Time | 101.7m | 85.9% | 33.9m | 33.9m |
| Total Meta-Evolution Time | 0.0m | 0.0% | 0.0m | 0.0m |
| **Total** | **118.4m** | **100.0%** | **39.5m** | **-** |

- **Evolution/Test ratio**: 0.16x
  - For every minute of testing, 0.16 minutes spent on evolution

## Claude CLI Usage & Cost Analysis
- **Total Cost**: $4.56
  - **Evaluation Cost**: $0.46
  - **Evolution CLI Cost**: $4.10
- **Total CLI Calls**: 4
- **Total CLI Input Tokens**: 820
- **Total CLI Output Tokens**: 61,859
- **Total Cache Created**: 206,606
- **Total Cache Read**: 2,506,026

### Cost Breakdown

| Operation | Cost | % of Total | Calls | Tokens In | Tokens Out |
|-----------|------|------------|-------|-----------|------------|
| Evaluation | $0.46 | 10.0% | - | - | - |
| Evolution (CLI) | $4.10 | 90.0% | 4 | 820 | 61,859 |
| **Total** | **$4.56** | **100%** | **4** | **820** | **61,859** |

**Answer: Evolution drives 90.0% of total costs**

### Detailed Per-Iteration Costs

| Iter | Total | **Eval** | **Evo** | Iter Eval | 1st Draft | Refl | Meta | Fresh | Strategy | Meta-strategy |
|------|------|------|------|------|------|------|------|------|------|------|
| **1** | $0.01 | **$0.01** | **-** | $0.01 | - | - | - | 20 | - | - |
| **2** | $1.00 | **$0.18** | **$0.81** | $0.18 | $0.70 | $0.11 | - | 36 | use_your_judgment | - |
| **3** | $3.55 | **$0.26** | **$3.28** | $0.26 | $2.65 | $0.64 | - | 45 | use_your_judgment | - |
|------|------|------|------|------|------|------|------|------|------|------|
| **TOTAL** | **$4.56** | **$0.46** | **$4.10** | $0.46 | $3.35 | $0.75 | $0.00 | **101** | - | - |

### Evolution Cost Breakdown (Avg per evolution)

| Phase | Avg Cost | % of Evolution | Avg Tokens In | Avg Tokens Out |
|-------|----------|----------------|---------------|----------------|
| First Draft (Round 1) | $1.68 | 81.8% | 407 | 28926 |
| Reflection | $0.37 | 18.2% | 3 | 2004 |
| **Total** | **$2.05** | **100%** | **410** | **30930** |

### Cache Performance

- **Cache Creation**: 206,606 tokens
- **Cache Hits**: 2,506,026 tokens (92.4% hit rate)

## Zero Accuracy Cases
No zero accuracy cases encountered ✅

## Exception Failures
No exception failures encountered ✅

## Evolution Health

No artifact validation failures ✅