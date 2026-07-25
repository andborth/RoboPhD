# Parallel Agent Research - Interim Report (Iteration 8)

## Experiment Configuration

**Run ID**: asta_paper_finder_20260724_004103
**Date**: 2026-07-24 05:30:03
**Iterations**: 8/999

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

#### **Use Your Judgment** (7 wins - 87.5% of all wins)
- **iter6_metadata_targeted_v1** (3 wins, created iteration 6)
- **iter3_evidence_rerank_v1** (2 wins, created iteration 3)
- **iter2_recall_max_v1** (1 win, created iteration 2)
- **iter4_facet_evidence_v1** (1 win, created iteration 4)

#### **Initial** (1 wins - 12.5% of all wins)
- **seed_k_udbj7b** (1 win, created iteration 0)

#### Summary Statistics
- **Total iterations with winners**: 8
- **Total wins counted**: 8 (includes ties)
- **Unique winning agents**: 5
- **Evolution strategies that produced winners**: 2


## Complete Performance Ranking Table

| Agent | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 | Iter 6 | Iter 7 | Iter 8 | Final Elo | Mean Rank |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|-----------|----------|
| iter6_metadata_targeted_v1 | - | - | - | - | - | **#1** 20.362 | **#1** 26.484 | **#1** 20.566 | **1589** | **1.00** |
| iter3_evidence_rerank_v1 | - | - | **#1** 31.614 | **#1** 18.589 | #3 15.393 | #2 13.364 | #2 21.860 | - | 1523 | **1.80** |
| iter5_grade3_rerank_v1 | - | - | - | - | #2 18.859 | - | - | #2 17.801 | 1505 | 2.00 |
| iter4_facet_evidence_v1 | - | - | - | #2 15.901 | **#1** 23.767 | #3 12.590 | - | - | 1502 | 2.00 |
| iter2_recall_max_v1 | - | **#1** 13.275 | #2 23.740 | #3 13.929 | - | - | - | - | 1483 | 2.00 |
| iter8_grade3_probe_v1 | - | - | - | - | - | - | - | #3 17.567 | 1472 | 3.00 |
| iter7_grade3_evidence_v1 | - | - | - | - | - | - | #3 16.146 | - | 1471 | 3.00 |
| seed_k_udbj7b | **#1** 7.622 | #2 7.259 | #3 4.680 | - | - | - | - | - | 1455 | 2.50 |

### Legend:
- **#1** = 1st place (winner of iteration)
- #2, #3, etc. = 2nd, 3rd place, etc.
- Score = Average score (0-1) on that iteration's problems
- **Bold Elo/Rank** = Top performer
- `-` = Agent not tested in that iteration



### Quick Summary

| Agent | Elo | Mean Score | Mean Cost | Tests |
|-------|-----|------------|-----------|-------|
| iter6_metadata_targeted_v1 | 1589 | 22.471 | $0.005 | 3 |
| iter3_evidence_rerank_v1 | 1523 | 20.164 | $0.003 | 5 |
| iter5_grade3_rerank_v1 | 1505 | 18.330 | $0.015 | 2 |
| iter4_facet_evidence_v1 | 1502 | 17.419 | $0.005 | 3 |
| iter2_recall_max_v1 | 1483 | 16.981 | $0.000 | 3 |
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

## Best Agent: iter6_metadata_targeted_v1
- Elo Score: 1589
- Mean Score: 22.471


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


## Deep Focus Evolution Timing Analysis
- **Iterations with evolution**: 7/8
- **Total evolution time**: 60.4 minutes
- **Average evolution time**: 8.6 minutes

### Time Breakdown by Evolution Phase

| Phase | Avg Time | % of Evolution |
|-------|----------|----------------|
| First Draft (Round 1) | 7.8m | 90.6% |
| Reflection | 0.8m | 9.4% |
| **Total** | **8.6m** | **100.0%** |

### Evolution vs Test vs Meta-Evolution Time

| Metric | Time | Percentage | Avg/Iter | Avg/Occurrence |
|--------|------|------------|----------|----------------|
| Total Evolution Time | 60.4m | 20.9% | 7.6m | 8.6m |
| Total Test Time | 228.6m | 79.1% | 28.6m | 28.6m |
| Total Meta-Evolution Time | 0.0m | 0.0% | 0.0m | 0.0m |
| **Total** | **289.0m** | **100.0%** | **36.1m** | **-** |

- **Evolution/Test ratio**: 0.26x
  - For every minute of testing, 0.26 minutes spent on evolution

## Claude CLI Usage & Cost Analysis
- **Total Cost**: $36.27
  - **Evaluation Cost**: $0.97
  - **Evolution CLI Cost**: $19.66
  - **Other (evaluator overhead) Cost**: $15.65
- **Total CLI Calls**: 14
- **Total CLI Input Tokens**: 8,008
- **Total CLI Output Tokens**: 260,006
- **Total Cache Created**: 786,909
- **Total Cache Read**: 10,478,717

### Cost Breakdown

| Operation | Cost | % of Total | Calls | Tokens In | Tokens Out |
|-----------|------|------------|-------|-----------|------------|
| Evaluation | $0.97 | 2.7% | - | - | - |
| Evolution (CLI) | $19.66 | 54.2% | 14 | 8,008 | 260,006 |
| Other (evaluator overhead) | $15.65 | 43.1% | - | - | - |
| **Total** | **$36.27** | **100%** | **14** | **8,008** | **260,006** |

**Answer: Evolution drives 54.2% of total costs**

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
|------|------|------|------|------|------|------|------|------|------|------|------|
| **TOTAL** | **$36.27** | **$0.97** | **$19.66** | $0.97 | $15.89 | $3.77 | $0.00 | **$15.65** | **232** | - | - |

### Evolution Cost Breakdown (Avg per evolution)

| Phase | Avg Cost | % of Evolution | Avg Tokens In | Avg Tokens Out |
|-------|----------|----------------|---------------|----------------|
| First Draft (Round 1) | $2.27 | 80.8% | 1140 | 34232 |
| Reflection | $0.54 | 19.2% | 4 | 2911 |
| **Total** | **$2.81** | **100%** | **1144** | **37144** |

### Cache Performance

- **Cache Creation**: 786,909 tokens
- **Cache Hits**: 10,478,717 tokens (93.0% hit rate)

## Zero Accuracy Cases
No zero accuracy cases encountered ✅

## Exception Failures
No exception failures encountered ✅

## Evolution Health

No artifact validation failures ✅