# Parallel Agent Research - Interim Report (Iteration 14)

## Experiment Configuration

**Run ID**: asta_paper_finder_20260730_031141
**Date**: 2026-07-30 13:46:13
**Iterations**: 14/999

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
- evolution_model: opus-5 (claude-opus-5)
- evolution_strategy: use_your_judgment
- evolution_timeout: 3600
- examples_per_iteration: 14
- initial_agents: ['seed_qh7zljny']
- max_workers: 8
- meta_evolution_cadence: 3
- meta_evolution_domain: asta_paper_finder
- meta_evolution_first_iteration: 4
- meta_evolution_model: opus-5 (claude-opus-5)
- new_agent_test_rounds: 0

### Configuration Changes by Iteration

(No configuration changes across iterations)


## Winning Agents by Evolution Strategy

#### **Use Your Judgment** (13 wins - 92.9% of all wins)
- **iter4_judge_aligned_rank** (2 wins, created iteration 4)
- **iter6_structured_metadata** (2 wins, created iteration 6)
- **iter7_criterion_repair** (2 wins, created iteration 7)
- **iter11_ensemble_conjunctive_rank** (2 wins, created iteration 11)
- **iter13_balanced_digest_wide_vetting** (2 wins, created iteration 13)
- **iter2_triage_rrf_deepevidence** (1 win, created iteration 2)
- **iter3_criterion_evidence** (1 win, created iteration 3)
- **iter10_criterion_coverage** (1 win, created iteration 10)

#### **Initial** (1 wins - 7.1% of all wins)
- **seed_qh7zljny** (1 win, created iteration 0)

#### Summary Statistics
- **Total iterations with winners**: 14
- **Total wins counted**: 14 (includes ties)
- **Unique winning agents**: 9
- **Evolution strategies that produced winners**: 2


## Complete Performance Ranking Table

| Agent | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 | Iter 6 | Iter 7 | Iter 8 | Iter 9 | Iter 10 | Iter 11 | Iter 12 | Iter 13 | Iter 14 | Final Elo | Mean Rank |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|-----------|----------|
| iter13_balanced_digest_wide_vetting | - | - | - | - | - | - | - | - | - | - | - | - | **#1** 39.498 | **#1** 49.888 | **1564** | **1.00** |
| iter11_ensemble_conjunctive_rank | - | - | - | - | - | - | - | - | - | - | **#1** 30.741 | **#1** 54.844 | #3 36.732 | - | 1530 | **1.67** |
| iter7_criterion_repair | - | - | - | - | - | - | #2 43.440 | **#1** 47.351 | **#1** 47.239 | #2 47.770 | #2 29.436 | #3 48.861 | #2 36.835 | #2 48.689 | 1524 | **1.88** |
| iter12_balanced_digest_rank | - | - | - | - | - | - | - | - | - | - | - | #2 51.285 | - | - | 1504 | 2.00 |
| iter10_criterion_coverage | - | - | - | - | - | - | - | - | - | **#1** 49.199 | #3 27.385 | - | - | - | 1504 | 2.00 |
| iter8_conjunction_recall | - | - | - | - | - | - | - | #2 43.070 | - | - | - | - | - | - | 1503 | 2.00 |
| iter9_cue_anchored_specific | - | - | - | - | - | - | - | - | #2 46.596 | - | - | - | - | - | 1503 | 2.00 |
| iter3_criterion_evidence | - | - | **#1** 27.148 | #3 41.822 | - | - | - | - | - | - | - | - | - | - | 1498 | 2.00 |
| iter6_structured_metadata | - | - | - | - | - | **#1** 56.946 | **#1** 44.395 | #3 40.389 | #3 44.216 | - | - | - | - | - | 1496 | 2.00 |
| iter4_judge_aligned_rank | - | - | - | **#1** 47.764 | **#1** 29.371 | #2 44.380 | #3 40.566 | - | - | #3 41.696 | - | - | - | - | 1492 | 2.00 |
| iter2_triage_rrf_deepevidence | - | **#1** 28.914 | #2 23.199 | #2 43.113 | #2 26.431 | #3 43.678 | - | - | - | - | - | - | - | - | 1485 | 2.00 |
| iter14_conjunctive_vote | - | - | - | - | - | - | - | - | - | - | - | - | - | #3 48.226 | 1471 | 3.00 |
| iter5_evidence_repair | - | - | - | - | #3 26.016 | - | - | - | - | - | - | - | - | - | 1471 | 3.00 |
| seed_qh7zljny | **#1** 5.164 | #2 7.224 | #3 2.744 | - | - | - | - | - | - | - | - | - | - | - | 1455 | 2.50 |

### Legend:
- **#1** = 1st place (winner of iteration)
- #2, #3, etc. = 2nd, 3rd place, etc.
- Score = Average score (0-1) on that iteration's problems
- **Bold Elo/Rank** = Top performer
- `-` = Agent not tested in that iteration



### Quick Summary

| Agent | Elo | Mean Score | Mean Cost | Tests |
|-------|-----|------------|-----------|-------|
| iter13_balanced_digest_wide_vetting | 1564 | 44.693 | $0.052 | 2 |
| iter11_ensemble_conjunctive_rank | 1530 | 40.772 | $0.048 | 3 |
| iter7_criterion_repair | 1524 | 43.703 | $0.032 | 8 |
| iter12_balanced_digest_rank | 1504 | 51.285 | $0.044 | 1 |
| iter10_criterion_coverage | 1504 | 38.292 | $0.040 | 2 |
| iter8_conjunction_recall | 1503 | 43.070 | $0.027 | 1 |
| iter9_cue_anchored_specific | 1503 | 46.596 | $0.033 | 1 |
| iter3_criterion_evidence | 1498 | 34.485 | $0.020 | 2 |
| iter6_structured_metadata | 1496 | 46.486 | $0.022 | 4 |
| iter4_judge_aligned_rank | 1492 | 40.755 | $0.022 | 5 |
| iter2_triage_rrf_deepevidence | 1485 | 33.067 | $0.016 | 5 |
| iter14_conjunctive_vote | 1471 | 48.226 | $0.047 | 1 |
| iter5_evidence_repair | 1471 | 26.016 | $0.019 | 1 |
| seed_qh7zljny | 1455 | 5.044 | $0.001 | 3 |

### Iteration Progression

| Iter | Winner(s) | Score | Evo Time | Test Time | Total Time |
|------|-----------|-------|----------|-----------|------------|
| 1 | seed_qh7zljny | 5.164 | - | 0.9m | 0.9m |
| 2 | iter2_triage_rrf_deepevidence | 28.914 | 12.9m | 8.3m | 21.2m |
| 3 | iter3_criterion_evidence | 27.148 | 15.0m | 20.1m | 35.1m |
| 4 | iter4_judge_aligned_rank | 47.764 | 11.3m | 17.0m | 28.2m |
| 5 | iter4_judge_aligned_rank | 29.371 | 11.2m | 25.8m | 37.0m |
| 6 | iter6_structured_metadata | 56.946 | 11.6m | 24.6m | 36.3m |
| 7 | iter6_structured_metadata | 44.395 | 13.9m | 37.9m | 51.8m |
| 8 | iter7_criterion_repair | 47.351 | 12.1m | 40.2m | 52.4m |
| 9 | iter7_criterion_repair | 47.239 | 10.9m | 31.1m | 42.0m |
| 10 | iter10_criterion_coverage | 49.199 | 10.3m | 47.4m | 57.7m |
| 11 | iter11_ensemble_conjunctive_rank | 30.741 | 11.6m | 76.1m | 87.6m |
| 12 | iter11_ensemble_conjunctive_rank | 54.844 | 8.8m | 56.0m | 64.8m |
| 13 | iter13_balanced_digest_wide_vetting | 39.498 | 10.9m | 48.9m | 59.8m |
| 14 | iter13_balanced_digest_wide_vetting | 49.888 | 10.3m | 49.4m | 59.7m |

## Best Agent: iter13_balanced_digest_wide_vetting
- Elo Score: 1564
- Mean Score: 44.693


## Elo Leadership Progression

Iteration-by-iteration Elo leaders:

| Iteration | Leader | Elo | Evolution Strategy |
|-----------|--------|-----|-------------------|
| 1 | seed_qh7zljny | 1500 | Initial |
| 2 | iter2_triage_rrf_deepevidence | 1516 | use_your_judgment |
| 3 | iter3_criterion_evidence | 1531 | use_your_judgment |
| 4 | iter4_judge_aligned_rank | 1533 | use_your_judgment |
| 5 | iter4_judge_aligned_rank | 1562 | use_your_judgment |
| 6 | iter4_judge_aligned_rank | 1557 | use_your_judgment |
| 7 | iter6_structured_metadata | 1565 | use_your_judgment |
| 8 | iter7_criterion_repair | 1538 | use_your_judgment |
| 9 | iter7_criterion_repair | 1567 | use_your_judgment |
| 10 | iter7_criterion_repair | 1562 | use_your_judgment |
| 11 | iter7_criterion_repair | 1558 | use_your_judgment |
| 12 | iter11_ensemble_conjunctive_rank | 1566 | use_your_judgment |
| 13 | iter13_balanced_digest_wide_vetting | 1535 | use_your_judgment |
| 14 | iter13_balanced_digest_wide_vetting | 1564 | use_your_judgment |


## Deep Focus Evolution Timing Analysis
- **Iterations with evolution**: 13/14
- **Total evolution time**: 150.7 minutes
- **Average evolution time**: 11.6 minutes

### Time Breakdown by Evolution Phase

| Phase | Avg Time | % of Evolution |
|-------|----------|----------------|
| First Draft (Round 1) | 10.2m | 88.2% |
| Reflection | 1.4m | 11.8% |
| **Total** | **11.6m** | **100.0%** |

### Evolution vs Test vs Meta-Evolution Time

| Metric | Time | Percentage | Avg/Iter | Avg/Occurrence |
|--------|------|------------|----------|----------------|
| Total Evolution Time | 150.7m | 23.8% | 10.8m | 11.6m |
| Total Test Time | 483.7m | 76.2% | 34.6m | 34.6m |
| Total Meta-Evolution Time | 0.0m | 0.0% | 0.0m | 0.0m |
| **Total** | **634.5m** | **100.0%** | **45.3m** | **-** |

- **Evolution/Test ratio**: 0.31x
  - For every minute of testing, 0.31 minutes spent on evolution

## Claude CLI Usage & Cost Analysis
- **Total Cost**: $86.72
  - **Evaluation Cost**: $11.04
  - **Evolution CLI Cost**: $47.69
  - **Other (evaluator overhead) Cost**: $27.99
- **Total CLI Calls**: 26
- **Total CLI Input Tokens**: 13,712
- **Total CLI Output Tokens**: 621,615
- **Total Cache Created**: 1,485,916
- **Total Cache Read**: 34,386,807

### Cost Breakdown

| Operation | Cost | % of Total | Calls | Tokens In | Tokens Out |
|-----------|------|------------|-------|-----------|------------|
| Evaluation | $11.04 | 12.7% | - | - | - |
| Evolution (CLI) | $47.69 | 55.0% | 26 | 13,712 | 621,615 |
| Other (evaluator overhead) | $27.99 | 32.3% | - | - | - |
| **Total** | **$86.72** | **100%** | **26** | **13,712** | **621,615** |

**Answer: Evolution drives 55.0% of total costs**

### Detailed Per-Iteration Costs

| Iter | Total | **Eval** | **Evo** | Iter Eval | 1st Draft | Refl | Meta | Other | Fresh | Strategy | Meta-strategy |
|------|------|------|------|------|------|------|------|------|------|------|------|
| **1** | $0.04 | **$0.01** | **-** | $0.01 | - | - | - | $0.03 | 14 | - | - |
| **2** | $4.49 | **$0.23** | **$3.21** | $0.23 | $2.87 | $0.34 | - | $1.05 | 27 | use_your_judgment | - |
| **3** | $7.71 | **$0.55** | **$4.97** | $0.55 | $3.63 | $1.34 | - | $2.18 | 34 | use_your_judgment | - |
| **4** | $7.06 | **$0.58** | **$3.92** | $0.58 | $3.63 | $0.29 | - | $2.56 | 29 | use_your_judgment | - |
| **5** | $5.35 | **$0.52** | **$3.44** | $0.52 | $3.10 | $0.34 | - | $1.39 | 34 | use_your_judgment | - |
| **6** | $4.80 | **$0.51** | **$3.10** | $0.51 | $2.82 | $0.28 | - | $1.19 | 23 | use_your_judgment | - |
| **7** | $9.59 | **$0.92** | **$4.65** | $0.92 | $4.10 | $0.55 | - | $4.02 | 34 | use_your_judgment | - |
| **8** | $5.96 | **$0.70** | **$3.23** | $0.70 | $2.85 | $0.38 | - | $2.03 | 29 | use_your_judgment | - |
| **9** | $6.93 | **$1.01** | **$3.90** | $1.01 | $3.57 | $0.33 | - | $2.02 | 31 | use_your_judgment | - |
| **10** | $5.82 | **$0.91** | **$2.95** | $0.91 | $2.68 | $0.26 | - | $1.97 | 25 | use_your_judgment | - |
| **11** | $8.83 | **$1.51** | **$3.76** | $1.51 | $3.45 | $0.32 | - | $3.56 | 32 | use_your_judgment | - |
| **12** | $6.53 | **$1.25** | **$2.89** | $1.25 | $2.59 | $0.29 | - | $2.40 | 30 | use_your_judgment | - |
| **13** | $6.44 | **$1.06** | **$4.09** | $1.06 | $3.81 | $0.28 | - | $1.29 | 22 | use_your_judgment | - |
| **14** | $7.16 | **$1.28** | **$3.58** | $1.28 | $3.28 | $0.30 | - | $2.30 | 26 | use_your_judgment | - |
|------|------|------|------|------|------|------|------|------|------|------|------|
| **TOTAL** | **$86.72** | **$11.04** | **$47.69** | $11.04 | $42.39 | $5.30 | $0.00 | **$27.99** | **390** | - | - |

### Evolution Cost Breakdown (Avg per evolution)

| Phase | Avg Cost | % of Evolution | Avg Tokens In | Avg Tokens Out |
|-------|----------|----------------|---------------|----------------|
| First Draft (Round 1) | $3.26 | 88.9% | 1051 | 42441 |
| Reflection | $0.41 | 11.1% | 4 | 5375 |
| **Total** | **$3.67** | **100%** | **1055** | **47817** |

### Cache Performance

- **Cache Creation**: 1,485,916 tokens
- **Cache Hits**: 34,386,807 tokens (95.9% hit rate)

## Zero Accuracy Cases
No zero accuracy cases encountered ✅

## Exception Failures
No exception failures encountered ✅

## Evolution Health

No artifact validation failures ✅