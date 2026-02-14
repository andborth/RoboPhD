# Critic Evaluation Results

**Updated:** February 8, 2026
**Dataset:** Competition programming problems (288 problems, LiveCodeBench test set)

## Overview

This document summarizes empirical results from critic evaluation runs. Part 1 covers 9 naive critic runs testing all combinations of coder and critic models across three capability tiers: haiku-4.5, sonnet-4.5, and opus-4.5. Part 2 covers evolved critic agents produced by RoboPhD.

The critic system works as follows:
1. **V1 (Coder):** Generate initial code solution
2. **Critic:** Review V1 and decide CORRECT or INCORRECT
3. **V2 (Revision):** If INCORRECT, critic generates revised code
4. **Acceptance:** Coder accepts/rejects critic's suggestions

## Part 1: Naive Critic (3x3 Model Grid)

### Table 1: Accuracy Impact

| Run ID | Coder → Critic | Relation | Valid | V1 | V2 | Delta | Net Fix |
|--------|----------------|----------|-------|----|----|-------|---------|
| run_20260121_203944 | haiku → haiku | = same | 288 | 51.7% | 51.0% | -0.7% | -2 |
| run_20260124_051813 | haiku → sonnet | ↑ stronger | 288 | 52.1% | 56.6% | +4.5% | +13 |
| run_20260127_232251 | haiku → opus | ↑ stronger | 288 | 51.4% | 59.7% | +8.3% | +24 |
| run_20260129_030511 | sonnet → haiku | ↓ weaker | 288 | 57.6% | 58.7% | +1.1% | +3 |
| run_20260121_221710 | sonnet → sonnet | = same | 288 | 58.0% | 58.0% | +0.0% | +0 |
| run_20260128_104741 | sonnet → opus | ↑ stronger | 287 | 58.2% | 65.9% | +7.7% | +22 |
| run_20260129_073319 | opus → haiku | ↓ weaker | 278 | 83.8% | 83.8% | +0.0% | +0 |
| run_20260128_220446 | opus → sonnet | ↓ weaker | 275 | 84.7% | 86.2% | +1.5% | +4 |
| run_20260127_180226 | opus → opus | = same | 272 | 84.9% | 87.1% | +2.2% | +6 |

### Table 2: Classification Performance

The critic's job is to detect wrong V1 answers. These metrics measure that capability:

- **Precision:** When critic says INCORRECT, how often is V1 actually wrong?
- **Recall:** Of all wrong V1 answers, what fraction does the critic catch?
- **F1:** Harmonic mean of precision and recall

| Run ID | Coder → Critic | Relation | TP | FP | Precision | Recall | F1 |
|--------|----------------|----------|----|----|-----------|--------|-----|
| run_20260121_203944 | haiku → haiku | = same | 57 | 16 | 78.1% | 41.0% | 53.8% |
| run_20260124_051813 | haiku → sonnet | ↑ stronger | 66 | 9 | 88.0% | 47.8% | 62.0% |
| run_20260127_232251 | haiku → opus | ↑ stronger | 119 | 31 | 79.3% | 85.0% | 82.1% |
| run_20260129_030511 | sonnet → haiku | ↓ weaker | 66 | 17 | 79.5% | 54.1% | 64.4% |
| run_20260121_221710 | sonnet → sonnet | = same | 52 | 13 | 80.0% | 43.0% | 55.9% |
| run_20260128_104741 | sonnet → opus | ↑ stronger | 99 | 33 | 75.0% | 82.5% | 78.6% |
| run_20260129_073319 | opus → haiku | ↓ weaker | 10 | 9 | 52.6% | 22.2% | 31.2% |
| run_20260128_220446 | opus → sonnet | ↓ weaker | 10 | 6 | 62.5% | 23.8% | 34.5% |
| run_20260127_180226 | opus → opus | = same | 13 | 8 | 61.9% | 31.7% | 41.9% |

### Table 3: Revision Effectiveness

When the critic flags code as INCORRECT and attempts revision:

| Run ID | Coder → Critic | Relation | Flagged | Improved | Regressed | Fix Rate |
|--------|----------------|----------|---------|----------|-----------|----------|
| run_20260121_203944 | haiku → haiku | = same | 73 | 2 | 4 | 2.7% |
| run_20260124_051813 | haiku → sonnet | ↑ stronger | 75 | 14 | 1 | 18.7% |
| run_20260127_232251 | haiku → opus | ↑ stronger | 150 | 28 | 4 | 18.7% |
| run_20260129_030511 | sonnet → haiku | ↓ weaker | 83 | 8 | 5 | 9.6% |
| run_20260121_221710 | sonnet → sonnet | = same | 65 | 6 | 6 | 9.2% |
| run_20260128_104741 | sonnet → opus | ↑ stronger | 132 | 27 | 5 | 20.5% |
| run_20260129_073319 | opus → haiku | ↓ weaker | 19 | 0 | 0 | 0.0% |
| run_20260128_220446 | opus → sonnet | ↓ weaker | 16 | 4 | 0 | 25.0% |
| run_20260127_180226 | opus → opus | = same | 21 | 7 | 1 | 33.3% |

## 3x3 Matrices

### Delta Matrix (V2 - V1 accuracy)

|  | haiku critic | sonnet critic | opus critic |
|--|--------------|---------------|-------------|
| **haiku coder** | -0.7% (= same) | +4.5% (↑ stronger) | +8.3% (↑ stronger) |
| **sonnet coder** | +1.1% (↓ weaker) | +0.0% (= same) | +7.7% (↑ stronger) |
| **opus coder** | +0.0% (↓ weaker) | +1.5% (↓ weaker) | +2.2% (= same) |

### Recall Matrix

|  | haiku critic | sonnet critic | opus critic |
|--|--------------|---------------|-------------|
| **haiku coder** | 41% | 48% | 85% |
| **sonnet coder** | 54% | 43% | 83% |
| **opus coder** | 22% | 24% | 32% |

### Net Fixes Matrix

|  | haiku critic | sonnet critic | opus critic |
|--|--------------|---------------|-------------|
| **haiku coder** | -2 | +13 | +24 |
| **sonnet coder** | +3 | +0 | +22 |
| **opus coder** | +0 | +4 | +6 |

## Summary by Relation (Critic vs Coder Capability)

| Relation | Runs | Avg Delta | Avg Recall | Avg Fix Rate | Net Fixes |
|----------|------|-----------|------------|--------------|-----------|
| **↑ Stronger critic** | haiku→sonnet, haiku→opus, sonnet→opus | **+6.8%** | **71.8%** | **19.3%** | **+59** |
| **= Same tier** | haiku→haiku, sonnet→sonnet, opus→opus | +0.5% | 38.6% | 15.1% | +4 |
| **↓ Weaker critic** | sonnet→haiku, opus→haiku, opus→sonnet | +0.9% | 33.4% | 11.5% | +7 |

**Note:** Same-tier and weaker averages are similar (+0.5% vs +0.9%), but this masks important differences:
- Same-tier includes opus→opus (+2.2%) which benefits from opus's high absolute capability
- Weaker includes sonnet→haiku (+1.1%) where haiku can still catch some of sonnet's errors
- The key differentiator is **recall**: stronger critics catch 72% of errors vs ~33-39% for others

## Summary by Critic Model

The clearest pattern emerges when grouping by critic capability:

| Critic | With haiku coder | With sonnet coder | With opus coder | Avg Delta |
|--------|------------------|-------------------|-----------------|-----------|
| **opus** | +8.3% | +7.7% | +2.2% | **+6.1%** |
| **sonnet** | +4.5% | +0.0% | +1.5% | +2.0% |
| **haiku** | -0.7% | +1.1% | +0.0% | +0.1% |

| Critic | Avg Recall | Avg Fix Rate |
|--------|------------|--------------|
| **opus** | 66.4% | 24.2% |
| **sonnet** | 39.8% | 17.6% |
| **haiku** | 39.1% | 4.1% |

**Key insight:** Opus critic helps all coders substantially. Sonnet critic only helps haiku coder meaningfully. Haiku critic provides negligible or negative value regardless of coder.

## Summary by Coder Model

| Coder | V1 Baseline | Best Critic | Best Delta | Best Net Fix |
|-------|-------------|-------------|------------|--------------|
| **haiku** | ~52% | opus | +8.3% | +24 |
| **sonnet** | ~58% | opus | +7.7% | +22 |
| **opus** | ~84% | opus | +2.2% | +6 |

**Key insight:** Lower-capability coders have more room for improvement, but even opus code benefits from opus critic.

## Key Findings

### 1. Opus Critic Dramatically Outperforms Others

Opus critic achieves **+6.1%** average improvement across all coders, versus +2.0% for sonnet and +0.1% for haiku. The key differentiator is **recall**: opus catches 66% of errors on average versus ~39% for sonnet/haiku critics.

### 2. Same-Tier Critics Are Unreliable

- **haiku→haiku:** Actually hurts (-0.7%)
- **sonnet→sonnet:** Breaks even (0%)
- **opus→opus:** Modest gain (+2.2%), but opus code is already 85% correct

Same-tier critics lack the capability gap needed to reliably identify and fix errors.

### 3. Weaker Critics Struggle with Stronger Code

When reviewing opus code (84% baseline accuracy):
- **opus→haiku:** 22% recall, 0% fix rate, +0.0% delta
- **opus→sonnet:** 24% recall, 25% fix rate, +1.5% delta
- **opus→opus:** 32% recall, 33% fix rate, +2.2% delta

Weaker critics can barely detect opus's errors, let alone fix them.

### 4. The Capability Gap Hypothesis

For meaningful improvement, the critic should be **stronger than the coder**:
- Opus critic helps everyone substantially
- Sonnet critic only helps haiku meaningfully (+4.5%)
- Haiku critic helps no one

However, even same-tier can work when the coder is already strong (opus→opus: +2.2%), suggesting the relationship is more nuanced than a simple "stronger = better" rule.

### 5. Detection ≠ Correction

Even when critics correctly identify wrong code (True Positives), they often fail to fix it:

| Coder → Critic | True Positives | Fixed | Fix Rate of TPs |
|----------------|----------------|-------|-----------------|
| haiku → haiku | 57 | 2 | 3.5% |
| haiku → sonnet | 66 | 14 | 21.2% |
| haiku → opus | 119 | 28 | 23.5% |
| sonnet → opus | 99 | 27 | 27.3% |
| opus → opus | 13 | 7 | 53.8% |

The capability gap affects both detection AND correction ability.

### 6. Opus Code Is Hard to Improve

- V1 accuracy already 84-85% (only ~42 errors to find in 288 problems)
- Even opus→opus only catches 32% of errors
- Weaker critics barely scratch the surface (22-24% recall)

### 7. Cost Efficiency

| Configuration | Net Fixes | Total Cost | Cost per Fix |
|---------------|-----------|------------|--------------|
| haiku → sonnet | +13 | ~$75 | ~$5.77 |
| haiku → opus | +24 | ~$129 | ~$5.38 |
| sonnet → opus | +22 | ~$193 | ~$8.77 |

**haiku→opus** provides the best value: highest improvement at moderate cost.

## Part 1 Conclusions

1. **Use opus as critic** for meaningful improvement. It's the only naive critic providing substantial gains across all coder tiers.

2. **Naive same-tier critics are noise** - they break even or hurt performance. (But see Part 2: evolution changes this.)

3. **Weaker critics waste compute** - they can't reliably identify or fix stronger models' mistakes.

4. **For production** (with naive critics), the optimal configuration depends on budget:
   - **Budget-conscious:** haiku coder + opus critic (+8.3%, ~$5.38/fix)
   - **Quality-focused:** sonnet coder + opus critic (+7.7%, higher baseline)
   - **Avoid:** Same-tier or weaker critic configurations

## Part 2: Evolved Critic (RoboPhD)

### Evolution Setup

Two evolved critic agents are compared, both using haiku-4.5 for coder and critic with opus-4.5 for evolution:

- **`0203_i005_reflection_refined_critic`** — from RoboPhD run `robophd_20260203_154803` (14 iterations, evolution set of 767 problems). Uses tool-only execution mode — a deterministic 875-line Python analyzer (`problem_analyzer.py`) generates critic feedback without any LLM call in the analysis phase. Key innovations include calibrated anti-TLE-rationalization rules, stronger logic bug detection, and an explicit forbidden rationalizations list in `eval_instructions.md`. Source: `main_codegen_agents/0203_i005_reflection_refined_critic`.

- **`0206_i013_unchanged_baseline_critic`** — from RoboPhD run `robophd_20260206_201828` (15 iterations, evolution set of 60 problems). Source: `critic_evaluations/run_20260207_230433`.

### Skip-a-Tier Result

The best evolved haiku critic matches naive sonnet critic performance:

| Config | Agent | V1 | V2 | Delta | Net Fix |
|--------|-------|----|----|-------|---------|
| haiku → haiku | naive_critic | 51.7% | 51.0% | -0.7% | -2 |
| haiku → sonnet | naive_critic | 52.1% | 56.6% | +4.5% | +13 |
| **haiku → haiku** | **0203_i005_reflection_refined_critic** | **52.1%** | **56.6%** | **+4.5%** | **+13** |
| haiku → haiku | 0206_i013_unchanged_baseline_critic | 54.9% | 55.6% | +0.7% | +2 |

The first evolved agent replicates the "skip-a-tier" finding from the RoboPhD Text2SQL paper: evolution on cheaper models provides the largest gains, enabling evolved Haiku to match naive Sonnet performance. The second evolved agent, from a different evolution run, shows that not all evolved agents achieve this — evolution quality varies significantly across runs.

### How the Evolved Agents Differ

| Metric | Naive haiku | Naive sonnet | 0203_i005 | 0206_i013 |
|--------|-------------|--------------|-----------|-----------|
| Flagged | 73 | 75 | **138** | 124 |
| TP (true positives) | 57 | 66 | **108** | 92 |
| FP (false positives) | 16 | 9 | 30 | 32 |
| Precision | 78.1% | 88.0% | 78.3% | 74.2% |
| Recall | 41.0% | 47.8% | **78.3%** | 70.8% |
| Fix Rate | 2.7% | 18.7% | 10.9% | 6.5% |
| Improved | 2 | 14 | 15 | 8 |
| Regressed | 4 | 1 | 2 | 6 |

Both evolved agents are more aggressive than naive critics — flagging far more problems with higher recall. The first evolved agent (`0203_i005`) achieves the best balance: 78% recall with a 10.9% fix rate and only 2 regressions, matching naive sonnet's +4.5% delta. The second (`0206_i013`) has similar recall (71%) but a lower fix rate (6.5%) and more regressions (6), yielding only +0.7% delta.

### Cost Analysis

Deployment cost is critic + revision (codegen is fixed across configurations; acceptance is only needed during evolution). Average cost per problem:

| Phase | Naive haiku→haiku | Naive haiku→sonnet | 0203_i005 | 0206_i013 |
|-------|-------------------|--------------------|-----------|-----------|
| critic | $0.042 | $0.123 | $0.116 | — |
| revision | $0.033 | $0.026 | $0.078 | — |
| **critic + revision** | **$0.075** | **$0.149** | **$0.194** | **$0.098** |

*Note: Per-phase cost breakdown for `0206_i013` is not available; total evaluation cost was $28.20 for 288 problems ($0.098/problem).*

The first evolved agent's critic phase ($0.116) is actually cheaper than naive sonnet's ($0.123) despite sending a 15x larger prompt — haiku's 3x lower per-token pricing more than compensates. The cost premium comes from revision: the evolved agent flags 138 problems vs 75, so nearly twice as many go through the expensive revision pipeline.

The first evolved agent costs **1.3x** naive sonnet for the same +4.5% delta. The second evolved agent is cheaper ($0.098/problem) but delivers far less improvement (+0.7%).

### Prompt Size

The evolved agent's critic prompt averages 14,161 bytes (363 lines) vs 944 bytes (32 lines) for the naive critic. The bulk of the extra context is `eval_instructions.md` (11,896 bytes, 294 lines), which provides detailed analysis rules. The tool analyzer output (`critic_feedback.txt`) contributes only ~1,000 bytes per problem.

Reducing `eval_instructions.md` from 294 lines to ~200 lines would bring evolved haiku's total cost under naive sonnet's, achieving the same accuracy at lower cost.

### Key Findings

1. **Evolution can break the capability gap requirement.** The naive critic results showed that same-tier critics were unreliable (haiku→haiku: -0.7%). The best evolved agent transforms haiku into an effective same-tier critic (+4.5%), but this is not guaranteed — the second evolved agent achieved only +0.7%.

2. **Evolution compensates for model capability with context.** Rather than relying on a stronger model's reasoning, the evolved agent feeds haiku detailed analytical context — effectively substituting compute (larger prompts) for capability (more expensive model).

3. **The skip-a-tier pattern generalizes across domains.** The RoboPhD Text2SQL paper showed evolved Haiku exceeding naive Sonnet on SQL generation. This result replicates the same pattern for code critique, suggesting it is a general property of the evolution approach.

4. **Cost parity is achievable.** The best evolved agent is currently 1.3x the cost of naive sonnet, driven primarily by aggressive flagging (138 vs 75 problems revised). Modest prompt trimming would bring costs below sonnet's while maintaining the same delta.

5. **Evolution quality varies across runs.** Two independent evolution runs produced agents with very different effectiveness (+4.5% vs +0.7%). Both agents adopted similar high-recall strategies, but the second agent's lower fix rate (6.5% vs 10.9%) and higher regression count (6 vs 2) erased most of the gains from detection. Consistent evolution quality remains an open challenge.

## Future Work

- Trim `eval_instructions.md` to achieve cost parity with naive sonnet
- Evolve critics for sonnet and opus coders
- Explore multi-round critique (critic reviews its own revision)
- Investigate why detection is easier than correction
- Test whether evolution can close the gap to opus-level critic performance (+8.3%)
