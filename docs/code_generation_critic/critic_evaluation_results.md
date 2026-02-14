# Critic Evaluation Results

**Date:** January 29, 2026
**Agent:** naive_critic
**Dataset:** Competition programming problems (288 problems)

## Overview

This document summarizes empirical results from 9 complete critic evaluation runs testing all combinations of coder and critic models across three capability tiers: haiku-4.5, sonnet-4.5, and opus-4.5.

The critic system works as follows:
1. **V1 (Coder):** Generate initial code solution
2. **Critic:** Review V1 and decide CORRECT or INCORRECT
3. **V2 (Revision):** If INCORRECT, critic generates revised code
4. **Acceptance:** Coder accepts/rejects critic's suggestions

## Complete Run Summary

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

## Conclusions

1. **Use opus as critic** for meaningful improvement. It's the only critic providing substantial gains across all coder tiers.

2. **Same-tier critics are noise** - they break even or hurt performance.

3. **Weaker critics waste compute** - they can't reliably identify or fix stronger models' mistakes.

4. **For production**, the optimal configuration depends on budget:
   - **Budget-conscious:** haiku coder + opus critic (+8.3%, ~$5.38/fix)
   - **Quality-focused:** sonnet coder + opus critic (+7.7%, higher baseline)
   - **Avoid:** Same-tier or weaker critic configurations

## Future Work

- Test with evolved/specialized critic agents instead of naive_critic
- Explore multi-round critique (critic reviews its own revision)
- Investigate why detection is easier than correction
- Test on different problem domains (Text-to-SQL vs competition programming)
