# Comparative Agent Analysis - Iteration 015

**Generated**: 2026-07-30 14:51:17

**Agents**: iter11_ensemble_conjunctive_rank, iter13_balanced_digest_wide_vetting, iter15_verdict_repair
**Total problems**: 14

## Score Summary

| Agent | Mean Raw Score | Mean Score | Problems |
|-------|----------------|------------|----------|
| iter13_balanced_digest_wide_vetting | 0.538 | 53.821 | 14 |
| iter15_verdict_repair | 0.538 | 53.812 | 14 |
| iter11_ensemble_conjunctive_rank | 0.517 | 51.688 | 14 |

**Aggregate notes** (how Mean Score was derived from Mean Raw Score):

- **iter13_balanced_digest_wide_vetting**: Mean agent cost $0.0479 within free zone (threshold $0.063); no penalty applied. Raw mean F1 0.5382 reported as percentage: 53.821.
- **iter15_verdict_repair**: Mean agent cost $0.0473 within free zone (threshold $0.063); no penalty applied. Raw mean F1 0.5381 reported as percentage: 53.812.
- **iter11_ensemble_conjunctive_rank**: Mean agent cost $0.0388 within free zone (threshold $0.063); no penalty applied. Raw mean F1 0.5169 reported as percentage: 51.688.

- **Solo wins**: iter13_balanced_digest_wide_vetting: 4 (semantic_172, semantic_2, semantic_205, semantic_8)
- **Solo wins**: iter15_verdict_repair: 3 (semantic_196, semantic_214, semantic_189)
- **Solo wins**: iter11_ensemble_conjunctive_rank: 2 (semantic_110, semantic_112)
- **Solo losses**: iter13_balanced_digest_wide_vetting: 1 (semantic_214)
- **Solo losses**: iter15_verdict_repair: 2 (semantic_110, semantic_112)
- **Solo losses**: iter11_ensemble_conjunctive_rank: 6 (semantic_172, semantic_2, semantic_196, semantic_205, semantic_8, semantic_189)

## Raw Score Comparison

| Problem | iter13_balanced_digest_wide_vetting | iter15_verdict_repair | iter11_ensemble_conjunctive_rank | Δ(best-worst) | Δ(#1-#2) |
|---------|--------|--------|--------|---------|---------|
| semantic_189 | 0.137 | 0.292 | 0.094 | 0.198 | 0.155 |
| semantic_205 | 0.262 | 0.234 | 0.142 | 0.120 | 0.028 |
| semantic_112 | 0.753 | 0.645 | 0.755 | 0.111 | 0.002 |
| semantic_8 | 0.374 | 0.332 | 0.315 | 0.059 | 0.042 |
| semantic_196 | 0.117 | 0.134 | 0.075 | 0.059 | 0.017 |
| semantic_172 | 0.797 | 0.786 | 0.756 | 0.041 | 0.011 |
| semantic_2 | 0.378 | 0.377 | 0.352 | 0.026 | 0.001 |
| semantic_214 | 0.070 | 0.089 | 0.087 | 0.019 | 0.002 |
| semantic_110 | 0.647 | 0.645 | 0.660 | 0.015 | 0.013 |
| specific_9 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| metadata_14 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| metadata_42 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| specific_15 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| specific_33 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
