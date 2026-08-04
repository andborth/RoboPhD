# Comparative Agent Analysis - Iteration 018

**Generated**: 2026-08-03 00:38:31

**Agents**: iter17_ambigunion_landmark_v1, iter18_cocite_largegold_v1, iter9_metafix_poolboost_v1
**Total problems**: 14

## Score Summary

| Agent | Mean Raw Score | Mean Score | Problems |
|-------|----------------|------------|----------|
| iter18_cocite_largegold_v1 | 0.415 | 41.532 | 14 |
| iter17_ambigunion_landmark_v1 | 0.404 | 40.142 | 14 |
| iter9_metafix_poolboost_v1 | 0.368 | 36.795 | 14 |

**Aggregate notes** (how Mean Score was derived from Mean Raw Score):

- **iter18_cocite_largegold_v1**: Mean agent cost $0.3478 within free zone (threshold $0.355); no penalty applied. Raw mean F1 0.4153 reported as percentage: 41.532.
- **iter17_ambigunion_landmark_v1**: Mean agent cost $0.3562 exceeded threshold $0.355 by $0.0012 = 0.03 errors of penalty (cost_per_error=$0.0355); subtracted 0.243 score pts from raw 40.385 → final 40.142 (percentage).
- **iter9_metafix_poolboost_v1**: Mean agent cost $0.2443 within free zone (threshold $0.355); no penalty applied. Raw mean F1 0.3679 reported as percentage: 36.795.

- **Solo wins**: iter18_cocite_largegold_v1: 5 (metadata_25, semantic_110, semantic_160, semantic_91, semantic_222)
- **Solo wins**: iter17_ambigunion_landmark_v1: 5 (semantic_170, semantic_196, semantic_220, semantic_138, semantic_70)
- **Solo wins**: iter9_metafix_poolboost_v1: 3 (semantic_101, semantic_174, semantic_152)
- **Solo losses**: iter18_cocite_largegold_v1: 2 (semantic_101, semantic_70)
- **Solo losses**: iter17_ambigunion_landmark_v1: 2 (semantic_174, semantic_152)
- **Solo losses**: iter9_metafix_poolboost_v1: 9 (metadata_25, semantic_170, semantic_196, semantic_220, semantic_110, semantic_160, semantic_138, semantic_91, semantic_222)

## Raw Score Comparison

| Problem | iter18_cocite_largegold_v1 | iter17_ambigunion_landmark_v1 | iter9_metafix_poolboost_v1 | Δ(best-worst) | Δ(#1-#2) |
|---------|--------|--------|--------|---------|---------|
| semantic_196 | 0.453 | 0.462 | 0.241 | 0.221 | 0.009 |
| semantic_222 | 0.344 | 0.297 | 0.160 | 0.184 | 0.047 |
| semantic_152 | 0.387 | 0.347 | 0.489 | 0.142 | 0.102 |
| semantic_110 | 0.844 | 0.811 | 0.705 | 0.138 | 0.033 |
| metadata_25 | 0.111 | 0.062 | 0.010 | 0.101 | 0.048 |
| semantic_138 | 0.601 | 0.615 | 0.515 | 0.100 | 0.014 |
| semantic_174 | 0.759 | 0.697 | 0.783 | 0.086 | 0.023 |
| semantic_91 | 0.534 | 0.460 | 0.453 | 0.081 | 0.075 |
| semantic_220 | 0.223 | 0.269 | 0.195 | 0.074 | 0.046 |
| semantic_70 | 0.109 | 0.158 | 0.157 | 0.049 | 0.002 |
| semantic_170 | 0.489 | 0.518 | 0.476 | 0.042 | 0.030 |
| semantic_101 | 0.453 | 0.463 | 0.475 | 0.022 | 0.012 |
| semantic_160 | 0.508 | 0.495 | 0.492 | 0.016 | 0.013 |
| semantic_43 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
