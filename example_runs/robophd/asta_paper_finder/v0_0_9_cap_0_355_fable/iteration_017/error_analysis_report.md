# Comparative Agent Analysis - Iteration 017

**Generated**: 2026-08-02 23:30:32

**Agents**: iter12_citerevive_multiref_v1, iter17_ambigunion_landmark_v1, iter9_metafix_poolboost_v1
**Total problems**: 14

## Score Summary

| Agent | Mean Raw Score | Mean Score | Problems |
|-------|----------------|------------|----------|
| iter17_ambigunion_landmark_v1 | 0.469 | 46.900 | 14 |
| iter12_citerevive_multiref_v1 | 0.375 | 37.452 | 14 |
| iter9_metafix_poolboost_v1 | 0.356 | 35.587 | 14 |

**Aggregate notes** (how Mean Score was derived from Mean Raw Score):

- **iter17_ambigunion_landmark_v1**: Mean agent cost $0.2320 within free zone (threshold $0.355); no penalty applied. Raw mean F1 0.4690 reported as percentage: 46.900.
- **iter12_citerevive_multiref_v1**: Mean agent cost $0.1967 within free zone (threshold $0.355); no penalty applied. Raw mean F1 0.3745 reported as percentage: 37.452.
- **iter9_metafix_poolboost_v1**: Mean agent cost $0.1632 within free zone (threshold $0.355); no penalty applied. Raw mean F1 0.3559 reported as percentage: 35.587.

- **Solo wins**: iter17_ambigunion_landmark_v1: 6 (semantic_108, semantic_186, semantic_170, specific_9, specific_39, semantic_189)
- **Solo wins**: iter12_citerevive_multiref_v1: 5 (semantic_100, metadata_25, semantic_155, metadata_42, semantic_193)
- **Solo wins**: iter9_metafix_poolboost_v1: 1 (semantic_77)
- **Solo losses**: iter17_ambigunion_landmark_v1: 2 (semantic_155, semantic_77)
- **Solo losses**: iter12_citerevive_multiref_v1: 2 (specific_39, semantic_189)
- **Solo losses**: iter9_metafix_poolboost_v1: 7 (semantic_108, semantic_186, semantic_170, semantic_100, metadata_25, metadata_42, semantic_193)

## Raw Score Comparison

| Problem | iter17_ambigunion_landmark_v1 | iter12_citerevive_multiref_v1 | iter9_metafix_poolboost_v1 | Δ(best-worst) | Δ(#1-#2) |
|---------|--------|--------|--------|---------|---------|
| specific_9 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| specific_39 | 0.444 | 0.000 | 0.333 | 0.444 | 0.111 |
| metadata_42 | 0.378 | 0.426 | 0.106 | 0.320 | 0.048 |
| semantic_155 | 0.244 | 0.504 | 0.435 | 0.260 | 0.069 |
| semantic_189 | 0.321 | 0.134 | 0.250 | 0.186 | 0.070 |
| semantic_108 | 0.661 | 0.622 | 0.516 | 0.145 | 0.039 |
| metadata_25 | 0.062 | 0.087 | 0.010 | 0.077 | 0.025 |
| semantic_186 | 0.451 | 0.406 | 0.384 | 0.066 | 0.045 |
| semantic_100 | 0.390 | 0.420 | 0.373 | 0.047 | 0.029 |
| semantic_170 | 0.518 | 0.512 | 0.476 | 0.042 | 0.006 |
| semantic_193 | 0.265 | 0.295 | 0.261 | 0.034 | 0.030 |
| semantic_77 | 0.141 | 0.147 | 0.149 | 0.008 | 0.002 |
| metadata_31 | 0.690 | 0.690 | 0.690 | 0.000 | 0.000 |
| specific_24 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
