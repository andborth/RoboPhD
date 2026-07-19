# Comparative Agent Analysis - Iteration 011

**Generated**: 2026-07-18 00:32:10

**Agents**: iter10_cite_expand, iter11_tail_saturate, iter4_judge_sim_ranker
**Total problems**: 14

## Score Summary

| Agent | Mean Raw Score | Mean Score | Problems |
|-------|----------------|------------|----------|
| iter11_tail_saturate | 0.442 | 44.202 | 14 |
| iter10_cite_expand | 0.423 | 42.340 | 14 |
| iter4_judge_sim_ranker | 0.349 | 34.914 | 14 |

**Aggregate notes** (how Mean Score was derived from Mean Raw Score):

- **iter11_tail_saturate**: Mean agent cost $0.0585 within free zone (threshold $0.06); no penalty applied. Raw mean F1 0.4420 reported as percentage: 44.202.
- **iter10_cite_expand**: Mean agent cost $0.0564 within free zone (threshold $0.06); no penalty applied. Raw mean F1 0.4234 reported as percentage: 42.340.
- **iter4_judge_sim_ranker**: Mean agent cost $0.0374 within free zone (threshold $0.06); no penalty applied. Raw mean F1 0.3491 reported as percentage: 34.914.

- **Solo wins**: iter11_tail_saturate: 5 (semantic_219, semantic_110, metadata_33, semantic_112, semantic_8)
- **Solo wins**: iter10_cite_expand: 3 (semantic_193, semantic_222, semantic_104)
- **Solo wins**: iter4_judge_sim_ranker: 3 (semantic_226, semantic_7, semantic_137)
- **Solo losses**: iter11_tail_saturate: 1 (semantic_104)
- **Solo losses**: iter10_cite_expand: 6 (semantic_226, semantic_219, semantic_7, semantic_112, semantic_8, semantic_137)
- **Solo losses**: iter4_judge_sim_ranker: 5 (specific_9, semantic_110, semantic_193, semantic_222, metadata_33)

## Raw Score Comparison

| Problem | iter11_tail_saturate | iter10_cite_expand | iter4_judge_sim_ranker | Δ(best-worst) | Δ(#1-#2) |
|---------|--------|--------|--------|---------|---------|
| specific_9 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| metadata_33 | 0.143 | 0.125 | 0.000 | 0.143 | 0.018 |
| semantic_112 | 0.749 | 0.625 | 0.657 | 0.124 | 0.092 |
| semantic_7 | 0.099 | 0.000 | 0.104 | 0.104 | 0.004 |
| semantic_110 | 0.442 | 0.413 | 0.347 | 0.095 | 0.029 |
| semantic_104 | 0.035 | 0.126 | 0.096 | 0.091 | 0.029 |
| semantic_137 | 0.108 | 0.094 | 0.175 | 0.081 | 0.067 |
| semantic_193 | 0.377 | 0.420 | 0.341 | 0.079 | 0.044 |
| semantic_222 | 0.338 | 0.362 | 0.284 | 0.077 | 0.023 |
| semantic_8 | 0.325 | 0.254 | 0.320 | 0.071 | 0.004 |
| semantic_219 | 0.199 | 0.149 | 0.156 | 0.050 | 0.043 |
| semantic_226 | 0.373 | 0.360 | 0.408 | 0.048 | 0.035 |
| specific_33 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| specific_24 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
