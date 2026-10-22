# Comparative Agent Analysis - Iteration 015

**Generated**: 2026-10-20 11:21:06

**Agents**: iter13_crosscast, iter15_bodycast, iter7_headcast
**Total problems**: 14

## Score Summary

| Agent | Mean Raw Score | Mean Score | Problems |
|-------|----------------|------------|----------|
| iter15_bodycast | 0.590 | 58.862 | 14 |
| iter13_crosscast | 0.584 | 58.413 | 14 |
| iter7_headcast | 0.518 | 51.810 | 14 |

**Aggregate notes** (how Mean Score was derived from Mean Raw Score):

- **iter15_bodycast**: Mean agent cost $0.0631 exceeded threshold $0.063 by $0.0001 = 0.02 errors of penalty (cost_per_error=$0.0063); subtracted 0.160 score pts from raw 59.022 → final 58.862 (percentage).
- **iter13_crosscast**: Mean agent cost $0.0528 within free zone (threshold $0.063); no penalty applied. Raw mean F1 0.5841 reported as percentage: 58.413.
- **iter7_headcast**: Mean agent cost $0.0486 within free zone (threshold $0.063); no penalty applied. Raw mean F1 0.5181 reported as percentage: 51.810.

- **Solo wins**: iter15_bodycast: 5 (semantic_140, semantic_8, semantic_193, semantic_155, semantic_87)
- **Solo wins**: iter13_crosscast: 3 (semantic_226, semantic_112, semantic_160)
- **Solo wins**: iter7_headcast: 1 (semantic_229)
- **Solo losses**: iter15_bodycast: 3 (semantic_112, specific_33, semantic_160)
- **Solo losses**: iter13_crosscast: 3 (semantic_229, semantic_140, semantic_193)
- **Solo losses**: iter7_headcast: 5 (semantic_226, semantic_8, metadata_31, semantic_155, semantic_87)

## Raw Score Comparison

| Problem | iter15_bodycast | iter13_crosscast | iter7_headcast | Δ(best-worst) | Δ(#1-#2) |
|---------|--------|--------|--------|---------|---------|
| metadata_31 | 0.750 | 0.750 | 0.061 | 0.689 | 0.000 |
| specific_33 | 0.400 | 0.667 | 0.667 | 0.267 | 0.000 |
| semantic_140 | 0.384 | 0.153 | 0.154 | 0.231 | 0.230 |
| semantic_155 | 0.440 | 0.306 | 0.245 | 0.195 | 0.134 |
| semantic_112 | 0.658 | 0.765 | 0.712 | 0.107 | 0.053 |
| semantic_226 | 0.265 | 0.283 | 0.181 | 0.101 | 0.018 |
| semantic_8 | 0.471 | 0.395 | 0.373 | 0.097 | 0.076 |
| semantic_160 | 0.324 | 0.403 | 0.370 | 0.078 | 0.032 |
| semantic_229 | 0.481 | 0.420 | 0.496 | 0.076 | 0.015 |
| semantic_87 | 0.848 | 0.837 | 0.793 | 0.055 | 0.011 |
| semantic_193 | 0.242 | 0.199 | 0.200 | 0.043 | 0.042 |
| specific_7 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| specific_9 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| specific_24 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
