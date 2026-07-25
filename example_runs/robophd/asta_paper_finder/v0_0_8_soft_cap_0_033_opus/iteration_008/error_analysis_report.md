# Comparative Agent Analysis - Iteration 008

**Generated**: 2026-07-24 05:30:03

**Agents**: iter5_grade3_rerank_v1, iter6_metadata_targeted_v1, iter8_grade3_probe_v1
**Total problems**: 14

## Score Summary

| Agent | Mean Raw Score | Mean Score | Problems |
|-------|----------------|------------|----------|
| iter6_metadata_targeted_v1 | 0.206 | 20.566 | 14 |
| iter5_grade3_rerank_v1 | 0.178 | 17.801 | 14 |
| iter8_grade3_probe_v1 | 0.176 | 17.567 | 14 |

**Aggregate notes** (how Mean Score was derived from Mean Raw Score):

- **iter6_metadata_targeted_v1**: Mean agent cost $0.0052 within free zone (threshold $0.033); no penalty applied. Raw mean F1 0.2057 reported as percentage: 20.566.
- **iter5_grade3_rerank_v1**: Mean agent cost $0.0154 within free zone (threshold $0.033); no penalty applied. Raw mean F1 0.1780 reported as percentage: 17.801.
- **iter8_grade3_probe_v1**: Mean agent cost $0.0053 within free zone (threshold $0.033); no penalty applied. Raw mean F1 0.1757 reported as percentage: 17.567.

- **Solo wins**: iter6_metadata_targeted_v1: 3 (semantic_203, semantic_148, semantic_152)
- **Solo wins**: iter5_grade3_rerank_v1: 5 (semantic_70, semantic_222, semantic_125, semantic_186, semantic_108)
- **Solo wins**: iter8_grade3_probe_v1: 2 (semantic_104, semantic_189)
- **Solo losses**: iter6_metadata_targeted_v1: 2 (semantic_189, semantic_70)
- **Solo losses**: iter5_grade3_rerank_v1: 2 (semantic_104, specific_10)
- **Solo losses**: iter8_grade3_probe_v1: 6 (semantic_203, semantic_148, semantic_222, semantic_186, semantic_108, semantic_152)

## Raw Score Comparison

| Problem | iter6_metadata_targeted_v1 | iter5_grade3_rerank_v1 | iter8_grade3_probe_v1 | Δ(best-worst) | Δ(#1-#2) |
|---------|--------|--------|--------|---------|---------|
| specific_10 | 1.000 | 0.667 | 1.000 | 0.333 | 0.000 |
| semantic_108 | 0.291 | 0.308 | 0.117 | 0.190 | 0.016 |
| semantic_104 | 0.270 | 0.147 | 0.309 | 0.162 | 0.039 |
| semantic_148 | 0.351 | 0.318 | 0.196 | 0.155 | 0.033 |
| semantic_203 | 0.201 | 0.143 | 0.078 | 0.123 | 0.058 |
| semantic_186 | 0.160 | 0.237 | 0.129 | 0.107 | 0.076 |
| semantic_189 | 0.136 | 0.136 | 0.242 | 0.106 | 0.105 |
| semantic_152 | 0.299 | 0.271 | 0.215 | 0.084 | 0.027 |
| semantic_125 | 0.000 | 0.081 | 0.000 | 0.081 | 0.081 |
| semantic_222 | 0.071 | 0.082 | 0.071 | 0.011 | 0.011 |
| semantic_70 | 0.100 | 0.104 | 0.102 | 0.003 | 0.001 |
| metadata_14 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| semantic_7 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| specific_39 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
