# Comparative Agent Analysis - Iteration 009

**Generated**: 2026-07-24 06:05:31

**Agents**: iter3_evidence_rerank_v1, iter6_metadata_targeted_v1, iter9_rerank_rich_v1
**Total problems**: 14

## Score Summary

| Agent | Mean Raw Score | Mean Score | Problems |
|-------|----------------|------------|----------|
| iter3_evidence_rerank_v1 | 0.122 | 12.227 | 14 |
| iter9_rerank_rich_v1 | 0.121 | 12.064 | 14 |
| iter6_metadata_targeted_v1 | 0.120 | 11.958 | 14 |

**Aggregate notes** (how Mean Score was derived from Mean Raw Score):

- **iter3_evidence_rerank_v1**: Mean agent cost $0.0031 within free zone (threshold $0.033); no penalty applied. Raw mean F1 0.1223 reported as percentage: 12.227.
- **iter9_rerank_rich_v1**: Mean agent cost $0.0065 within free zone (threshold $0.033); no penalty applied. Raw mean F1 0.1206 reported as percentage: 12.064.
- **iter6_metadata_targeted_v1**: Mean agent cost $0.0056 within free zone (threshold $0.033); no penalty applied. Raw mean F1 0.1196 reported as percentage: 11.958.

- **Solo wins**: iter3_evidence_rerank_v1: 4 (semantic_98, semantic_152, semantic_77, semantic_33)
- **Solo wins**: iter9_rerank_rich_v1: 4 (semantic_7, semantic_43, semantic_2, semantic_229)
- **Solo wins**: iter6_metadata_targeted_v1: 3 (semantic_203, semantic_123, semantic_137)
- **Solo losses**: iter3_evidence_rerank_v1: 5 (metadata_25, semantic_43, semantic_2, semantic_123, semantic_137)
- **Solo losses**: iter9_rerank_rich_v1: 3 (semantic_203, semantic_152, semantic_33)
- **Solo losses**: iter6_metadata_targeted_v1: 3 (semantic_98, semantic_77, semantic_229)

## Raw Score Comparison

| Problem | iter3_evidence_rerank_v1 | iter9_rerank_rich_v1 | iter6_metadata_targeted_v1 | Δ(best-worst) | Δ(#1-#2) |
|---------|--------|--------|--------|---------|---------|
| semantic_98 | 0.363 | 0.163 | 0.141 | 0.221 | 0.199 |
| semantic_123 | 0.071 | 0.073 | 0.194 | 0.123 | 0.121 |
| semantic_203 | 0.144 | 0.078 | 0.201 | 0.123 | 0.056 |
| semantic_7 | 0.000 | 0.099 | 0.000 | 0.099 | 0.099 |
| semantic_43 | 0.112 | 0.202 | 0.170 | 0.090 | 0.032 |
| semantic_2 | 0.122 | 0.187 | 0.124 | 0.065 | 0.063 |
| semantic_229 | 0.245 | 0.267 | 0.218 | 0.050 | 0.022 |
| semantic_152 | 0.329 | 0.290 | 0.299 | 0.039 | 0.030 |
| metadata_25 | 0.000 | 0.010 | 0.010 | 0.010 | 0.000 |
| semantic_77 | 0.147 | 0.144 | 0.140 | 0.007 | 0.004 |
| semantic_33 | 0.085 | 0.081 | 0.081 | 0.004 | 0.004 |
| semantic_137 | 0.095 | 0.095 | 0.095 | 0.001 | 0.000 |
| metadata_14 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| metadata_33 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
