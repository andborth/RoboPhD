# Comparative Agent Analysis - Iteration 020

**Generated**: 2026-08-01 08:32:53

**Agents**: iter19_deep_screen_bulk_evidence, iter20_prf_coverage_expansion, iter9_bulk_passage_harvest
**Total problems**: 14

## Score Summary

| Agent | Mean Raw Score | Mean Score | Problems |
|-------|----------------|------------|----------|
| iter19_deep_screen_bulk_evidence | 0.394 | 39.426 | 14 |
| iter9_bulk_passage_harvest | 0.368 | 36.780 | 14 |
| iter20_prf_coverage_expansion | 0.347 | 34.746 | 14 |

**Aggregate notes** (how Mean Score was derived from Mean Raw Score):

- **iter19_deep_screen_bulk_evidence**: Mean agent cost $0.2375 within free zone (threshold $0.355); no penalty applied. Raw mean F1 0.3943 reported as percentage: 39.426.
- **iter9_bulk_passage_harvest**: Mean agent cost $0.2097 within free zone (threshold $0.355); no penalty applied. Raw mean F1 0.3678 reported as percentage: 36.780.
- **iter20_prf_coverage_expansion**: Mean agent cost $0.2920 within free zone (threshold $0.355); no penalty applied. Raw mean F1 0.3475 reported as percentage: 34.746.

- **Solo wins**: iter19_deep_screen_bulk_evidence: 7 (metadata_42, metadata_26, semantic_104, semantic_174, semantic_186, semantic_110, semantic_77)
- **Solo wins**: iter9_bulk_passage_harvest: 1 (semantic_222)
- **Solo wins**: iter20_prf_coverage_expansion: 3 (semantic_205, semantic_189, semantic_22)
- **Solo losses**: iter19_deep_screen_bulk_evidence: 1 (semantic_222)
- **Solo losses**: iter9_bulk_passage_harvest: 5 (semantic_174, semantic_205, semantic_110, semantic_189, semantic_22)
- **Solo losses**: iter20_prf_coverage_expansion: 5 (metadata_42, metadata_26, semantic_104, semantic_186, semantic_77)

## Raw Score Comparison

| Problem | iter19_deep_screen_bulk_evidence | iter9_bulk_passage_harvest | iter20_prf_coverage_expansion | Δ(best-worst) | Δ(#1-#2) |
|---------|--------|--------|--------|---------|---------|
| semantic_104 | 0.625 | 0.506 | 0.300 | 0.325 | 0.119 |
| metadata_26 | 0.250 | 0.200 | 0.000 | 0.250 | 0.050 |
| semantic_77 | 0.385 | 0.359 | 0.276 | 0.109 | 0.026 |
| semantic_189 | 0.454 | 0.404 | 0.511 | 0.107 | 0.057 |
| semantic_174 | 0.835 | 0.751 | 0.788 | 0.084 | 0.047 |
| semantic_205 | 0.365 | 0.344 | 0.386 | 0.042 | 0.021 |
| semantic_22 | 0.279 | 0.272 | 0.312 | 0.040 | 0.033 |
| semantic_222 | 0.272 | 0.309 | 0.305 | 0.038 | 0.005 |
| semantic_110 | 0.755 | 0.723 | 0.727 | 0.032 | 0.028 |
| semantic_186 | 0.505 | 0.493 | 0.483 | 0.022 | 0.011 |
| metadata_42 | 0.045 | 0.037 | 0.027 | 0.018 | 0.008 |
| metadata_25 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| metadata_15 | 0.750 | 0.750 | 0.750 | 0.000 | 0.000 |
| metadata_33 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
