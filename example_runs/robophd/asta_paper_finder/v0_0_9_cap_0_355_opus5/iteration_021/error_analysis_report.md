# Comparative Agent Analysis - Iteration 021

**Generated**: 2026-08-01 09:46:04

**Agents**: iter19_deep_screen_bulk_evidence, iter21_gold_rubric_and_hard_predicates, iter8_criterion_window_evidence
**Total problems**: 14

## Score Summary

| Agent | Mean Raw Score | Mean Score | Problems |
|-------|----------------|------------|----------|
| iter21_gold_rubric_and_hard_predicates | 0.581 | 58.092 | 14 |
| iter8_criterion_window_evidence | 0.552 | 55.207 | 14 |
| iter19_deep_screen_bulk_evidence | 0.544 | 54.383 | 14 |

**Aggregate notes** (how Mean Score was derived from Mean Raw Score):

- **iter21_gold_rubric_and_hard_predicates**: Mean agent cost $0.2604 within free zone (threshold $0.355); no penalty applied. Raw mean F1 0.5809 reported as percentage: 58.092.
- **iter8_criterion_window_evidence**: Mean agent cost $0.1823 within free zone (threshold $0.355); no penalty applied. Raw mean F1 0.5521 reported as percentage: 55.207.
- **iter19_deep_screen_bulk_evidence**: Mean agent cost $0.2483 within free zone (threshold $0.355); no penalty applied. Raw mean F1 0.5438 reported as percentage: 54.383.

- **Solo wins**: iter21_gold_rubric_and_hard_predicates: 6 (semantic_155, semantic_189, semantic_108, semantic_57, semantic_222, semantic_140)
- **Solo wins**: iter8_criterion_window_evidence: 1 (semantic_123)
- **Solo wins**: iter19_deep_screen_bulk_evidence: 3 (semantic_170, semantic_172, semantic_220)
- **Solo losses**: iter21_gold_rubric_and_hard_predicates: 0
- **Solo losses**: iter8_criterion_window_evidence: 5 (semantic_170, semantic_172, semantic_108, semantic_220, semantic_140)
- **Solo losses**: iter19_deep_screen_bulk_evidence: 6 (metadata_4, semantic_123, semantic_155, semantic_189, semantic_57, semantic_222)

## Raw Score Comparison

| Problem | iter21_gold_rubric_and_hard_predicates | iter8_criterion_window_evidence | iter19_deep_screen_bulk_evidence | Δ(best-worst) | Δ(#1-#2) |
|---------|--------|--------|--------|---------|---------|
| semantic_123 | 0.250 | 0.444 | 0.186 | 0.258 | 0.194 |
| metadata_4 | 0.667 | 0.667 | 0.545 | 0.121 | 0.000 |
| semantic_220 | 0.397 | 0.305 | 0.421 | 0.115 | 0.023 |
| semantic_189 | 0.553 | 0.495 | 0.454 | 0.098 | 0.058 |
| semantic_155 | 0.746 | 0.672 | 0.652 | 0.094 | 0.074 |
| semantic_170 | 0.557 | 0.501 | 0.593 | 0.092 | 0.036 |
| semantic_140 | 0.547 | 0.459 | 0.495 | 0.088 | 0.052 |
| semantic_57 | 0.640 | 0.561 | 0.556 | 0.084 | 0.079 |
| semantic_222 | 0.333 | 0.275 | 0.272 | 0.061 | 0.057 |
| semantic_108 | 0.570 | 0.514 | 0.563 | 0.056 | 0.007 |
| semantic_172 | 0.874 | 0.835 | 0.878 | 0.042 | 0.003 |
| specific_7 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| metadata_33 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| metadata_14 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
