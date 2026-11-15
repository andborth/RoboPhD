# Comparative Agent Analysis - Iteration 019

**Generated**: 2026-11-12 16:55:28

**Agents**: iter13_citation_snowball, iter18_stage1_keyed_order, iter19_backdated_citation_counts
**Total problems**: 14

## Score Summary

| Agent | Mean Raw Score | Mean Score | Problems |
|-------|----------------|------------|----------|
| iter18_stage1_keyed_order | 0.589 | 58.948 | 14 |
| iter19_backdated_citation_counts | 0.586 | 58.591 | 14 |
| iter13_citation_snowball | 0.483 | 48.265 | 14 |

**Aggregate notes** (how Mean Score was derived from Mean Raw Score):

- **iter18_stage1_keyed_order**: Mean agent cost $0.2410 within free zone (threshold $0.279); no penalty applied. Raw mean F1 0.5895 reported as percentage: 58.948.
- **iter19_backdated_citation_counts**: Mean agent cost $0.2419 within free zone (threshold $0.279); no penalty applied. Raw mean F1 0.5859 reported as percentage: 58.591.
- **iter13_citation_snowball**: Mean agent cost $0.2338 within free zone (threshold $0.279); no penalty applied. Raw mean F1 0.4826 reported as percentage: 48.265.

- **Solo wins**: iter18_stage1_keyed_order: 5 (semantic_152, semantic_174, semantic_120, semantic_226, semantic_8)
- **Solo wins**: iter19_backdated_citation_counts: 4 (semantic_140, metadata_42, semantic_203, semantic_220)
- **Solo wins**: iter13_citation_snowball: 1 (semantic_112)
- **Solo losses**: iter18_stage1_keyed_order: 1 (semantic_203)
- **Solo losses**: iter19_backdated_citation_counts: 3 (semantic_152, semantic_112, semantic_8)
- **Solo losses**: iter13_citation_snowball: 7 (semantic_174, semantic_140, metadata_42, specific_20, semantic_120, semantic_220, semantic_226)

## Raw Score Comparison

| Problem | iter18_stage1_keyed_order | iter19_backdated_citation_counts | iter13_citation_snowball | Δ(best-worst) | Δ(#1-#2) |
|---------|--------|--------|--------|---------|---------|
| specific_20 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| semantic_220 | 0.571 | 0.653 | 0.475 | 0.179 | 0.082 |
| semantic_120 | 0.260 | 0.121 | 0.082 | 0.178 | 0.139 |
| semantic_152 | 0.643 | 0.471 | 0.613 | 0.172 | 0.030 |
| metadata_42 | 0.542 | 0.636 | 0.505 | 0.132 | 0.094 |
| semantic_140 | 0.281 | 0.399 | 0.278 | 0.122 | 0.119 |
| semantic_174 | 0.866 | 0.826 | 0.777 | 0.089 | 0.040 |
| semantic_203 | 0.210 | 0.286 | 0.217 | 0.076 | 0.069 |
| semantic_226 | 0.374 | 0.356 | 0.312 | 0.062 | 0.018 |
| semantic_112 | 0.852 | 0.817 | 0.855 | 0.038 | 0.003 |
| semantic_8 | 0.653 | 0.636 | 0.643 | 0.017 | 0.010 |
| semantic_145 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| specific_10 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| specific_24 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
