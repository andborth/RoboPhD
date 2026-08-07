# Comparative Agent Analysis - Iteration 014

**Generated**: 2026-08-05 05:02:01

**Agents**: iter13_any_author_gate, iter14_title_channel, iter6_graph_recall
**Total problems**: 14

## Score Summary

| Agent | Mean Raw Score | Mean Score | Problems |
|-------|----------------|------------|----------|
| iter14_title_channel | 0.427 | 42.673 | 14 |
| iter13_any_author_gate | 0.388 | 38.842 | 14 |
| iter6_graph_recall | 0.379 | 37.942 | 14 |

**Aggregate notes** (how Mean Score was derived from Mean Raw Score):

- **iter14_title_channel**: Mean agent cost $0.0599 within free zone (threshold $0.063); no penalty applied. Raw mean F1 0.4267 reported as percentage: 42.673.
- **iter13_any_author_gate**: Mean agent cost $0.0571 within free zone (threshold $0.063); no penalty applied. Raw mean F1 0.3884 reported as percentage: 38.842.
- **iter6_graph_recall**: Mean agent cost $0.0420 within free zone (threshold $0.063); no penalty applied. Raw mean F1 0.3794 reported as percentage: 37.942.

- **Solo wins**: iter14_title_channel: 7 (semantic_186, semantic_104, semantic_123, semantic_170, semantic_193, semantic_33, semantic_7)
- **Solo wins**: iter13_any_author_gate: 2 (semantic_224, semantic_8)
- **Solo wins**: iter6_graph_recall: 2 (semantic_148, semantic_110)
- **Solo losses**: iter14_title_channel: 0
- **Solo losses**: iter13_any_author_gate: 5 (semantic_148, semantic_170, semantic_110, semantic_33, semantic_7)
- **Solo losses**: iter6_graph_recall: 6 (semantic_186, semantic_104, semantic_123, semantic_193, semantic_224, semantic_8)

## Raw Score Comparison

| Problem | iter14_title_channel | iter13_any_author_gate | iter6_graph_recall | Δ(best-worst) | Δ(#1-#2) |
|---------|--------|--------|--------|---------|---------|
| semantic_7 | 0.292 | 0.091 | 0.100 | 0.201 | 0.192 |
| semantic_8 | 0.432 | 0.465 | 0.322 | 0.144 | 0.034 |
| semantic_186 | 0.451 | 0.420 | 0.322 | 0.129 | 0.031 |
| semantic_110 | 0.633 | 0.626 | 0.737 | 0.111 | 0.103 |
| semantic_170 | 0.515 | 0.407 | 0.449 | 0.109 | 0.067 |
| semantic_193 | 0.298 | 0.251 | 0.192 | 0.106 | 0.046 |
| semantic_104 | 0.333 | 0.267 | 0.232 | 0.101 | 0.067 |
| semantic_148 | 0.574 | 0.507 | 0.587 | 0.080 | 0.013 |
| semantic_123 | 0.259 | 0.199 | 0.198 | 0.061 | 0.060 |
| semantic_224 | 0.446 | 0.467 | 0.433 | 0.033 | 0.021 |
| semantic_33 | 0.165 | 0.162 | 0.164 | 0.002 | 0.000 |
| metadata_26 | 0.263 | 0.263 | 0.263 | 0.000 | 0.000 |
| metadata_25 | 0.313 | 0.313 | 0.313 | 0.000 | 0.000 |
| specific_7 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
