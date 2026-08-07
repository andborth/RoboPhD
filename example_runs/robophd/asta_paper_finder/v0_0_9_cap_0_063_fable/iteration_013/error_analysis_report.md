# Comparative Agent Analysis - Iteration 013

**Generated**: 2026-08-05 03:01:27

**Agents**: iter10_deadline_guard, iter12_salvage_rank, iter13_any_author_gate
**Total problems**: 14

## Score Summary

| Agent | Mean Raw Score | Mean Score | Problems |
|-------|----------------|------------|----------|
| iter13_any_author_gate | 0.424 | 42.398 | 14 |
| iter10_deadline_guard | 0.426 | 41.839 | 14 |
| iter12_salvage_rank | 0.417 | 41.741 | 14 |

**Aggregate notes** (how Mean Score was derived from Mean Raw Score):

- **iter13_any_author_gate**: Mean agent cost $0.0603 within free zone (threshold $0.063); no penalty applied. Raw mean F1 0.4240 reported as percentage: 42.398.
- **iter10_deadline_guard**: Mean agent cost $0.0637 exceeded threshold $0.063 by $0.0007 = 0.11 errors of penalty (cost_per_error=$0.0063); subtracted 0.768 score pts from raw 42.607 → final 41.839 (percentage).
- **iter12_salvage_rank**: Mean agent cost $0.0578 within free zone (threshold $0.063); no penalty applied. Raw mean F1 0.4174 reported as percentage: 41.741.

- **Solo wins**: iter13_any_author_gate: 4 (semantic_221, semantic_214, semantic_137, semantic_226)
- **Solo wins**: iter10_deadline_guard: 4 (semantic_222, semantic_196, semantic_77, semantic_57)
- **Solo wins**: iter12_salvage_rank: 3 (semantic_220, semantic_174, semantic_123)
- **Solo losses**: iter13_any_author_gate: 2 (semantic_222, semantic_174)
- **Solo losses**: iter10_deadline_guard: 4 (semantic_221, semantic_220, semantic_123, semantic_226)
- **Solo losses**: iter12_salvage_rank: 4 (semantic_214, semantic_137, semantic_196, semantic_57)

## Raw Score Comparison

| Problem | iter13_any_author_gate | iter10_deadline_guard | iter12_salvage_rank | Δ(best-worst) | Δ(#1-#2) |
|---------|--------|--------|--------|---------|---------|
| semantic_214 | 0.169 | 0.118 | 0.000 | 0.169 | 0.052 |
| semantic_77 | 0.000 | 0.146 | 0.000 | 0.146 | 0.146 |
| semantic_174 | 0.637 | 0.716 | 0.773 | 0.136 | 0.057 |
| semantic_221 | 0.590 | 0.460 | 0.553 | 0.130 | 0.037 |
| semantic_123 | 0.199 | 0.141 | 0.241 | 0.100 | 0.042 |
| semantic_222 | 0.197 | 0.292 | 0.222 | 0.095 | 0.070 |
| semantic_220 | 0.304 | 0.217 | 0.304 | 0.086 | 0.000 |
| semantic_196 | 0.483 | 0.531 | 0.451 | 0.080 | 0.049 |
| semantic_57 | 0.612 | 0.659 | 0.608 | 0.051 | 0.048 |
| semantic_137 | 0.179 | 0.131 | 0.129 | 0.050 | 0.048 |
| semantic_226 | 0.266 | 0.253 | 0.262 | 0.013 | 0.004 |
| metadata_15 | 0.800 | 0.800 | 0.800 | 0.000 | 0.000 |
| specific_20 | 0.500 | 0.500 | 0.500 | 0.000 | 0.000 |
| specific_33 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
