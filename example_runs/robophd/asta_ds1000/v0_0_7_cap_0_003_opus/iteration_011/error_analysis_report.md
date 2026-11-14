# Comparative Agent Analysis - Iteration 011

**Generated**: 2026-07-21 05:21:03

**Agents**: iter10_strong_reindent (70.0%), iter11_strong_assign (60.0%), seed__mxgdywk (65.0%)

## Summary
- Total questions: 20
- Consensus correct: 10 (50.0%)
- Consensus failures: 5 (25.0%)
- Split decisions: 5 (25.0%)

## Agent Accuracy

| Agent | Correct | Failed | Errors | Mean Raw Score | Mean Score |
|-------|---------|--------|--------|----------------|------------|
| iter10_strong_reindent | 14 | 6 | 0 | 0.7000 | 70.000 |
| iter11_strong_assign | 12 | 8 | 0 | 0.6000 | 60.000 |
| seed__mxgdywk | 13 | 7 | 0 | 0.6500 | 65.000 |

**Aggregate notes** (how Mean Score was derived from Mean Raw Score):

- **iter10_strong_reindent**: Mean cost $0.0015 within free zone (threshold $0.003); no penalty applied. Raw accuracy 0.7000 reported as percentage: 70.000.
- **iter11_strong_assign**: Mean cost $0.0016 within free zone (threshold $0.003); no penalty applied. Raw accuracy 0.6000 reported as percentage: 60.000.
- **seed__mxgdywk**: Mean cost $0.0005 within free zone (threshold $0.003); no penalty applied. Raw accuracy 0.6500 reported as percentage: 65.000.

## Consensus Failures

All agents failed on 5 questions: 165, 444, 706, 883, 887

## Split Decisions

Total split decisions: 5

- ✓ iter10_strong_reindent, iter11_strong_assign | ✗ seed__mxgdywk: **426**, **944**
- ✓ iter10_strong_reindent, seed__mxgdywk | ✗ iter11_strong_assign: **723**, **838**
- ✓ seed__mxgdywk | ✗ iter10_strong_reindent, iter11_strong_assign: **910**
