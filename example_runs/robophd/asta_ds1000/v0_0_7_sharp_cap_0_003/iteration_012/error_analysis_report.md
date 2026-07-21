# Comparative Agent Analysis - Iteration 012

**Generated**: 2026-07-21 05:28:32

**Agents**: iter10_strong_reindent (80.0%), iter12_strong_toplevel (90.0%), iter6_strong_oneshot (85.0%)

## Summary
- Total questions: 20
- Consensus correct: 15 (75.0%)
- Consensus failures: 1 (5.0%)
- Split decisions: 4 (20.0%)

## Agent Accuracy

| Agent | Correct | Failed | Errors | Mean Raw Score | Mean Score |
|-------|---------|--------|--------|----------------|------------|
| iter10_strong_reindent | 16 | 4 | 0 | 0.8000 | 80.000 |
| iter12_strong_toplevel | 18 | 2 | 0 | 0.9000 | 90.000 |
| iter6_strong_oneshot | 17 | 3 | 0 | 0.8500 | 85.000 |

**Aggregate notes** (how Mean Score was derived from Mean Raw Score):

- **iter10_strong_reindent**: Mean cost $0.0016 within free zone (threshold $0.003); no penalty applied. Raw accuracy 0.8000 reported as percentage: 80.000.
- **iter12_strong_toplevel**: Mean cost $0.0016 within free zone (threshold $0.003); no penalty applied. Raw accuracy 0.9000 reported as percentage: 90.000.
- **iter6_strong_oneshot**: Mean cost $0.0016 within free zone (threshold $0.003); no penalty applied. Raw accuracy 0.8500 reported as percentage: 85.000.

## Consensus Failures

All agents failed on 1 questions: 887

## Split Decisions

Total split decisions: 4

- ✓ iter10_strong_reindent, iter12_strong_toplevel | ✗ iter6_strong_oneshot: **526**
- ✓ iter12_strong_toplevel | ✗ iter10_strong_reindent, iter6_strong_oneshot: **451**
- ✓ iter12_strong_toplevel, iter6_strong_oneshot | ✗ iter10_strong_reindent: **977**
- ✓ iter6_strong_oneshot | ✗ iter10_strong_reindent, iter12_strong_toplevel: **10**
