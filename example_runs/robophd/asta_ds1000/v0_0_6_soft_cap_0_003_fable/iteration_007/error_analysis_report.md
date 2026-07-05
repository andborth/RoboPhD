# Comparative Agent Analysis - Iteration 007

**Generated**: 2026-07-04 14:08:02

**Agents**: iter2_exec_verify_ensemble (75.0%), iter6_audited_cascade (75.0%), iter7_lean_audited_cascade (80.0%)

## Summary
- Total questions: 20
- Consensus correct: 15 (75.0%)
- Consensus failures: 4 (20.0%)
- Split decisions: 1 (5.0%)

## Agent Accuracy

| Agent | Correct | Failed | Errors | Mean Raw Score | Mean Score |
|-------|---------|--------|--------|----------------|------------|
| iter2_exec_verify_ensemble | 15 | 5 | 0 | 0.7500 | 72.283 |
| iter6_audited_cascade | 15 | 5 | 0 | 0.7500 | 71.646 |
| iter7_lean_audited_cascade | 16 | 4 | 0 | 0.8000 | 80.000 |

**Aggregate notes** (how Mean Score was derived from Mean Raw Score):

- **iter2_exec_verify_ensemble**: Mean cost $0.0035 exceeded threshold $0.003 by $0.0005 = 0.54 errors of penalty (cost_per_error=$0.0010); subtracted 2.717 score pts from raw 75.000 → final 72.283 (percentage).
- **iter6_audited_cascade**: Mean cost $0.0037 exceeded threshold $0.003 by $0.0007 = 0.67 errors of penalty (cost_per_error=$0.0010); subtracted 3.354 score pts from raw 75.000 → final 71.646 (percentage).
- **iter7_lean_audited_cascade**: Mean cost $0.0023 within free zone (threshold $0.003); no penalty applied. Raw accuracy 0.8000 reported as percentage: 80.000.

## Consensus Failures

All agents failed on 4 questions: 238, 165, 269, 445

## Split Decisions

Total split decisions: 1

- ✓ iter7_lean_audited_cascade | ✗ iter2_exec_verify_ensemble, iter6_audited_cascade: **446**
