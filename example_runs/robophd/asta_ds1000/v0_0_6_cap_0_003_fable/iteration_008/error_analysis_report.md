# Comparative Agent Analysis - Iteration 008

**Generated**: 2026-07-04 14:39:29

**Agents**: iter2_exec_verify_ensemble (95.0%), iter7_lean_audited_cascade (90.0%), iter8_expected_diff_cascade (95.0%)

## Summary
- Total questions: 20
- Consensus correct: 18 (90.0%)
- Consensus failures: 1 (5.0%)
- Split decisions: 1 (5.0%)

## Agent Accuracy

| Agent | Correct | Failed | Errors | Mean Raw Score | Mean Score |
|-------|---------|--------|--------|----------------|------------|
| iter2_exec_verify_ensemble | 19 | 1 | 0 | 0.9500 | 87.053 |
| iter7_lean_audited_cascade | 18 | 2 | 0 | 0.9000 | 90.000 |
| iter8_expected_diff_cascade | 19 | 1 | 0 | 0.9500 | 93.892 |

**Aggregate notes** (how Mean Score was derived from Mean Raw Score):

- **iter2_exec_verify_ensemble**: Mean cost $0.0046 exceeded threshold $0.003 by $0.0016 = 1.59 errors of penalty (cost_per_error=$0.0010); subtracted 7.947 score pts from raw 95.000 → final 87.053 (percentage).
- **iter7_lean_audited_cascade**: Mean cost $0.0028 within free zone (threshold $0.003); no penalty applied. Raw accuracy 0.9000 reported as percentage: 90.000.
- **iter8_expected_diff_cascade**: Mean cost $0.0032 exceeded threshold $0.003 by $0.0002 = 0.22 errors of penalty (cost_per_error=$0.0010); subtracted 1.108 score pts from raw 95.000 → final 93.892 (percentage).

## Consensus Failures

All agents failed on 1 questions: 812

## Split Decisions

Total split decisions: 1

- ✓ iter2_exec_verify_ensemble, iter8_expected_diff_cascade | ✗ iter7_lean_audited_cascade: **919**
