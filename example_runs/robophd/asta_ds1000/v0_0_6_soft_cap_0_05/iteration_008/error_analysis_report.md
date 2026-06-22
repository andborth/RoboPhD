# Comparative Agent Analysis - Iteration 008

**Generated**: 2026-06-21 21:19:04

**Agents**: iter3_safe_repair_ds1000 (90.0%), iter7_grounded_reconcile_ds1000 (75.0%), iter8_perspective_consensus_ds1000 (95.0%)

## Summary
- Total questions: 20
- Consensus correct: 15 (75.0%)
- Consensus failures: 1 (5.0%)
- Split decisions: 4 (20.0%)

## Agent Accuracy

| Agent | Correct | Failed | Errors | Mean Raw Score | Mean Score |
|-------|---------|--------|--------|----------------|------------|
| iter3_safe_repair_ds1000 | 18 | 2 | 0 | 0.9000 | 90.000 |
| iter7_grounded_reconcile_ds1000 | 15 | 5 | 0 | 0.7500 | 75.000 |
| iter8_perspective_consensus_ds1000 | 19 | 1 | 0 | 0.9500 | 95.000 |

**Aggregate notes** (how Mean Score was derived from Mean Raw Score):

- **iter3_safe_repair_ds1000**: Mean cost $0.0021 within free zone (threshold $0.05); no penalty applied. Raw accuracy 0.9000 reported as percentage: 90.000.
- **iter7_grounded_reconcile_ds1000**: Mean cost $0.0054 within free zone (threshold $0.05); no penalty applied. Raw accuracy 0.7500 reported as percentage: 75.000.
- **iter8_perspective_consensus_ds1000**: Mean cost $0.0065 within free zone (threshold $0.05); no penalty applied. Raw accuracy 0.9500 reported as percentage: 95.000.

## Consensus Failures

All agents failed on 1 questions: 883

## Split Decisions

Total split decisions: 4

- ✓ iter3_safe_repair_ds1000, iter8_perspective_consensus_ds1000 | ✗ iter7_grounded_reconcile_ds1000: **667**, **723**, **906**
- ✓ iter8_perspective_consensus_ds1000 | ✗ iter3_safe_repair_ds1000, iter7_grounded_reconcile_ds1000: **763**
