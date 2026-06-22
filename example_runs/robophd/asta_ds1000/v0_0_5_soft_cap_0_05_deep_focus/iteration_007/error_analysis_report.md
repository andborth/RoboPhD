# Comparative Agent Analysis - Iteration 007

**Generated**: 2026-06-21 20:52:10

**Agents**: iter3_safe_repair_ds1000 (95.0%), iter6_grounded_repair_ds1000 (85.0%), iter7_grounded_reconcile_ds1000 (90.0%)

## Summary
- Total questions: 20
- Consensus correct: 17 (85.0%)
- Consensus failures: 0 (0.0%)
- Split decisions: 3 (15.0%)

## Agent Accuracy

| Agent | Correct | Failed | Errors | Mean Raw Score | Mean Score |
|-------|---------|--------|--------|----------------|------------|
| iter3_safe_repair_ds1000 | 19 | 1 | 0 | 0.9500 | 95.000 |
| iter6_grounded_repair_ds1000 | 17 | 3 | 0 | 0.8500 | 85.000 |
| iter7_grounded_reconcile_ds1000 | 18 | 2 | 0 | 0.9000 | 90.000 |

**Aggregate notes** (how Mean Score was derived from Mean Raw Score):

- **iter3_safe_repair_ds1000**: Mean cost $0.0024 within free zone (threshold $0.05); no penalty applied. Raw accuracy 0.9500 reported as percentage: 95.000.
- **iter6_grounded_repair_ds1000**: Mean cost $0.0048 within free zone (threshold $0.05); no penalty applied. Raw accuracy 0.8500 reported as percentage: 85.000.
- **iter7_grounded_reconcile_ds1000**: Mean cost $0.0064 within free zone (threshold $0.05); no penalty applied. Raw accuracy 0.9000 reported as percentage: 90.000.

## Split Decisions

Total split decisions: 3

- ✓ iter3_safe_repair_ds1000 | ✗ iter6_grounded_repair_ds1000, iter7_grounded_reconcile_ds1000: **667**, **723**
- ✓ iter7_grounded_reconcile_ds1000 | ✗ iter3_safe_repair_ds1000, iter6_grounded_repair_ds1000: **129**
