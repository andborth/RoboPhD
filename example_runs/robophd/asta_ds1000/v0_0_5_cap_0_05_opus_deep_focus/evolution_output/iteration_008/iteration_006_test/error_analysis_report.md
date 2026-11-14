# New vs Baseline Analysis - Test Round

**Generated**: 2026-06-21 21:03:28

**New Agent**: iter8_perspective_consensus_ds1000 (85.0%)
**Baselines**: iter6_grounded_repair_ds1000 (80.0%), iter3_safe_repair_ds1000 (75.0%), iter5_strong_dualcheck_ds1000 (75.0%)

## Summary
- Unique successes (new succeeded, all baselines failed): 1
- Unique failures (new failed, all baselines succeeded): 0
- Consensus failures (all failed): 3
- Mixed results: 2

## Agent Accuracy

| Agent | Correct | Failed | Errors | Mean Raw Score | Mean Score |
|-------|---------|--------|--------|----------------|------------|
| iter8_perspective_consensus_ds1000 | 17 | 3 | 0 | 0.8500 | 85.000 |
| iter6_grounded_repair_ds1000 | 16 | 4 | 0 | 0.8000 | 80.000 |
| iter3_safe_repair_ds1000 | 15 | 5 | 0 | 0.7500 | 75.000 |
| iter5_strong_dualcheck_ds1000 | 15 | 5 | 0 | 0.7500 | 75.000 |

**Aggregate notes** (how Mean Score was derived from Mean Raw Score):

- **iter8_perspective_consensus_ds1000**: Mean cost $0.0053 within free zone (threshold $0.05); no penalty applied. Raw accuracy 0.8500 reported as percentage: 85.000.
- **iter6_grounded_repair_ds1000**: Mean cost $0.0033 within free zone (threshold $0.05); no penalty applied. Raw accuracy 0.8000 reported as percentage: 80.000.
- **iter3_safe_repair_ds1000**: Mean cost $0.0019 within free zone (threshold $0.05); no penalty applied. Raw accuracy 0.7500 reported as percentage: 75.000.
- **iter5_strong_dualcheck_ds1000**: Mean cost $0.0106 within free zone (threshold $0.05); no penalty applied. Raw accuracy 0.7500 reported as percentage: 75.000.

## Unique Successes

New agent succeeded but all baselines failed (1): 238

## Consensus Failures

New agent AND all baselines failed (3): 165, 129, 706

## Mixed Results

Total mixed results: 2

- ✅ NEW | ✓ iter3_safe_repair_ds1000, iter6_grounded_repair_ds1000 | ✗ iter5_strong_dualcheck_ds1000: **365**
- ✅ NEW | ✓ iter5_strong_dualcheck_ds1000, iter6_grounded_repair_ds1000 | ✗ iter3_safe_repair_ds1000: **961**
