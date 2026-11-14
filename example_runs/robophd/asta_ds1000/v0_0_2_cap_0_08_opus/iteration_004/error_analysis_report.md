# Comparative Agent Analysis - Iteration 004

**Generated**: 2026-05-15 01:32:02

**Agents**: iter2_ds1000_verify_repair (95.0%), iter3_ds1000_format_aware (95.0%), iter4_ds1000_idiom_probe (100.0%)

## Summary
- Total questions: 20
- Consensus correct: 19 (95.0%)
- Consensus failures: 0 (0.0%)
- Split decisions: 1 (5.0%)

## Agent Accuracy

| Agent | Correct | Failed | Errors | Mean Raw Score | Mean Score |
|-------|---------|--------|--------|----------------|------------|
| iter2_ds1000_verify_repair | 19 | 1 | 0 | 0.9500 | 95.000 |
| iter3_ds1000_format_aware | 19 | 1 | 0 | 0.9500 | 95.000 |
| iter4_ds1000_idiom_probe | 20 | 0 | 0 | 1.0000 | 100.000 |

**Aggregate notes** (how Mean Score was derived from Mean Raw Score):

- **iter2_ds1000_verify_repair**: Mean cost $0.0066 within free zone (threshold $0.08); no tiebreaker penalty applied. Raw accuracy 0.9500 reported as percentage: 95.000.
- **iter3_ds1000_format_aware**: Mean cost $0.0120 within free zone (threshold $0.08); no tiebreaker penalty applied. Raw accuracy 0.9500 reported as percentage: 95.000.
- **iter4_ds1000_idiom_probe**: Mean cost $0.0103 within free zone (threshold $0.08); no tiebreaker penalty applied. Raw accuracy 1.0000 reported as percentage: 100.000.

## Split Decisions

Total split decisions: 1

- ✓ iter4_ds1000_idiom_probe | ✗ iter2_ds1000_verify_repair, iter3_ds1000_format_aware: **427**
