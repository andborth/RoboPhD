# Comparative Agent Analysis - Iteration 003

**Generated**: 2026-05-15 01:14:39

**Agents**: iter2_ds1000_verify_repair (75.0%), iter3_ds1000_format_aware (80.0%), seed_yyg6m9ud (65.0%)

## Summary
- Total questions: 20
- Consensus correct: 12 (60.0%)
- Consensus failures: 4 (20.0%)
- Split decisions: 4 (20.0%)

## Agent Accuracy

| Agent | Correct | Failed | Errors | Mean Raw Score | Mean Score |
|-------|---------|--------|--------|----------------|------------|
| iter2_ds1000_verify_repair | 15 | 5 | 0 | 0.7500 | 75.000 |
| iter3_ds1000_format_aware | 16 | 4 | 0 | 0.8000 | 80.000 |
| seed_yyg6m9ud | 13 | 7 | 0 | 0.6500 | 65.000 |

**Aggregate notes** (how Mean Score was derived from Mean Raw Score):

- **iter2_ds1000_verify_repair**: Mean cost $0.0051 within free zone (threshold $0.08); no tiebreaker penalty applied. Raw accuracy 0.7500 reported as percentage: 75.000.
- **iter3_ds1000_format_aware**: Mean cost $0.0096 within free zone (threshold $0.08); no tiebreaker penalty applied. Raw accuracy 0.8000 reported as percentage: 80.000.
- **seed_yyg6m9ud**: Mean cost $0.0005 within free zone (threshold $0.08); no tiebreaker penalty applied. Raw accuracy 0.6500 reported as percentage: 65.000.

## Consensus Failures

All agents failed on 4 questions: 706, 165, 269, 420

## Split Decisions

Total split decisions: 4

- ✓ iter2_ds1000_verify_repair, iter3_ds1000_format_aware | ✗ seed_yyg6m9ud: **426**, **763**, **887**
- ✓ iter3_ds1000_format_aware, seed_yyg6m9ud | ✗ iter2_ds1000_verify_repair: **723**
