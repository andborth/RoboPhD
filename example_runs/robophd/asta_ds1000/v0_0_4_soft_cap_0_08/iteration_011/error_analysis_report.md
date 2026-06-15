# Comparative Agent Analysis - Iteration 011

**Generated**: 2026-06-15 00:15:10

**Agents**: iter11_ds1000_tridtype_judge (100.0%), iter3_ds1000_ensemble_judge (90.0%), iter8_ds1000_strongjudge (95.0%)

## Summary
- Total questions: 20
- Consensus correct: 18 (90.0%)
- Consensus failures: 0 (0.0%)
- Split decisions: 2 (10.0%)

## Agent Accuracy

| Agent | Correct | Failed | Errors | Mean Raw Score | Mean Score |
|-------|---------|--------|--------|----------------|------------|
| iter11_ds1000_tridtype_judge | 20 | 0 | 0 | 1.0000 | 100.000 |
| iter3_ds1000_ensemble_judge | 18 | 2 | 0 | 0.9000 | 90.000 |
| iter8_ds1000_strongjudge | 19 | 1 | 0 | 0.9500 | 95.000 |

**Aggregate notes** (how Mean Score was derived from Mean Raw Score):

- **iter11_ds1000_tridtype_judge**: Mean cost $0.0450 within free zone (threshold $0.08); no penalty applied. Raw accuracy 1.0000 reported as percentage: 100.000.
- **iter3_ds1000_ensemble_judge**: Mean cost $0.0114 within free zone (threshold $0.08); no penalty applied. Raw accuracy 0.9000 reported as percentage: 90.000.
- **iter8_ds1000_strongjudge**: Mean cost $0.0208 within free zone (threshold $0.08); no penalty applied. Raw accuracy 0.9500 reported as percentage: 95.000.

## Split Decisions

Total split decisions: 2

- ✓ iter11_ds1000_tridtype_judge | ✗ iter3_ds1000_ensemble_judge, iter8_ds1000_strongjudge: **165**
- ✓ iter11_ds1000_tridtype_judge, iter8_ds1000_strongjudge | ✗ iter3_ds1000_ensemble_judge: **142**
