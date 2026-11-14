# Comparative Agent Analysis - Iteration 014

**Generated**: 2026-06-09 20:51:16

**Agents**: iter14_filemock_adjudicate (100.0%), iter6_inplace_expect_adjudicate (85.0%), iter8_refquirk_adjudicate (90.0%)

## Summary
- Total questions: 20
- Consensus correct: 17 (85.0%)
- Consensus failures: 0 (0.0%)
- Split decisions: 3 (15.0%)

## Agent Accuracy

| Agent | Correct | Failed | Errors | Mean Raw Score | Mean Score |
|-------|---------|--------|--------|----------------|------------|
| iter14_filemock_adjudicate | 20 | 0 | 0 | 1.0000 | 100.000 |
| iter6_inplace_expect_adjudicate | 17 | 3 | 0 | 0.8500 | 85.000 |
| iter8_refquirk_adjudicate | 18 | 2 | 0 | 0.9000 | 90.000 |

**Aggregate notes** (how Mean Score was derived from Mean Raw Score):

- **iter14_filemock_adjudicate**: Mean cost $0.0537 within free zone (threshold $0.06); no penalty applied. Raw accuracy 1.0000 reported as percentage: 100.000.
- **iter6_inplace_expect_adjudicate**: Mean cost $0.0522 within free zone (threshold $0.06); no penalty applied. Raw accuracy 0.8500 reported as percentage: 85.000.
- **iter8_refquirk_adjudicate**: Mean cost $0.0516 within free zone (threshold $0.06); no penalty applied. Raw accuracy 0.9000 reported as percentage: 90.000.

## Split Decisions

Total split decisions: 3

- ✓ iter14_filemock_adjudicate | ✗ iter6_inplace_expect_adjudicate, iter8_refquirk_adjudicate: **269**, **883**
- ✓ iter14_filemock_adjudicate, iter8_refquirk_adjudicate | ✗ iter6_inplace_expect_adjudicate: **284**
