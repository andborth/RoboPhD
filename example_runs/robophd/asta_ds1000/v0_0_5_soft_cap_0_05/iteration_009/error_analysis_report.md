# Comparative Agent Analysis - Iteration 009

**Generated**: 2026-06-18 19:06:44

**Agents**: iter3_fmt_strong_cascade (80.0%), iter7_agree_escalate (80.0%), iter9_reason_agree (80.0%)

## Summary
- Total questions: 20
- Consensus correct: 15 (75.0%)
- Consensus failures: 2 (10.0%)
- Split decisions: 3 (15.0%)

## Agent Accuracy

| Agent | Correct | Failed | Errors | Mean Raw Score | Mean Score |
|-------|---------|--------|--------|----------------|------------|
| iter3_fmt_strong_cascade | 16 | 4 | 0 | 0.8000 | 80.000 |
| iter7_agree_escalate | 16 | 4 | 0 | 0.8000 | 80.000 |
| iter9_reason_agree | 16 | 4 | 0 | 0.8000 | 80.000 |

**Aggregate notes** (how Mean Score was derived from Mean Raw Score):

- **iter3_fmt_strong_cascade**: Mean cost $0.0042 within free zone (threshold $0.05); no penalty applied. Raw accuracy 0.8000 reported as percentage: 80.000.
- **iter7_agree_escalate**: Mean cost $0.0104 within free zone (threshold $0.05); no penalty applied. Raw accuracy 0.8000 reported as percentage: 80.000.
- **iter9_reason_agree**: Mean cost $0.0221 within free zone (threshold $0.05); no penalty applied. Raw accuracy 0.8000 reported as percentage: 80.000.

## Consensus Failures

All agents failed on 2 questions: 445, 883

## Split Decisions

Total split decisions: 3

- ✓ iter3_fmt_strong_cascade | ✗ iter7_agree_escalate, iter9_reason_agree: **269**
- ✓ iter7_agree_escalate | ✗ iter3_fmt_strong_cascade, iter9_reason_agree: **706**
- ✓ iter9_reason_agree | ✗ iter3_fmt_strong_cascade, iter7_agree_escalate: **812**
