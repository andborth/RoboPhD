# Comparative Agent Analysis - Iteration 010

**Generated**: 2026-06-18 19:21:51

**Agents**: iter10_literal_consensus (100.0%), iter3_fmt_strong_cascade (90.0%), iter7_agree_escalate (95.0%)

## Summary
- Total questions: 20
- Consensus correct: 18 (90.0%)
- Consensus failures: 0 (0.0%)
- Split decisions: 2 (10.0%)

## Agent Accuracy

| Agent | Correct | Failed | Errors | Mean Raw Score | Mean Score |
|-------|---------|--------|--------|----------------|------------|
| iter10_literal_consensus | 20 | 0 | 0 | 1.0000 | 100.000 |
| iter3_fmt_strong_cascade | 18 | 2 | 0 | 0.9000 | 90.000 |
| iter7_agree_escalate | 19 | 1 | 0 | 0.9500 | 95.000 |

**Aggregate notes** (how Mean Score was derived from Mean Raw Score):

- **iter10_literal_consensus**: Mean cost $0.0127 within free zone (threshold $0.05); no penalty applied. Raw accuracy 1.0000 reported as percentage: 100.000.
- **iter3_fmt_strong_cascade**: Mean cost $0.0049 within free zone (threshold $0.05); no penalty applied. Raw accuracy 0.9000 reported as percentage: 90.000.
- **iter7_agree_escalate**: Mean cost $0.0073 within free zone (threshold $0.05); no penalty applied. Raw accuracy 0.9500 reported as percentage: 95.000.

## Split Decisions

Total split decisions: 2

- ✓ iter10_literal_consensus | ✗ iter3_fmt_strong_cascade, iter7_agree_escalate: **445**
- ✓ iter10_literal_consensus, iter7_agree_escalate | ✗ iter3_fmt_strong_cascade: **706**
