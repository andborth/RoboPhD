# Comparative Agent Analysis - Iteration 006

**Generated**: 2026-10-14 11:03:42

**Agents**: iter2_exec_verified_sonnet (85.0%), iter3_ensemble_exec_vote (80.0%), iter6_ensemble_exec_vote_escalate (90.0%)

## Summary
- Total questions: 20
- Consensus correct: 15 (75.0%)
- Consensus failures: 1 (5.0%)
- Split decisions: 4 (20.0%)

## Agent Accuracy

| Agent | Correct | Failed | Errors | Mean Raw Score | Mean Score |
|-------|---------|--------|--------|----------------|------------|
| iter2_exec_verified_sonnet | 17 | 3 | 0 | 0.8500 | 85.000 |
| iter3_ensemble_exec_vote | 16 | 4 | 0 | 0.8000 | 80.000 |
| iter6_ensemble_exec_vote_escalate | 18 | 2 | 0 | 0.9000 | 90.000 |

**Aggregate notes** (how Mean Score was derived from Mean Raw Score):

- **iter2_exec_verified_sonnet**: Mean cost $0.0123 within free zone (threshold $0.127); no penalty applied. Raw accuracy 0.8500 reported as percentage: 85.000.
- **iter3_ensemble_exec_vote**: Mean cost $0.0397 within free zone (threshold $0.127); no penalty applied. Raw accuracy 0.8000 reported as percentage: 80.000.
- **iter6_ensemble_exec_vote_escalate**: Mean cost $0.0652 within free zone (threshold $0.127); no penalty applied. Raw accuracy 0.9000 reported as percentage: 90.000.

## Consensus Failures

All agents failed on 1 questions: 165

## Split Decisions

Total split decisions: 4

- ✓ iter2_exec_verified_sonnet | ✗ iter3_ensemble_exec_vote, iter6_ensemble_exec_vote_escalate: **906**
- ✓ iter2_exec_verified_sonnet, iter6_ensemble_exec_vote_escalate | ✗ iter3_ensemble_exec_vote: **822**
- ✓ iter3_ensemble_exec_vote, iter6_ensemble_exec_vote_escalate | ✗ iter2_exec_verified_sonnet: **883**
- ✓ iter6_ensemble_exec_vote_escalate | ✗ iter2_exec_verified_sonnet, iter3_ensemble_exec_vote: **238**
