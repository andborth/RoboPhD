# Comparative Agent Analysis - Iteration 005

**Generated**: 2026-10-14 10:31:40

**Agents**: iter2_exec_verified_sonnet (100.0%), iter3_ensemble_exec_vote (100.0%), iter5_ensemble_exec_vote_dtype_judge (100.0%)

## Summary
- Total questions: 20
- Consensus correct: 20 (100.0%)
- Consensus failures: 0 (0.0%)
- Split decisions: 0 (0.0%)

## Agent Accuracy

| Agent | Correct | Failed | Errors | Mean Raw Score | Mean Score |
|-------|---------|--------|--------|----------------|------------|
| iter2_exec_verified_sonnet | 20 | 0 | 0 | 1.0000 | 100.000 |
| iter3_ensemble_exec_vote | 20 | 0 | 0 | 1.0000 | 100.000 |
| iter5_ensemble_exec_vote_dtype_judge | 20 | 0 | 0 | 1.0000 | 100.000 |

**Aggregate notes** (how Mean Score was derived from Mean Raw Score):

- **iter2_exec_verified_sonnet**: Mean cost $0.0134 within free zone (threshold $0.127); no penalty applied. Raw accuracy 1.0000 reported as percentage: 100.000.
- **iter3_ensemble_exec_vote**: Mean cost $0.0450 within free zone (threshold $0.127); no penalty applied. Raw accuracy 1.0000 reported as percentage: 100.000.
- **iter5_ensemble_exec_vote_dtype_judge**: Mean cost $0.0480 within free zone (threshold $0.127); no penalty applied. Raw accuracy 1.0000 reported as percentage: 100.000.

## Clone Detection

⚠️ **iter5_ensemble_exec_vote_dtype_judge** identified as exact clone of **iter2_exec_verified_sonnet** (identical scores on all 20 problems). Elo penalized by 200; excluded from winner selection.
