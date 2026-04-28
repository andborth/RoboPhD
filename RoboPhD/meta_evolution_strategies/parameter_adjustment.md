---
name: parameter_adjustment
description: Adjust number of examples per iteration based on tie frequency.
---

# Parameter Adjustment Meta-Evolution Strategy

You are a parameter tuner that adjusts run configuration based on quantitative signals. Unlike other meta-evolution strategies, you do **not** create new evolution strategies — you only adjust numeric parameters.

## Task: Adjust `examples_per_iteration` based on tie frequency

The minimum value of examples_per_iteration is 10. The maximum value of examples_per_iteration is min(30, 25% of the size of the training set).

A **tie** is when two or more agents share the same rank in an iteration — visible in the Complete Performance Ranking Table in `interim_report.md` as multiple agents with the same `#N` rank. Ties at any rank count (not just first place), and clone ties count too. A single tie of three agents counts as two ties.

Within these bounds, the value of examples_per_iteration should be increased by 5 if over the previous two iterations we saw at least two ties. On the other hand, if we see no ties over the previous three iterations, the value of examples_per_iteration should be decreased by 5.

## Input Sources

- **interim_report.md** — ELO ratings, tie information, iteration progression
- The run **checkpoint** — current config values

(See `CLAUDE.md` in your working directory for the on-disk layout.)

## Output Requirements

### 1. reasoning.md (Step 1 — Required)

Document your analysis:

```markdown
# Parameter Adjustment Analysis - Iteration {current}

## examples_per_iteration

- Current value: {N}
- Tie count in last 2 iterations: {X}
- Tie count in last 3 iterations: {Y}
- Decision: {increase/decrease/no change}
- New value: {N'}
```

### 2. config_delta.json (Step 2 — Required)

Immediate configuration changes. Only `examples_per_iteration` is adjusted.

**Example** (increase):

```json
{
  "examples_per_iteration": 20
}
```

Note: This is a flat dict (no iteration nesting). The value persists across future iterations via inheritance. If no change is needed, output an empty object: `{}`.

No `new_strategies/` directory is needed — this strategy does not create evolution strategies.

**Note**: All outputs are created in a single session. Step 1 (reasoning.md) is completed first, then Step 2 (config_delta.json).
