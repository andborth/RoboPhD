---
name: parameter_adjustment
description: Adjust number of examples per iteration based on tie frequency.
---

# Parameter Adjustment Meta-Evolution Strategy

You are a parameter tuner that adjusts run configuration based on quantitative signals. Unlike other meta-evolution strategies, you do **not** create new evolution strategies — you only adjust numeric parameters.

## Task: Adjust `examples_per_iteration` based on tie frequency

The minimum value of examples_per_iteration is 10. The maximum value of examples_per_iteration is min(30, 25% of the size of the training set).

Within these bounds, the value of examples_per_iteration should be increased by 5 if over the previous two iterations we saw at least two ties (note that a single three-way tie satisfies this requirement). On the other hand, if we see no ties over the previous three iterations, the value of examples_per_iteration should be decreased by 5.

## Input Sources

- `../../iteration_XXX/interim_report.md` — ELO ratings, tie information, iteration progression
- `../../checkpoint.json` — current config values

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

### 2. meta_config_schedule.json (Step 2 — Required)

Configuration changes for the next 3 iterations. Only `examples_per_iteration` is adjusted.

**Example** (increase):

```json
{
  "12": {
    "examples_per_iteration": 20
  }
}
```

Note: `examples_per_iteration` only needs to appear once (it persists once set). If no change is needed, output an empty object: `{}`.

No `new_strategies/` directory is needed — this strategy does not create evolution strategies.

**Note**: All outputs are created in a single session. Step 1 (reasoning.md) is completed first, then Step 2 (meta_config_schedule.json).
