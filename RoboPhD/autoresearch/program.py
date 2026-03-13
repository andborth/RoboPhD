"""
Generate program.md — the autoresearch agent's operating manual.

Domain-agnostic protocol. Injects task-specific details (objective,
background, file structure, budget) into the template.
"""


def generate_program(
    task,
    file_mapping: dict[str, str],
    train_example_ids: list[str],
    val_size: int,
    evaluation_budget: int,
) -> str:
    """Generate the program.md content for the autoresearch agent."""

    # Build file structure description
    file_lines = []
    for key, path in file_mapping.items():
        file_lines.append(f"  - `{path}` (candidate key: `{key}`)")
    file_structure = "\n".join(file_lines)

    return f"""# Autoresearch Program

## Objective

{task.objective}

## Candidate Files

Your candidate is defined by these files in the workspace root:

{file_structure}

Modify these files to improve performance. You can use any packages installed
in the current Python environment (not limited to stdlib).

## Evaluation

Use `_evaluate.py` to evaluate your candidate:

```bash
# Score on specific training examples (returns scores + diagnostics)
python _evaluate.py train <id> [<id> ...]

# Score on held-out validation set (returns only mean_score, no per-example data)
python _evaluate.py val

# Check remaining budget without consuming calls
python _evaluate.py budget
```

Training example IDs are listed in `_train_examples.json` ({len(train_example_ids)} examples).
Each training evaluation costs 1 budget per example.
Each validation evaluation costs {val_size} budget (entire val set).
Total budget: {evaluation_budget} evaluator calls.

**Important**: Validation returns only the mean score — no diagnostics or per-example
breakdown. Use training evaluations to study failure modes in detail, then validate
to confirm overall improvement.

## Protocol

### 1. Baseline
Evaluate the seed candidate on the validation set to establish a baseline score.

### 2. Experiment Loop
Repeat until budget is exhausted:

1. **Study**: Evaluate on a few training examples to get detailed diagnostics.
   Analyze the diagnostics to understand failure modes.
2. **Hypothesize**: Form a specific hypothesis about what change will improve scores.
3. **Modify**: Edit the candidate files to implement your hypothesis.
4. **Evaluate (train)**: Test on training examples to verify the change helps.
5. **Evaluate (val)**: If training looks promising, run a validation sweep.
6. **Commit or revert**:
   - If val score improved: `git add -A && git commit -m "Experiment N: <description> (val=X.XXX)"`
   - If val score did not improve: `git checkout -- .` to revert all changes
7. **Log**: Append to `_experiment_log.jsonl`:
   ```json
   {{"experiment": N, "hypothesis": "...", "val_score": X.XXX, "baseline_val_score": X.XXX, "kept": true/false, "budget_remaining": N}}
   ```

### 3. Budget Management
- Check budget before expensive operations (`python _evaluate.py budget`)
- Training evaluations are cheap (1 call each) — use them to probe and diagnose
- Validation sweeps are expensive ({val_size} calls) — use sparingly to confirm gains
- When budget is exhausted, the evaluator returns an error. Stop experimenting.

### 4. Strategy Tips
- Start with a small training probe (2-3 examples) to understand the problem
- Make targeted changes based on diagnostic evidence, not guesses
- Keep changes small and testable — one hypothesis per experiment
- If a change helps on training but not validation, it may be overfitting
- The best candidate is whatever is in the workspace at HEAD when you stop

## Version Control
- Git is initialized in this workspace. The seed candidate is the initial commit.
- Always commit improvements so they aren't lost if a later experiment fails.
- Use `git log --oneline` to review your experiment history.
- Use `git diff` to see what changed since the last commit.
"""
