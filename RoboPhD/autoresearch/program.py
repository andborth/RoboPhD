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

## Candidate files

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
Each training evaluation costs 1 budget unit per example.
Each validation evaluation costs {val_size} budget (entire val set).
Total budget: {evaluation_budget} evaluator calls.

**Important**: Validation returns only the mean score — no diagnostics or per-example
breakdown. Use training evaluations to study failure modes in detail, then validate
to confirm overall improvement.

## The experiment loop

**The goal is simple: get the highest val score.** Everything is fair game: rewrite
the candidate files however you want. The only constraint is that the evaluator can
still score your candidate.

Always start and end with a validation sweep. Everything in between is up to you.

**First**: Run `python _evaluate.py val` on the unmodified candidate to establish the
baseline. Record this as experiment 0 in the log. This gives you a concrete number
to beat.

**Last**: Reserve exactly {val_size} budget for a final validation sweep. When your
remaining budget equals {val_size}, stop experimenting and run one last
`python _evaluate.py val`. Commit or revert based on the result.

**In between**: Use the remaining budget however you see fit — training evals,
validation sweeps, whatever mix works. The loop below is a suggested rhythm,
not a rigid protocol.

LOOP:

1. Check state: `git log --oneline`, `python _evaluate.py budget`
2. Research: run one or more rounds of training evaluation
   (`python _evaluate.py train <id> ...`), read diagnostics, form a hypothesis
3. Edit the candidate files to implement your idea
4. Run a validation sweep: `python _evaluate.py val`
5. Log results to `_experiment_log.jsonl`:
   ```json
   {{"experiment": N, "hypothesis": "...", "val_score": X.XXX, "baseline_val_score": X.XXX, "kept": true/false, "budget_remaining": N}}
   ```
6. If val score improved: `git add -A && git commit -m "Experiment N: <description> (val=X.XXX)"`
7. If val score did not improve: `git checkout -- .` to revert all changes

The idea is that you are a completely autonomous researcher trying things out.
If they work, keep. If they don't, discard. And you're advancing the branch so
that you can iterate. If you feel like you're getting stuck in some way, you can
rewind, but you should probably do this very sparingly (if ever).

The best candidate is whatever is in the workspace at HEAD when you stop.

**Timeout and crashes**: If your session gets interrupted or crashes, the
infrastructure will resume you automatically. Your git commits and experiment log
are durable — check them on resume to see where you left off and continue from there.

**NEVER STOP**: Do NOT pause to ask the human if you should continue. Do NOT ask
"should I keep going?" or "is this a good stopping point?". The human might be
asleep or away from the computer and expects you to continue working until your
budget is exhausted or you are manually stopped. You are autonomous. If you run
out of ideas, think harder — re-read the diagnostics, try combining previous
near-misses, try more radical changes. The loop runs until the budget runs out.

"""
