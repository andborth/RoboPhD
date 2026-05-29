## Task: Sudoku Solver Evolution

You are evolving `agent.py`, a Python file with a `solve()` function that
solves 9x9 Sudoku puzzles.

### Agent Interface

```python
def solve(puzzle: str) -> str:
```

Parameters:
- `puzzle`: 81-character string representing a 9x9 Sudoku grid, read
  left-to-right, top-to-bottom. Digits '1'-'9' are given clues; '.' marks
  empty cells.

Returns: 81-character string with all cells filled (digits '1'-'9').

### Example

```
Input:  "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79"
Output: "534678912672195348198342567859761423426853791713924856961537284287419635345286179"
```

### Diagnostics

Any `print()` output from the agent is captured and included in evaluation
diagnostics as `agent_stdout`. Use `print()` to log any information you think
would be helpful for you to see in improving the agent in later rounds of
testing and refinement.

### How scoring works

The score is based on correctness and speed only:

1. **Correctness** (gate): If the returned solution doesn't match the expected
   answer, the score is **0.0**.
2. **Time score**: `max(0.0, 1.0 - elapsed_seconds * 100)`.
   A solve time of 0ms scores 1.0; 10ms scores 0.0. Faster is better.

Final score = time_score (if correct), 0.0 otherwise.

### Constraints

- **Pure Python only.** Do not use `ctypes`, `subprocess`, `cffi`, `os.system`,
  or any mechanism to compile or call external C/C++ code. Agents that import
  forbidden modules will score 0.
- Do not use numpy or heavy libraries — pure Python with bit tricks is fastest
  for this problem size.

### What makes a good solver

- **Constraint propagation** (naked singles, hidden singles) eliminates most
  cells without search and is very fast.
- **Backtracking with MRV** (minimum remaining values) picks the most
  constrained cell first, pruning the search tree.
- **Bitmask representations** (`1 << (digit-1)`) allow O(1) constraint checks
  via bitwise AND/OR.
- **Precomputed structures** (peers, units, box indices) avoid repeated
  calculation during search.

### Dataset

Puzzles come from the `sapientinc/sudoku-extreme` dataset on HuggingFace.
Difficulty ratings range from 0 (easy) to 465 (extreme). The training set
includes a stratified sample across difficulty levels.

## Scratch space

Your working directory is your iteration's evolution dir. If you need to drop a small test harness or scratch script while debugging, write it there directly — `/tmp` is outside the write scope, and there's no need to clean up after yourself afterwards. Leftover debugging artifacts in the iteration dir are useful for retrospective analysis.
