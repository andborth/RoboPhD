"""
Baseline Sudoku solver — simple recursive backtracking.

Deliberately minimal: no constraint propagation, no bitmasks, no heuristics.
Finds the first empty cell and tries digits 1-9, backtracking on conflicts.
Correct but slow on hard puzzles.
"""


def solve(puzzle):
    """Solve an 81-char Sudoku puzzle string. Returns 81-char solution."""
    grid = list(puzzle)

    def is_valid(grid, pos, digit):
        row, col = divmod(pos, 9)
        # Check row
        for c in range(9):
            if grid[row * 9 + c] == digit:
                return False
        # Check column
        for r in range(9):
            if grid[r * 9 + col] == digit:
                return False
        # Check 3x3 box
        box_r, box_c = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_r, box_r + 3):
            for c in range(box_c, box_c + 3):
                if grid[r * 9 + c] == digit:
                    return False
        return True

    def backtrack(grid):
        # Find first empty cell
        for i in range(81):
            if grid[i] == '.':
                for d in "123456789":
                    if is_valid(grid, i, d):
                        grid[i] = d
                        if backtrack(grid):
                            return True
                        grid[i] = '.'
                return False
        return True

    print(f"Givens: {sum(1 for c in puzzle if c != '.')}")
    backtrack(grid)
    result = ''.join(grid)
    print(f"Solved: {'.' not in result}")
    return result
