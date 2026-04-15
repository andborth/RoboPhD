#!/usr/bin/env python3
"""Minimal problem analyzer stub. Basic stats for the critic."""
from pathlib import Path


def main():
    output_dir = Path("tool_output")
    output_dir.mkdir(exist_ok=True)

    problem = Path("problem.md").read_text() if Path("problem.md").exists() else ""
    solution = Path("solution.py").read_text() if Path("solution.py").exists() else ""

    problem_lines = len(problem.splitlines()) if problem else 0
    solution_lines = len(solution.splitlines()) if solution else 0

    with open(output_dir / "analysis.txt", "w") as f:
        f.write(f"Problem: {problem_lines} lines\n")
        f.write(f"Solution: {solution_lines} lines\n")


if __name__ == "__main__":
    main()
