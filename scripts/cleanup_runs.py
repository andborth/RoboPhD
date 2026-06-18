#!/usr/bin/env python3
"""
Clean up short/experimental RoboPhD and GEPA runs.

Usage:
    # Dry-run: show what would be cleaned up
    python scripts/cleanup_runs.py

    # Move short runs to trash directory
    python scripts/cleanup_runs.py --move

    # Permanently delete short runs
    python scripts/cleanup_runs.py --delete

    # Custom threshold and location
    python scripts/cleanup_runs.py --min-iterations 3 --runs-dir /path/to/runs
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Resolve default relative to repo root, not CWD
_PROJECT_ROOT = Path(__file__).parent.parent


def safe_rmtree(path: Path) -> None:
    """Remove directory tree, unlinking symlinks first to avoid following them."""
    for root, dirs, files in os.walk(path, topdown=False):
        root_path = Path(root)
        for name in files:
            p = root_path / name
            if p.is_symlink():
                p.unlink()
        for name in dirs:
            p = root_path / name
            if p.is_symlink():
                p.unlink()
    shutil.rmtree(path)


def get_dir_size(path: Path) -> int:
    """Get total size of directory in bytes."""
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file() and not f.is_symlink():
                total += f.stat().st_size
    except (PermissionError, OSError):
        pass
    return total


def fmt_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f}MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f}GB"


def scan_robophd_runs(runs_dir: Path, threshold: int) -> list[dict]:
    """Scan robophd/ for short runs based on checkpoint.json."""
    results = []
    robophd_dir = runs_dir / "robophd"
    if not robophd_dir.exists():
        return results

    for run_dir in sorted(robophd_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        checkpoint = run_dir / "checkpoint.json"
        iterations = None
        if checkpoint.exists():
            try:
                data = json.loads(checkpoint.read_text())
                iterations = data.get("last_completed_iteration", 0)
            except (json.JSONDecodeError, OSError):
                iterations = 0
        else:
            # No checkpoint = never completed an iteration
            iterations = 0

        if iterations < threshold:
            size = get_dir_size(run_dir)
            results.append({
                "path": run_dir,
                "type": "robophd",
                "iterations": iterations,
                "size": size,
            })

    return results


def scan_gepa_runs(runs_dir: Path, threshold: int) -> list[dict]:
    """Scan gepa/ for short runs based on optimization_summary.json."""
    results = []
    gepa_dir = runs_dir / "gepa"
    if not gepa_dir.exists():
        return results

    for run_dir in sorted(gepa_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        summary = run_dir / "optimization_summary.json"
        candidates = None
        if summary.exists():
            try:
                data = json.loads(summary.read_text())
                candidates = data.get("num_candidates_explored", 0)
            except (json.JSONDecodeError, OSError):
                candidates = 0
        else:
            candidates = 0

        if candidates < threshold:
            size = get_dir_size(run_dir)
            results.append({
                "path": run_dir,
                "type": "gepa",
                "candidates": candidates,
                "size": size,
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="Clean up short/experimental runs")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Root directory for experiment outputs (default: ../robophd_runs relative to repo root)",
    )
    parser.add_argument(
        "--min-iterations",
        type=int,
        default=5,
        help="Minimum iterations/candidates to keep (default: 5)",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--move",
        action="store_true",
        help="Move short runs to <runs-dir>_trash/",
    )
    action.add_argument(
        "--delete",
        action="store_true",
        help="Permanently delete short runs",
    )
    args = parser.parse_args()

    if args.runs_dir is None:
        args.runs_dir = _PROJECT_ROOT.parent / "robophd_runs"
    runs_dir = args.runs_dir.resolve()
    if not runs_dir.exists():
        print(f"Runs directory not found: {runs_dir}")
        sys.exit(1)

    threshold = args.min_iterations

    # Scan
    robophd_targets = scan_robophd_runs(runs_dir, threshold)
    gepa_targets = scan_gepa_runs(runs_dir, threshold)
    all_targets = robophd_targets + gepa_targets

    # Report
    if not all_targets:
        print(f"No short runs found (threshold: {threshold} iterations/candidates)")
        print(f"Scanned: {runs_dir}")
        return

    total_size = sum(t["size"] for t in all_targets)
    print(f"Found {len(all_targets)} short run(s) below threshold ({threshold}):\n")

    for t in all_targets:
        rel = t["path"].relative_to(runs_dir)
        if t["type"] == "robophd":
            detail = f"{t['iterations']} iterations"
        else:
            detail = f"{t['candidates']} candidates"
        print(f"  {str(rel):<50s}  {detail:<20s}  {fmt_size(t['size'])}")

    print(f"\nTotal: {fmt_size(total_size)}")

    # Act
    if args.move:
        trash_dir = runs_dir.parent / f"{runs_dir.name}_trash"
        print(f"\nMoving {len(all_targets)} run(s) to {trash_dir}/...")
        for t in all_targets:
            rel = t["path"].relative_to(runs_dir)
            dest = trash_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(t["path"]), str(dest))
            print(f"  Moved {rel}")
        print("Done.")

    elif args.delete:
        total_size = sum(t["size"] for t in all_targets)
        answer = input(f"\nPermanently delete {len(all_targets)} run(s) totaling {fmt_size(total_size)}? [y/N] ")
        if answer.strip().lower() != "y":
            print("Aborted.")
            return
        for t in all_targets:
            rel = t["path"].relative_to(runs_dir)
            safe_rmtree(t["path"])
            print(f"  Deleted {rel}")
        print(f"Freed {fmt_size(total_size)}.")

    else:
        print("\nDry run — use --move or --delete to act.")


if __name__ == "__main__":
    main()
