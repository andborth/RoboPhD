#!/usr/bin/env python3
"""
Clean up short/experimental/failed runs.

Targets, by engine:
  - robophd/      : last_completed_iteration < --min-iterations (abandoned early)
  - gepa/         : no best_agent/ directory (the run produced no agent)
  - autoresearch/ : no best_agent/ directory (the run produced no agent)

  gepa/autoresearch are judged by best_agent/ presence (not a candidate/
  iteration count): a run that explored few candidates but still produced a
  best agent is valid. The agent file name varies by task (agent.py,
  eval_instructions.md, system_prompt.md, ...), so the directory is checked,
  not a specific file.

Safety:
  - runs referenced by a symlink under <runs-dir>/results/ (e.g. a recorded
    result's results/runs/<id> link) are never cleaned, so cleanup can't
    orphan a recorded result.
  - runs modified within --min-age-hours are skipped (they may still be
    running); this guards every engine, since a mid-run gepa/autoresearch run
    has no best_agent/ yet and a mid-run robophd run looks short.

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
import time
from pathlib import Path

# Resolve default relative to repo root, not CWD
_PROJECT_ROOT = Path(__file__).parent.parent

# Directory names under an engine subdir that are NOT evolution runs and must
# never be treated as cleanup targets (archives, seed/scratch dirs, etc.).
IGNORE_DIR_NAMES = {"archived"}


def is_ignored_dir(run_dir: Path) -> bool:
    """True for non-run helper dirs: the ignore-list or any '_'-prefixed name."""
    name = run_dir.name
    return name in IGNORE_DIR_NAMES or name.startswith("_")


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


def has_best_agent(run_dir: Path) -> bool:
    """True if the run produced a non-empty best_agent/ directory.

    The agent artifact's filename varies by task (agent.py, eval_instructions.md,
    system_prompt.md, tools/, ...), so success is judged by the directory having
    any content, not by a specific file name.
    """
    ba = run_dir / "best_agent"
    try:
        return ba.is_dir() and any(ba.iterdir())
    except OSError:
        return False


def newest_mtime(path: Path) -> float:
    """Most-recent mtime of the dir or any non-symlink file under it (0 if none).

    Used as an 'is it still being written?' signal: an actively-running run
    writes continuously, so a recent mtime means it may still be in progress.
    """
    latest = 0.0
    try:
        latest = path.stat().st_mtime
    except OSError:
        return latest
    for f in path.rglob("*"):
        try:
            if not f.is_symlink():
                latest = max(latest, f.stat().st_mtime)
        except OSError:
            continue
    return latest


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
        if not run_dir.is_dir() or is_ignored_dir(run_dir):
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


def scan_gepa_runs(runs_dir: Path) -> list[dict]:
    """Scan gepa/ for FAILED runs — ones that never produced a best_agent/.

    The candidate count is NOT used as the criterion: a run that explored only
    a few candidates but still produced a best agent is a valid result. A run
    is cleanup-worthy only if it has no non-empty best_agent/ directory.
    """
    return _scan_by_best_agent(runs_dir, "gepa")


def scan_autoresearch_runs(runs_dir: Path) -> list[dict]:
    """Scan autoresearch/ for FAILED runs — no best_agent/ directory produced.

    Autoresearch has no iteration/candidate ladder, so success is judged the
    same way as gepa: did it produce a (non-empty) best_agent/?
    """
    return _scan_by_best_agent(runs_dir, "autoresearch")


def _scan_by_best_agent(runs_dir: Path, engine: str) -> list[dict]:
    """Flag runs under <runs-dir>/<engine>/ that lack a non-empty best_agent/."""
    results = []
    engine_dir = runs_dir / engine
    if not engine_dir.exists():
        return results
    for run_dir in sorted(engine_dir.iterdir()):
        if not run_dir.is_dir() or is_ignored_dir(run_dir):
            continue
        if not has_best_agent(run_dir):
            results.append({
                "path": run_dir,
                "type": engine,
                "detail": "no best_agent/",
                "size": get_dir_size(run_dir),
            })
    return results


def collect_referenced_run_dirs(runs_dir: Path) -> set[str]:
    """Realpath targets of every symlink under <runs-dir>/results/.

    A run that is the target of such a symlink is referenced by a recorded
    result (e.g. results/runs/<id> -> <run_dir>), so it must NOT be cleaned
    even if it looks short/failed — that would orphan a recorded result.
    Dangling symlinks still resolve to their intended absolute target, so a
    run is protected as soon as any results/ symlink names it.
    """
    referenced: set[str] = set()
    results_dir = runs_dir / "results"
    if not results_dir.exists():
        return referenced
    for p in results_dir.rglob("*"):
        try:
            if p.is_symlink():
                referenced.add(os.path.realpath(p))
        except OSError:
            continue
    return referenced


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
        help="Minimum iterations for a robophd run to keep (default: 5). gepa/autoresearch use best_agent/ presence instead.",
    )
    parser.add_argument(
        "--min-age-hours",
        type=float,
        default=2.0,
        help="Never clean a run modified within this many hours — it may still "
             "be running (default: 2.0)",
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
    gepa_targets = scan_gepa_runs(runs_dir)
    autoresearch_targets = scan_autoresearch_runs(runs_dir)
    candidates_to_clean = robophd_targets + gepa_targets + autoresearch_targets

    # Safety: never clean a run that a results/ symlink points to.
    referenced = collect_referenced_run_dirs(runs_dir)
    protected = [t for t in candidates_to_clean
                 if os.path.realpath(t["path"]) in referenced]
    all_targets = [t for t in candidates_to_clean
                   if os.path.realpath(t["path"]) not in referenced]

    if protected:
        print(f"Protected {len(protected)} run(s) referenced by a results/ symlink "
              f"(will NOT be cleaned):")
        for t in protected:
            print(f"  {t['path'].relative_to(runs_dir)}")
        print()

    # Safety: never clean a run that's been written to recently — it may still
    # be running (a mid-run robophd/gepa run looks short; a mid-run autoresearch
    # run has no best_agent/agent.py yet).
    min_age = args.min_age_hours * 3600
    now = time.time()
    active, fresh_enough = [], []
    for t in all_targets:
        (active if now - newest_mtime(t["path"]) < min_age else fresh_enough).append(t)
    all_targets = fresh_enough
    if active:
        print(f"Skipped {len(active)} run(s) modified in the last {args.min_age_hours:g}h "
              f"(possibly still running):")
        for t in active:
            print(f"  {t['path'].relative_to(runs_dir)}")
        print()

    # Report
    if not all_targets:
        print(f"No cleanable runs found (robophd: < {threshold} iterations; "
              f"gepa/autoresearch: no best_agent/)")
        print(f"Scanned: {runs_dir}")
        return

    total_size = sum(t["size"] for t in all_targets)
    print(f"Found {len(all_targets)} cleanable run(s) "
          f"(robophd: < {threshold} iterations; gepa/autoresearch: no best_agent/):\n")

    for t in all_targets:
        rel = t["path"].relative_to(runs_dir)
        if t["type"] == "robophd":
            detail = f"{t['iterations']} iterations"
        else:  # gepa / autoresearch
            detail = t["detail"]
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
