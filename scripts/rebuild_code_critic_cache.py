#!/usr/bin/env python3
"""
Rebuild the code_critic cache, ensuring every entry has all 4 canonical files:
  problem.md, meta.json, solution.py, reflection.md

Usage:
  python scripts/rebuild_code_critic_cache.py [--cache-dir PATH] [--coder-model MODEL] [--limit N] [--dry-run]

Steps:
  1. Bootstrap from HuggingFace if cache is empty (problem.md + meta.json)
  2. For entries missing solution.py OR reflection.md: regenerate both in one
     fresh Claude CLI session (call 1: solution, call 2: reflection)
  3. Report summary

Idempotent: safe to re-run; only touches incomplete entries.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add project root to path (for RoboPhD.* and utilities.* imports)
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

from RoboPhD.config import CLAUDE_CLI_MODEL_MAP

DEFAULT_CACHE_DIR = PROJECT_ROOT / "RoboPhD" / "data" / "code_critic" / "cache"

# Reuse the exact prompt from run_critic_evaluation.py
CALL_1_PROMPT = """Read problem.md and create solution.py to solve it.

Requirements:
- If examples show "Input: variable = value" format and ask you to "Return" something:
  Create a Solution class with the appropriate method (e.g., `class Solution: def solve(self, nums): ...`)
- If the problem has "Input" and "Output" sections describing line-by-line format:
  Create a standalone script that reads from stdin and writes to stdout
- Code must run in under 6 seconds on every test case
- Include brief comments explaining your approach"""

REFLECTION_PROMPT = (
    "Create reflection.md describing your solution approach. "
    "Include a 'Categories' section listing one or more problem categories that apply."
)


def find_claude_cli() -> str:
    """Find Claude CLI path."""
    local_claude = Path.home() / ".claude" / "local" / "claude"
    if local_claude.exists():
        return str(local_claude)
    result = subprocess.run(["which", "claude"], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    raise RuntimeError("Claude CLI not found")


def run_claude_code(
    claude_path: str,
    prompt: str,
    working_dir: Path,
    model: str,
    timeout: int = 1200,
    session_id: str | None = None,
) -> tuple[dict, str | None]:
    """Run Claude Code CLI. Returns (result_dict, session_id)."""
    cli_model = CLAUDE_CLI_MODEL_MAP.get(model, model)

    settings = {"env": {"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "128000"}}
    cmd = [
        claude_path, "--print", "--output-format", "json",
        "--append-system-prompt", "Do not read or write MEMORY.md files.",
        "--permission-mode", "bypassPermissions",
        "--settings", json.dumps(settings),
        "--add-dir", str(working_dir.resolve()),
        "--model", cli_model,
    ]

    if session_id:
        cmd.extend(["--resume", session_id])

    cmd.extend(["-p", prompt])

    # Build env without CLAUDECODE to allow nested CLI invocations
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    result = subprocess.run(
        cmd, cwd=working_dir, capture_output=True, text=True, timeout=timeout, env=env,
    )

    output = {}
    if result.stdout:
        try:
            output = json.loads(result.stdout)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON output: {result.stdout[:200]}")

    if result.returncode != 0 and result.stderr:
        logger.warning(f"CLI stderr: {result.stderr[:500]}")

    sid = output.get("session_id")
    return output, sid


def regenerate_entry(
    claude_path: str,
    entry_dir: Path,
    model: str,
    timeout: int,
) -> bool:
    """
    Regenerate solution.py and reflection.md for a single entry.

    Uses a temporary workspace so we don't pollute the cache dir with
    session artifacts. Copies results back on success.
    Returns True on success.
    """
    question_id = entry_dir.name

    # Work in a temp directory, symlink problem.md
    workspace = entry_dir / "_workspace"
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir()

    try:
        # Symlink problem.md into workspace
        (workspace / "problem.md").symlink_to((entry_dir / "problem.md").resolve())

        # Call 1: Generate solution
        logger.info(f"  [{question_id}] generating solution.py")
        _, session_id = run_claude_code(
            claude_path, CALL_1_PROMPT, workspace, model, timeout=timeout,
        )

        if not session_id:
            logger.error(f"  [{question_id}] call 1 failed: no session_id")
            return False

        if not (workspace / "solution.py").exists():
            logger.error(f"  [{question_id}] call 1 failed: no solution.py created")
            return False

        # Call 2: Generate reflection (same session)
        logger.info(f"  [{question_id}] generating reflection.md")
        run_claude_code(
            claude_path, REFLECTION_PROMPT, workspace, model,
            timeout=timeout, session_id=session_id,
        )

        if not (workspace / "reflection.md").exists():
            logger.error(f"  [{question_id}] call 2 failed: no reflection.md created")
            return False

        # Delete stale files from cache entry before copying fresh ones
        for fname in ("solution.py", "reflection.md"):
            target = entry_dir / fname
            if target.exists():
                target.unlink()

        # Copy results back
        shutil.copy(workspace / "solution.py", entry_dir / "solution.py")
        shutil.copy(workspace / "reflection.md", entry_dir / "reflection.md")

        logger.info(f"  [{question_id}] OK")
        return True

    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Rebuild code_critic cache")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--coder-model", default="haiku-4.5")
    parser.add_argument("--limit", type=int, default=0, help="Max entries to regenerate (0=all)")
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    cache_dir = args.cache_dir

    # Step 1: Bootstrap from HuggingFace if cache is empty
    if not cache_dir.exists() or not any(cache_dir.iterdir()):
        logger.info("Cache empty — bootstrapping from HuggingFace...")
        from RoboPhD.adapters.gepa_codegen import _bootstrap_cache
        _bootstrap_cache(cache_dir)

    # Step 2: Find incomplete entries
    total = 0
    incomplete = []
    for entry_dir in sorted(cache_dir.iterdir()):
        if not entry_dir.is_dir() or entry_dir.name.startswith("."):
            continue
        if not (entry_dir / "meta.json").exists():
            continue
        total += 1
        if not (entry_dir / "solution.py").exists() or not (entry_dir / "reflection.md").exists():
            incomplete.append(entry_dir)

    logger.info(f"Cache: {total} entries, {len(incomplete)} incomplete")

    if not incomplete:
        logger.info("All entries complete. Nothing to do.")
        return

    if args.dry_run:
        for entry_dir in incomplete:
            missing = []
            if not (entry_dir / "solution.py").exists():
                missing.append("solution.py")
            if not (entry_dir / "reflection.md").exists():
                missing.append("reflection.md")
            logger.info(f"  Would regenerate {entry_dir.name}: missing {', '.join(missing)}")
        logger.info(f"Dry run: {len(incomplete)} entries would be regenerated")
        return

    # Step 3: Regenerate
    claude_path = find_claude_cli()
    to_process = incomplete[:args.limit] if args.limit > 0 else incomplete
    logger.info(f"Regenerating {len(to_process)} entries (max_concurrent={args.max_concurrent})")

    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=args.max_concurrent) as executor:
        futures = {
            executor.submit(
                regenerate_entry, claude_path, entry_dir, args.coder_model, args.timeout,
            ): entry_dir
            for entry_dir in to_process
        }

        for future in as_completed(futures):
            entry_dir = futures[future]
            try:
                if future.result():
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                logger.error(f"  [{entry_dir.name}] exception: {e}")
                fail_count += 1

    logger.info(f"Done: {success_count} regenerated, {fail_count} failed, {total} total")


if __name__ == "__main__":
    main()
