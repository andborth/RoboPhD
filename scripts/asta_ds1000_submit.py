#!/usr/bin/env python3
"""
DS-1000 → AstaBench leaderboard submission runner.

Run from anywhere — paths are resolved relative to the repo root:

    python scripts/asta_ds1000_submit.py

Per submission, the script:
  1. Verifies the working tree is clean (otherwise eval_spec.revision.dirty
     would record `true`, weakening the reproducibility claim).
  2. Stages a working dir at submissions/asta_ds1000/<dir>/ (gitignored)
     with agent.py copied from example_runs/ and model_registry.py copied
     from examples/asta_ds1000/.
  3. Runs `astabench eval` against DS_1000_test (no --ignore-git).
  4. Runs `astabench score` to produce scores.json + summary_stats.json.
  5. Tarballs logs/full_test/ for HuggingFace upload.

After all submissions, prints an accuracy summary parsed from each scores.json.

Idempotent: skips eval if logs/full_test/*.eval already exists for that
submission. submissions/ is gitignored so the working dirs don't dirty
the tree.

Cost / time, sequential:
    v0_seed_gpt54_mini    ~$0.50    30-60 min   (gpt-5.4-mini one-shot)
    v0_seed_sonnet_4_6    ~$2.20    30-60 min   (claude-sonnet-4-6 one-shot)
    v0_soft_cap_0_04      ~$23.00   2-4 hr      (4-candidate jury w/ smoke-test)
    --
    total                 ~$25.70   4-6 hr

Prerequisites:
  - OPENAI_API_KEY              (required by all three; v0_soft_cap_0_04
                                 calls it as one of four candidates)
  - ANTHROPIC_API_KEY           (or ANTHROPIC_API_KEY_FOR_ROBOPHD;
                                 model_registry reads either)
  - GOOGLE_API_KEY              (v0_soft_cap_0_04's Gemini candidate;
                                 mostly times out but the registry import
                                 needs the env var present)
  - Docker daemon running       (DS-1000 sandbox tier requires it)
  - LITELLM_LOCAL_MODEL_COST_MAP=True is set inside subprocesses

The script does NOT submit. After it finishes, three .tar.gz files are
ready for manual upload via the HF Spaces leaderboard form.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

# Repo root (this script lives at <repo>/scripts/).
REPO = Path(__file__).resolve().parent.parent

EXAMPLES_DIR = REPO / "examples" / "asta_ds1000"
SOURCE_BASE = REPO / "example_runs" / "robophd" / "asta_ds1000"
WORKING_BASE = REPO / "submissions" / "asta_ds1000"
LOG_SUBDIR = "logs/full_test"


class Submission(NamedTuple):
    # Dir name under both example_runs/robophd/asta_ds1000/ and
    # submissions/asta_ds1000/. Becomes the .tar.gz basename too.
    name: str
    # agent.py path relative to the example_runs source dir for this
    # submission. Evolved runs nest under agents/<agent_name>/; seeds
    # are flat at the dir root.
    agent_rel_path: str
    # --model arg passed to `astabench eval`. For multi-model agents
    # we use `none` so the recorded eval.model field doesn't claim a
    # single primary; per-call usage is captured in stats.model_usage.
    model_arg: str


SUBMISSIONS = [
    # Cheapest first — if anything's wrong with the path we catch it
    # before spending $23 on the iter13 run.
    Submission(
        name="v0_seed_gpt54_mini",
        agent_rel_path="agent.py",
        model_arg="openai/gpt-5.4-mini",
    ),
    Submission(
        name="v0_seed_sonnet_4_6",
        agent_rel_path="agent.py",
        model_arg="anthropic/claude-sonnet-4-6",
    ),
    Submission(
        name="v0_soft_cap_0_04",
        agent_rel_path="agents/iter13_style_aware_lean/agent.py",
        model_arg="none",
    ),
]


def verify_clean_tree() -> None:
    """Refuse to run if git status --porcelain returns anything.

    A dirty tree would cause Inspect to record eval_spec.revision.dirty=true
    in the .eval log, weakening the reproducibility claim for the leaderboard
    submission.
    """
    out = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if out:
        sys.stderr.write(
            "Working tree is dirty — refusing to run.\n"
            "The recorded eval_spec.revision.dirty would be true, weakening\n"
            "the leaderboard submission's reproducibility claim.\n"
            "Either commit or stash the following before re-running:\n\n"
            f"{out}\n"
        )
        sys.exit(1)


def run(cmd: list[str], *, cwd: Path, extra_env: dict | None = None) -> int:
    """Stream subprocess output live. Returns exit code."""
    env = {**os.environ, **(extra_env or {})}
    print(f"\n$ cd {cwd}")
    print(f"$ {' '.join(cmd)}\n")
    return subprocess.run(cmd, cwd=cwd, env=env).returncode


def stage(s: Submission) -> Path:
    """Copy agent.py + model_registry.py into a fresh working dir.

    Returns the working dir path.
    """
    src_agent = SOURCE_BASE / s.name / s.agent_rel_path
    src_registry = EXAMPLES_DIR / "model_registry.py"
    dst_dir = WORKING_BASE / s.name
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_agent, dst_dir / "agent.py")
    shutil.copy(src_registry, dst_dir / "model_registry.py")
    return dst_dir


def already_evaluated(working_dir: Path) -> bool:
    log_dir = working_dir / LOG_SUBDIR
    return log_dir.exists() and any(log_dir.glob("*.eval"))


def eval_submission(s: Submission, working_dir: Path) -> bool:
    if already_evaluated(working_dir):
        print(f"[skip eval] {s.name}: existing .eval found in {working_dir / LOG_SUBDIR}")
        return True
    log_dir = working_dir / LOG_SUBDIR
    log_dir.mkdir(parents=True, exist_ok=True)
    rc = run(
        [
            "astabench", "eval",
            "--solver", "agent.py",
            "--model", s.model_arg,
            "--split", "test",
            "--task", "DS_1000_test",
            "--log-dir", str(log_dir),
            "--display", "plain",
        ],
        cwd=working_dir,
        # LITELLM_LOCAL_MODEL_COST_MAP is required by `astabench score`;
        # passing during eval too is harmless and keeps env consistent.
        extra_env={"LITELLM_LOCAL_MODEL_COST_MAP": "True"},
    )
    return rc == 0


def score_submission(working_dir: Path) -> bool:
    rc = run(
        ["astabench", "score", str(working_dir / LOG_SUBDIR)],
        cwd=working_dir,
        extra_env={"LITELLM_LOCAL_MODEL_COST_MAP": "True"},
    )
    return rc == 0


def tar_submission(s: Submission, working_dir: Path) -> bool:
    out = WORKING_BASE / f"{s.name}.tar.gz"
    rc = run(
        # -C cd's into the working dir, then we tar the leaf dir name
        # so the archive contents are `full_test/...eval` rather than
        # absolute or deeply-nested paths.
        ["tar", "czfv", str(out), "-C", str(working_dir / "logs"), "full_test"],
        cwd=working_dir,
    )
    return rc == 0


def summarize() -> None:
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    for s in SUBMISSIONS:
        scores_path = WORKING_BASE / s.name / LOG_SUBDIR / "scores.json"
        if not scores_path.exists():
            print(f"  {s.name:25s}  (no scores.json)")
            continue
        try:
            data = json.loads(scores_path.read_text())
            result = data["results"][0]
            metrics = {m["name"]: m["value"] for m in result["metrics"]}
            acc = metrics.get("ds1000_scorer/accuracy", float("nan"))
            print(f"  {s.name:25s}  accuracy = {acc:.4f}")
        except Exception as e:
            print(f"  {s.name:25s}  (parse error: {e})")
    print("=" * 70)
    print("\nReady-to-upload tarballs:")
    for s in SUBMISSIONS:
        out = WORKING_BASE / f"{s.name}.tar.gz"
        marker = "✓" if out.exists() else "✗"
        print(f"  {marker} {out}")
    print("\nUpload at:")
    print("  https://huggingface.co/spaces/allenai/asta-bench-leaderboard")
    print("\nMetadata for the form (suggested):")
    print("  Openness:    Open source, closed weights")
    print("  Tools tier:  Standard")


def main() -> int:
    verify_clean_tree()
    failures = []
    for s in SUBMISSIONS:
        print(f"\n{'#' * 70}\n# {s.name}\n{'#' * 70}")
        working_dir = stage(s)
        if not eval_submission(s, working_dir):
            print(f"!! eval failed for {s.name}; continuing to next submission")
            failures.append((s.name, "eval"))
            continue
        if not score_submission(working_dir):
            print(f"!! score failed for {s.name}")
            failures.append((s.name, "score"))
            continue
        if not tar_submission(s, working_dir):
            print(f"!! tar failed for {s.name}")
            failures.append((s.name, "tar"))

    summarize()
    if failures:
        print(f"\nFailures: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
