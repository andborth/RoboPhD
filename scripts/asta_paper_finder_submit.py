#!/usr/bin/env python3
"""
PaperFindingBench → AstaBench leaderboard submission runner.

Run from anywhere — paths are resolved relative to the repo root:

    python scripts/asta_paper_finder_submit.py [--only NAME ...] [--limit N]

Per submission, the script:
  1. Verifies the working tree is clean (otherwise eval_spec.revision.dirty
     would record `true`, weakening the reproducibility claim).
  2. Verifies provider keys, ASTA_TOOL_KEY / HF_ACCESS_TOKEN, and that the
     installed litellm's BUNDLED price map covers every model the staged
     agent calls (the DS-1000 v0_0_3 lesson: an unpriced model makes
     `astabench score` emit cost=null and the entry ships costless).
  3. Stages a working dir at submissions/asta_paper_finder/<name>/
     (gitignored) with agent.py (resilience wrapper), agent_inner.py (the
     evolved agent from example_runs/), seed_agent.py (fallback tier), and
     model_registry.py.
  4. Runs `astabench eval` against paper_finder_test (no Docker — this
     task has no sandbox tier).
  5. Runs `astabench score` to produce scores.json + summary_stats.json,
     refusing to continue on cost=null.
  6. Tarballs logs/full_test/ for HuggingFace upload (full runs only —
     a --limit smoke run logs to logs/smoke_limit_<N>/ and is never
     tarred, so a partial run can't masquerade as a submission or trip
     the full run's idempotency skip).

The script does NOT submit. After a full run, one .tar.gz per selected
submission is ready for manual upload via the HF Spaces leaderboard form.

Cost / time (full test split, 267 queries, sequential):
    v0_0_7_soft_cap_0_06_fable   ~$200-270   12-18 hr
        (~12 min/semantic query at --max-samples 4; wall clock is unscored
        officially, so duration buys evidence quality, not points)
        agent  ≈ $15  (internal test measured $14.83)
        judge  ≈ $205-245 — official judging is FRESH and UNCAPPED:
        ~194 semantic queries × ~250 submitted papers ≈ 48.5K GPT-4o
        verdicts at a measured ~$0.0042/paper, billed to OPENAI_API_KEY
        during the eval. (Internal capped+cached judging cost $88.38.)
    A `--limit 3` smoke run costs ~$3 and validates the whole path first.

Prerequisites:
  - OPENAI_API_KEY               (agent models gpt-5.4-mini/gpt-5.4 AND the
                                  benchmark's GPT-4o relevance judge)
  - ANTHROPIC_API_KEY            (or ANTHROPIC_API_KEY_FOR_ROBOPHD;
                                  model_registry instantiates all handles
                                  at import, Anthropic validates eagerly)
  - GOOGLE_API_KEY               (registry import needs it present)
  - ASTA_TOOL_KEY                (Asta MCP corpus tools — the task's only
                                  retrieval surface)
  - HF_ACCESS_TOKEN              (test-split dataset download)
  - pip install litellm==1.88.1  (submission-scoring price map)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

# Repo root (this script lives at <repo>/scripts/).
REPO = Path(__file__).resolve().parent.parent

EXAMPLES_DIR = REPO / "examples" / "asta_paper_finder"
SOURCE_BASE = REPO / "example_runs" / "robophd" / "asta_paper_finder"
WORKING_BASE = REPO / "submissions" / "asta_paper_finder"
FULL_LOG_SUBDIR = "logs/full_test"
# The astabench CLI --task filter matches the config's task NAME
# (config/v1.0.0.yml: name PaperFindingBench_test, path
# astabench/paper_finder_test) — the path does not match.
TASK_NAME = "PaperFindingBench_test"

# Every model the staged agents call, as litellm bundled-map keys (litellm
# strips the provider prefix). Checked against the installed litellm before
# any eval spend — an unpriced model would surface only after the full run
# as `astabench score` cost=null.
AGENT_MODELS = ["gpt-5.4-mini", "gpt-5.4-2026-03-05"]

# Official-judging cost projection, printed before eval. Constants measured
# from the internal test eval of run asta_paper_finder_20260717_170858:
# $88.38 judge spend / 20,915 capped verdicts ≈ $0.0042/paper; iter12
# submits 250 papers on every semantic query; test split has 194 semantic.
JUDGE_COST_PER_PAPER_USD = 0.0042
SEMANTIC_TEST_QUERIES = 194
PAPERS_PER_SEMANTIC_QUERY = 250


class Submission(NamedTuple):
    # Dir name under both example_runs/robophd/asta_paper_finder/ and
    # submissions/asta_paper_finder/. Becomes the .tar.gz basename too.
    name: str
    # agent.py path relative to the example_runs source dir for this
    # submission. Snapshots are run-shaped (ds1000 precedent): the
    # winning agent nests under agents/<agent_name>/.
    agent_rel_path: str
    # --model arg for `astabench eval`. `none` for multi-model agents so
    # the recorded eval.model doesn't claim a single primary; per-call
    # usage is captured in stats.model_usage.
    model_arg: str


SUBMISSIONS = [
    Submission(
        name="v0_0_7_soft_cap_0_06_fable",
        agent_rel_path="agents/iter12_body_conjunction/agent.py",
        model_arg="none",
        # Run robophd-asta_paper_finder-003 (fable-5-evolved), winner
        # iter12_body_conjunction (Elo 1581, 8 test rounds). Internal test
        # 0.3724 mean F1 @ $0.0556/query (free zone $0.06). Models:
        # gpt-5.4-mini + gpt-5.4-2026-03-05 — both priced in the litellm
        # 1.88.1 bundled map (verified; see AGENT_MODELS preflight).
        # Patch 0_0_7 continues the cross-benchmark sequence after
        # DS-1000's v0_0_6.
    ),
    Submission(
        name="v0_0_8_soft_cap_0_033_opus",
        agent_rel_path="agents/iter9_rerank_rich_v1/agent.py",
        model_arg="none",
        # Run robophd-asta_paper_finder-006 (opus-4.8-evolved; luna
        # no-prose training judge), winner iter9_rerank_rich_v1 (Elo 1589,
        # 7 test rounds) — the platform's own Elo pick, submitted as such
        # even though iter14 finished 1.07 Elo behind on a higher train
        # mean (see the snapshot README). Internal test 0.2754 mean F1 @
        # $0.006/query on a full stock GPT-4o re-eval (free zone $0.033).
        # Sole model gpt-5.4-mini, already covered by AGENT_MODELS.
        # The cheap counterpart to v0_0_7, not a replacement: the two are
        # distinct Pareto points (0.3749 @ $0.0533 vs 0.2754 @ $0.006).
        # model_arg stays "none" despite the single model — the recorded
        # eval.model would otherwise claim a primary, and per-call usage
        # is already captured in stats.model_usage.
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


def resolve_anthropic_key() -> str:
    """Return ANTHROPIC_API_KEY value to inject into subprocesses.

    Convention: keep ANTHROPIC_API_KEY unset in the interactive shell so
    the user's Claude Code CLI subscription credentials aren't clobbered,
    and use ANTHROPIC_API_KEY_FOR_ROBOPHD for evaluation (model_registry
    reads either). Injected into subprocess env only.
    """
    return (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY_FOR_ROBOPHD")
        or ""
    )


def verify_credentials() -> None:
    """Hard-fail at startup if any required env var is missing.

    All three provider keys are required even though the agent only calls
    OpenAI models — model_registry.py creates handles for every family at
    import time, and the Anthropic provider validates its key eagerly.
    ASTA_TOOL_KEY is the task's only retrieval surface; HF_ACCESS_TOKEN
    gates the test-split download. Failing here beats failing minutes into
    a paid run.
    """
    missing = []
    if not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not resolve_anthropic_key():
        missing.append("ANTHROPIC_API_KEY (or ANTHROPIC_API_KEY_FOR_ROBOPHD)")
    if not os.environ.get("GOOGLE_API_KEY"):
        missing.append("GOOGLE_API_KEY")
    if not os.environ.get("ASTA_TOOL_KEY"):
        missing.append("ASTA_TOOL_KEY")
    if not os.environ.get("HF_ACCESS_TOKEN"):
        missing.append("HF_ACCESS_TOKEN")
    if missing:
        sys.stderr.write(
            f"Missing required env vars: {', '.join(missing)}\n"
            "See examples/asta_paper_finder/README.md for setup instructions.\n"
        )
        sys.exit(1)


def verify_model_pricing() -> None:
    """Assert the installed litellm's bundled map prices every AGENT_MODELS
    entry — before any eval spend.

    `astabench score` runs with LITELLM_LOCAL_MODEL_COST_MAP=True (it
    refuses otherwise), which forces the BUNDLED map: any model the agent
    called that the map lacks prices the whole run to cost=null, and the
    leaderboard entry would ship without a cost figure (DS-1000 v0_0_3).
    """
    try:
        import litellm
    except ImportError:
        sys.stderr.write(
            "litellm is not importable — `pip install litellm==1.88.1` "
            "(the submission-scoring price map) and re-run.\n"
        )
        sys.exit(1)
    unpriced = [
        m for m in AGENT_MODELS
        if not (litellm.model_cost.get(m) or {}).get("input_cost_per_token")
    ]
    if unpriced:
        sys.stderr.write(
            f"litellm's bundled price map does not price: {unpriced}\n"
            "`astabench score` would emit cost=null for the whole run.\n"
            "Upgrade litellm (1.88.1 is known-good for these models) and "
            "re-run.\n"
        )
        sys.exit(1)
    print(f"[pricing] litellm bundled map prices all agent models: {AGENT_MODELS}")


def print_cost_projection(limit: int | None) -> None:
    judge = SEMANTIC_TEST_QUERIES * PAPERS_PER_SEMANTIC_QUERY * JUDGE_COST_PER_PAPER_USD
    print(
        f"\n[cost] Official judging is fresh + uncapped: "
        f"~{SEMANTIC_TEST_QUERIES} semantic queries × "
        f"~{PAPERS_PER_SEMANTIC_QUERY} papers × ${JUDGE_COST_PER_PAPER_USD}/paper "
        f"≈ ${judge:.0f} judge + ~$15 agent ≈ ${judge + 15:.0f} total (full run)."
    )
    if limit is not None:
        print(f"[cost] --limit {limit} smoke run: a few dollars; logs are "
              f"kept separate and never tarred.")


def run(cmd: list[str], *, cwd: Path, extra_env: dict | None = None) -> int:
    """Stream subprocess output live. Returns exit code."""
    env = {**os.environ, **(extra_env or {})}
    print(f"\n$ cd {cwd}")
    print(f"$ {' '.join(cmd)}\n")
    return subprocess.run(cmd, cwd=cwd, env=env).returncode


# Auto-generated resilience wrapper inserted as `agent.py` in the staged
# working dir. The original agent code is renamed to `agent_inner.py` and
# imported from here.
#
# Why this exists: RoboPhD's evolution evaluator runs each sample in a
# subprocess, so a per-sample solver crash returns raw_score=0 and the run
# continues. AstaBench's CLI runs all samples in one process and aborts on
# any uncaught solver exception. The wrapper bridges those two contracts so
# the AstaBench-CLI score reflects the same crash-tolerance the recorded
# RoboPhD-internal score was produced under.
#
# Caught: any `Exception` subclass (which on Python 3.11+ includes
# asyncio.TimeoutError). NOT caught: KeyboardInterrupt, SystemExit,
# asyncio.CancelledError — BaseException-only signals stay unhandled so
# user/runtime cancellation still works correctly.
WRAPPER_TEMPLATE = '''"""Auto-generated resilience wrapper for AstaBench submission.

Two-tier fallback: primary agent (agent_inner.py) -> seed agent
(seed_agent.py) -> empty-but-valid submission. Each layer catches
Exception so a transient provider failure or evolved-agent bug on one
query doesn't abort the eval, and the seed gets a shot at recovering the
score before we give up.

The last-resort output is NOT an empty string: PaperFindingBench's scorer
falls back to an LLM re-parse on unparseable completions, so we emit the
schema-valid empty submission ({"output": {"query_id": ..., "results": []}}),
which parses cleanly and scores 0 with no extra machinery.

Per-sample wall-clock timeout: both tiers are wrapped in
`asyncio.wait_for(..., timeout=1500)` so a hung primary (rate-limit
retry storm, provider connection deadlock, snippet_search stall) can't
wedge the eval indefinitely. Internal training capped queries at 1800s;
1500s leaves room for the seed tier within the same ballpark.
asyncio.TimeoutError is a subclass of Exception on Python 3.11+, so the
`except Exception` blocks catch the timeout and fall through naturally.

Wrapper recipe lives in scripts/asta_paper_finder_submit.py:WRAPPER_TEMPLATE.
"""
import asyncio
import json
import traceback

from inspect_ai.solver import Generate, TaskState, solver

from agent_inner import make_solver as _inner_make_solver
from seed_agent import make_solver as _seed_make_solver


PRIMARY_TIMEOUT_S = 3000  # 50 min — a hang bound, not a work budget.
# It has to clear whatever pacing the staged agent has, and agents differ:
# v0_0_7's iter12 self-paced (SOFT_DEADLINE=1300 / TAIL_DEADLINE=1550,
# evolved against training's 1800s external cap), while v0_0_8's iter9 has
# no deadline constants at all — its runtime comes from a fixed work plan,
# and with no per-call timeouts a hung tool call runs until this ceiling
# fires. For a self-pacing agent a tight ceiling guillotines it INSIDE its
# planned budget: at 1500s the first official attempt seed-fell-back on 5
# of 9 samples (0.57->0.06 on semantic_5); at 2100 the second watched
# semantic_25 finish 48s short as MCP latency stretched solves to ~30 min.
# For a non-pacing agent the ceiling should only ever fire on a true hang.
# Either way firing means a seed-tier fallback for that sample. Wall clock
# is unscored officially, so generosity costs nothing — size this off the
# slowest agent in SUBMISSIONS, never the newest.
SEED_TIMEOUT_S = 1500     # fallback tier, applied independently


def _empty_submission(state: TaskState) -> str:
    return json.dumps({"output": {"query_id": str(state.sample_id), "results": []}})


@solver
def make_solver():
    inner = _inner_make_solver()
    seed = _seed_make_solver()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if state.metadata is None:
            state.metadata = {}
        try:
            return await asyncio.wait_for(inner(state, generate), timeout=PRIMARY_TIMEOUT_S)
        except Exception as primary:
            print(f"[{state.sample_id}] WRAPPER primary caught {type(primary).__name__}: {primary}")
            print(f"[{state.sample_id}] WRAPPER primary traceback (truncated):")
            print(traceback.format_exc()[:1500])
            state.metadata["__wrapper_primary_caught"] = repr(primary)[:500]
            state.metadata["__wrapper_primary_traceback"] = traceback.format_exc()[:2000]
            try:
                return await asyncio.wait_for(seed(state, generate), timeout=SEED_TIMEOUT_S)
            except Exception as fallback:
                print(f"[{state.sample_id}] WRAPPER seed fallback ALSO caught {type(fallback).__name__}: {fallback}")
                print(f"[{state.sample_id}] WRAPPER seed fallback traceback (truncated):")
                print(traceback.format_exc()[:1500])
                state.output.completion = _empty_submission(state)
                state.metadata["__wrapper_fallback_caught"] = repr(fallback)[:500]
                state.metadata["__wrapper_fallback_traceback"] = traceback.format_exc()[:2000]
                return state

    return solve
'''


def stage(s: Submission) -> Path:
    """Stage a working dir with the two-tier resilience wrapper.

    Layout in dst_dir:
      agent.py          — auto-generated wrapper (the file --solver references)
      agent_inner.py    — the primary evolved agent source (renamed)
      seed_agent.py     — the baseline seed agent (the fallback tier)
      model_registry.py — copied from examples/asta_paper_finder/

    Returns the working dir path.
    """
    src_agent = SOURCE_BASE / s.name / s.agent_rel_path
    src_seed = EXAMPLES_DIR / "seeds" / "baseline" / "agent.py"
    src_registry = EXAMPLES_DIR / "model_registry.py"
    dst_dir = WORKING_BASE / s.name
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_agent, dst_dir / "agent_inner.py")
    shutil.copy(src_seed, dst_dir / "seed_agent.py")
    (dst_dir / "agent.py").write_text(WRAPPER_TEMPLATE)
    shutil.copy(src_registry, dst_dir / "model_registry.py")
    return dst_dir


def log_subdir(limit: int | None) -> str:
    """Full runs log to logs/full_test (the tarred, idempotency-checked
    dir). --limit smoke runs log to their own dir so a partial run can
    neither be tarred as a submission nor trip the full run's skip."""
    return FULL_LOG_SUBDIR if limit is None else f"logs/smoke_limit_{limit}"


def _log_status(log_dir: Path) -> tuple[str | None, int]:
    """(status, completed_samples) of the newest .eval in log_dir, read
    header-only. (None, 0) when no log exists.

    Status matters: an interrupted run leaves a log with status
    "cancelled"/"error" that contains only the samples that finished.
    Treating any *.eval as done would skip the eval on re-run and then
    score+tar a PARTIAL log as a submission. Instead, only "success"
    skips; anything else re-runs `astabench eval` with the same log dir,
    which flows into `inspect eval-set`'s retry path — completed samples
    are reused (dataset unshuffled + stable query ids), so a resume
    re-runs only what didn't finish.
    """
    logs = sorted(log_dir.glob("*.eval")) if log_dir.exists() else []
    if not logs:
        return None, 0
    from inspect_ai.log import read_eval_log

    log = read_eval_log(str(logs[-1]), header_only=True)
    completed = getattr(getattr(log, "results", None), "completed_samples", 0) or 0
    return log.status, completed


def eval_submission(s: Submission, working_dir: Path, limit: int | None) -> bool:
    status, completed = _log_status(working_dir / log_subdir(limit))
    if status == "success":
        print(f"[skip eval] {s.name}: successful .eval "
              f"({completed} samples) in {working_dir / log_subdir(limit)}")
        return True
    if status is not None:
        print(f"[resume] {s.name}: previous eval status={status!r} with "
              f"{completed} completed sample(s) — re-running; inspect "
              f"eval-set reuses completed samples and runs the rest")
    log_dir = working_dir / log_subdir(limit)
    log_dir.mkdir(parents=True, exist_ok=True)
    # LITELLM_LOCAL_MODEL_COST_MAP is required by `astabench score`;
    # passing during eval too keeps env consistent. ANTHROPIC_API_KEY is
    # injected here (not in the parent process) — see resolve_anthropic_key().
    extra_env = {
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "ANTHROPIC_API_KEY": resolve_anthropic_key(),
    }
    cmd = [
        "astabench", "eval",
        "--solver", "agent.py",
        "--model", s.model_arg,
        "--split", "test",
        "--task", TASK_NAME,
        "--log-dir", str(log_dir),
        "--display", "plain",
        # Concurrency: no Docker tier for this task, so no --max-sandboxes.
        # --max-samples 6 keeps aggregate Asta MCP tool traffic under the
        # 10 req/s per-endpoint server-side rate limit (the agent fans out
        # snippet searches within a query; it evolved and was internally
        # tested at 8-wide against this endpoint, so 6 is inside its
        # native habitat). 40 connections: agents' grading
        # calls and the scorer's per-paper GPT-4o judge fan-out share ONE
        # pool — at 20, agents queued behind judge bursts and samples
        # stretched from ~12 min (smoke, no scoring overlap) to ~25 min,
        # into the wrapper ceiling.
        "--max-samples", "6",
        "--max-connections", "40",
    ]
    if limit is not None:
        # Forwarded to `inspect eval-set` via agenteval's
        # ignore_unknown_options Click context.
        cmd += ["--limit", str(limit)]
    rc = run(cmd, cwd=working_dir, extra_env=extra_env)
    if rc != 0:
        return False
    # Belt-and-suspenders: a zero exit with a non-success log (cancelled
    # mid-write, partial eval-set) must not flow into score+tar.
    status, completed = _log_status(log_dir)
    if status != "success":
        print(f"!! eval exited 0 but log status={status!r} "
              f"({completed} samples) — not treating as success")
        return False
    return True


def score_submission(working_dir: Path, limit: int | None) -> bool:
    # LITELLM_LOCAL_MODEL_COST_MAP=True is REQUIRED by `astabench score`
    # (it refuses to run without it). It forces litellm's *bundled* price
    # map — verify_model_pricing() pre-checked coverage, and the null-cost
    # gate below catches anything that still slips through.
    rc = run(
        ["astabench", "score", str(working_dir / log_subdir(limit))],
        cwd=working_dir,
        extra_env={"LITELLM_LOCAL_MODEL_COST_MAP": "True"},
    )
    if rc != 0:
        return False
    stats_path = working_dir / log_subdir(limit) / "summary_stats.json"
    try:
        stats = json.loads(stats_path.read_text())["stats"]
        # One task per run; resolve its key instead of hardcoding, so an
        # astabench task-name change fails loudly here rather than KeyError.
        # Match both naming styles ("PaperFindingBench_test" config name,
        # "paper_finder_test" task path).
        task_keys = [
            k for k in stats
            if "paperfinding" in k.lower().replace("_", "")
            or "paperfinder" in k.lower().replace("_", "")
        ] or list(stats)
        cost = stats[task_keys[0]]["cost"]
    except Exception as e:
        print(f"!! could not read cost from {stats_path}: {e}")
        return False
    if cost is None:
        print(
            "!! astabench score produced cost=null — a model in the eval "
            "log isn't in the installed litellm's bundled price map. "
            "Upgrade litellm (bundled map must cover every model the "
            "agent called) and re-run. Refusing to tar a costless "
            "submission."
        )
        return False
    print(f"scored: cost/problem = ${cost:.4f}")
    return True


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


def summarize(limit: int | None) -> None:
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    for s in SUBMISSIONS:
        scores_path = WORKING_BASE / s.name / log_subdir(limit) / "scores.json"
        if not scores_path.exists():
            print(f"  {s.name:30s}  (no scores.json)")
            continue
        try:
            data = json.loads(scores_path.read_text())
            result = data["results"][0]
            metrics = {m["name"]: m["value"] for m in result["metrics"]}
            # PaperFindingBench's headline metric is adjusted_f1_micro_avg
            # (grouped mean over all samples); fall back to printing
            # everything if the name shifts upstream.
            headline = next(
                (v for k, v in metrics.items() if "adjusted_f1" in k or k.endswith("_micro_avg")),
                None,
            )
            if headline is not None:
                print(f"  {s.name:30s}  adjusted_f1 = {headline:.4f}")
            else:
                print(f"  {s.name:30s}  metrics = {metrics}")
        except Exception as e:
            print(f"  {s.name:30s}  (parse error: {e})")
    print("=" * 70)
    if limit is not None:
        print(f"\n--limit {limit} smoke run: logs kept in "
              f"{log_subdir(limit)}/ — NOT tarred, not a submission.")
        return
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
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--only", nargs="+", metavar="NAME",
                        help="restrict to a subset of SUBMISSIONS")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="smoke run: evaluate only the first N samples; "
                             "logs to a separate dir, never tarred")
    args = parser.parse_args()

    selected = SUBMISSIONS
    if args.only:
        names = set(args.only)
        unknown = names - {s.name for s in SUBMISSIONS}
        if unknown:
            print(f"unknown submission(s): {sorted(unknown)}")
            print(f"known: {[s.name for s in SUBMISSIONS]}")
            return 2
        selected = [s for s in SUBMISSIONS if s.name in names]

    verify_clean_tree()
    verify_credentials()
    verify_model_pricing()
    print_cost_projection(args.limit)

    failures = []
    for s in selected:
        print(f"\n{'#' * 70}\n# {s.name}\n{'#' * 70}")
        working_dir = stage(s)
        if not eval_submission(s, working_dir, args.limit):
            print(f"!! eval failed for {s.name}; continuing to next submission")
            failures.append((s.name, "eval"))
            continue
        if not score_submission(working_dir, args.limit):
            print(f"!! score failed for {s.name}")
            failures.append((s.name, "score"))
            continue
        if args.limit is None and not tar_submission(s, working_dir):
            print(f"!! tar failed for {s.name}")
            failures.append((s.name, "tar"))

    summarize(args.limit)
    if failures:
        print(f"\nFailures: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
