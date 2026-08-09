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
    v0_0_1_seed_gpt54_mini   ~$0.50    30-60 min   (gpt-5.4-mini one-shot)
    v0_0_1_soft_cap_0_16     ~$20-30   2-4 hr      (quad-diverse ensemble + Opus critic)
    v0_0_2_soft_cap_0_08     ~$12-15   1-2 hr      (Sonnet primary + rare Opus fallback)
    v0_0_3_soft_cap_0_06     ~$40-50   2-4 hr      (dual-gen + GPT-5.5 adjudication;
                                                    internal final eval cost $41.58)
    --
    total                   ~$73-95   5.5-11 hr   (use --only to run a subset)

Prerequisites:
  - OPENAI_API_KEY              (seed runs gpt-5.4-mini; v0_0_1_soft_cap_0_16
                                 calls GPT-5.4 in its four-candidate ensemble)
  - ANTHROPIC_API_KEY           (or ANTHROPIC_API_KEY_FOR_ROBOPHD;
                                 model_registry reads either; v0_0_1_soft_cap_0_16
                                 uses Sonnet+Opus and v0_0_2_soft_cap_0_08
                                 uses Sonnet primary + Opus fallback)
  - GOOGLE_API_KEY              (v0_0_1_soft_cap_0_16's Gemini candidate;
                                 mostly times out but the registry import
                                 needs the env var present)
  - Docker daemon running       (DS-1000 sandbox tier requires it)
  - LITELLM_LOCAL_MODEL_COST_MAP=True is set inside subprocesses

The script does NOT submit. After it finishes, one .tar.gz per selected
submission is ready for manual upload via the HF Spaces leaderboard form.
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
    # Historical: drove ASTA_DS1000_ALLOW_STRONGER_MODELS=1 when
    # model_registry gated the strong handles (CLAUDE_OPUS_4_7, GPT_5_5,
    # GEMINI_3_1_PRO_PREVIEW) behind it. The gate was removed in 7494ff7
    # — all handles import unconditionally — so the env var is a no-op
    # against the registry at HEAD. Kept (and still set for the v0_0_1/
    # v0_0_2 entries) so re-staging those submissions reproduces their
    # original eval-time env. Note that env reproduction is no longer
    # sufficient for those two entries: since 2065ce4 the live registry
    # no longer exports CLAUDE_OPUS_4_7, so they fail at import when
    # staged at HEAD (see the per-entry notes below).
    needs_stronger_models: bool = False


SUBMISSIONS = [
    # Cheapest first — if anything's wrong with the path (wrapper
    # template, stage(), extra_env, etc.) we catch it on the ~$0.50
    # seed run before spending $20-30 on the iter10 ensemble.
    Submission(
        name="v0_0_1_seed_gpt54_mini",
        agent_rel_path="agent.py",
        model_arg="openai/gpt-5.4-mini",
    ),
    # NOT runnable at HEAD: the v0_0_1/v0_0_2 archived agents import
    # CLAUDE_OPUS_4_7, dropped from the live registry in 2065ce4 — a
    # deliberate substitution-avoidance choice (aliasing it to 4.8 would
    # silently re-rank old agents on a model they weren't evaluated on),
    # so the archived agents stay as-is. To re-stage either entry, check
    # out the revision recorded in its snapshot's eval log (any
    # pre-2065ce4 commit works).
    Submission(
        name="v0_0_1_soft_cap_0_16",
        agent_rel_path="agents/iter10_idiomatic_loop_guard_v1/agent.py",
        model_arg="none",
        needs_stronger_models=True,
    ),
    Submission(
        name="v0_0_2_soft_cap_0_08",
        agent_rel_path="agents/iter4_ds1000_idiom_probe/agent.py",
        model_arg="none",
        needs_stronger_models=True,
    ),
    Submission(
        name="v0_0_3_soft_cap_0_06",
        agent_rel_path="agents/iter14_filemock_adjudicate/agent.py",
        model_arg="none",
        # No needs_stronger_models: the gate was removed from
        # model_registry.py (7494ff7) — all handles, including the
        # GPT-5.5 adjudicator this agent leans on, import unconditionally.
    ),
    Submission(
        name="v0_0_4_soft_cap_0_08",
        agent_rel_path="agents/iter11_ds1000_tridtype_judge/agent.py",
        model_arg="none",
        # Run robophd-asta_ds1000-028 (Opus-4.8-evolved). Three-family
        # ensemble: gpt-5.4-2026-03-05 / claude-sonnet-4-6 /
        # gemini-3.1-pro-preview — all three are in the leaderboard price
        # map, so cost computes (unlike v0_0_3's gemini-3.5-flash). The
        # needs_stronger_models field is inert at HEAD (gate removed).
    ),
    Submission(
        name="v0_0_5_soft_cap_0_05",
        agent_rel_path="agents/iter10_literal_consensus/agent.py",
        model_arg="none",
        # Run robophd-asta_ds1000-029 (Opus-4.8-evolved). Lean 2-model
        # cross-agreement: gpt-5.4-2026-03-05 + claude-sonnet-4-6 (no
        # Gemini) -- both priced on the board, so cost computes. Cheapest
        # RoboPhD agent at the ~0.84 tier ($0.0173/problem; the sonnet-only
        # v0_0_2 is cheaper at $0.01 but only 0.809); internal 0.8378 would
        # Pareto-dominate the Button leaderboard entry (0.836 @ $0.03).
    ),
    Submission(
        name="v0_0_5_soft_cap_0_05_deep_focus",
        agent_rel_path="agents/iter8_perspective_consensus_ds1000/agent.py",
        model_arg="none",
        # Run robophd-asta_ds1000-032 (Opus-4.8-evolved). Deep-Focus variant
        # of v0_0_5: same config except new_agent_test_rounds=1 (vs 0). It
        # converged on a different, cheaper agent -- a gpt-5.4-mini
        # 3-perspective ensemble + gpt-5.4 no-consensus escalation (both
        # priced on the board, so cost computes). Internal 0.7678 @ $0.0069;
        # projected new cheap frontier point dominating ReAct o3 (0.749 @
        # $0.01) and Asta v0 (0.748 @ $0.01), pending the official number.
    ),
    Submission(
        name="v0_0_6_soft_cap_0_003_fable",
        agent_rel_path="agents/iter8_expected_diff_cascade/agent.py",
        model_arg="none",
        # Run robophd-asta_ds1000-036 (FIRST fable-5-evolved submission).
        # Ultra-low cap $0.003/$0.001. Budget-tier cascade: gpt-5.4-mini +
        # gemini-3.1-flash-lite generators, claude-haiku-4-5 escalation,
        # gpt-5.4 only as a rare 450-token arbiter -- all four priced in the
        # litellm 1.88.1 bundled map (verified), so cost computes. Train mean
        # 86.8 over 11 Elo rounds (~220 problems); internal 900-test
        # 0.7322 @ $0.00294 (98% of cap); targets ReAct/GPT-5-Mini
        # (0.710 @ $0.003 true cost) on the official frontier.
    ),
    Submission(
        name="v0_0_7_sharp_cap_0_003",
        agent_rel_path="agents/iter12_strong_toplevel/agent.py",
        model_arg="none",
        # Run robophd-asta_ds1000-044 (opus-4.8-evolved). Sharp-penalty
        # follow-up to v0_0_6: cost_threshold=0.003, cost_per_error=0.0003
        # (3.33x sharper -- v0_0_6's optimizer rationally bought through
        # the milder penalty). Single gpt-5.4 one-shot + deterministic
        # extraction fixes, gpt-5.4-mini fallback -- both OpenAI, cleanly
        # priced in the bundled map. Internal 900-test 0.7333 @ $0.00168
        # measured entirely on the post-fix billing basis (first such
        # submission: official cost should replicate, unlike v0_0_6's
        # +46%); internally Pareto-dominates ReAct/GPT-5-Mini
        # (0.710 @ $0.003 true cost).
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

    Inspect-AI's CLI resolves `--model anthropic/...` at startup by
    instantiating an Anthropic client, which reads ANTHROPIC_API_KEY
    directly from env. Our convention is to keep ANTHROPIC_API_KEY
    unset so the user's Claude Code CLI subscription credentials don't
    get clobbered, and use ANTHROPIC_API_KEY_FOR_ROBOPHD for our
    evaluation needs (which model_registry.py reads). This shim bridges
    the two: prefer ANTHROPIC_API_KEY if already set, else fall back
    to ANTHROPIC_API_KEY_FOR_ROBOPHD. The returned value is injected
    into the subprocess env only — the parent process's env is left
    alone, so the user's Claude Code CLI keeps using its own creds.
    """
    return (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY_FOR_ROBOPHD")
        or ""
    )


def verify_credentials() -> None:
    """Hard-fail at startup if any required provider key is missing.

    All three keys are required even though some submissions only use
    one model — model_registry.py creates handles for all six models at
    import time, and the Anthropic provider validates its key eagerly.
    Failing here beats failing 30 minutes into a $23 iter13 run.
    """
    missing = []
    if not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not resolve_anthropic_key():
        missing.append("ANTHROPIC_API_KEY (or ANTHROPIC_API_KEY_FOR_ROBOPHD)")
    if not os.environ.get("GOOGLE_API_KEY"):
        missing.append("GOOGLE_API_KEY")
    if missing:
        sys.stderr.write(
            f"Missing required env vars: {', '.join(missing)}\n"
            "All three provider keys must be set — model_registry.py creates\n"
            "all six model handles at import time, and the Anthropic provider\n"
            "validates its key eagerly. See examples/asta_ds1000/README.md\n"
            "for setup instructions.\n"
        )
        sys.exit(1)


def run(cmd: list[str], *, cwd: Path, extra_env: dict | None = None) -> int:
    """Stream subprocess output live. Returns exit code."""
    env = {**os.environ, **(extra_env or {})}
    print(f"\n$ cd {cwd}")
    print(f"$ {' '.join(cmd)}\n")
    return subprocess.run(cmd, cwd=cwd, env=env).returncode


# Auto-generated resilience wrapper inserted as `agent.py` in the staged
# working dir. The original agent code is renamed to `agent_inner.py` and
# imported from here. Wraps the inner make_solver in a try/except that
# converts uncaught solver exceptions to empty `state.output.completion`,
# which the DS-1000 scorer marks as 'I' (incorrect → score 0).
#
# Why this exists: RoboPhD's evolution evaluator runs each sample in a
# subprocess, so a per-sample solver crash returns raw_score=0 and the run
# continues. AstaBench's CLI runs all samples in one process and aborts on
# any uncaught solver exception. The wrapper bridges those two contracts so
# the AstaBench-CLI score reflects the same crash-tolerance the recorded
# RoboPhD-internal score was produced under.
#
# Caught: any `Exception` subclass (which on Python 3.11+ includes
# asyncio.TimeoutError, since it's an alias for builtins.TimeoutError).
# NOT caught: KeyboardInterrupt, SystemExit, asyncio.CancelledError —
# these BaseException-only signals stay unhandled so user/runtime
# cancellation still works correctly.
WRAPPER_TEMPLATE = '''"""Auto-generated resilience wrapper for AstaBench submission.

Two-tier fallback: primary agent (agent_inner.py) -> seed agent
(seed_agent.py) -> empty completion. Each layer catches Exception so a
transient provider failure or evolved-agent bug on one example doesn't
abort the eval, and the seed gets a shot at recovering the score before
we give up and emit 'I'.

This matches the per-sample crash semantics of RoboPhD's internal
subprocess-isolated evaluator (with `fallback_candidate=seed`) under
which the recorded RoboPhD-internal test_results_final.json scores
were produced. Inspect-AI's `--score-on-error` flag (added in 0.3.220)
would be a clean framework-level equivalent, but AstaBench pins to
0.3.203, so we apply the same semantic at the agent level.

Stopping-behavior asymmetry vs the internal evaluator's fallback: when
the seed ALSO errors, this wrapper emits empty completion (scorer marks
'I' -> 0) because Inspect requires a TaskState back to score. The
internal evaluator (Ds1000Evaluator.__call__) returns the seed's error
diagnostics as-is, since its upstream layer (ExternalEvaluatorDomain)
handles errors natively. Same intent (no third retry, no recursion);
different output shape forced by the calling convention.

Per-sample wall-clock timeout: both tiers are wrapped in
`asyncio.wait_for(..., timeout=1200)` so a hung primary (rate-limit
retry storm, provider connection deadlock, etc.) can't wedge the eval
indefinitely. asyncio.TimeoutError is a subclass of Exception on
Python 3.11+, so the existing `except Exception` blocks catch the
timeout and fall to the next tier naturally.

Per-sample module isolation: agent_inner and seed_agent are loaded as FRESH
module instances per sample (see _isolated below), so anything an evolved
agent keeps at module scope is per-sample here exactly as it is under
RoboPhD's one-subprocess-per-sample evaluator. Without this, --max-samples
concurrent samples share one module namespace and per-sample state silently
becomes global. model_registry deliberately stays shared.

Wrapper recipe lives in scripts/asta_ds1000_submit.py:WRAPPER_TEMPLATE.
"""
import asyncio
import importlib.util
import itertools
import sys
import traceback
import uuid
from pathlib import Path

from inspect_ai.solver import Generate, TaskState, solver

# LOAD-BEARING — do not delete as an "unused import". This runs while
# inspect's chdir_python() still has the submission dir on sys.path, which
# pins model_registry into sys.modules. agent_inner does `from model_registry
# import ...` at ITS module scope, and the per-sample loads below happen long
# after chdir_python.__exit__ restored sys.path (inspect_ai/_util/path.py:60).
# Without this, every sample dies with ModuleNotFoundError.
import model_registry  # noqa: F401


_HERE = Path(__file__).resolve().parent
_TAG = uuid.uuid4().hex[:8]   # load_module() re-execs this file on every
_SEQ = itertools.count()      # solver_from_spec, so a bare counter collides
_WARNED = set()


def _warn_once(key: str, detail: str) -> None:
    """One line per distinct problem, not one per sample.

    These conditions are not fatal to a sample but they ARE load-bearing for
    the run, so swallowing them is wrong: the eval would finish and score
    plausibly with no trace of what changed.
    """
    if key not in _WARNED:
        _WARNED.add(key)
        print(f"[WRAPPER] WARNING: {detail}", flush=True)


# Private inspect API, resolved once at load rather than per sample. If it
# moves, the wrapper still runs, but agent_inner's @solver keeps the bare
# "make_solver" registry entry it clobbers on every load — and eval-set's
# RESUME path (evalset.py resolve_solver, inside task_identifier) looks the
# solver up by that name, so a resumed run would execute agent_inner with NO
# wrapper: no wait_for ceiling, no seed tier, and the shared module state this
# whole wrapper exists to prevent. Loud here, once.
try:
    from inspect_ai._util.registry import registry_add as _registry_add
    from inspect_ai._util.registry import registry_info as _registry_info
except Exception as _reg_err:  # pragma: no cover - private API moved
    _registry_add = _registry_info = None
    print(f"[WRAPPER] WARNING: inspect_ai._util.registry unavailable "
          f"({type(_reg_err).__name__}: {_reg_err}); the solver registry entry "
          f"cannot be restored after per-sample loads. Do NOT resume this "
          f"eval-set — a resume may run agent_inner with no wrapper.",
          flush=True)


def _isolated(stem: str):
    """Load <stem>.py as a FRESH module: its module scope becomes per-sample.

    `astabench eval` runs --max-samples samples as asyncio tasks in ONE
    process; RoboPhD training and internal test run one sample per subprocess.
    Agents are therefore evolved where module scope means "per sample" and
    graded where it means "shared across concurrent samples". No DS-1000 agent
    shipped to date keeps mutable state there, but nothing prevents the next
    one from doing so, and the failure is silent — see the paper_finder
    submission where a module-scope deadline clock cost 30% of samples.

    Isolation boundary: modules loaded from THIS directory under a synthetic
    name. model_registry stays SHARED by design (canonical name in
    sys.modules — one connection pool). An agent that mutates a shared
    module's globals is not isolated by this.
    """
    # chdir_python() had this dir on sys.path only while agent.py itself was
    # loading, then __exit__ restored the snapshot. Third-line defence only:
    # the load below uses an absolute path, and model_registry is already
    # pinned in sys.modules by the module-scope import above. It is here
    # because the failure it guards (ModuleNotFoundError on every sample)
    # zeroes an entire run.
    # The entry persists for the process lifetime. That is safe because
    # `astabench eval` runs exactly ONE submission per process — the submit
    # script spawns a subprocess per submission — so no second submission dir
    # can ever be shadowed by it.
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    name = f"{stem}__iso{_TAG}_{next(_SEQ)}"
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{stem}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build a module spec for {stem}.py")
    mod = importlib.util.module_from_spec(spec)
    # Register BEFORE exec: the @solver decorator calls find_spec() on this
    # name, which is a full sys.path scan whenever the name is absent.
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        # Drop the name so hundreds of module dicts don't accumulate. Never
        # mod.__dict__.clear() — a cancelled asyncio task may still be
        # unwinding through these globals.
        sys.modules.pop(name, None)
    # agent_inner's own @solver registered under the same bare name
    # ("make_solver") and clobbered ours. inspect's eval-set resume path
    # resolves the solver by that name, so put it back. Absent during the
    # warm-up below, which runs before the wrapper is defined.
    _wrapper = globals().get("make_solver")
    if _wrapper is not None and _registry_add is not None:
        try:
            _registry_add(_wrapper, _registry_info(_wrapper))
        except Exception as e:
            _warn_once(
                "registry_restore",
                f"could not restore the solver registry entry after loading "
                f"{stem} ({type(e).__name__}: {e}). Samples still run, but do "
                f"NOT resume this eval-set — a resume may run agent_inner "
                f"with no wrapper.")
    return mod


# Warm-up load, at module scope so it runs INSIDE inspect's chdir_python
# window. Compiles the pyc once so no sample pays compile cost, and surfaces a
# broken agent_inner here rather than on sample 1. Must stay ABOVE the @solver
# below so the wrapper is the last thing registered.
_isolated("agent_inner")


PRIMARY_TIMEOUT_S = 1200  # 20 min — bounds hung primaries
SEED_TIMEOUT_S = 1200     # same — applied independently to the fallback tier


@solver
def make_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if state.metadata is None:
            state.metadata = {}
        try:
            # Fresh module per sample. INSIDE the try on purpose: a load
            # failure (transient OSError, MemoryError, a half-written source
            # file) must degrade to the seed tier, not abort the whole eval.
            inner = _isolated("agent_inner").make_solver()
            return await asyncio.wait_for(inner(state, generate), timeout=PRIMARY_TIMEOUT_S)
        except Exception as primary:
            print(f"[{state.sample_id}] WRAPPER primary caught {type(primary).__name__}: {primary}")
            print(f"[{state.sample_id}] WRAPPER primary traceback (truncated):")
            print(traceback.format_exc()[:1500])
            state.metadata["__wrapper_primary_caught"] = repr(primary)[:500]
            state.metadata["__wrapper_primary_traceback"] = traceback.format_exc()[:2000]
            try:
                # Lazy: the seed is loaded only on the failure path, so the
                # happy path pays zero seed loads.
                seed = _isolated("seed_agent").make_solver()
                return await asyncio.wait_for(seed(state, generate), timeout=SEED_TIMEOUT_S)
            except Exception as fallback:
                print(f"[{state.sample_id}] WRAPPER seed fallback ALSO caught {type(fallback).__name__}: {fallback}")
                print(f"[{state.sample_id}] WRAPPER seed fallback traceback (truncated):")
                print(traceback.format_exc()[:1500])
                state.output.completion = ""
                state.metadata["__wrapper_fallback_caught"] = repr(fallback)[:500]
                state.metadata["__wrapper_fallback_traceback"] = traceback.format_exc()[:2000]
                return state

    return solve
'''


def stage(s: Submission, restage: bool = False) -> Path:
    """Stage a working dir with the two-tier resilience wrapper.

    Layout in dst_dir:
      agent.py         — auto-generated wrapper (the file --solver references)
      agent_inner.py   — the primary evolved/seed agent source (renamed)
      seed_agent.py    — the baseline seed agent (the fallback tier)
      model_registry.py — copied from examples/asta_ds1000/

    The wrapper imports make_solver from agent_inner (primary) and
    seed_agent (fallback). On primary Exception it invokes the seed;
    on the seed also raising it emits empty completion. See
    WRAPPER_TEMPLATE above for the rationale.

    FROZEN once the submission has been evaluated. Staging is unconditional
    and runs BEFORE the already_evaluated() skip, so without this guard a
    re-run silently rewrites the staged source of an already-posted entry
    while skipping the eval — an invisible loss of the record of what produced
    it. Especially relevant here: the module docstring already notes v0_0_1 and
    v0_0_2 are NOT runnable at HEAD, so their staged trees are the only
    surviving description of those runs. --restage overwrites deliberately.

    Returns the working dir path.
    """
    dst_dir = WORKING_BASE / s.name
    if already_evaluated(dst_dir) and not restage:
        print(f"[skip stage] {s.name}: frozen (already evaluated); "
              f"pass --restage to overwrite")
        return dst_dir
    src_agent = SOURCE_BASE / s.name / s.agent_rel_path
    src_seed = EXAMPLES_DIR / "seeds" / "baseline" / "agent.py"
    src_registry = EXAMPLES_DIR / "model_registry.py"
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_agent, dst_dir / "agent_inner.py")
    shutil.copy(src_seed, dst_dir / "seed_agent.py")
    (dst_dir / "agent.py").write_text(WRAPPER_TEMPLATE)
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
    # LITELLM_LOCAL_MODEL_COST_MAP is required by `astabench score`;
    # passing during eval too is harmless and keeps env consistent.
    # ANTHROPIC_API_KEY is injected here (not in the parent process)
    # so Inspect's --model anthropic/... resolution succeeds without
    # clobbering the user's Claude Code CLI subscription credentials
    # in their interactive shell. See resolve_anthropic_key().
    extra_env = {
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "ANTHROPIC_API_KEY": resolve_anthropic_key(),
    }
    if s.needs_stronger_models:
        # The agent's source imports gated handles (CLAUDE_OPUS_4_7,
        # GPT_5_5, GEMINI_3_1_PRO_PREVIEW) which model_registry only
        # defines when this env var is set at module-load time. Without
        # it, `astabench eval` aborts at startup with an ImportError
        # before any solver runs. Scoped per-Submission so a seed-only
        # tarball doesn't claim it needs stronger handles it isn't
        # importing.
        extra_env["ASTA_DS1000_ALLOW_STRONGER_MODELS"] = "1"
    rc = run(
        [
            "astabench", "eval",
            "--solver", "agent.py",
            "--model", s.model_arg,
            "--split", "test",
            "--task", "DS_1000_test",
            "--log-dir", str(log_dir),
            "--display", "plain",
            # Concurrency knobs forwarded to `inspect eval-set` via
            # agenteval's ignore_unknown_options Click context. Sized so
            # max-samples is the binding constraint:
            #   - 6 in-flight samples (half the 12-core M4 Pro)
            #   - 6 Docker sandboxes (one per sample; iter10 packs its 4
            #     candidates into one container via exec() namespaces)
            #   - 30 connections (peak Anthropic burst = 6 samples × 2
            #     concurrent calls = 12; 30 is 2.5× headroom for the
            #     sequential downstream Anthropic calls that follow)
            "--max-samples", "6",
            "--max-sandboxes", "6",
            "--max-connections", "30",
        ],
        cwd=working_dir,
        extra_env=extra_env,
    )
    return rc == 0


def score_submission(working_dir: Path) -> bool:
    # LITELLM_LOCAL_MODEL_COST_MAP=True is REQUIRED by `astabench score`
    # (it refuses to run without it). It forces litellm's *bundled* price
    # map — so the installed litellm's bundled map must include every
    # model the agent actually called, or agenteval prices the run to
    # cost=null and the leaderboard entry ships without a cost figure.
    # v0_0_3 hit this: gemini-3.5-flash and gpt-5.5 are absent from
    # litellm <=1.83 bundled maps (agent-eval 0.1.50 over-pins
    # litellm<=1.83.13; litellm 1.88.1 works in practice and prices
    # both). The null-cost check below catches it either way.
    rc = run(
        ["astabench", "score", str(working_dir / LOG_SUBDIR)],
        cwd=working_dir,
        extra_env={"LITELLM_LOCAL_MODEL_COST_MAP": "True"},
    )
    if rc != 0:
        return False
    stats_path = working_dir / LOG_SUBDIR / "summary_stats.json"
    try:
        stats = json.loads(stats_path.read_text())
        cost = stats["stats"]["task/DS_1000_test"]["cost"]
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


def warm_base_image() -> None:
    """Best-effort pre-pull of the sandbox base image before evaluating.

    astabench's DS-1000 sandbox uses `build: .` against a Dockerfile whose
    `FROM` is a Docker Hub tag, so every per-sample sandbox build resolves
    that tag against Hub. Warming the image locally first reduces (does not
    eliminate) the per-build Hub auth contact that intermittently 404s
    mid-run. Authenticating with `docker login` is the stronger lever for
    that (higher rate limits, stabler tokens) but is per-machine secret
    state, so it stays a documented prerequisite rather than living here.

    The image TAG is read from astabench's own Dockerfile rather than
    hardcoded, so a future astabench base bump can't silently leave us
    warming the wrong image. (The Dockerfile's package-relative PATH is a
    structural assumption; a layout change degrades safely to skip-warm,
    since the eval pulls on its own.)

    Strictly non-fatal: image-resolution errors, a missing/broken docker
    binary, a hung pull (timeout), and a non-zero pull all only warn. A
    best-effort warm must never block the run -- the eval surfaces a clearer
    error if Docker is truly unavailable, and the caller already skips this
    entirely when no eval is pending.
    """
    try:
        import astabench

        dockerfile = Path(astabench.__file__).parent / "util" / "sandbox" / "Dockerfile"
        # FROM [--flag=val ...] <image> [AS <stage>] -- the image is the
        # first non-flag token after FROM (flag-aware so a future
        # `FROM --platform=... python:...` resolves to the image, not --platform).
        image = None
        for ln in dockerfile.read_text().splitlines():
            if ln.strip().upper().startswith("FROM "):
                image = next((t for t in ln.split()[1:] if not t.startswith("--")), None)
                break
        if not image:
            print("[warm] no FROM image in astabench sandbox Dockerfile; skipping pre-pull")
            return
    except Exception as e:
        print(f"[warm] could not resolve sandbox base image; skipping pre-pull ({e})")
        return
    print(f"[warm] pre-pulling sandbox base image {image} (best-effort)")
    try:
        rc = subprocess.run(["docker", "pull", image], timeout=600).returncode
    except subprocess.TimeoutExpired:
        print(f"[warm] pre-pull of {image} timed out (>600s); continuing -- the eval "
              "will attempt its own pull (see README 'Docker prerequisites')")
        return
    except OSError as e:
        print(f"[warm] could not run docker pull ({e}); continuing -- the eval will "
              "surface a clearer error if Docker is unavailable")
        return
    if rc != 0:
        print(f"[warm] pre-pull of {image} failed; continuing -- the eval will "
              "attempt its own pull (see README 'Docker prerequisites')")


def main() -> int:
    # --only NAME [NAME ...]: restrict to a subset of SUBMISSIONS.
    # Default remains all submissions (idempotency skips completed
    # evals, but --only also avoids re-scoring/re-tarring old entries
    # whose tarballs were already uploaded).
    args = sys.argv[1:]
    # --restage: overwrite a frozen (already-evaluated) working dir. Pulled out
    # first so the positional --only parsing below is unchanged.
    restage = "--restage" in args
    args = [a for a in args if a != "--restage"]
    selected = SUBMISSIONS
    if args:
        if args[0] != "--only" or len(args) < 2:
            print(f"usage: {sys.argv[0]} [--restage] [--only NAME [NAME ...]]")
            return 2
        names = set(args[1:])
        unknown = names - {s.name for s in SUBMISSIONS}
        if unknown:
            print(f"unknown submission(s): {sorted(unknown)}")
            print(f"known: {[s.name for s in SUBMISSIONS]}")
            return 2
        selected = [s for s in SUBMISSIONS if s.name in names]

    verify_clean_tree()
    verify_credentials()
    # Only warm the sandbox base image if an eval is actually pending --
    # a re-run that just re-scores/re-tars already-evaluated submissions
    # shouldn't trigger a redundant pull.
    if any(not already_evaluated(WORKING_BASE / s.name) for s in selected):
        warm_base_image()
    failures = []
    for s in selected:
        print(f"\n{'#' * 70}\n# {s.name}\n{'#' * 70}")
        working_dir = stage(s, restage)
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
