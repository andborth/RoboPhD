"""
DS-1000 evaluator for AstaBench (Standard tools tier, Docker sandbox).

Each candidate is an Inspect `@solver` factory exported as `make_solver`
in agent.py. The evaluator runs the candidate against one DS-1000
sample at a time via `inspect.eval()` with a 1-sample dataset, attaches
the `python_session` tool inside a Docker sandbox, and returns the
binary score (0.0 or 1.0) along with cost/usage diagnostics.

Two deliberate properties:

1. The DS-1000 scorer is fully programmatic. No judge LLM. So the
   "agent vs judge" cost split that DiscoveryBench needs reduces to
   a single agent_cost_usd here.

2. `state.metadata` carries `code_context` (the test harness AND the
   reference code) and `perturbation_type` (a training-set membership
   signal). The scorer must read `code_context`; the solver must not.
   We solve this by wrapping the candidate's solver in an outer
   higher-order function that pops both keys from `state.metadata`
   before delegating, then restores them in a `finally` block so the
   scorer (which runs after the solver) sees the canonical metadata.
"""

import base64
import importlib.util
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from contextlib import redirect_stdout
from importlib import resources
from pathlib import Path
from typing import Any

from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Scorer, accuracy, scorer, stderr
from inspect_ai.solver import solver, use_tools
from inspect_ai.util import sandbox

from astabench.tools import python_session
from inspect_evals.ds1000.ds1000 import ds1000_scorer, postprocess

logger = logging.getLogger(__name__)

# Tracks whether we've already warned about a model litellm couldn't price.
# Once-per-process to avoid log spam across hundreds of evals.
_unpriced_models_warned: set[str] = set()


# Per-example score during training (apply_cost_penalty=True):
#   score = SCORE_SCALE * raw_score - cost_penalty
#   cost_penalty = min(1.0, max(0, agent_cost_usd - MIN_COST_THRESHOLD)
#                          / (COST_PENALTY_SATURATION - MIN_COST_THRESHOLD))
# raw_score is binary (1.0 for "C" / 0.0 for "I"). At MIN_COST_THRESHOLD
# the penalty is 0 (free zone for trivial agents); at COST_PENALTY_SATURATION
# it hits its max of 1.0. SCORE_SCALE puts raw_score on a [0, 100] axis
# so the [0, 1] penalty acts as a tiebreaker that never overrides
# correctness gaps. Test scores remain raw 0/1 — leaderboard parity.
SCORE_SCALE = 100.0
MIN_COST_THRESHOLD = 0.01      # matches DiscoveryBench
COST_PENALTY_SATURATION = 1.0  # matches DiscoveryBench

# Keys in state.metadata the agent must NOT see. The scorer reads
# `code_context`; we keep it on the Sample but pop it from state.metadata
# at solve time and restore it before the scorer runs. `perturbation_type`
# names whether this problem is an "Origin" (memorizable) or perturbed
# variant — leaking it would let a clever agent route to a memorized cache.
LEAK_KEYS: tuple[str, ...] = ("code_context", "perturbation_type")

# Expected score values from inspect_evals.ds1000.ds1000_scorer.
# Defensive: assert the read-out matches one of these so a future
# upstream restructure (e.g., switching to numeric scores) surfaces
# loudly instead of silently scoring everything 0.0.
SCORE_C = "C"
SCORE_I = "I"


def _check_upstream_invariants() -> None:
    """Warn if upstream inspect_evals drifts away from the structures we depend on.

    Brittle by design — a one-time startup warning beats silent scoring
    bugs when the upstream package version-bumps.

    Three assumptions:
      1. The scorer returns string `"C"` (correct) or `"I"` (incorrect).
         Source: inspect_evals/ds1000/ds1000.py:107.
      2. The scorer reads `state.metadata["code_context"]` to construct
         the test program (line 89). Our metadata-scrubbing wrapper
         depends on this — we restore the key before the scorer runs.
      3. `postprocess(text) -> str` exists with the standard signature.
         Our value-capture probe imports it (line 47) and calls it on
         the agent's completion to extract the code body the scorer
         would see. A signature change (e.g., adding required kwargs)
         would silently break the probe path; renames would fail at
         import time, which is loud enough.
    """
    try:
        import inspect as _inspect
        import importlib
        # `inspect_evals.ds1000` is a package; its __init__ re-exports the
        # *function* `ds1000` from the same-named submodule, so a normal
        # `import inspect_evals.ds1000.ds1000` resolves the attribute path
        # to the function (614 chars), not the submodule (4.8K). Use
        # importlib to force the submodule lookup.
        _mod = importlib.import_module("inspect_evals.ds1000.ds1000")

        src = _inspect.getsource(_mod)
        if 'value="C" if' not in src:
            logger.warning(
                "Upstream inspect_evals.ds1000 no longer returns string "
                "'C'/'I' — score conversion in evaluator may be wrong."
            )
        if 'state.metadata["code_context"]' not in src:
            logger.warning(
                "Upstream inspect_evals.ds1000 scorer no longer reads "
                "state.metadata['code_context'] — metadata scrubbing "
                "wrapper assumptions are stale."
            )
        if "def postprocess(code: str)" not in src:
            logger.warning(
                "Upstream inspect_evals.ds1000.postprocess no longer has "
                "signature `(code: str) -> str` — value-capture probe "
                "may misformat solutions."
            )
    except Exception:
        # Source inspection is best-effort.
        pass


_check_upstream_invariants()


# ---------------------------------------------------------------------------
# Public dataset loader
# ---------------------------------------------------------------------------

def load_ds1000(split: str = "validation") -> list[Sample]:
    """Load DS-1000 samples via AstaBench's wrapper.

    `split` is "validation" (100 samples) or "test" (900 samples).
    The wrapper appends AstaBench's FORMAT_INSTRUCTIONS to each
    sample.input and pins the canonical 100/900 split.

    Note: AstaBench's task constructor chains into
    `inspect_evals.ds1000.ds1000()`, which calls `get_model()` to pick
    a model-specific system message. That requires `INSPECT_EVAL_MODEL`
    to be set. We don't need a real model just to load the dataset, so
    we set a placeholder for the duration of this call (and unset it
    again in the finally block) when nothing is configured. This is
    benign — the model is only consulted to choose between two
    system-message strings, and the actual model used at evaluate()
    time comes from the `model=` arg passed to inspect.eval().
    """
    from astabench.evals.inspect_eval_wrappers.ds1000 import ds1000 as astabench_ds1000

    prev = os.environ.get("INSPECT_EVAL_MODEL")
    if not prev:
        os.environ["INSPECT_EVAL_MODEL"] = "openai/gpt-4o-mini"
    try:
        task = astabench_ds1000(split=split)
        return list(task.dataset)
    finally:
        if prev is None:
            os.environ.pop("INSPECT_EVAL_MODEL", None)


# ---------------------------------------------------------------------------
# Docker pre-flight
# ---------------------------------------------------------------------------

def _check_docker_available() -> None:
    if shutil.which("docker") is None:
        raise RuntimeError(
            "Docker is not installed. DS-1000 requires a running Docker "
            "daemon for the python_session sandbox. See examples/"
            "asta_ds1000/README.md for setup options."
        )
    try:
        subprocess.run(
            ["docker", "info"], check=True, capture_output=True, timeout=10
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(
            "Docker daemon is not responding. DS-1000 requires a running "
            "Docker daemon. See examples/asta_ds1000/README.md for setup."
        ) from e


# ---------------------------------------------------------------------------
# Candidate import
# ---------------------------------------------------------------------------

def _import_candidate_solver(agent_code: str) -> Any:
    """Materialize candidate agent.py into a temp module and return its
    @solver factory function (must be exported as `make_solver`).
    """
    mod_name = f"_ds1000_candidate_{uuid.uuid4().hex}"
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(agent_code)
        spec = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        try:
            spec.loader.exec_module(mod)
            if not hasattr(mod, "make_solver"):
                raise RuntimeError(
                    "candidate agent.py must define a function named "
                    "`make_solver` decorated with @solver "
                    "(see seeds/baseline/agent.py)"
                )
            return mod.make_solver
        finally:
            sys.modules.pop(mod_name, None)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Metadata scrubbing wrapper
# ---------------------------------------------------------------------------

@solver
def _wrap_with_metadata_scrub(inner_solver):
    """Solver factory that wraps the candidate's solver instance.

    Pops LEAK_KEYS from state.metadata before delegating, restores them
    in a finally block so the scorer (which runs after the solver) can
    still read state.metadata['code_context']. The candidate solver only
    ever sees state.metadata['library'].

    Decorated with @solver so Inspect's registry can track the wrapper
    — without it, inspect.eval raises "does not have registry info".
    """
    async def solve(state, generate):
        saved = {k: state.metadata.pop(k) for k in LEAK_KEYS if k in state.metadata}
        try:
            return await inner_solver(state, generate)
        finally:
            state.metadata.update(saved)
    return solve


# ---------------------------------------------------------------------------
# Diagnostic-artifact-capturing scorer wrapper
# ---------------------------------------------------------------------------

# Probe body (everything after `solution = <repr>`). Static — built by
# concatenation so we never run str.format / % over arbitrary Python.
# code_context can legitimately contain f-strings, dict literals, and
# format-spec patterns ({n:>5}, {x!r}); using .format() to substitute
# the template would raise KeyError on those and the outer try/except
# would swallow it, dropping the artifact silently.
_VALUE_PROBE_BODY = """
_test_input, _expected = generate_test_case(1)
_actual = '<not captured>'
try:
    _code = exec_context.replace('[insert]', solution)
    _env = {'test_input': _test_input}
    exec(_code, _env)
    _actual = _env.get('result')
except BaseException as _e:
    _actual = f'<exec failed: {type(_e).__name__}: {_e}>'
print('===EXPECTED===')
print(repr(_expected)[:4000])
print('===ACTUAL===')
print(repr(_actual)[:4000])
"""


# Tracks whether we've already logged a probe-execution failure
# (issue 3 mitigation: silent-fail bucket gets one warning per process,
# not per evaluation).
_probe_failure_logged = False


@scorer(metrics=[accuracy(), stderr()])
def _ds1000_scorer_with_artifacts() -> Scorer:
    """Wrap the canonical ds1000_scorer to also capture diagnostic artifacts.

    Two pieces, both best-effort (any failure is silent):

    1. **Matplotlib PNG capture**: read /ds1000/output.png (agent's render)
       and /ds1000/ans.png (reference render) as bytes; base64-encode and
       stash in Score.metadata under '__actual_png_b64' / '__expected_png_b64'.
       The double-underscore prefix marks them for parent-side decoding by
       Ds1000Evaluator._write_png_artifacts (binary files don't fit the
       diagnostic-key-as-string convention).

    2. **Value-based capture (non-Matplotlib)**: run a probe that mirrors
       test_execution but captures `result` instead of asserting. Stashes
       repr(actual) and repr(expected) (each capped to 4KB) in Score.metadata
       under 'actual_repr' / 'expected_repr'.

    The value probe runs even on Matplotlib problems, but `expected_result`
    is None for them (image-based comparison) — useful as a confirmation
    signal.
    """
    inner = ds1000_scorer()

    async def score(state, target):
        result = await inner(state, target)
        artifacts: dict[str, Any] = {}
        sb = sandbox()

        # 1. Matplotlib PNG capture. Silent miss for non-Matplotlib problems
        # (file not found is the dominant outcome there).
        for src_name, dst_key in (
            ("output.png", "__actual_png_b64"),
            ("ans.png", "__expected_png_b64"),
        ):
            try:
                data = await sb.read_file(f"/ds1000/{src_name}", text=False)
                if data:
                    artifacts[dst_key] = base64.b64encode(data).decode("ascii")
            except Exception:
                pass

        # 2. Value-based capture via probe. We pull code_context from
        # state.metadata (which the metadata-scrubbing solver wrapper has
        # already restored by the time the scorer runs).
        cc = state.metadata.get("code_context", "") if state.metadata else ""
        if (
            cc
            and "exec_context" in cc
            and "generate_test_case" in cc
            and "[insert]" in cc
        ):
            global _probe_failure_logged
            try:
                solution = postprocess(state.output.completion or "")
                # Build by concatenation — never .format() over `cc` because
                # arbitrary Python source has braces that would be
                # misinterpreted as format specs.
                probe = cc + "\nsolution = " + repr(solution) + _VALUE_PROBE_BODY
                # Note: probe file persists in the sandbox after this scoring
                # pass. Bounded harmless since each sample gets a fresh
                # container and the container is torn down at sample end.
                # If container reuse is ever introduced, add an explicit
                # cleanup after `sb.exec` returns.
                probe_path = f"/ds1000/_probe_{state.sample_id}.py"
                await sb.write_file(probe_path, probe)
                probe_res = await sb.exec(
                    ["python", probe_path], cwd="/ds1000", timeout=30
                )
                if probe_res and probe_res.success and probe_res.stdout:
                    out = probe_res.stdout
                    if "===EXPECTED===" in out and "===ACTUAL===" in out:
                        expected_part = (
                            out.split("===EXPECTED===", 1)[1]
                            .split("===ACTUAL===", 1)[0]
                            .strip()
                        )
                        actual_part = out.split("===ACTUAL===", 1)[1].strip()
                        if expected_part:
                            artifacts["expected_repr"] = expected_part[:4000]
                        if actual_part:
                            artifacts["actual_repr"] = actual_part[:4000]
                elif not _probe_failure_logged:
                    _probe_failure_logged = True
                    logger.warning(
                        "Value probe returned non-success or empty stdout for "
                        "sample %s. Subsequent failures will be silent. "
                        "Stderr: %s",
                        state.sample_id,
                        (getattr(probe_res, "stderr", "") or "")[:500],
                    )
            except Exception as e:
                if not _probe_failure_logged:
                    _probe_failure_logged = True
                    logger.warning(
                        "Value probe raised %s for sample %s. Subsequent "
                        "failures will be silent. Probable causes: upstream "
                        "DS-1000 renamed exec_context/generate_test_case, or "
                        "code_context structure changed.",
                        type(e).__name__,
                        state.sample_id,
                    )

        merged = {**(result.metadata or {}), **artifacts}
        return Score(
            value=result.value,
            answer=result.answer,
            explanation=result.explanation,
            metadata=merged,
        )

    return score


# ---------------------------------------------------------------------------
# Sandbox compose path
# ---------------------------------------------------------------------------

def _sandbox_compose_path() -> str:
    """Path to AstaBench's docker compose file used by `python_session`."""
    return (resources.files("astabench.util.sandbox") / "sandbox_compose.yaml").as_posix()


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Ds1000Evaluator:
    """RoboPhD evaluator for DS-1000.

    Contract: __call__(candidate, example, *, problem_dir=None) -> (score, diagnostics).

    `candidate` is a {"agent.py": "<source>"} mapping (RoboPhD's standard
    file_mapping shape).
    `example` is a dict (Sample.model_dump()) — main.py converts before
    calling so RoboPhD's domain layer can SHA256 stable IDs.
    """

    def __init__(
        self,
        log_dir: str | None = None,
        skip_docker_check: bool = False,
        subprocess_isolation: bool = True,
        eval_timeout: int = 600,
        apply_cost_penalty: bool = True,
        min_cost_threshold: float = MIN_COST_THRESHOLD,
        cost_penalty_saturation: float = COST_PENALTY_SATURATION,
    ):
        # Hard requirement: every provider key the registry references
        # must be set, even if the seed only uses one. Evolution can
        # produce an agent that calls Claude or Gemini at any iteration,
        # and the failure mode if the key is missing is a 401 mid-run
        # (worst time to discover the gap). Fail loudly at startup
        # instead. ANTHROPIC accepts either ANTHROPIC_API_KEY or
        # ANTHROPIC_API_KEY_FOR_ROBOPHD per RoboPhD convention; the
        # FOR_ROBOPHD variant lets the user's Claude Code CLI keep
        # using its own subscription credentials while RoboPhD uses a
        # separate API key (see model_registry.py).
        missing = []
        if not os.environ.get("OPENAI_API_KEY"):
            missing.append("OPENAI_API_KEY")
        if not (os.environ.get("ANTHROPIC_API_KEY") or
                os.environ.get("ANTHROPIC_API_KEY_FOR_ROBOPHD")):
            missing.append("ANTHROPIC_API_KEY (or ANTHROPIC_API_KEY_FOR_ROBOPHD)")
        if not os.environ.get("GOOGLE_API_KEY"):
            missing.append("GOOGLE_API_KEY")
        if missing:
            raise RuntimeError(
                f"Missing required env vars: {', '.join(missing)}. "
                f"DS-1000 evolution may pick any of three solver models "
                f"(gpt-5.4-mini, claude-haiku-4-5-20251001, "
                f"gemini-3.1-flash-lite-preview); all three provider keys "
                f"must be set before running, or evolution will fail "
                f"mid-run when it produces an agent that uses a model "
                f"with no key configured."
            )
        if not skip_docker_check:
            _check_docker_available()

        # Resolve registry handles at construction time (after the env-var
        # check so the friendly error fires first if keys are missing).
        # Storing self._default_model surfaces any registry-resolution
        # failure here rather than waiting for the first .generate()
        # inside inspect.eval.
        from model_registry import GPT_5_4_MINI as _DEFAULT_MODEL
        self._default_model = _DEFAULT_MODEL

        self.subprocess_isolation = subprocess_isolation
        # True for training (RoboPhD's ELO competition): score is
        # SCORE_SCALE * raw_score - cost_penalty. False for test paths
        # (eval_candidate / eval_run / --eval-test-set / --eval-only) —
        # score is raw 0/1 so the agent lands at its true point on the
        # Pareto cost-vs-score curve. agent_cost_usd is recorded in
        # diagnostics in both modes for the audit trail.
        self.apply_cost_penalty = apply_cost_penalty
        # Catch the misconfiguration class up-front: equal endpoints
        # would divide by zero when computing cost_penalty; crossed
        # endpoints would give a negative ramp width and produce huge
        # penalties for tiny costs (sign-flip). Negative threshold
        # would mean every agent pays a penalty — likely not intended.
        if min_cost_threshold < 0:
            raise ValueError(
                f"min_cost_threshold must be >= 0; got {min_cost_threshold}"
            )
        if min_cost_threshold >= cost_penalty_saturation:
            raise ValueError(
                f"min_cost_threshold ({min_cost_threshold}) must be strictly "
                f"less than cost_penalty_saturation ({cost_penalty_saturation}); "
                f"otherwise the cost-penalty ramp has zero or negative width"
            )
        self.min_cost_threshold = min_cost_threshold
        self.cost_penalty_saturation = cost_penalty_saturation

        # Subprocess kill-after timeout MUST be less than RoboPhD's
        # eval_timeout so the subprocess gets SIGKILLed before RoboPhD's
        # reaper would leak the thread. Buffer of 30s.
        self.eval_timeout = eval_timeout
        self.subprocess_timeout = max(eval_timeout - 30, 60)

        self._log_dir = log_dir or tempfile.mkdtemp(prefix="ds1000_eval_")
        self.total_eval_cost = 0.0
        self._cost_lock = threading.Lock()

    # -- Construction helpers -----------------------------------------------

    def with_overrides(self, **overrides: Any) -> "Ds1000Evaluator":
        """Return a sibling evaluator inheriting this one's config.

        Used by main.py to derive the test evaluator from the training
        one with `apply_cost_penalty=False`. Any future constructor
        field added to Ds1000Evaluator must also be added here.
        """
        base = {
            "eval_timeout": self.eval_timeout,
            "apply_cost_penalty": self.apply_cost_penalty,
            "min_cost_threshold": self.min_cost_threshold,
            "cost_penalty_saturation": self.cost_penalty_saturation,
            "subprocess_isolation": self.subprocess_isolation,
            "skip_docker_check": True,
        }
        base.update(overrides)
        return Ds1000Evaluator(**base)

    # -- RoboPhD evaluator contract -----------------------------------------

    def __call__(self, candidate: dict, example, *, problem_dir=None) -> tuple[float, dict]:
        if self.subprocess_isolation:
            score, diagnostics = self._evaluate_via_subprocess(candidate, example)
        else:
            score, diagnostics = self.evaluate(candidate, example, problem_dir=problem_dir)
        # Decode any PNG artifacts (Matplotlib problems) and write to
        # problem_dir as binary files. Strips the b64 keys regardless,
        # since they don't fit ExternalEvaluatorDomain's str-only file
        # convention and would otherwise persist as huge text blobs.
        self._write_png_artifacts(diagnostics, problem_dir)
        return score, diagnostics

    @staticmethod
    def _write_png_artifacts(diagnostics: dict, problem_dir: Any) -> None:
        """Pop PNG b64 entries from diagnostics; write decoded bytes to
        problem_dir/{actual,expected}_output.png if problem_dir is set.
        Always pops the keys (so they don't bloat persisted diagnostics)
        even when problem_dir is None.
        """
        for src_key, dst_name in (
            ("__actual_png_b64", "actual_output.png"),
            ("__expected_png_b64", "expected_output.png"),
        ):
            b64_data = diagnostics.pop(src_key, None)
            if not b64_data or not problem_dir:
                continue
            try:
                Path(problem_dir, dst_name).write_bytes(base64.b64decode(b64_data))
            except Exception as e:
                logger.warning("failed to write %s to %s: %s", dst_name, problem_dir, e)

    def _evaluate_via_subprocess(self, candidate: dict, example) -> tuple[float, dict]:
        """Run one evaluation in a fresh Python subprocess.

        Inspect-AI's `inspect.eval()` raises if two calls are in flight
        in the same Python process. Subprocess isolation gives us real
        parallelism across RoboPhD's worker threads.
        """
        if isinstance(example, dict):
            example_dict = example
        elif hasattr(example, "model_dump"):
            example_dict = example.model_dump()
        else:
            return 0.0, {
                "error": f"_evaluate_via_subprocess expects Sample or dict; got {type(example).__name__}"
            }

        worker_path = Path(__file__).resolve().parent / "_eval_worker.py"
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as inf:
            json.dump({
                "candidate": candidate,
                "example": example_dict,
                "apply_cost_penalty": self.apply_cost_penalty,
                "min_cost_threshold": self.min_cost_threshold,
                "cost_penalty_saturation": self.cost_penalty_saturation,
            }, inf, default=str)
            inf_path = inf.name
        out_path = inf_path + ".out"

        try:
            try:
                result = subprocess.run(
                    [sys.executable, str(worker_path), inf_path, out_path],
                    capture_output=True,
                    text=True,
                    timeout=self.subprocess_timeout,
                )
            except subprocess.TimeoutExpired as e:
                return 0.0, {
                    "error": f"subprocess timed out after {self.subprocess_timeout}s",
                    "subprocess_stderr": (e.stderr or "")[-2000:] if e.stderr else "",
                }
            if result.returncode != 0:
                return 0.0, {
                    "error": f"subprocess failed (exit {result.returncode})",
                    "subprocess_stderr": result.stderr[-2000:] if result.stderr else "",
                }
            try:
                with open(out_path) as f:
                    payload = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                return 0.0, {
                    "error": f"subprocess produced no valid output: {type(e).__name__}: {e}",
                    "subprocess_stderr": result.stderr[-2000:] if result.stderr else "",
                }

            score = float(payload.get("score", 0.0))
            diagnostics = payload.get("diagnostics", {}) or {}
            agent_only_cost = diagnostics.get("agent_cost_usd", 0.0) or 0.0
            with self._cost_lock:
                self.total_eval_cost += agent_only_cost
            return score, diagnostics
        finally:
            for p in (inf_path, out_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    def evaluate(self, candidate: dict, example, *, problem_dir=None) -> tuple[float, dict]:
        agent_code = candidate.get("agent.py", "")
        if not agent_code:
            return 0.0, {"error": "candidate missing agent.py"}

        if isinstance(example, dict):
            example = Sample(**example)
        elif not isinstance(example, Sample):
            raise TypeError(
                f"evaluate() expects Sample or dict (from Sample.model_dump); "
                f"got {type(example).__name__}"
            )

        try:
            solver_factory = _import_candidate_solver(agent_code)
        except Exception as e:
            return 0.0, {"error": f"candidate import failed: {type(e).__name__}: {e}"}

        # Wrap the candidate's solver instance so `state.metadata`'s
        # leak-prone keys are hidden during solve and restored before
        # the scorer reads them.
        wrapped_solver = _wrap_with_metadata_scrub(solver_factory())

        task = Task(
            dataset=MemoryDataset([example]),
            solver=wrapped_solver,
            setup=[use_tools(python_session())],
            scorer=_ds1000_scorer_with_artifacts(),
            sandbox=("docker", _sandbox_compose_path()),
        )

        captured = io.StringIO()
        try:
            with redirect_stdout(captured):
                logs = inspect_eval(
                    task,
                    model=self._default_model,
                    display="none",
                    log_dir=self._log_dir,
                    log_format="json",
                    log_level="warning",
                )
        except Exception as e:
            return 0.0, {
                "error": f"inspect.eval crashed: {type(e).__name__}: {e}",
                "agent_stdout": captured.getvalue(),
            }

        log = logs[0]
        return self._extract_score_and_diagnostics(log, example, captured.getvalue())

    # -- Result extraction --------------------------------------------------

    def _extract_score_and_diagnostics(
        self, log, example: Sample, agent_stdout: str
    ) -> tuple[float, dict]:
        diagnostics: dict[str, Any] = {
            "agent_stdout": agent_stdout,
            "sample_id": str(example.id),
        }

        # Surface the prompt the agent received and (for engineer-side
        # debugging only — never read by evolution) the reference target.
        diagnostics["problem.md"] = str(example.input or "")
        if example.target:
            target_str = (
                example.target if isinstance(example.target, str)
                else "\n".join(example.target) if isinstance(example.target, list)
                else str(example.target)
            )
            diagnostics["reference.md"] = target_str

        if not getattr(log, "samples", None):
            diagnostics["error"] = "no samples in eval log"
            return 0.0, diagnostics

        sample_log = log.samples[0]

        sample_err = getattr(sample_log, "error", None)
        if sample_err is not None:
            err_msg = getattr(sample_err, "message", None) or str(sample_err)
            diagnostics["error"] = err_msg[:1000]

        # Score read-out. The DS-1000 scorer returns string "C" or "I".
        # Anything else is a sign the upstream package changed its
        # surface — fail loud rather than silently scoring 0.
        scores = getattr(sample_log, "scores", None) or {}
        score_value = 0.0
        score_explanation = ""
        score_metadata: dict[str, Any] = {}
        for sc in scores.values():
            v = getattr(sc, "value", None)
            if v == SCORE_C:
                score_value = 1.0
            elif v == SCORE_I:
                score_value = 0.0
            else:
                logger.warning(
                    "Unexpected score value %r (expected %r/%r). "
                    "Returning 0.0; check inspect_evals.ds1000 for upstream changes.",
                    v, SCORE_C, SCORE_I,
                )
                score_value = 0.0
            score_explanation = getattr(sc, "explanation", "") or ""
            score_metadata = getattr(sc, "metadata", None) or {}
            break  # single scorer; first entry wins

        # The scorer's explanation contains the test program's stdout
        # and stderr — most actionable signal for diagnosing failures.
        if score_explanation:
            diagnostics["test_result.md"] = str(score_explanation)

        # Diagnostic artifacts from the wrapped scorer. expected/actual
        # repr go through the standard string-diagnostic-as-file path;
        # PNG b64 keys (double-underscore prefix) are decoded and written
        # as binary by __call__'s _write_png_artifacts step before returning.
        if score_metadata.get("expected_repr"):
            diagnostics["expected_output.md"] = str(score_metadata["expected_repr"])
        if score_metadata.get("actual_repr"):
            diagnostics["actual_output.md"] = str(score_metadata["actual_repr"])
        for b64_key in ("__actual_png_b64", "__expected_png_b64"):
            if score_metadata.get(b64_key):
                diagnostics[b64_key] = score_metadata[b64_key]

        # Agent's emitted completion (verbatim, before postprocess).
        completion = ""
        if sample_log.output:
            completion = getattr(sample_log.output, "completion", "") or ""
        if completion:
            diagnostics["extracted_code.md"] = completion[:4000]

        # Cost. DS-1000 has no judge LLM, so model_usage is all agent.
        agent_cost_usd = 0.0
        usage_summary: dict[str, Any] = {}
        try:
            stats = getattr(log, "stats", None)
            model_usage = getattr(stats, "model_usage", None) if stats else None
            if model_usage:
                for model_name, u in model_usage.items():
                    counts = {
                        "input_tokens": getattr(u, "input_tokens", 0),
                        "output_tokens": getattr(u, "output_tokens", 0),
                        "total_tokens": getattr(u, "total_tokens", 0),
                    }
                    usage_summary[model_name] = counts
                    agent_cost_usd += self._estimate_cost(model_name, counts)
        except Exception:
            pass

        # Cost penalty (training only). raw_score is rescaled to
        # [0, SCORE_SCALE] and a bounded penalty in [0, 1] is subtracted,
        # proportional to cost above min_cost_threshold up to
        # cost_penalty_saturation. The penalty's [0, 1] range is two
        # orders of magnitude smaller than the score scale, so it acts as
        # a tiebreaker and never reorders agents whose raw scores differ
        # (DS-1000 raw is binary 1.0/0.0). Test paths skip this — score
        # stays raw 0/1 for leaderboard parity.
        raw_score = score_value
        if self.apply_cost_penalty:
            cost_excess = max(0.0, agent_cost_usd - self.min_cost_threshold)
            cost_excess_range = self.cost_penalty_saturation - self.min_cost_threshold
            cost_penalty = min(1.0, cost_excess / cost_excess_range)
            score_value = SCORE_SCALE * raw_score - cost_penalty
        else:
            cost_penalty = 0.0

        with self._cost_lock:
            self.total_eval_cost += agent_cost_usd

        diagnostics["score"] = score_value           # post-penalty (training) or raw (test)
        diagnostics["raw_score"] = raw_score         # always 1.0 or 0.0 — pre-penalty
        diagnostics["cost_penalty"] = cost_penalty   # [0, 1] subtracted on training; 0.0 on test
        diagnostics["cost_usd"] = agent_cost_usd
        diagnostics["agent_cost_usd"] = agent_cost_usd
        diagnostics["usage"] = usage_summary
        diagnostics["library"] = (example.metadata or {}).get("library")
        # Numeric fields (raw_score, cost_penalty) are skipped by
        # ExternalEvaluatorDomain's str-only file persistence. Surface
        # them as a readable per-key file so failure forensics can
        # distinguish "wrong answer" (raw_score=0) from "correct but
        # expensive" (raw_score=1, cost_penalty>0) without re-running.
        diagnostics["cost_breakdown.md"] = (
            f"raw_score:        {raw_score}\n"
            f"agent_cost_usd:   ${agent_cost_usd:.6f}\n"
            f"cost_penalty:     {cost_penalty:.4f}\n"
            f"penalty_applied:  {self.apply_cost_penalty}\n"
            f"final score:      {score_value}\n"
        )

        return score_value, diagnostics

    # -- Cost helpers -------------------------------------------------------

    @staticmethod
    def _estimate_cost(model_name: str, counts: dict) -> float:
        """Estimate cost via litellm.cost_per_token.

        Returns 0.0 if litellm isn't installed, can't price the model,
        or raises. Logs a one-shot warning the first time a model with
        non-trivial usage prices to $0 — that pattern almost always
        means litellm's pricing table doesn't know about a recent
        model, in which case the cost cap silently never fires. Loud
        failure beats silently-zero spend tracking.
        """
        try:
            import litellm
        except ImportError:
            return 0.0
        input_tokens = counts.get("input_tokens", 0)
        output_tokens = counts.get("output_tokens", 0)
        try:
            pin, pout = litellm.cost_per_token(
                model=model_name,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
            )
            cost = (pin or 0.0) + (pout or 0.0)
        except Exception:
            cost = 0.0

        if cost == 0.0 and (input_tokens + output_tokens) > 100 and model_name not in _unpriced_models_warned:
            _unpriced_models_warned.add(model_name)
            logger.warning(
                "litellm priced %r at $0 despite %d input + %d output tokens. "
                "Cost cap will not fire for this model. Update litellm or "
                "patch cost_per_token mapping.",
                model_name, input_tokens, output_tokens,
            )
        return cost
