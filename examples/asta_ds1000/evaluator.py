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
from inspect_ai.solver import solver, use_tools

from astabench.tools import python_session
from inspect_evals.ds1000.ds1000 import ds1000_scorer

logger = logging.getLogger(__name__)

# Tracks whether we've already warned about a model litellm couldn't price.
# Once-per-process to avoid log spam across hundreds of evals.
_unpriced_models_warned: set[str] = set()


DEFAULT_COST_BUDGET = 0.06
COST_BREACH_PENALTY = 0.9

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

    Two assumptions:
      1. The scorer returns string `"C"` (correct) or `"I"` (incorrect).
         Source: inspect_evals/ds1000/ds1000.py:107.
      2. The scorer reads `state.metadata["code_context"]` to construct
         the test program (line 89). Our metadata-scrubbing wrapper
         depends on this — we restore the key before the scorer runs.
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
        model: str = "openai/gpt-5.4-mini",
        cost_budget: float = DEFAULT_COST_BUDGET,
        log_dir: str | None = None,
        skip_docker_check: bool = False,
        subprocess_isolation: bool = True,
        eval_timeout: int = 600,
        apply_cost_penalty: bool = True,
    ):
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. The default solver model is "
                "GPT-5.4 Mini via OpenAI. Set the env var before running."
            )
        if not skip_docker_check:
            _check_docker_available()

        self.model = model
        self.cost_budget = cost_budget
        self.subprocess_isolation = subprocess_isolation
        # True for training (RoboPhD's ELO competition) — soft penalty
        # nudges evolution toward cheaper agents. False for test paths
        # (eval_candidate / eval_run / --eval-test-set / --eval-only) —
        # test scores are raw 0/1 so the agent lands at its true point
        # on the Pareto cost-vs-score curve.
        self.apply_cost_penalty = apply_cost_penalty

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
            "model": self.model,
            "cost_budget": self.cost_budget,
            "eval_timeout": self.eval_timeout,
            "apply_cost_penalty": self.apply_cost_penalty,
            "subprocess_isolation": self.subprocess_isolation,
            "skip_docker_check": True,
        }
        base.update(overrides)
        return Ds1000Evaluator(**base)

    # -- RoboPhD evaluator contract -----------------------------------------

    def __call__(self, candidate: dict, example, *, problem_dir=None) -> tuple[float, dict]:
        if self.subprocess_isolation:
            return self._evaluate_via_subprocess(candidate, example)
        return self.evaluate(candidate, example, problem_dir=problem_dir)

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
                "model": self.model,
                "cost_budget": self.cost_budget,
                "apply_cost_penalty": self.apply_cost_penalty,
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
            scorer=ds1000_scorer(),
            sandbox=("docker", _sandbox_compose_path()),
        )

        captured = io.StringIO()
        try:
            with redirect_stdout(captured):
                logs = inspect_eval(
                    task,
                    model=self.model,
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
            break  # single scorer; first entry wins

        # The scorer's explanation contains the test program's stdout
        # and stderr — most actionable signal for diagnosing failures.
        if score_explanation:
            diagnostics["test_result.md"] = str(score_explanation)

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

        cost_breached = agent_cost_usd > self.cost_budget
        cost_penalty_applied = cost_breached and self.apply_cost_penalty
        if cost_penalty_applied:
            score_value = score_value * COST_BREACH_PENALTY

        with self._cost_lock:
            self.total_eval_cost += agent_cost_usd

        diagnostics["score"] = score_value
        diagnostics["cost_usd"] = agent_cost_usd
        diagnostics["agent_cost_usd"] = agent_cost_usd
        diagnostics["cost_breached"] = cost_breached
        diagnostics["cost_penalty_applied"] = cost_penalty_applied
        diagnostics["cost_budget"] = self.cost_budget
        diagnostics["usage"] = usage_summary
        diagnostics["library"] = (example.metadata or {}).get("library")

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
