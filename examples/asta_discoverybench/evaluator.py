"""
DiscoveryBench evaluator for AstaBench (Standard tools tier, Docker sandbox).

Each candidate is an Inspect `@solver` factory exported as `make_solver` in
agent.py. The evaluator runs the candidate against one DiscoveryBench
sample at a time via `inspect.eval()` with a 1-sample dataset, attaches
the `python_session` tool inside a Docker sandbox, and returns the HMS
score along with cost/usage diagnostics.

Bypassing inspect.eval() and constructing TaskState manually was
considered for lower per-call latency but rejected — the Docker sandbox
lifecycle, AstaBench's submission tarball machinery, and the scorer's
gpt-4o judge calls all assume an Inspect-driven runtime.

Cost accounting splits agent vs judge:
  - `judge_cost_usd`: 5 fixed gpt-4o-2024-08-06 calls per sample
    (≈$0.029); excluded from the per-example cap because it's outside
    agent control.
  - `agent_cost_usd`: everything else (the candidate's own
    `get_model().generate()` calls and any wrapped out-of-band calls).
    Capped at $0.10/sample with a 0.9 score multiplier on breach.
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
from typing import Any

from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import use_tools

from astabench.evals.discoverybench.task import (
    load_discoverybench_hf,
    json_to_sample,
    score_discoverybench,
)
from astabench.tools import python_session

logger = logging.getLogger(__name__)


# IMPORTANT: this must match what the upstream scorer uses for judging.
# Source: astabench.evals.discoverybench.task.score_discoverybench passes
# `llm_used="gpt-4o-2024-08-06"` to run_eval_gold_vs_gen_NL_hypo_workflow.
# If the upstream version-bumps the judge model, this constant must
# update too — otherwise our agent-vs-judge cost split silently
# misclassifies the new judge as agent spend, breaching the cap. We
# do an import-time check below to surface that drift loudly.
JUDGE_MODEL_SHORT = "gpt-4o-2024-08-06"
DEFAULT_COST_BUDGET = 0.10
COST_BREACH_PENALTY = 0.9

# Inspect-AI's `inspect.eval()` (and its async counterpart `eval_async`)
# raise "Multiple concurrent calls to eval_async are not allowed" if two
# evaluations are in flight simultaneously in the same process. RoboPhD's
# default --max-workers is >1, so we serialize with a process-global lock.
#
# Practically, this makes --max-workers a no-op for *throughput*: the
# inspect.eval call dominates per-evaluation wall-clock (~50s/sample is
# almost entirely inside the lock), so concurrency reduces to ~1.
# RoboPhD's `--max-workers` argument is preserved for API compatibility
# but does not deliver parallelism for this evaluator. To restore real
# parallelism, the inspect.eval call would need to run in a subprocess
# per worker (see "Open" in README.md).
_INSPECT_EVAL_LOCK = threading.Lock()


def _check_upstream_invariants() -> None:
    """Warn if upstream astabench drifts away from the structures we depend on.

    Brittle by design — better a single startup warning than silent
    misclassification when astabench updates. Two things we depend on:

    1. The judge model. Source: astabench.evals.discoverybench.task.score_discoverybench
       passes `llm_used="gpt-4o-2024-08-06"` to run_eval_gold_vs_gen_NL_hypo_workflow.
       If upstream version-bumps the judge, JUDGE_MODEL_SHORT is wrong and our
       agent-vs-judge cost split silently misclassifies the new judge as agent spend.

    2. The Score.metadata structure. Source: astabench.evals.discoverybench.eval_utils
       returns a dict whose top-level includes `context_score` and `var_rel`, with
       `var_rel` containing `var.score.f1` and `rel.score`. _extract_score_and_diagnostics
       indexes those paths to surface per-dimension HMS in diagnostics; if upstream
       restructures the dict, those fields silently become None.
    """
    try:
        import inspect as _inspect
        from astabench.evals.discoverybench import task as _task
        from astabench.evals.discoverybench import eval_utils as _eval_utils

        task_src = _inspect.getsource(_task)
        eval_src = _inspect.getsource(_eval_utils)

        if JUDGE_MODEL_SHORT not in task_src:
            logger.warning(
                f"Upstream astabench.evals.discoverybench.task no longer "
                f"references {JUDGE_MODEL_SHORT!r} — the judge model may have "
                f"changed. Update JUDGE_MODEL_SHORT in this evaluator or "
                f"agent/judge cost classification will be wrong."
            )
        # Per-dim metadata structure — both names appear in the upstream
        # eval_utils source (in the dict construction). If they're absent,
        # the structure has changed and our extraction needs updating.
        for marker in ("context_score", "var_rel"):
            if marker not in eval_src:
                logger.warning(
                    f"Upstream astabench.evals.discoverybench.eval_utils no "
                    f"longer references {marker!r} — Score.metadata structure "
                    f"may have changed. Per-dim HMS extraction in "
                    f"_extract_score_and_diagnostics needs review."
                )
    except Exception:
        # Source inspection is best-effort. If it fails, assume the
        # invariants still hold rather than spamming false warnings.
        pass


_check_upstream_invariants()


# ---------------------------------------------------------------------------
# Public dataset loaders
# ---------------------------------------------------------------------------

def load_real(split: str = "validation") -> list[Sample]:
    """Load DiscoveryBench real/ samples via AstaBench's loader.

    `split` is "validation" (25) or "test" (239).
    """
    raw = load_discoverybench_hf(split=split)
    return [json_to_sample(x, split=split) for x in raw]


def load_synth(split: str):
    """Re-export from load_synth.py for a unified import surface."""
    from load_synth import load_synth as _load
    return _load(split)


# ---------------------------------------------------------------------------
# Docker pre-flight
# ---------------------------------------------------------------------------

def _check_docker_available() -> None:
    """Raise a clear error if Docker isn't installed or the daemon isn't running."""
    if shutil.which("docker") is None:
        raise RuntimeError(
            "Docker is not installed. DiscoveryBench requires a running Docker "
            "daemon for the python_session sandbox. See examples/asta_discoverybench/"
            "README.md for setup options (Docker Desktop / colima / OrbStack)."
        )
    try:
        subprocess.run(
            ["docker", "info"], check=True, capture_output=True, timeout=10
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(
            "Docker daemon is not responding. DiscoveryBench requires a running "
            "Docker daemon. See examples/asta_discoverybench/README.md for setup."
        ) from e


# ---------------------------------------------------------------------------
# Candidate import
# ---------------------------------------------------------------------------

def _import_candidate_solver(agent_code: str) -> Any:
    """Materialize candidate agent.py into a temp module and return its
    @solver factory function (must be exported as `make_solver`).

    Cleans up both the temp file and the sys.modules entry after the
    factory is captured. Function.__globals__ keeps a direct reference to
    the module's __dict__ so the factory remains executable after the
    sys.modules cleanup.
    """
    mod_name = f"_discoverybench_candidate_{uuid.uuid4().hex}"
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
# Sandbox compose path
# ---------------------------------------------------------------------------

def _sandbox_compose_path() -> str:
    """Path to AstaBench's docker compose file used by `python_session`."""
    return (resources.files("astabench.util.sandbox") / "sandbox_compose.yaml").as_posix()


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class DiscoveryBenchEvaluator:
    """RoboPhD evaluator for DiscoveryBench.

    Contract: evaluate(candidate, example) -> (float_score, diagnostics_dict).

    `candidate` is a {"agent.py": "<source>"} mapping (RoboPhD's standard
    file_mapping shape).
    `example` is an Inspect Sample loaded via load_real() or load_synth().
    """

    def __init__(
        self,
        model: str = "openai/gpt-5-mini",
        cost_budget: float = DEFAULT_COST_BUDGET,
        log_dir: str | None = None,
        skip_docker_check: bool = False,
    ):
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. The DiscoveryBench scorer judges "
                "hypotheses with gpt-4o-2024-08-06, and the default solver "
                "model is GPT-5 Mini — both go through OpenAI. Set the env "
                "var before running."
            )
        if not skip_docker_check:
            _check_docker_available()

        self.model = model
        self.cost_budget = cost_budget
        self._log_dir = log_dir or tempfile.mkdtemp(prefix="discoverybench_eval_")
        self.total_eval_cost = 0.0
        self._cost_lock = threading.Lock()

    # -- RoboPhD evaluator contract -----------------------------------------
    # RoboPhD invokes the evaluator object directly:
    #   evaluator(candidate, example, problem_dir=...) -> (score, diagnostics)

    def __call__(self, candidate: dict, example, *, problem_dir=None) -> tuple[float, dict]:
        return self.evaluate(candidate, example, problem_dir=problem_dir)

    def evaluate(self, candidate: dict, example, *, problem_dir=None) -> tuple[float, dict]:
        agent_code = candidate.get("agent.py", "")
        if not agent_code:
            return 0.0, {"error": "candidate missing agent.py"}

        # RoboPhD's domain layer wants JSON-serializable examples (it
        # SHA256s them for stable IDs), so main.py converts Sample to
        # dict via .model_dump() before passing to optimize_anything().
        # Reconstruct here, and fail loud on anything else (a programmatic
        # caller bypassing main.py would otherwise hit a deep AttributeError
        # later instead of seeing the boundary mismatch).
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

        task = Task(
            dataset=MemoryDataset([example]),
            solver=solver_factory(),
            setup=[use_tools(python_session())],
            scorer=score_discoverybench(),
            sandbox=("docker", _sandbox_compose_path()),
        )

        captured = io.StringIO()
        try:
            with _INSPECT_EVAL_LOCK, redirect_stdout(captured):
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
            "sample_id": example.id,
            "split": example.metadata.get("split", "real"),
        }

        if not getattr(log, "samples", None):
            diagnostics["error"] = "no samples in eval log"
            return 0.0, diagnostics

        sample_log = log.samples[0]

        sample_err = getattr(sample_log, "error", None)
        if sample_err is not None:
            err_msg = getattr(sample_err, "message", None) or str(sample_err)
            diagnostics["error"] = err_msg[:1000]

        # HMS score and per-dimension breakdown.
        # DiscoveryBench attaches a single scorer per sample; multiple
        # scorer entries would mean the upstream task definition changed
        # and we'd be silently picking whichever lands first.
        scores = getattr(sample_log, "scores", None) or {}
        if len(scores) > 1:
            logger.warning(
                f"Sample {example.id} produced {len(scores)} scores; "
                f"expected 1. Using first floatable. Names: {list(scores)}"
            )
        score_value = 0.0
        score_metadata: dict[str, Any] = {}
        for sc in scores.values():
            v = getattr(sc, "value", 0)
            try:
                score_value = float(v)
            except (TypeError, ValueError):
                score_value = 0.0
            md = getattr(sc, "metadata", None) or {}
            score_metadata.update(md)
            break  # single scorer; first entry wins

        # Per-dimension HMS pieces. Source: astabench.evals.discoverybench.eval_utils
        # returns metadata = {context_score, var_rel: {var: {score: {f1, ...}},
        # rel: {score, ...}}, HMS, ...}. _check_upstream_invariants() at module
        # import time warns if these key names disappear from the upstream source.
        if "context_score" in score_metadata:
            diagnostics["context_score"] = score_metadata["context_score"]
        var_rel = score_metadata.get("var_rel") or {}
        var_score = (var_rel.get("var") or {}).get("score") or {}
        if isinstance(var_score, dict):
            diagnostics["var_f1"] = var_score.get("f1")
        rel_score = (var_rel.get("rel") or {}).get("score")
        if rel_score is not None:
            diagnostics["rel_score"] = rel_score

        # Cost split: agent vs judge
        agent_cost_usd = 0.0
        judge_cost_usd = 0.0
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
                    cost = self._estimate_cost(model_name, counts)
                    if self._is_judge_model(model_name):
                        judge_cost_usd += cost
                    else:
                        agent_cost_usd += cost
        except Exception:
            pass

        # Cost cap penalty
        cost_breached = agent_cost_usd > self.cost_budget
        if cost_breached:
            score_value = score_value * COST_BREACH_PENALTY

        with self._cost_lock:
            self.total_eval_cost += agent_cost_usd + judge_cost_usd

        diagnostics["score"] = score_value
        # RoboPhD's domain layer reads `cost_usd` from diagnostics and
        # surfaces it as `eval_cost` in cost reports. Report total run
        # spend (agent + judge); evolution sees the agent-only number
        # via the cap penalty, but humans tracking $ want the total.
        diagnostics["cost_usd"] = agent_cost_usd + judge_cost_usd
        diagnostics["agent_cost_usd"] = agent_cost_usd
        diagnostics["judge_cost_usd"] = judge_cost_usd
        diagnostics["cost_breached"] = cost_breached
        diagnostics["cost_budget"] = self.cost_budget
        diagnostics["usage"] = usage_summary
        diagnostics["agent_output"] = (
            getattr(sample_log.output, "completion", "")[:1000]
            if sample_log.output else ""
        )

        return score_value, diagnostics

    # -- Cost helpers -------------------------------------------------------

    @staticmethod
    def _is_judge_model(model_name: str) -> bool:
        # Inspect surfaces model usage with provider prefix sometimes
        # ("openai/gpt-4o-2024-08-06") and sometimes without; substring
        # match handles both. JUDGE_MODEL_SHORT is the source of truth.
        return JUDGE_MODEL_SHORT.lower() in model_name.lower()

    @staticmethod
    def _estimate_cost(model_name: str, counts: dict) -> float:
        try:
            import litellm
        except ImportError:
            return 0.0
        try:
            pin, pout = litellm.cost_per_token(
                model=model_name,
                prompt_tokens=counts.get("input_tokens", 0),
                completion_tokens=counts.get("output_tokens", 0),
            )
            return (pin or 0.0) + (pout or 0.0)
        except Exception:
            return 0.0
