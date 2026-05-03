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
from pathlib import Path
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

# Metadata fields that should reach the agent at runtime. Anything else
# (split / domain / question_type / difficulty added by load_synth) is
# evaluator-side bookkeeping and would let the agent build a runtime
# synth-vs-real branch detector — which silently overfits to the
# training mixture instead of learning the underlying task. We strip
# them from a *copy* of the example before constructing the Task; the
# originals stay on `example.metadata` for diagnostics extraction.
#
# This allowlist must stay aligned with AstaBench's own json_to_sample
# for DiscoveryBench:
#   astabench/evals/discoverybench/task.py:json_to_sample
# That function constructs Sample with metadata={"query", "metadata"}.
# If a future astabench version adds new fields to real samples (e.g. a
# "context" field), update this allowlist to include them — otherwise
# the agent will be silently denied access to fields a leaderboard
# agent would receive. Allowlist (not denylist) chosen because new
# fields default to hidden, which is the safer failure mode for
# leakage but the noisier one for missed inputs.
AGENT_VISIBLE_METADATA_KEYS = frozenset({"query", "metadata"})

# Inspect-AI's `inspect.eval()` raises if two evaluations are in flight
# in the same Python process. To get real parallelism across RoboPhD's
# worker threads we route each call through a subprocess (see
# `_eval_worker.py`). Each subprocess has its own process-global state,
# so workers don't fight over the eval_async singleton.
#
# `subprocess_isolation=False` falls back to in-process execution. That
# path is correct for single-threaded callers (smoke tests, the
# subprocess worker calling itself) but will deadlock/error if multiple
# threads enter it concurrently.

# Flag to ensure the "explanation extraction silently empty" debug log
# fires at most once per process. Per-evaluation logs would spam.
_explanation_drift_logged = False


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
       `var_rel` containing `var.score.f1` and `rel.score`. The judge prose lives
       in inner-dict keys: var.score has "explanation" and "f1"/"intersection"/etc.,
       rel has "answer" (a JSON-stringified judge response with "answer"/"explanation"
       fields). _extract_score_and_diagnostics indexes those paths to surface
       per-dimension HMS and judge prose; if upstream restructures the dict, those
       fields silently become None and judge_explanation.md goes empty.
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
        # Both top-level keys (context_score, var_rel) and the inner keys
        # we extract from var.score (f1, explanation) and rel (answer)
        # appear as quoted string literals in upstream eval_utils source.
        # Quotes prevent false positives from generic English usage of
        # words like "explanation" in comments.
        for marker in ('context_score', 'var_rel', '"f1"', '"explanation"', '"answer"'):
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
        subprocess_isolation: bool = True,
        eval_timeout: int = 600,
        apply_cost_penalty: bool = True,
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
        self.subprocess_isolation = subprocess_isolation
        # Whether to multiply score by 0.9 when agent_cost_usd > cost_budget.
        # True for training (RoboPhD's ELO competition) — the soft penalty
        # nudges evolution toward cheaper agents. False for test paths
        # (eval_candidate / eval_run / --eval-test-set / --eval-only /
        # --eval-agent) — test scores are raw HMS so the agent lands at
        # its true point on the Pareto cost-vs-score curve. cost_breached
        # is recorded in diagnostics in both modes for the audit trail.
        self.apply_cost_penalty = apply_cost_penalty

        # Subprocess kill-after timeout MUST be less than RoboPhD's
        # eval_timeout. RoboPhD's reaper writes "EVAL TIMEOUT" and scores
        # 0, but cannot interrupt Python threads — the thread keeps
        # blocking on subprocess.run until our subprocess_timeout fires.
        # If subprocess_timeout > eval_timeout, the gap leaves Docker and
        # judge-LLM resources tied up after RoboPhD gave up, with each
        # leaked thread burning a CPU core (see the warning that fires at
        # the start of the next iteration). Buffer of 30s gives Python
        # time to SIGKILL the subprocess and the thread to return cleanly.
        self.eval_timeout = eval_timeout
        self.subprocess_timeout = max(eval_timeout - 30, 60)

        self._log_dir = log_dir or tempfile.mkdtemp(prefix="discoverybench_eval_")
        self.total_eval_cost = 0.0
        self._cost_lock = threading.Lock()

    # -- RoboPhD evaluator contract -----------------------------------------
    # RoboPhD invokes the evaluator object directly:
    #   evaluator(candidate, example, problem_dir=...) -> (score, diagnostics)

    def __call__(self, candidate: dict, example, *, problem_dir=None) -> tuple[float, dict]:
        if self.subprocess_isolation:
            return self._evaluate_via_subprocess(candidate, example)
        return self.evaluate(candidate, example, problem_dir=problem_dir)

    def _evaluate_via_subprocess(self, candidate: dict, example) -> tuple[float, dict]:
        """Run one evaluation in a fresh Python subprocess.

        Each subprocess imports inspect-ai/astabench (~7s overhead) but
        has its own process-global state, so concurrent calls from
        RoboPhD's worker threads don't collide on inspect.eval_async's
        singleton check. This is what lets --max-workers > 1 actually
        deliver parallelism.
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
            # `default=str` is a defensive belt-and-braces: candidates SHOULD
            # be `{"agent.py": "<source>"}` and examples should round-trip
            # through Sample.model_dump() cleanly. If a future caller passes
            # objects that aren't JSON-native (Path, datetime, custom class),
            # default=str silently stringifies them so the worker can still
            # parse the JSON. This is a known fidelity loss; if it ever
            # bites a debugging session, the right fix is to enforce
            # JSON-serializable types at the boundary, not to remove
            # default=str.
            json.dump({
                "candidate": candidate,
                "example": example_dict,
                "model": self.model,
                "cost_budget": self.cost_budget,
                "apply_cost_penalty": self.apply_cost_penalty,
            }, inf, default=str)
            inf_path = inf.name
        out_path = inf_path + ".out"

        # Note: `subprocess_stderr` in the diagnostic dict is truncated to the
        # last 2000 chars to avoid bloating context for downstream evolution.
        # For deep-traceback debugging during incident response, set the
        # SUBPROCESS_STDERR_FULL env var (read by no code yet — placeholder
        # for the convention) or read raw stderr by re-running with the
        # in-process path (subprocess_isolation=False).
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
            # Aggregate cost on the parent side too so total_eval_cost is correct.
            cost = (diagnostics.get("agent_cost_usd", 0.0) or 0.0) + (
                diagnostics.get("judge_cost_usd", 0.0) or 0.0
            )
            with self._cost_lock:
                self.total_eval_cost += cost
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

        # Build the agent-visible Sample with only the metadata fields a real
        # AstaBench DiscoveryBench query would carry. The evaluator's
        # diagnostics extraction below still reads from the *original*
        # example.metadata, so split/domain/etc. remain visible to
        # evolution post-hoc — they just stop flowing through
        # TaskState.metadata to the runtime solver. (See
        # AGENT_VISIBLE_METADATA_KEYS for rationale.)
        agent_visible_metadata = {
            k: v for k, v in (example.metadata or {}).items()
            if k in AGENT_VISIBLE_METADATA_KEYS
        }
        example_for_solver = example.model_copy(update={"metadata": agent_visible_metadata})

        try:
            solver_factory = _import_candidate_solver(agent_code)
        except Exception as e:
            return 0.0, {"error": f"candidate import failed: {type(e).__name__}: {e}"}

        task = Task(
            dataset=MemoryDataset([example_for_solver]),
            solver=solver_factory(),
            setup=[use_tools(python_session())],
            scorer=score_discoverybench(),
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
            "split": example.metadata.get("split", "real"),
        }

        # Save the agent-visible inputs (query + datasets metadata) into
        # the problem dir so a reader of any per-problem dir can see
        # exactly what the agent was asked. Mirrors the convention of
        # other examples where the question/document inputs are written
        # alongside the agent's output.
        query = example.metadata.get("query", "")
        if query:
            diagnostics["query.md"] = str(query)
        inner_metadata = example.metadata.get("metadata")
        if inner_metadata:
            diagnostics["metadata.md"] = (
                "```json\n" + json.dumps(inner_metadata, indent=2, default=str) + "\n```"
            )

        # Expose the gold hypothesis + workflow to evolution. The AGENT
        # never sees state.target at runtime; this only flows into the
        # post-hoc diagnostics that evolution reads when deciding how to
        # mutate the next candidate. With this, evolution can compare
        # what the agent emitted (agent_output) against what was
        # expected (gold_hypothesis) and learn from specific failures.
        target = example.target
        if isinstance(target, list) and len(target) >= 1:
            diagnostics["gold_hypothesis.md"] = str(target[0])
            if len(target) >= 2 and target[1]:
                diagnostics["gold_workflow.md"] = str(target[1])

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
        rel_obj = var_rel.get("rel") or {}
        rel_score = rel_obj.get("score")
        if rel_score is not None:
            diagnostics["rel_score"] = rel_score

        # Expose the judge's per-dimension explanations. These are the
        # most directly actionable signal evolution can get: the judge
        # literally tells us which variables matched/missed and whether
        # the relationship form was right. Surfaced as a single
        # `judge_explanation.md` so RoboPhD writes one readable file.
        explanation_lines = []
        var_explanation = var_score.get("explanation") if isinstance(var_score, dict) else None
        if var_explanation:
            explanation_lines.append(
                f"## Variables (f1={var_score.get('f1')}, "
                f"intersection={var_score.get('intersection')}, "
                f"sizeA={var_score.get('sizeA')}, sizeB={var_score.get('sizeB')})\n\n"
                + str(var_explanation)
            )
        # rel.answer is the LLM judge's raw response — JSON-stringified
        # `{"answer": "A) very similar", "explanation": "..."}`. Try to
        # parse the explanation field out of it.
        rel_answer = rel_obj.get("answer")
        if rel_answer:
            try:
                parsed = json.loads(rel_answer)
                rel_expl = parsed.get("explanation", "")
                rel_verdict = parsed.get("answer", "")
            except (json.JSONDecodeError, AttributeError, TypeError) as e:
                # Drift signal: the upstream judge's response is no longer
                # JSON-shaped (e.g., switched to plain text). We fall back
                # to raw text so judge_explanation.md still has content,
                # but log so a future code review can spot the change.
                logger.debug(
                    "rel_answer not JSON-parseable (%s); falling back to raw "
                    "text. Verdict will be empty in judge_explanation.md.",
                    type(e).__name__,
                )
                rel_expl = str(rel_answer)
                rel_verdict = ""
            explanation_lines.append(
                f"## Relationship (score={rel_score}, verdict={rel_verdict!r})\n\n"
                + rel_expl
            )
        # Drift signal #2: var_rel was populated (so the scoring path ran)
        # but neither var nor rel produced a usable explanation. That
        # almost certainly means the inner-dict key names ("explanation",
        # "answer") changed upstream. Log once per process so it's
        # visible in run logs without spamming per-evaluation.
        global _explanation_drift_logged
        if var_rel and not explanation_lines and not _explanation_drift_logged:
            logger.debug(
                "var_rel populated but no var.score.explanation or "
                "rel.answer found — judge metadata key names may have "
                "changed upstream. _check_upstream_invariants() should "
                "have warned at import time."
            )
            _explanation_drift_logged = True
        if explanation_lines:
            diagnostics["judge_explanation.md"] = "\n\n".join(explanation_lines)

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

        # Cost cap penalty. Applied during training (apply_cost_penalty=True)
        # to nudge evolution toward cheaper agents; suppressed at test time
        # so reported scores are raw HMS for the Pareto curve.
        cost_breached = agent_cost_usd > self.cost_budget
        cost_penalty_applied = cost_breached and self.apply_cost_penalty
        if cost_penalty_applied:
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
        diagnostics["cost_penalty_applied"] = cost_penalty_applied
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
