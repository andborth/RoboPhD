"""
PaperFindingBench evaluator for AstaBench (Standard tools tier).

Each candidate is an Inspect `@solver` factory exported as `make_solver` in
agent.py. The evaluator runs the candidate against one PaperFindingBench
sample at a time via `inspect.eval()` with a 1-sample dataset, attaches the
Asta MCP corpus tools (or a Semantic Scholar fallback when ASTA_TOOL_KEY is
absent for dev), and reports the scorer's adjusted-F1 score along with
cost/usage diagnostics.

Bypassing inspect.eval() and constructing TaskState manually was considered
for lower per-call latency but rejected: AstaBench's submission tarball
machinery, cost accounting (record_model_usage_with_inspect), and the
scorer's get_llm_relevance for semantic queries all assume an Inspect-driven
runtime. Profile first; switch only if the loop is the bottleneck.
"""

import importlib.util
import io
import json
import logging
import os
import sys
import tempfile
import threading
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import use_tools

from astabench.evals.paper_finder.paper_finder_utils import (
    PF_INSERTED_BEFORE,
    get_inserted_before_per_dataset_type,
)
from astabench.evals.paper_finder.task import (
    pf_final_score_name_per_type,
    score_paper_finder,
    score_paper_finder_with_all_name,
)

logger = logging.getLogger(__name__)

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


def load_paper_finder(split: str = "validation") -> list[Sample]:
    """Load the PaperFindingBench split as a list of Inspect Samples.

    Thin wrapper around astabench's loader. `split` is "validation" (66) or
    "test" (267). Litqa2 splits are also accepted by the loader but are out
    of scope for this example.
    """
    from astabench.evals.paper_finder.task import load_paper_finder_bench
    ds = load_paper_finder_bench(split)
    return list(ds)


def _build_tools(sample_id: str, tool_source: str):
    """Return the list[Tool] to attach for this sample.

    `tool_source`:
      - "mcp"    : Asta MCP corpus tools (the leaderboard's Standard kit;
                   requires ASTA_TOOL_KEY).
      - "search" : Public-Semantic-Scholar paper_search + snippet_search
                   factories. Lets us smoke-test without the MCP key, but
                   scores will not match the leaderboard exactly.
    """
    inserted_before = get_inserted_before_per_dataset_type(sample_id)
    if tool_source == "mcp":
        from astabench.tools import make_asta_mcp_tools
        return make_asta_mcp_tools(insertion_date=inserted_before)
    if tool_source == "search":
        from astabench.tools.search import make_paper_search, make_snippet_search
        return [
            make_paper_search(inserted_before=inserted_before),
            make_snippet_search(inserted_before=inserted_before),
        ]
    raise ValueError(f"unknown tool_source: {tool_source!r}")


def _import_candidate_solver(agent_code: str) -> Any:
    """Materialize candidate agent.py into a temp module and return its
    @solver factory function (must be exported as `make_solver`).

    Cleans up both the temp file and the sys.modules entry after the
    factory is captured. Function.__globals__ keeps a direct reference to
    the module's __dict__ so the factory remains executable after the
    sys.modules cleanup.
    """
    mod_name = f"_paper_finder_candidate_{uuid.uuid4().hex}"
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


class PaperFinderEvaluator:
    """RoboPhD evaluator for PaperFindingBench.

    Contract: evaluate(candidate, example) -> (float_score, diagnostics_dict).

    `candidate` is a {"agent.py": "<source>"} mapping (RoboPhD's standard
    file_mapping shape).
    `example` is an Inspect Sample loaded by load_paper_finder().
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        tool_source: str | None = None,
        log_dir: str | None = None,
        subprocess_isolation: bool = True,
        subprocess_timeout: int = 900,
    ):
        self.model = model
        self.subprocess_isolation = subprocess_isolation
        self.subprocess_timeout = subprocess_timeout
        # Default to MCP if ASTA_TOOL_KEY is set, else fall back to search-only
        # for offline development. This is honest about which mode you're in
        # via the diagnostics["tool_source"] field per evaluation.
        explicit_choice = tool_source is not None
        if tool_source is None:
            tool_source = "mcp" if os.environ.get("ASTA_TOOL_KEY") else "search"
        self.tool_source = tool_source

        # Surface the silent-fallback case loudly. A user expecting
        # leaderboard-tier scores who forgets ASTA_TOOL_KEY would
        # otherwise get search-tier results without realizing it.
        if not explicit_choice and tool_source == "search":
            logger.warning(
                "ASTA_TOOL_KEY is not set; falling back to tool_source='search' "
                "(public Semantic Scholar). Scores will NOT match the AstaBench "
                "leaderboard — the Standard tier requires the Asta MCP corpus. "
                "Set ASTA_TOOL_KEY for leaderboard-comparable runs, or pass "
                "tool_source='search' explicitly to silence this warning."
            )

        # inspect.eval() insists on a log dir; use a per-evaluator temp dir
        # so multiple parallel evaluators don't fight over the same path.
        self._log_dir = log_dir or tempfile.mkdtemp(prefix="paper_finder_eval_")
        self.total_eval_cost = 0.0
        self._cost_lock = threading.Lock()

    # -- RoboPhD evaluator contract ------------------------------------------
    # RoboPhD invokes the evaluator object directly:
    #   evaluator(candidate, example, problem_dir=...) -> (score, diagnostics)

    def __call__(self, candidate: dict, example, *, problem_dir=None) -> tuple[float, dict]:
        if self.subprocess_isolation:
            return self._evaluate_via_subprocess(candidate, example)
        return self.evaluate(candidate, example, problem_dir=problem_dir)

    def _evaluate_via_subprocess(self, candidate: dict, example) -> tuple[float, dict]:
        """Run one evaluation in a fresh Python subprocess.

        Mirror of the DiscoveryBench implementation. See its docstring
        for the rationale: each subprocess has its own inspect.eval
        process-global state, so RoboPhD's parallel worker threads can
        actually overlap work instead of serializing through one lock.
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
                "tool_source": self.tool_source,
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
            cost = float(diagnostics.get("cost_usd", 0.0) or 0.0)
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

        try:
            solver_factory = _import_candidate_solver(agent_code)
        except Exception as e:
            return 0.0, {"error": f"candidate import failed: {type(e).__name__}: {e}"}

        try:
            tools = _build_tools(example.id, self.tool_source)
        except Exception as e:
            return 0.0, {
                "error": f"tool init failed: {type(e).__name__}: {e}",
                "tool_source": self.tool_source,
            }

        # Build a 1-sample Task wrapping just this example.
        # The scorer factory wraps score_paper_finder via @scorer; calling
        # it with () instantiates the Scorer that Task expects.
        scorer = score_paper_finder_with_all_name(
            pf_final_score_name_per_type["paper_finder_bench"]
        )()
        task = Task(
            dataset=MemoryDataset([example]),
            solver=solver_factory(),
            setup=[use_tools(tools)],
            scorer=scorer,
        )

        # Capture the candidate's stdout (print() lines surface as diagnostics,
        # mirroring docfinqa/protein_go's agent_stdout convention).
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
                "tool_source": self.tool_source,
                "agent_stdout": captured.getvalue(),
            }

        log = logs[0]
        return self._extract_score_and_diagnostics(log, example, captured.getvalue())

    # -- Result extraction ---------------------------------------------------

    def _extract_score_and_diagnostics(
        self, log, example: Sample, agent_stdout: str
    ) -> tuple[float, dict]:
        diagnostics: dict[str, Any] = {
            "score_type": example.metadata.get("score_type"),
            "tool_source": self.tool_source,
            "agent_stdout": agent_stdout,
            "sample_id": str(example.id),
        }

        # Expose the gold scorer criteria to evolution. The AGENT never
        # sees state.target at runtime; this only flows into post-hoc
        # diagnostics that evolution reads when deciding how to mutate.
        # For specific/metadata queries the target is the gold corpus_id
        # list; for semantic queries it's the relevance criteria dict.
        if example.target:
            diagnostics["gold_criteria.md"] = str(example.target)

        # samples is None if eval failed catastrophically
        if not getattr(log, "samples", None):
            diagnostics["error"] = "no samples in eval log"
            return 0.0, diagnostics

        sample_log = log.samples[0]

        # Sample-level errors (e.g. a tool call raised) end up here, not in
        # the outer try/except, because inspect.eval catches and records them.
        sample_err = getattr(sample_log, "error", None)
        if sample_err is not None:
            err_msg = getattr(sample_err, "message", None) or str(sample_err)
            diagnostics["error"] = err_msg[:1000]

        scores = getattr(sample_log, "scores", None) or {}
        # PaperFindingBench attaches a single scorer per sample; multiple
        # scorer entries would mean the upstream task definition changed
        # and we'd be silently picking whichever lands first.
        if len(scores) > 1:
            logger.warning(
                f"Sample {example.id} produced {len(scores)} scores; "
                f"expected 1. Using first floatable. Names: {list(scores)}"
            )
        score_value = 0.0
        for sc in scores.values():
            v = getattr(sc, "value", 0)
            try:
                score_value = float(v)
                break
            except (TypeError, ValueError):
                pass

        # Cost from inspect's model usage tracking.
        cost_usd = 0.0
        usage_summary: dict[str, Any] = {}
        try:
            stats = getattr(log, "stats", None)
            model_usage = getattr(stats, "model_usage", None) if stats else None
            if model_usage:
                for model_name, u in model_usage.items():
                    usage_summary[model_name] = {
                        "input_tokens": getattr(u, "input_tokens", 0),
                        "output_tokens": getattr(u, "output_tokens", 0),
                        "total_tokens": getattr(u, "total_tokens", 0),
                    }
                cost_usd = self._estimate_cost(usage_summary)
        except Exception:
            pass

        with self._cost_lock:
            self.total_eval_cost += cost_usd

        diagnostics["score"] = score_value
        diagnostics["cost_usd"] = cost_usd
        diagnostics["usage"] = usage_summary
        diagnostics["agent_output"] = (
            getattr(sample_log.output, "completion", "")[:1000] if sample_log.output else ""
        )

        return score_value, diagnostics

    @staticmethod
    def _estimate_cost(usage: dict) -> float:
        """Estimate $ cost from per-model token counts via litellm pricing."""
        try:
            import litellm
        except ImportError:
            return 0.0
        total = 0.0
        for model_name, u in usage.items():
            try:
                pin, pout = litellm.cost_per_token(
                    model=model_name,
                    prompt_tokens=u.get("input_tokens", 0),
                    completion_tokens=u.get("output_tokens", 0),
                )
                total += (pin or 0.0) + (pout or 0.0)
            except Exception:
                continue
        return total
