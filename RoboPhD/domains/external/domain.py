"""
External evaluator domain: wraps an evaluator function as a RoboPhD domain.

This enables RoboPhD's evolution loop to consume any benchmark that provides
an evaluator function, without requiring domain-specific code in the core.

The evaluator function interface:
    evaluator(candidate: dict, example: dict) -> (score: float, diagnostics: dict)

Where:
    - candidate is a dict[str, str] of named text components (e.g., eval_instructions, tool_code)
    - example is a dict identifying the problem (e.g., {"question_id": "abc314_c"})
    - score is a float in [0, 1] (0.0 = wrong, 1.0 = perfect; intermediate values allowed)
    - diagnostics is a dict of execution trace data (ASI)
"""

import hashlib
import json
import logging
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..base import DomainInterface, EvaluationResult, SampledProblems
from RoboPhD.adapters.candidate_utils import extract_candidate, materialize_candidate
from RoboPhD.eval_utils import EvalRateLimitError, is_rate_limit_error

logger = logging.getLogger(__name__)


class ExternalEvaluatorDomain(DomainInterface):
    """
    Domain backed by an external evaluator function.

    Bridges between RoboPhD's agent-directory representation and
    the flat candidate-dict representation used by GEPA and similar frameworks.

    Config keys:
        evaluator_fn: Callable[[dict, dict], tuple[float, dict]]
        dataset: list[dict] - flat list of example dicts
        file_mapping: dict[str, str] - candidate key -> agent file path
        evolution_strategies_directory: str - strategy directory name (optional)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Required: evaluator function, dataset, and file mapping.
        # These are non-serializable — passed via runtime_config in
        # ParallelAgentResearcher, not stored in checkpoints. If resuming
        # an external-domain run, the caller must reconstruct them
        # (run_robophd.py does this; bare researcher.py --resume does not).
        for key in ("evaluator_fn", "dataset", "file_mapping"):
            if key not in config:
                raise ValueError(
                    f"ExternalEvaluatorDomain requires '{key}' in config. "
                    f"External-domain runs must be resumed via run_robophd.py "
                    f"(which reconstructs runtime_config from --task + --config), "
                    f"not via researcher.py --resume directly."
                )
        self._evaluator_fn: Callable = config["evaluator_fn"]
        self._dataset: List[Dict] = config["dataset"]
        self._file_mapping: Dict[str, str] = config["file_mapping"]

        # Optional configuration
        self._strategies_dir = config.get(
            "evolution_strategies_directory", "evolution_strategies"
        )

        # Cache for loaded problems
        self._problems_cache: Optional[Dict[str, List[Dict]]] = None

        # Track threads leaked by eval timeouts (can't kill Python threads)
        self._leaked_threads: int = 0

    # -----------------------------------------------------------------
    # Core interface
    # -----------------------------------------------------------------

    def prepare_eval_input(
        self, workspace: Path, context: str, problem: Optional[Dict] = None
    ) -> Path:
        """Write example data to workspace for agent analysis."""
        input_path = workspace / "example.json"
        example = self._get_example(context)
        with open(input_path, "w") as f:
            json.dump(example, f, indent=2)
        return input_path

    def evaluate(
        self,
        solution: str,
        problem: Dict,
        context: str,
        predictions_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Not used directly — run_evaluation handles everything."""
        raise NotImplementedError(
            "ExternalEvaluatorDomain uses run_evaluation() directly."
        )

    def load_problems(self) -> Dict[str, List[Dict]]:
        """
        Load problems as a flat map: example_id -> [example_dict].

        Uses the first available identifier field as the key.
        """
        if self._problems_cache is not None:
            return self._problems_cache

        problems = {}
        for example in self._dataset:
            # Try common ID fields (must be deterministic across processes —
            # hash() is randomized per PYTHONHASHSEED and breaks on --resume)
            for key in ("question_id", "problem_id", "id", "example_id"):
                if key in example:
                    eid = example[key]
                    break
            else:
                eid = hashlib.sha256(json.dumps(example, sort_keys=True).encode()).hexdigest()[:12]
            problems[str(eid)] = [example]

        self._problems_cache = problems
        return problems

    def get_contexts(self) -> List[str]:
        return list(self.load_problems().keys())

    def sample_problems(
        self,
        config: Dict[str, Any],
        rng: random.Random,
        available_contexts: Optional[List[str]] = None,
    ) -> SampledProblems:
        """Flat sampling: each example is its own context."""
        examples_per_iteration = config.get("examples_per_iteration", 50)

        if available_contexts is None:
            available_contexts = self.get_contexts()

        if len(available_contexts) <= examples_per_iteration:
            sampled_ids = list(available_contexts)
        else:
            sampled_ids = rng.sample(available_contexts, examples_per_iteration)

        sampled_ids = sorted(sampled_ids)
        all_problems = self.load_problems()
        problems_by_context = {
            pid: all_problems[pid] for pid in sampled_ids if pid in all_problems
        }

        return SampledProblems(
            contexts=sampled_ids,
            problems_by_context=problems_by_context,
        )

    def run_evaluation(
        self,
        sampled: SampledProblems,
        agent_path: Path,
        output_dir: Path,
        config: Dict[str, Any],
    ) -> EvaluationResult:
        """
        Run evaluation by calling the external evaluator function.

        Creates per-problem directories, supports eval result caching via
        symlinks (same pattern as CodeGenDomain), and passes problem_dir
        to the evaluator so it can write artifacts there.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract candidate from agent directory
        candidate = extract_candidate(agent_path, self._file_mapping)

        # --- Eval result cache: find prior (agent, problem) results ---
        agent_id = config.get('agent_id', '')
        cache_enabled = config.get('eval_result_cache', True)
        cached_results: Dict[str, Dict] = {}  # problem_id -> result_data

        if cache_enabled and agent_id:
            cached_map = self._find_cached_results(
                agent_id, sampled.contexts, output_dir
            )

            if cached_map:
                problems_dir = output_dir / "problems"
                problems_dir.mkdir(parents=True, exist_ok=True)

                for problem_id, source_dir in list(cached_map.items()):
                    try:
                        with open(source_dir / "result.json") as f:
                            cached_results[problem_id] = json.load(f)
                    except (json.JSONDecodeError, IOError) as e:
                        self.logger.warning(
                            f"Failed to read cached result for {problem_id}: {e}"
                        )
                        continue

                    symlink = problems_dir / problem_id
                    if not symlink.exists():
                        symlink.symlink_to(source_dir)

        cached_count = len(cached_results)
        fresh_problem_ids = [p for p in sampled.contexts if p not in cached_results]
        fresh_count = len(fresh_problem_ids)

        if cached_count > 0:
            self.logger.debug(
                f"Eval cache: {cached_count} cached, {fresh_count} fresh "
                f"(out of {len(sampled.contexts)} total)"
            )

        # --- Build results from cached entries ---
        results = []
        score_sum = 0.0

        for problem_id, result_data in cached_results.items():
            score = result_data.get("score", 0.0)
            score_sum += score
            results.append({
                "question_id": problem_id,
                "score": score,
                "cached": True,
            })

        # --- All-cached fast path ---
        if fresh_count == 0 and cached_count > 0:
            self.logger.info("All problems cached — skipping evaluator")
            total = cached_count
            average_score = (score_sum / total) if total else 0.0
            eval_data = {
                "summary": {"total_problems": total, "score_sum": score_sum, "average_score": average_score},
                "results": {r["question_id"]: r for r in results},
            }
            with open(output_dir / "evaluation.json", "w") as f:
                json.dump(eval_data, f, indent=2)
            return EvaluationResult(
                average_score=average_score, total=total, score_sum=score_sum,
                results=results,
                metadata={"fresh_count": 0, "cached_count": cached_count, "eval_cost": 0.0},
            )

        # --- Warn about leaked threads from prior timeouts ---
        if self._leaked_threads > 0:
            self.logger.warning(
                f"{self._leaked_threads} timed-out eval thread(s) still running "
                f"from prior iterations (each burns one CPU core)"
            )

        # --- Evaluate fresh problems ---
        problems_dir = output_dir / "problems"
        problems_dir.mkdir(parents=True, exist_ok=True)

        max_workers = config.get("max_workers")

        def _evaluate_one(problem_id: str) -> dict:
            try:
                examples = sampled.problems_by_context.get(problem_id, [])
                example = examples[0] if examples else {"id": problem_id}

                problem_dir = problems_dir / problem_id
                problem_dir.mkdir(parents=True, exist_ok=True)

                try:
                    score, diagnostics = self._evaluator_fn(
                        candidate, example, problem_dir=problem_dir
                    )
                except Exception as e:
                    if is_rate_limit_error(e):
                        raise
                    self.logger.error(f"Evaluator failed on {problem_id}: {e}")
                    score = 0.0
                    diagnostics = {"error.md": str(e)}

                for key in ("question_id", "problem_id", "id", "example_id"):
                    if key in example:
                        eid = str(example[key])
                        break
                else:
                    eid = problem_id

                # Extract cost from diagnostics (e.g., AIME evaluator
                # returns cost_usd from litellm.completion_cost)
                cost_usd = 0.0
                if isinstance(diagnostics, dict):
                    raw = diagnostics.get("cost_usd", 0.0)
                    try:
                        cost_usd = float(raw)
                    except (TypeError, ValueError):
                        pass

                has_error = "error.md" in diagnostics if isinstance(diagnostics, dict) else False
                result_entry = {
                    "question_id": eid,
                    "score": score,
                    "eval_cost": cost_usd,
                    "error": has_error,
                }

                # Write result.json for future caching (don't overwrite
                # a richer one the evaluator may have already written)
                result_path = problem_dir / "result.json"
                if not result_path.exists():
                    with open(result_path, "w") as f:
                        json.dump(result_entry, f, indent=2)

                # Write diagnostics as readable files for evolution.
                # String values → {key}.md (or {key} if it already has an
                # extension), skipping keys already on disk.
                if diagnostics:
                    for key, val in diagnostics.items():
                        if not isinstance(val, str) or not val:
                            continue
                        fname = key
                        out_path = problem_dir / fname
                        if not out_path.exists():
                            out_path.write_text(val)

                return result_entry

            except Exception as e:
                self.logger.error(f"Unexpected error for {problem_id}: {e}")
                return {
                    "question_id": problem_id,
                    "score": 0.0,
                }

        eval_timeout = config.get("eval_timeout", 300)
        timed_out = False
        timed_out_pids: set[str] = set()  # Don't resubmit problems that already timed out

        executor = ThreadPoolExecutor(max_workers=max_workers)
        try:
            future_to_pid = {
                executor.submit(_evaluate_one, pid): pid
                for pid in fresh_problem_ids
            }
            remaining = set(future_to_pid.keys())

            while remaining:
                done, not_done = wait(
                    remaining, timeout=eval_timeout,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    # No future completed within timeout — abandon hung
                    # futures but resubmit queued ones (unless already timed out).
                    still_remaining = set()
                    for future in not_done:
                        pid = future_to_pid[future]
                        if future.cancel():
                            if pid in timed_out_pids:
                                # Already resubmitted once — give up
                                self.logger.warning(
                                    f"EVAL TIMEOUT: {pid} never started after resubmit — scored 0"
                                )
                                result_entry = {"question_id": pid, "score": 0.0}
                                self._write_timeout_result(problems_dir, pid, result_entry, eval_timeout)
                                results.append(result_entry)
                            else:
                                # Queued (never ran) — resubmit once
                                self.logger.info(
                                    f"EVAL TIMEOUT: {pid} never started within {eval_timeout}s — resubmitting"
                                )
                                timed_out_pids.add(pid)
                                new_future = executor.submit(_evaluate_one, pid)
                                future_to_pid[new_future] = pid
                                still_remaining.add(new_future)
                        else:
                            # Running and hung — score 0, leak thread
                            self._leaked_threads += 1
                            timed_out = True
                            timed_out_pids.add(pid)
                            self.logger.warning(
                                f"EVAL TIMEOUT: {pid} exceeded {eval_timeout}s — "
                                f"scored 0, thread leaked ({self._leaked_threads} total leaked)"
                            )
                            result_entry = {"question_id": pid, "score": 0.0}
                            self._write_timeout_result(problems_dir, pid, result_entry, eval_timeout)
                            results.append(result_entry)
                    remaining = still_remaining
                    continue

                for future in done:
                    try:
                        result_entry = future.result()
                    except Exception as e:
                        if is_rate_limit_error(e):
                            for f in not_done:
                                f.cancel()
                            raise EvalRateLimitError(str(e)) from e
                        raise
                    results.append(result_entry)
                    score_sum += result_entry["score"]
                remaining = not_done
        except EvalRateLimitError:
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            if timed_out:
                # Don't block on hung threads — cancel queued futures and move on.
                # Running threads keep burning CPU until process exit (Python limitation).
                executor.shutdown(wait=False, cancel_futures=True)
            else:
                executor.shutdown(wait=True)

        total = len(results)
        average_score = (score_sum / total) if total else 0.0

        # Aggregate evaluation costs from fresh results
        total_eval_cost = sum(
            r.get("eval_cost", 0.0) for r in results if not r.get("cached")
        )

        # Write evaluation.json for compatibility
        eval_data = {
            "summary": {
                "total_problems": total,
                "score_sum": score_sum,
                "average_score": average_score,
            },
            "results": {r["question_id"]: r for r in results},
        }
        with open(output_dir / "evaluation.json", "w") as f:
            json.dump(eval_data, f, indent=2)

        return EvaluationResult(
            average_score=average_score,
            total=total,
            score_sum=score_sum,
            results=results,
            metadata={
                "fresh_count": fresh_count,
                "cached_count": cached_count,
                "eval_cost": total_eval_cost,
            },
        )

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def solution_name(self) -> str:
        return "output"

    @property
    def evolution_strategies_dir(self) -> str:
        return self._strategies_dir

    @property
    def context_label(self) -> str:
        return "Example"

    @property
    def file_mapping(self) -> Dict[str, str]:
        return self._file_mapping

    @property
    def task_description(self) -> str:
        return self.config.get("task_description", "")

    @property
    def task_objective(self) -> str:
        return self.config.get("task_objective", "")

    @property
    def task_background(self) -> str:
        return self.config.get("task_background", "")

    @property
    def task_name(self) -> str:
        return self.config.get("task_name", "external")

    @property
    def experiment_structure_docs(self) -> str:
        # Build agent file listing from file_mapping
        file_lines = []
        for key, path in sorted(self._file_mapping.items()):
            file_lines.append(f"      {path:<30s} ← {key}")
        file_listing = "\n".join(file_lines)

        # Build diagnostic file listing
        diagnostic_files = self.config.get("diagnostic_files", {})
        if diagnostic_files:
            diag_lines = []
            for fname, desc in diagnostic_files.items():
                diag_lines.append(f"        {fname:<30s} ← {desc}")
            diag_listing = "\n".join(diag_lines)
        else:
            diag_listing = "        {key}.md                 ← Diagnostics from evaluator"

        return f"""```
../../iteration_XXX/
  agent_<AGENT_NAME>/
    evaluation.json                ← Summary metrics for all examples
    problems/
      <problem_id>/               ← Per-problem directory (symlink if cached)
        result.json               ← Score and metadata for caching
{diag_listing}

Agent source code:
  ../../agents/
    <agent_name>/
{file_listing}
```"""

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _write_timeout_result(self, problems_dir: Path, pid: str, result_entry: dict, eval_timeout: int) -> None:
        """Write result.json and error diagnostic for a timed-out eval.

        Writes the same ``error`` diagnostic file that evaluator exceptions
        produce (line ~274 / ~314), so evolution AI discovers timeout errors
        the same way it discovers evaluator errors — by reading problem dirs.

        The hung thread may eventually finish and overwrite this with the real
        result — that's fine (better for future cache hits). The current
        iteration already scored 0.
        """
        try:
            problem_dir = problems_dir / pid
            problem_dir.mkdir(parents=True, exist_ok=True)
            result_path = problem_dir / "result.json"
            if not result_path.exists():
                with open(result_path, "w") as f:
                    json.dump(result_entry, f, indent=2)
            # Write diagnostic file (same pattern as evaluator exceptions)
            error_path = problem_dir / "error.md"
            if not error_path.exists():
                error_path.write_text(f"Evaluation timed out after {eval_timeout}s")
        except OSError as e:
            self.logger.debug(f"Failed to write timeout result for {pid}: {e}")

    def _find_cached_results(
        self,
        agent_id: str,
        problem_ids: List[str],
        output_dir: Path,
    ) -> Dict[str, Path]:
        """
        Find prior iteration results for (agent, problem) pairs.

        Scans iteration directories before the current one for existing results
        that can be reused via symlinks. Same pattern as CodeGenDomain.

        A result is cacheable if its result.json exists and is valid JSON.

        Returns:
            Dict mapping problem_id -> resolved Path to the cached problem directory
        """
        # output_dir = <run_dir>/iteration_XXX/agent_<agent_id>
        iteration_dir = output_dir.parent
        run_dir = iteration_dir.parent

        iter_name = iteration_dir.name
        try:
            current_iter_num = int(iter_name.split("_")[1])
        except (IndexError, ValueError):
            self.logger.warning(f"Cannot parse iteration number from {iter_name}")
            return {}

        agent_dir_name = output_dir.name

        cached = {}
        remaining = set(problem_ids)

        for prior_iter in range(current_iter_num - 1, 0, -1):
            if not remaining:
                break

            prior_problems_dir = (
                run_dir / f"iteration_{prior_iter:03d}" / agent_dir_name / "problems"
            )
            if not prior_problems_dir.is_dir():
                continue

            for problem_id in list(remaining):
                problem_dir = prior_problems_dir / problem_id
                result_file = problem_dir / "result.json"
                if not result_file.exists():
                    continue

                try:
                    with open(result_file) as f:
                        result_data = json.load(f)
                except (json.JSONDecodeError, IOError):
                    continue

                # Resolve to avoid symlink chains
                cached[problem_id] = problem_dir.resolve()
                remaining.discard(problem_id)

        return cached

    def _get_example(self, context_id: str) -> Dict:
        """Look up example by context ID."""
        problems = self.load_problems()
        examples = problems.get(context_id, [])
        return examples[0] if examples else {"id": context_id}
