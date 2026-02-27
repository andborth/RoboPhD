"""
External evaluator domain: wraps an evaluator function as a RoboPhD domain.

This enables RoboPhD's evolution loop to consume any benchmark that provides
an evaluator function, without requiring domain-specific code in the core.

The evaluator function interface:
    evaluator(candidate: dict, example: dict) -> (score: float, diagnostics: dict)

Where:
    - candidate is a dict[str, str] of named text components (e.g., eval_instructions, tool_code)
    - example is a dict identifying the problem (e.g., {"question_id": "abc314_c"})
    - score is 0.0 or 1.0 (binary pass/fail)
    - diagnostics is a dict of execution trace data (ASI)
"""

import json
import logging
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..base import DomainInterface, EvaluationResult, SampledProblems
from RoboPhD.adapters.candidate_utils import extract_candidate, materialize_candidate

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

    # -----------------------------------------------------------------
    # Core interface
    # -----------------------------------------------------------------

    def prepare_phase1_input(
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
            # Try common ID fields
            eid = (
                example.get("question_id")
                or example.get("id")
                or example.get("example_id")
                or str(hash(json.dumps(example, sort_keys=True)))
            )
            problems[eid] = [example]

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
            self.logger.info(
                f"Eval cache: {cached_count} cached, {fresh_count} fresh "
                f"(out of {len(sampled.contexts)} total)"
            )

        # --- Build results from cached entries ---
        results = []
        correct_count = 0

        for problem_id, result_data in cached_results.items():
            is_correct = result_data.get("correct", result_data.get("score", 0) >= 0.5)
            score = result_data.get("score", 1.0 if is_correct else 0.0)
            if is_correct:
                correct_count += 1
            results.append({
                "question_id": problem_id,
                "correct": is_correct,
                "score": score,
                "error": result_data.get("error"),
                "cached": True,
            })

        # --- All-cached fast path ---
        if fresh_count == 0 and cached_count > 0:
            self.logger.info("All problems cached — skipping evaluator")
            total = cached_count
            accuracy = (correct_count / total * 100) if total else 0.0
            eval_data = {
                "summary": {"total_problems": total, "correct": correct_count, "accuracy": accuracy},
                "results": {r["question_id"]: r for r in results},
            }
            with open(output_dir / "evaluation.json", "w") as f:
                json.dump(eval_data, f, indent=2)
            return EvaluationResult(
                accuracy=accuracy, total=total, correct=correct_count,
                results=results,
                metadata={"fresh_count": 0, "cached_count": cached_count},
            )

        # --- Evaluate fresh problems ---
        problems_dir = output_dir / "problems"
        problems_dir.mkdir(parents=True, exist_ok=True)

        max_concurrent = config.get("max_concurrent", 12)

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
                    is_correct = score >= 0.5
                except Exception as e:
                    self.logger.error(f"Evaluator failed on {problem_id}: {e}")
                    score = 0.0
                    diagnostics = {"error": str(e)}
                    is_correct = False

                eid = (
                    example.get("question_id")
                    or example.get("id")
                    or problem_id
                )

                result_entry = {
                    "question_id": eid,
                    "correct": is_correct,
                    "score": score,
                    "error": diagnostics.get("error"),
                }

                # Write result.json for future caching (don't overwrite
                # a richer one the evaluator may have already written)
                result_path = problem_dir / "result.json"
                if not diagnostics.get("error") and not result_path.exists():
                    with open(result_path, "w") as f:
                        json.dump(
                            {k: v for k, v in result_entry.items() if k != "error"},
                            f, indent=2,
                        )

                # Write diagnostics as readable files for evolution.
                # String values → {key}.md (or {key} if it already has an
                # extension), skipping keys already on disk.
                if diagnostics:
                    for key, val in diagnostics.items():
                        if not isinstance(val, str) or not val:
                            continue
                        fname = key if "." in key else f"{key}.md"
                        out_path = problem_dir / fname
                        if not out_path.exists():
                            out_path.write_text(val)

                return result_entry

            except Exception as e:
                self.logger.error(f"Unexpected error for {problem_id}: {e}")
                return {
                    "question_id": problem_id,
                    "correct": False,
                    "score": 0.0,
                    "error": str(e),
                }

        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = {
                executor.submit(_evaluate_one, pid): pid
                for pid in fresh_problem_ids
            }
            for future in as_completed(futures):
                result_entry = future.result()
                results.append(result_entry)
                if result_entry["correct"]:
                    correct_count += 1

        total = cached_count + fresh_count
        accuracy = (correct_count / total * 100) if total else 0.0

        # Write evaluation.json for compatibility
        eval_data = {
            "summary": {
                "total_problems": total,
                "correct": correct_count,
                "accuracy": accuracy,
            },
            "results": {r["question_id"]: r for r in results},
        }
        with open(output_dir / "evaluation.json", "w") as f:
            json.dump(eval_data, f, indent=2)

        return EvaluationResult(
            accuracy=accuracy,
            total=total,
            correct=correct_count,
            results=results,
            metadata={"fresh_count": fresh_count, "cached_count": cached_count},
        )

    def load_agent_results(
        self, agent_dir: Path, contexts: List[str]
    ) -> Dict[str, Any]:
        """Load results from evaluation.json."""
        results = {
            "overall_accuracy": 0.0,
            "total_questions": 0,
            "correct": 0,
            "by_context": {},
        }

        eval_file = agent_dir / "evaluation.json"
        if not eval_file.exists():
            return results

        try:
            with open(eval_file) as f:
                eval_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            return results

        summary = eval_data.get("summary", {})
        results["overall_accuracy"] = summary.get("accuracy", 0.0)
        results["total_questions"] = summary.get("total_problems", 0)
        results["correct"] = summary.get("correct", 0)

        for qid, r in eval_data.get("results", {}).items():
            is_correct = r.get("correct", False)
            results["by_context"][qid] = {
                "accuracy": 100.0 if is_correct else 0.0,
                "correct": 1 if is_correct else 0,
                "total": 1,
            }

        return results

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def phase1_input_name(self) -> str:
        return "example"

    @property
    def solution_name(self) -> str:
        return "output"

    @property
    def evolution_strategies_dir(self) -> str:
        return self._strategies_dir

    @property
    def phase1_display_name(self) -> str:
        return "Analysis"

    @property
    def phase2_display_name(self) -> str:
        return "Evaluation"

    @property
    def context_label(self) -> str:
        return "Example"

    @property
    def phase1_short_label(self) -> str:
        return "Example"

    @property
    def is_hierarchical(self) -> bool:
        return False

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
    def task_name(self) -> str:
        return self.config.get("task_name", "external")

    @property
    def experiment_structure_docs(self) -> str:
        # Build agent file listing from file_mapping
        file_lines = []
        for key, path in sorted(self._file_mapping.items()):
            file_lines.append(f"      {path:<30s} ← {key}")
        file_listing = "\n".join(file_lines)

        return f"""```
../../iteration_XXX/
  agent_<AGENT_NAME>/
    evaluation.json                ← Summary metrics for all examples
    report.md                      ← Human-readable evaluation report
    problems/
      <problem_id>/               ← Per-problem directory (symlink if cached)
        result.json               ← Score and metadata for caching
        {key}.md / {key.ext}      ← Diagnostics from evaluator (problem text, model response, etc.)

Agent source code:
  ../../agents/
    <agent_name>/
{file_listing}
```"""

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

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

        A result is cacheable if its result.json exists and has no "error" key.

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

                if "error" in result_data:
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
