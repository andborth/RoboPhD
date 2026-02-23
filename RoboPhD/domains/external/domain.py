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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..base import DomainInterface, EvaluationResult, SampledProblems
from RoboPhD.adapters.gepa_codegen import extract_candidate, materialize_candidate

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

        # Required: evaluator function and dataset
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
        contexts_per_iteration = config.get("contexts_per_iteration", 50)

        if available_contexts is None:
            available_contexts = self.get_contexts()

        if len(available_contexts) <= contexts_per_iteration:
            sampled_ids = list(available_contexts)
        else:
            sampled_ids = rng.sample(available_contexts, contexts_per_iteration)

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

        Extracts a candidate dict from the agent directory, then evaluates
        each sampled example through the evaluator function.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract candidate from agent directory
        candidate = extract_candidate(agent_path, self._file_mapping)

        results = []
        correct_count = 0
        total = 0

        for context_id in sampled.contexts:
            examples = sampled.problems_by_context.get(context_id, [])
            for example in examples:
                total += 1
                try:
                    score, diagnostics = self._evaluator_fn(candidate, example)
                    is_correct = score >= 0.5
                except Exception as e:
                    self.logger.error(f"Evaluator failed on {context_id}: {e}")
                    score = 0.0
                    diagnostics = {"error": str(e)}
                    is_correct = False

                if is_correct:
                    correct_count += 1

                eid = (
                    example.get("question_id")
                    or example.get("id")
                    or context_id
                )
                results.append({
                    "question_id": eid,
                    "correct": is_correct,
                    "score": score,
                    "error": diagnostics.get("error"),
                })

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
            metadata={"fresh_count": total, "cached_count": 0},
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
    def experiment_structure_docs(self) -> str:
        return """```
../../iteration_XXX/
  agent_<AGENT_NAME>/
    evaluation.json                ← Summary metrics for all examples
    report.md                      ← Human-readable evaluation report

Agent source code (three-artifact packages):
  ../../agents/
    <agent_name>/
      agent.md              ← Agent definition
      eval_instructions.md  ← Evaluation instructions
      tools/                ← Analysis scripts (optional)
```"""

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _get_example(self, context_id: str) -> Dict:
        """Look up example by context ID."""
        problems = self.load_problems()
        examples = problems.get(context_id, [])
        return examples[0] if examples else {"id": context_id}
