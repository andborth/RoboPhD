"""
Base domain interface for RoboPhD research system.

Domains differ in how problems are loaded, sampled, and evaluated.
Everything else (evolution, Elo, checkpointing, agent selection) is domain-agnostic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, TYPE_CHECKING


@dataclass
class SampledProblems:
    """
    Domain-agnostic container for sampled problems.

    This opaque container is passed from sample_problems() to run_evaluation()
    without the evolution system needing to understand its structure.

    Attributes:
        contexts: List of context identifiers (db names for Text2SQL, problem IDs for CodeGen)
        problems_by_context: Dict mapping context_id -> list of problem dicts
    """
    contexts: List[str]
    problems_by_context: Dict[str, List[Dict]]


@dataclass
class EvaluationResult:
    """
    Domain-agnostic container for evaluation results.

    Returned by run_evaluation() to provide the evolution system with
    score metrics and per-question results.

    Attributes:
        average_score: Mean score across all problems (0-1 scale).
            For binary domains this equals accuracy as a fraction.
        total: Total number of problems evaluated
        score_sum: Sum of all per-problem scores (float, not count).
            For binary domains this equals the number of correct solutions.
        results: List of per-question result dicts with at minimum:
            - question_id or problem_id
            - score: float (raw score, 0-1)
            - Additional domain-specific details
        metadata: Optional dict for domain-specific metadata (timing, costs, etc.)
    """
    average_score: float
    total: int
    score_sum: float
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


class DomainInterface(ABC):
    """
    Minimal interface for domain-specific behavior.

    Domains differ only in:
    1. What gets fed into evaluation (prepare_eval_input)
    2. How solutions are evaluated (evaluate)
    3. How problems are loaded (load_problems, get_problems_for_context)

    Everything else (evolution, Elo, checkpointing, agent selection) is domain-agnostic.
    """

    @abstractmethod
    def prepare_eval_input(self, workspace: Path, context: str, problem: Optional[Dict] = None) -> Path:
        """
        Prepare the input for evaluation in the workspace.

        This method sets up whatever the agent needs to analyze:
        - Text2SQL: Creates symlink to database.sqlite
        - CodeGen: Writes problem_context.json with {question, code_v1, approach}

        Args:
            workspace: The workspace directory for this analysis
            context: Context identifier (database name for Text2SQL, problem_id for CodeGen)
            problem: Optional problem dict (used by CodeGen to include code_v1, approach)

        Returns:
            Path to the primary input file/directory the agent will analyze
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        solution: str,
        problem: Dict,
        context: str,
        predictions_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a solution against ground truth.

        Args:
            solution: The generated solution (SQL query or code)
            problem: Problem dict with ground truth and metadata
            context: Context identifier (database name or problem_id)
            predictions_path: Optional path to save predictions

        Returns:
            Evaluation result dict with at minimum:
            - correct: bool - whether the solution is correct
            - details: Any - domain-specific details
        """
        pass

    @abstractmethod
    def load_problems(self) -> Dict[str, List[Dict]]:
        """
        Load all problems grouped by context.

        Returns:
            Dict mapping context_id -> list of problem dicts

        For Text2SQL: Maps database_name -> list of questions for that database
        For CodeGen: Maps problem_id -> list containing single problem dict
        """
        pass

    @abstractmethod
    def get_contexts(self) -> List[str]:
        """
        Get list of all available contexts.

        Returns:
            List of context identifiers (database names for Text2SQL,
            problem categories or IDs for CodeGen)
        """
        pass

    @abstractmethod
    def sample_problems(
        self,
        config: Dict[str, Any],
        rng: 'random.Random',
        available_contexts: Optional[List[str]] = None
    ) -> SampledProblems:
        """
        Sample problems for an iteration based on domain-specific config.

        Sampling logic differs by domain:
        - Text2SQL: Uses examples_per_iteration + problems_per_context
          (two-level hierarchy: sample databases, then questions within each)
        - CodeGen: Uses examples_per_iteration only
          (flat sampling from problem pool, each context IS a problem)

        Args:
            config: Configuration dict containing sampling parameters
            rng: Random number generator for reproducibility
            available_contexts: Optional list of contexts to sample from.
                              If None, uses all available contexts.

        Returns:
            SampledProblems containing:
            - contexts: List of context identifiers selected
            - problems_by_context: Dict mapping context_id -> list of problem dicts
        """
        pass

    @abstractmethod
    def run_evaluation(
        self,
        sampled: SampledProblems,
        agent_path: Path,
        output_dir: Path,
        config: Dict[str, Any]
    ) -> EvaluationResult:
        """
        Run evaluation on sampled problems with given agent.

        This method orchestrates the full evaluation pipeline for the domain:
        - Text2SQL: DB analysis, SQL generation, result comparison
        - CodeGen: Critic feedback, coder revision, test execution

        Args:
            sampled: SampledProblems from sample_problems()
            agent_path: Path to agent directory containing agent.md, eval_instructions.md, tools/
            output_dir: Directory to write evaluation outputs
            config: Configuration dict with eval_model, timeouts, etc.

        Returns:
            EvaluationResult with accuracy and per-question results
        """
        pass

    @property
    @abstractmethod
    def solution_name(self) -> str:
        """
        Human-readable name for what gets generated.

        Used in logging and prompts:
        - Text2SQL: "SQL"
        - CodeGen: "code"
        """
        pass

    @property
    @abstractmethod
    def evolution_strategies_dir(self) -> str:
        """
        Directory name for evolution strategies.

        Returns the directory name (not full path) relative to RoboPhD/.
        All domains use "evolution_strategies".

        The full path is constructed as: RoboPhD/{evolution_strategies_dir}/
        """
        pass

    @property
    def name(self) -> str:
        """Domain name for identification."""
        return self.__class__.__name__.replace('Domain', '').lower()

    @property
    @abstractmethod
    def context_label(self) -> str:
        """
        Human-readable label for contexts (singular).

        Used in reports and logs:
        - Text2SQL: "Database"
        - CodeGen: "Problem"
        """
        pass

    @property
    def context_label_plural(self) -> str:
        """
        Plural form of context_label.

        Used in reports and logs:
        - Text2SQL: "databases"
        - CodeGen: "problems"
        """
        return self.context_label.lower() + "s"

    @property
    @abstractmethod
    def experiment_structure_docs(self) -> str:
        """
        Return documentation of experiment directory structure for evolution prompts.

        This describes where agents can find evaluation results, outputs, and source code.
        Each domain has its own structure. The string should use 'iteration_XXX' as a
        placeholder which will be replaced with the actual iteration number.
        """
        pass

    # === Internal workspace methods ===
    # These are implementation details used by some domains internally.
    # Not part of the public interface - use run_evaluation() instead.

    def setup_context_workspace(
        self,
        workspace: Path,
        context_name: str,
        agent_path: Path,
        problem: Optional[Dict] = None
    ) -> None:
        """
        Set up workspace for evaluating a single context.

        Creates the standard workspace structure:
        - .claude/agents/agent.md (from agent_path)
        - eval_instructions.md (from agent_path)
        - tools/ (from agent_path, if exists)
        - Context-specific input (database symlink or problem_context.json)
        - output/, tool_output/, results/ directories

        Args:
            workspace: Directory for this context's evaluation
            context_name: Context identifier (database name or problem_id)
            agent_path: Path to agent directory with agent.md, eval_instructions.md, tools/
            problem: Optional problem dict (used by CodeGen for code_v1, approach)

        Note: This is an internal method used by some domains' run_evaluation()
        implementations. External callers should use run_evaluation() directly.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement setup_context_workspace(). "
            "Use run_evaluation() instead."
        )

    def run_evaluation_in_workspace(
        self,
        workspace: Path,
        problems: List[Dict],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run evaluation in an existing workspace.

        The workspace must already be set up via setup_context_workspace().
        This runs the full evaluation pipeline for a single context.

        Args:
            workspace: Workspace directory (already set up)
            problems: List of problem dicts for this context
            config: Configuration dict with:
                - eval_model: Model for generation
                - timeout: Timeout for operations
                - Additional domain-specific options

        Returns:
            Dict with:
            - accuracy: Accuracy percentage (0-100)
            - correct: Number of correct solutions
            - total: Total number of problems
            - error: Optional error message if evaluation failed
            - eval_cost: Optional API cost (if tracked)

        Note: This is an internal method used by some domains' run_evaluation()
        implementations. External callers should use run_evaluation() directly.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement run_evaluation_in_workspace(). "
            "Use run_evaluation() instead."
        )
