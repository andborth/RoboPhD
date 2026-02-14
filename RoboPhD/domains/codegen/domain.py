"""
Code Generation domain implementation.

Implements the coder/critic paradigm for LiveCodeBench.
See docs/code_generation_critic/robophd_code_generation.md for design.
"""

import json
import logging
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import DomainInterface, SampledProblems, EvaluationResult


class CodeGenDomain(DomainInterface):
    """
    Code Generation domain with evolved critics.

    Phase 1 Input: Bundle {question, code_v1, approach_description}
    Phase 1 Output: Approach-specific critic feedback
    Phase 2 Output: Revised code (Code v2) via session resumption
    Evaluation: Binary pass/fail on hidden tests

    Key architectural difference from Text2SQL:
    - Text2SQL Phase 2: Fresh API call with generated system prompt
    - CodeGen Phase 2: Resume coder's session with critic feedback

    The session_id is part of the cached problem data (alongside code_v1
    and approach_description). This enables the coder to evaluate critic
    feedback against its original reasoning, preserved in session context.

    Cached problem data (from preprocessing):
    - question: Problem statement from LiveCodeBench
    - code_v1: Initial solution from Coder Call 1
    - approach_description: Self-reported approach from Coder Call 1.5
    - session_id: Claude Code session ID for resumption in Call 2

    NOT YET IMPLEMENTED - This is a stub for the domain abstraction.
    """

    # Date cutoff for test set (problems on or after this date are test set)
    TEST_SET_CUTOFF = "2024-11-01"

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Code Generation domain.

        Args:
            config: Configuration dictionary with:
                - coder_model: Model used for coder (determines cache path). Default: 'haiku-4.5'
                - cache_dir: Override cache directory (default: codegen_cache/{coder_model}_v6)
                - codegen_split: "evolution" (before cutoff) or "test" (after cutoff)
                - test_set_cutoff: Override cutoff date (default: '2024-11-01')
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Configure cache directory
        coder_model = config.get('coder_model', 'haiku-4.5')
        if 'cache_dir' in config:
            self.cache_dir = Path(config['cache_dir'])
        else:
            self.cache_dir = Path(f"codegen_cache/{coder_model}_v6")

        # Date filtering options
        self.codegen_split = config.get('codegen_split', 'evolution')
        if self.codegen_split not in ('evolution', 'test'):
            raise ValueError(f"codegen_split must be 'evolution' or 'test', got: {self.codegen_split}")
        self.test_set_cutoff = config.get('test_set_cutoff', self.TEST_SET_CUTOFF)

        # Cache for loaded problems
        self._problems_cache: Optional[Dict[str, List[Dict]]] = None

    def prepare_phase1_input(self, workspace: Path, context: str, problem: Optional[Dict] = None) -> Path:
        """
        Prepare problem context for Phase 1 (critic) analysis.

        NOTE: CodeGen's run_evaluation() handles everything via subprocess.
        This method is only called if the orchestrator path is used.
        Returns a placeholder path for interface compatibility.

        For the actual critic workflow, run_critic_evaluation.py loads problems
        from cache and handles all phases internally.

        Args:
            workspace: Workspace directory
            context: Problem identifier
            problem: Problem dict (unused in direct evaluation mode)

        Returns:
            Path to expected problem_context.json (may not exist)
        """
        # No-op for CodeGen - run_evaluation() handles everything via subprocess
        # The subprocess (run_critic_evaluation.py) loads problems from cache
        # and generates v1 code, critic feedback, and revisions internally
        return workspace / "problem_context.json"

    def evaluate(
        self,
        solution: str,
        problem: Dict,
        context: str,
        predictions_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Evaluate code solution against hidden tests.

        Args:
            solution: Generated code (Code v2)
            problem: Problem dict with test cases
            context: Problem identifier
            predictions_path: Optional path to save predictions

        Returns:
            Evaluation result with 'correct' (all tests pass) and details
        """
        # Single-problem evaluation is handled by run_critic_evaluation.py
        # This method is not used directly - use run_evaluation() instead
        raise NotImplementedError(
            "Single-problem evaluation not implemented. "
            "Use run_evaluation() with SampledProblems instead."
        )

    def load_problems(self) -> Dict[str, List[Dict]]:
        """
        Load problems from codegen_cache directory.

        Iterates problem directories, loading those with at least problem.md and meta.json.
        Applies date filtering based on test_set_only/evolution_set_only config.

        Returns:
            Dict mapping problem_id -> [problem_dict]
            Each problem_dict contains:
            - question_id: Problem identifier
            - _cache_dir: Path to cache directory (for downstream lazy generation)
            - difficulty: Problem difficulty
            - contest_date: Date of contest
        """
        if self._problems_cache is not None:
            return self._problems_cache

        problems: Dict[str, List[Dict]] = {}

        if not self.cache_dir.exists():
            self.logger.warning(f"Cache directory does not exist: {self.cache_dir}")
            self._problems_cache = problems
            return problems

        for problem_dir in sorted(self.cache_dir.iterdir()):
            if not problem_dir.is_dir():
                continue

            # Require problem.md and meta.json
            problem_md = problem_dir / "problem.md"
            meta_path = problem_dir / "meta.json"

            if not problem_md.exists() or not meta_path.exists():
                continue

            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                self.logger.warning(f"Failed to load meta.json for {problem_dir.name}: {e}")
                continue

            # Extract fields
            question_id = meta.get('question_id', problem_dir.name)
            contest_date = meta.get('contest_date', '')

            # Apply date filtering based on codegen_split
            if self.codegen_split == "test" and contest_date < self.test_set_cutoff:
                continue
            if self.codegen_split == "evolution" and contest_date >= self.test_set_cutoff:
                continue

            # Build problem dict
            problem_dict = {
                'question_id': question_id,
                '_cache_dir': str(problem_dir),
                'difficulty': meta.get('difficulty', ''),
                'contest_date': contest_date,
                'question_title': meta.get('question_title', ''),
            }

            # CodeGen has 1:1 mapping of context to problem
            problems[question_id] = [problem_dict]

        self._problems_cache = problems
        return problems

    def get_contexts(self) -> List[str]:
        """
        Get list of problem identifiers.

        Returns:
            List of problem IDs from loaded problems
        """
        return list(self.load_problems().keys())

    @property
    def phase1_input_name(self) -> str:
        """Human-readable name for Phase 1 input."""
        return "problem context"

    @property
    def solution_name(self) -> str:
        """Human-readable name for generated solutions."""
        return "code"

    @property
    def evolution_strategies_dir(self) -> str:
        """Directory name for CodeGen evolution strategies."""
        return "evolution_strategies_codegen"

    @property
    def phase1_display_name(self) -> str:
        """Display name for Phase 1: Problem Analysis."""
        return "Problem Analysis"

    @property
    def phase2_display_name(self) -> str:
        """Display name for Phase 2: Code Generation."""
        return "Code Generation"

    @property
    def error_analysis_tool_dir(self) -> str:
        """Directory path for CodeGen-specific error analysis tools."""
        return "domains/codegen/tools"

    @property
    def context_label(self) -> str:
        """Human-readable label for contexts."""
        return "Problem"

    @property
    def phase1_short_label(self) -> str:
        """Short label for Phase 1 context."""
        return "Problem"

    @property
    def is_hierarchical(self) -> bool:
        """CodeGen is flat: each problem is its own context (1:1 mapping)."""
        return False

    @property
    def experiment_structure_docs(self) -> str:
        """Return CodeGen experiment directory structure documentation."""
        return '''```
../../iteration_XXX/
  agent_<AGENT_NAME>/
    evaluation.json                ← Summary metrics for all problems
    report.md                      ← Human-readable evaluation report
    problems/
      <problem_id>/
        solution.py                ← Original code (v1)
        reflection.md              ← Coder's self-reflection on their approach
        tool_output/
          critic_feedback.txt      ← Tool-generated analysis
        critic_prompt.md           ← Full prompt sent to critic (includes eval_instructions)
        feedback.md                ← Critic's verdict (CORRECT/INCORRECT) and feedback
        solution_v2.py             ← Revised code (or symlink to v1 if verdict=CORRECT)
        acceptance.md              ← Coder's explanation of changes (if revised)
        result.json                ← Per-problem evaluation result

Agent source code (three-artifact packages):
  ../../agents/
    <agent_name>/
      agent.md              ← Problem analysis agent definition
      eval_instructions.md  ← Code generation instructions
      tools/                ← Analysis scripts (optional)
```'''

    def load_agent_results(self, agent_dir: Path, contexts: List[str]) -> Dict[str, Any]:
        """
        Load evaluation results from an agent's output directory.

        For CodeGen, results are stored in:
        agent_dir/evaluation.json (flat structure)

        Args:
            agent_dir: Path to agent's output directory
            contexts: List of problem IDs to load results for

        Returns:
            Dict with overall_accuracy, total_questions, correct, and by_context
        """
        results = {
            'overall_accuracy': 0.0,
            'total_questions': 0,
            'correct': 0,
            'by_context': {},
        }

        eval_file = agent_dir / "evaluation.json"
        if not eval_file.exists():
            self.logger.warning(f"evaluation.json not found at {eval_file}")
            return results

        try:
            with open(eval_file) as f:
                eval_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            self.logger.warning(f"Failed to load {eval_file}: {e}")
            return results

        # Extract summary data
        summary = eval_data.get('summary', {})
        results['overall_accuracy'] = summary.get('v2_pass_rate', 0.0)
        results['total_questions'] = summary.get('total_problems', len(contexts))
        results['correct'] = summary.get('v2_passed', 0)

        # Extract per-problem results
        results_raw = eval_data.get('results', {})
        results_iter = results_raw.values() if isinstance(results_raw, dict) else results_raw

        for r in results_iter:
            problem_id = r.get('question_id')
            if problem_id:
                # Use standard 'correct' field (normalized by run_evaluation)
                # Fall back to v2_passed for backward compatibility
                is_correct = r.get('correct', r.get('v2_passed', False))
                results['by_context'][problem_id] = {
                    'accuracy': 100.0 if is_correct else 0.0,
                    'correct': 1 if is_correct else 0,
                    'total': 1,
                    'v1_passed': r.get('v1_passed', False),
                    'v2_passed': r.get('v2_passed', False),
                    'improved': r.get('improved', False),
                    'regressed': r.get('regressed', False),
                }

        return results

    # Note: setup_context_workspace and run_evaluation_in_workspace are not implemented
    # for CodeGen. The domain uses run_evaluation() directly for all evaluation.
    # These methods have default NotImplementedError in base.py.

    def sample_problems(
        self,
        config: Dict[str, Any],
        rng: random.Random,
        available_contexts: Optional[List[str]] = None
    ) -> SampledProblems:
        """
        Sample problems for an iteration.

        CodeGen uses flat sampling: contexts_per_iteration problems
        from the problem pool. Each problem is its own context (1:1 mapping).

        Args:
            config: Configuration with contexts_per_iteration
            rng: Random number generator for reproducibility
            available_contexts: Optional list of problem IDs to sample from

        Returns:
            SampledProblems with sampled problem IDs and problem dicts
        """
        contexts_per_iteration = config.get("contexts_per_iteration", 50)

        # Get available problem IDs
        if available_contexts is None:
            available_contexts = self.get_contexts()

        # Sample problem IDs
        if len(available_contexts) <= contexts_per_iteration:
            sampled_ids = list(available_contexts)
        else:
            sampled_ids = rng.sample(available_contexts, contexts_per_iteration)

        # Sort for reproducibility
        sampled_ids = sorted(sampled_ids)

        # Build problems_by_context (1:1 mapping for CodeGen)
        all_problems = self.load_problems()
        problems_by_context: Dict[str, List[Dict]] = {}

        for problem_id in sampled_ids:
            if problem_id in all_problems:
                problems_by_context[problem_id] = all_problems[problem_id]

        return SampledProblems(
            contexts=sampled_ids,
            problems_by_context=problems_by_context
        )

    def run_evaluation(
        self,
        sampled: SampledProblems,
        agent_path: Path,
        output_dir: Path,
        config: Dict[str, Any]
    ) -> EvaluationResult:
        """
        Run evaluation on sampled problems with given critic agent.

        Calls run_critic_evaluation.py as subprocess with:
        - --problem-ids: Comma-separated list of sampled problem IDs
        - --critic-agent: Path to agent directory
        - --output-dir: Output directory for results
        - --cache-dir: Cache directory for problem data

        Args:
            sampled: SampledProblems with problem IDs
            agent_path: Path to critic agent directory (agent.md, eval_instructions.md, tools/)
            output_dir: Directory to write evaluation outputs
            config: Configuration with coder_model, critic_model, timeouts, etc.

        Returns:
            EvaluationResult with v2 accuracy and per-problem results
        """
        # Extract config (dynamic defaults resolved by ConfigManager - fail fast if missing)
        coder_model = config['coder_model']
        critic_model = config['critic_model']
        timeout = config['phase2_timeout']
        max_concurrent = config['max_concurrent']

        # Build problem IDs list
        problem_ids = ','.join(sampled.contexts)

        # Find run_critic_evaluation.py relative to RoboPhD package
        # This file is at RoboPhD/domains/codegen/domain.py
        # Script is at RoboPhD/tools/run_critic_evaluation.py
        import RoboPhD
        robophd_root = Path(RoboPhD.__file__).parent
        script_path = robophd_root / "tools" / "run_critic_evaluation.py"

        if not script_path.exists():
            raise FileNotFoundError(
                f"run_critic_evaluation.py not found at {script_path}. "
                f"Expected location relative to RoboPhD package: RoboPhD/tools/run_critic_evaluation.py"
            )

        # Build command
        cmd = [
            sys.executable,
            str(script_path),
            "--coder-model", coder_model,
            "--critic-model", critic_model,
            "--critic-agent", str(agent_path),
            "--output-dir", str(output_dir),
            "--cache-dir", str(self.cache_dir),
            "--problem-ids", problem_ids,
            "--timeout", str(timeout),
            "--max-concurrent", str(max_concurrent),
        ]

        # Add test-set or evolution-set flags based on codegen_split
        if self.codegen_split == "test":
            cmd.append("--test-set")
        else:
            cmd.append("--evolution-set")

        self.logger.info(f"Running critic evaluation on {len(sampled.contexts)} problems")
        self.logger.debug(f"Command: {' '.join(cmd)}")

        # Calculate subprocess timeout based on expected runtime
        # - Each problem may have up to 4 Claude calls (critic, revision, acceptance, regen)
        # - Problems run concurrently (max_concurrent workers)
        # - 1.5x safety margin + 10 min buffer for startup/teardown
        # This timeout guards against catastrophic hangs; normal operation completes faster
        calls_per_problem = 4
        estimated_runtime = (len(sampled.contexts) / max_concurrent) * calls_per_problem * timeout
        subprocess_timeout = int(estimated_runtime * 1.5) + 600

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=subprocess_timeout
            )

            if result.returncode != 0:
                self.logger.error(f"run_critic_evaluation.py failed with code {result.returncode}")
                if result.stderr:
                    self.logger.error(f"stderr: {result.stderr[:1000]}")
                # Return empty result on failure
                return EvaluationResult(
                    accuracy=0.0,
                    total=len(sampled.contexts),
                    correct=0,
                    results=[],
                    metadata={'error': f'Subprocess failed with code {result.returncode}'}
                )

        except subprocess.TimeoutExpired:
            self.logger.error("run_critic_evaluation.py timed out")
            return EvaluationResult(
                accuracy=0.0,
                total=len(sampled.contexts),
                correct=0,
                results=[],
                metadata={'error': 'Subprocess timeout'}
            )

        # Parse evaluation.json from output directory
        eval_file = Path(output_dir) / "evaluation.json"
        if not eval_file.exists():
            self.logger.error(f"evaluation.json not found at {eval_file}")
            return EvaluationResult(
                accuracy=0.0,
                total=len(sampled.contexts),
                correct=0,
                results=[],
                metadata={'error': 'evaluation.json not found'}
            )

        try:
            with open(eval_file) as f:
                eval_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            self.logger.error(f"Failed to parse evaluation.json: {e}")
            return EvaluationResult(
                accuracy=0.0,
                total=len(sampled.contexts),
                correct=0,
                results=[],
                metadata={'error': f'Failed to parse evaluation.json: {e}'}
            )

        # Extract results
        summary = eval_data.get('summary', {})
        results_raw = eval_data.get('results', {})

        # Handle both dict (standard) and list (legacy) formats
        # Dict: iterate over values; List: iterate directly
        results_iter = results_raw.values() if isinstance(results_raw, dict) else results_raw
        results_count = len(results_raw)

        # Use v2 pass rate as accuracy (critic's impact)
        v2_pass_rate = summary.get('v2_pass_rate', 0.0)
        v2_passed = summary.get('v2_passed', 0)
        total_problems = summary.get('total_problems', len(sampled.contexts))

        # Extract timing/cost info from nested structures
        # CodeGen tracks costs in eval_timing and codegen_timing
        eval_timing = summary.get('eval_timing', {})
        codegen_timing = summary.get('codegen_timing', {})

        # Calculate total Phase 2 cost (critic evaluation only)
        # Note: codegen_timing costs are excluded because they represent cached/one-time
        # costs from initial code generation, not costs incurred during this evaluation.
        # Including them would cause double-counting when multiple agents evaluate
        # the same cached problems.
        total_phase2_cost = eval_timing.get('total_cost_usd', 0.0)

        # Distribute cost evenly across problems for per-question tracking
        # (CodeGen doesn't have per-question costs, so we approximate)
        per_problem_cost = total_phase2_cost / results_count if results_count else 0.0

        # Convert results to standard format
        formatted_results = []
        for r in results_iter:
            formatted_results.append({
                'question_id': r.get('question_id'),
                # Use standard 'correct' field, with fallback for legacy
                'correct': r.get('correct', False),
                'v1_passed': r.get('v1_passed', False),
                'v2_passed': r.get('v2_passed', False),
                'improved': r.get('improved', False),
                'regressed': r.get('regressed', False),
                'verdict_correct': r.get('verdict_correct', False),
                'error': r.get('error'),
                # Include phase2_cost for cost report compatibility
                'phase2_cost': per_problem_cost,
            })

        # Only count eval tokens (codegen tokens are from cached generation, not this run)
        phase2_tokens_in = eval_timing.get('input_tokens', 0)
        phase2_tokens_out = eval_timing.get('output_tokens', 0)

        return EvaluationResult(
            accuracy=v2_pass_rate,
            total=total_problems,
            correct=v2_passed,
            results=formatted_results,
            metadata={
                # Map to expected format for researcher.py cost tracking
                'phase2_cost': total_phase2_cost,
                'phase2_tokens_in': phase2_tokens_in,
                'phase2_tokens_out': phase2_tokens_out,
                # CodeGen-specific fields
                'v1_pass_rate': summary.get('v1_pass_rate', 0.0),
                'v1_passed': summary.get('v1_passed', 0),
                'improvement': summary.get('improvement', 0.0),
                'problems_improved': summary.get('problems_improved', 0),
                'problems_regressed': summary.get('problems_regressed', 0),
                # Preserve original timing data for detailed analysis
                'eval_timing': eval_timing,
                'codegen_timing': codegen_timing,
            }
        )
