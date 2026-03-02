"""
Code Generation domain implementation.

Implements the coder/critic paradigm for LiveCodeBench.
See local_docs/code_generation_critic/robophd_code_generation.md for design.
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
from RoboPhD.config import LMSTUDIO_DEFAULT_BASE_URL


class CodeGenDomain(DomainInterface):
    """
    Code Generation domain with evolved critics.

    Phase 1 Input: Bundle {question, code_v1}
    Phase 1 Output: Critic feedback
    Phase 2 Output: Revised code (Code v2) via session resumption
    Evaluation: Binary pass/fail on hidden tests

    Key architectural difference from Text2SQL:
    - Text2SQL Phase 2: Fresh API call with generated system prompt
    - CodeGen Phase 2: Resume coder's session with critic feedback

    The session_id is part of the cached problem data (alongside code_v1).
    This enables the coder to evaluate critic feedback against its original
    reasoning, preserved in session context.

    Cached problem data (from preprocessing):
    - question: Problem statement from LiveCodeBench
    - code_v1: Initial solution from Coder Call 1
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

        # Configure cache directory (default: outside git repo to avoid CLAUDE.md contamination)
        coder_model = config.get('coder_model', 'haiku-4.5')
        if 'cache_dir' in config:
            self.cache_dir = Path(config['cache_dir'])
        else:
            # Sanitize model name for filesystem (e.g., 'qwen/qwen3-coder-30b' -> 'qwen--qwen3-coder-30b')
            cache_model_name = coder_model.replace("/", "--")
            # Append variant tag for cache isolation across quantization/context settings
            tag = config.get('coder_model_tag', '')
            if tag:
                cache_model_name = f"{cache_model_name}_{tag}"
            runs_dir = Path(config.get('runs_directory', '../robophd_runs'))
            self.cache_dir = runs_dir / "codegen_cache" / f"{cache_model_name}_v6"

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
        return "evolution_strategies"

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
    problems/
      <problem_id>/
        solution.py                ← Original code (v1)
        tool_output/
          critic_feedback.txt      ← Tool-generated analysis
        critic_prompt.md           ← Full prompt sent to critic (includes eval_instructions)
        feedback.md                ← Critic's verdict (CORRECT/INCORRECT) and feedback
        solution_v2.py             ← Revised code (or symlink to v1 if verdict=CORRECT)
        acceptance.md              ← Coder's explanation of changes (if revised)
        result.json                ← Per-problem evaluation result

Agent source code:
  ../../agents/
    <agent_name>/
      eval_instructions.md  ← Code review instructions
      tools/
        problem_analyzer.py ← Analysis tool
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

        CodeGen uses flat sampling: examples_per_iteration problems
        from the problem pool. Each problem is its own context (1:1 mapping).

        Args:
            config: Configuration with examples_per_iteration
            rng: Random number generator for reproducibility
            available_contexts: Optional list of problem IDs to sample from

        Returns:
            SampledProblems with sampled problem IDs and problem dicts
        """
        examples_per_iteration = config.get("examples_per_iteration", 50)

        # Get available problem IDs
        if available_contexts is None:
            available_contexts = self.get_contexts()

        # Sample problem IDs
        if len(available_contexts) <= examples_per_iteration:
            sampled_ids = list(available_contexts)
        else:
            sampled_ids = rng.sample(available_contexts, examples_per_iteration)

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

    def _find_cached_results(
        self,
        agent_id: str,
        problem_ids: List[str],
        output_dir: Path
    ) -> Dict[str, Path]:
        """
        Find prior iteration results for (agent, problem) pairs.

        Scans iteration directories before the current one for existing results
        that can be reused via symlinks.

        A result is cacheable only if its result.json:
        - Exists on disk
        - Has no "error" key
        - Has "revision_failed_reason": null

        Args:
            agent_id: Agent identifier
            problem_ids: List of problem IDs to look up
            output_dir: Current iteration's agent output dir
                (e.g., <runs_dir>/evolution/<run_name>/iteration_014/agent_iter9)

        Returns:
            Dict mapping problem_id -> resolved Path to the cached problem directory
        """
        # Derive iteration number and run directory from output_dir
        # output_dir = <runs_dir>/evolution/<run_name>/iteration_XXX/agent_<agent_id>
        iteration_dir = output_dir.parent  # iteration_XXX
        run_dir = iteration_dir.parent     # <runs_dir>/evolution/<run_name>

        iter_name = iteration_dir.name  # "iteration_014"
        try:
            current_iter_num = int(iter_name.split("_")[1])
        except (IndexError, ValueError):
            self.logger.warning(f"Cannot parse iteration number from {iter_name}")
            return {}

        # Build agent dir name for lookup (same agent_id across iterations)
        agent_dir_name = output_dir.name  # "agent_iter9"

        cached = {}
        remaining = set(problem_ids)

        # Scan prior iterations in reverse order (most recent first for freshest results)
        for prior_iter in range(current_iter_num - 1, 0, -1):
            if not remaining:
                break

            prior_iter_dir = run_dir / f"iteration_{prior_iter:03d}" / agent_dir_name / "problems"
            if not prior_iter_dir.is_dir():
                continue

            for problem_id in list(remaining):
                problem_dir = prior_iter_dir / problem_id
                result_file = problem_dir / "result.json"
                if not result_file.exists():
                    continue

                try:
                    with open(result_file) as f:
                        result_data = json.load(f)
                except (json.JSONDecodeError, IOError):
                    continue

                # Only cache successful results (no errors, no revision failures)
                if "error" in result_data:
                    continue
                if result_data.get("revision_failed_reason") is not None:
                    continue

                # Resolve to avoid symlink chains
                cached[problem_id] = problem_dir.resolve()
                remaining.discard(problem_id)

        return cached

    def _build_summary_from_results(
        self,
        all_results: List[Dict[str, Any]],
        fresh_eval_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute evaluation summary from merged results (cached + fresh).

        Preserves cost/timing fields from fresh_eval_data since only fresh
        evaluations incur API cost.

        Args:
            all_results: List of per-problem result dicts
            fresh_eval_data: The subprocess's evaluation.json data (None if all cached)

        Returns:
            Complete evaluation.json-compatible dict with summary and results
        """
        total = len(all_results)
        v1_passed = sum(1 for r in all_results if r.get('v1_passed', False))
        v2_passed = sum(1 for r in all_results if r.get('v2_passed', False))
        improved = sum(1 for r in all_results if r.get('improved', False))
        regressed = sum(1 for r in all_results if r.get('regressed', False))

        summary = {
            'total_problems': total,
            'v1_passed': v1_passed,
            'v2_passed': v2_passed,
            'v1_pass_rate': (v1_passed / total * 100) if total else 0.0,
            'v2_pass_rate': (v2_passed / total * 100) if total else 0.0,
            'improvement': ((v2_passed - v1_passed) / total * 100) if total else 0.0,
            'problems_improved': improved,
            'problems_regressed': regressed,
        }

        # Preserve cost/timing from fresh evaluation (cached results are zero-cost)
        if fresh_eval_data:
            fresh_summary = fresh_eval_data.get('summary', {})
            summary['eval_timing'] = fresh_summary.get('eval_timing', {})
            summary['codegen_timing'] = fresh_summary.get('codegen_timing', {})
        else:
            summary['eval_timing'] = {}
            summary['codegen_timing'] = {}

        # Build results dict keyed by question_id
        results_dict = {}
        for r in all_results:
            qid = r.get('question_id')
            if qid:
                results_dict[qid] = r

        return {'summary': summary, 'results': results_dict}

    def _build_evaluation_result(
        self,
        eval_data: Dict[str, Any],
        total_sampled: int,
        cached_count: int,
        fresh_count: int
    ) -> EvaluationResult:
        """
        Convert evaluation.json data into an EvaluationResult.

        Shared by both the cached and non-cached code paths.

        Args:
            eval_data: Parsed evaluation.json dict
            total_sampled: Total number of sampled problems
            cached_count: Number of problems served from cache
            fresh_count: Number of problems evaluated fresh

        Returns:
            EvaluationResult with accuracy, results, and metadata
        """
        summary = eval_data.get('summary', {})
        results_raw = eval_data.get('results', {})

        # Handle both dict (standard) and list (legacy) formats
        results_iter = results_raw.values() if isinstance(results_raw, dict) else results_raw
        results_count = len(results_raw)

        v2_pass_rate = summary.get('v2_pass_rate', 0.0)
        v2_passed = summary.get('v2_passed', 0)
        total_problems = summary.get('total_problems', total_sampled)

        eval_timing = summary.get('eval_timing', {})
        codegen_timing = summary.get('codegen_timing', {})

        # Calculate total Phase 2 cost (critic evaluation only)
        total_phase2_cost = eval_timing.get('total_cost_usd', 0.0)

        # Distribute cost evenly across fresh problems only (cached problems are zero-cost)
        # For display, spread across all problems to match existing per-problem cost pattern
        per_problem_cost = total_phase2_cost / results_count if results_count else 0.0

        formatted_results = []
        for r in results_iter:
            formatted_results.append({
                'question_id': r.get('question_id'),
                'correct': r.get('correct', False),
                'v1_passed': r.get('v1_passed', False),
                'v2_passed': r.get('v2_passed', False),
                'improved': r.get('improved', False),
                'regressed': r.get('regressed', False),
                'verdict_correct': r.get('verdict_correct', False),
                'error': r.get('error'),
                'phase2_cost': per_problem_cost,
            })

        phase2_tokens_in = eval_timing.get('input_tokens', 0)
        phase2_tokens_out = eval_timing.get('output_tokens', 0)

        return EvaluationResult(
            accuracy=v2_pass_rate,
            total=total_problems,
            correct=v2_passed,
            results=formatted_results,
            metadata={
                'phase2_cost': total_phase2_cost,
                'phase2_tokens_in': phase2_tokens_in,
                'phase2_tokens_out': phase2_tokens_out,
                'v1_pass_rate': summary.get('v1_pass_rate', 0.0),
                'v1_passed': summary.get('v1_passed', 0),
                'improvement': summary.get('improvement', 0.0),
                'problems_improved': summary.get('problems_improved', 0),
                'problems_regressed': summary.get('problems_regressed', 0),
                'eval_timing': eval_timing,
                'codegen_timing': codegen_timing,
                'cached_count': cached_count,
                'fresh_count': fresh_count,
            }
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
        codegen_timeout = config['codegen_call_timeout']
        critic_timeout = config['critic_call_timeout']
        max_concurrent = config['max_concurrent']

        # --- Eval result cache: find prior (agent, problem) results ---
        agent_id = config.get('agent_id', '')
        cache_enabled = config.get('eval_result_cache', True)
        cached_results_map: Dict[str, Path] = {}
        cached_result_data: Dict[str, Dict] = {}

        if cache_enabled and agent_id:
            cached_results_map = self._find_cached_results(
                agent_id, sampled.contexts, output_dir
            )

            if cached_results_map:
                # Create problems/ dir and symlink cached problem dirs
                problems_dir = output_dir / "problems"
                problems_dir.mkdir(parents=True, exist_ok=True)

                for problem_id, source_dir in list(cached_results_map.items()):
                    # Load result.json for merging later
                    try:
                        with open(source_dir / "result.json") as f:
                            cached_result_data[problem_id] = json.load(f)
                    except (json.JSONDecodeError, IOError) as e:
                        self.logger.warning(f"Failed to read cached result for {problem_id}: {e}")
                        # Remove from cache — will be evaluated fresh
                        del cached_results_map[problem_id]
                        continue

                    # Create symlink for cached problem dir
                    symlink_target = problems_dir / problem_id
                    if not symlink_target.exists():
                        symlink_target.symlink_to(source_dir)

        cached_count = len(cached_result_data)
        fresh_problem_ids = [p for p in sampled.contexts if p not in cached_result_data]
        fresh_count = len(fresh_problem_ids)

        if cached_count > 0:
            self.logger.info(
                f"Eval cache: {cached_count} cached, {fresh_count} fresh "
                f"(out of {len(sampled.contexts)} total)"
            )

        # --- All-cached fast path: skip subprocess entirely ---
        if fresh_count == 0 and cached_count > 0:
            self.logger.info("All problems cached — skipping subprocess")

            merged_eval = self._build_summary_from_results(
                list(cached_result_data.values()), None
            )
            # Write merged evaluation.json
            eval_file = output_dir / "evaluation.json"
            with open(eval_file, 'w') as f:
                json.dump(merged_eval, f, indent=2)

            return self._build_evaluation_result(
                merged_eval, len(sampled.contexts), cached_count, fresh_count
            )

        # Build problem IDs list (only uncached problems if cache is active)
        problem_ids = ','.join(fresh_problem_ids)

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
        lmstudio_base_url = config.get('lmstudio_base_url', LMSTUDIO_DEFAULT_BASE_URL)
        cmd = [
            sys.executable,
            str(script_path),
            "--coder-model", coder_model,
            "--critic-model", critic_model,
            "--critic-agent", str(agent_path),
            "--output-dir", str(output_dir),
            "--cache-dir", str(self.cache_dir),
            "--problem-ids", problem_ids,
            "--codegen-timeout", str(codegen_timeout),
            "--critic-timeout", str(critic_timeout),
            "--max-concurrent", str(max_concurrent),
            "--lmstudio-base-url", lmstudio_base_url,
        ]

        # Add test-set or evolution-set flags based on codegen_split
        if self.codegen_split == "test":
            cmd.append("--test-set")
        else:
            cmd.append("--evolution-set")

        self.logger.info(f"Running critic evaluation on {fresh_count} problems ({cached_count} cached)")
        self.logger.debug(f"Command: {' '.join(cmd)}")

        # Calculate subprocess timeout based on expected runtime
        # - Each problem uses codegen calls (regen, session check) + critic calls (critic, revision, acceptance)
        # - Problems run concurrently (max_concurrent workers)
        # - 1.5x safety margin + 10 min buffer for startup/teardown
        # This timeout guards against catastrophic hangs; normal operation completes faster
        estimated_runtime = (fresh_count / max_concurrent) * (2 * codegen_timeout + 2 * critic_timeout)
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

        # --- Merge cached results with fresh subprocess results ---
        if cached_result_data:
            fresh_results = eval_data.get('results', {})
            if isinstance(fresh_results, list):
                fresh_results = {r.get('question_id'): r for r in fresh_results if r.get('question_id')}

            # Combine cached + fresh results
            all_results = list(cached_result_data.values()) + list(fresh_results.values())
            merged_eval = self._build_summary_from_results(all_results, eval_data)

            # Write merged evaluation.json (overwrites subprocess's partial version)
            with open(eval_file, 'w') as f:
                json.dump(merged_eval, f, indent=2)

            return self._build_evaluation_result(
                merged_eval, len(sampled.contexts), cached_count, fresh_count
            )

        # No cache — use subprocess output directly
        return self._build_evaluation_result(
            eval_data, len(sampled.contexts), 0, len(sampled.contexts)
        )
