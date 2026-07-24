"""
Deep Focus Evolution Manager for RoboPhD Phase 2.

Orchestrates multi-round agent evolution within a single Claude Code session:
- Round 1: Strategy-guided analysis, planning, and implementation (reasoning.md + artifacts)
- Rounds 2+: Test against prior iterations and refine based on performance

Each round uses explicit session ID management (--session-id and --resume) to maintain
context while preventing interference from concurrent Claude Code sessions.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from RoboPhD.domains.base import DomainInterface

from RoboPhD.domains.base import SampledProblems, EvaluationResult
from RoboPhD.config import CLAUDE_CLI_MODEL_MAP, build_evolution_env
from utilities.claude_cli import call_claude_cli, claude_cli_settings, RateLimitExceeded

logger = logging.getLogger(__name__)

EVOLUTION_ENVIRONMENT_GUIDE = """\
# Evolution Environment

## Available Data

Use your available tools to explore the experiment directory. Key artifacts:

- `../../agents/<name>/` — agent source code (one directory per agent)
- `../../iteration_NNN/` — evaluation results per iteration:
  - `error_analysis_report.md` — cross-agent score comparison and failure summary
  - `error_index.json` — machine-readable score data (error analysis report was derived from this)
  - `cost_report.md` — per-agent cost breakdown. Useful if you are instructed to pay attention to cost
  - `agent_<name>/problems/<id>/` — per-problem results and diagnostics

During testing and refinement rounds (Rounds 2+), your results appear in `./iteration_NNN_test/` (in your working directory, not the experiment root).

## Strategy Tools

If a `strategy_tools/` directory exists in your working directory, it contains Python helper scripts provided by your evolution strategy. **Run them** — they analyze prior iteration data and produce structured output to guide your work. Use `__PYTHON_EXECUTABLE__ strategy_tools/<script>.py --help` to discover usage.

## Scratch space

Your working directory is your iteration's evolution dir. If you need to drop a small test harness or scratch script while debugging, write it there directly — `/tmp` is outside the write scope, and there's no need to clean up after yourself afterwards. Leftover debugging artifacts in the iteration dir are useful for retrospective analysis.

"""


class EvolutionResult(NamedTuple):
    """Result of a deep focus evolution run."""
    artifact_paths: Dict[str, Path]  # rel_path -> abs_path for each file_mapping artifact
    timing_info: Dict[str, Any]
    cost_info: Dict[str, Any]
    session_id: Optional[str]


class DeepFocusEvolutionManager:
    """
    Manages multi-round Deep Focus evolution process.

    Architecture:
    - Round 1: Analysis, planning, and implementation (reasoning.md + artifacts) - strategy-guided
    - Rounds 2+: Test and refine loop - data-driven

    Each round maintains context via explicit session ID management
    (--session-id for first call, --resume for subsequent calls).
    """

    def __init__(
        self,
        test_rounds: int = 2,
        test_round_offset: int = -2,
        evolution_model: str = "opus-5",
        timeout: int = 1800,
        max_workers: int | None = None,
        llm_call_timeout: int = 120,
        domain: Optional['DomainInterface'] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Deep Focus Evolution Manager.

        Args:
            test_rounds: Number of prior iterations to test against (default 2)
                        0 = Round 1 only (no testing)
                        1 = Rounds 1, 2 (test against 1 iteration)
                        2 = Rounds 1, 2, 3 (test against 2 iterations) [DEFAULT]
            test_round_offset: Starting offset from current iteration (default -2).
                        At iteration 8 with offset -2, first test is against iteration 6.
                        Iterations < 1 are skipped.
            evolution_model: Model for evolution/planning (default opus-5)
            timeout: Timeout in seconds for Claude CLI calls (default 1800)
            max_workers: Maximum concurrent context processing (None = Python default)
            llm_call_timeout: Per-call LLM timeout in seconds (default 120, for local models)
            domain: Optional domain interface for domain-specific workspace setup.
                   If not provided, defaults to Text2SQL-style database symlink.
            config: Full configuration from ConfigManager. Passed to domain evaluation
                   to ensure all domain-specific fields (coder_model, critic_model, etc.)
                   are available without manual extraction.
        """
        self.test_rounds = test_rounds
        self.test_round_offset = test_round_offset
        self.evolution_model = evolution_model
        self.timeout = timeout
        self.max_workers = max_workers
        self.llm_call_timeout = llm_call_timeout
        self.domain = domain
        self.config = config or {}
        self._file_mapping = getattr(domain, 'file_mapping', {}) if domain else {}
        self._task_objective = getattr(domain, 'task_objective', '') if domain else ''
        self._task_background = getattr(domain, 'task_background', '') if domain else ''

        # Set during evolve_agent() call
        self.working_dir: Optional[Path] = None
        self.experiment_dir: Optional[Path] = None
        self.current_iteration: Optional[int] = None
        self.session_id: Optional[str] = None
        self.timing_info: Dict[str, float] = {}

    def evolve_agent(
        self,
        working_dir: Path,
        experiment_dir: Path,
        current_iteration: int,
        evolution_strategy_name: str,
        evolution_prompt: str,
        contexts: Dict[int, List[str]],
    ) -> Tuple[Path, Path, Optional[Path], Dict[str, float]]:
        """
        Main orchestration method for deep focus evolution.

        Args:
            working_dir: Evolution workspace (evolution_output/iteration_XXX/)
            experiment_dir: Root experiment directory
            current_iteration: Current iteration number
            evolution_strategy_name: Name of evolution strategy (for Round 1)
            evolution_prompt: Strategy-specific prompt (for Round 1)
            contexts: Dict mapping test_iteration -> list of context names
                      e.g., {34: ['db1', 'db2'], 33: ['db1', 'db2']}
                      Used for Rounds 3+ testing against prior iterations

        Returns:
            EvolutionResult with artifact_paths, timing_info, cost_info, session_id
            timing_info contains: error_analyzer, planning, test_refine_1,
                                 test_refine_2, total (all in seconds)
            cost_info contains: cost and token breakdown for each phase

        Side Effects:
            - Creates session_summary.md in working_dir (readable markdown summary)
            This summary enables debugging and future meta-evolution analysis

        Raises:
            RuntimeError: If any round fails
            NotImplementedError: If domain is CodeGen (not yet supported)
        """
        self.working_dir = working_dir
        self.experiment_dir = experiment_dir
        self.current_iteration = current_iteration

        # Drop the sandbox settings into this iteration's cwd. Claude
        # CLI only loads .claude/settings.local.json from the cwd it's
        # invoked with (no walk-up), so each iteration needs its own copy.
        from RoboPhD.researcher import install_iteration_sandbox
        install_iteration_sandbox(working_dir, experiment_dir)

        # Generate unique session ID for this evolution run
        self.session_id = str(uuid.uuid4())

        # Reset timing info for this evolution
        self.timing_info = {
            'reflection': 0.0
        }

        # Reset cost info for this evolution
        # Test rounds have nested structure to separate eval and evolution (refinement) costs
        self.cost_info = {
            'first_draft': {'cost': 0.0, 'tokens_in': 0, 'tokens_out': 0, 'cache_created': 0, 'cache_read': 0},
            'reflection': {'cost': 0.0, 'tokens_in': 0, 'tokens_out': 0, 'cache_created': 0, 'cache_read': 0},
            'total': 0.0
        }

        # Dynamically add test_refine_N entries based on test_rounds config
        # This supports any number of test rounds (meta-evolution can set 3+)
        for i in range(1, self.test_rounds + 1):
            self.cost_info[f'test_refine_{i}'] = {
                'eval': {'cost': 0.0, 'tokens_in': 0, 'tokens_out': 0, 'cache_created': 0, 'cache_read': 0},
                'evolution': {'cost': 0.0, 'tokens_in': 0, 'tokens_out': 0, 'cache_created': 0, 'cache_read': 0},
                'cost': 0.0, 'tokens_in': 0, 'tokens_out': 0, 'cache_created': 0, 'cache_read': 0
            }

        logger.info(f"Deep Focus session ID: {self.session_id}")

        # Write CLAUDE.md with domain background to evolution_output/.
        # Claude Code traverses up to find it, so all iteration subdirs inherit it.
        evolution_output_dir = self.experiment_dir / "evolution_output"
        claude_md_path = evolution_output_dir / "CLAUDE.md"
        if not claude_md_path.exists():
            sections = []
            if self._task_background:
                sections.append(f"# Domain Background\n\n{self._task_background}")
            if self._task_objective:
                sections.append(f"# Domain Objective\n\n{self._task_objective}")
            guide = EVOLUTION_ENVIRONMENT_GUIDE.replace("__PYTHON_EXECUTABLE__", sys.executable)

            # Dynamically detect and advertise available CLI tools
            available_tools = []
            missing_tools = []
            for tool in ["jq", "tree"]:
                (available_tools if shutil.which(tool) else missing_tools).append(tool)
            if available_tools:
                guide += "\n\n## CLI Tools\n\nAvailable: " + ", ".join(f"`{t}`" for t in available_tools)
            if missing_tools:
                logger.warning("Recommended CLI tools not found: %s. Install for better evolution results.", ", ".join(missing_tools))

            sections.append(guide)
            claude_md_path.write_text("\n\n".join(sections))
            logger.info(f"CLAUDE.md written to: {claude_md_path}")

        # Create symlink to strategy tools for state persistence
        # This allows research-driven strategies to maintain state (e.g., papers_pool.json)
        strategy_tools_src = self.experiment_dir / "evolution_strategies" / evolution_strategy_name / "strategy_tools"
        if strategy_tools_src.exists():
            strategy_tools_link = working_dir / "strategy_tools"
            if not strategy_tools_link.exists():
                # Compute relative path from symlink location to target
                # This ensures the symlink works regardless of where experiment_dir is located
                relative_target = os.path.relpath(strategy_tools_src, working_dir)
                strategy_tools_link.symlink_to(relative_target, target_is_directory=True)
                logger.info(f"✓ Linked strategy tools: {strategy_tools_src.name} (relative: {relative_target})")

        # Track total evolution time
        total_start_time = time.time()

        try:
            # Note: Previous iteration already has error_analysis_report.md (generated by researcher.py)
            # Evolution will reference it directly from ../../iteration_XXX/
            self.timing_info['error_analyzer'] = 0.0

            # Round 1: Analysis, Planning, and Implementation (ALWAYS)
            logger.info("=" * 60)
            logger.info("ROUND 1: Analysis, Planning, and Implementation")
            logger.info("=" * 60)
            start_time = time.time()
            self._current_call_costs = []  # Reset for this round
            self._round1_analysis_and_implementation(evolution_prompt, continue_from_analyzer=False)
            self.timing_info['first_draft'] = time.time() - start_time
            self._aggregate_round_costs('first_draft')
            logger.info(f"Planning and implementation time: {self.timing_info['first_draft']/60:.1f} minutes")

            # Rounds 2+: Test and Refine (OPTIONAL - based on test_rounds)
            test_round_count = 0
            for test_round in range(self.test_rounds):
                round_num = 2 + test_round
                test_iteration = current_iteration + self.test_round_offset - test_round

                if test_iteration < 1:
                    logger.warning(f"Cannot test against iteration {test_iteration} (doesn't exist)")
                    break

                # Get contexts for this test iteration
                test_contexts = contexts.get(test_iteration, [])
                if not test_contexts:
                    logger.info(f"No contexts available for iteration {test_iteration}, skipping Round {round_num}")
                    break

                logger.info("=" * 60)
                logger.info(f"ROUND {round_num}: Testing and Refinement")
                logger.info(f"Testing against iteration {test_iteration}")
                logger.info(f"Problems: {', '.join(test_contexts)}")
                logger.info("=" * 60)

                start_time = time.time()
                self._current_call_costs = []  # Reset for this round
                self._test_and_refine_round(
                    round_num=round_num,
                    test_iteration=test_iteration,
                    contexts=test_contexts,
                )
                self.timing_info[f'test_refine_{test_round + 1}'] = time.time() - start_time
                self._aggregate_round_costs(f'test_refine_{test_round + 1}')
                test_round_count += 1

            # Set unused test rounds to 0
            for i in range(test_round_count, self.test_rounds):
                self.timing_info[f'test_refine_{i + 1}'] = 0.0
                # Cost info already initialized to 0

            # Save final artifacts at top level (copy from last snapshot)
            logger.info("=" * 60)
            logger.info("Saving final artifacts")
            logger.info("=" * 60)

            final_agent = self._save_final_artifacts()

            # Calculate total evolution time and cost
            self.timing_info['total'] = time.time() - total_start_time
            self.cost_info['total'] = sum(
                phase['cost']
                for key, phase in self.cost_info.items()
                if key != 'total' and isinstance(phase, dict)
            )

            logger.info(f"Deep Focus evolution complete for iteration {current_iteration}")
            logger.info(f"Total rounds executed: {1 + test_round_count}")
            logger.info(f"Total evolution time: {self.timing_info['total']/60:.1f} minutes")
            logger.info(f"Total evolution cost: ${self.cost_info['total']:.2f}")

            # Request evolution reflection for meta-learning and debugging
            reflection_start = time.time()
            self._request_evolution_reflection()
            self.timing_info['reflection'] = time.time() - reflection_start

            # Recalculate total cost to include reflection
            self.cost_info['total'] = sum(
                phase['cost']
                for key, phase in self.cost_info.items()
                if key != 'total' and isinstance(phase, dict)
            )

            # Recalculate total time to include reflection
            self.timing_info['total'] = time.time() - total_start_time

            # Save evolution session transcript for debugging and meta-evolution
            self._save_session_transcript()

            return EvolutionResult(
                artifact_paths=final_agent,
                timing_info=self.timing_info,
                cost_info=self.cost_info,
                session_id=self.session_id,
            )

        except Exception as e:
            logger.error(f"Deep Focus evolution failed: {e}")
            raise RuntimeError(f"Deep Focus evolution failed: {e}") from e

    def _extract_agent_name_from_reasoning(self, reasoning_path: Path) -> str:
        """
        Extract agent name from reasoning.md.

        Looks for a line like:
            Name: my-agent-name
            ## Agent Name: my-agent-name

        Args:
            reasoning_path: Path to reasoning.md file

        Returns:
            Agent name with 'agent_' prefix (defaults to 'agent_new' if not found)
        """
        import re

        if not reasoning_path.exists():
            logger.warning(f"Reasoning file not found: {reasoning_path}, using default 'agent_new'")
            return 'agent_new'

        try:
            with open(reasoning_path, 'r') as f:
                for line in f:
                    stripped = line.strip().lstrip('#').strip()
                    match = re.match(r'^(?:agent\s+)?name:\s*(.+)', stripped, re.IGNORECASE)
                    if match:
                        name = match.group(1).strip().replace('-', '_').replace(' ', '_')
                        if not name.startswith('agent_'):
                            name = f'agent_{name}'
                        return name
        except Exception as e:
            logger.warning(f"Error extracting agent name from {reasoning_path}: {e}")

        logger.warning("Agent name not found in reasoning.md, using default 'agent_new'")
        return 'agent_new'

    def _generate_new_vs_baseline_analysis(self, test_workspace: Path, test_iteration: int, new_agent_name: str):
        """
        Generate new vs baseline error analysis for test round.

        Runs create_deep_focus_error_index.py to generate:
        - error_index.json: Structured data for programmatic access
        - error_analysis_report.md: Human-readable summary comparing new agent vs baselines

        Args:
            test_workspace: Path to test workspace (contains new agent + baseline symlinks)
            test_iteration: Iteration number being tested against
            new_agent_name: Name of new agent being tested
        """
        baseline_iter = self.experiment_dir / f"iteration_{test_iteration:03d}"

        # Paths
        error_index_path = test_workspace / "error_index.json"
        error_report_path = test_workspace / "error_analysis_report.md"

        try:
            # Run create_deep_focus_error_index.py with both directories
            cmd = [
                sys.executable, "-m",
                "RoboPhD.tools.error_analysis.create_deep_focus_error_index",
                "--iteration-dirs", f"{baseline_iter},{test_workspace}",
                "--output", str(error_index_path),
                "--new-agent", new_agent_name
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                logger.error(f"Failed to generate error index: {result.stderr}")
                return

            # Generate simple markdown report from JSON
            if error_index_path.exists():
                with open(error_index_path, 'r') as f:
                    index = json.load(f)

                # Create markdown report
                report_lines = [
                    f"# New vs Baseline Analysis - Test Round",
                    "",
                    f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    ""
                ]

                from RoboPhD.report_generator import (
                    is_continuous_scoring, format_continuous_score_table_deep_focus,
                    format_binary_report_deep_focus,
                )
                scores_by_question = index.get('scores_by_question', {})
                summary = index.get('summary', {})
                newest_agent = summary.get('new_agent', 'unknown')
                baseline_agents = summary.get('baseline_agents', [])
                agent_explanations = summary.get('agent_explanations', {}) or {}
                agent_aggregate_scores = summary.get('agent_aggregate_scores', {}) or {}

                if scores_by_question and is_continuous_scoring(scores_by_question):
                    report_lines.append(f"**New Agent**: {newest_agent}")
                    if baseline_agents:
                        report_lines.append(f"**Baselines**: {', '.join(baseline_agents)}")
                    report_lines.append("")
                    report_lines.extend(format_continuous_score_table_deep_focus(
                        scores_by_question, newest_agent, baseline_agents,
                        agent_explanations=agent_explanations,
                        agent_aggregate_scores=agent_aggregate_scores,
                    ))
                else:
                    report_lines.extend(format_binary_report_deep_focus(index))

                # Write report
                with open(error_report_path, 'w') as f:
                    f.write('\n'.join(report_lines))

                logger.info(f"✓ Generated error analysis: {error_report_path}")
            else:
                logger.warning(f"Error index not generated: {error_index_path}")

        except subprocess.TimeoutExpired:
            logger.error(f"Error analysis generation timed out after 300 seconds")
        except Exception as e:
            logger.error(f"Failed to generate error analysis: {e}")

    def _round1_analysis_and_implementation(self, evolution_prompt: str, continue_from_analyzer: bool = False):
        """
        Round 1: Strategy-guided analysis, planning, and implementation.

        Creates reasoning.md with strategic analysis and improvement plan,
        then implements the 3-artifact agent package:
        - agent artifacts (from file_mapping)
        - eval_instructions.md (evaluation instructions)
        - tools/ (optional analysis scripts, if in file_mapping)

        Saves snapshot after completion.

        Args:
            evolution_prompt: Strategy-specific prompt from evolution strategy
            continue_from_analyzer: Whether to use --continue (if error analyzer ran)
        """
        logger.info("Executing Round 1: Analysis, planning, and implementation")

        # Build artifact listing from file_mapping
        artifact_lines = []
        for idx, (key, path) in enumerate(sorted(self._file_mapping.items()), 1):
            artifact_lines.append(f"{idx}. `{path}` - {key.replace('_', ' ')}")
        artifact_listing = "\n".join(artifact_lines)

        prompt = f"""
{evolution_prompt}

## Round 1: Analysis, Planning, and Implementation

Based on the evolution strategy guidance above, please complete both steps below.

### Step 1: Analysis and Planning

1. Analyze the provided data (agent performance, errors, patterns)
2. Develop a strategic improvement plan
3. Document your reasoning and planned changes

**Error Analysis Available:**
- Previous iteration error analysis: `../../iteration_{self.current_iteration-1:03d}/error_analysis_report.md`

Create a file called `reasoning.md` with your analysis and plan.
Include a `Name:` line (e.g. `Name: my-agent-name`) for agent identification.

### Step 2: Implementation

Based on your analysis and plan in `reasoning.md`, create the agent artifacts:

{artifact_listing}

Create these artifacts in the current directory.

After completing both steps, respond with: "ROUND 1 COMPLETE"
"""

        # Save full prompt for debugging and reproducibility
        (self.working_dir / "evolution_prompt.md").write_text(prompt)
        logger.info(f"Evolution prompt saved to: {self.working_dir / 'evolution_prompt.md'}")

        # Call Claude Code for Round 1
        success = self._call_claude_code(
            prompt=prompt,
            continue_session=continue_from_analyzer,  # Continue if error analyzer ran
            expected_completion="ROUND 1 COMPLETE"
        )

        if not success:
            raise RuntimeError("Round 1 (analysis, planning, and implementation) failed")

        # Verify reasoning.md was created
        reasoning_file = self.working_dir / "reasoning.md"
        if not reasoning_file.exists():
            description = "This file should contain your analysis and planning from Step 1."
            if not self._recover_missing_file(reasoning_file, "reasoning.md", description):
                raise RuntimeError("Round 1 did not create reasoning.md")

        logger.info(f"✓ reasoning.md created ({reasoning_file.stat().st_size} bytes): {reasoning_file}")

        # Verify artifacts from file_mapping were created
        for key, rel_path in self._file_mapping.items():
            artifact = self.working_dir / rel_path
            if not artifact.exists():
                description = f"This file should contain the {key.replace('_', ' ')}."
                if not self._recover_missing_file(artifact, rel_path, description):
                    raise RuntimeError(f"Round 1 did not create {rel_path}")

        logger.info(f"✓ Round 1 complete: reasoning.md and agent artifacts created")
        for key, rel_path in self._file_mapping.items():
            artifact = self.working_dir / rel_path
            if artifact.exists():
                logger.info(f"  - {rel_path}: {artifact.stat().st_size} bytes")

        # Save Round 1 snapshot
        self._save_snapshot(round_num=1)

    def _load_baseline_problems(
        self,
        test_iteration: int,
        contexts: List[str],
    ) -> Dict[str, List[Dict]]:
        """
        Load problems for each context, reusing baseline question IDs where available.

        Args:
            test_iteration: Iteration number to get baseline questions from
            contexts: List of context names (database names or problem_ids)

        Returns:
            Dict mapping context_name -> list of problem dicts
        """
        import glob
        import random

        problems_by_context: Dict[str, List[Dict]] = {}

        # Load all problems from domain
        all_problems = self.domain.load_problems()

        # For each context, try to reuse baseline question IDs
        baseline_iter_dir = self.experiment_dir / f"iteration_{test_iteration:03d}"

        for context_name in contexts:
            context_problems = all_problems.get(context_name, [])

            # Try to load baseline question IDs
            baseline_question_ids = None
            if baseline_iter_dir.exists():
                # Look for evaluation.json in any agent's context directory
                eval_pattern = str(baseline_iter_dir / "agent_*" / context_name / "results" / "evaluation.json")
                eval_files = glob.glob(eval_pattern)

                if eval_files:
                    try:
                        with open(eval_files[0], 'r') as f:
                            baseline_data = json.load(f)
                            baseline_question_ids = set(baseline_data['results'].keys())
                        logger.info(f"    Reusing {len(baseline_question_ids)} question IDs from iteration {test_iteration} for {context_name}")
                    except Exception as e:
                        logger.warning(f"    Could not read baseline questions from {eval_files[0]}: {e}")

            if baseline_question_ids:
                # Filter to only questions matching baseline IDs
                problems_by_context[context_name] = [
                    q for q in context_problems
                    if str(q.get('question_id')) in baseline_question_ids
                ]
            else:
                # Use all available problems
                problems_by_context[context_name] = list(context_problems)

        return problems_by_context

    def _test_and_refine_round(
        self,
        round_num: int,
        test_iteration: int,
        contexts: List[str],
    ):
        """
        Execute a test and refinement round.

        Process:
        1. Create test workspace (iteration_XXX_test/)
        2. Run agent on test contexts using existing infrastructure
        3. Generate standard evaluation report
        4. Generate comparison report vs historical agents
        5. Prompt Claude to refine based on results
        6. Save snapshot of refined artifacts

        Args:
            round_num: Round number (3, 4, 5, ...)
            test_iteration: Iteration number to test against
            contexts: Contexts to test on (databases for Text2SQL, problem_ids for CodeGen)
        """
        logger.info(f"Round {round_num}: Testing against iteration {test_iteration}")

        # Create test workspace
        test_workspace = self.working_dir / f"iteration_{test_iteration:03d}_test"
        test_workspace.mkdir(parents=True, exist_ok=True)

        logger.info(f"Test workspace: {test_workspace}")

        # Verify current agent artifacts exist (from working_dir top level)
        for key, rel_path in self._file_mapping.items():
            if not (self.working_dir / rel_path).exists():
                raise RuntimeError(f"Agent artifact {rel_path} not found for Round {round_num}")

        # Extract agent name from reasoning.md
        reasoning_path = self.working_dir / "reasoning.md"
        agent_name = self._extract_agent_name_from_reasoning(reasoning_path)
        logger.info(f"Testing agent: {agent_name}")

        # Create new agent directory directly in test_workspace.
        # experiment_dir is already outside the RoboPhD repo (in robophd_runs/),
        # so no symlink split needed — CLAUDE.md contamination is avoided.
        new_agent_dir = test_workspace / agent_name
        new_agent_dir.mkdir(parents=True, exist_ok=True)

        # Create symlinks to baseline agents for comparison
        # Use absolute path so symlinks work from any location
        baseline_iter_dir = (self.experiment_dir / f"iteration_{test_iteration:03d}").resolve()
        baseline_scores = {}
        if baseline_iter_dir.exists():
            baseline_agents = list(baseline_iter_dir.glob("agent_*"))
            logger.info(f"Creating symlinks to {len(baseline_agents)} baseline agents")
            for baseline_agent_dir in baseline_agents:
                symlink_path = test_workspace / baseline_agent_dir.name
                if not symlink_path.exists():
                    try:
                        symlink_path.symlink_to(baseline_agent_dir, target_is_directory=True)
                        # Read baseline score from evaluation.json
                        score_str = ""
                        try:
                            eval_json = baseline_agent_dir / "evaluation.json"
                            if eval_json.exists():
                                eval_data = json.loads(eval_json.read_text())
                                score = eval_data.get("summary", {}).get("average_score")
                                if score is not None:
                                    baseline_scores[baseline_agent_dir.name] = score
                                    score_str = f" (score: {score:.3f})"
                        except Exception:
                            pass
                        logger.info(f"  ✓ Linked {baseline_agent_dir.name}{score_str}")
                    except Exception as e:
                        logger.warning(f"  ⚠️  Failed to link {baseline_agent_dir.name}: {e}")
        else:
            logger.warning(f"Baseline iteration not found: {baseline_iter_dir}")

        # Run agent testing using domain-agnostic approach
        context_label = self.domain.context_label_plural if self.domain else "contexts"
        logger.info(f"Running agent on {len(contexts)} {context_label} (max workers: {self.max_workers})...")

        # Create agent package directory with evolved agent artifacts
        agent_package_dir = test_workspace / "agent_package"
        agent_package_dir.mkdir(parents=True, exist_ok=True)
        for key, rel_path in self._file_mapping.items():
            src = self.working_dir / rel_path
            if src.exists():
                dst = agent_package_dir / rel_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                if src.is_dir():
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)

        # Load problems for each context, reusing baseline question IDs where available
        problems_by_context = self._load_baseline_problems(
            test_iteration=test_iteration,
            contexts=contexts,
        )

        # Build SampledProblems (same interface as main loop in researcher.py)
        missing = [c for c in contexts if c not in problems_by_context]
        if missing:
            available_keys = list(problems_by_context.keys())[:5]
            logger.error(
                f"Deep focus: {len(missing)}/{len(contexts)} contexts missing from problems_by_context. "
                f"First missing: {missing[0]!r}, "
                f"available keys (first 5): {available_keys}"
            )
        sampled = SampledProblems(
            contexts=contexts,
            problems_by_context=problems_by_context
        )

        # Build config for run_evaluation (same pattern as main loop)
        eval_config = self.config.copy()
        eval_config.update({
            'agent_id': agent_name,
            'cache_manager': None,  # No caching in DeepFocus testing
            'experiment_dir': self.experiment_dir,
            'max_workers': self.max_workers,
            'llm_call_timeout': self.llm_call_timeout,
        })

        # Single run_evaluation call (same as main loop in researcher.py)
        # Output goes to: new_agent_dir / context_name / results / evaluation.json
        eval_result = self.domain.run_evaluation(
            sampled=sampled,
            agent_path=agent_package_dir,
            output_dir=new_agent_dir,
            config=eval_config
        )

        # Process results from EvaluationResult into per-context format for reports
        results = {}
        for r in eval_result.results:
            # Get context name (db_id for Text2SQL, question_id for CodeGen)
            context_name = r.get('db_id') or r.get('question_id')
            if context_name not in results:
                results[context_name] = {
                    'average_score': 0.0,
                    'score_sum': 0.0,
                    'total': 0,
                }
            results[context_name]['total'] += 1
            results[context_name]['score_sum'] += r.get('score', 0)

        # Calculate per-context average score
        for context_name in results:
            r = results[context_name]
            r['average_score'] = (r['score_sum'] / r['total']) if r['total'] > 0 else 0.0

        # Accumulate evaluation cost
        if eval_result.metadata.get('eval_cost', 0) > 0:
            if not hasattr(self, '_current_call_costs'):
                self._current_call_costs = []
            eval_cost_info = {
                'cost': eval_result.metadata.get('eval_cost', 0),
                'tokens_in': eval_result.metadata.get('eval_tokens_in', 0),
                'tokens_out': eval_result.metadata.get('eval_tokens_out', 0),
                'cache_created': 0,
                'cache_read': 0,
                'call_type': 'eval'
            }
            self._current_call_costs.append(eval_cost_info)

        overall_score = eval_result.average_score
        total_score_sum = eval_result.score_sum
        total_questions = eval_result.total
        aggregate_explanation = (eval_result.metadata or {}).get('aggregate_explanation', '')

        # Compute rank relative to baselines
        rank_str = ""
        if baseline_scores:
            all_scores = list(baseline_scores.values()) + [overall_score]
            # Round to 6 decimals for tie detection (consistent with ranking_table.py)
            rounded_new = round(overall_score, 6)
            higher_count = sum(1 for s in baseline_scores.values() if round(s, 6) > rounded_new)
            tied_count = sum(1 for s in baseline_scores.values() if round(s, 6) == rounded_new)
            rank = higher_count + 1
            total = len(all_scores)
            if tied_count > 0:
                rank_str = f" — rank #{rank} (tied) of {total}"
            else:
                rank_str = f" — rank #{rank} of {total}"
        logger.info(f"Overall score: {overall_score:.3f} ({total_score_sum:.1f}/{total_questions}){rank_str}")

        # Generate new vs baseline error analysis
        try:
            self._generate_new_vs_baseline_analysis(test_workspace, test_iteration, agent_name)
        except Exception as e:
            logger.warning(f"Error analysis generation failed: {e}")
            logger.warning("Continuing without error analysis...")

        # Prompt Claude to refine based on results
        logger.info("Prompting for refinement...")
        self._prompt_for_refinement(
            round_num=round_num,
            test_iteration=test_iteration,
            overall_score=overall_score,
            total_score_sum=total_score_sum,
            total_questions=total_questions,
            agent_name=agent_name,
            aggregate_explanation=aggregate_explanation,
        )

        # Save snapshot after refinement
        self._save_snapshot(round_num=round_num)

        logger.info(f"✓ Round {round_num} complete")

    def _prompt_for_refinement(
        self,
        round_num: int,
        test_iteration: int,
        overall_score: float,
        total_score_sum: float,
        total_questions: int,
        agent_name: str,
        aggregate_explanation: str = "",
    ):
        """
        Prompt Claude Code to refine agent based on test results.

        Uses --continue to maintain context from previous rounds.

        Args:
            round_num: Current round number
            test_iteration: Iteration that was tested against
            overall_score: Aggregator's output for this batch (the
                canonical score Elo compares — see aggregate_explanation
                for how it was derived from raw correctness).
            total_score_sum: Sum of all per-problem raw scores
            total_questions: Total number of questions
            agent_name: Name of the new agent being tested
            aggregate_explanation: Optional human-readable explanation
                of how the aggregator converted raw correctness to
                overall_score (e.g. DS-1000 cost-penalty math). Empty
                for tasks using the default mean aggregator.
        """
        # Check if error analysis report exists (always generated now)
        test_workspace_name = f"iteration_{test_iteration:03d}_test"
        error_analysis_path = self.working_dir / test_workspace_name / "error_analysis_report.md"
        error_analysis_section = ""
        if error_analysis_path.exists():
            error_analysis_section = f"""
### Error Analysis Report
A structured error analysis comparing your new agent against baselines has been generated:
- Report: `./{test_workspace_name}/error_analysis_report.md`
- Error index: `./{test_workspace_name}/error_index.json`
"""

        # Build agent output directory listing
        agent_dirs_section = ""
        test_workspace_path = self.working_dir / test_workspace_name
        if test_workspace_path.exists():
            new_agent_line = f"- `./{test_workspace_name}/{agent_name}/` ← **your agent**"
            baseline_lines = []
            for d in sorted(test_workspace_path.iterdir()):
                if d.is_dir() and d.name != agent_name and d.name != "agent_package" and d.name.startswith("agent_"):
                    baseline_lines.append(f"- `./{test_workspace_name}/{d.name}/`")
            if baseline_lines:
                agent_dirs_section = f"""
### Agent Outputs
Your new agent's per-problem output:
{new_agent_line}

Baseline agents from iteration {test_iteration} (for comparison):
{chr(10).join(baseline_lines)}
"""
            else:
                agent_dirs_section = f"""
### Agent Outputs
Your new agent's per-problem output:
{new_agent_line}
"""

        aggregate_section = (
            f"Aggregator notes: {aggregate_explanation}\n"
            if aggregate_explanation
            else ""
        )
        prompt = f"""
## Round {round_num}: Testing and Refinement

Your agent was tested on data from iteration {test_iteration}.

### Results
Overall score: {overall_score:.3f} ({total_score_sum:.1f}/{total_questions})
{aggregate_section}{error_analysis_section}{agent_dirs_section}
### Your Task
Review the performance results and diagnostic outputs listed above.

Based on what you observe, provide updated versions of any artifacts that need changes.

After refinements, respond with: "ROUND {round_num} COMPLETE"
"""

        # Save refinement prompt for debugging
        (self.working_dir / f"refinement_prompt_round{round_num}.md").write_text(prompt)

        success = self._call_claude_code(
            prompt=prompt,
            continue_session=True,  # Continue from previous rounds
            expected_completion=f"ROUND {round_num} COMPLETE"
        )

        if not success:
            raise RuntimeError(f"Round {round_num} (refinement) failed")

        logger.info(f"✓ Round {round_num} refinement complete")

    def _compact_session(self) -> bool:
        """
        Compact Claude Code session to free context.

        Returns:
            True if compact succeeded, False otherwise
        """
        logger.info("Compacting session to free context...")

        # Build compact command
        claude_cli = self._get_claude_cli_path()
        model = CLAUDE_CLI_MODEL_MAP.get(self.evolution_model, self.evolution_model)

        cmd = [
            str(claude_cli),
            "--model", model,
            "--resume", self.session_id,
            "--print", "/compact",
            "--permission-mode", "bypassPermissions"
        ]

        extra_env = build_evolution_env(
            self.evolution_model, self.experiment_dir,
            iteration_dir=self.working_dir,
        )

        try:
            result = call_claude_cli(
                cmd=cmd,
                cwd=self.working_dir,
                timeout=600,  # 10 minutes for compact (can take a while with large sessions)
                logger=logger,
                extra_env=extra_env
            )

            if result.returncode != 0:
                logger.error(f"Compact failed with return code {result.returncode}")
                logger.error(f"stdout: {result.stdout[:500]}")
                logger.error(f"stderr: {result.stderr}")
                return False

            logger.info("✓ Session compacted successfully")
            return True

        except subprocess.TimeoutExpired:
            logger.error("Compact timed out after 600s")
            return False
        except RateLimitExceeded as e:
            logger.error(f"Rate limit exceeded during compact: {e}")
            return False
        except Exception as e:
            logger.error(f"Compact failed: {e}")
            return False

    def _call_claude_code(
        self,
        prompt: str,
        continue_session: bool,
        expected_completion: Optional[str]
    ) -> bool:
        """
        Call Claude Code CLI with the given prompt.

        Args:
            prompt: Prompt to send to Claude Code
            continue_session: Whether to resume existing session (True) or create new (False)
            expected_completion: Expected completion message (None = don't check)

        Returns:
            True if call succeeded, False otherwise
        """
        # Build command
        claude_cli = self._get_claude_cli_path()
        # Map API model name to Claude CLI name (e.g., 'sonnet-4.5' -> 'sonnet')
        model = CLAUDE_CLI_MODEL_MAP.get(self.evolution_model, self.evolution_model)

        cmd = [
            str(claude_cli),
            "--model", model
        ]

        # Use explicit session management to prevent interference
        if continue_session:
            # Resume existing session by ID
            cmd.extend(["--resume", self.session_id])
        else:
            # Create new session with explicit ID
            cmd.extend(["--session-id", self.session_id])

        cmd.extend([
            "--print", prompt,
            "--output-format", "json",  # Get JSON output for cost tracking
            "--permission-mode", "bypassPermissions",  # Allow automation without prompts
            "--settings", claude_cli_settings()  # autoCompact + repo-read deny
        ])

        logger.debug(f"Calling Claude Code: {' '.join(cmd[:4])}...")

        extra_env = build_evolution_env(
            self.evolution_model, self.experiment_dir,
            iteration_dir=self.working_dir,
        )

        try:
            # Run in working directory with rate limit handling
            result = call_claude_cli(
                cmd=cmd,
                cwd=self.working_dir,
                timeout=self.timeout,
                logger=logger,
                extra_env=extra_env
            )

            if result.returncode != 0:
                logger.error(f"Claude Code call failed with return code {result.returncode}")
                logger.error(f"stdout: {result.stdout[:1000]}")  # First 1000 chars
                logger.error(f"stderr: {result.stderr}")

                # Check if it's a recoverable size-related error (context or prompt too large)
                # Check both stdout and stderr as error messages may appear in either
                output = result.stdout + result.stderr
                if any(msg in output for msg in ["Context low", "Run /compact", "Prompt is too long"]):
                    logger.warning("Recoverable error detected (context/prompt size), attempting compact and retry...")
                    if self._compact_session():
                        logger.info("Retrying after successful compact...")
                        # Build retry command - IMPORTANT: use --resume instead of --session-id
                        # since the session now exists after the first attempt
                        retry_cmd = [
                            str(claude_cli),
                            "--model", model
                        ]

                        # Use --resume for retry (session already exists)
                        retry_cmd.extend(["--resume", self.session_id])

                        retry_cmd.extend([
                            "--print", prompt,
                            "--output-format", "json",
                            "--permission-mode", "bypassPermissions",
                            "--settings", claude_cli_settings()  # autoCompact + repo-read deny
                        ])

                        # Retry with the corrected command (also handles rate limits)
                        retry_result = call_claude_cli(
                            cmd=retry_cmd,
                            cwd=self.working_dir,
                            timeout=self.timeout,
                            logger=logger,
                            extra_env=extra_env
                        )

                        if retry_result.returncode != 0:
                            logger.error(f"Retry failed with return code {retry_result.returncode}")
                            logger.error(f"stdout: {retry_result.stdout[:1000]}")
                            logger.error(f"stderr: {retry_result.stderr}")
                            return False

                        logger.info("✓ Retry succeeded after compact")
                        # Use retry result for completion check below
                        result = retry_result
                    else:
                        logger.error("Compact failed, cannot retry")
                        return False
                else:
                    # Not a recoverable error, just fail
                    return False

            # Parse JSON output for cost tracking
            if result.returncode == 0 and result.stdout:
                try:
                    import json
                    json_output = json.loads(result.stdout)
                    usage = json_output.get('usage', {})
                    call_cost = {
                        'cost': json_output.get('total_cost_usd', 0.0),
                        'tokens_in': usage.get('input_tokens', 0),
                        'tokens_out': usage.get('output_tokens', 0),
                        'cache_created': usage.get('cache_creation_input_tokens', 0),
                        'cache_read': usage.get('cache_read_input_tokens', 0),
                        'call_type': 'evolution'  # Tag evolution calls
                    }

                    # Accumulate into cost_info dict
                    if not hasattr(self, '_current_call_costs'):
                        self._current_call_costs = []
                    self._current_call_costs.append(call_cost)

                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to parse cost data from evolution call: {e}")
                    # Continue anyway - cost tracking is not critical

            # Check for expected completion message in output (if specified)
            if expected_completion and expected_completion not in result.stdout:
                logger.warning(f"Expected completion message not found: {expected_completion}")
                logger.warning(f"Output preview: {result.stdout[:500]}")
                # Don't fail - Claude might have completed successfully anyway

            return True

        except subprocess.TimeoutExpired as e:
            # Check if the CLI actually finished (marker present) but hung on exit
            output = (e.stdout or "") + (e.stderr or "")
            if expected_completion and expected_completion in output:
                logger.warning(
                    f"Claude Code timed out after {self.timeout}s but completion "
                    f"marker found in stdout — treating as success"
                )
                return True
            # With --output-format json the CLI buffers stdout until exit, so
            # captured output may be empty even when the model produced the
            # marker. Fall back to the on-disk session transcript.
            if expected_completion and self._marker_in_transcript(expected_completion):
                logger.warning(
                    f"Claude Code timed out after {self.timeout}s but completion "
                    f"marker found in session transcript — treating as success"
                )
                return True
            logger.error(f"Claude Code call timed out after {self.timeout}s")
            return False
        except RateLimitExceeded as e:
            # Let rate limit exceeded propagate for checkpoint/exit handling
            logger.error(f"Rate limit exceeded: {e}")
            raise
        except Exception as e:
            logger.error(f"Claude Code call failed: {e}")
            return False

    def _aggregate_round_costs(self, round_key: str):
        """
        Aggregate costs from _current_call_costs into cost_info for the given round.
        For test rounds, separates eval and evolution (refinement) costs.

        Args:
            round_key: Key in cost_info dict ('first_draft', 'test_refine_1', etc.)
        """
        if not hasattr(self, '_current_call_costs') or not self._current_call_costs:
            return

        # Test rounds have nested structure (dynamically detect any test_refine_N)
        is_test_round = round_key.startswith('test_refine_')

        for call_cost in self._current_call_costs:
            call_type = call_cost.get('call_type', 'evolution')  # Default to evolution for backward compat

            if is_test_round:
                # Accumulate into nested structure (eval or evolution)
                self.cost_info[round_key][call_type]['cost'] += call_cost['cost']
                self.cost_info[round_key][call_type]['tokens_in'] += call_cost['tokens_in']
                self.cost_info[round_key][call_type]['tokens_out'] += call_cost['tokens_out']
                self.cost_info[round_key][call_type]['cache_created'] += call_cost['cache_created']
                self.cost_info[round_key][call_type]['cache_read'] += call_cost['cache_read']

                # Also accumulate into round-level totals
                self.cost_info[round_key]['cost'] += call_cost['cost']
                self.cost_info[round_key]['tokens_in'] += call_cost['tokens_in']
                self.cost_info[round_key]['tokens_out'] += call_cost['tokens_out']
                self.cost_info[round_key]['cache_created'] += call_cost['cache_created']
                self.cost_info[round_key]['cache_read'] += call_cost['cache_read']
            else:
                # Simple rounds (planning, implementation) - no nested structure
                self.cost_info[round_key]['cost'] += call_cost['cost']
                self.cost_info[round_key]['tokens_in'] += call_cost['tokens_in']
                self.cost_info[round_key]['tokens_out'] += call_cost['tokens_out']
                self.cost_info[round_key]['cache_created'] += call_cost['cache_created']
                self.cost_info[round_key]['cache_read'] += call_cost['cache_read']

        # Clear for next round
        self._current_call_costs = []

    def _get_claude_cli_path(self) -> Path:
        """
        Get path to Claude CLI executable.

        Returns:
            Path to claude executable

        Raises:
            RuntimeError: If Claude CLI not found
        """
        # Try local installation first
        local_cli = Path.home() / ".claude" / "local" / "claude"
        if local_cli.exists():
            return local_cli

        # Try system installation
        try:
            result = subprocess.run(
                ["which", "claude"],
                capture_output=True,
                text=True,
                check=True
            )
            system_cli = Path(result.stdout.strip())
            if system_cli.exists():
                return system_cli
        except subprocess.CalledProcessError:
            pass

        raise RuntimeError(
            "Claude CLI not found. Install from: https://docs.anthropic.com/en/docs/claude-code"
        )

    def _save_snapshot(self, round_num: int):
        """
        Save snapshot of agent artifacts after a round completes.

        Args:
            round_num: Round number (2, 3, 4, ...)
        """
        snapshot_dir = self.working_dir / f"round{round_num}_snapshot"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Copy artifacts from file_mapping to snapshot
        for key, rel_path in self._file_mapping.items():
            src = self.working_dir / rel_path
            if src.exists():
                dst = snapshot_dir / rel_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                if src.is_dir():
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)

        logger.info(f"Saved Round {round_num} snapshot to {snapshot_dir}")

    def _save_final_artifacts(self) -> Dict[str, Path]:
        """
        Save final version of artifacts at top level.

        Returns:
            Dict mapping relative path -> absolute path for each artifact.
        """
        artifact_paths = {}

        # Collect artifacts from file_mapping
        for key, rel_path in self._file_mapping.items():
            src = self.working_dir / rel_path
            if src.exists():
                artifact_paths[rel_path] = src
                logger.info(f"  - {rel_path}: {src}")
            else:
                description = f"This file should contain the {key.replace('_', ' ')}."
                if self._recover_missing_file(src, rel_path, description):
                    artifact_paths[rel_path] = src
                    logger.info(f"  - {rel_path}: {src} (recovered)")
                else:
                    logger.warning(f"  ⚠️ Artifact not found: {rel_path}")

        logger.info(f"Final artifacts verified: {len(artifact_paths)} files")

        return artifact_paths

    def _marker_in_transcript(self, marker: str) -> bool:
        """Return True if the marker appears in an assistant text block
        in the on-disk session transcript.

        We only scan assistant-text blocks because the marker also appears
        in the user prompt instructions (e.g. 'respond with: "ROUND 1 COMPLETE"'),
        which would false-positive a whole-file substring check.
        """
        try:
            from RoboPhD.utilities.transcript_summarizer import find_transcript

            chat_file = find_transcript(self.working_dir, self.session_id)
            if not chat_file:
                return False
            with open(chat_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if msg.get('type') != 'assistant':
                        continue
                    content = msg.get('message', {}).get('content')
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get('type') == 'text':
                                if marker in block.get('text', ''):
                                    return True
                    elif isinstance(content, str):
                        if marker in content:
                            return True
            return False
        except Exception as e:
            logger.warning(f"Transcript marker check failed: {e}")
            return False

    def _save_session_transcript(self):
        """
        Summarize Claude Code session transcript and save to evolution output directory.

        Side effects:
            - Creates session_summary.md in working_dir

        Errors are logged but do not raise exceptions - transcript saving
        should never break the research run.
        """
        try:
            from RoboPhD.utilities.transcript_summarizer import find_transcript, summarize_transcript

            chat_file = find_transcript(self.working_dir, self.session_id)
            if not chat_file:
                logger.warning(f"Session transcript not found for session {self.session_id}")
                return

            summary_path = summarize_transcript(chat_file, self.working_dir / "session_summary.md")
            summary_size = summary_path.stat().st_size
            logger.info(f"Saved session summary: {os.path.relpath(summary_path)} ({summary_size/1024:.1f} KB)")

        except Exception as e:
            logger.warning(f"Failed to save session summary: {e}")

    def _request_evolution_reflection(self):
        """
        Request reflection from Claude about evolution process.

        Asks Claude to provide advice for future evolution sessions based on
        the completed work. This reflection:
        - Captures insights about what worked/didn't work
        - Informs potential improvements to evolution strategies
        - Feeds into future meta-evolution analysis

        The reflection is saved to evolution_reflection.md and captured
        in the session transcript.

        Side effects:
            - Creates evolution_reflection.md in working_dir
            - Adds cost to self.cost_info['reflection']

        Errors are logged but do not raise exceptions - reflection should
        never break the research run.
        """
        logger.info("Requesting evolution reflection...")

        reflection_prompt = """Thanks for your help on this evolution project. Looking back at the entire process, is there advice you could offer to future instances working on this task?

This might lead to changes in the prompt which will make future evolution sessions more efficient or help them achieve higher accuracy.

Please consider:
- What approaches worked well?
- What was challenging or time-consuming?
- Were the provided tools helpful? Did you encounter difficulties using them?
- What would you do differently?
- Any insights about the evolution strategy itself?
- Suggestions for improving the prompts or process?

**Keep your reflection concise - 300 lines or less.**

Save your reflection to a file called `evolution_reflection.md`.

After saving the reflection, respond with: "REFLECTION COMPLETE"
"""

        try:
            # Reset call costs for this reflection
            self._current_call_costs = []

            # Call Claude Code for reflection (continue existing session)
            success = self._call_claude_code(
                prompt=reflection_prompt,
                continue_session=True,  # Continue the evolution session
                expected_completion="REFLECTION COMPLETE"
            )

            # Aggregate costs into reflection phase
            self._aggregate_round_costs('reflection')

            if success:
                reflection_file = self.working_dir / "evolution_reflection.md"
                if reflection_file.exists():
                    logger.info(f"✓ Evolution reflection saved: {reflection_file}")
                else:
                    logger.warning("Reflection completed but evolution_reflection.md not found")
            else:
                logger.warning("Evolution reflection call failed (non-critical)")

        except Exception as e:
            logger.warning(f"Failed to request evolution reflection: {e}")
            # Don't raise - reflection should never break the research run

    def _recover_missing_file(self, expected_path: Path, filename: str, description: str) -> bool:
        """
        Attempt to recover a missing file by prompting Claude to create/move it.

        Args:
            expected_path: Absolute path where file should be
            filename: Name of the file (for logging)
            description: What the file should contain

        Returns:
            True if recovery successful, False otherwise
        """
        logger.warning(f"  ⚠️  {filename} not found at expected location")
        logger.info(f"  Attempting automatic recovery...")

        # Simple prompt: tell Claude exactly where the file should be
        recovery_prompt = f"""The file `{filename}` is missing or not in the correct location.

**Expected absolute path:** `{expected_path.resolve()}`

{description}

Please ensure `{filename}` exists at the exact path specified above. If it exists elsewhere, move it. If it doesn't exist, create it.

Respond with "FILE READY" when `{filename}` exists at `{expected_path.resolve()}`.
"""

        # Make recovery call (cost tracked automatically via _call_claude_code)
        success = self._call_claude_code(
            prompt=recovery_prompt,
            continue_session=True,  # Continue same session
            expected_completion="FILE READY"
        )

        # Verify file is now in correct location
        if success and expected_path.exists():
            logger.info(f"  ✓ Recovery successful: {filename} now exists at correct location")
            return True
        else:
            logger.error(f"  ✗ Recovery failed: {filename} still not found at {expected_path}")
            return False
