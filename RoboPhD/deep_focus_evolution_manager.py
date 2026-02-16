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
from RoboPhD.comparison_report_generator import ComparisonReportGenerator
from RoboPhD.config import CLAUDE_CLI_MODEL_MAP, get_lmstudio_env
from utilities.claude_cli import call_claude_cli, RateLimitExceeded

logger = logging.getLogger(__name__)


class EvolutionResult(NamedTuple):
    """Result of a deep focus evolution run."""
    agent_md_path: Path
    eval_instructions_path: Path
    tools_path: Optional[Path]
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
        evolution_model: str = "opus-4.5",
        eval_model: str = "haiku-4.5",
        analysis_model: str = "haiku-4.5",
        timeout: int = 1800,
        max_concurrent: int = 8,
        verification_retries: int = 2,
        temperature_strategy: str = "progressive",
        debug_log_probability: float = 0.02,
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
            evolution_model: Model for evolution/planning (default opus-4.5)
            eval_model: Model for SQL generation (default haiku-4.5)
            analysis_model: Model for database analysis (default haiku-4.5)
            timeout: Timeout in seconds for Claude CLI calls (default 1800)
            max_concurrent: Maximum concurrent context processing (default 8)
            verification_retries: Number of SQL verification attempts (default 2)
            temperature_strategy: Temperature strategy for SQL generation (default "progressive")
            debug_log_probability: Probability (0.0-1.0) of logging API calls for debugging (default 0.02)
            llm_call_timeout: Per-call LLM timeout in seconds (default 120, for local models)
            domain: Optional domain interface for domain-specific workspace setup.
                   If not provided, defaults to Text2SQL-style database symlink.
            config: Full configuration from ConfigManager. Passed to domain evaluation
                   to ensure all domain-specific fields (coder_model, critic_model, etc.)
                   are available without manual extraction.
        """
        self.test_rounds = test_rounds
        self.evolution_model = evolution_model
        self.eval_model = eval_model
        self.analysis_model = analysis_model
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.verification_retries = verification_retries
        self.temperature_strategy = temperature_strategy
        self.debug_log_probability = debug_log_probability
        self.llm_call_timeout = llm_call_timeout
        self.domain = domain
        self.config = config or {}

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
        problems_per_context: int,
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
            problems_per_context: Problems per context for testing

        Returns:
            Tuple of (agent.md path, eval_instructions.md path, tools/ path, timing_info dict, cost_info dict)
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

        # Generate unique session ID for this evolution run
        self.session_id = str(uuid.uuid4())

        # Reset timing info for this evolution
        self.timing_info = {
            'reflection': 0.0
        }

        # Reset cost info for this evolution
        # Test rounds have nested structure to separate phase1 (DB analysis), phase2 (SQL gen), and evolution (refinement)
        self.cost_info = {
            'first_draft': {'cost': 0.0, 'tokens_in': 0, 'tokens_out': 0, 'cache_created': 0, 'cache_read': 0},
            'reflection': {'cost': 0.0, 'tokens_in': 0, 'tokens_out': 0, 'cache_created': 0, 'cache_read': 0},
            'total': 0.0
        }

        # Dynamically add test_refine_N entries based on test_rounds config
        # This supports any number of test rounds (meta-evolution can set 3+)
        for i in range(1, self.test_rounds + 1):
            self.cost_info[f'test_refine_{i}'] = {
                'phase1': {'cost': 0.0, 'tokens_in': 0, 'tokens_out': 0, 'cache_created': 0, 'cache_read': 0},
                'phase2': {'cost': 0.0, 'tokens_in': 0, 'tokens_out': 0, 'cache_created': 0, 'cache_read': 0},
                'evolution': {'cost': 0.0, 'tokens_in': 0, 'tokens_out': 0, 'cache_created': 0, 'cache_read': 0},
                'cost': 0.0, 'tokens_in': 0, 'tokens_out': 0, 'cache_created': 0, 'cache_read': 0
            }

        logger.info(f"Starting Deep Focus evolution for iteration {current_iteration}")
        logger.info(f"Deep Focus session ID: {self.session_id}")
        logger.info(f"Test rounds configured: {self.test_rounds}")
        logger.info(f"Evolution strategy: {evolution_strategy_name}")

        # Save evolution prompt for debugging and reproducibility
        evolution_prompt_file = working_dir / "evolution_prompt.md"
        evolution_prompt_file.write_text(evolution_prompt)
        logger.info(f"Evolution prompt saved to: {evolution_prompt_file}")

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
                test_iteration = current_iteration - 1 - test_round

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
                logger.info(f"Contexts: {', '.join(test_contexts)}")
                logger.info("=" * 60)

                start_time = time.time()
                self._current_call_costs = []  # Reset for this round
                self._test_and_refine_round(
                    round_num=round_num,
                    test_iteration=test_iteration,
                    contexts=test_contexts,
                    problems_per_context=problems_per_context,
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
                agent_md_path=final_agent[0],
                eval_instructions_path=final_agent[1],
                tools_path=final_agent[2],
                timing_info=self.timing_info,
                cost_info=self.cost_info,
                session_id=self.session_id,
            )

        except Exception as e:
            logger.error(f"Deep Focus evolution failed: {e}")
            raise RuntimeError(f"Deep Focus evolution failed: {e}") from e

    def _extract_agent_name_from_md(self, agent_md_path: Path) -> str:
        """
        Extract agent name from YAML header in agent.md.

        Looks for:
            ---
            name: agent_name
            ---

        Args:
            agent_md_path: Path to agent.md file

        Returns:
            Agent name with 'agent_' prefix (defaults to 'agent_new' if not found)
        """
        if not agent_md_path.exists():
            logger.warning(f"Agent file not found: {agent_md_path}, using default 'agent_new'")
            return 'agent_new'

        try:
            with open(agent_md_path, 'r') as f:
                in_header = False
                for line in f:
                    line = line.strip()
                    if line == '---':
                        if not in_header:
                            in_header = True
                        else:
                            # End of header, name not found
                            break
                    elif in_header and line.startswith('name:'):
                        # Extract name after 'name:'
                        name = line.split('name:', 1)[1].strip()
                        # Add agent_ prefix if not present
                        if not name.startswith('agent_'):
                            name = f'agent_{name}'
                        return name
        except Exception as e:
            logger.warning(f"Error extracting agent name from {agent_md_path}: {e}")

        # Default if name not found
        logger.warning("Agent name not found in YAML header, using default 'agent_new'")
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

        logger.info(f"Generating new vs baseline analysis for test workspace")

        # Paths
        script_path = Path(__file__).parent / "tools" / "error_analysis" / "create_deep_focus_error_index.py"
        error_index_path = test_workspace / "error_index.json"
        error_report_path = test_workspace / "error_analysis_report.md"

        try:
            # Run create_deep_focus_error_index.py with both directories
            cmd = [
                sys.executable,
                str(script_path),
                "--iteration-dirs", f"{baseline_iter},{test_workspace}",
                "--output", str(error_index_path),
                "--new-agent", new_agent_name
            ]
            # Add --flat-domain for non-hierarchical domains (e.g., CodeGen)
            if self.domain and not self.domain.is_hierarchical:
                cmd.append("--flat-domain")

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

                # Summary section
                summary = index.get('summary', {})
                newest_agent = summary.get('new_agent', 'unknown')
                baseline_agents = summary.get('baseline_agents', [])
                # Get accuracies from by_agent dict
                by_agent = index.get('by_agent', {})
                accuracies = {name: stats.get('accuracy', 0) for name, stats in by_agent.items()}

                report_lines.append(f"**New Agent**: {newest_agent} ({accuracies.get(newest_agent, 0)}%)")
                if baseline_agents:
                    baseline_strs = [f"{a} ({accuracies.get(a, 0)}%)" for a in baseline_agents]
                    report_lines.append(f"**Baselines**: {', '.join(baseline_strs)}")
                report_lines.append("")

                # Cross-agent analysis
                cross_agent = index.get('cross_agent_analysis', {}).get('new_vs_baseline', {})
                unique_successes = cross_agent.get('unique_successes', [])
                unique_errors = cross_agent.get('unique_errors', [])
                consensus_errors = cross_agent.get('consensus_errors', [])
                mixed = cross_agent.get('mixed_results', [])

                report_lines.extend([
                    "## Summary",
                    f"- Unique successes (new succeeded, all baselines failed): {len(unique_successes)}",
                    f"- Unique errors (new failed, all baselines succeeded): {len(unique_errors)}",
                    f"- Consensus errors (all failed): {len(consensus_errors)}",
                    f"- Mixed results: {len(mixed)}",
                    ""
                ])

                # Per-database breakdown
                by_database = index.get('by_database', {})

                # Accuracy by Database and Agent table
                if by_database:
                    all_agents = [newest_agent] + baseline_agents

                    report_lines.extend([
                        "## Accuracy by Database and Agent",
                        ""
                    ])

                    # Header row
                    header = "| Database |"
                    separator = "|----------|"
                    for agent in all_agents:
                        header += f" {agent} |"
                        separator += "--------|"

                    report_lines.append(header)
                    report_lines.append(separator)

                    # Data rows (sorted alphabetically by database)
                    for db_name in sorted(by_database.keys()):
                        db_stats = by_database[db_name]
                        agent_stats = db_stats.get('agent_stats', {})

                        row = f"| {db_name} |"
                        for agent in all_agents:
                            if agent in agent_stats:
                                accuracy = agent_stats[agent].get('accuracy', 0.0)
                                row += f" {accuracy:.1f}% |"
                            else:
                                row += " - |"

                        report_lines.append(row)

                    report_lines.append("")

                # 1. Unique Successes by database (new succeeded, baselines failed)
                if by_database:
                    db_unique_successes = []
                    for db_name, db_stats in by_database.items():
                        cross_analysis = db_stats.get('cross_agent_analysis', {}).get('new_vs_baseline', {})
                        success_ids = cross_analysis.get('unique_successes', [])
                        if success_ids:
                            db_unique_successes.append((db_name, success_ids))

                    if db_unique_successes:
                        db_unique_successes.sort(key=lambda x: len(x[1]), reverse=True)

                        report_lines.extend([
                            "## Unique Successes by Database",
                            "",
                            "New agent succeeded but all baselines failed:",
                            "",
                            "| Database | Count | Question IDs |",
                            "|----------|-------|--------------|"
                        ])

                        for db_name, success_ids in db_unique_successes:
                            count = len(success_ids)
                            if count <= 10:
                                id_str = ', '.join(success_ids)
                            else:
                                id_str = ', '.join(success_ids[:10]) + ', ...'
                            report_lines.append(f"| {db_name} | {count} | {id_str} |")

                        report_lines.append("")

                # 2. Unique Errors by database (new failed, baselines succeeded)
                if by_database:
                    db_unique_errors = []
                    for db_name, db_stats in by_database.items():
                        cross_analysis = db_stats.get('cross_agent_analysis', {}).get('new_vs_baseline', {})
                        error_ids = cross_analysis.get('unique_errors', [])
                        if error_ids:
                            db_unique_errors.append((db_name, error_ids))

                    if db_unique_errors:
                        db_unique_errors.sort(key=lambda x: len(x[1]), reverse=True)

                        report_lines.extend([
                            "## Unique Errors by Database",
                            "",
                            "New agent failed but all baselines succeeded:",
                            "",
                            "| Database | Count | Question IDs |",
                            "|----------|-------|--------------|"
                        ])

                        for db_name, error_ids in db_unique_errors:
                            count = len(error_ids)
                            if count <= 10:
                                id_str = ', '.join(error_ids)
                            else:
                                id_str = ', '.join(error_ids[:10]) + ', ...'
                            report_lines.append(f"| {db_name} | {count} | {id_str} |")

                        report_lines.append("")

                # 3. Consensus errors by database (all failed including new)
                if by_database:
                    db_consensus_errors = []
                    for db_name, db_stats in by_database.items():
                        cross_analysis = db_stats.get('cross_agent_analysis', {}).get('new_vs_baseline', {})
                        consensus_ids = cross_analysis.get('consensus_errors', [])
                        if consensus_ids:
                            db_consensus_errors.append((db_name, consensus_ids))

                    if db_consensus_errors:
                        db_consensus_errors.sort(key=lambda x: len(x[1]), reverse=True)

                        report_lines.extend([
                            "## Consensus Errors by Database",
                            "",
                            "New agent AND all baselines failed:",
                            "",
                            "| Database | Count | Question IDs |",
                            "|----------|-------|--------------|"
                        ])

                        for db_name, error_ids in db_consensus_errors:
                            count = len(error_ids)
                            if count <= 10:
                                id_str = ', '.join(error_ids)
                            else:
                                id_str = ', '.join(error_ids[:10]) + ', ...'
                            report_lines.append(f"| {db_name} | {count} | {id_str} |")

                        report_lines.append("")

                # Mixed results by database
                if by_database:
                    db_mixed = []
                    for db_name, db_stats in by_database.items():
                        cross_analysis = db_stats.get('cross_agent_analysis', {}).get('new_vs_baseline', {})
                        mixed_results = cross_analysis.get('mixed_results', {})
                        if mixed_results:
                            db_mixed.append((db_name, mixed_results))

                    if db_mixed:
                        total_mixed = sum(len(mixed_results) for _, mixed_results in db_mixed)
                        db_mixed.sort(key=lambda x: len(x[1]), reverse=True)

                        report_lines.extend([
                            "## Mixed Results by Database",
                            "",
                            f"Total mixed results: {total_mixed}",
                            ""
                        ])

                        from collections import defaultdict
                        for db_name, mixed_results in db_mixed:
                            report_lines.extend([
                                f"### {db_name} ({len(mixed_results)} mixed results)",
                                ""
                            ])

                            # Group by pattern (new_agent_correct, correct baselines, wrong baselines)
                            pattern_groups = defaultdict(list)

                            for question_id, split_info in mixed_results.items():
                                new_agent_correct = split_info.get('new_agent_correct', None)
                                correct_baselines = tuple(sorted(split_info.get('baseline_correct', [])))
                                wrong_baselines = tuple(sorted(split_info.get('baseline_wrong', [])))

                                pattern = (new_agent_correct, correct_baselines, wrong_baselines)
                                pattern_groups[pattern].append(question_id)

                            # Sort patterns by count (most common first), then alphabetically
                            sorted_patterns = sorted(
                                pattern_groups.items(),
                                key=lambda x: (-len(x[1]), x[0])
                            )

                            for (new_agent_correct, correct_baselines, wrong_baselines), question_ids in sorted_patterns:
                                new_agent_status = "✅ NEW" if new_agent_correct else "❌ NEW"
                                correct_str = ', '.join(correct_baselines)
                                wrong_str = ', '.join(wrong_baselines)
                                question_ids_str = ', '.join(f"**{qid}**" for qid in sorted(question_ids))

                                report_lines.append(f"- {new_agent_status} | ✓ {correct_str} | ✗ {wrong_str}: {question_ids_str}")

                            report_lines.append("")

                report_lines.extend([
                    "## Extract Details",
                    "",
                    "To extract detailed error information for specific questions:",
                    "```bash",
                    "python RoboPhD/tools/error_analysis/extract_error_details.py \\",
                    f"  --question-ids 'ID1,ID2' \\",
                    f"  --iteration-dir '{test_workspace.name}'",
                    "```",
                    "",
                    "The tool returns complete evaluation data including SQL, results, and error details.",
                    ""
                ])

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
        - agent.md (database analysis agent)
        - eval_instructions.md (SQL generation instructions)
        - tools/ (optional analysis scripts)

        Saves snapshot after completion.

        Args:
            evolution_prompt: Strategy-specific prompt from evolution strategy
            continue_from_analyzer: Whether to use --continue (if error analyzer ran)
        """
        logger.info("Executing Round 1: Analysis, planning, and implementation")

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

**For detailed question analysis, use the CLI tool:**

```bash
python RoboPhD/tools/error_analysis/extract_error_details.py \\
  --question-ids '1234,5678,9012' \\
  --iteration-dir '../../iteration_{self.current_iteration-1:03d}'
```

The tool returns complete evaluation data including SQL queries, predicted results, ground truth,
verification attempts, and error details for each question across all agents.

Create a file called `reasoning.md` with your analysis and plan.

### Step 2: Implementation

Based on your analysis and plan in `reasoning.md`, create the agent artifacts:

1. `agent.md` - Database analysis agent with model configuration
2. `eval_instructions.md` - Direct SQL generation instructions for eval model
3. `tools/` - Optional Python/shell analysis scripts (if needed)

Create these artifacts in the current directory.

After completing both steps, respond with: "ROUND 1 COMPLETE"
"""

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

        logger.info(f"✓ reasoning.md created ({reasoning_file.stat().st_size} bytes)")

        # Verify agent artifacts were created
        agent_md = self.working_dir / "agent.md"
        eval_instructions_md = self.working_dir / "eval_instructions.md"

        if not agent_md.exists():
            description = "This file should contain the database analysis agent instructions."
            if not self._recover_missing_file(agent_md, "agent.md", description):
                raise RuntimeError("Round 1 did not create agent.md")

        if not eval_instructions_md.exists():
            description = "This file should contain the SQL generation instructions for the eval model."
            if not self._recover_missing_file(eval_instructions_md, "eval_instructions.md", description):
                raise RuntimeError("Round 1 did not create eval_instructions.md")

        logger.info(f"✓ Round 1 complete: reasoning.md and agent artifacts created")
        logger.info(f"  - agent.md: {agent_md.stat().st_size} bytes")
        logger.info(f"  - eval_instructions.md: {eval_instructions_md.stat().st_size} bytes")

        tools_dir = self.working_dir / "tools"
        if tools_dir.exists():
            tool_count = len(list(tools_dir.iterdir()))
            logger.info(f"  - tools/: {tool_count} files")

        # Save Round 1 snapshot
        self._save_snapshot(round_num=1)

    def _load_baseline_problems(
        self,
        test_iteration: int,
        contexts: List[str],
        questions_per_context: int
    ) -> Dict[str, List[Dict]]:
        """
        Load problems for each context, reusing baseline question IDs where available.

        For Text2SQL: Loads question IDs from baseline evaluation.json files
        For CodeGen: Loads problem data from domain

        Args:
            test_iteration: Iteration number to get baseline questions from
            contexts: List of context names (database names or problem_ids)
            questions_per_context: Max questions per context (fallback if no baseline)

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
            elif questions_per_context and len(context_problems) > questions_per_context:
                # Fallback: sample randomly if no baseline available
                logger.warning(f"    Could not load baseline questions for {context_name}, sampling {questions_per_context} randomly")
                problems_by_context[context_name] = random.sample(context_problems, questions_per_context)
            else:
                # Use all available problems
                problems_by_context[context_name] = list(context_problems)

        return problems_by_context

    def _test_and_refine_round(
        self,
        round_num: int,
        test_iteration: int,
        contexts: List[str],
        problems_per_context: int,
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
            problems_per_context: Problems per context
        """
        logger.info(f"Round {round_num}: Testing against iteration {test_iteration}")

        # Create test workspace
        test_workspace = self.working_dir / f"iteration_{test_iteration:03d}_test"
        test_workspace.mkdir(parents=True, exist_ok=True)

        logger.info(f"Test workspace: {test_workspace}")

        # Load current agent artifacts (from working_dir top level)
        agent_md = self.working_dir / "agent.md"
        eval_instructions_md = self.working_dir / "eval_instructions.md"
        tools_dir = self.working_dir / "tools"

        if not agent_md.exists() or not eval_instructions_md.exists():
            raise RuntimeError(f"Agent artifacts not found for Round {round_num}")

        # Extract agent name from agent.md
        agent_name = self._extract_agent_name_from_md(agent_md)
        logger.info(f"Testing agent: {agent_name}")

        # Create new agent directory outside git repo (prevents CLAUDE.md contamination for coder/critic calls).
        # Symlink from evolution tree for easy browsing.
        runs_dir = Path(self.config.get('runs_directory', '../robophd_runs'))
        run_name = self.experiment_dir.name if self.experiment_dir else "unknown"
        real_agent_dir = runs_dir / "evolution" / run_name / f"iteration_{test_iteration:03d}_test" / agent_name
        real_agent_dir.mkdir(parents=True, exist_ok=True)

        new_agent_dir_symlink = test_workspace / agent_name
        try:
            new_agent_dir_symlink.symlink_to(real_agent_dir.resolve())
        except FileExistsError:
            pass

        new_agent_dir = real_agent_dir

        # Create symlinks to baseline agents for comparison
        # Use absolute path so symlinks work from any location
        baseline_iter_dir = (self.experiment_dir / f"iteration_{test_iteration:03d}").resolve()
        if baseline_iter_dir.exists():
            baseline_agents = list(baseline_iter_dir.glob("agent_*"))
            logger.info(f"Creating symlinks to {len(baseline_agents)} baseline agents")
            for baseline_agent_dir in baseline_agents:
                symlink_path = test_workspace / baseline_agent_dir.name
                if not symlink_path.exists():
                    try:
                        symlink_path.symlink_to(baseline_agent_dir, target_is_directory=True)
                        logger.info(f"  ✓ Linked {baseline_agent_dir.name}")
                    except Exception as e:
                        logger.warning(f"  ⚠️  Failed to link {baseline_agent_dir.name}: {e}")
        else:
            logger.warning(f"Baseline iteration not found: {baseline_iter_dir}")

        # Run agent testing using domain-agnostic approach
        context_label = self.domain.context_label_plural if self.domain else "contexts"
        logger.info(f"Running agent on {len(contexts)} {context_label} (max concurrent: {self.max_concurrent})...")

        # Create agent package directory with evolved agent artifacts
        agent_package_dir = test_workspace / "agent_package"
        agent_package_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(agent_md, agent_package_dir / "agent.md")
        shutil.copy2(eval_instructions_md, agent_package_dir / "eval_instructions.md")
        if tools_dir.exists():
            package_tools_dir = agent_package_dir / "tools"
            if package_tools_dir.exists():
                shutil.rmtree(package_tools_dir)
            shutil.copytree(tools_dir, package_tools_dir)

        # Load problems for each context, reusing baseline question IDs where available
        problems_by_context = self._load_baseline_problems(
            test_iteration=test_iteration,
            contexts=contexts,
            questions_per_context=problems_per_context
        )

        # Build SampledProblems (same interface as main loop in researcher.py)
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
            'eval_model': self.eval_model,
            'analysis_model': self.analysis_model,
            'phase2_timeout': self.timeout,
            'max_concurrent': self.max_concurrent,
            'verification_retries': self.verification_retries,
            'temperature_strategy': self.temperature_strategy,
            'debug_log_probability': self.debug_log_probability,
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
                    'accuracy': 0.0,
                    'correct': 0,
                    'total': 0,
                }
            results[context_name]['total'] += 1
            if r.get('correct'):
                results[context_name]['correct'] += 1

        # Calculate per-context accuracy
        for context_name in results:
            r = results[context_name]
            r['accuracy'] = (r['correct'] / r['total'] * 100) if r['total'] > 0 else 0.0

        # Accumulate Phase 2 cost
        if eval_result.metadata.get('phase2_cost', 0) > 0:
            if not hasattr(self, '_current_call_costs'):
                self._current_call_costs = []
            phase2_cost_info = {
                'cost': eval_result.metadata.get('phase2_cost', 0),
                'tokens_in': eval_result.metadata.get('phase2_tokens_in', 0),
                'tokens_out': eval_result.metadata.get('phase2_tokens_out', 0),
                'cache_created': 0,
                'cache_read': 0,
                'call_type': 'phase2'
            }
            self._current_call_costs.append(phase2_cost_info)

        overall_accuracy = eval_result.accuracy
        total_correct = eval_result.correct
        total_questions = eval_result.total

        logger.info(f"Overall accuracy: {overall_accuracy:.1f}% ({total_correct}/{total_questions})")

        # Generate standard evaluation report inside agent directory
        report_path = new_agent_dir / "agent_evaluation_report.md"
        self._generate_evaluation_report(results, overall_accuracy, report_path)

        # Generate comparison report (pass domain for domain-agnostic loading)
        comparison_path: Optional[Path] = None

        if self.domain and self.domain.supports_comparison_report:
            comparison_path = self.working_dir / f"iteration_{test_iteration:03d}_comparison.md"
            report_gen = ComparisonReportGenerator(self.experiment_dir, domain=self.domain)
            report_gen.generate_comparison_report(
                test_iteration=test_iteration,
                new_agent_workspace=new_agent_dir,
                databases=contexts,
                output_path=comparison_path
            )
            logger.info(f"📊 Comparison report: {comparison_path}")
        else:
            logger.info("Skipping comparison report (not supported for this domain)")

        # Generate new vs baseline error analysis (ALWAYS - no longer optional)
        logger.info("Generating new vs baseline error analysis...")
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
            report_path=report_path,
            comparison_path=comparison_path
        )

        # Save snapshot after refinement
        self._save_snapshot(round_num=round_num)

        logger.info(f"✓ Round {round_num} complete")

    def _generate_evaluation_report(
        self,
        results: Dict[str, Dict],
        overall_accuracy: float,
        output_path: Path
    ):
        """
        Generate standard evaluation report.

        Args:
            results: Dictionary mapping db_name -> results dict
            overall_accuracy: Overall accuracy percentage
            output_path: Where to save the report
        """
        # Use domain-specific terminology
        context_label = self.domain.context_label if self.domain else "Database"

        report_lines = [
            "# Agent Evaluation Report",
            "",
            "## Overall Performance",
            "",
            f"**Overall Accuracy**: {overall_accuracy:.1f}%",
            "",
            f"## Performance by {context_label}",
            "",
            f"| {context_label} | Accuracy | Correct | Total |",
            "|----------|----------|---------|-------|"
        ]

        # Sort contexts by name
        errors = {}
        tool_only_failures = {}
        for context_name in sorted(results.keys()):
            r = results[context_name]
            if 'error' in r:
                report_lines.append(f"| {context_name} | ERROR | - | - |")
                errors[context_name] = r['error']
            else:
                report_lines.append(
                    f"| {context_name} | {r['accuracy']:.1f}% | {r['correct']} | {r['total']} |"
                )
                # Track tool-only failures (tool failed but agent succeeded)
                if 'tool_only_error' in r:
                    tool_only_failures[context_name] = r['tool_only_error']

        # Add errors section if there were any errors
        if errors:
            context_label_lower = context_label.lower() + "s"  # "databases" or "contexts"
            report_lines.extend([
                "",
                "## Errors",
                "",
                f"The following {context_label_lower} encountered errors during testing:",
                ""
            ])
            for context_name in sorted(errors.keys()):
                report_lines.extend([
                    f"### {context_name}",
                    "```",
                    errors[context_name],
                    "```",
                    ""
                ])

        # Add tool-only failures section if there were any
        if tool_only_failures:
            context_label_lower = context_label.lower() + "s"  # "databases" or "contexts"
            report_lines.extend([
                "",
                "## Tool-Only Failures",
                "",
                f"The following {context_label_lower} had tool-only execution failures but succeeded with agent fallback:",
                ""
            ])
            for context_name in sorted(tool_only_failures.keys()):
                report_lines.extend([
                    f"### {context_name}",
                    "```",
                    tool_only_failures[context_name],
                    "```",
                    ""
                ])

        # Use domain-specific paths in documentation
        context_path_label = "database" if (self.domain is None or self.domain.__class__.__name__ == 'Text2SQLDomain') else "context"
        report_lines.extend([
            "",
            "## Detailed Results",
            "",
            f"Review individual {context_path_label} results in:",
            "```",
            f"./<{context_path_label}>/results/evaluation.json",
            f"./<{context_path_label}>/output/system_prompt.txt" if context_path_label == "database" else f"./<{context_path_label}>/output/",
            "```"
        ])

        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))

    def _prompt_for_refinement(
        self,
        round_num: int,
        test_iteration: int,
        report_path: Path,
        comparison_path: Optional[Path]
    ):
        """
        Prompt Claude Code to refine agent based on test results.

        Uses --continue to maintain context from previous rounds.

        Args:
            round_num: Current round number
            test_iteration: Iteration that was tested against
            report_path: Path to evaluation report
            comparison_path: Path to comparison report (None if not supported)
        """
        # Read reports
        with open(report_path) as f:
            eval_report = f.read()

        comparison_report = ""
        if comparison_path and comparison_path.exists():
            with open(comparison_path) as f:
                comparison_report = f.read()

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

**For detailed question analysis, use the CLI tool:**

```bash
python RoboPhD/tools/error_analysis/extract_error_details.py \\
  --question-ids '1234,5678' \\
  --iteration-dir '{test_workspace_name}'
```

The tool returns complete evaluation data including SQL queries, predicted results, ground truth,
verification attempts, and error details. All agents from the baseline iteration are accessible
via symlinks in the test workspace.
"""

        comparison_section = ""
        if comparison_report:
            comparison_section = f"""
### Comparison Report
{comparison_report}
"""

        prompt = f"""
## Round {round_num}: Testing and Refinement

Your agent was tested on data from iteration {test_iteration}.

### Standard Evaluation Report
{eval_report}
{comparison_section}{error_analysis_section}
### Your Task
Review the performance results above and the generated outputs in the test workspace:
- Evaluation results: `./iteration_{test_iteration:03d}_test/<database>/results/evaluation.json`
- Database analysis outputs: `./iteration_{test_iteration:03d}_test/<database>/output/agent_output.txt`

Based on what you observe, provide updated versions of any artifacts that need changes
(agent.md, eval_instructions.md, or tools/).

After refinements, respond with: "ROUND {round_num} COMPLETE"
"""

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

        # Get LM Studio env overrides for non-Anthropic evolution models
        extra_env = get_lmstudio_env(self.evolution_model)

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

        # Add MCP config for error analysis tools
        # Path to MCP config relative to project root
        project_root = self.experiment_dir.parent
        mcp_config_path = project_root / "RoboPhD" / "mcp_configs" / "error_analysis_tools.json"
        if mcp_config_path.exists():
            cmd.extend(["--mcp-config", str(mcp_config_path)])

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
            "--settings", '{"autoCompact": true}'  # Proactively compact when context gets low
        ])

        logger.debug(f"Calling Claude Code: {' '.join(cmd[:4])}...")

        # Get LM Studio env overrides for non-Anthropic evolution models
        extra_env = get_lmstudio_env(self.evolution_model)

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

                        # Add MCP config if present
                        if mcp_config_path.exists():
                            retry_cmd.extend(["--mcp-config", str(mcp_config_path)])

                        # Use --resume for retry (session already exists)
                        retry_cmd.extend(["--resume", self.session_id])

                        retry_cmd.extend([
                            "--print", prompt,
                            "--output-format", "json",
                            "--permission-mode", "bypassPermissions",
                            "--settings", '{"autoCompact": true}'
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

        except subprocess.TimeoutExpired:
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
        For test rounds, separates phase1 (DB analysis), phase2 (SQL generation), and evolution (refinement) costs.

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
                # Accumulate into nested structure (phase1 or evolution)
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
            "Claude CLI not found. Install with: pip install claude-cli"
        )

    def _save_snapshot(self, round_num: int):
        """
        Save snapshot of agent artifacts after a round completes.

        Args:
            round_num: Round number (2, 3, 4, ...)
        """
        snapshot_dir = self.working_dir / f"round{round_num}_snapshot"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Copy artifacts from top level to snapshot
        agent_md = self.working_dir / "agent.md"
        eval_instructions_md = self.working_dir / "eval_instructions.md"
        tools_dir = self.working_dir / "tools"

        if agent_md.exists():
            shutil.copy2(agent_md, snapshot_dir / "agent.md")

        if eval_instructions_md.exists():
            shutil.copy2(eval_instructions_md, snapshot_dir / "eval_instructions.md")

        if tools_dir.exists() and tools_dir.is_dir():
            tools_snapshot = snapshot_dir / "tools"
            if tools_snapshot.exists():
                shutil.rmtree(tools_snapshot)
            shutil.copytree(tools_dir, tools_snapshot)

        logger.info(f"Saved Round {round_num} snapshot to {snapshot_dir}")

    def _save_final_artifacts(self) -> Tuple[Path, Path, Optional[Path]]:
        """
        Save final version of artifacts at top level.

        Final artifacts are copies of the last snapshot.

        Returns:
            Tuple of (agent.md path, eval_instructions.md path, tools/ path)
        """
        # Artifacts should already be at top level from last round
        # This method just verifies they exist and returns their paths

        agent_md = self.working_dir / "agent.md"
        eval_instructions_md = self.working_dir / "eval_instructions.md"
        tools_dir = self.working_dir / "tools"

        if not agent_md.exists():
            description = "This is the final database analysis agent file."
            if not self._recover_missing_file(agent_md, "agent.md", description):
                raise RuntimeError("Final agent.md not found")

        if not eval_instructions_md.exists():
            description = "This is the final SQL generation instructions file."
            if not self._recover_missing_file(eval_instructions_md, "eval_instructions.md", description):
                raise RuntimeError("Final eval_instructions.md not found")

        tools_path = tools_dir if tools_dir.exists() else None

        logger.info("Final artifacts verified:")
        logger.info(f"  - agent.md: {agent_md}")
        logger.info(f"  - eval_instructions.md: {eval_instructions_md}")
        if tools_path:
            logger.info(f"  - tools/: {tools_path}")

        return (agent_md, eval_instructions_md, tools_path)

    def _save_session_transcript(self):
        """
        Summarize Claude Code session transcript and save to evolution output directory.

        Reads the session chat.jsonl file from ~/.claude/projects/[session_id]/
        and produces a human/LLM-readable session_summary.md with the full
        narrative of assistant reasoning and one-line tool call summaries.

        Side effects:
            - Creates session_summary.md in working_dir

        Errors are logged but do not raise exceptions - transcript saving
        should never break the research run.
        """
        # Derive project directory name from working directory
        # Claude CLI sanitizes path: /foo/bar_baz → -foo-bar-baz (both / and _ become -)
        project_dir_name = str(self.working_dir.resolve()).replace('/', '-').replace('_', '-')

        session_dir = Path.home() / ".claude" / "projects" / project_dir_name
        chat_file = session_dir / f"{self.session_id}.jsonl"

        if not chat_file.exists():
            logger.warning(f"Session transcript not found: {chat_file}")
            logger.warning(f"Session ID: {self.session_id}")
            return

        try:
            from RoboPhD.utilities.transcript_summarizer import summarize_transcript

            summary_path = summarize_transcript(chat_file, self.working_dir / "session_summary.md")
            summary_size = summary_path.stat().st_size
            logger.info(f"Saved session summary: {summary_path.name} ({summary_size/1024:.1f} KB)")

        except Exception as e:
            logger.warning(f"Failed to save session summary: {e}")
            # Don't raise - transcript saving should never break the research run

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
                    logger.info(f"✓ Evolution reflection saved: {reflection_file.name}")
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
