#!/usr/bin/env python3
"""
RoboPhD Parallel Agent Researcher - Complete Migration from APE
This file contains the full researcher.py implementation for RoboPhD.
Due to size constraints, this will replace the partial researcher.py file.
"""

import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from RoboPhD.domains.base import DomainInterface

# RoboPhD imports - handle both module and script execution
try:
    from .ranking_table import generate_ranking_table, calculate_mean_ranks
    from .config import (
        API_KEY_ENV_VAR,
        CLAUDE_CLI_MODEL_MAP,
        DEFAULT_MODEL,
        SUPPORTED_MODELS
    )
    from .domains.base import SampledProblems
    from .domains import get_domain
    from .evolution import EvolutionStrategySelector
    from .config_manager import ConfigManager, ConfigSource
    from .report_generator import ReportGenerator, is_continuous_scoring
    from .deep_focus_evolution_manager import DeepFocusEvolutionManager
    from .utilities.cached_sql_executor import close_all_connections as close_robophd_connections
    from .eval_utils import EvalRateLimitError
except ImportError:
    # When run as a script, use absolute imports
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from RoboPhD.ranking_table import generate_ranking_table, calculate_mean_ranks
    from RoboPhD.config import (
        API_KEY_ENV_VAR,
        CLAUDE_CLI_MODEL_MAP,
        DEFAULT_MODEL,
        SUPPORTED_MODELS
    )
    from RoboPhD.domains import get_domain
    from RoboPhD.evolution import EvolutionStrategySelector
    from RoboPhD.config_manager import ConfigManager, ConfigSource
    from RoboPhD.report_generator import ReportGenerator, is_continuous_scoring
    from RoboPhD.deep_focus_evolution_manager import DeepFocusEvolutionManager
    from RoboPhD.utilities.cached_sql_executor import close_all_connections as close_robophd_connections
    from RoboPhD.domains.base import SampledProblems
    from RoboPhD.eval_utils import EvalRateLimitError

# Import root-level utilities for evaluation pool cleanup
from utilities.cached_sql_executor import close_all_connections as close_eval_connections

# Utilities
import psutil

# Setup logger
logger = logging.getLogger(__name__)


# Infrastructure errors that indicate system bugs (not agent failures)
# These should abort the run to prevent corrupted/incomplete data
CRITICAL_INFRASTRUCTURE_ERRORS = [
    "dictionary changed size during iteration",  # Threading race condition (now fixed)
    "Too many open files",                        # File descriptor exhaustion
    "Database is locked",                         # SQLite lock contention
    "MemoryError",                                # OOM condition
    "Connection refused",                         # API/network infrastructure
    "No space left on device",                    # Disk space
    "status_code=529",                            # Anthropic API overloaded (transient)
]


class MemoryMonitor:
    """Monitor system memory usage."""
    
    def __init__(self, threshold_percent: float = 80.0):
        self.threshold_percent = threshold_percent
        
    def check_memory(self) -> bool:
        """Check if memory usage is below threshold."""
        memory = psutil.virtual_memory()
        if memory.percent > self.threshold_percent:
            print(f"⚠️ Memory usage high: {memory.percent:.1f}%")
            print(f"   Available: {memory.available / (1024**3):.1f} GB")
            return False
        return True


class StuckProcessReaper:
    """Background daemon that kills orphaned Python subprocesses.

    The Claude CLI evolution session spawns subprocesses via its Bash tool.
    When the CLI times out or is killed, these child processes become orphaned
    and run indefinitely at 100% CPU. This reaper periodically scans for and
    kills any Python process whose cwd is inside the experiment directory
    and that has been running longer than the configured threshold.

    Agent code (agent.py) is safe — it runs via exec() in-process, not as
    a separate OS process.
    """

    def __init__(self, experiment_dir: Path, process_age_threshold: int = 1200):
        self._stop_event = threading.Event()
        self._thread = None
        self._stopped = False
        self._total_killed = 0
        self._experiment_dir = str(Path(experiment_dir).resolve())
        self.process_age_threshold = process_age_threshold
        self.scan_interval = process_age_threshold // 2

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("StuckProcessReaper started (threshold=%ds, interval=%ds)",
                     self.process_age_threshold, self.scan_interval)

    def stop(self):
        if self._stopped:
            return
        self._stopped = True
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._total_killed > 0:
            logger.warning("StuckProcessReaper stopped — killed %d stuck processes total",
                           self._total_killed)
        else:
            logger.info("StuckProcessReaper stopped — no stuck processes found")

    def _run(self):
        while not self._stop_event.is_set():
            self._scan_and_kill()
            self._stop_event.wait(self.scan_interval)

    def _scan_and_kill(self):
        now = time.time()
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
            try:
                cmdline = proc.info.get('cmdline') or []
                if not cmdline:
                    continue
                if not cmdline[0].startswith('python'):
                    continue
                age = now - (proc.info.get('create_time') or now)
                if age <= self.process_age_threshold:
                    continue
                # Check cwd is inside experiment directory
                try:
                    proc_cwd = proc.cwd()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                if not proc_cwd.startswith(self._experiment_dir):
                    continue
                logger.warning("Killing stuck process pid=%d age=%.0fs cmd=%s",
                               proc.pid, age, ' '.join(cmdline[:5]))
                proc.kill()
                self._total_killed += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue


class ParallelAgentEvolver:
    """Manages agent evolution using Claude."""

    def __init__(self, experiment_dir: Path, config: Dict[str, Any], domain: Optional['DomainInterface'] = None):
        """Initialize the evolver with resolved config dict.

        Args:
            experiment_dir: Path to experiment directory
            config: Resolved configuration dict from ConfigManager
                   Contains all parameters including evolution_strategy
            domain: Optional domain interface for domain-specific settings.
                   Used to determine default evolution strategies directory.
        """
        self.experiment_dir = Path(experiment_dir)
        self.domain = domain
        self.config = config  # Store full config for DeepFocusEvolutionManager

        # Extract all parameters from config
        self.evolution_model = config["evolution_model"]
        self.evolution_timeout = config["evolution_timeout"]
        self.agents_directory = config.get("agents_directory")
        self.strategies_directory = config.get("strategies_directory")
        self.new_agent_test_rounds = config["new_agent_test_rounds"]
        self.new_agent_test_round_offset = config["new_agent_test_round_offset"]
        self.max_workers = config["max_workers"]
        self.llm_call_timeout = config["llm_call_timeout"]

        # Evolution tracking
        self.evolution_count = 0
        self.evolution_retries = []
        self.five_hour_limit_incidents = []
        self.restart_from_iteration = None  # Changed from boolean to track specific iteration
        self.evolution_history = []

        # Challenger mode flag (set per-iteration)
        self.use_challenger_selection = False
        # Greedy mode flag (set per-iteration)
        self.use_greedy_selection = False
        self.is_first_evolution_call = True
        
        # Setup paths
        # Evolution strategies will be loaded from experiment directory after strategies are copied
        self.evolution_prompts_dir = self.experiment_dir / "evolution_strategies"
        self.available_strategies = {}
        # Note: _load_evolution_strategies() called after load_initial_strategies() in run()

        # Claude CLI path
        self.claude_path = os.path.expanduser("~/.claude/local/claude")
        if not Path(self.claude_path).exists():
            self.claude_path = "claude"  # Try system PATH


    def _load_evolution_strategies(self):
        """
        Load available evolution strategies from experiment directory.

        Strategies are now two-artifact packages in subdirectories:
        - <experiment_dir>/evolution_strategies/strategy_name/strategy.md
        - <experiment_dir>/evolution_strategies/strategy_name/tools/ (optional)
        """
        if not self.evolution_prompts_dir.exists():
            print(f"⚠️ Evolution strategies directory not found: {self.evolution_prompts_dir}")
            return

        # Scan for directories containing strategy.md
        for strategy_dir in self.evolution_prompts_dir.iterdir():
            if not strategy_dir.is_dir():
                continue

            # Skip special directories
            if strategy_dir.name.startswith('.'):
                continue

            # Check for strategy.md
            strategy_file = strategy_dir / "strategy.md"
            if strategy_file.exists():
                strategy_name = strategy_dir.name
                self.available_strategies[strategy_name] = strategy_file

        print(f"📋 Loaded {len(self.available_strategies)} evolution strategies from {self.evolution_prompts_dir}")

    def list_strategies(self) -> List[str]:
        """List all available evolution strategies."""
        return sorted(list(self.available_strategies.keys()))
    
    def list_all_strategies(self) -> List[str]:
        """List all strategies including special ones."""
        strategies = list(self.available_strategies.keys())
        strategies.append("none")
        strategies.append("weighted_random")
        return sorted(strategies)

    def _get_iteration_contexts(self, iteration: int) -> List[str]:
        """
        Get contexts used in a specific iteration from test_history.

        Args:
            iteration: Iteration number (1-indexed)

        Returns:
            List of context IDs used in that iteration, or empty list if not available
        """
        if iteration < 1 or iteration > len(self.test_history):
            return []

        # Get iteration data (test_history is 0-indexed)
        iteration_data = self.test_history[iteration - 1]
        if not iteration_data:
            return []

        # Get contexts tested from first agent (all agents test same contexts)
        first_agent_data = next(iter(iteration_data.values()))
        # Support both new key and legacy checkpoints
        return first_agent_data.get('contexts_tested') or first_agent_data.get('databases_tested', [])

    def create_new_agent(self,
                        agent_pool: Dict,
                        performance_records: Dict,
                        recent_results: Dict,
                        iteration: int,
                        test_history: List,
                        strategy_name: str,
                        was_random: Union[bool, str] = False) -> Optional[Tuple]:
        """
        Create a new evolved agent using Deep Focus multi-round evolution.

        Args:
            agent_pool: Pool of available agents
            performance_records: Performance records for all agents
            recent_results: Results from previous iteration
            iteration: Current iteration number
            test_history: History of all test results
            strategy_name: Evolution strategy to use (required)
            was_random: Whether strategy was randomly selected

        Returns:
            Tuple of (agent_id, reasoning, package_info) or None if evolution failed
        """
        # Note: strategy_name is now always provided by caller from resolved config

        print(f"\n🧬 DEEP FOCUS EVOLUTION (Iteration {iteration}) | Strategy: {strategy_name} | Test Rounds: {self.new_agent_test_rounds} | Model: {self.evolution_model}")

        # Load evolution strategy
        if strategy_name not in self.available_strategies:
            print(f"❌ Strategy '{strategy_name}' not found")
            return None
        
        strategy_content = self.available_strategies[strategy_name].read_text()
        
        # Build evolution prompt
        prompt = self._build_evolution_prompt(
            strategy_content,
            agent_pool,
            performance_records,
            recent_results,
            iteration,
            test_history
        )

        # Create evolution workspace
        evolution_dir = self.experiment_dir / "evolution_output" / f"iteration_{iteration:03d}"
        evolution_dir.mkdir(parents=True, exist_ok=True)

        # Build database mapping for test rounds
        databases_map = {}
        for test_round in range(self.new_agent_test_rounds):
            test_iteration = iteration + self.new_agent_test_round_offset - test_round
            if test_iteration >= 1:
                databases = self._get_iteration_contexts(test_iteration)
                if databases:
                    databases_map[test_iteration] = databases

        # Deep focus test rounds don't cache, so all evals are fresh
        # All domains use flat evaluation (each context IS one problem)
        self._current_deep_focus_fresh_evals = sum(
            len(dbs) for dbs in databases_map.values()
        )

        # Create Deep Focus Evolution Manager
        # Pass full config so domains get all required fields (coder_model, critic_model, etc.)
        manager = DeepFocusEvolutionManager(
            test_rounds=self.new_agent_test_rounds,
            test_round_offset=self.new_agent_test_round_offset,
            evolution_model=self.evolution_model,
            timeout=self.evolution_timeout,
            max_workers=self.max_workers,
            llm_call_timeout=self.llm_call_timeout,
            domain=self.domain,
            config=self.config
        )

        # Run Deep Focus evolution
        try:
            result = manager.evolve_agent(
                working_dir=evolution_dir,
                experiment_dir=self.experiment_dir,
                current_iteration=iteration,
                evolution_strategy_name=strategy_name,
                evolution_prompt=prompt,
                contexts=databases_map,
            )
        except Exception as e:
            print(f"❌ Deep Focus evolution failed: {e}")
            import traceback
            traceback.print_exc()
            return None


        # Track evolution
        self.evolution_count += 1
        evolution_entry = {
            'iteration': iteration,
            'strategy': strategy_name,
            'timestamp': datetime.now().isoformat(),
            'deep_focus_rounds': 2 + len(databases_map),  # Rounds 1 & 2 + test rounds
            'timing': result.timing_info  # Deep Focus timing breakdown
        }
        if was_random == 'weighted':
            evolution_entry['was_weighted_random'] = True
        elif was_random:
            evolution_entry['was_random'] = True
        self.evolution_history.append(evolution_entry)

        # Read reasoning.md for agent naming and context
        reasoning_file = evolution_dir / "reasoning.md"
        reasoning = reasoning_file.read_text() if reasoning_file.exists() else ""

        # Generate agent ID from reasoning.md content
        agent_id = self._generate_agent_id(reasoning, iteration)

        # Build package info
        package_info = {
            'type': 'three_artifact',
            'artifact_paths': result.artifact_paths,
            'evolution_dir': evolution_dir,
            'timing': result.timing_info,
            'cost': result.cost_info,
            'session_id': result.session_id,
        }

        print(f"✅ Deep Focus evolution complete")
        print(f"   Agent ID: {agent_id}")
        print(f"   Artifacts: {evolution_dir}")
        print(f"   Evolution time: {result.timing_info['total']/60:.1f} minutes")
        print(f"   Evolution cost: ${result.cost_info['total']:.2f}")

        return (agent_id, reasoning, package_info)
    
    def _substitute_template_variables(self, text: str, iteration: int) -> str:
        """
        Substitute template variables in evolution strategy text.

        Args:
            text: Strategy text with template variables
            iteration: Current iteration number

        Returns:
            Text with variables substituted
        """
        # Template variables
        variables = {
            'iteration': str(iteration),
            'previous_iteration': str(iteration - 1),
            'experiment_dir': str(self.experiment_dir),
        }

        # Simple variable substitution
        result = text
        for var_name, var_value in variables.items():
            result = result.replace(f'{{{var_name}}}', var_value)

        return result

    def _build_evolution_prompt(self,
                               strategy_content: str,
                               agent_pool: Dict,
                               performance_records: Dict,
                               recent_results: Dict,
                               iteration: int,
                               test_history: List) -> str:
        """Build the complete evolution prompt."""
        lines = []

        # Add performance rankings and agent pool (can be disabled to reduce ELO-leader fixation)
        include_rankings = self.config.get("include_evolution_rankings", True)
        if include_rankings:
            lines.append("## Performance Rankings Across All Iterations\n")
            ranking_table = self._generate_ranking_table(test_history, performance_records, for_evolution=True)
            lines.append(ranking_table)

        # Add previous iteration summary with per-database breakdown
        lines.append("\n")
        summary = self._get_previous_iteration_summary(iteration - 1, test_history)
        lines.append(summary)

        # Add experiment structure
        lines.append("\n## Experiment Directory Structure\n")
        structure = self._get_experiment_structure(iteration - 1)
        lines.append(structure)

        # Add agent pool summary
        if include_rankings:
            lines.append("\n## Agent Pool\n")
            pool_summary = self._format_agent_pool_summary(agent_pool, performance_records)
            lines.append(pool_summary)

        # Substitute template variables in strategy content
        strategy_with_vars = self._substitute_template_variables(strategy_content, iteration)

        # NOW add strategy content after context is established
        lines.append("\n## Evolution Strategy: " + strategy_with_vars.split('\n')[0].strip('#').strip())
        lines.append("\n" + strategy_with_vars)
        
        # Add output requirements for file creation (driven by domain file_mapping)
        lines.append("\n## OUTPUT REQUIREMENTS\n")
        lines.append(f"Create the following files in evolution_output/iteration_{iteration:03d}/:\n")
        lines.append("1. **reasoning.md** - Your analysis and improvement strategy")
        lines.append("   Must include a `Name:` line (e.g. `Name: my-agent-name`) for agent identification.\n")

        # List required artifacts from file_mapping
        file_mapping = getattr(self.domain, 'file_mapping', {})
        task_objective = getattr(self.domain, 'task_objective', '')
        artifact_num = 2
        for key, path in sorted(file_mapping.items()):
            description = key.replace("_", " ")
            lines.append(f"{artifact_num}. **{path}** - {description}")
            artifact_num += 1

        if task_objective:
            lines.append(f"\n**Task objective**: {task_objective}")
        
        return "\n".join(lines)
    
    def _generate_ranking_table(self, test_history: List, performance_records: Dict, for_evolution: bool = False, clone_agent_ids: set = None) -> str:
        """Generate comprehensive ranking table for agents across all iterations."""
        return generate_ranking_table(test_history, performance_records, for_evolution, clone_agent_ids=clone_agent_ids)
    
    def _calculate_mean_ranks(self, records: Dict) -> Dict[str, float]:
        """Calculate mean average rank for each agent across iterations."""
        return calculate_mean_ranks(records)
    
    def _get_experiment_structure(self, iteration: int) -> str:
        """Get a structured overview of experiment files for analysis."""
        lines = []
        iter_dir = self.experiment_dir / f"iteration_{iteration:03d}"

        if not iter_dir.exists() or iteration < 1:
            return "No previous iteration data available yet."

        lines.append("Experiment directory structure (paths relative to evolution workspace):")
        lines.append("")

        agent_dirs = sorted(iter_dir.glob("agent_*"))
        if not agent_dirs:
            return "No agent data available yet."

        # Use domain-specific structure documentation
        structure_docs = self.domain.experiment_structure_docs
        # Replace XXX with actual iteration number
        structure_docs = structure_docs.replace("iteration_XXX", f"iteration_{iteration:03d}")
        lines.append(structure_docs)
        lines.append("")

        # List agents tested
        agent_names = [agent_dir.name for agent_dir in agent_dirs]
        lines.append(f"**Agents tested ({len(agent_names)}):**")
        for agent_name in agent_names:
            clean_name = agent_name.replace('agent_', '')
            lines.append(f"- {agent_name} (source: ../../agents/{clean_name}/)")

        return "\n".join(lines)
    
    def _get_previous_iteration_summary(self, iteration: int, test_history: List) -> str:
        """Get performance breakdown for the previous iteration."""
        if iteration < 1 or not test_history or iteration > len(test_history):
            return "## Previous Iteration Results\n\nNo previous iteration data available yet."

        prev_results = test_history[iteration - 1]  # test_history is 0-indexed

        lines = [f"## Previous Iteration Results (Iteration {iteration})"]
        lines.append("")

        if not prev_results:
            lines.append("No results available.")
            return "\n".join(lines)

        agents = sorted(prev_results.keys())

        # Agent score table
        lines.append("### Agent Scores")
        lines.append("")
        lines.append("| Agent | Score | Score Sum / Total |")
        lines.append("|-------|-------|-------------------|")

        for agent_id in agents:
            agent_data = prev_results[agent_id]
            average_score = agent_data.get('average_score', 0.0)
            score_sum = agent_data.get('score_sum', 0)
            total = agent_data.get('total', 0)
            agent_display = agent_id[:30] + "..." if len(agent_id) > 30 else agent_id
            lines.append(f"| {agent_display} | {average_score:.3f} | {score_sum:.1f}/{total} |")

        lines.append("")

        # Failed problem IDs from evaluation.json (skip for continuous-score domains)
        if self.domain:
            # Single pass: collect scores and per-agent failed lists
            scores_by_question: Dict[str, Dict[str, float]] = {}
            failed_by_agent: Dict[str, List[str]] = {}
            for agent_id in agents:
                iter_agent_dir = self.experiment_dir / f"iteration_{iteration:03d}" / f"agent_{agent_id}"
                eval_file = iter_agent_dir / "evaluation.json"
                if eval_file.exists():
                    try:
                        with open(eval_file, 'r') as f:
                            eval_data = json.load(f)
                        agent_failed = []
                        for pid, r in eval_data.get('results', {}).items():
                            score = r.get('score', 0)
                            if pid not in scores_by_question:
                                scores_by_question[pid] = {}
                            scores_by_question[pid][agent_id] = score
                            if score < 0.5:
                                agent_failed.append(pid)
                        if agent_failed:
                            failed_by_agent[agent_id] = agent_failed
                    except Exception:
                        pass

            if not is_continuous_scoring(scores_by_question):
                for agent_id, failed in failed_by_agent.items():
                    lines.append(f"**{agent_id}** failed problems ({len(failed)}): {', '.join(failed[:20])}")
                    if len(failed) > 20:
                        lines.append(f"  ... and {len(failed) - 20} more")

            lines.append("")

        return "\n".join(lines)
    
    def _format_agent_pool_summary(self, agent_pool: Dict, performance_records: Dict) -> str:
        """Format agent pool summary."""
        lines = []
        for agent_id in sorted(agent_pool.keys()):
            perf = performance_records.get(agent_id, {})
            elo_score = perf.get('elo', 1500)
            lines.append(f"- {agent_id}: {perf.get('mean_score', 0):.3f} (ELO: {elo_score:.0f})")
        return "\n".join(lines)

    def _generate_agent_id(self, content: str, iteration: int) -> str:
        """
        Generate ID for agent based on reasoning.md content.

        Looks for a line like:
            Name: my-agent-name
            ## Agent Name: my-agent-name
            name: my-agent-name  (legacy YAML frontmatter)
        """
        for line in content.split('\n'):
            stripped = line.strip().lstrip('#').strip()
            # Match "Name: value" or "Agent Name: value" (case-insensitive)
            match = re.match(r'^(?:agent\s+)?name:\s*(.+)', stripped, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Clean name for filesystem
                name = name.replace("-", "_").replace(" ", "_")
                # Strip any existing iter prefix (current or stale)
                name = re.sub(r'^iter\d+_', '', name)
                if name:
                    return f"iter{iteration}_{name}"

        # Fallback to generic name
        return f"iter{iteration}_evolved_{int(time.time() % 10000)}"


# Continue in next part due to size...

class ParallelAgentResearcher:
    """Research system for evolving database analysis agents."""
    
    def __init__(self,
                 config_manager: ConfigManager,
                 num_iterations: int,
                 random_seed: Optional[int] = None,
                 resume_mode: bool = False,
                 resume_from_iteration: Optional[int] = None,
                 resume_checkpoint: Optional[Dict] = None,
                 resume_experiment_dir: Optional[Path] = None,
                 dev_eval_mode: bool = False,
                 custom_experiment_name: Optional[str] = None,
                 api_key: Optional[str] = None,
                 runtime_config: Optional[Dict] = None,
                 task_config: Optional[Dict] = None):
        """
        Initialize the parallel agent researcher.

        Args:
            config_manager: ConfigManager instance with all configuration
            num_iterations: Number of iterations to run (CLI-only parameter)
            resume_mode: Whether resuming from checkpoint
            resume_from_iteration: Specific iteration to restart from
            resume_checkpoint: Checkpoint data if resuming
            resume_experiment_dir: Experiment directory if resuming
            dev_eval_mode: Whether running dev evaluation
            custom_experiment_name: Custom name for experiment
            api_key: API key for SQL generation
            runtime_config: Non-serializable domain config (e.g. evaluator_fn,
                dataset, file_mapping). Merged into domain config in _load_data().
                Never serialized to checkpoint — reconstructed on resume.
            task_config: Task-level config (split, model, cost_budget, etc.).
                Persisted to checkpoint for reproducibility and resume.
        """
        # Store config manager and CLI-only params
        self.config_manager = config_manager
        self.num_iterations = num_iterations
        self.resume_mode = resume_mode
        self.resume_from_iteration = resume_from_iteration
        self.runtime_config = runtime_config or {}
        self.task_config = task_config if task_config is not None else (
            resume_checkpoint.get('task_config', {}) if resume_checkpoint else {}
        )

        # Required for Text2SQL (SQL generation via API), optional for external domain
        # (which uses Claude Code CLI subprocesses instead).
        self.api_key = api_key

        # Get iteration 1 config for initialization
        config = config_manager.get_config(1)

        # Extract all parameters from config
        self.domain_name = config.get("domain", "external")
        self.dataset = config["dataset"]
        self.examples_per_iteration = config["examples_per_iteration"]
        self.agents_per_iteration = config["agents_per_iteration"]
        self.evolution_model = config["evolution_model"]
        self.max_workers = config["max_workers"]
        self.evolution_timeout = config["evolution_timeout"]
        self.llm_call_timeout = config["llm_call_timeout"]
        self.new_agent_test_rounds = config["new_agent_test_rounds"]
        self.new_agent_test_round_offset = config["new_agent_test_round_offset"]
        self.agents_directory = config["agents_directory"]
        self.strategies_directory = config.get("strategies_directory")


        # Handle resume vs fresh start
        if resume_mode:
            # Restore state from checkpoint
            self.experiment_dir = resume_experiment_dir
            self.agent_pool = self._restore_agent_pool(
                resume_checkpoint['agent_pool'],
                from_iteration=resume_from_iteration
            )
            self.performance_records = resume_checkpoint['performance_records']

            # Migrate old checkpoints: add tracking fields if missing
            for agent_id, perf in self.performance_records.items():
                if 'last_win_iteration' not in perf:
                    perf['last_win_iteration'] = None
                if 'last_test_iteration' not in perf:
                    perf['last_test_iteration'] = None

            self.test_history = resume_checkpoint['test_history']
            self.iteration_times = resume_checkpoint.get('iteration_times', [])
            self.iteration_claude_costs = resume_checkpoint.get('iteration_claude_costs', [])
            self.iteration_fresh_evals = resume_checkpoint.get('iteration_fresh_evals', [])
            self.evolution_times = resume_checkpoint.get('evolution_times', [])
            self.meta_evolution_times = resume_checkpoint.get('meta_evolution_times', [])
            self.zero_accuracy_cases = [tuple(e) for e in resume_checkpoint.get('zero_accuracy_cases', [])]
            self.exception_failures = [tuple(e) for e in resume_checkpoint.get('exception_failures', [])]
            self.five_hour_limit_incidents = [tuple(e) for e in resume_checkpoint.get('five_hour_limit_incidents', [])]
            self.clone_detections = [tuple(e) for e in resume_checkpoint.get('clone_detections', [])]
            self.current_iteration_evolution_cost = None

            # Validate checkpoint integrity before any mutation
            self.original_seed = resume_checkpoint.get("random_seed")
            if self.original_seed is None:
                raise ValueError(
                    "Checkpoint missing random_seed at root level. "
                    "Cannot resume - checkpoint may be corrupted."
                )

            last_completed = resume_checkpoint.get('last_completed_iteration', len(resume_checkpoint.get('test_history', [])))
            # Always archive partial work from crashed iterations
            self.archive_iterations(resume_from_iteration or (last_completed + 1))
            # Only restore performance tracking when explicitly rewinding with --from-iteration
            # On auto-resume, the checkpoint already has correct data; rebuilding is harmful
            # because it can incorrectly grant clone agents pending-winner status
            if resume_from_iteration and resume_from_iteration <= last_completed:
                self._restore_performance_tracking_before_iteration(resume_from_iteration)

            current_iteration = resume_from_iteration if resume_from_iteration else last_completed + 1
            self.random_seed = (self.original_seed + current_iteration * 10000) % (2**32)
            random.seed(self.random_seed)
            print(f"🎲 Resume seed: {self.random_seed}")
        else:
            # Fresh start initialization
            # Setup random seed (from parameter, set by main())
            if random_seed is not None:
                self.original_seed = random_seed
                self.random_seed = random_seed
                random.seed(random_seed)
            else:
                # Generate random seed
                self.random_seed = random.randint(0, 10000)
                random.seed(self.random_seed)
                self.original_seed = self.random_seed

            # Setup experiment directory
            if dev_eval_mode and custom_experiment_name:
                self.experiment_dir = Path("robophd_evaluation") / custom_experiment_name
            else:
                runs_dir_cfg = (runtime_config or {}).get("runs_dir")
                if runs_dir_cfg:
                    runs_dir = Path(runs_dir_cfg)
                else:
                    runs_dir = Path("../robophd_runs")
                task_name = runtime_config.get("task_name", "unknown") if runtime_config else "unknown"
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.experiment_dir = runs_dir / "robophd" / f"{task_name}_{timestamp}"
            self.experiment_dir.mkdir(parents=True, exist_ok=True)

            # Set evaluator debug_log_dir now that experiment_dir is known
            evaluator_fn = self.runtime_config.get("evaluator_fn")
            if evaluator_fn and hasattr(evaluator_fn, "debug_log_dir") and evaluator_fn.debug_log_dir is None:
                evaluator_fn.debug_log_dir = self.experiment_dir / "debug_logs" / "eval"

            # Initialize as git repo so evolution AI gets scoped per-run memory.
            # Claude CLI recursive lookup will still find parent code-gen-critic/CLAUDE.md.
            git_dir = self.experiment_dir / ".git"
            if not git_dir.exists():
                subprocess.run(["git", "init"], cwd=str(self.experiment_dir),
                               capture_output=True, check=False)

            # Store evaluation modes
            self.dev_eval_mode = dev_eval_mode


            # Initialize state
            self.agent_pool = {}
            self.performance_records = {}
            self.test_history = []
            self.iteration_times = []
            self.iteration_claude_costs = []
            self.iteration_fresh_evals = []
            self._current_deep_focus_fresh_evals = 0
            self.current_iteration_evolution_cost = None
            self.evolution_times = []
            self.meta_evolution_times = []
            self.zero_accuracy_cases = []
            self.exception_failures = []
            self.five_hour_limit_incidents = []
            self.clone_detections = []  # [(clone_id, matched_id, iteration), ...]

        # Ensure eval mode is always set
        if not hasattr(self, 'dev_eval_mode'):
            self.dev_eval_mode = dev_eval_mode

        # Initialize evolver (will be recreated per iteration with current config)
        # For now, create with iteration 1 config - run() will recreate per iteration
        self.evolver = ParallelAgentEvolver(
            experiment_dir=self.experiment_dir,
            config=config
        )

        # Restore evolver state if resuming
        if resume_mode and resume_checkpoint:
            # Note: evolver tracking state is not in new checkpoint format
            # We'll preserve this for now to avoid breaking resume
            pass

        # Apply pending evolution reset if needed
        if hasattr(self, '_pending_evolution_reset'):
            self._reset_evolution_tracking_for_iteration(self._pending_evolution_reset)
            delattr(self, '_pending_evolution_reset')

        self.memory_monitor = MemoryMonitor()
        self.process_reaper = StuckProcessReaper(experiment_dir=self.experiment_dir)

        # Load data
        self._load_data()

        # Update evolver with domain (created before _load_data)
        self.evolver.domain = self.domain

        # Initialize meta-evolution manager (after _load_data so self.domain exists)
        from RoboPhD.meta_evolution_manager import MetaEvolutionManager
        self.meta_evolution_manager = MetaEvolutionManager(
            experiment_dir=self.experiment_dir,
            config_manager=self.config_manager,
            domain_name=config.get("meta_evolution_domain", self.domain_name),
            domain=self.domain,
        )

        # Pass references to evolver for Deep Focus
        self.evolver.test_history = self.test_history

        # Legacy test_eval_mode for BIRD benchmark test set evaluation was
        # removed in the external architecture migration. See git history if needed.

        # Initialize report generator
        self.report_generator = ReportGenerator(self)

        print(f"\n🔬 RoboPhD Parallel Agent Researcher initialized")
        print(f"📂 Experiment directory: {self.experiment_dir}")
        print(f"🎲 Random seed: {self.random_seed}")
    
    def _load_data(self):
        """Load questions and databases using domain abstraction."""
        # Use full resolved config so domain gets all fields (coder_model, codegen_split, etc.)
        domain_config = dict(self.config_manager.get_config(1))
        # Add runtime fields not managed by config_manager
        domain_config['api_key'] = self.api_key
        # Merge non-serializable runtime config (evaluator_fn, dataset, file_mapping, etc.)
        domain_config.update(self.runtime_config)
        self.domain = get_domain(self.domain_name, domain_config)

        # Load problems grouped by context (database for Text2SQL, problem_id for CodeGen)
        self.problems_by_context = self.domain.load_problems()

        # Flatten for all_problems (backward compatibility)
        self.all_problems = []
        for context_problems in self.problems_by_context.values():
            self.all_problems.extend(context_problems)

        # Get available contexts (databases for Text2SQL, problem_ids for CodeGen)
        self.contexts = self.domain.get_contexts()

        print(f"📊 Loaded {len(self.all_problems)} problems from {len(self.contexts)} {self.domain.context_label_plural}")
    
    @classmethod
    def load_checkpoint(cls, experiment_dir: Path) -> Dict:
        """Load checkpoint from an experiment directory."""
        checkpoint_path = experiment_dir / 'checkpoint.json'
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    
    def _restore_agent_pool(self, pool_data: Dict, from_iteration: Optional[int] = None) -> Dict:
        """
        Restore agent pool from checkpoint data.

        Args:
            pool_data: Agent pool data from checkpoint
            from_iteration: If set, exclude agents created in this iteration or later

        Returns:
            Restored agent pool dictionary
        """
        restored_pool = {}
        for agent_id, agent_info in pool_data.items():
            # When restarting from a specific iteration, skip agents created in that iteration or later
            if from_iteration is not None:
                created_iteration = agent_info.get('created_iteration', 0)
                if created_iteration >= from_iteration:
                    # Skip this agent - it was created in an iteration we're redoing
                    continue
            # Resolve package_dir (stored as experiment-relative path)
            package_dir_str = agent_info.get('package_dir')
            if not package_dir_str:
                print(f"  ⚠️ Skipping agent {agent_id}: no package_dir in checkpoint")
                continue

            package_dir = Path(package_dir_str)
            if not package_dir.is_absolute():
                package_dir = (self.experiment_dir / package_dir).resolve()

            if not package_dir.exists():
                raise FileNotFoundError(
                    f"Agent package directory not found: {package_dir}\n"
                    f"Agent: {agent_id}\n"
                    f"Original path in checkpoint: {package_dir_str}"
                )

            restored_agent = {
                'source': agent_info.get('source', 'restored'),
                'created_iteration': agent_info.get('created_iteration', 0),
                'evolution_strategy': agent_info.get('evolution_strategy', None),
                'session_id': agent_info.get('session_id', None),
                'package_dir': package_dir,
            }

            restored_pool[agent_id] = restored_agent
        
        return restored_pool
    
    def _restore_performance_tracking_before_iteration(self, from_iteration: int):
        """
        Restore last_win_iteration and last_test_iteration to their state
        before from_iteration, based on test_history.

        This is needed when using --from-iteration to ensure agents aren't
        incorrectly treated as "pending winners" for wins that are being re-executed.

        Applies the current tie-breaking mode (oldest_agent_wins_ties,
        random_agent_wins_ties) during replay so pending-winner state is
        consistent with the configured behavior.

        Args:
            from_iteration: Iteration number to restore before (1-indexed)
        """
        print(f"🧹 Restoring performance tracking to state before iteration {from_iteration}")
        restored_count = 0

        # Build set of cloned agent IDs for each iteration
        clone_agents_by_iter = {}
        for clone_id, _matched_id, iter_num in self.clone_detections:
            clone_agents_by_iter.setdefault(iter_num, set()).add(clone_id)

        # Get tie-breaking config
        current_config = self.config_manager.get_config(1)
        oawt = current_config.get("oldest_agent_wins_ties", False)
        rawt = current_config.get("random_agent_wins_ties", False)

        # Determine winners per iteration (with tie-breaking)
        # Maps agent_id -> most recent winning iteration
        last_win_by_agent = {agent_id: None for agent_id in self.performance_records}

        for iter_idx in range(min(from_iteration - 1, len(self.test_history))):
            iteration_results = self.test_history[iter_idx]
            if not iteration_results:
                continue
            iter_num = iter_idx + 1
            clones_this_iter = clone_agents_by_iter.get(iter_num, set())
            eligible = {k: v for k, v in iteration_results.items()
                       if k not in clones_this_iter}
            if not eligible:
                continue

            max_score = max(eligible[k]['average_score'] for k in eligible)
            winners = [k for k, v in eligible.items()
                      if round(v['average_score'], 6) == round(max_score, 6)]

            # Apply tie-breaking
            if len(winners) > 1 and oawt:
                winners.sort(key=lambda a: (self.agent_pool[a].get('created_iteration', 0), random.random()))
                winners = [winners[0]]
            elif len(winners) > 1 and rawt:
                winners = [random.choice(winners)]

            for winner_id in winners:
                last_win_by_agent[winner_id] = iter_num

        # Restore last_win_iteration and last_test_iteration
        for agent_id in self.performance_records.keys():
            last_win = last_win_by_agent.get(agent_id)

            # Find most recent test before from_iteration
            last_test = None
            for iter_idx in range(from_iteration - 2, -1, -1):
                if iter_idx < len(self.test_history):
                    if agent_id in self.test_history[iter_idx]:
                        last_test = iter_idx + 1
                        break

            old_win = self.performance_records[agent_id].get('last_win_iteration')
            old_test = self.performance_records[agent_id].get('last_test_iteration')

            if old_win != last_win or old_test != last_test:
                self.performance_records[agent_id]['last_win_iteration'] = last_win
                self.performance_records[agent_id]['last_test_iteration'] = last_test
                restored_count += 1

        if restored_count > 0:
            print(f"  Restored tracking for {restored_count} agent(s)")

    def archive_iterations(self, from_iteration: int):
        """Archive existing iterations from a specific point onwards."""
        import shutil
        from datetime import datetime
        
        # Find iterations to archive
        iterations_to_archive = []
        for item in self.experiment_dir.iterdir():
            if item.is_dir() and item.name.startswith('iteration_'):
                try:
                    iter_num = int(item.name.split('_')[1])
                    if iter_num >= from_iteration:
                        iterations_to_archive.append(item)
                except (IndexError, ValueError):
                    continue
        
        # Find evolution_output directories to archive
        evolution_dirs_to_archive = []
        evolution_output_dir = self.experiment_dir / "evolution_output"
        if evolution_output_dir.exists():
            for item in evolution_output_dir.iterdir():
                if item.is_dir() and item.name.startswith('iteration_'):
                    try:
                        iter_num = int(item.name.split('_')[1])
                        if iter_num >= from_iteration:
                            evolution_dirs_to_archive.append(item)
                    except (IndexError, ValueError):
                        continue

        # Find meta_evolution_output directories to archive
        meta_evolution_dirs_to_archive = []
        meta_evolution_output_dir = self.experiment_dir / "meta_evolution_output"
        if meta_evolution_output_dir.exists():
            for item in meta_evolution_output_dir.iterdir():
                if item.is_dir() and item.name.startswith('iteration_'):
                    try:
                        iter_num = int(item.name.split('_')[1])
                        if iter_num >= from_iteration:
                            meta_evolution_dirs_to_archive.append(item)
                    except (IndexError, ValueError):
                        continue

        # Archive if there's anything to archive (iterations, evolution_output, or meta_evolution_output)
        if iterations_to_archive or evolution_dirs_to_archive or meta_evolution_dirs_to_archive:
            # Create archive directory with timestamp
            archive_dir = self.experiment_dir / f"archived_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            archive_dir.mkdir(exist_ok=True)

            # Copy checkpoint.json and final_report.md to archive (for reference)
            checkpoint_file = self.experiment_dir / 'checkpoint.json'
            final_report_file = self.experiment_dir / 'final_report.md'

            if checkpoint_file.exists():
                shutil.copy2(str(checkpoint_file), str(archive_dir / 'checkpoint.json'))
                print(f"📄 Copied checkpoint.json to archive")

            if final_report_file.exists():
                shutil.copy2(str(final_report_file), str(archive_dir / 'final_report.md'))
                print(f"📄 Copied final_report.md to archive")

            # Archive iterations if any
            if iterations_to_archive:
                print(f"📦 Archiving {len(iterations_to_archive)} iterations to {archive_dir.name}/")
                for iteration_dir in iterations_to_archive:
                    dest = archive_dir / iteration_dir.name
                    print(f"  Moving {iteration_dir.name} to archive...")
                    # Use copytree + rmtree for consistency and reliability
                    shutil.copytree(str(iteration_dir), str(dest), dirs_exist_ok=True, symlinks=True)
                    shutil.rmtree(str(iteration_dir))
            
            # Archive evolution_output directories if any
            if evolution_dirs_to_archive:
                print(f"📦 Archiving {len(evolution_dirs_to_archive)} evolution_output directories to {archive_dir.name}/")
                archive_evolution = archive_dir / "evolution_output"
                archive_evolution.mkdir(exist_ok=True)
                for evo_dir in evolution_dirs_to_archive:
                    dest = archive_evolution / evo_dir.name
                    print(f"  Moving evolution_output/{evo_dir.name} to archive...")
                    # Use copytree + rmtree instead of move to ensure all contents are archived
                    # This is more reliable for complex directory structures with subdirectories
                    shutil.copytree(str(evo_dir), str(dest), dirs_exist_ok=True, symlinks=True)
                    shutil.rmtree(str(evo_dir))

            # Archive meta_evolution_output directories if any
            if meta_evolution_dirs_to_archive:
                print(f"📦 Archiving {len(meta_evolution_dirs_to_archive)} meta_evolution_output directories to {archive_dir.name}/")
                archive_meta_evolution = archive_dir / "meta_evolution_output"
                archive_meta_evolution.mkdir(exist_ok=True)
                for meta_evo_dir in meta_evolution_dirs_to_archive:
                    dest = archive_meta_evolution / meta_evo_dir.name
                    print(f"  Moving meta_evolution_output/{meta_evo_dir.name} to archive...")
                    # Use copytree + rmtree instead of move to ensure all contents are archived
                    # This is more reliable for complex directory structures with subdirectories
                    shutil.copytree(str(meta_evo_dir), str(dest), dirs_exist_ok=True, symlinks=True)
                    shutil.rmtree(str(meta_evo_dir))

        # Archive and remove agents created in archived iterations
        agents_to_archive = []
        agents_archive_dir = None
        
        if hasattr(self, 'agent_pool'):
            for agent_id, agent_info in self.agent_pool.items():
                if agent_info.get('created_iteration', 0) >= from_iteration:
                    agents_to_archive.append(agent_id)
            
            if agents_to_archive:
                print(f"📦 Archiving {len(agents_to_archive)} agents created in iterations {from_iteration}+")
                
                # Create agents archive directory
                agents_archive_dir = archive_dir / 'agents'
                agents_archive_dir.mkdir(exist_ok=True)
                
                # Move agent directories to archive (removes from original location)
                agents_dir = self.experiment_dir / "agents"
                for agent_id in agents_to_archive:
                    src = agents_dir / agent_id
                    if src.exists():
                        dest = agents_archive_dir / agent_id
                        print(f"  Moving agent {agent_id} to archive...")
                        shutil.move(str(src), str(dest))
                    
                    # Remove from agent pool
                    del self.agent_pool[agent_id]
                    
                    # Don't delete from performance_records, just clean iteration_results
                    # This preserves the agent's history up to the archive point
                
                print(f"  🧹 Removed {len(agents_to_archive)} agents from active pool and agents/ directory")
        
        # Also check for orphaned agent directories (created but not in pool)
        # These can occur when evolution fails after creating directories
        agents_dir = self.experiment_dir / "agents"
        if agents_dir.exists():
            orphaned_agents = []
            for agent_dir in agents_dir.iterdir():
                if agent_dir.is_dir() and agent_dir.name.startswith('iter'):
                    # Extract iteration number from agent name (e.g., iter25_resilient_fusion -> 25)
                    try:
                        iter_num = int(agent_dir.name.split('_')[0].replace('iter', ''))
                        if iter_num >= from_iteration and agent_dir.name not in agents_to_archive:
                            orphaned_agents.append(agent_dir.name)
                    except (ValueError, IndexError):
                        continue
            
            if orphaned_agents:
                print(f"📦 Archiving {len(orphaned_agents)} orphaned agents from iterations {from_iteration}+")
                # Ensure agents archive directory exists
                if agents_archive_dir is None:
                    agents_archive_dir = archive_dir / 'agents'
                    agents_archive_dir.mkdir(exist_ok=True)
                
                for agent_id in orphaned_agents:
                    src = agents_dir / agent_id
                    dest = agents_archive_dir / agent_id
                    print(f"  Moving orphaned agent {agent_id} to archive...")
                    shutil.move(str(src), str(dest))
                print(f"  🧹 Removed {len(orphaned_agents)} orphaned agents from agents/ directory")
        
        # Trim data arrays to remove archived iterations
        if from_iteration > 1:
            self.test_history = self.test_history[:from_iteration - 1]
            
            # Clean up iteration_results in performance_records for archived iterations
            # This prevents duplicate entries when resuming
            agents_to_remove_from_perf = []
            for agent_id in self.performance_records:
                if 'iteration_results' in self.performance_records[agent_id]:
                    # Remove any results from archived iterations
                    cleaned_results = [
                        result for result in self.performance_records[agent_id]['iteration_results']
                        if result.get('iteration', 0) < from_iteration
                    ]
                    self.performance_records[agent_id]['iteration_results'] = cleaned_results

                    # Recalculate summary statistics based on cleaned results
                    if cleaned_results:
                        total_score_sum = sum(r.get('score_sum', 0.0) for r in cleaned_results if 'score_sum' in r)
                        total_questions = sum(r.get('examples', 0) for r in cleaned_results)

                        self.performance_records[agent_id]['test_count'] = len(cleaned_results)
                        self.performance_records[agent_id]['total_score_sum'] = total_score_sum
                        self.performance_records[agent_id]['total_questions'] = total_questions
                        if total_questions > 0:
                            self.performance_records[agent_id]['mean_score'] = total_score_sum / total_questions
                    else:
                        # No results left - mark for removal from performance_records
                        # (No point preserving agents with no historical data)
                        agents_to_remove_from_perf.append(agent_id)

            # Remove agents with no history from performance_records
            if agents_to_remove_from_perf:
                print(f"  🧹 Removing {len(agents_to_remove_from_perf)} agents with no history from performance_records")
                for agent_id in agents_to_remove_from_perf:
                    del self.performance_records[agent_id]
            
            # Recalculate all ELO scores from the cleaned test_history
            # This ensures consistency after archiving
            print("  🎲 Recalculating ELO scores from cleaned test history...")
            self._recalculate_all_elo_scores()

            # Always truncate evolution_times to prevent duplicates when restarting
            if len(self.evolution_times) >= from_iteration:
                self.evolution_times = self.evolution_times[:from_iteration - 1]

            # Always truncate meta_evolution_times to prevent duplicates when restarting
            if len(self.meta_evolution_times) >= from_iteration:
                self.meta_evolution_times = self.meta_evolution_times[:from_iteration - 1]

            # Truncate time/cost arrays for archived iterations
            if len(self.iteration_times) >= from_iteration:
                archived_time = sum(self.iteration_times[from_iteration - 1:])
                self.iteration_times = self.iteration_times[:from_iteration - 1]
                self.iteration_claude_costs = self.iteration_claude_costs[:from_iteration - 1]
                self.iteration_fresh_evals = self.iteration_fresh_evals[:from_iteration - 1]

                print(f"  ⏱️  Subtracted archived time: {archived_time/60:.1f} minutes")
            
            # Clear failure records for archived iterations
            original_zero_cases = len(self.zero_accuracy_cases) if hasattr(self, 'zero_accuracy_cases') else 0
            if hasattr(self, 'zero_accuracy_cases'):
                self.zero_accuracy_cases = [
                    entry for entry in self.zero_accuracy_cases
                    if entry[-2] < from_iteration  # iter_num is second-to-last in both 3-tuple and legacy 4-tuple
                ]

            original_exception_failures = len(self.exception_failures) if hasattr(self, 'exception_failures') else 0
            if hasattr(self, 'exception_failures'):
                self.exception_failures = [
                    (agent_id, db_name, iter_num, error_msg, total_q)
                    for agent_id, db_name, iter_num, error_msg, total_q in self.exception_failures
                    if iter_num < from_iteration
                ]

            # Clear clone detections for archived iterations
            if hasattr(self, 'clone_detections'):
                self.clone_detections = [
                    (clone_id, matched_id, iter_num)
                    for clone_id, matched_id, iter_num in self.clone_detections
                    if iter_num < from_iteration
                ]

            if original_zero_cases > 0 and original_zero_cases != len(self.zero_accuracy_cases):
                print(f"  🧹 Cleared {original_zero_cases - len(self.zero_accuracy_cases)} zero accuracy records")
            if original_exception_failures > 0 and original_exception_failures != len(self.exception_failures):
                print(f"  🧹 Cleared {original_exception_failures - len(self.exception_failures)} exception failure records")
            
            # Reset evolution tracking to match the new starting point
            self._reset_evolution_tracking_for_iteration(from_iteration)
    
    def _reset_evolution_tracking_for_iteration(self, from_iteration: int):
        """
        Reset evolution tracking when restarting from a specific iteration.
        
        This ensures that:
        1. Evolution count is properly adjusted
        2. Evolution history is trimmed to match archived iterations
        3. The evolver's first_evolution_call flag is reset appropriately
        
        Args:
            from_iteration: The iteration we're restarting from
        """
        # Only reset if evolver exists (it won't exist yet during __init__)
        if not hasattr(self, 'evolver'):
            # Store the reset request for later when evolver is initialized
            self._pending_evolution_reset = from_iteration
            return
            
        # Calculate how many evolutions occurred before from_iteration
        evolutions_before = 0
        for hist_entry in self.evolver.evolution_history:
            if hist_entry['iteration'] < from_iteration:
                if hist_entry['strategy'].lower() not in ['none', 'skip']:
                    evolutions_before += 1
        
        # Reset evolution count to match what it should be at from_iteration - 1
        self.evolver.evolution_count = evolutions_before
        
        # Trim evolution history to remove archived iterations
        # This ensures we don't have duplicate entries when re-running iterations
        self.evolver.evolution_history = [
            entry for entry in self.evolver.evolution_history
            if entry['iteration'] < from_iteration
        ]

        # Count how many random selections we're keeping
        # This preserves the random selection sequence
        remaining_random_count = sum(1 for entry in self.evolver.evolution_history
                                    if entry.get('was_random', False))
        print(f"     Keeping {remaining_random_count} random selections in history")

        # Trim retries to only include those before from_iteration
        self.evolver.evolution_retries = [
            retry for retry in self.evolver.evolution_retries
            if retry.get('iteration', 999) < from_iteration
        ]

        # Trim validation failures and header repairs
        if hasattr(self.evolver, 'evolution_validation_failures'):
            self.evolver.evolution_validation_failures = [
                failure for failure in self.evolver.evolution_validation_failures
                if failure.get('iteration', 999) < from_iteration
            ]

        if hasattr(self.evolver, 'header_repairs'):
            self.evolver.header_repairs = [
                repair for repair in self.evolver.header_repairs
                if repair.get('iteration', 999) < from_iteration
            ]
        
        # Reset the first evolution call flag based on whether we've done any evolutions
        # If we're at iteration 9 and have done evolutions, this should be False
        self.evolver.is_first_evolution_call = (evolutions_before == 0)
        
        print(f"  🔄 Reset evolution tracking:")
        print(f"     Evolution count: {self.evolver.evolution_count}")
        print(f"     Evolution history entries: {len(self.evolver.evolution_history)}")
        print(f"     First evolution call: {self.evolver.is_first_evolution_call}")
    
    def _is_valid_agent_dir(self, agent_dir: Path) -> bool:
        """Check if a directory is a valid agent by looking for file_mapping files."""
        file_mapping = getattr(self.domain, 'file_mapping', {}) if self.domain else {}
        if not file_mapping:
            return False
        return all((agent_dir / rel_path).exists() for rel_path in file_mapping.values())

    def load_initial_agents(self, agent_list: Optional[List[str]] = None):
        """
        Load initial agents from agents directory.

        Discovers agents by checking for all files from file_mapping.

        Args:
            agent_list: Optional list of specific agent names to load
        """
        # Use custom agents directory if specified, otherwise default to RoboPhD/agents/
        if self.agents_directory:
            agents_dir = Path(self.agents_directory)
        else:
            agents_dir = Path(__file__).parent / 'agents'

        if not agent_list:
            # Auto-discover agent directories
            agent_dirs = [d for d in agents_dir.iterdir()
                          if d.is_dir() and self._is_valid_agent_dir(d)]
        else:
            # Load specific agents
            agent_dirs = []
            for name in agent_list:
                agent_dir = agents_dir / name
                if agent_dir.exists() and agent_dir.is_dir():
                    if self._is_valid_agent_dir(agent_dir):
                        agent_dirs.append(agent_dir)
                    else:
                        print(f"  ⚠️ Agent directory has no recognized artifacts: {name}")
                else:
                    print(f"  ⚠️ Agent not found: {name}")

        for agent_dir in agent_dirs:
            agent_id = agent_dir.name

            # Copy entire agent directory to local agents directory
            local_agents_dir = self.experiment_dir / "agents"
            local_agents_dir.mkdir(exist_ok=True)
            local_agent_dir = local_agents_dir / agent_id

            if local_agent_dir.exists():
                shutil.rmtree(local_agent_dir)

            shutil.copytree(agent_dir, local_agent_dir, symlinks=True)

            self.agent_pool[agent_id] = {
                'source': 'initial',
                'created_iteration': 0,
                'package_dir': local_agent_dir,
            }

            # Initialize performance record
            self.performance_records[agent_id] = {
                'test_count': 0,
                'total_score_sum': 0.0,
                'total_questions': 0,
                'mean_score': 0.0,
                'elo': 1500,
                'iteration_results': [],
                'last_win_iteration': None,
                'last_test_iteration': None
            }

            print(f"  🤖 Loaded agent: {agent_id}")

        print(f"\n✅ Loaded {len(self.agent_pool)} initial agents")

    # Strategies that don't correspond to file-based strategy directories
    NON_FILE_STRATEGIES = {"none", "challenger", "greedy", "random"}

    def _collect_referenced_strategies(self) -> List[str]:
        """
        Scan the full config for all evolution_strategy references.

        Checks:
        - Base evolution_strategy
        - config_schedule values
        - weighted_random_configs entries

        Returns:
            Deduplicated list of file-based strategy names, or empty list
            if none found (which triggers auto-discover-all fallback).
        """
        config = self.config_manager.get_config(1)
        strategies = set()

        # Base evolution_strategy
        base = config.get("evolution_strategy")
        if base:
            strategies.add(base)

        # config_schedule values
        for delta in config.get("config_schedule", {}).values():
            if isinstance(delta, dict) and "evolution_strategy" in delta:
                strategies.add(delta["evolution_strategy"])

        # weighted_random_configs entries
        for entry in config.get("weighted_random_configs", []):
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                cfg, _weight = entry
                if isinstance(cfg, dict) and "evolution_strategy" in cfg:
                    strategies.add(cfg["evolution_strategy"])

        # Filter out non-file-based strategies and None
        strategies = {s for s in strategies if s and s not in self.NON_FILE_STRATEGIES}

        return sorted(strategies) if strategies else []

    def load_initial_strategies(self, strategy_list: Optional[List[str]] = None):
        """
        Load initial evolution strategies from strategies directory.

        Copies strategy directories to <experiment_dir>/evolution_strategies/.
        For research_driven strategies, shuffles the papers pool.

        Args:
            strategy_list: Optional list of specific strategy names to load
        """
        # Use custom strategies directory if specified, otherwise use domain's default
        if self.strategies_directory:
            strategies_dir = Path(self.strategies_directory)
        elif self.domain is not None:
            strategies_dir = Path(__file__).parent / self.domain.evolution_strategies_dir
        else:
            raise ValueError(
                "ParallelAgentEvolver requires a domain to be set for loading strategies. "
                "Set evolver.domain after creation."
            )

        if not strategy_list:
            # Auto-discover all strategy directories
            strategy_dirs = [d for d in strategies_dir.iterdir()
                           if d.is_dir() and (d / 'strategy.md').exists()]
        else:
            # Load specific strategies
            strategy_dirs = []
            for name in strategy_list:
                strategy_dir = strategies_dir / name
                if strategy_dir.exists() and strategy_dir.is_dir():
                    if (strategy_dir / 'strategy.md').exists():
                        strategy_dirs.append(strategy_dir)
                    else:
                        print(f"  ⚠️ Strategy directory missing strategy.md: {name}")
                else:
                    print(f"  ⚠️ Strategy not found: {name}")

        # Create local evolution_strategies directory
        local_strategies_dir = self.experiment_dir / "evolution_strategies"
        local_strategies_dir.mkdir(exist_ok=True)

        for strategy_dir in strategy_dirs:
            strategy_id = strategy_dir.name
            local_strategy_dir = local_strategies_dir / strategy_id

            # Remove existing directory if it exists
            if local_strategy_dir.exists():
                shutil.rmtree(local_strategy_dir)

            # Copy the entire directory
            shutil.copytree(strategy_dir, local_strategy_dir, symlinks=True)

            # Special handling for research_driven strategies: shuffle papers pool
            if 'research_driven' in strategy_id:
                papers_pool_path = local_strategy_dir / 'tools' / 'papers_pool.json'
                if papers_pool_path.exists():
                    with open(papers_pool_path, 'r') as f:
                        pool = json.load(f)

                    # Shuffle papers for this experiment
                    papers = pool.get('papers', [])
                    random.shuffle(papers)
                    pool['papers'] = papers
                    pool['used_papers'] = []

                    with open(papers_pool_path, 'w') as f:
                        json.dump(pool, f, indent=2)

                    print(f"  ✓ Shuffled papers pool for {strategy_id} ({len(papers)} papers)")

        if len(strategy_dirs) == 0:
            if strategy_list:
                raise ValueError(
                    f"No valid strategies found from requested list: {strategy_list}\n"
                    f"Strategies directory: {strategies_dir}"
                )
            else:
                raise ValueError(
                    f"No evolution strategies found in {strategies_dir}\n"
                    f"Ensure the directory contains subdirectories with strategy.md files"
                )

        print(f"📋 Loaded {len(strategy_dirs)} initial strategies to {local_strategies_dir}")

    def run_iteration(self, iteration: int, selected_agents: List[str], contexts: List[str]) -> Dict:
        """
        Run one iteration testing selected agents on contexts.

        All domains use a unified evaluation flow via domain.run_evaluation().
        Parallelism is handled internally by the domain (e.g., Text2SQL parallelizes
        across databases, CodeGen uses subprocess with internal parallelism).

        Args:
            iteration: Iteration number
            selected_agents: List of agent IDs to test
            contexts: List of context identifiers to test on

        Returns:
            Tuple of (iteration_results, results_by_agent, costs_by_context, eval_cache_stats)
        """
        from datetime import datetime

        print(f"Agents: {', '.join(selected_agents)}")
        print(f"{self.domain.context_label_plural.title()}: {', '.join(str(c) for c in contexts)}")

        # CRITICAL: Sample problems once per context for this iteration (sequential, before threading)
        # This ensures ALL agents test IDENTICAL problems (fair comparison + deterministic)
        self.current_iteration_problems = {}
        for context_name in contexts:
            problems = self.problems_by_context.get(context_name, [])
            if problems:
                sampled = random.sample(
                    problems,
                    min(1, len(problems))
                )
                self.current_iteration_problems[context_name] = sampled
            else:
                self.current_iteration_problems[context_name] = []

        # Build SampledProblems once (all agents test identical problems)
        sampled = SampledProblems(
            contexts=contexts,
            problems_by_context=self.current_iteration_problems
        )

        # Initialize cost tracking for this iteration
        iteration_cost_dict = {
            'eval_cost': 0.0,
            'eval_tokens_in': 0,
            'eval_tokens_out': 0,
            'evolution_cost': 0.0,
            'evolution_calls': 0,
            'evolution_tokens_in': 0,
            'evolution_tokens_out': 0,
            'evolution_cache_created': 0,
            'evolution_cache_read': 0,
            'evolution_breakdown': None,
        }

        # Track results for each agent
        results_by_agent: Dict[str, List[Dict]] = {agent_id: [] for agent_id in selected_agents}
        costs_by_context: Dict[str, Dict[str, Dict[str, float]]] = {}  # {agent_id: {context: {'eval': $}}}
        iteration_results: Dict[str, Dict] = {}
        eval_cache_stats: Dict[str, Dict[str, int]] = {}  # {agent_id: {'cached': N, 'fresh': M}}

        # Process each agent (sequential - agents are few, ~2-4)
        # Domain handles internal parallelism (e.g., Text2SQL parallelizes across databases)
        for agent_id in selected_agents:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"    [{timestamp}] {agent_id}: Starting evaluation on {len(contexts)} {self.domain.context_label_plural}...")

            agent_info = self.agent_pool.get(agent_id)
            if not agent_info:
                print(f"    ❌ {agent_id}: Agent not found")
                iteration_results[agent_id] = {
                    'average_score': 0.0,
                    'score_sum': 0.0,
                    'total': sum(len(p) for p in self.current_iteration_problems.values()),
                    'failures': len(contexts)
                }
                continue

            # Create output directory for this agent.
            # experiment_dir is already outside the RoboPhD repo (in robophd_runs/),
            # so no symlink split needed — CLAUDE.md contamination is avoided.
            agent_output_dir = self.experiment_dir / f"iteration_{iteration:03d}" / f"agent_{agent_id}"
            agent_output_dir.mkdir(parents=True, exist_ok=True)

            # Build config for domain evaluation
            # Pass full config directly - domains extract what they need
            # ConfigManager resolves all defaults (fail fast if missing)
            current_config = self.config_manager.get_config(iteration)
            eval_config = current_config.copy()
            eval_config.update({
                'agent_id': agent_id,
                'experiment_dir': self.experiment_dir,
                'api_key': self.api_key,
            })

            try:
                # Universal interface - domain handles internals
                eval_result = self.domain.run_evaluation(
                    sampled=sampled,
                    agent_path=agent_info.get('package_dir'),
                    output_dir=agent_output_dir,
                    config=eval_config
                )

                # Convert EvaluationResult to per-context results for compatibility
                for r in eval_result.results:
                    context_name = r.get('db_id') or r.get('question_id', 'unknown')
                    results_by_agent[agent_id].append({
                        'success': r.get('error') is None,
                        'context': context_name,
                        'agent_id': agent_id,
                        'score': r.get('score', 0),
                        'total': 1,
                        'error': r.get('error'),
                        # Include eval_cost for cost reporting
                        'eval_cost': r.get('eval_cost', 0.0),
                    })

                # Calculate agent metrics
                average_score = eval_result.average_score
                score_sum = eval_result.score_sum
                total_questions = eval_result.total

                # Get contexts tested successfully from metadata
                metadata = eval_result.metadata or {}

                iteration_results[agent_id] = {
                    'average_score': average_score,
                    'score_sum': score_sum,
                    'total': total_questions,
                    'contexts_tested': list(contexts),
                    'failures': 0
                }

                # Update performance records
                perf = self.performance_records[agent_id]
                perf['test_count'] += 1
                perf['total_score_sum'] += score_sum
                perf['total_questions'] += total_questions
                perf['mean_score'] = (perf['total_score_sum'] / perf['total_questions']) if perf['total_questions'] > 0 else 0
                perf['iteration_results'].append({
                    'iteration': iteration,
                    'average_score': average_score,
                    'score_sum': score_sum,
                    'examples': len(contexts)
                })

                # Track zero score cases
                if average_score == 0 and total_questions > 0:
                    self.zero_accuracy_cases.append((agent_id, iteration, total_questions))

                # Accumulate evaluation costs from metadata
                if metadata.get('eval_cost'):
                    iteration_cost_dict['eval_cost'] += metadata['eval_cost']
                    iteration_cost_dict['eval_tokens_in'] += metadata.get('eval_tokens_in', 0)
                    iteration_cost_dict['eval_tokens_out'] += metadata.get('eval_tokens_out', 0)

                # Extract per-context costs from metadata for cost report
                if metadata.get('costs_by_context'):
                    costs_by_context[agent_id] = metadata['costs_by_context']

                # Track eval result cache stats
                if metadata.get('cached_count', 0) > 0:
                    eval_cache_stats[agent_id] = {
                        'cached': metadata['cached_count'],
                        'fresh': metadata.get('fresh_count', total_questions),
                    }

                cached_count = metadata.get('cached_count', 0)
                cache_suffix = f" [cached {cached_count}/{total_questions}]" if cached_count else ""
                print(f"\n{agent_id}: {average_score:.3f} ({score_sum:.1f}/{total_questions}){cache_suffix}")

            except EvalRateLimitError as e:
                print(f"\n❌ RATE LIMIT EXCEEDED - Aborting research run")
                print(f"   Agent: {agent_id}")
                print(f"   Error: {e}")
                raise

            except Exception as e:
                import traceback
                error_str = str(e)

                for infra_error in CRITICAL_INFRASTRUCTURE_ERRORS:
                    if infra_error in error_str:
                        print(f"\n❌ CRITICAL INFRASTRUCTURE ERROR - Aborting research run")
                        print(f"   Agent: {agent_id}")
                        print(f"   Error: {error_str}")
                        raise

                # Log error but continue with other agents
                print(f"    ❌ {agent_id}: Error - {e}")
                traceback.print_exc()

                total_questions = sum(len(p) for p in self.current_iteration_problems.values())
                iteration_results[agent_id] = {
                    'average_score': 0.0,
                    'score_sum': 0.0,
                    'total': total_questions,
                    'failures': len(contexts)
                }

                self.exception_failures.append((agent_id, 'all', iteration, str(e), total_questions))

        # Determine winner(s)
        if not iteration_results:
            error_msg = (
                f"\n{'='*60}\n"
                f"❌ FATAL: No agents completed testing in iteration {iteration}\n"
                f"{'='*60}\n"
                f"Agents attempted: {', '.join(selected_agents)}\n"
                f"{self.domain.context_label_plural.title()} attempted: {', '.join(str(c) for c in contexts)}\n"
            )
            raise RuntimeError(error_msg)

        # Detect exact-clone agents: newly created agents with identical
        # per-problem scores to any other agent in this iteration.
        clone_agents = set()
        for agent_id in selected_agents:
            agent_info = self.agent_pool.get(agent_id, {})
            if agent_info.get('created_iteration') != iteration:
                continue  # Only check newly created agents
            # Build score vector for this agent
            new_scores = {r['context']: r['score'] for r in results_by_agent.get(agent_id, [])}
            if not new_scores:
                continue
            # Compare against all other agents in this iteration
            for other_id in selected_agents:
                if other_id == agent_id:
                    continue
                other_scores = {r['context']: r['score'] for r in results_by_agent.get(other_id, [])}
                if new_scores == other_scores:
                    clone_agents.add(agent_id)
                    self.clone_detections.append((agent_id, other_id, iteration))
                    print(f"    ⚠️ Clone detected: {agent_id} has identical scores to {other_id} — ELO penalty applied, excluded from winners")
                    break

        # Find winner(s) — exclude clone agents
        eligible = {k: v for k, v in iteration_results.items() if k not in clone_agents}
        assert eligible, "All agents are clones — this should be impossible (at most 1 new agent per iteration)"
        max_score = round(max(r['average_score'] for r in eligible.values()), 6)
        winners = [k for k, v in eligible.items() if round(v['average_score'], 6) == max_score]

        # Tie breaking
        if len(winners) > 1 and current_config.get("oldest_agent_wins_ties", False):
            winners.sort(key=lambda a: (self.agent_pool[a].get('created_iteration', 0), random.random()))
            print(f"\n🏆 Iteration {iteration} winner: {winners[0]} ({max_score:.3f})")
            print(f"     (tie-break over {', '.join(winners[1:])} — oldest agent wins)")
            winners = [winners[0]]
        elif len(winners) > 1 and current_config.get("random_agent_wins_ties", False):
            winner = random.choice(winners)
            others = [w for w in winners if w != winner]
            print(f"\n🏆 Iteration {iteration} winner: {winner} ({max_score:.3f})")
            print(f"     (random tie-break over {', '.join(others)})")
            winners = [winner]
        elif len(winners) == 1:
            print(f"\n🏆 Iteration {iteration} winner: {winners[0]} ({max_score:.3f})")
        else:
            print(f"\n🏆 Iteration {iteration} tied winners: {', '.join(winners)} ({max_score:.3f})")

        # Update last_win_iteration for ALL winners
        for winner_id in winners:
            self.performance_records[winner_id]['last_win_iteration'] = iteration

        # Store results in test_history
        self.test_history.append(iteration_results)

        # Update ELO scores (includes clone penalties via _recalculate_all_elo_scores)
        self._update_elo_scores(iteration_results)

        # Populate evolution costs from temporary storage
        if self.current_iteration_evolution_cost is not None:
            evo_cost_info = self.current_iteration_evolution_cost
            iteration_cost_dict['evolution_cost'] = evo_cost_info.get('total', 0.0)
            iteration_cost_dict['evolution_calls'] = sum(
                1 for key, phase in evo_cost_info.items()
                if key != 'total' and isinstance(phase, dict) and phase.get('cost', 0) > 0
            )
            for key, phase in evo_cost_info.items():
                if key != 'total' and isinstance(phase, dict):
                    iteration_cost_dict['evolution_tokens_in'] += phase.get('tokens_in', 0)
                    iteration_cost_dict['evolution_tokens_out'] += phase.get('tokens_out', 0)
                    iteration_cost_dict['evolution_cache_created'] += phase.get('cache_created', 0)
                    iteration_cost_dict['evolution_cache_read'] += phase.get('cache_read', 0)
            iteration_cost_dict['evolution_breakdown'] = evo_cost_info
            self.current_iteration_evolution_cost = None

        # Store iteration costs
        self.iteration_claude_costs.append(iteration_cost_dict)

        # Cleanup database connections
        close_robophd_connections()
        close_eval_connections()

        return iteration_results, results_by_agent, costs_by_context, eval_cache_stats

    @staticmethod
    def _calculate_elo_updates(current_elos: Dict[str, float], iteration_results: Dict, k: int = 32) -> Dict[str, float]:
        """
        Calculate updated ELO scores based on head-to-head results, properly handling ties.

        Args:
            current_elos: Dictionary of agent_id -> current ELO score
            iteration_results: Dictionary of agent_id -> {'average_score': float, ...}
            k: K-factor for ELO calculations (default 32)

        Returns:
            Dictionary of agent_id -> updated ELO score
        """
        # Create a copy to avoid modifying the input
        updated_elos = current_elos.copy()
        agents = list(iteration_results.keys())

        # Group agents by score to identify ties
        # Round to 6 decimals to collapse floating-point noise into ties
        score_groups = {}
        for agent in agents:
            score = round(iteration_results[agent]['average_score'], 6)
            if score not in score_groups:
                score_groups[score] = []
            score_groups[score].append(agent)

        # Process ties within groups (each agent draws against others in same group)
        for score, group in score_groups.items():
            if len(group) > 1:
                # Process all pairs within the tied group
                for i, agent1 in enumerate(group):
                    for agent2 in group[i+1:]:
                        # Handle as a draw (0.5 points each)
                        elo1 = updated_elos[agent1]
                        elo2 = updated_elos[agent2]

                        expected1 = 1 / (1 + 10**((elo2 - elo1) / 400))
                        expected2 = 1 / (1 + 10**((elo1 - elo2) / 400))

                        updated_elos[agent1] += k * (0.5 - expected1)
                        updated_elos[agent2] += k * (0.5 - expected2)

        # Process wins/losses between different score groups
        sorted_groups = sorted(score_groups.keys(), reverse=True)
        for i, higher_score in enumerate(sorted_groups[:-1]):
            for lower_score in sorted_groups[i+1:]:
                for winner in score_groups[higher_score]:
                    for loser in score_groups[lower_score]:
                        # Winner beats loser
                        winner_elo = updated_elos[winner]
                        loser_elo = updated_elos[loser]

                        # ELO calculation
                        expected_winner = 1 / (1 + 10**((loser_elo - winner_elo) / 400))
                        expected_loser = 1 / (1 + 10**((winner_elo - loser_elo) / 400))

                        updated_elos[winner] += k * (1 - expected_winner)
                        updated_elos[loser] += k * (0 - expected_loser)

        return updated_elos
    
    def _recalculate_all_elo_scores(self):
        """
        Recalculate all ELO scores from scratch based on test_history.
        This ensures consistency and prevents accumulated errors.
        """
        # Reset all ELO scores to base
        cumulative_elo_scores = {}
        
        # Process all iterations in test_history
        for iteration_data in self.test_history:
            # Initialize new agents with base ELO
            for agent in iteration_data:
                if agent not in cumulative_elo_scores:
                    cumulative_elo_scores[agent] = 1500.0
            
            # Get scores for this iteration (already 0-1 scale)
            iteration_results = {
                agent: {'average_score': data['average_score']}
                for agent, data in iteration_data.items()
            }

            # Calculate updated ELO scores using the shared logic
            current_elos_for_iteration = {
                agent: cumulative_elo_scores[agent]
                for agent in iteration_results
            }
            updated_elos = self._calculate_elo_updates(current_elos_for_iteration, iteration_results)

            # Update the cumulative scores
            for agent, new_elo in updated_elos.items():
                cumulative_elo_scores[agent] = new_elo

        # Update all performance_records with recalculated ELO scores
        for agent_id in self.performance_records:
            if agent_id in cumulative_elo_scores:
                self.performance_records[agent_id]['elo'] = cumulative_elo_scores[agent_id]
            else:
                # Agent hasn't been tested yet, keep base ELO
                self.performance_records[agent_id]['elo'] = 1500.0

        # Apply persistent clone penalties
        for clone_id, _matched_id, _iteration in self.clone_detections:
            if clone_id in self.performance_records:
                self.performance_records[clone_id]['elo'] -= 200
    
    def _update_elo_scores(self, iteration_results: Dict):
        """
        Update ELO scores by recalculating from scratch based on all test history.
        This ensures consistency and prevents accumulated errors.
        """
        # Instead of incremental updates, recalculate everything from test_history
        # This prevents drift and ensures consistency
        self._recalculate_all_elo_scores()
    
    def _calculate_elo_progression(self) -> List[Dict]:
        """
        Calculate ELO progression to track the leader after each iteration.
        
        Returns:
            List of dictionaries containing iteration number, leader name, ELO score, and average_score
        """
        # We need to maintain a cumulative ELO score dictionary
        cumulative_elo_scores = {}
        leaders = []
        
        for iter_num, iteration_data in enumerate(self.test_history, 1):
            # Initialize new agents with base ELO
            for agent in iteration_data:
                if agent not in cumulative_elo_scores:
                    cumulative_elo_scores[agent] = 1500.0
            
            # Get scores for this iteration (already 0-1 scale)
            iteration_results = {
                agent: {'average_score': data['average_score']}
                for agent, data in iteration_data.items()
            }

            # Calculate updated ELO scores using the shared logic
            # Important: We update the cumulative scores, not reset them
            current_elos_for_iteration = {
                agent: cumulative_elo_scores[agent]
                for agent in iteration_results
            }
            updated_elos = self._calculate_elo_updates(current_elos_for_iteration, iteration_results)

            # Update the cumulative scores with the new values
            for agent, new_elo in updated_elos.items():
                cumulative_elo_scores[agent] = new_elo

            # Find the leader after this iteration (from ALL agents, not just tested ones)
            if cumulative_elo_scores:
                leader_agent = max(cumulative_elo_scores.items(), key=lambda x: x[1])
                leaders.append({
                    'iteration': iter_num,
                    'leader': leader_agent[0],
                    'elo': leader_agent[1],
                    'average_score': iteration_data.get(leader_agent[0], {}).get('average_score', None)
                })
        
        return leaders
    
    def _get_agent_evolution_strategy(self, agent_id: str) -> str:
        """
        Get the evolution strategy that created an agent.

        Args:
            agent_id: The agent identifier

        Returns:
            Strategy name or "Initial" for non-evolved agents
        """
        if agent_id not in self.agent_pool:
            return "Unknown"

        agent_info = self.agent_pool[agent_id]

        # Check if it's an evolved agent
        if agent_info.get('source') == 'evolution':
            # First check if we stored the strategy directly (new approach)
            if 'evolution_strategy' in agent_info:
                return agent_info['evolution_strategy']

            # Fallback: look it up from evolution history by iteration
            created_iter = agent_info.get('created_iteration')
            if created_iter:
                for entry in self.evolver.evolution_history:
                    if entry['iteration'] == created_iter:
                        return entry.get('strategy', 'Unknown')
            return "Evolution (unknown)"
        else:
            # Initial agent
            return "Initial"
    
    def _get_pending_winners(self) -> List[str]:
        """
        Find all agents that won an iteration but haven't been tested since their win.

        An agent is "pending" if:
        - It has won at least once (last_win_iteration is not None)
        - It hasn't been tested after that win (last_test_iteration <= last_win_iteration)

        Returns:
            List of pending winner agent IDs, sorted by most recent win first, then by ELO
        """
        pending = []

        for agent_id, perf in self.performance_records.items():
            last_win = perf.get('last_win_iteration')
            last_test = perf.get('last_test_iteration', -1)  # Treat None as -1

            # Check if agent won and hasn't been tested after the win
            if last_win is not None:
                # Handle None for last_test (never tested scenario, though shouldn't happen)
                if last_test is None or last_test <= last_win:
                    pending.append(agent_id)

        # Sort by most recent win first, then by ELO (descending)
        pending.sort(key=lambda agent_id: (
            -self.performance_records[agent_id].get('last_win_iteration', 0),
            -self.performance_records[agent_id].get('elo', 1500)
        ))

        return pending

    def _select_challenger_agents(self, iteration: int) -> List[str]:
        """
        Select agents for challenger round - targets under-tested high-ELO agents.

        Excludes pending winners to "break dynasties" and find hidden gems.
        Criteria: ELO > 1500
        Selection: Sort by test count ascending (random within ties)

        Args:
            iteration: Current iteration number

        Returns:
            List of agent IDs to test
        """
        # Print selection header
        print(f"\n📋 CHALLENGER AGENT SELECTION FOR ITERATION {iteration}")
        print("═" * 60)

        # Get pending winners to exclude
        pending_winners = self._get_pending_winners()
        exclude_set = set(pending_winners)

        if pending_winners:
            print(f"\n🚫 Excluding {len(pending_winners)} pending winner(s): {', '.join(pending_winners)}")

        # Find eligible challengers: ELO > 1500 (excluding pending winners)
        print(f"\n🎯 Challenger mode: Under-tested high-performers (ELO > 1500)")
        challengers = []
        for agent_id, perf in self.performance_records.items():
            # Skip pending winners
            if agent_id in exclude_set:
                continue

            # Challenger criteria: high-performing agents
            elo = perf.get('elo', 1500)
            test_count = perf.get('test_count', 0)

            if elo > 1500 and test_count > 0:
                challengers.append((agent_id, elo, test_count))

        # Group challengers by test count for display
        by_test_count = defaultdict(list)
        for agent_id, elo, test_count in challengers:
            by_test_count[test_count].append((agent_id, elo))

        # Sort each test count group by ELO, then shuffle for random tie-breaking
        sorted_challengers = []
        for test_count in sorted(by_test_count.keys()):
            agents_at_count = by_test_count[test_count]
            # Shuffle agents with same test count for random selection
            random.shuffle(agents_at_count)
            for agent_id, elo in agents_at_count:
                sorted_challengers.append((agent_id, elo, test_count))

        # Select top k agents
        selected = []
        num_to_select = min(self.agents_per_iteration, len(sorted_challengers))

        if sorted_challengers:
            print(f"  Mode: Deterministic by fewest tests (random within ties)")
            print(f"  Need to fill: {num_to_select} slot(s)")
            print(f"\n  Candidate pool grouped by test count:")

            # Display grouped by test count (show up to first few tiers)
            displayed_tiers = 0
            max_tiers_to_show = 10
            for test_count in sorted(by_test_count.keys()):
                if displayed_tiers >= max_tiers_to_show:
                    remaining_tiers = len(by_test_count) - displayed_tiers
                    print(f"    ... ({remaining_tiers} more test count tiers)")
                    break

                agents = by_test_count[test_count]
                print(f"    Tests = {test_count} ({len(agents)} agent{'s' if len(agents) != 1 else ''}):")
                for agent_id, elo in sorted(agents, key=lambda x: x[1], reverse=True):
                    print(f"      - {agent_id} (ELO: {elo:.0f})")
                displayed_tiers += 1

            # Select top k from sorted list
            selected = [agent_id for agent_id, elo, test_count in sorted_challengers[:num_to_select]]
            print(f"\n  Selected: {selected}")
        else:
            print(f"  ⚠️  No agents found with ELO > 1500 (excluding pending winners)")

        # Final fallback: If still not enough agents, include pending winners
        # (Better to test someone than fail with empty list)
        if len(selected) < self.agents_per_iteration:
            remaining_slots = self.agents_per_iteration - len(selected)
            print(f"\n🔄 Final fallback: Including pending winners to fill {remaining_slots} slot(s)")

            # Get pending winners that aren't already selected
            already_selected = set(selected)
            available_pending = [agent_id for agent_id in pending_winners
                                if agent_id not in already_selected]

            # Select up to remaining_slots from pending winners
            num_to_add = min(remaining_slots, len(available_pending))
            for agent_id in available_pending[:num_to_add]:
                selected.append(agent_id)
                perf = self.performance_records[agent_id]
                elo = perf.get('elo', 1500)
                test_count = perf.get('test_count', 0)
                print(f"  ✓ Pending winner: {agent_id} (ELO: {elo:.0f}, tests: {test_count})")

        # Final fallback: agents with ELO <= 1500, ordered by ELO
        if len(selected) < self.agents_per_iteration:
            remaining_slots = self.agents_per_iteration - len(selected)
            already_selected = set(selected)

            # Get all agents with ELO <= 1500, not already selected
            low_elo_agents = []
            for agent_id, perf in self.performance_records.items():
                if agent_id in already_selected:
                    continue
                elo = perf.get('elo', 1500)
                if elo <= 1500 and perf.get('test_count', 0) > 0:
                    low_elo_agents.append((agent_id, elo))

            # Sort by ELO descending (best of the low-ELO agents first)
            low_elo_agents.sort(key=lambda x: x[1], reverse=True)

            if low_elo_agents:
                print(f"\n🔄 Final fallback: Including agents with ELO ≤ 1500 to fill {remaining_slots} slot(s)")
                for agent_id, elo in low_elo_agents[:remaining_slots]:
                    selected.append(agent_id)
                    print(f"  ✓ Low-ELO agent: {agent_id} (ELO: {elo:.0f})")

        print(f"\n🎯 Final Challenger Selection: {selected}")
        print("=" * 60)

        # Update last_test_iteration for all selected agents
        for agent_id in selected:
            self.performance_records[agent_id]['last_test_iteration'] = iteration

        # Clear the challenger flag
        self.evolver.use_challenger_selection = False

        return selected

    def _select_greedy_agents(self, iteration: int,
                             evolved_agent_id: Optional[str] = None) -> List[str]:
        """
        Greedy selection: deterministic top-k by ELO.

        Uses normal priority flow (1-3) but changes Priority 4 to deterministic selection.

        Priority order:
        1. Pending Winners (all winners not yet retested)
        2. Newly evolved agent (if provided)
        3. Untested agents (test_count == 0)
        4. Deterministic top-k by ELO (no randomization)

        Args:
            iteration: Current iteration number
            evolved_agent_id: ID of newly evolved agent to include (if any)

        Returns:
            List of agent IDs to test
        """
        selected = []
        available = list(self.agent_pool.keys())

        # Print selection header
        print(f"\n📋 GREEDY AGENT SELECTION FOR ITERATION {iteration}")
        print("═" * 60)

        # Priority 1: Pending Winners
        print("\nPriority 1 - Pending Winners:")
        pending_winners = self._get_pending_winners()
        if pending_winners:
            available_pending = [agent for agent in pending_winners if agent in available]

            if available_pending:
                num_to_select = min(self.agents_per_iteration, len(available_pending))
                chosen = available_pending[:num_to_select]

                for agent_id in chosen:
                    selected.append(agent_id)
                    available.remove(agent_id)
                    last_win = self.performance_records[agent_id]['last_win_iteration']
                    elo_score = self.performance_records[agent_id]['elo']
                    print(f"  ✓ Selected: {agent_id} (won iteration {last_win}, ELO: {elo_score:.0f})")

                print(f"  Found {len(pending_winners)} total pending winner(s), selected {len(chosen)}")
            else:
                print(f"  Found {len(pending_winners)} pending winner(s), but none available")
        else:
            print("  ✗ No pending winners (first iteration or all winners retested)")

        # Priority 2: Newly evolved agent (ALWAYS gets a slot if provided)
        print("\nPriority 2 - Newly Evolved Agent:")
        if evolved_agent_id and evolved_agent_id in available:
            # If we're at capacity, randomly drop a pending winner to make room
            if len(selected) >= self.agents_per_iteration:
                dropped = random.choice(selected)
                selected.remove(dropped)
                available.append(dropped)
                print(f"  ⚠️  At capacity - randomly dropping pending winner: {dropped}")
                print(f"     (Will remain a pending winner for future iterations)")

            selected.append(evolved_agent_id)
            available.remove(evolved_agent_id)
            print(f"  ✓ Selected: {evolved_agent_id} (just evolved)")
        elif evolved_agent_id and evolved_agent_id not in available:
            print(f"  ✗ Evolved agent {evolved_agent_id} not available")
        else:
            print("  ✗ No evolution occurred this iteration")

        # If we already have enough agents, return what we have
        if len(selected) >= self.agents_per_iteration:
            print(f"\n🎯 Final Greedy Selection: {selected[:self.agents_per_iteration]}")
            print("=" * 60)

            final_selected = selected[:self.agents_per_iteration]
            for agent_id in final_selected:
                self.performance_records[agent_id]['last_test_iteration'] = iteration

            # Clear the greedy flag
            self.evolver.use_greedy_selection = False

            return final_selected

        # Priority 3: Untested agents
        untested = [a for a in available if self.performance_records[a]['test_count'] == 0]
        tested = [a for a in available if self.performance_records[a]['test_count'] > 0]

        slots_remaining = self.agents_per_iteration - len(selected)

        print("\nPriority 3 - Untested Agents:")
        if untested and slots_remaining > 0:
            if len(untested) > slots_remaining:
                # Randomly select from untested agents
                print(f"  ✓ Selecting {slots_remaining} untested agent(s) from pool of {len(untested)}:")
                print(f"    Pool: {untested}")
                untested_selected = random.sample(untested, slots_remaining)
                selected.extend(untested_selected)
                print(f"    Selected: {untested_selected} (random selection)")
            else:
                # Take all untested agents if we have fewer than needed
                selected.extend(untested)
                print(f"  ✓ Selected all {len(untested)} untested agent(s): {untested}")
            slots_remaining = self.agents_per_iteration - len(selected)
        else:
            if not untested:
                print("  ✗ No untested agents available")
            else:
                print("  ✗ No slots remaining for untested agents")

        # Priority 4: Deterministic top-k ELO selection (GREEDY DIFFERENCE)
        if slots_remaining > 0 and tested:
            print("\nPriority 4 - Deterministic Top-k ELO Selection:")
            # Sort tested agents by ELO
            sorted_tested = sorted(tested,
                                 key=lambda a: self.performance_records[a]['elo'],
                                 reverse=True)

            # Greedy: Take top k deterministically (no randomization)
            num_to_select = min(slots_remaining, len(sorted_tested))
            candidate_pool = sorted_tested[:num_to_select]

            print(f"  Mode: Deterministic top-{num_to_select} by ELO (greedy)")
            print(f"  Need to fill: {slots_remaining} slot(s)")
            print(f"  Selected agents (top {num_to_select} by ELO):")
            for i, agent in enumerate(candidate_pool, 1):
                elo = self.performance_records[agent]['elo']
                test_count = self.performance_records[agent]['test_count']
                print(f"    {i}. {agent} (ELO: {elo:.0f}, tested: {test_count} times)")

            selected.extend(candidate_pool)
        elif slots_remaining > 0:
            print("\nPriority 4 - Deterministic Top-k ELO Selection:")
            print("  ✗ No tested agents available for ELO-based selection")

        print(f"\n🎯 Final Greedy Selection: {selected[:self.agents_per_iteration]}")
        print("=" * 60)

        # Update last_test_iteration for all selected agents
        final_selected = selected[:self.agents_per_iteration]
        for agent_id in final_selected:
            self.performance_records[agent_id]['last_test_iteration'] = iteration

        # Clear the greedy flag
        self.evolver.use_greedy_selection = False

        return final_selected

    def select_agents_for_iteration(self, iteration: int,
                                    evolved_agent_id: Optional[str] = None,
                                    skip_evolution: bool = False) -> List[str]:
        """
        Select agents to test in this iteration.

        Priority order:
        1. Pending Winners (all winners not yet retested)
        2. Newly evolved agent (if provided)
        3. Untested agents (test_count == 0)
        4. ELO-based selection:
           - With evolution: Random from top 2*j agents
           - Without evolution: Deterministic top j agents

        Args:
            iteration: Current iteration
            evolved_agent_id: ID of newly evolved agent to include (if any)
            skip_evolution: If True, use deterministic top ELO selection

        Returns:
            List of agent IDs to test
        """
        # Check if this is a greedy round
        if self.evolver.use_greedy_selection:
            return self._select_greedy_agents(iteration, evolved_agent_id)

        # Check if this is a challenger round
        if self.evolver.use_challenger_selection:
            return self._select_challenger_agents(iteration)

        selected = []
        available = list(self.agent_pool.keys())

        # Print selection header
        print(f"\n📋 AGENT SELECTION FOR ITERATION {iteration}")
        print("═" * 60)

        # Priority 1: Pending Winners (expanded from "Previous Winner")
        print("\nPriority 1 - Pending Winners:")
        pending_winners = self._get_pending_winners()
        if pending_winners:
            # Filter to only available agents
            available_pending = [agent for agent in pending_winners if agent in available]

            if available_pending:
                # Take up to k pending winners
                num_to_select = min(self.agents_per_iteration, len(available_pending))
                chosen = available_pending[:num_to_select]

                for agent_id in chosen:
                    selected.append(agent_id)
                    available.remove(agent_id)
                    last_win = self.performance_records[agent_id]['last_win_iteration']
                    elo_score = self.performance_records[agent_id]['elo']
                    print(f"  ✓ Selected: {agent_id} (won iteration {last_win}, ELO: {elo_score:.0f})")

                print(f"  Found {len(pending_winners)} total pending winner(s), selected {len(chosen)}")
            else:
                print(f"  Found {len(pending_winners)} pending winner(s), but none available")
        else:
            print("  ✗ No pending winners (first iteration or all winners retested)")
        
        # Priority 2: Newly evolved agent (ALWAYS gets a slot if provided)
        print("\nPriority 2 - Newly Evolved Agent:")
        if evolved_agent_id and evolved_agent_id in available:
            # If we're at capacity, randomly drop a pending winner to make room
            if len(selected) >= self.agents_per_iteration:
                dropped = random.choice(selected)  # Randomly select a pending winner to drop
                selected.remove(dropped)
                available.append(dropped)  # Return to pool for potential ELO selection
                print(f"  ⚠️  At capacity - randomly dropping pending winner: {dropped}")
                print(f"     (Will remain a pending winner for future iterations)")

            selected.append(evolved_agent_id)
            available.remove(evolved_agent_id)
            print(f"  ✓ Selected: {evolved_agent_id} (just evolved)")
        elif evolved_agent_id and evolved_agent_id not in available:
            print(f"  ✗ Evolved agent {evolved_agent_id} not available")
        else:
            print("  ✗ No evolution occurred this iteration")

        # If we already have enough agents, return what we have
        if len(selected) >= self.agents_per_iteration:
            print(f"\n🎯 Final Selection: {selected[:self.agents_per_iteration]}")
            print("=" * 60)

            # Update last_test_iteration before returning
            final_selected = selected[:self.agents_per_iteration]
            for agent_id in final_selected:
                self.performance_records[agent_id]['last_test_iteration'] = iteration

            return final_selected
        
        # Priority 3: Untested agents
        untested = [a for a in available if self.performance_records[a]['test_count'] == 0]
        tested = [a for a in available if self.performance_records[a]['test_count'] > 0]
        
        slots_remaining = self.agents_per_iteration - len(selected)
        
        print("\nPriority 3 - Untested Agents:")
        if untested and slots_remaining > 0:
            if len(untested) > slots_remaining:
                # Randomly select from untested agents
                print(f"  ✓ Selecting {slots_remaining} untested agent(s) from pool of {len(untested)}:")
                print(f"    Pool: {untested}")
                untested_selected = random.sample(untested, slots_remaining)
                selected.extend(untested_selected)
                print(f"    Selected: {untested_selected} (random selection)")
            else:
                # Take all untested agents if we have fewer than needed
                selected.extend(untested)
                print(f"  ✓ Selected all {len(untested)} untested agent(s): {untested}")
            slots_remaining = self.agents_per_iteration - len(selected)
        else:
            if not untested:
                print("  ✗ No untested agents available")
            else:
                print("  ✗ No slots remaining for untested agents")
        
        # Priority 4: ELO-based selection (threshold: > 1500)
        if slots_remaining > 0 and tested:
            print("\nPriority 4 - ELO-Based Selection:")

            # Filter to high-performing agents (ELO > 1500)
            high_elo = [(a, self.performance_records[a]['elo'])
                        for a in tested
                        if self.performance_records[a]['elo'] > 1500]
            high_elo.sort(key=lambda x: x[1], reverse=True)

            if high_elo:
                # Random selection from top 2*k high-ELO agents
                pool_size = min(slots_remaining * 2, len(high_elo))
                candidate_pool = [a for a, _ in high_elo[:pool_size]]
                num_to_select = min(slots_remaining, len(candidate_pool))

                print(f"  Mode: Random selection from top {pool_size} agents (ELO > 1500)")
                print(f"  Need to fill: {slots_remaining} slot(s)")
                print(f"  Candidate pool:")
                for agent, elo in high_elo[:pool_size]:
                    test_count = self.performance_records[agent]['test_count']
                    print(f"    - {agent} (ELO: {elo:.0f}, tested: {test_count} times)")

                elo_selected = random.sample(candidate_pool, num_to_select)
                selected.extend(elo_selected)
                slots_remaining -= len(elo_selected)
                print(f"  Selected: {elo_selected} (random from pool)")
            else:
                print(f"  ⚠️ No agents with ELO > 1500 available")

            # Fallback: agents with ELO <= 1500, deterministic by ELO
            if slots_remaining > 0:
                already_selected = set(selected)
                low_elo = [(a, self.performance_records[a]['elo'])
                           for a in tested
                           if a not in already_selected and self.performance_records[a]['elo'] <= 1500]
                low_elo.sort(key=lambda x: x[1], reverse=True)

                if low_elo:
                    print(f"\n  Fallback: Filling {slots_remaining} slot(s) from agents with ELO ≤ 1500 (by ELO)")
                    for agent, elo in low_elo[:slots_remaining]:
                        selected.append(agent)
                        test_count = self.performance_records[agent]['test_count']
                        print(f"    ✓ {agent} (ELO: {elo:.0f}, tested: {test_count} times)")
        elif slots_remaining > 0:
            print("\nPriority 4 - ELO-Based Selection:")
            print("  ✗ No tested agents available for ELO-based selection")
        
        print(f"\n🎯 Final Selection: {selected[:self.agents_per_iteration]}")
        print("=" * 60)

        # Update last_test_iteration for all selected agents
        final_selected = selected[:self.agents_per_iteration]
        for agent_id in final_selected:
            self.performance_records[agent_id]['last_test_iteration'] = iteration

        return final_selected
    
    def _validate_datasets(self):
        """Validate dataset integrity using validation scripts."""
        datasets_dir = Path(__file__).parent.parent / "benchmark_resources" / "datasets"

        # Determine which validation script to run
        if self.dev_eval_mode:
            script_name = "validate_dev.sh"
            dataset_name = "dev"
        else:
            script_name = "validate_train.sh"
            dataset_name = f"{self.dataset} and train-filtered"

        script_path = datasets_dir / script_name

        if not script_path.exists():
            raise RuntimeError(f"Validation script not found: {script_path}")

        print(f"\n🔍 Validating {dataset_name} dataset integrity...")

        try:
            result = subprocess.run(
                [str(script_path)],
                cwd=str(datasets_dir),
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                print("✓ Dataset validation passed")
            else:
                print(f"\n❌ Dataset validation failed!")
                print(result.stdout)
                if result.stderr:
                    print(result.stderr)
                print("\nPlease fix dataset issues before continuing.")
                raise RuntimeError(f"Dataset validation failed for {dataset_name}")

        except subprocess.TimeoutExpired as e:
            print(f"\n❌ Dataset validation timed out after 30 seconds!")
            raise RuntimeError(f"Dataset validation timed out for {dataset_name}") from e
        except RuntimeError:
            # Re-raise our own RuntimeError (from validation failure above)
            raise
        except Exception as e:
            print(f"\n❌ Dataset validation error: {e}")
            raise RuntimeError(f"Dataset validation failed for {dataset_name}") from e

    def run(self, initial_agents: Optional[List[str]] = None):
        """
        Run the complete parallel agent research experiment.

        Args:
            initial_agents: Optional list of specific agents to start with
        """
        start_time = time.time()

        # Validate mutually exclusive config options
        iter1_config = self.config_manager.get_config(1)
        if iter1_config.get("oldest_agent_wins_ties") and iter1_config.get("random_agent_wins_ties"):
            raise ValueError("Cannot set both oldest_agent_wins_ties and random_agent_wins_ties")

        print("\n" + "="*60)
        print("PARALLEL AGENT RESEARCH EXPERIMENT" + (" (RESUMED)" if self.resume_mode else ""))
        print("="*60)

        # Validate datasets before starting
        ## self._validate_datasets()

        # Load initial agents and strategies only if not resuming
        if not self.resume_mode:
            self.load_initial_agents(initial_agents)

            # Load initial strategies (auto-derived from all config references)
            initial_strategies = self._collect_referenced_strategies()
            self.load_initial_strategies(initial_strategies or None)

            # Load evolution strategies from experiment directory
            self.evolver._load_evolution_strategies()
        else:
            print(f"📂 Resumed from: {self.experiment_dir}")
            print(f"📊 Agents in pool: {len(self.agent_pool)}")
            if self.resume_from_iteration:
                print(f"🔄 Restarting from iteration: {self.resume_from_iteration}")
            else:
                last_completed = self.test_history[-1] if self.test_history else 0
                print(f"🔄 Continuing from iteration: {len(self.test_history) + 1}")

            # Load evolution strategies from experiment directory (needed for resumed runs)
            self.evolver._load_evolution_strategies()

        # Determine starting iteration
        if self.resume_mode:
            start_iteration = self.resume_from_iteration if self.resume_from_iteration else len(self.test_history) + 1
        else:
            start_iteration = 1
        
        # Start background process reaper for stuck solution.py processes
        self.process_reaper.start()

        # Main research loop (using while to allow restart)
        completed_normally = True
        iteration = start_iteration
        while iteration <= self.num_iterations:
            # Check memory
            self.memory_monitor.check_memory()

            # Print iteration banner
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration}")
            print(f"{'='*60}")

            # Set current iteration for lazy evaluation (prevents future iteration caching)
            self.config_manager.set_current_iteration(iteration)

            # Get config for THIS iteration
            config = self.config_manager.get_config(iteration)

            # Update random seed for this iteration
            self.random_seed = (self.original_seed + iteration * 10000) % (2**32)
            random.seed(self.random_seed)

            # Update mutable parameters from config
            self.examples_per_iteration = config["examples_per_iteration"]
            self.agents_per_iteration = config["agents_per_iteration"]
            self.evolution_model = config["evolution_model"]
            self.max_workers = config["max_workers"]
            self.evolution_timeout = config["evolution_timeout"]
            self.llm_call_timeout = config["llm_call_timeout"]
            self.new_agent_test_rounds = config["new_agent_test_rounds"]
            self.new_agent_test_round_offset = config["new_agent_test_round_offset"]

            # Recreate evolver with current iteration's config
            self.evolver = ParallelAgentEvolver(
                experiment_dir=self.experiment_dir,
                config=config,
                domain=self.domain
            )
            # Restore evolver references
            self.evolver.test_history = self.test_history

            # Load evolution strategies (needed after recreating evolver)
            self.evolver._load_evolution_strategies()

            # Select contexts for this iteration and sort alphabetically
            contexts = sorted(random.sample(self.contexts,
                                           min(self.examples_per_iteration, len(self.contexts))))

            # Select agents to test
            if iteration == 1:
                # Randomly select initial agents
                available_agents = list(self.agent_pool.keys())
                if len(available_agents) > self.agents_per_iteration:
                    selected_agents = random.sample(available_agents, self.agents_per_iteration)
                else:
                    selected_agents = available_agents

                # No evolution in iteration 1
                self.evolution_times.append(None)
                self._current_deep_focus_fresh_evals = 0
            else:
                # Get evolution strategy and analyzer from config
                evolution_strategy = config["evolution_strategy"]

                # Check if this strategy was selected via weighted random
                # by looking at config_change_history
                was_random = False
                for entry in reversed(self.config_manager.config_change_history):
                    if entry["iteration"] == iteration and entry["source"] == "weighted_random":
                        was_random = True
                        break

                # Check if evolution should be skipped
                skip_evolution = (evolution_strategy == "none")

                # Check for greedy strategy
                if evolution_strategy == 'greedy':
                    print(f"\n🎯 Greedy round: deterministic top-k selection by ELO")
                    self.evolver.use_greedy_selection = True
                    skip_evolution = True  # No evolution, deterministic selection
                    # Track in evolution_history
                    self.evolver.evolution_history.append({
                        'iteration': iteration,
                        'strategy': 'greedy',
                        'was_random': was_random
                    })
                    # Note: evolution_times.append(None) happens in the generic skip_evolution block below

                # Check for challenger strategy
                if evolution_strategy == 'challenger':
                    print(f"\n🎯 Challenger round: targeting under-tested high-ELO agents")
                    self.evolver.use_challenger_selection = True
                    skip_evolution = True  # No evolution, but custom selection
                    # Track in evolution_history
                    self.evolver.evolution_history.append({
                        'iteration': iteration,
                        'strategy': 'challenger',
                        'was_random': was_random
                    })
                    # Note: evolution_times.append(None) happens in the generic skip_evolution block below

                evolved_agent_id = None

                if not skip_evolution and self.test_history:
                    # Create new agent based on previous results
                    recent_results = self.test_history[-1]

                    result = self.evolver.create_new_agent(
                        self.agent_pool,
                        self.performance_records,
                        recent_results,
                        iteration,
                        self.test_history,
                        strategy_name=evolution_strategy,
                        was_random=was_random
                    )
                    
                    # Check if evolution failed
                    if result is None or result[0] is None:
                        # Check for 5-hour limit restart
                        if hasattr(self.evolver, 'restart_from_iteration') and self.evolver.restart_from_iteration is not None:
                            restart_iter = self.evolver.restart_from_iteration
                            iterations_to_redo = iteration - restart_iter + 1
                            print(f"\n🔄 Restarting from iteration {restart_iter} due to 5-hour limit")
                            print(f"   Will redo {iterations_to_redo} iteration(s): {restart_iter} through {iteration}")

                            # Archive the failed iterations
                            self.archive_iterations(restart_iter)

                            # Reset the restart flag
                            self.evolver.restart_from_iteration = None

                            # Update five_hour_limit_incidents to be persistent
                            self.five_hour_limit_incidents = self.evolver.five_hour_limit_incidents.copy()

                            # Save checkpoint before restart
                            self._save_checkpoint(restart_iter - 1)

                            # Restart loop from the specified iteration
                            iteration = restart_iter
                            continue  # Skip to next iteration of while loop
                        else:
                            # Normal failure - end experiment
                            print(f"\n🏁 Ending experiment early after {iteration-1} successful iterations")
                            print(f"   Evolution failed for iteration {iteration} - cannot continue")
                            completed_normally = False
                            break
                    
                    # Unpack the result
                    new_agent_id, reasoning, package_info = result
                    evolved_agent_id = new_agent_id

                    # Extract and store evolution timing if available
                    if 'timing' in package_info:
                        self.evolution_times.append(package_info['timing'])
                    else:
                        self.evolution_times.append(None)  # For iterations without Deep Focus

                    # Store evolution cost info for later use in run_iteration()
                    # (iteration_cost_dict doesn't exist yet - it's created in run_iteration)
                    if 'cost' in package_info:
                        self.current_iteration_evolution_cost = package_info['cost']
                    else:
                        self.current_iteration_evolution_cost = None

                    # Stash deep focus fresh eval count (computed in create_new_agent)
                    self._current_deep_focus_fresh_evals = getattr(self.evolver, '_current_deep_focus_fresh_evals', 0)

                    # Install package based on type
                    if package_info['type'] == 'three_artifact':
                        # File-mapping-driven package — install complete
                        package_dir = self._install_three_artifact_package(
                            new_agent_id, package_info, iteration
                        )
                    elif package_info['type'] in ['parsed', 'single_artifact']:
                        # Fallback: create empty agent directory
                        print(f"  📝 Creating agent from {package_info['type']} (no artifacts to copy)")
                        package_dir = self.experiment_dir / "agents" / new_agent_id
                        package_dir.mkdir(parents=True, exist_ok=True)
                    else:
                        print(f"  ⚠️ Unexpected package type: {package_info['type']}")
                        package_dir = self.experiment_dir / "agents" / new_agent_id
                        package_dir.mkdir(parents=True, exist_ok=True)

                    # Add to pool with package info
                    self.agent_pool[new_agent_id] = {
                        'source': 'evolution',
                        'created_iteration': iteration,
                        'evolution_strategy': evolution_strategy,
                        'session_id': package_info.get('session_id'),
                        'package_dir': package_dir,
                    }
                    
                    # Initialize performance record
                    self.performance_records[new_agent_id] = {
                        'test_count': 0,
                        'total_score_sum': 0.0,
                        'total_questions': 0,
                        'mean_score': 0.0,
                        'elo': 1500,
                        'iteration_results': [],
                        'last_win_iteration': None,  # Track when agent last won
                        'last_test_iteration': None  # Track when agent was last tested
                    }
                    
                    print(f"\n✨ Created new agent: {new_agent_id}")
                elif skip_evolution:
                    print(f"\n⏭️ Skipping evolution for iteration {iteration} (configured in evolution schedule)")
                    # Track that evolution was skipped
                    self.evolver.evolution_history.append({
                        'iteration': iteration,
                        'strategy': 'none (skipped)',
                        'was_random': False
                    })
                    # No evolution timing for skipped evolution
                    self.evolution_times.append(None)
                    self._current_deep_focus_fresh_evals = 0

                # Select agents for this iteration
                selected_agents = self.select_agents_for_iteration(
                    iteration, 
                    evolved_agent_id=evolved_agent_id,
                    skip_evolution=skip_evolution
                )
            
            # Track iteration timing
            iteration_start_time = time.time()
            
            # Run iteration
            iteration_results, results_by_agent, costs_by_context, eval_cache_stats = self.run_iteration(iteration, selected_agents, contexts)

            # Calculate iteration metrics
            iteration_time = time.time() - iteration_start_time

            # Store per-iteration metrics
            self.iteration_times.append(iteration_time)

            # Track fresh (non-cached) evaluations for evaluation_budget
            iteration_fresh = sum(
                eval_cache_stats.get(aid, {}).get('fresh', iteration_results[aid]['total'])
                for aid in iteration_results
            )
            iteration_fresh += self._current_deep_focus_fresh_evals
            self._current_deep_focus_fresh_evals = 0
            self.iteration_fresh_evals.append(iteration_fresh)

            # Initialize meta-evolution time to 0 (will be updated if meta-evolution runs)
            self.meta_evolution_times.append(0)

            # Store results (already done in run_iteration before ELO calculation)

            # Generate interim report after this iteration
            self.report_generator.generate_interim_report(start_time, iteration)

            # Generate cost analysis report
            self._generate_iteration_cost_report(iteration, results_by_agent, costs_by_context, eval_cache_stats)

            # Generate comparative error analysis
            self._generate_comparative_analysis(iteration)

            # PHASE 1: Save checkpoint (protects iteration work)
            self._save_checkpoint(iteration)

            # Validate config consistency after iteration
            is_valid, errors = self.config_manager.validate_consistency(iteration)
            if not is_valid:
                logger.error(f"❌ Config consistency validation failed after iteration {iteration}:")
                for error in errors:
                    logger.error(f"  - {error}")
                raise RuntimeError(f"Config validation failed at iteration {iteration}")

            # PHASE 2: Run meta-evolution if configured
            if self.meta_evolution_manager.should_run_meta_evolution(iteration):
                meta_start_time = time.time()
                try:
                    # Run meta-evolution
                    meta_result = self.meta_evolution_manager.run_meta_evolution(iteration)

                    # Store meta-evolution costs
                    if len(self.iteration_claude_costs) >= iteration:
                        self.iteration_claude_costs[iteration - 1]['meta_evolution_cost'] = meta_result.cost_data.get('total_cost', 0.0)
                        self.iteration_claude_costs[iteration - 1]['meta_evolution_calls'] = meta_result.cost_data.get('calls', 0)
                        self.iteration_claude_costs[iteration - 1]['meta_evolution_tokens_in'] = meta_result.cost_data.get('tokens_in', 0)
                        self.iteration_claude_costs[iteration - 1]['meta_evolution_tokens_out'] = meta_result.cost_data.get('tokens_out', 0)

                    # If meta-evolution proposed changes, integrate them
                    if meta_result.meta_config_schedule:
                        self.config_manager.integrate_meta_config_schedule(meta_result.meta_config_schedule, iteration)
                        logger.info(f"✓ Integrated meta_config_schedule with {len(meta_result.meta_config_schedule)} iteration changes")

                    if meta_result.config_delta:
                        self.config_manager.apply_delta(
                            iteration=iteration + 1,
                            delta=meta_result.config_delta,
                            source=ConfigSource.META_EVOLUTION,
                            rationale=f"Immediate meta-evolution parameter adjustment: {meta_result.config_delta}"
                        )
                        logger.info(f"✓ Applied immediate config delta: {meta_result.config_delta}")

                    # Save checkpoint again with meta-evolution results
                    self._save_checkpoint(iteration)

                    # Validate after meta-evolution
                    is_valid, errors = self.config_manager.validate_consistency(iteration)
                    if not is_valid:
                        logger.error("❌ Config consistency validation failed after meta-evolution:")
                        for error in errors:
                            logger.error(f"  - {error}")
                        raise RuntimeError("Meta-evolution broke config consistency")

                except Exception as e:
                    # Meta-evolution failed - iteration work is already saved
                    logger.error(f"❌ Meta-evolution failed: {e}")
                    logger.info("Triggering graceful termination...")

                    # Generate final report
                    self.process_reaper.stop()
                    self.report_generator.generate_final_report(start_time)

                    # Exit gracefully
                    print(f"\n🏁 Ending experiment after {iteration} iterations due to meta-evolution failure")
                    return True
                finally:
                    # Update meta-evolution time (even if it failed)
                    self.meta_evolution_times[iteration - 1] = time.time() - meta_start_time
            # else: meta-evolution not run - time remains 0 (already initialized above)

            # Check budget and maybe terminate
            if self.meta_evolution_manager.check_budget_and_maybe_terminate(iteration):
                # Budget exhausted - generate final report before terminating
                self.process_reaper.stop()
                self.report_generator.generate_final_report(start_time)
                print(f"\n🏁 Ending experiment after {iteration} iterations due to budget exhaustion")
                return True

            # Check evaluation budget
            evaluation_budget = config.get('evaluation_budget')
            if evaluation_budget is not None:
                total_fresh = sum(self.iteration_fresh_evals)
                logger.info(f"Evaluation budget: {total_fresh}/{evaluation_budget} fresh evals")
                if total_fresh >= evaluation_budget:
                    self.process_reaper.stop()
                    self.report_generator.generate_final_report(start_time)
                    print(f"\n🏁 Ending experiment after {iteration} iterations: evaluation budget exhausted ({total_fresh}/{evaluation_budget} fresh evals)")
                    return True

            # Increment iteration for next loop
            iteration += 1

        # Stop background process reaper
        self.process_reaper.stop()

        # Generate final report
        self.report_generator.generate_final_report(start_time)

        total_time = time.time() - start_time
        print(f"\n✅ Research complete!")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Results saved to: {self.experiment_dir}")

        return completed_normally

    def _install_three_artifact_package(self, agent_id: str, package_info: Dict, iteration: int) -> Path:
        """
        Install an agent package to the experiment agents directory.

        Copies artifacts listed in package_info['artifact_paths'] (from
        file_mapping) to the agent directory.

        Args:
            agent_id: ID for the agent
            package_info: Package information from evolution
            iteration: Current iteration number

        Returns:
            Path to the installed package directory
        """
        package_dir = self.experiment_dir / "agents" / agent_id
        package_dir.mkdir(parents=True, exist_ok=True)

        artifact_paths = package_info.get('artifact_paths', {})

        for rel_path, src_path in artifact_paths.items():
            src = Path(src_path)
            if not src.exists():
                print(f"  ⚠️  Artifact not found: {src}")
                continue
            dst = package_dir / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst, symlinks=True)
            else:
                shutil.copy2(src, dst)

        print(f"  📦 Installed agent package to {package_dir.name}")

        return package_dir

    def _serialize_evolution_default(self, default: Tuple[str, Optional[str]]) -> Union[str, List]:
        """Convert evolution default tuple to JSON-serializable format."""
        strategy, analyzer = default
        if analyzer is None:
            return strategy
        else:
            return [strategy, analyzer]

    def _serialize_evolution_schedule(self, schedule: Dict[int, Tuple[str, Optional[str]]]) -> Dict[int, Union[str, List]]:
        """Convert evolution schedule tuples to JSON-serializable format."""
        serialized = {}
        for iteration, (strategy, analyzer) in schedule.items():
            if analyzer is None:
                serialized[iteration] = strategy
            else:
                serialized[iteration] = [strategy, analyzer]
        return serialized

    def _serialize_weighted_random(self, weighted_random: Dict[Tuple[str, Optional[str]], int]) -> List:
        """Convert weighted random dict to JSON-serializable array format."""
        if not weighted_random:
            return []

        result = []
        for (strategy, analyzer), weight in weighted_random.items():
            if analyzer is None:
                strategy_spec = strategy
            else:
                strategy_spec = [strategy, analyzer]
            result.append([strategy_spec, weight])
        return result

    def _generate_comparative_analysis(self, iteration: int):
        """
        Generate comparative error analysis for completed iteration.

        Runs create_comparative_error_index.py to generate:
        - error_index.json: Structured data for programmatic access
        - error_analysis_report.md: Human-readable summary

        Args:
            iteration: Iteration number to analyze
        """
        iteration_dir = self.experiment_dir / f"iteration_{iteration:03d}"

        if not iteration_dir.exists():
            print(f"⚠️ Cannot generate analysis: iteration directory not found: {iteration_dir}")
            return

        print(f"📊 Generating comparative error analysis for iteration {iteration}")

        # Paths
        error_index_path = iteration_dir / "error_index.json"
        error_report_path = iteration_dir / "error_analysis_report.md"

        try:
            # Run create_comparative_error_index.py
            result = subprocess.run(
                [
                    sys.executable, "-m",
                    "RoboPhD.tools.error_analysis.create_comparative_error_index",
                    "--iteration-dir", str(iteration_dir),
                    "--output", str(error_index_path)
                ],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                print(f"❌ Failed to generate error index: {result.stderr}")
                return

            # Generate simple markdown report from JSON
            if error_index_path.exists():
                with open(error_index_path, 'r') as f:
                    index = json.load(f)

                # Create markdown report
                report_lines = [
                    f"# Comparative Agent Analysis - Iteration {iteration:03d}",
                    "",
                    f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    ""
                ]

                from RoboPhD.report_generator import (
                    is_continuous_scoring, format_continuous_score_table,
                    format_binary_report_comparative, format_agent_errors,
                )
                scores_by_question = index.get('scores_by_question', {})
                summary = index.get('summary', {})
                agents = summary.get('agents', [])
                total_q = summary.get('total_questions', 0)

                if scores_by_question and is_continuous_scoring(scores_by_question):
                    report_lines.append(f"**Agents**: {', '.join(agents)}")
                    report_lines.append(f"**Total problems**: {total_q}")
                    report_lines.append("")
                    report_lines.extend(format_continuous_score_table(scores_by_question, agents))
                    report_lines.extend(format_agent_errors(index))
                else:
                    report_lines.extend(format_binary_report_comparative(index))

                # Append clone detection section if any clones found this iteration
                iteration_clones = [(c, m) for c, m, i in self.clone_detections if i == iteration]
                if iteration_clones:
                    report_lines.append("## Clone Detection")
                    report_lines.append("")
                    for clone_id, matched_id in iteration_clones:
                        report_lines.append(
                            f"⚠️ **{clone_id}** identified as exact clone of **{matched_id}** "
                            f"(identical scores on all {total_q} problems). "
                            f"ELO penalized by 200; excluded from winner selection."
                        )
                    report_lines.append("")

                # Write report
                with open(error_report_path, 'w') as f:
                    f.write('\n'.join(report_lines))

                print(f"✓ Generated error analysis: {error_report_path}")
            else:
                print(f"⚠️ Error index not generated: {error_index_path}")

        except subprocess.TimeoutExpired:
            print(f"❌ Error analysis generation timed out after 300 seconds")
        except Exception as e:
            print(f"❌ Failed to generate error analysis: {e}")

    def _generate_iteration_cost_report(self, iteration: int, results_by_agent: Dict,
                                         costs_by_context: Optional[Dict] = None,
                                         eval_cache_stats: Optional[Dict] = None):
        """
        Generate cost analysis report for this iteration.

        Args:
            iteration: Current iteration number
            results_by_agent: Dict mapping agent_id to list of result dicts
            costs_by_context: Dict mapping agent_id to {context: {'eval': $}}
            eval_cache_stats: Dict mapping agent_id to {'cached': N, 'fresh': M}
        """
        iteration_dir = self.experiment_dir / f"iteration_{iteration:03d}"
        if not iteration_dir.exists():
            return

        if not results_by_agent:
            return

        print(f"📊 Generating cost analysis for iteration {iteration}")

        # Collect all contexts and agents
        all_contexts = set()
        all_agents = sorted(results_by_agent.keys())

        # Build cost matrix
        cost_matrix = {}  # {context: {agent_id: {'eval': $}}}

        for agent_id in all_agents:
            if costs_by_context and agent_id in costs_by_context:
                for context_name, costs in costs_by_context[agent_id].items():
                    all_contexts.add(context_name)
                    if context_name not in cost_matrix:
                        cost_matrix[context_name] = {}
                    cost_matrix[context_name][agent_id] = {
                        'eval': costs.get('eval', 0.0),
                    }
            else:
                for result in results_by_agent.get(agent_id, []):
                    context_name = result['context']
                    all_contexts.add(context_name)
                    if context_name not in cost_matrix:
                        cost_matrix[context_name] = {}
                    eval_cost = result.get('eval_cost', 0.0)
                    cost_matrix[context_name][agent_id] = {
                        'eval': eval_cost,
                    }

        sorted_contexts = sorted(all_contexts)

        # Calculate totals
        total_eval = 0.0
        ctx_totals = {}
        agent_totals = {agent: {'eval': 0.0} for agent in all_agents}

        for ctx_name in sorted_contexts:
            ctx_totals[ctx_name] = {'eval': 0.0}
            for agent_id in all_agents:
                if agent_id in cost_matrix.get(ctx_name, {}):
                    costs = cost_matrix[ctx_name][agent_id]
                    ctx_totals[ctx_name]['eval'] += costs['eval']
                    agent_totals[agent_id]['eval'] += costs['eval']
                    total_eval += costs['eval']

        num_tests = sum(len(results) for results in results_by_agent.values())

        context_label = "Problems"

        # Generate markdown report
        report_lines = [
            f"# Cost Analysis - Iteration {iteration}",
            "",
            f"**Total Evaluation Cost: ${total_eval:.2f}**",
            "",
            f"**Agents Tested**: {len(all_agents)} agents",
            f"**{context_label} Tested**: {len(sorted_contexts)} {context_label.lower()}",
            f"**Total Tests**: {num_tests} (agent x {context_label.lower()[:-1]} pairs)",
            "",
        ]

        # Agent cost summary table
        has_cache = eval_cache_stats and any(
            s.get('cached', 0) > 0 for s in eval_cache_stats.values()
        )

        if has_cache:
            header = "| Agent | Eval Cost | Cached | Total |"
            separator = "|-------|-----------|--------|-------|"
        else:
            header = "| Agent | Eval Cost |"
            separator = "|-------|-----------|"

        report_lines.extend(["## Agent Cost Summary", "", header, separator])

        for agent_id in all_agents:
            at = agent_totals[agent_id]
            if has_cache:
                cs = eval_cache_stats.get(agent_id, {})
                cached = cs.get('cached', 0)
                total_problems = cached + cs.get('fresh', 0)
                cache_str = f"{cached}/{total_problems}" if cached > 0 else "-"
                report_lines.append(
                    f"| {agent_id} | ${at['eval']:.2f} | {cache_str} | **${at['eval']:.2f}** |"
                )
            else:
                report_lines.append(
                    f"| {agent_id} | **${at['eval']:.2f}** |"
                )

        # Total row
        if has_cache:
            total_cached = sum(s.get('cached', 0) for s in eval_cache_stats.values())
            total_all = sum(s.get('cached', 0) + s.get('fresh', 0) for s in eval_cache_stats.values())
            report_lines.append(
                f"| **Total** | **${total_eval:.2f}** | **{total_cached}/{total_all}** | **${total_eval:.2f}** |"
            )
        else:
            report_lines.append(
                f"| **Total** | **${total_eval:.2f}** |"
            )

        # Cost insights
        report_lines.extend(["", "---", "", "## Cost Insights", ""])

        # Most expensive agents
        agent_costs = [(agent, agent_totals[agent]['eval']) for agent in all_agents]
        agent_costs.sort(key=lambda x: x[1], reverse=True)

        report_lines.append("### Most Expensive Agents")
        context_label_singular = "problem"
        for i, (agent, cost) in enumerate(agent_costs, 1):
            pct = (cost / total_eval * 100) if total_eval > 0 else 0
            avg = cost / len(sorted_contexts) if sorted_contexts else 0
            report_lines.append(f"{i}. {agent}: ${cost:.2f} ({pct:.1f}%, avg ${avg:.2f}/{context_label_singular})")
        report_lines.append("")

        # Write report
        cost_report_path = iteration_dir / "cost_report.md"
        with open(cost_report_path, 'w') as f:
            f.write('\n'.join(report_lines))

        print(f"✓ Generated cost report: {cost_report_path}")

    def _validate_checkpoint_consistency(self, iteration: int):
        """
        Validate that checkpoint data structures are consistent with completed iterations.

        Args:
            iteration: Number of completed iterations

        Raises:
            RuntimeError: If any inconsistencies are detected
        """
        errors = []

        # All these arrays should have exactly 'iteration' entries
        arrays_to_check = {
            'iteration_times': self.iteration_times,
            'test_history': self.test_history,
            'evolution_times': self.evolution_times,
            'meta_evolution_times': self.meta_evolution_times,
            'iteration_claude_costs': self.iteration_claude_costs,
            'iteration_fresh_evals': self.iteration_fresh_evals
        }

        for name, array in arrays_to_check.items():
            expected_length = iteration
            actual_length = len(array)

            if actual_length != expected_length:
                errors.append(
                    f"{name}: expected {expected_length} entries, found {actual_length}"
                )

        if errors:
            error_msg = "Checkpoint consistency validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.debug(f"✓ Checkpoint consistency validated: {iteration} iterations, all arrays match")

    def _save_checkpoint(self, iteration: int):
        """Save checkpoint after each iteration."""
        # Validate checkpoint consistency before saving
        self._validate_checkpoint_consistency(iteration)

        # Convert agent_pool to serializable format
        serializable_pool = {}
        for agent_id, agent_info in self.agent_pool.items():
            serializable_agent = {
                'source': agent_info.get('source', 'unknown'),
                'created_iteration': agent_info.get('created_iteration', 0),
                'evolution_strategy': agent_info.get('evolution_strategy', None),
                'session_id': agent_info.get('session_id', None),
            }

            # Save package_dir as relative path from experiment directory for portability
            if 'package_dir' in agent_info:
                package_dir_path = Path(agent_info['package_dir'])
                try:
                    relative_path = package_dir_path.relative_to(self.experiment_dir)
                    serializable_agent['package_dir'] = str(relative_path)
                except ValueError:
                    serializable_agent['package_dir'] = str(package_dir_path)

            serializable_pool[agent_id] = serializable_agent
        
        checkpoint = {
            'last_completed_iteration': iteration,
            'num_iterations': self.num_iterations,
            'random_seed': self.random_seed,
            'agent_pool': serializable_pool,
            'performance_records': self.performance_records,
            'test_history': self.test_history,
            'iteration_times': self.iteration_times,
            'iteration_claude_costs': self.iteration_claude_costs,
            'iteration_fresh_evals': self.iteration_fresh_evals,
            'evolution_times': self.evolution_times,
            'meta_evolution_times': self.meta_evolution_times,
            'zero_accuracy_cases': self.zero_accuracy_cases,
            'exception_failures': self.exception_failures,
            'five_hour_limit_incidents': self.five_hour_limit_incidents,
            'clone_detections': self.clone_detections,
            'config_manager': self.config_manager.to_checkpoint(),
            'task_config': self.task_config,
        }

        # Preserve meta_evolution_session_id and meta_evolution_session_created if they exist.
        # Currently unused — meta-evolution creates a fresh session each iteration.
        # This scaffolding would be needed for cross-iteration session persistence.
        checkpoint_file = self.experiment_dir / 'checkpoint.json'
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r') as f:
                    existing_checkpoint = json.load(f)
                    if 'meta_evolution_session_id' in existing_checkpoint:
                        checkpoint['meta_evolution_session_id'] = existing_checkpoint['meta_evolution_session_id']
                    if 'meta_evolution_session_created' in existing_checkpoint:
                        checkpoint['meta_evolution_session_created'] = existing_checkpoint['meta_evolution_session_created']
            except:
                pass  # If reading fails, just proceed without preserving

        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)



