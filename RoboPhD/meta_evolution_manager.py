"""
Meta-Evolution Manager for RoboPhD Phase 3.

Orchestrates meta-evolution process that analyzes research system performance
and creates new evolution strategies while adjusting configuration parameters.

Architecture:
- Single Claude Code session per meta-evolution call
- Three-phase execution: Planning & Implementation → Validation → Installation
- Budget tracking across all phases (Evaluation + Evolution + Meta-evolution)
- Automatic termination when budget exhausted
"""

import json
import logging
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Any

from RoboPhD.config_manager import ConfigManager, ConfigSource
from utilities.claude_cli import call_claude_cli, RateLimitExceeded

logger = logging.getLogger(__name__)


class MetaEvolutionResult(NamedTuple):
    """Result from a meta-evolution run."""
    meta_config_schedule: Optional[Dict[str, Any]]
    config_delta: Optional[Dict[str, Any]]
    cost_data: Dict[str, Any]

META_EVOLUTION_ENVIRONMENT_GUIDE = """\
# Meta-Evolution Environment

## Cadence

You are called every {cadence} iterations: first firing at iter {first_iteration}, then iter {first_plus_cadence}, {first_plus_2cadence}, … Plan your `meta_config_schedule.json` decisions with this {cadence}-iteration horizon in mind — any change you propose will run for ~{cadence} iterations before you see its effect and can revise.

The Claude Code session persists across all firings within a run; subsequent firings deliver brief status updates against this same session.

## Working Directory

Your working directory is the run's `meta_evolution_output/` directory, which is stable across all firings within a run (so the persistent Claude Code session can be resumed each iteration). Iteration-specific subdirectories live as children:
- `iteration_NNN/` — per-firing output (reasoning.md, meta_config_schedule.json, new_strategies/, etc.)
- `../iteration_NNN/` — per-iteration outputs from the main run (interim_report.md, cost_report.md, error_analysis_report.md, agent dirs)
- `../evolution_strategies/` — installed evolution strategies (yours land here after validation)

## Per-Iteration Reports

These reports are generated after each iteration at `../iteration_NNN/` (relative to your working dir):
- `error_analysis_report.md` — cross-agent score comparison & failure summary
- `error_index.json` — raw per-problem score data (source for the report)
- `cost_report.md` — per-agent LLM cost breakdown (tokens, cache hits, USD)

## CLI Tools

{cli_tools}

## Required Outputs

Each firing must produce, at minimum:
- `iteration_NNN/reasoning.md` — your analysis. The format and content are whatever your meta-evolution strategy specifies.
- `iteration_NNN/meta_config_schedule.json` — config changes for upcoming iterations. Can be empty (`{{}}`) if you propose no schedule changes; the file itself must exist.

Optional, only if your strategy authorizes:
- `iteration_NNN/config_delta.json` — immediate parameter change starting next iteration (persists until overwritten).
- `iteration_NNN/new_strategies/<name>/strategy.md` — a new evolution strategy.

Missing required outputs trigger a correction prompt within the same session; persistent failure terminates the run.

## Strategy Packages

If you create a new evolution strategy, it lives at `iteration_NNN/new_strategies/<name>/` as a package containing:
- `strategy.md` (required) — YAML frontmatter with `name` and `description` fields, followed by instructions for the evolution AI on how to create agents.
- `strategy_tools/` (optional) — Python helper scripts the evolution AI can run (custom error analysis, state tracking, specialized reports). Details below.

Review existing strategies under `../evolution_strategies/` for patterns and structure to follow. The format and content of `reasoning.md` is whatever your meta-evolution strategy specifies.

### Strategy tools details

When you include `strategy_tools/` in a package, those tools are **symlinked into the evolution working directory** as `strategy_tools/`. Reference them as `python strategy_tools/<script>.py` in your strategy.md instructions.

- Tools should use only stdlib and libraries already installed in the environment
- Include `--help` support so Claude can discover usage
- Reference them with imperative language in strategy.md (e.g., "Run `python strategy_tools/analyze_failures.py ...`" not "If the tool is available...")
- The symlink will exist — do NOT include fallback instructions suggesting the tool might be missing

## Strategy Naming

Pick a hyphenated, lowercase name for your strategy (e.g. `cost_mechanism_aware`) — it ends up in installed paths and the schedule, so legibility matters.

When you create a new evolution strategy at `iteration_NNN/new_strategies/<name>/`, the system installs it as `evolution_strategies/iter{{N}}_<name>/` — your name is automatically prefixed with `iter{{N}}_` to keep each firing's contribution unique (mirroring how evolved agents get an iter prefix). **Reference the prefixed form in `meta_config_schedule.json`.**

For example, if at iteration 7 you create `new_strategies/cost_mechanism_aware/`, it installs as `evolution_strategies/iter7_cost_mechanism_aware/`. Your schedule should reference `"evolution_strategy": "iter7_cost_mechanism_aware"`, not `"cost_mechanism_aware"`.

If you reference a name that doesn't resolve, you'll get a correction prompt with the full list of installed strategies and the prefixed form of any strategy you just created.

## Configuration Persistence

Configurations persist across iterations once set. A `meta_config_schedule.json` entry like `{{"4": {{"evolution_strategy": "X"}}}}` does NOT mean "use X at iteration 4 only" — it means "starting at iteration 4, use X until another entry overrides it." To restrict X to a single iteration, schedule both the change AND the revert: `{{"4": {{"evolution_strategy": "X"}}, "5": {{"evolution_strategy": "Y"}}}}`.

## Schedule Format

`meta_config_schedule.json` is a top-level mapping from iteration-number strings to delta dicts. Example:

```json
{{
  "11": {{"evolution_strategy": "iter11_my_strategy"}},
  "13": {{"evolution_strategy": "iter4_my_other_strategy"}}
}}
```

Iteration 11 starts using `iter11_my_strategy` (a strategy you just created); iteration 12 inherits it (no override scheduled); iteration 13 switches to `iter4_my_other_strategy` (an older strategy you created in a prior firing). See Configuration Persistence above for the inheritance rule.

## Horizon

The run may be extended beyond the current iteration count. Don't treat any iteration as "final" or optimize for a specific end point — make decisions based on strategy performance trends, not on how many iterations remain.

## Framework Behavior (post-firing)

After your firing completes, the framework will:
- Discover strategies by scanning `iteration_NNN/new_strategies/` for subdirectories
- Validate each strategy package (frontmatter, syntax) and prompt you to correct any errors
- Install valid strategies to `evolution_strategies/iter{{N}}_<name>/`
- Validate that every `evolution_strategy` reference in your schedule resolves to an installed strategy; prompt for correction if not
- Integrate `meta_config_schedule` via ConfigManager (your changes take effect at their scheduled iterations)"""


class MetaEvolutionManager:
    """
    Manages meta-evolution process for optimizing research system configuration.

    Meta-evolution runs as the last step of iteration K, analyzing K's results
    and proposing configuration changes for K+1 and beyond.
    """

    def __init__(
        self,
        experiment_dir: Path,
        config_manager: ConfigManager,
        domain_name: str,
        domain=None,
        session_id: Optional[str] = None,
        initial_firing_complete: bool = False,
    ):
        """
        Initialize Meta-Evolution Manager.

        Args:
            experiment_dir: Root directory of research experiment
            config_manager: Configuration manager for the experiment
            domain_name: Domain identifier (e.g., "codegen", "text2sql")
            domain: Optional domain object for deriving header from task metadata
            session_id: Optional pre-existing Claude Code session ID. Restored from
                checkpoint on resume. If the initial firing did NOT complete (see
                ``initial_firing_complete``), this id will be discarded and a fresh
                one minted at the next firing — the abandoned transcript stays on
                disk for diagnostics.
            initial_firing_complete: True iff a prior firing successfully passed
                ``_parse_and_validate_outputs`` (so the persistent session contains
                the strategy + task background and follow-up prompts are safe).
                Restored from checkpoint on resume.
        """
        self.experiment_dir = experiment_dir
        self.config_manager = config_manager
        self.domain_name = domain_name
        self._domain = domain
        self._task_background = getattr(domain, 'task_background', '') if domain else ''
        self._task_objective = getattr(domain, 'task_objective', '') if domain else ''
        self.strategies_dir = Path("RoboPhD/meta_evolution_strategies")  # Source directory
        self.output_dir = experiment_dir / "meta_evolution_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Persistent Claude Code session shared across all firings within the run.
        self.session_id: Optional[str] = session_id
        self.initial_firing_complete: bool = initial_firing_complete

    def get_session_id(self) -> Optional[str]:
        """Return the Claude Code session id used by meta-evolution, or None if not yet started."""
        return self.session_id

    def is_initial_firing_complete(self) -> bool:
        """Return True iff a prior firing successfully completed validation (follow-up safe)."""
        return self.initial_firing_complete

    def should_run_meta_evolution(self, iteration: int) -> bool:
        """
        Check if meta-evolution should run for this iteration.

        Fires when iteration >= first_iteration and (iteration - first) % cadence == 0.
        First/cadence are read from config and locked at iteration 1 (IMMUTABLE_PARAMS).

        Args:
            iteration: Current iteration number

        Returns:
            True if meta-evolution should run this iteration.
        """
        config = self.config_manager.get_config(iteration)
        strategy = config.get("meta_evolution_strategy")
        if strategy is None or strategy == "none":
            return False
        first = config.get("meta_evolution_first_iteration", 4)
        cadence = config.get("meta_evolution_cadence", 3)
        if iteration < first:
            return False
        return (iteration - first) % cadence == 0

    def run_meta_evolution(
        self,
        iteration: int
    ) -> MetaEvolutionResult:
        """
        Execute meta-evolution for this iteration.

        All firings within a run share one continuous Claude Code session
        (`self.session_id`). The first firing carries the full setup (strategy,
        task background, cadence info); subsequent firings deliver brief status
        updates and rely on the in-session memory of prior decisions.

        Args:
            iteration: Current iteration number

        Returns:
            MetaEvolutionResult with meta_config_schedule, config_delta, and cost_data
        """
        config = self.config_manager.get_config(iteration)
        strategy_name = config["meta_evolution_strategy"]
        model = config.get("meta_evolution_model", "opus-4.5")
        cadence = config.get("meta_evolution_cadence", 3)
        first_iteration = config.get("meta_evolution_first_iteration", 4)

        # If the initial firing hasn't completed, this firing IS the initial firing.
        # Mint a fresh session id (discarding any abandoned id from a prior crashed
        # attempt — its transcript stays on disk for diagnostics) so Claude starts
        # with a clean conversation.
        is_first_firing = not self.initial_firing_complete
        if is_first_firing:
            self.session_id = str(uuid.uuid4())
        else:
            assert self.session_id is not None, (
                "Invariant: initial_firing_complete=True implies session_id is set"
            )

        firing_label = "Initial firing" if is_first_firing else "Follow-up firing"
        logger.info(f"\n{'=' * 60}")
        logger.info(f"🧬 META-EVOLUTION (Iteration {iteration}) | Strategy: {strategy_name} | "
                    f"{firing_label} | Model: {model}")
        logger.info(f"Session ID: {self.session_id}")
        logger.info(f"{'=' * 60}\n")

        # Create iteration-specific output directory
        iteration_output = self.output_dir / f"iteration_{iteration:03d}"
        iteration_output.mkdir(parents=True, exist_ok=True)

        # Track total cost for this meta-evolution run
        total_cost_data = {
            'total_cost': 0.0,
            'calls': 0,
            'tokens_in': 0,
            'tokens_out': 0,
            'cache_created': 0,
            'cache_read': 0
        }

        # PHASE 1: Planning and Implementation (initial OR follow-up firing)
        if is_first_firing:
            logger.info("📋 Phase 1: Initial planning, reasoning, and implementation...")
            context = self._gather_context(iteration, config)
            strategy = self._load_meta_strategy(strategy_name)
            cost_data = self._execute_planning_and_implementation(
                strategy=strategy,
                context=context,
                iteration=iteration,
                iteration_output=iteration_output,
                model=model,
                session_id=self.session_id,
                cadence=cadence,
                first_iteration=first_iteration,
            )
        else:
            logger.info("📋 Phase 1: Follow-up status update (resuming session)...")
            context = self._gather_context(iteration, config)
            cost_data = self._execute_followup_firing(
                iteration=iteration,
                iteration_output=iteration_output,
                model=model,
                session_id=self.session_id,
                cadence=cadence,
                context=context,
            )
        self._accumulate_costs(total_cost_data, cost_data)

        # Validate reasoning.md exists; correct if missing (cheap inline check).
        reasoning_path = iteration_output / "reasoning.md"
        if not reasoning_path.exists():
            logger.warning("⚠️  reasoning.md not found, prompting for correction...")
            cost_data = self._prompt_for_correction(
                iteration=iteration,
                model=model,
                error_message=f"reasoning.md is missing. Please create it at iteration_{iteration:03d}/reasoning.md as specified.",
                session_id=self.session_id,
                working_dir=self.output_dir
            )
            self._accumulate_costs(total_cost_data, cost_data)
            if not reasoning_path.exists():
                raise RuntimeError("Planning failed to create reasoning.md even after correction attempt")

        logger.info(f"✓ reasoning.md created ({reasoning_path.stat().st_size} bytes): {os.path.relpath(reasoning_path)}")

        # PHASE 2: Validation and installation of artifacts
        logger.info("✅ Phase 2: Validating and installing outputs...")
        meta_config_schedule, config_delta = self._parse_and_validate_outputs(
            iteration=iteration,
            model=model,
            total_cost_data=total_cost_data,
            session_id=self.session_id,
            working_dir=self.output_dir
        )

        # Validation succeeded — the session now contains a complete planning
        # round (strategy + task background + validated artifacts). Future firings
        # are safe to use the brief follow-up prompt against this session.
        self.initial_firing_complete = True

        # PHASE 3: Reflection
        logger.info("💭 Phase 3: Requesting meta-evolution reflection...")
        cost_data = self._request_reflection(
            iteration=iteration,
            model=model,
            session_id=self.session_id,
            working_dir=self.output_dir,
            iteration_dir=iteration_output,
        )
        self._accumulate_costs(total_cost_data, cost_data)

        # Save per-firing session transcript summary. Lookup uses self.output_dir
        # (matches the cwd Claude CLI saw); output is per-firing so each iteration
        # keeps its own snapshot.
        self._save_session_transcript(
            self.session_id,
            lookup_dir=self.output_dir,
            output_path=iteration_output / "session_summary.md",
        )

        logger.info(f"\n{'=' * 60}")
        logger.info(f"✓ Meta-evolution complete for iteration {iteration}")
        logger.info(f"Cost: ${total_cost_data['total_cost']:.4f}")
        logger.info(f"{'=' * 60}\n")

        return MetaEvolutionResult(meta_config_schedule, config_delta, total_cost_data)

    def check_budget_and_maybe_terminate(self, iteration: int) -> bool:
        """
        Check if budget is exhausted after iteration completes.

        Calculates total cost across all phases:
        - Evaluation costs (from iteration_claude_costs)
        - Evolution costs (from iteration_claude_costs)
        - Evolution: Claude CLI costs
        - Meta-evolution: Claude CLI costs

        Args:
            iteration: Just-completed iteration number

        Returns:
            True if budget exhausted and should terminate, False otherwise

        Side effects:
            - If budget exhausted: creates final_report.md and triggers reflection
        """
        config = self.config_manager.get_config(iteration)
        budget = config.get("dollar_budget")

        if budget is None:
            return False  # No budget limit

        # Calculate total cost across all phases
        total_cost = self._calculate_total_cost(iteration)
        remaining = budget - total_cost

        if remaining <= 0:
            logger.info(
                f"💰 Budget exhausted: ${total_cost:.2f} / ${budget:.2f} "
                f"(after {iteration} iterations)"
            )

            # Note: Final report is generated by researcher.py before termination
            return True  # Terminate

        logger.info(
            f"💰 Budget status: ${total_cost:.2f} / ${budget:.2f} "
            f"(${remaining:.2f} remaining)"
        )

        return False  # Continue

    def _calculate_total_cost(self, through_iteration: int) -> float:
        """
        Calculate total cost through specified iteration.

        Returns sum of:
        - Evaluation costs (from iteration_claude_costs)
        - Evolution costs (from iteration_claude_costs)
        - Meta-evolution costs (from iteration_claude_costs)

        This matches the calculation in report_generator.py.

        Args:
            through_iteration: Calculate costs through this iteration

        Returns:
            Total cost in dollars
        """
        checkpoint_path = self.experiment_dir / "checkpoint.json"
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)

        eval_cost = 0.0
        evolution_cost = 0.0
        meta_evolution_cost = 0.0

        iteration_costs = checkpoint.get("iteration_claude_costs", [])
        for iter_num in range(1, through_iteration + 1):
            if iter_num - 1 < len(iteration_costs):
                cost_dict = iteration_costs[iter_num - 1]
                eval_cost += cost_dict.get('eval_cost', 0.0)
                evolution_cost += cost_dict.get('evolution_cost', 0.0)
                meta_evolution_cost += cost_dict.get('meta_evolution_cost', 0.0)

        return eval_cost + evolution_cost + meta_evolution_cost

    def _calculate_evals_consumed(self, through_iteration: int) -> int:
        """
        Calculate total fresh evaluator calls consumed through specified iteration.

        Reads `iteration_fresh_evals` from the checkpoint (a list whose i-th entry
        is the fresh-eval count for iteration i+1). Mirrors the budget calculation
        in researcher.py's evaluation-budget-exhaustion check.

        Args:
            through_iteration: Sum fresh evals for iterations 1..through_iteration.

        Returns:
            Total fresh evaluations consumed.
        """
        checkpoint_path = self.experiment_dir / "checkpoint.json"
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)

        fresh_evals = checkpoint.get("iteration_fresh_evals", [])
        return sum(fresh_evals[:through_iteration])

    def _load_meta_strategy(self, strategy_name: str) -> str:
        """
        Load meta-evolution strategy from source directory.

        Meta-evolution strategies are NOT copied to experiment directory -
        they're loaded directly from RoboPhD/meta_evolution_strategies/
        (unlike evolution strategies which ARE copied).

        Model is controlled by meta_evolution_model config parameter,
        NOT by the strategy file.

        Args:
            strategy_name: Name of meta-evolution strategy

        Returns:
            Strategy instructions (str)

        Raises:
            ValueError: If strategy file not found
        """
        strategy_path = self.strategies_dir / f"{strategy_name}.md"

        if not strategy_path.exists():
            raise ValueError(
                f"Meta-evolution strategy '{strategy_name}' not found at {strategy_path}"
            )

        content = strategy_path.read_text()

        # Strip YAML frontmatter if present (just name/description metadata)
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                content = parts[2]  # Instructions without frontmatter

        return content

    def _validate_strategy_package(self, strategy_dir: Path) -> List[str]:
        """
        Validate evolution strategy package structure.

        Checks:
        - Directory exists
        - strategy.md exists and has valid YAML frontmatter
        - tools/ directory exists (if referenced)
        - Python tools have valid syntax

        Args:
            strategy_dir: Path to strategy package directory

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        if not strategy_dir.exists():
            errors.append(f"Strategy directory does not exist: {strategy_dir}")
            return errors

        # Check strategy.md
        strategy_file = strategy_dir / "strategy.md"
        if not strategy_file.exists():
            errors.append("Missing strategy.md file")
        else:
            # TODO: Validate YAML frontmatter if needed
            pass

        # Check tools/ directory if it exists
        tools_dir = strategy_dir / "tools"
        if tools_dir.exists():
            # Validate Python files have valid syntax
            for py_file in tools_dir.glob("*.py"):
                try:
                    compile(py_file.read_text(), str(py_file), 'exec')
                except SyntaxError as e:
                    errors.append(f"Syntax error in {py_file.name}: {e}")

        return errors

    def _installed_strategy_names(self) -> set:
        """Return the set of strategy names currently in ``evolution_strategies/``."""
        strategies_dir = self.experiment_dir / "evolution_strategies"
        if not strategies_dir.exists():
            return set()
        return {d.name for d in strategies_dir.iterdir() if d.is_dir()}

    def _find_unresolved_strategy_refs(self, meta_config_schedule: Dict) -> List:
        """
        Walk a meta_config_schedule for evolution_strategy references that don't
        resolve to an installed strategy.

        Checks both top-level ``evolution_strategy`` values and entries inside
        ``weighted_random_configs``. The literal string ``"none"`` is allowed
        (means "skip evolution this iteration").

        Args:
            meta_config_schedule: Parsed schedule dict from
                meta_config_schedule.json.

        Returns:
            List of (iter_str, strategy_name, location) tuples for each
            unresolved reference. Empty list = all references resolve.
        """
        installed = self._installed_strategy_names()
        unresolved = []
        for iter_str, delta in meta_config_schedule.items():
            if not isinstance(delta, dict):
                continue
            if "evolution_strategy" in delta:
                name = delta["evolution_strategy"]
                if name and name != "none" and name not in installed:
                    unresolved.append((iter_str, name, "evolution_strategy"))
            if "weighted_random_configs" in delta:
                for entry_idx, entry in enumerate(delta.get("weighted_random_configs") or []):
                    if isinstance(entry, list) and len(entry) >= 1:
                        config_dict = entry[0]
                        if isinstance(config_dict, dict) and "evolution_strategy" in config_dict:
                            name = config_dict["evolution_strategy"]
                            if name and name != "none" and name not in installed:
                                unresolved.append(
                                    (iter_str, name, f"weighted_random_configs[{entry_idx}]")
                                )
        return unresolved

    def _format_unresolved_strategy_error(
        self, unresolved: List, iteration: int, installed_this_firing: set
    ) -> str:
        """Build a data-rich correction prompt listing what was wrong + what's available."""
        lines = ["Your meta_config_schedule.json references strategies that don't exist:"]
        for iter_str, name, loc in unresolved:
            lines.append(f"- Iteration {iter_str} ({loc}): \"{name}\" — not found")

        installed = sorted(self._installed_strategy_names())
        if installed:
            lines.append("")
            lines.append("Strategies currently installed:")
            for name in installed:
                lines.append(f"- {name}")

        if installed_this_firing:
            lines.append("")
            lines.append("Strategies just created in this firing (now installed with iter prefix):")
            prefix = f"iter{iteration}_"
            for prefixed in sorted(installed_this_firing):
                unprefixed = prefixed[len(prefix):] if prefixed.startswith(prefix) else prefixed
                lines.append(f"- {unprefixed} → {prefixed}")

        lines.append("")
        lines.append(
            "Reminder: new strategies are installed with the iter{N}_ prefix. "
            "Reference one of the names above and update meta_config_schedule.json."
        )
        return "\n".join(lines)

    def _install_strategy_package(self, strategy_name: str, iteration: int) -> str:
        """
        Install validated strategy package to evolution_strategies/ directory.

        Mirrors the agent-naming convention in _generate_agent_id: the installed
        directory name is prefixed with ``iter{iteration}_`` so each firing's
        strategy lands at a unique path. This eliminates the silent-overwrite
        cycle that occurred when meta-evolution wrote successive revisions of
        a same-named strategy across firings.

        The meta-agent should reference the prefixed name in
        ``meta_config_schedule.json`` (see META_EVOLUTION_ENVIRONMENT_GUIDE).
        Validation in _parse_and_validate_outputs catches references that
        forget the prefix and re-prompts for correction.

        Args:
            strategy_name: Unprefixed name as written by meta-evolution
                (the directory name under iteration_NNN/new_strategies/).
            iteration: Iteration that created this strategy. Used to mint the
                unique installed name.

        Returns:
            The installed (prefixed) name, e.g. "iter4_cost_mechanism_aware".
        """
        source_dir = (
            self.output_dir /
            f"iteration_{iteration:03d}" /
            "new_strategies" /
            strategy_name
        )

        installed_name = f"iter{iteration}_{strategy_name}"
        dest_dir = self.experiment_dir / "evolution_strategies" / installed_name

        logger.info(
            f"Installing strategy '{strategy_name}' to evolution_strategies/{installed_name}/"
        )

        # Create evolution_strategies directory if it doesn't exist
        dest_dir.parent.mkdir(parents=True, exist_ok=True)

        # Recovery path: if a directory with this prefixed name already exists,
        # the most likely cause is a mid-firing crash between install and
        # initial_firing_complete=True — on resume, the same iteration's firing
        # runs again and re-installs at the same prefixed path. Log loudly (the
        # state is unusual) but proceed, since the new install supersedes the
        # crashed partial.
        if dest_dir.exists():
            logger.warning(
                f"Strategy directory {dest_dir} already exists; replacing. "
                f"This typically indicates iteration {iteration}'s meta-evolution "
                f"is running again after a mid-firing crash + resume. The prior "
                f"partial install will be overwritten."
            )
            import shutil
            shutil.rmtree(dest_dir)

        import shutil
        shutil.copytree(source_dir, dest_dir)

        logger.info(f"✓ Installed strategy '{installed_name}'")
        return installed_name

    def _update_meta_evolution_costs(self, iteration: int, cost_data: Dict) -> None:
        """
        Update checkpoint with meta-evolution costs for this iteration.

        Costs are tracked in checkpoint.iteration_claude_costs[iteration]['meta_evolution_cost']
        parallel to eval_cost and evolution_cost.

        Args:
            iteration: Current iteration number
            cost_data: Dictionary containing cost information from Claude CLI
        """
        checkpoint_path = self.experiment_dir / "checkpoint.json"
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)

        # Get or create cost entry for this iteration
        iteration_costs = checkpoint.get("iteration_claude_costs", [])
        while len(iteration_costs) < iteration:
            iteration_costs.append({})

        # Extract cost from cost_data
        meta_cost = cost_data.get('total_cost', 0.0)

        # Update meta_evolution_cost field
        if len(iteration_costs) >= iteration:
            iteration_costs[iteration - 1]['meta_evolution_cost'] = meta_cost
            iteration_costs[iteration - 1]['meta_evolution_calls'] = cost_data.get('calls', 0)
            iteration_costs[iteration - 1]['meta_evolution_tokens_in'] = cost_data.get('tokens_in', 0)
            iteration_costs[iteration - 1]['meta_evolution_tokens_out'] = cost_data.get('tokens_out', 0)

        checkpoint['iteration_claude_costs'] = iteration_costs

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def _get_claude_cli_path(self) -> Path:
        """
        Get path to Claude CLI executable.

        Returns:
            Path to claude CLI

        Raises:
            FileNotFoundError: If claude CLI not found
        """
        import subprocess
        import sys

        # Check local installation first
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
            return Path(result.stdout.strip())
        except subprocess.CalledProcessError:
            raise FileNotFoundError(
                "Claude CLI not found. Install from: https://docs.anthropic.com/en/docs/claude-code"
            )

    def _call_claude_code(
        self,
        prompt: str,
        model: str,
        session_id: str,
        working_dir: Path,
        resume_session: bool = False
    ) -> Dict[str, Any]:
        """
        Call Claude Code CLI with the given prompt.

        Args:
            prompt: Prompt to send to Claude Code
            model: Model to use (API name like "sonnet-4.5")
            session_id: Session ID for this meta-evolution call
            working_dir: Working directory (cwd) for Claude Code. Must stay STABLE
                across all firings within a run — Claude CLI hashes cwd to locate
                the on-disk session transcript at ~/.claude/projects/<sanitized>/.
                Meta-evolution uses self.output_dir (meta_evolution_output/) for
                this reason; iteration-specific subdirs live as `iteration_NNN/`.
            resume_session: True → use ``--resume <session_id>`` (continue an existing
                Claude Code session). False → use ``--session-id <session_id>`` to
                create a new session with that explicit id.

        Returns:
            Dictionary with cost and usage information

        Raises:
            RuntimeError: If Claude Code call fails
        """
        from RoboPhD.config import CLAUDE_CLI_MODEL_MAP, get_lmstudio_env

        # Build command
        claude_cli = self._get_claude_cli_path()
        # Map API model name to Claude CLI name (e.g., 'sonnet-4.5' -> 'sonnet')
        cli_model = CLAUDE_CLI_MODEL_MAP.get(model, model)

        cmd = [
            str(claude_cli),
            "--model", cli_model
        ]

        # Use explicit session management to prevent interference
        if resume_session:
            cmd.extend(["--resume", session_id])
        else:
            cmd.extend(["--session-id", session_id])

        cmd.extend([
            "--print", prompt,
            "--output-format", "json",  # Get JSON output for cost tracking
            "--permission-mode", "bypassPermissions",  # Allow automation without prompts
            "--settings", '{"autoCompact": true}'  # Proactively compact when context gets low
        ])

        logger.debug(f"Calling Claude Code: {' '.join(cmd[:4])}...")

        # Get LM Studio env overrides for non-Anthropic models
        extra_env = get_lmstudio_env(model)

        try:
            # Run in iteration-specific working directory with rate limit handling
            result = call_claude_cli(
                cmd=cmd,
                cwd=working_dir,
                timeout=1800,  # 30 minutes default
                logger=logger,
                extra_env=extra_env
            )

            if result.returncode != 0:
                logger.error(f"Claude Code call failed with return code {result.returncode}")
                logger.error(f"stdout: {result.stdout[:1000]}")
                logger.error(f"stderr: {result.stderr}")
                raise RuntimeError(f"Claude Code call failed: {result.stderr}")

            # Parse JSON output for cost tracking
            if result.stdout:
                try:
                    json_output = json.loads(result.stdout)
                    usage = json_output.get('usage', {})
                    return {
                        'total_cost': json_output.get('total_cost_usd', 0.0),
                        'calls': 1,
                        'tokens_in': usage.get('input_tokens', 0),
                        'tokens_out': usage.get('output_tokens', 0),
                        'cache_created': usage.get('cache_creation_input_tokens', 0),
                        'cache_read': usage.get('cache_read_input_tokens', 0)
                    }
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to parse cost data: {e}")
                    return {'total_cost': 0.0, 'calls': 1}

            return {'total_cost': 0.0, 'calls': 1}

        except subprocess.TimeoutExpired:
            raise RuntimeError("Claude Code call timed out after 1800s")
        except RateLimitExceeded:
            # Let rate limit exceeded propagate for checkpoint/exit handling
            raise
        except Exception as e:
            raise RuntimeError(f"Claude Code call failed: {e}")

    def _accumulate_costs(self, total: Dict, new: Dict) -> None:
        """Accumulate cost data from multiple rounds."""
        total['total_cost'] += new.get('total_cost', 0.0)
        total['calls'] += new.get('calls', 0)
        total['tokens_in'] += new.get('tokens_in', 0)
        total['tokens_out'] += new.get('tokens_out', 0)
        total['cache_created'] += new.get('cache_created', 0)
        total['cache_read'] += new.get('cache_read', 0)

    def _prompt_for_correction(
        self,
        iteration: int,
        model: str,
        error_message: str,
        session_id: str,
        working_dir: Path
    ) -> Dict[str, Any]:
        """
        Prompt meta-evolution to correct validation errors.

        Resumes the session and asks Claude Code to fix specific issues.

        Args:
            iteration: Current iteration number
            model: Model to use for correction
            error_message: Specific error(s) to fix
            session_id: Session ID for this meta-evolution call
            working_dir: Working directory for Claude Code

        Returns:
            Cost data dictionary
        """
        correction_prompt = f"""
## Validation Error - Please Correct

Your implementation has validation errors:

{error_message}

Please fix these issues in `iteration_{iteration:03d}/`. See `CLAUDE.md` for the strategy-package structure and schedule rules if needed.
"""

        # Resume session and prompt for correction
        return self._call_claude_code(
            prompt=correction_prompt,
            model=model,
            session_id=session_id,
            working_dir=working_dir,
            resume_session=True  # Corrections always continue the active session
        )

    def _execute_planning_and_implementation(
        self,
        strategy: str,
        context: Dict,
        iteration: int,
        iteration_output: Path,
        model: str,
        session_id: str,
        cadence: int,
        first_iteration: int,
    ) -> Dict[str, Any]:
        """
        Execute initial-firing planning and implementation.

        Sends full context (strategy + cadence info + task background + reports),
        asks Claude to create reasoning.md and implement strategies/config changes.
        This is the FIRST firing only; subsequent firings use _execute_followup_firing.

        Args:
            strategy: Meta-evolution strategy text
            context: Gathered context (rankings, reports, budget)
            iteration: Current iteration number
            iteration_output: Output directory for this iteration
            model: Model to use
            session_id: Session ID (will be created with --session-id)
            cadence: Iterations between firings (used in cadence paragraph)
            first_iteration: First firing iteration (used in cadence paragraph)

        Returns:
            Cost data dictionary
        """
        budget_info = self._format_budget_status(context["budget"], iteration)
        strategy_with_budget = strategy.replace(
            "**Budget Status**:",
            budget_info
        )

        # Write CLAUDE.md with domain background to parent (meta_evolution_output/).
        # Claude Code traverses up to find it, so all iteration subdirs inherit it.
        # The GUIDE template uses .format() placeholders for cadence values; all
        # other curly braces in the template are doubled ({{ }}) so they survive
        # formatting unchanged.
        #
        # Write-once-per-run: the `if not exists` guard means a resumed run with
        # initial_firing_complete=False (which mints a fresh session) inherits the
        # prior run's CLAUDE.md. This is intentional and safe — cadence and
        # first_iteration are IMMUTABLE_PARAMS, so the inherited file is still
        # correct. If you ever add a re-write path here, first verify that
        # nothing relies on CLAUDE.md staying stable across same-run firings.
        claude_md_path = self.output_dir / "CLAUDE.md"
        if not claude_md_path.exists():
            # Detect available CLI tools (mirrors deep_focus_evolution_manager:209).
            # Don't claim tools are available if they aren't — meta-agent will try
            # to use them and fail.
            import shutil
            available_tools = []
            missing_tools = []
            for tool in ["jq", "tree"]:
                (available_tools if shutil.which(tool) else missing_tools).append(tool)
            if missing_tools:
                logger.warning(
                    "Recommended CLI tools not found: %s. Install for better meta-evolution results.",
                    ", ".join(missing_tools),
                )
            cli_tools_text = (
                "Available: " + ", ".join(f"`{t}`" for t in available_tools)
                if available_tools
                else "(none of the recommended tools — `jq`, `tree` — detected on this system)"
            )

            sections = []
            if self._task_background:
                sections.append(f"# Domain Background\n\n{self._task_background}")
            if self._task_objective:
                sections.append(f"# Domain Objective\n\n{self._task_objective}")
            sections.append(
                META_EVOLUTION_ENVIRONMENT_GUIDE.format(
                    cadence=cadence,
                    first_iteration=first_iteration,
                    first_plus_cadence=first_iteration + cadence,
                    first_plus_2cadence=first_iteration + 2 * cadence,
                    cli_tools=cli_tools_text,
                )
            )
            claude_md_path.write_text("\n\n".join(sections))
            logger.info(f"CLAUDE.md written to: {claude_md_path}")

        prompt = f"""
{strategy_with_budget}

## Current State (Iteration {iteration})

### Recent Performance
{self._format_interim_reports(context.get("interim_reports", []))}

## Your Task

Produce the artifacts for this firing in `iteration_{iteration:03d}/`. Per your strategy:
- `reasoning.md` (REQUIRED) — your analysis, formatted per your strategy's instructions
- `meta_config_schedule.json` (REQUIRED) — can be `{{}}` if no changes
- `new_strategies/<name>/strategy.md` and/or `config_delta.json` — only if your strategy authorizes them

See `CLAUDE.md` (already in your context) for: cadence, strategy-package structure, naming convention, schedule format, schedule semantics, forbidden parameters, weighted-random override, and the framework's post-firing actions.
"""

        # Save meta-evolution prompt for debugging and reproducibility
        meta_prompt_file = iteration_output / "meta_evolution_prompt.md"
        meta_prompt_file.write_text(prompt)
        logger.info(f"Meta-evolution prompt saved to: {meta_prompt_file}")

        # Single call for planning and implementation. cwd is the parent
        # `meta_evolution_output/` so the persistent Claude Code session has a
        # stable working dir across all firings within this run (Claude CLI
        # stores transcripts under a hash of cwd; varying cwd would make
        # `--resume` fail with "no conversation found").
        return self._call_claude_code(
            prompt=prompt,
            model=model,
            session_id=session_id,
            working_dir=self.output_dir,
            resume_session=False  # Initial firing creates the session
        )

    def _execute_followup_firing(
        self,
        iteration: int,
        iteration_output: Path,
        model: str,
        session_id: str,
        cadence: int,
        context: Dict,
    ) -> Dict[str, Any]:
        """
        Execute a follow-up meta-evolution firing within the existing session.

        Sends a brief status update referencing the latest iteration's reports
        and asks for the standard artifacts. The strategy text and task background
        are NOT re-fed — Claude has them from the initial firing.

        Args:
            iteration: Current iteration number
            iteration_output: Output directory for this iteration
            model: Model to use
            session_id: Persistent session id (will be resumed with --resume)
            cadence: Iterations between firings (used to mention next firing)
            context: Gathered context (used for budget formatting)

        Returns:
            Cost data dictionary
        """
        budget_info = self._format_budget_status(context["budget"], iteration)

        prompt = f"""## Meta-Evolution Firing — Iteration {iteration}

Iteration {iteration} has just completed. Updated reports for this iteration (paths relative to your meta_evolution_output/ working dir):
- Interim report: `../iteration_{iteration:03d}/interim_report.md`
- Cost report: `../iteration_{iteration:03d}/cost_report.md`
- Error analysis: `../iteration_{iteration:03d}/error_analysis_report.md`

{budget_info}

Next firing: iteration {iteration + cadence} (or run end if budget exhausts first).

Please produce the standard artifacts in `iteration_{iteration:03d}/`:
- `iteration_{iteration:03d}/reasoning.md` — your analysis and plan (reference your prior decisions and what the new data shows)
- `iteration_{iteration:03d}/meta_config_schedule.json` — config changes for upcoming iterations (REQUIRED, can be empty `{{}}` if no changes)
- `iteration_{iteration:03d}/config_delta.json` — immediate parameter change starting next iteration; include only if your strategy authorizes parameter changes.
- `iteration_{iteration:03d}/new_strategies/<name>/strategy.md` — a new evolution strategy; include only if your strategy authorizes creating new evolution strategies.

After completing, respond with: "META-EVOLUTION ITERATION {iteration} COMPLETE"
"""

        # Save follow-up prompt for debugging and reproducibility
        meta_prompt_file = iteration_output / "meta_evolution_prompt.md"
        meta_prompt_file.write_text(prompt)
        logger.info(f"Meta-evolution follow-up prompt saved to: {meta_prompt_file}")

        return self._call_claude_code(
            prompt=prompt,
            model=model,
            session_id=session_id,
            working_dir=self.output_dir,
            resume_session=True  # Follow-up firings resume the persistent session
        )

    def _gather_context(self, iteration: int, config: Dict[str, Any]) -> Dict:
        """
        Gather information for meta-evolution analysis.

        Returns dictionary with interim reports and budget (both dollar and
        evaluation budgets, when set).
        """
        # Dollar budget (optional)
        dollar_budget = config.get("dollar_budget")
        total_cost = self._calculate_total_cost(iteration)
        dollar_remaining = (dollar_budget - total_cost) if dollar_budget else None

        # Evaluation budget (typically set; default 1500 for the example mains)
        eval_budget = config.get("evaluation_budget")
        eval_consumed = self._calculate_evals_consumed(iteration)
        eval_remaining = (eval_budget - eval_consumed) if eval_budget else None

        context = {
            "current_iteration": iteration,
            "interim_reports": [],
            "budget": {
                "dollar_total": dollar_budget,
                "dollar_consumed": total_cost,
                "dollar_remaining": dollar_remaining,
                "eval_total": eval_budget,
                "eval_consumed": eval_consumed,
                "eval_remaining": eval_remaining,
            }
        }

        # Find most recent interim report (they are cumulative)
        # Search backwards from current iteration to find latest report
        latest_report = None
        for iter_num in range(iteration, 1, -1):  # iteration down to 2
            iter_dir = self.experiment_dir / f"iteration_{iter_num:03d}"
            report_path = iter_dir / "interim_report.md"
            if report_path.exists():
                latest_report = {
                    "iteration": iter_num,
                    "path": f"iteration_{iter_num:03d}/interim_report.md"
                }
                break

        context["interim_reports"] = [latest_report] if latest_report else []

        return context

    def _find_first_meta_evolution_iteration(self) -> int:
        """
        Find the first iteration where meta-evolution ran.

        Checks resolved configs to find the first iteration where meta_evolution_strategy
        was active (not None and not "none"). This works regardless of whether the strategy
        was set via direct delta, config_schedule, or meta_config_schedule.

        Returns:
            First iteration number where meta_evolution_strategy was active
        """
        checkpoint = self._load_checkpoint()
        resolved_configs = checkpoint.get('config_manager', {}).get('resolved_configs', {})

        # Check resolved configs in order
        for iter_num in sorted([int(k) for k in resolved_configs.keys()]):
            config = resolved_configs[str(iter_num)]
            strategy = config.get('meta_evolution_strategy')
            if strategy and strategy != 'none':
                return iter_num

        return 2  # Default fallback if not found

    def _find_previous_meta_evolution_iteration(self, current_iteration: int) -> Optional[int]:
        """
        Find the most recent iteration where meta-evolution ran before current iteration.

        Checks resolved configs in reverse order from current_iteration - 1.

        Args:
            current_iteration: Current iteration number

        Returns:
            Previous meta-evolution iteration number, or None if not found
        """
        checkpoint = self._load_checkpoint()
        resolved_configs = checkpoint.get('config_manager', {}).get('resolved_configs', {})

        # Check resolved configs in reverse order before current iteration
        for iter_num in sorted([int(k) for k in resolved_configs.keys()], reverse=True):
            if iter_num >= current_iteration:
                continue  # Skip current and future iterations

            config = resolved_configs[str(iter_num)]
            strategy = config.get('meta_evolution_strategy')
            if strategy and strategy != 'none':
                return iter_num

        return None  # No previous meta-evolution found

    def _load_checkpoint(self) -> Dict:
        """Load checkpoint JSON."""
        checkpoint_path = self.experiment_dir / "checkpoint.json"
        if not checkpoint_path.exists():
            return {}

        with open(checkpoint_path) as f:
            return json.load(f)

    def _format_budget_status(self, budget_info: Dict, iteration: int) -> str:
        """Format budget status (evaluation and dollar budgets) for the prompt."""
        lines = []

        eval_total = budget_info.get("eval_total")
        if eval_total is not None:
            eval_consumed = budget_info["eval_consumed"]
            eval_remaining = budget_info["eval_remaining"]
            lines.append(
                f"- **Evaluations**: {eval_consumed} / {eval_total} consumed "
                f"({eval_remaining} remaining)"
            )

        dollar_total = budget_info.get("dollar_total")
        if dollar_total is not None:
            dollar_consumed = budget_info["dollar_consumed"]
            dollar_remaining = budget_info["dollar_remaining"]
            lines.append(
                f"- **Dollars**: ${dollar_consumed:.2f} / ${dollar_total:.2f} "
                f"(${dollar_remaining:.2f} remaining)"
            )

        if not lines:
            return "**Budget Status**: No budget limits set"

        lines.append(f"- **Iterations completed**: {iteration}")
        return "**Budget Status**:\n" + "\n".join(lines)

    def _format_interim_reports(self, reports: List[Dict]) -> str:
        """Format interim reports reference for prompt."""
        if not reports:
            return "(No interim reports available)"

        # Only reference the most recent report (they are cumulative)
        latest = reports[-1]

        return f"""The most recent interim report is available at:
`{latest['path']}`

This report is cumulative and includes performance data across all iterations."""

    def _parse_and_validate_outputs(
        self,
        iteration: int,
        model: str,
        total_cost_data: Dict,
        session_id: str,
        working_dir: Path
    ) -> tuple[Dict, Optional[Dict]]:
        """
        Parse and validate meta-evolution outputs (Phase 4).

        Discovers strategies by scanning new_strategies/ directory.
        Validates both strategies and config.
        Installs valid strategies.

        Args:
            iteration: Current iteration number
            model: Model name for correction prompts
            total_cost_data: Cost accumulator (updated if corrections needed)
            session_id: Session ID for this meta-evolution call
            working_dir: Working directory for Claude Code

        Returns:
            Tuple of (meta_config_schedule, config_delta):
            - meta_config_schedule: Dict mapping iteration numbers to config deltas
            - config_delta: Optional flat dict of immediate parameter changes
        """
        iteration_dir = self.output_dir / f"iteration_{iteration:03d}"
        new_strategies_dir = iteration_dir / "new_strategies"

        # Discover strategies by scanning new_strategies/
        strategy_names = []
        if new_strategies_dir.exists():
            strategy_names = [
                d.name for d in new_strategies_dir.iterdir()
                if d.is_dir() and not d.name.startswith('.')
            ]

        logger.info(f"Discovered {len(strategy_names)} new strategies: {strategy_names}")

        # Parse config_delta.json (optional — used by parameter_adjustment)
        config_delta = None
        config_delta_file = iteration_dir / "config_delta.json"
        if config_delta_file.exists():
            with open(config_delta_file) as f:
                config_delta = json.load(f)

            if not isinstance(config_delta, dict):
                raise RuntimeError(f"config_delta.json must be a flat dict, got {type(config_delta).__name__}")

            if config_delta:
                # Validate no forbidden parameters
                forbidden = self.config_manager._get_meta_evolution_forbidden_params()
                forbidden_found = [p for p in config_delta if p in forbidden]
                if forbidden_found:
                    raise RuntimeError(
                        f"config_delta.json contains forbidden parameters: {forbidden_found}. "
                        f"Forbidden: {sorted(forbidden)}"
                    )

                # Validate all parameters are known
                try:
                    self.config_manager._validate_parameters(
                        config_delta,
                        "config_delta.json",
                        ConfigSource.META_EVOLUTION
                    )
                except ValueError as e:
                    raise RuntimeError(f"config_delta.json has invalid parameters: {e}")

                logger.info(f"✓ Loaded config_delta.json: {config_delta}")
            else:
                logger.info("✓ config_delta.json is empty (no changes)")
                config_delta = None  # Normalize empty dict to None

        # Load meta_config_schedule.json (required unless config_delta.json exists)
        config_file = iteration_dir / "meta_config_schedule.json"
        has_config_delta = config_delta is not None or config_delta_file.exists()

        if not config_file.exists() and not has_config_delta:
            # Neither file exists — prompt for correction
            logger.warning("⚠️  Neither meta_config_schedule.json nor config_delta.json found, prompting for correction...")
            cost_data = self._prompt_for_correction(
                iteration=iteration,
                model=model,
                error_message="Neither meta_config_schedule.json nor config_delta.json found (at least one is required). "
                             "Please create meta_config_schedule.json or config_delta.json.",
                session_id=session_id,
                working_dir=working_dir
            )
            self._accumulate_costs(total_cost_data, cost_data)

            if not config_file.exists() and not config_delta_file.exists():
                raise RuntimeError(
                    "Neither meta_config_schedule.json nor config_delta.json found after correction attempt"
                )

            # Re-check: correction may have created config_delta.json
            if config_delta_file.exists() and config_delta is None:
                with open(config_delta_file) as f:
                    config_delta = json.load(f) or None

        if config_file.exists():
            with open(config_file) as f:
                meta_config_schedule = json.load(f)
        else:
            logger.info("meta_config_schedule.json not found (config_delta.json present, skipping)")
            meta_config_schedule = {}

        # Validate structure: all keys must be numeric iteration strings > current iteration
        doc_keys = [k for k in meta_config_schedule.keys() if k.startswith('_')]
        non_numeric_keys = [k for k in meta_config_schedule.keys() if not k.isdigit()]

        if doc_keys or non_numeric_keys:
            # Invalid keys found - prompt for correction
            invalid = doc_keys + non_numeric_keys
            logger.warning(f"⚠️  Invalid keys in meta_config_schedule: {invalid}")
            logger.warning("Prompting Claude to generate clean JSON...")

            cost_data = self._prompt_for_correction(
                iteration=iteration,
                model=model,
                error_message=f"meta_config_schedule.json contains invalid keys: {invalid}. "
                             f"Requirements:\n"
                             f"1. ALL top-level keys must be numeric iteration strings (e.g., '3', '4', '5')\n"
                             f"2. NO documentation fields (keys starting with '_')\n"
                             f"3. NO other non-numeric keys\n"
                             f"Please regenerate with ONLY numeric iteration keys > {iteration}.",
                session_id=session_id,
                working_dir=working_dir
            )
            self._accumulate_costs(total_cost_data, cost_data)

            # Reload and re-validate
            with open(config_file) as f:
                meta_config_schedule = json.load(f)

            # Validate ALL keys are numeric iteration strings
            non_numeric_keys = [k for k in meta_config_schedule.keys() if not k.isdigit()]
            if non_numeric_keys:
                raise RuntimeError(
                    f"meta_config_schedule.json still has non-numeric keys after correction: {non_numeric_keys}. "
                    f"All top-level keys must be iteration numbers (e.g., '3', '4', '5')."
                )

            # Validate all iteration numbers are in the future (> current iteration)
            invalid_iterations = [k for k in meta_config_schedule.keys() if int(k) <= iteration]
            if invalid_iterations:
                raise RuntimeError(
                    f"meta_config_schedule.json has invalid iteration numbers after correction: {invalid_iterations}. "
                    f"All iterations must be > {iteration} (current iteration)."
                )

        logger.info(f"Loaded meta_config_schedule with changes for {len(meta_config_schedule)} iterations")

        # Validate no forbidden parameters in meta_config_schedule
        forbidden = self.config_manager._get_meta_evolution_forbidden_params()
        forbidden_params_found = []

        for iter_str, delta in meta_config_schedule.items():
            for param in delta.keys():
                if param in forbidden:
                    forbidden_params_found.append((iter_str, param))

        if forbidden_params_found:
            # Build error message
            error_lines = ["meta_config_schedule.json contains forbidden parameters:"]
            for iter_str, param in forbidden_params_found:
                error_lines.append(f"  - Iteration {iter_str}: '{param}'")
            error_lines.append(f"\nForbidden parameters: {sorted(forbidden)}")
            error_lines.append("\nThese parameters control meta-evolution itself or system state, creating circular dependencies.")
            error_lines.append("Please remove these parameters from the schedule.")

            error_msg = "\n".join(error_lines)
            logger.warning(f"⚠️  {error_msg}")
            logger.warning("Prompting Claude to fix forbidden parameters...")

            cost_data = self._prompt_for_correction(
                iteration=iteration,
                model=model,
                error_message=error_msg,
                session_id=session_id,
                working_dir=working_dir
            )
            self._accumulate_costs(total_cost_data, cost_data)

            # Reload and re-validate
            with open(config_file) as f:
                meta_config_schedule = json.load(f)

            # Check again for forbidden parameters
            forbidden_params_found = []
            for iter_str, delta in meta_config_schedule.items():
                for param in delta.keys():
                    if param in forbidden:
                        forbidden_params_found.append((iter_str, param))

            if forbidden_params_found:
                raise RuntimeError(
                    f"meta_config_schedule.json still has forbidden parameters after correction: {forbidden_params_found}. "
                    f"Forbidden parameters that cannot be modified by meta-evolution: {sorted(forbidden)}"
                )

        logger.info("✓ No forbidden parameters in meta_config_schedule")

        # Validate all parameters are known (no typos or invalid parameter names)
        unknown_params_errors = []
        for iter_str, delta in meta_config_schedule.items():
            try:
                self.config_manager._validate_parameters(
                    delta,
                    f"Meta-evolution schedule for iteration {iter_str}",
                    ConfigSource.META_EVOLUTION
                )
            except ValueError as e:
                unknown_params_errors.append((iter_str, str(e)))

        if unknown_params_errors:
            # Build error message
            error_lines = ["meta_config_schedule.json contains unknown/invalid parameters:"]
            for iter_str, error_msg in unknown_params_errors:
                # Extract just the parameter names from the error message
                error_lines.append(f"  - Iteration {iter_str}: {error_msg}")
            error_lines.append("\nPlease fix these parameter names or remove invalid parameters.")

            error_msg = "\n".join(error_lines)
            logger.warning(f"⚠️  {error_msg}")
            logger.warning("Prompting Claude to fix unknown parameters...")

            cost_data = self._prompt_for_correction(
                iteration=iteration,
                model=model,
                error_message=error_msg,
                session_id=session_id,
                working_dir=working_dir
            )
            self._accumulate_costs(total_cost_data, cost_data)

            # Reload and re-validate
            with open(config_file) as f:
                meta_config_schedule = json.load(f)

            # Check again for unknown parameters
            unknown_params_errors = []
            for iter_str, delta in meta_config_schedule.items():
                try:
                    self.config_manager._validate_parameters(
                        delta,
                        f"Meta-evolution schedule for iteration {iter_str}",
                        ConfigSource.META_EVOLUTION
                    )
                except ValueError as e:
                    unknown_params_errors.append((iter_str, str(e)))

            if unknown_params_errors:
                error_details = "; ".join([f"iter {i}: {e}" for i, e in unknown_params_errors])
                raise RuntimeError(
                    f"meta_config_schedule.json still has unknown parameters after correction: {error_details}"
                )

        logger.info("✓ All parameters in meta_config_schedule are valid")

        # Phase E: Validate weighted_random_configs conflicts
        weighted_random_errors = []

        # Helper function to extract parameters affected by weighted_random_configs
        def extract_weighted_random_params(configs_list):
            """Extract all unique parameter names from weighted_random_configs pool."""
            params = set()
            if not isinstance(configs_list, list):
                return params
            for entry in configs_list:
                if isinstance(entry, (list, tuple)) and len(entry) >= 1:
                    config_dict = entry[0]
                    if isinstance(config_dict, dict):
                        params.update(config_dict.keys())
            return params

        # Check E1: weighted_random_configs format validation
        for iter_str, delta in meta_config_schedule.items():
            if "weighted_random_configs" in delta:
                pool = delta["weighted_random_configs"]
                iter_num = int(iter_str)

                # Check it's a list
                if not isinstance(pool, list):
                    weighted_random_errors.append((
                        iter_str,
                        f"weighted_random_configs must be a list, got {type(pool).__name__}"
                    ))
                    continue

                # Check each entry is [dict, int]
                for i, entry in enumerate(pool):
                    if not isinstance(entry, (list, tuple)):
                        weighted_random_errors.append((
                            iter_str,
                            f"weighted_random_configs[{i}] must be a list or tuple, got {type(entry).__name__}"
                        ))
                        continue

                    if len(entry) != 2:
                        weighted_random_errors.append((
                            iter_str,
                            f"weighted_random_configs[{i}] must have exactly 2 elements [config_dict, weight], got {len(entry)}"
                        ))
                        continue

                    config_dict, weight = entry

                    if not isinstance(config_dict, dict):
                        weighted_random_errors.append((
                            iter_str,
                            f"weighted_random_configs[{i}][0] must be a dict, got {type(config_dict).__name__}"
                        ))

                    if not isinstance(weight, int):
                        weighted_random_errors.append((
                            iter_str,
                            f"weighted_random_configs[{i}][1] must be an int, got {type(weight).__name__}"
                        ))

                # Check weights sum to 100
                try:
                    total = sum(weight for _, weight in pool if isinstance(_, dict))
                    if total != 100:
                        weighted_random_errors.append((
                            iter_str,
                            f"weighted_random_configs weights must sum to 100%, got {total}%"
                        ))
                except (TypeError, ValueError) as e:
                    weighted_random_errors.append((
                        iter_str,
                        f"weighted_random_configs has invalid weight values: {e}"
                    ))

        # Check E2: No conflicting parameters in same iteration
        for iter_str, delta in meta_config_schedule.items():
            if delta.get("use_weighted_random", False):
                if "weighted_random_configs" not in delta:
                    # Will inherit from previous iteration - check that separately in E3
                    continue

                # Extract parameters that will be set by weighted random
                weighted_params = extract_weighted_random_params(delta["weighted_random_configs"])

                # Check if any of those parameters are also set directly in this iteration
                conflicting_params = []
                for param in delta.keys():
                    if param in weighted_params and param not in ["use_weighted_random", "weighted_random_configs"]:
                        conflicting_params.append(param)

                if conflicting_params:
                    weighted_random_errors.append((
                        iter_str,
                        f"Iteration has use_weighted_random: true but also sets parameters that "
                        f"weighted_random_configs will override: {sorted(conflicting_params)}. "
                        f"Remove these parameters or set use_weighted_random: false."
                    ))

        # Check E3: Must explicitly disable weighted random when overriding its parameters
        for iter_str in sorted(meta_config_schedule.keys(), key=int):
            iter_num = int(iter_str)
            delta = meta_config_schedule[iter_str]

            # Skip if this iteration explicitly enables weighted random (E2 handles that case)
            if delta.get("use_weighted_random", False):
                continue

            # Get resolved config from iteration N-1 to see if it has weighted random enabled
            # We need to check what N-1's resolved config would be
            if iter_num > iteration + 1:
                # Get previous iteration's delta from meta_config_schedule
                prev_iter_str = str(iter_num - 1)
                if prev_iter_str in meta_config_schedule:
                    prev_delta = meta_config_schedule[prev_iter_str]

                    # Check if previous iteration enables weighted random
                    prev_has_weighted_random = prev_delta.get("use_weighted_random", False)
                    if prev_has_weighted_random and "weighted_random_configs" in prev_delta:
                        # Extract parameters that weighted random affects
                        weighted_params = extract_weighted_random_params(prev_delta["weighted_random_configs"])

                        # Check if current iteration tries to set any of those parameters
                        conflicting_params = []
                        for param in delta.keys():
                            if param in weighted_params and param not in ["use_weighted_random", "weighted_random_configs", "comment"]:
                                conflicting_params.append(param)

                        # If there are conflicts and current iteration doesn't explicitly disable weighted random
                        if conflicting_params and "use_weighted_random" not in delta:
                            weighted_random_errors.append((
                                iter_str,
                                f"Iteration {prev_iter_str} has use_weighted_random: true affecting parameters: {sorted(weighted_params)}. "
                                f"Iteration {iter_str} tries to set {sorted(conflicting_params)} without explicitly setting "
                                f"use_weighted_random: false. Add 'use_weighted_random': false to iteration {iter_str}."
                            ))

        if weighted_random_errors:
            # Build error message
            error_lines = ["meta_config_schedule.json has weighted_random_configs conflicts:"]
            for iter_str, error_msg in weighted_random_errors:
                error_lines.append(f"\n  Iteration {iter_str}:")
                error_lines.append(f"    {error_msg}")
            error_lines.append("\nPlease fix these conflicts:")
            error_lines.append("  1. If using weighted_random_configs, ensure weights sum to 100% and format is [[{dict}, int], ...]")
            error_lines.append("  2. Don't set use_weighted_random: true AND the same parameters directly")
            error_lines.append("  3. When changing from weighted random, explicitly set use_weighted_random: false")

            error_msg = "\n".join(error_lines)
            logger.warning(f"⚠️  {error_msg}")
            logger.warning("Prompting Claude to fix weighted random conflicts...")

            cost_data = self._prompt_for_correction(
                iteration=iteration,
                model=model,
                error_message=error_msg,
                session_id=session_id,
                working_dir=working_dir
            )
            self._accumulate_costs(total_cost_data, cost_data)

            # Reload and re-validate
            with open(config_file) as f:
                meta_config_schedule = json.load(f)

            # Re-run all weighted random checks
            weighted_random_errors = []

            # Re-run E1
            for iter_str, delta in meta_config_schedule.items():
                if "weighted_random_configs" in delta:
                    pool = delta["weighted_random_configs"]

                    if not isinstance(pool, list):
                        weighted_random_errors.append((iter_str, f"weighted_random_configs must be a list"))
                        continue

                    for i, entry in enumerate(pool):
                        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                            weighted_random_errors.append((iter_str, f"weighted_random_configs[{i}] invalid format"))
                            continue
                        if not isinstance(entry[0], dict):
                            weighted_random_errors.append((iter_str, f"weighted_random_configs[{i}][0] must be dict"))
                        if not isinstance(entry[1], int):
                            weighted_random_errors.append((iter_str, f"weighted_random_configs[{i}][1] must be int"))

                    try:
                        total = sum(weight for _, weight in pool if isinstance(_, dict))
                        if total != 100:
                            weighted_random_errors.append((iter_str, f"weights must sum to 100%, got {total}%"))
                    except (TypeError, ValueError):
                        weighted_random_errors.append((iter_str, "invalid weight values"))

            # Re-run E2
            for iter_str, delta in meta_config_schedule.items():
                if delta.get("use_weighted_random", False) and "weighted_random_configs" in delta:
                    weighted_params = extract_weighted_random_params(delta["weighted_random_configs"])
                    conflicting = [p for p in delta.keys() if p in weighted_params and p not in ["use_weighted_random", "weighted_random_configs"]]
                    if conflicting:
                        weighted_random_errors.append((iter_str, f"conflicts with weighted random: {conflicting}"))

            # Re-run E3
            for iter_str in sorted(meta_config_schedule.keys(), key=int):
                iter_num = int(iter_str)
                delta = meta_config_schedule[iter_str]
                if delta.get("use_weighted_random", False):
                    continue
                if iter_num > iteration + 1:
                    prev_iter_str = str(iter_num - 1)
                    if prev_iter_str in meta_config_schedule:
                        prev_delta = meta_config_schedule[prev_iter_str]
                        if prev_delta.get("use_weighted_random", False) and "weighted_random_configs" in prev_delta:
                            weighted_params = extract_weighted_random_params(prev_delta["weighted_random_configs"])
                            conflicting = [p for p in delta.keys() if p in weighted_params and p not in ["use_weighted_random", "weighted_random_configs", "comment"]]
                            if conflicting and "use_weighted_random" not in delta:
                                weighted_random_errors.append((iter_str, f"must set use_weighted_random: false (conflicts: {conflicting})"))

            if weighted_random_errors:
                error_details = "; ".join([f"iter {i}: {e}" for i, e in weighted_random_errors])
                raise RuntimeError(
                    f"meta_config_schedule.json still has weighted_random_configs conflicts after correction: {error_details}"
                )

        logger.info("✓ No weighted_random_configs conflicts in meta_config_schedule")

        # Re-discover strategies after all corrections — Claude may have created
        # additional strategies while fixing validation errors. Only log if the
        # set actually changed (no point announcing a rescan that found nothing new).
        if new_strategies_dir.exists():
            previous_strategy_names = strategy_names
            strategy_names = [
                d.name for d in new_strategies_dir.iterdir()
                if d.is_dir() and not d.name.startswith('.')
            ]
            if set(strategy_names) != set(previous_strategy_names):
                logger.info(f"Re-discovered {len(strategy_names)} strategies after corrections: {strategy_names}")

        # Validate each discovered strategy
        installed_this_firing = set()
        for strategy_name in strategy_names:
            strategy_dir = new_strategies_dir / strategy_name
            errors = self._validate_strategy_package(strategy_dir)

            if errors:
                error_msg = f"Strategy '{strategy_name}' validation errors:\n" + \
                           "\n".join(f"  - {e}" for e in errors)
                logger.warning(f"⚠️  {error_msg}")
                logger.warning("Prompting for correction...")

                cost_data = self._prompt_for_correction(
                    iteration=iteration,
                    model=model,
                    error_message=error_msg,
                    session_id=session_id,
                    working_dir=working_dir
                )
                self._accumulate_costs(total_cost_data, cost_data)

                # Re-validate after correction
                errors = self._validate_strategy_package(strategy_dir)
                if errors:
                    raise RuntimeError(
                        f"Strategy '{strategy_name}' still has validation errors after correction:\n" +
                        "\n".join(f"  - {e}" for e in errors)
                    )

            # Install validated strategy (prefixed with iter{N}_; capture installed name)
            installed_this_firing.add(
                self._install_strategy_package(strategy_name, iteration)
            )

        # Validate that every evolution_strategy reference in the schedule resolves
        # to an installed strategy. The installer prefixes new strategies with
        # iter{N}_ — references that forget the prefix won't resolve. Catches the
        # case where the meta-agent writes the schedule before realizing the
        # convention.
        unresolved = self._find_unresolved_strategy_refs(meta_config_schedule)
        if unresolved:
            error_msg = self._format_unresolved_strategy_error(
                unresolved, iteration, installed_this_firing
            )
            logger.warning(f"⚠️  Unresolved strategy references in meta_config_schedule")
            logger.warning("Prompting for correction...")

            cost_data = self._prompt_for_correction(
                iteration=iteration,
                model=model,
                error_message=error_msg,
                session_id=session_id,
                working_dir=working_dir,
            )
            self._accumulate_costs(total_cost_data, cost_data)

            # Reload and re-validate. If still unresolved, fail loudly.
            with open(config_file) as f:
                meta_config_schedule = json.load(f)
            unresolved = self._find_unresolved_strategy_refs(meta_config_schedule)
            if unresolved:
                raise RuntimeError(
                    "meta_config_schedule.json still references unresolved strategies "
                    f"after correction: {unresolved}"
                )

        return meta_config_schedule, config_delta

    def _request_reflection(
        self,
        iteration: int,
        model: str,
        session_id: str,
        working_dir: Path,
        iteration_dir: Path,
    ) -> Dict[str, Any]:
        """
        Request reflection from Claude about meta-evolution process.

        Asks Claude to provide advice for future meta-evolution sessions based on
        the completed work. This reflection captures insights about what worked,
        what didn't, and suggestions for process improvement.

        The reflection is saved to meta_evolution_reflection.md.

        Args:
            iteration: Current iteration number
            model: Model to use for the reflection
            session_id: Session ID for this meta-evolution call
            working_dir: Working directory for Claude Code

        Returns:
            Cost data dictionary

        Note:
            Errors are logged but do not raise exceptions - reflection should
            never break the research run.
        """
        prompt = f"""Take a moment to reflect on this firing's work. You're in a persistent session — your next firing (a few iterations from now) will have this iteration's context already in memory, so this reflection serves two purposes: an audit-trail checkpoint for the human reviewing the run, and a chance to consolidate the insights you most want to carry forward.

Please consider:
- What patterns or insights from this iteration's data are worth emphasizing for next time?
- What was challenging or time-consuming about the analysis or implementation?
- Were the provided tools and reports helpful? Anything you wished you had?
- What would you do differently in the next firing?
- Any prompt or tooling changes worth flagging for the human maintainer of this system?

**Keep your reflection concise - 300 lines or less.**

Save your reflection to `iteration_{iteration:03d}/meta_evolution_reflection.md`.

After saving the reflection, respond with: "REFLECTION COMPLETE"
"""

        try:
            cost_data = self._call_claude_code(
                prompt=prompt,
                model=model,
                session_id=session_id,
                working_dir=working_dir,
                resume_session=True  # Reflection continues the active session
            )

            reflection_file = iteration_dir / "meta_evolution_reflection.md"
            if reflection_file.exists():
                logger.info(f"✓ Meta-evolution reflection saved: {os.path.relpath(reflection_file)}")
            else:
                logger.warning("Reflection completed but meta_evolution_reflection.md not found")

            return cost_data

        except Exception as e:
            logger.warning(f"Failed to request meta-evolution reflection: {e}")
            return {'total_cost': 0.0, 'calls': 0}

    def _save_session_transcript(self, session_id: str, lookup_dir: Path, output_path: Path):
        """
        Summarize Claude Code session transcript and save it.

        Args:
            session_id: Claude Code session ID
            lookup_dir: Directory whose path-hash Claude CLI used to store the
                transcript under ~/.claude/projects/<sanitized>/. Must match the
                cwd= passed to Claude CLI (self.output_dir for meta-evolution).
            output_path: Where to write the human-readable summary.

        Errors are logged but do not raise exceptions — transcript saving should
        never break the research run.
        """
        try:
            from RoboPhD.utilities.transcript_summarizer import find_transcript, summarize_transcript

            chat_file = find_transcript(lookup_dir, session_id)
            if not chat_file:
                logger.warning(f"Session transcript not found for session {session_id}")
                return

            summary_path = summarize_transcript(chat_file, output_path)
            summary_size = summary_path.stat().st_size
            logger.info(f"Saved session summary: {os.path.relpath(summary_path)} ({summary_size/1024:.1f} KB)")

        except Exception as e:
            logger.warning(f"Failed to save session summary: {e}")
