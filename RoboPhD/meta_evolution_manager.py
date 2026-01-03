"""
Meta-Evolution Manager for RoboPhD Phase 3.

Orchestrates meta-evolution process that analyzes research system performance
and creates new evolution strategies while adjusting configuration parameters.

Architecture:
- Single persistent Claude Code session across entire experiment
- Four-phase execution: Planning → Validation → Implementation → Validation
- Budget tracking across all phases (Phase 1 + Phase 2 + Evolution + Meta-evolution)
- Automatic termination when budget exhausted
"""

import json
import logging
import subprocess
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any

from RoboPhD.config_manager import ConfigManager, ConfigSource

logger = logging.getLogger(__name__)


class MetaEvolutionManager:
    """
    Manages meta-evolution process for optimizing research system configuration.

    Meta-evolution runs as the last step of iteration K, analyzing K's results
    and proposing configuration changes for K+1 and beyond.
    """

    def __init__(self, experiment_dir: Path, config_manager: ConfigManager):
        """
        Initialize Meta-Evolution Manager.

        Args:
            experiment_dir: Root directory of research experiment
            config_manager: Configuration manager for the experiment
        """
        self.experiment_dir = experiment_dir
        self.config_manager = config_manager
        self.strategies_dir = Path("RoboPhD/meta_evolution_strategies")  # Source directory
        self.output_dir = experiment_dir / "meta_evolution_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def should_run_meta_evolution(self, iteration: int) -> bool:
        """
        Check if meta-evolution should run for this iteration.

        Args:
            iteration: Current iteration number

        Returns:
            True if meta_evolution_strategy is configured and not "none"
        """
        config = self.config_manager.get_config(iteration)
        strategy = config.get("meta_evolution_strategy")
        return strategy is not None and strategy != "none"

    def run_meta_evolution(
        self,
        iteration: int
    ) -> tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute meta-evolution for this iteration using fresh session.

        Args:
            iteration: Current iteration number

        Returns:
            Tuple of (meta_config_schedule, cost_data):
            - meta_config_schedule: Dict mapping iteration numbers to config deltas (or None if no changes)
            - cost_data: Dict containing cost tracking information (total_cost, calls, tokens_in, tokens_out)
        """
        config = self.config_manager.get_config(iteration)
        strategy_name = config["meta_evolution_strategy"]
        model = config.get("meta_evolution_model", "opus-4.5")

        # Generate fresh session ID for this meta-evolution call
        session_id = str(uuid.uuid4())

        logger.info(f"\n{'=' * 60}")
        logger.info(f"🧬 META-EVOLUTION - Iteration {iteration}")
        logger.info(f"Strategy: {strategy_name}")
        logger.info(f"Model: {model}")
        logger.info(f"Session ID: {session_id}")
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

        # PHASE 1: Planning (Round 1)
        logger.info("📋 Phase 1: Planning and reasoning...")
        context = self._gather_context(iteration, config)
        strategy = self._load_meta_strategy(strategy_name)

        cost_data = self._execute_round_1(
            strategy=strategy,
            context=context,
            iteration=iteration,
            iteration_output=iteration_output,
            model=model,
            session_id=session_id
        )
        self._accumulate_costs(total_cost_data, cost_data)

        # PHASE 2: Validation (check reasoning.md exists)
        logger.info("✅ Phase 2: Validating reasoning.md...")
        reasoning_path = iteration_output / "reasoning.md"
        if not reasoning_path.exists():
            logger.warning("⚠️  reasoning.md not found, prompting for correction...")
            cost_data = self._prompt_for_correction(
                iteration=iteration,
                model=model,
                error_message="reasoning.md is missing. Please create it as specified in Round 1 instructions.",
                session_id=session_id,
                working_dir=iteration_output
            )
            self._accumulate_costs(total_cost_data, cost_data)

            # Re-check after correction
            if not reasoning_path.exists():
                raise RuntimeError("Phase 1 failed to create reasoning.md even after correction attempt")

        logger.info(f"✓ reasoning.md created ({reasoning_path.stat().st_size} bytes)")

        # PHASE 3: Implementation (Round 2)
        logger.info("🔨 Phase 3: Creating strategies and configuration...")
        cost_data = self._execute_round_2(
            iteration=iteration,
            iteration_output=iteration_output,
            model=model,
            session_id=session_id
        )
        self._accumulate_costs(total_cost_data, cost_data)

        # PHASE 4: Validation and installation
        logger.info("✅ Phase 4: Validating and installing outputs...")
        meta_config_schedule = self._parse_and_validate_outputs(
            iteration=iteration,
            model=model,
            total_cost_data=total_cost_data,
            session_id=session_id,
            working_dir=iteration_output
        )

        # PHASE 5: Champion viability assessment
        logger.info("🤔 Phase 5: Assessing champion viability...")
        cost_data = self._assess_champion_viability(
            iteration=iteration,
            model=model,
            session_id=session_id,
            working_dir=iteration_output,
            total_cost_data=total_cost_data
        )
        self._accumulate_costs(total_cost_data, cost_data)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"✓ Meta-evolution complete for iteration {iteration}")
        logger.info(f"Cost: ${total_cost_data['total_cost']:.4f}")
        logger.info(f"{'=' * 60}\n")

        return meta_config_schedule, total_cost_data

    def check_budget_and_maybe_terminate(self, iteration: int) -> bool:
        """
        Check if budget is exhausted after iteration completes.

        Calculates total cost across all phases:
        - Phase 1 (DB Analysis): Claude CLI costs
        - Phase 2 (SQL Generation): API costs (from checkpoint.total_cost)
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
        budget = config.get("meta_evolution_budget")

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
        Calculate total cost across all phases through specified iteration.

        Returns sum of:
        - Phase 1 costs (from iteration_claude_costs)
        - Phase 2 API costs (from checkpoint.total_cost)
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

        # Phase 2 API costs (already in checkpoint)
        phase2_cost = checkpoint.get("total_cost", 0.0)

        # Phase 1, Evolution, and Meta-evolution costs (from Claude CLI)
        phase1_cost = 0.0
        evolution_cost = 0.0
        meta_evolution_cost = 0.0

        iteration_costs = checkpoint.get("iteration_claude_costs", [])
        for iter_num in range(1, through_iteration + 1):
            if iter_num - 1 < len(iteration_costs):
                cost_dict = iteration_costs[iter_num - 1]
                phase1_cost += cost_dict.get('phase1_cost', 0.0)
                evolution_cost += cost_dict.get('evolution_cost', 0.0)
                meta_evolution_cost += cost_dict.get('meta_evolution_cost', 0.0)

        return phase1_cost + phase2_cost + evolution_cost + meta_evolution_cost

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

    def _install_strategy_package(self, strategy_name: str, iteration: int):
        """
        Install validated strategy package to evolution_strategies/ directory.

        Copies strategy from meta_evolution_output/iteration_XXX/new_strategies/NAME/
        to <experiment_dir>/evolution_strategies/NAME/

        Args:
            strategy_name: Name of strategy to install
            iteration: Iteration that created this strategy
        """
        source_dir = (
            self.output_dir /
            f"iteration_{iteration:03d}" /
            "new_strategies" /
            strategy_name
        )

        dest_dir = self.experiment_dir / "evolution_strategies" / strategy_name

        logger.info(f"Installing strategy '{strategy_name}' to evolution_strategies/")

        # Create evolution_strategies directory if it doesn't exist
        dest_dir.parent.mkdir(parents=True, exist_ok=True)

        # Copy strategy package
        if dest_dir.exists():
            import shutil
            shutil.rmtree(dest_dir)

        import shutil
        shutil.copytree(source_dir, dest_dir)

        logger.info(f"✓ Installed strategy '{strategy_name}'")

    def _update_meta_evolution_costs(self, iteration: int, cost_data: Dict) -> None:
        """
        Update checkpoint with meta-evolution costs for this iteration.

        Costs are tracked in checkpoint.iteration_claude_costs[iteration]['meta_evolution_cost']
        parallel to phase1_cost and evolution_cost.

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
                "Claude CLI not found. Install with: pip install claude-cli"
            )

    def _call_claude_code(
        self,
        prompt: str,
        model: str,
        session_id: str,
        working_dir: Path,
        session_created: bool = False
    ) -> Dict[str, Any]:
        """
        Call Claude Code CLI with the given prompt.

        Args:
            prompt: Prompt to send to Claude Code
            model: Model to use (API name like "sonnet-4.5")
            session_id: Session ID for this meta-evolution call
            working_dir: Working directory for Claude Code (iteration-specific output dir)
            session_created: True if this is a continuation of an existing session

        Returns:
            Dictionary with cost and usage information

        Raises:
            RuntimeError: If Claude Code call fails
        """
        from RoboPhD.config import CLAUDE_CLI_MODEL_MAP

        # Build command
        claude_cli = self._get_claude_cli_path()
        # Map API model name to Claude CLI name (e.g., 'sonnet-4.5' -> 'sonnet')
        cli_model = CLAUDE_CLI_MODEL_MAP.get(model, model)

        cmd = [
            str(claude_cli),
            "--model", cli_model
        ]

        # Add MCP config for error analysis tools
        # Path to MCP config relative to project root
        project_root = self.experiment_dir.parent
        mcp_config_path = project_root / "RoboPhD" / "mcp_configs" / "error_analysis_tools.json"
        if mcp_config_path.exists():
            cmd.extend(["--mcp-config", str(mcp_config_path)])

        # Use explicit session management to prevent interference
        if session_created:
            # Resume existing session by ID
            cmd.extend(["--resume", session_id])
        else:
            # Create new session with explicit ID
            cmd.extend(["--session-id", session_id])

        cmd.extend([
            "--print", prompt,
            "--output-format", "json",  # Get JSON output for cost tracking
            "--permission-mode", "bypassPermissions",  # Allow automation without prompts
            "--settings", '{"autoCompact": true}'  # Proactively compact when context gets low
        ])

        logger.debug(f"Calling Claude Code: {' '.join(cmd[:4])}...")

        try:
            # Run in iteration-specific working directory
            result = subprocess.run(
                cmd,
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes default
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

Please fix these issues and recreate the required files.

Remember:
- `meta_config_schedule.json` is REQUIRED (even if empty: {{}})
- Strategies go in `meta_evolution_output/iteration_{iteration:03d}/new_strategies/strategy_name/`
- Each strategy needs `strategy.md` with valid YAML frontmatter (name and description fields)
- Python tools must have valid syntax
"""

        # Resume session and prompt for correction
        return self._call_claude_code(
            prompt=correction_prompt,
            model=model,
            session_id=session_id,
            working_dir=working_dir,
            session_created=True  # Always a continuation when correcting
        )

    def _execute_round_1(
        self,
        strategy: str,
        context: Dict,
        iteration: int,
        iteration_output: Path,
        model: str,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Execute Round 1: Planning and reasoning.

        With fresh sessions, always includes full strategy text since every
        session starts from scratch with no prior context.

        Args:
            session_id: Session ID for this meta-evolution call

        Returns:
            Cost data dictionary
        """
        # Always include full strategy text for fresh sessions
        budget_info = self._format_budget_status(context["budget"], iteration)
        strategy_with_budget = strategy.replace(
            "**Budget Status**:",
            budget_info
        )

        prompt = f"""
## Performance Rankings Across All Iterations

{context['ranking_table']}

{strategy_with_budget}

## Current State (Iteration {iteration})

### Recent Performance
{self._format_interim_reports(context.get("interim_reports", []))}

## Understanding Evolution Strategies

If you propose creating new evolution strategies, you need to understand their structure.

### Evolution Strategy Structure

Each strategy is a package in `evolution_strategies/strategy_name/`:

**Required:**
- `strategy.md` - Main strategy prompt with:
  - YAML frontmatter: `name` and `description` fields
  - Instructions for evolution AI on how to create agents
  - Typical sections: Context, Strategy approach, Required outputs, Success metrics

**Optional but Powerful:**
- `strategy_tools/` - Helper scripts for complex workflows
  - **Example 1**: Research paper selection with state tracking (used by research_driven strategies)
  - **Example 2**: Custom error analysis scripts that generate specialized reports
  - Evolution AI calls these via bash commands in the strategy
  - Can maintain state across iterations via JSON files

### Existing Strategies (for reference)

Review strategies in `evolution_strategies/` to understand patterns:
- Refinement-based approaches
- Cross-pollination of successful patterns
- Research-driven with academic paper integration
- Judgment-based with different focus areas

Examine their `strategy.md` files to see structure and best practices.

### When Proposing New Strategies

In your `reasoning.md`, be specific:
- **Name**: What to call it (hyphenated, lowercase)
- **Core Idea**: What makes it different from existing strategies
- **Implementation Approach**: Key sections the strategy.md will need
- **Strategy Tools**: What helper scripts would make this strategy more effective
  - Consider: specialized error analysis, state tracking across iterations
- **Expected Benefits**: Why this will improve evolution outcomes

### Implementation Details (for Round 2)

If you decide to create new strategies, in Round 2 you will put them in:
`new_strategies/strategy_name/`

- Put ALL new strategies in the `new_strategies/` directory
- Each subdirectory under `new_strategies/` represents one strategy
- Each strategy needs: `strategy.md` with YAML frontmatter (name, description fields)
- Optional: `strategy_tools/` directory with Python/shell scripts
- You can create 0, 1, or multiple strategies
- Review `../../evolution_strategies/` for examples and patterns

## Your Task - Round 1: Planning and Reasoning

Analyze the system's evolution performance and document your reasoning.

Create: `reasoning.md`

Include:
1. **Performance Analysis**: What patterns do you see in recent iterations?
2. **Strategy Assessment**: Which strategies are working? Which aren't?
3. **Opportunities Identified**: What gaps or patterns suggest new approaches?
4. **Proposed Actions**:
   - New evolution strategies to create (if any)
   - Configuration changes to make
   - Expected impact and rationale

**Do not create strategy files or meta_config_schedule.json yet** - this round is for planning only.
"""

        # First call in this meta-evolution session
        return self._call_claude_code(
            prompt=prompt,
            model=model,
            session_id=session_id,
            working_dir=iteration_output,
            session_created=False  # First call
        )

    def _execute_round_2(
        self,
        iteration: int,
        iteration_output: Path,
        model: str,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Execute Round 2: Implementation.

        Args:
            session_id: Session ID for this meta-evolution call

        Returns:
            Cost data dictionary
        """
        prompt = f"""
## Meta-Evolution Implementation - Iteration {iteration} (Round 2)

You previously created reasoning.md with your analysis and proposed actions.

## Your Task - Round 2: Implementation

Implement the plan from reasoning.md:

1. **New Evolution Strategies** (if proposed): Create strategy packages in:
   `new_strategies/strategy_name/`
   - Put ALL new strategies in the `new_strategies/` directory
   - Each subdirectory under `new_strategies/` represents one strategy
   - Each strategy needs: `strategy.md` with YAML frontmatter (name, description fields)
   - Optional: `strategy_tools/` directory with Python/shell scripts for complex workflows
   - Review `../../evolution_strategies/` for examples and patterns (especially research_driven* for strategy_tools usage)
   - You can create 0, 1, or multiple strategies

2. **Configuration Changes** (required): Create:
   `meta_config_schedule.json`
   - Configuration changes for upcoming iterations
   - Must be consistent with reasoning.md
   - This file is REQUIRED even if no strategies are created
   - Format: {{"<iteration>": {{"param": value}}, ...}}

Example meta_config_schedule.json:
{{
  "12": {{
    "use_weighted_random": true,
    "weighted_random_configs": [
      [{{"evolution_strategy": "refinement"}}, 25],
      [{{"evolution_strategy": "cross_pollination"}}, 50],
      [{{"evolution_strategy": "none"}}, 25]
    ]
  }},
  "15": {{
    "databases_per_iteration": 10,
    "questions_per_database": 30
  }}
}}

**Forbidden Parameters** (do NOT include in meta_config_schedule.json):
- Initial config: `initial_agents`, `agents_directory`, `initial_strategies`, `strategies_directory`
- Dataset: `dataset`
- Meta-evolution self-reference: `meta_evolution_strategy`, `meta_evolution_model`, `meta_evolution_budget`

These create circular dependencies or modify immutable system state.

**CRITICAL: Weighted Random Override Behavior**
If `use_weighted_random: true` is enabled (either in current config or inherited from previous iteration):
- Weighted random selection applies BEFORE meta_config_schedule
- To use a specific `evolution_strategy` for an iteration, you MUST explicitly set `use_weighted_random: false` in that iteration's config
- Example: If you want iteration 9 to use "evidence_driven_refinement", you must include:
  {{
    "9": {{
      "use_weighted_random": false,
      "evolution_strategy": "evidence_driven_refinement",
      ...
    }}
  }}
- Without `use_weighted_random: false`, the weighted random pool will override your intended strategy

Framework will:
- Discover strategies by scanning `new_strategies/` for subdirectories
- Validate both strategies and config
- Install valid strategies to evolution_strategies/
- Integrate meta_config_schedule via ConfigManager
"""

        # Resume session from Round 1
        return self._call_claude_code(
            prompt=prompt,
            model=model,
            session_id=session_id,
            working_dir=iteration_output,
            session_created=True  # Continuation from Round 1
        )

    def _gather_context(self, iteration: int, config: Dict[str, Any]) -> Dict:
        """
        Gather information for meta-evolution analysis.

        Returns dictionary with interim reports, ranking table, and budget.
        """
        # Get budget information
        budget = config.get("meta_evolution_budget")
        total_cost = self._calculate_total_cost(iteration)
        budget_remaining = (budget - total_cost) if budget else None

        context = {
            "current_iteration": iteration,
            "interim_reports": [],
            "budget": {
                "total": budget,
                "consumed": total_cost,
                "remaining": budget_remaining
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

        # Generate performance rankings table
        checkpoint = self._load_checkpoint()
        if checkpoint:
            from RoboPhD.ranking_table import generate_ranking_table

            ranking_table = generate_ranking_table(
                test_history=checkpoint.get('test_history', []),
                performance_records=checkpoint.get('performance_records', {}),
                for_evolution=True  # Use simplified format like regular evolution
            )
            context["ranking_table"] = ranking_table
        else:
            context["ranking_table"] = "(No performance data available yet)"

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
        """Format budget status for injection into strategy prompt."""
        total = budget_info["total"]
        consumed = budget_info["consumed"]
        remaining = budget_info["remaining"]

        if total is None:
            return "**Budget Status**: No budget limit set"

        return f"""**Budget Status**:
- **Total budget**: ${total:.2f}
- **Consumed so far**: ${consumed:.2f}
- **Remaining**: ${remaining:.2f}
- **Iterations completed**: {iteration}"""

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
    ) -> Dict:
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
            Validated meta_config_schedule
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

        # Validate meta_config_schedule.json (REQUIRED)
        config_file = iteration_dir / "meta_config_schedule.json"
        if not config_file.exists():
            logger.warning("⚠️  meta_config_schedule.json not found, prompting for correction...")
            cost_data = self._prompt_for_correction(
                iteration=iteration,
                model=model,
                error_message="meta_config_schedule.json is missing (required). "
                             "Please create it even if proposing no configuration changes.",
                session_id=session_id,
                working_dir=working_dir
            )
            self._accumulate_costs(total_cost_data, cost_data)

            # Re-check after correction
            if not config_file.exists():
                raise RuntimeError(
                    "meta_config_schedule.json still missing after correction attempt"
                )

        with open(config_file) as f:
            meta_config_schedule = json.load(f)

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

        # Re-discover strategies after all corrections
        # (Claude may have created strategies during correction prompts)
        if new_strategies_dir.exists():
            strategy_names = [
                d.name for d in new_strategies_dir.iterdir()
                if d.is_dir() and not d.name.startswith('.')
            ]
            if strategy_names:
                logger.info(f"Re-discovered {len(strategy_names)} strategies after corrections: {strategy_names}")

        # Validate each discovered strategy
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

            # Install validated strategy
            self._install_strategy_package(strategy_name, iteration)

        return meta_config_schedule

    def _assess_champion_viability(
        self,
        iteration: int,
        model: str,
        session_id: str,
        working_dir: Path,
        total_cost_data: Dict
    ) -> Dict[str, Any]:
        """
        Query meta-evolution agent about champion viability for early termination.

        This is a self-assessment where meta-evolution evaluates its own proposed
        changes - asking: "Will the changes YOU just made help beat the current champion?"

        Args:
            iteration: Current iteration number
            model: Model to use for the query
            session_id: Session ID for this meta-evolution call
            working_dir: Working directory for Claude Code
            total_cost_data: Cost accumulator (updated if corrections needed)

        Returns:
            Cost data dictionary

        Raises:
            RuntimeError: If validation fails after retry
        """
        # Get current champion info from checkpoint
        checkpoint = self._load_checkpoint()
        performance_records = checkpoint.get('performance_records', {})

        if not performance_records:
            logger.warning("No performance records available for champion assessment")
            # Create empty assessment file and return
            assessment_path = working_dir / "champion_viability_assessment.md"
            assessment_path.write_text(
                "# Champion Viability Assessment\n\n"
                "(No performance records available yet)\n"
            )
            return {'total_cost': 0.0, 'calls': 0}

        # Find current ELO leader
        champion = max(performance_records.keys(),
                      key=lambda a: performance_records[a]['elo'])
        champion_elo = performance_records[champion]['elo']
        champion_accuracy = performance_records[champion]['mean_accuracy']

        prompt = f"""## Champion Viability Assessment - Iteration {iteration}

You have just completed meta-evolution for iteration {iteration}. You analyzed the system's
performance and created `meta_config_schedule.json` with proposed configuration changes.

**Current Champion**: {champion}
- ELO Score: {champion_elo:.0f}
- Mean Accuracy: {champion_accuracy:.1f}%

**Your Configuration Changes**: See `meta_config_schedule.json`

## Self-Assessment Question

This is a self-assessment of YOUR OWN WORK. You just proposed changes in `meta_config_schedule.json`.

**Question**: Do you think it is likely that in the next four iterations we will produce an
agent which would be able to exceed the current champion in ELO if they were repeatedly tested
head to head over many iterations, or do you think it more likely that the current champion
would win?

Consider:
1. The configuration changes YOU just proposed
2. Recent evolution trends and strategy effectiveness
3. Whether the champion's approach has been thoroughly explored
4. Potential for breakthrough innovations with your proposed changes

Please provide your analysis and prediction.

Save your assessment to: `champion_viability_assessment.md`

Include:
1. **Analysis** (2-3 paragraphs):
   - Assessment of your proposed changes
   - Recent trends in agent evolution
   - Champion's strengths and potential vulnerabilities
   - Likelihood of breakthrough innovations

2. **Prediction** (must include EXACTLY ONE of these phrases):
   - "A NEW AGENT MORE LIKELY TO WIN" - if you believe your changes will likely produce a superior agent
   - "CURRENT CHAMPION MORE LIKELY TO DEFEAT ALL NEW AGENTS" - if you believe the champion would likely remain dominant

After saving, respond with: "ASSESSMENT COMPLETE"
"""

        # Resume existing session for this query
        cost_data = self._call_claude_code(
            prompt=prompt,
            model=model,
            session_id=session_id,
            working_dir=working_dir,
            session_created=True  # Continuation
        )

        # Validate the assessment
        assessment_path = working_dir / "champion_viability_assessment.md"

        # Check 1: File exists
        if not assessment_path.exists():
            logger.warning("⚠️  champion_viability_assessment.md not found, prompting for correction...")
            correction_cost = self._prompt_for_correction(
                iteration=iteration,
                model=model,
                error_message=(
                    "champion_viability_assessment.md is missing. Please create it with:\n"
                    "1. Analysis section (2-3 paragraphs)\n"
                    "2. Prediction with EXACTLY ONE of: 'A NEW AGENT MORE LIKELY TO WIN' or 'CURRENT CHAMPION MORE LIKELY TO DEFEAT ALL NEW AGENTS'"
                ),
                session_id=session_id,
                working_dir=working_dir
            )
            self._accumulate_costs(total_cost_data, correction_cost)

            # Re-check after correction
            if not assessment_path.exists():
                raise RuntimeError(
                    "champion_viability_assessment.md still missing after correction attempt"
                )

        # Check 2: File contains required prediction
        assessment_text = assessment_path.read_text()
        has_new_agent = "A NEW AGENT MORE LIKELY TO WIN" in assessment_text
        has_champion = "CURRENT CHAMPION MORE LIKELY TO DEFEAT ALL NEW AGENTS" in assessment_text

        if not (has_new_agent or has_champion):
            logger.warning("⚠️  Assessment missing required prediction, prompting for correction...")
            correction_cost = self._prompt_for_correction(
                iteration=iteration,
                model=model,
                error_message=(
                    "champion_viability_assessment.md is missing the required prediction.\n"
                    "You must include EXACTLY ONE of these phrases:\n"
                    "  - 'A NEW AGENT MORE LIKELY TO WIN'\n"
                    "  - 'CURRENT CHAMPION MORE LIKELY TO DEFEAT ALL NEW AGENTS'\n\n"
                    "Please update the file to include your prediction."
                ),
                session_id=session_id,
                working_dir=working_dir
            )
            self._accumulate_costs(total_cost_data, correction_cost)

            # Re-check after correction
            assessment_text = assessment_path.read_text()
            has_new_agent = "A NEW AGENT MORE LIKELY TO WIN" in assessment_text
            has_champion = "CURRENT CHAMPION MORE LIKELY TO DEFEAT ALL NEW AGENTS" in assessment_text

            if not (has_new_agent or has_champion):
                raise RuntimeError(
                    "champion_viability_assessment.md still missing required prediction after correction. "
                    "Must include 'A NEW AGENT MORE LIKELY TO WIN' or 'CURRENT CHAMPION MORE LIKELY TO DEFEAT ALL NEW AGENTS'."
                )

        # Check 3: File contains both predictions (ambiguous)
        if has_new_agent and has_champion:
            logger.warning("⚠️  Assessment contains both predictions (ambiguous), prompting for correction...")
            correction_cost = self._prompt_for_correction(
                iteration=iteration,
                model=model,
                error_message=(
                    "champion_viability_assessment.md contains BOTH predictions, making it ambiguous.\n"
                    "You must include EXACTLY ONE of:\n"
                    "  - 'A NEW AGENT MORE LIKELY TO WIN'\n"
                    "  - 'CURRENT CHAMPION MORE LIKELY TO DEFEAT ALL NEW AGENTS'\n\n"
                    "Please update to include only your final prediction."
                ),
                session_id=session_id,
                working_dir=working_dir
            )
            self._accumulate_costs(total_cost_data, correction_cost)

            # Re-check after correction
            assessment_text = assessment_path.read_text()
            has_new_agent = "A NEW AGENT MORE LIKELY TO WIN" in assessment_text
            has_champion = "CURRENT CHAMPION MORE LIKELY TO DEFEAT ALL NEW AGENTS" in assessment_text

            if has_new_agent and has_champion:
                raise RuntimeError(
                    "champion_viability_assessment.md still contains both predictions after correction. "
                    "Must include EXACTLY ONE of: 'A NEW AGENT MORE LIKELY TO WIN' or 'CURRENT CHAMPION MORE LIKELY TO DEFEAT ALL NEW AGENTS'."
                )
            if not (has_new_agent or has_champion):
                raise RuntimeError(
                    "champion_viability_assessment.md missing prediction after correction. "
                    "Must include EXACTLY ONE of: 'A NEW AGENT MORE LIKELY TO WIN' or 'CURRENT CHAMPION MORE LIKELY TO DEFEAT ALL NEW AGENTS'."
                )

        # Validation passed
        prediction = "A NEW AGENT MORE LIKELY TO WIN" if has_new_agent else "CURRENT CHAMPION MORE LIKELY TO DEFEAT ALL NEW AGENTS"
        logger.info(f"✓ Champion viability assessment complete: {prediction}")
        logger.info(f"  Assessment saved: {assessment_path.name}")

        return cost_data
