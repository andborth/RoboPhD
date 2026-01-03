"""
Configuration Manager for RoboPhD

Implements unified delta-based configuration system where:
- All parameters are treated uniformly
- Configuration evolves through deltas over iterations
- Full traceability via config_change_history
- Meta-evolution ready
"""

import json
import random
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class ConfigSource(Enum):
    """Source of configuration changes."""
    CLI = "cli"
    SCHEDULE = "schedule"
    WEIGHTED_RANDOM = "weighted_random"
    META_EVOLUTION = "meta_evolution"
    USER_MODIFICATION = "user_modification"
    RESUME = "resume"
    EXTEND = "extend"


class ConfigManager:
    """
    Manages configuration across iterations using delta-based inheritance.

    Key concepts:
    - Iteration 0: Pure system defaults (immutable after init)
    - Iteration 1: User initial configuration (delta from defaults)
    - Iteration 2+: Deltas from schedule, weighted random, or meta-evolution

    All parameters are treated uniformly - no special handling.
    """

    def __init__(self):
        """Initialize empty ConfigManager."""
        self.iteration_configs: Dict[int, Dict[str, Any]] = {}
        self.resolved_configs: Dict[int, Dict[str, Any]] = {}
        self.config_change_history: List[Dict[str, Any]] = []
        self.current_iteration: int = 0  # Track current iteration for lazy evaluation

    def get_defaults(self) -> Dict[str, Any]:
        """
        Return system defaults for all parameters.

        These defaults represent iteration 0 configuration.
        """
        return {
            # Dataset and sampling
            "dataset": "train-filtered",
            "databases_per_iteration": 5,
            "questions_per_database": 30,
            "agents_per_iteration": 3,

            # Models
            "eval_model": "haiku-4.5",
            "analysis_model": "haiku-4.5",
            "evolution_model": "opus-4.5",

            # Evolution parameters (NO LONGER SPECIAL!)
            "evolution_strategy": "cross_pollination_tool_only",

            # Meta-evolution parameters
            "meta_evolution_strategy": None,       # Which meta-evolution strategy to use
            "meta_evolution_model": "opus-4.5",    # Model for meta-evolution
            "meta_evolution_budget": 100.0,        # Total budget in dollars (default: $100)

            # Deep Focus
            "new_agent_test_rounds": 1,

            # SQL generation
            "verification_retries": 2,
            "temperature_strategy": "progressive",

            # Performance
            "max_concurrent_dbs": 12,

            # Timeouts
            "phase1_timeout": 1800,
            "sql_timeout": 3600,
            "evolution_timeout": 1800,
            "llm_call_timeout": 120,  # Per-call LLM timeout (2 min) - affects local models

            # Other
            "debug_log_probability": 0.02,

            # Meta-parameters (control config system behavior)
            "config_schedule": {},
            "meta_config_schedule": {},
            "weighted_random_configs": [],
            "use_weighted_random": False,

            # Immutable parameters (user-set once at iteration 1, cannot change after)
            "initial_agents": ["naive"],
            "agents_directory": None,
            "initial_strategies": ["cross_pollination_tool_only"],
            "strategies_directory": None
        }

    def set_initial_config(self,
                          user_config: Dict[str, Any],
                          source: ConfigSource = ConfigSource.CLI) -> None:
        """
        Set iteration 0 (defaults) and iteration 1 (user overrides).

        Args:
            user_config: User-provided configuration overrides
            source: Source of configuration (CLI, RESUME, etc.)

        Raises:
            ValueError: If user_config contains unknown parameters
        """
        # Validate parameters
        self._validate_parameters(user_config, "Initial configuration", source)

        # Iteration 0 = Pure defaults
        defaults = self.get_defaults()
        self.iteration_configs[0] = defaults.copy()
        self.resolved_configs[0] = defaults.copy()

        # Iteration 1 = User overrides only (delta from defaults)
        self.iteration_configs[1] = user_config.copy()

        # Resolve iteration 1 = defaults + user overrides
        resolved = defaults.copy()
        resolved.update(user_config)
        self.resolved_configs[1] = resolved

        # Record in history
        self.config_change_history.append({
            "iteration": 1,
            "source": source.value,
            "delta": user_config.copy(),
            "rationale": "Initial user configuration"
        })

    def apply_delta(self,
                   iteration: int,
                   delta: Dict[str, Any],
                   source: ConfigSource,
                   rationale: str = "") -> None:
        """
        Apply delta to specific iteration.

        Args:
            iteration: Iteration number to apply delta to
            delta: Configuration changes to apply
            source: Source of the delta
            rationale: Explanation for the change

        Raises:
            ValueError: If trying to modify immutable parameters after iteration 1
                       or if delta contains unknown parameters
        """
        # Validate parameters
        self._validate_parameters(delta, f"Iteration {iteration} delta", source)

        # Validate immutable parameters
        self._validate_immutable_params(iteration, delta)

        # Store delta (merge with existing to preserve previous deltas)
        if iteration not in self.iteration_configs:
            self.iteration_configs[iteration] = {}
        self.iteration_configs[iteration].update(delta)

        # Resolve config
        resolved = self._resolve_config(iteration)
        self.resolved_configs[iteration] = resolved

        # Clear cached configs for iterations > iteration since they may have inherited old values
        to_clear = [i for i in self.resolved_configs if i > iteration]
        for i in to_clear:
            del self.resolved_configs[i]

        # Record in history
        self.config_change_history.append({
            "iteration": iteration,
            "source": source.value,
            "delta": delta.copy(),
            "rationale": rationale
        })

    def integrate_meta_config_schedule(self, meta_config_schedule: Dict[str, Any], iteration: int) -> None:
        """
        Integrate meta-evolution's config schedule into the configuration system.

        Meta-evolution produces a schedule like:
        {
            "12": {"use_weighted_random": true, "weighted_random_configs": [...]},
            "15": {"evolution_strategy": "pattern_exploiter"}
        }

        This method stores the schedule as a regular config parameter, following the
        "just another parameter" philosophy. It will be applied BEFORE user config_schedule
        in _resolve_config(). User config_schedule always takes precedence.

        Args:
            meta_config_schedule: Dictionary mapping iteration numbers to config deltas
            iteration: Current iteration number where meta-evolution ran

        Raises:
            ValueError: If meta_config_schedule contains forbidden parameters
        """
        # Use passed iteration number instead of max(keys()) to avoid bugs
        # when iterations don't have explicit deltas in iteration_configs
        current_iteration = iteration

        # Validate each iteration's delta
        for iter_str, delta in meta_config_schedule.items():
            # Validate parameters
            self._validate_parameters(delta, f"Meta-evolution schedule for iteration {iter_str}", ConfigSource.META_EVOLUTION)

            # Validate no forbidden parameters
            forbidden = self._get_meta_evolution_forbidden_params()
            for param in delta.keys():
                if param in forbidden:
                    raise ValueError(
                        f"Meta-evolution attempted to modify forbidden parameter '{param}'. "
                        f"Forbidden parameters: {forbidden}"
                    )

        # Filter out any entries for iterations <= current_iteration
        # These iterations have already run and been cached, so retroactive
        # schedule entries would cause validation mismatches
        filtered_schedule = {
            k: v for k, v in meta_config_schedule.items()
            if int(k) > current_iteration
        }

        if not filtered_schedule:
            # No future iterations to schedule
            return

        # Store schedule at next iteration (current iteration has already completed)
        # This matches --modify-config behavior: changes apply to future iterations
        self.apply_delta(
            iteration=current_iteration + 1,
            delta={"meta_config_schedule": filtered_schedule},
            source=ConfigSource.META_EVOLUTION,
            rationale=f"Meta-evolution schedule with {len(filtered_schedule)} iteration changes"
        )

        # Note: We do NOT pre-record history entries for scheduled iterations.
        # History entries are created when scheduled changes actually take effect
        # (e.g., when weighted_random executes and records its selection).

    def _get_meta_evolution_forbidden_params(self) -> List[str]:
        """
        Return list of parameters meta-evolution cannot modify.

        These parameters are either system-managed or control meta-evolution itself,
        so allowing meta-evolution to modify them would create circular dependencies
        or system instability.
        """
        return [
            # System-managed parameters
            "num_iterations",
            "random_seed",

            # Initial configuration (immutable after iteration 1)
            "initial_agents",
            "agents_directory",
            "initial_strategies",
            "strategies_directory",

            # Dataset (changing mid-run would invalidate comparisons)
            "dataset",

            # Meta-evolution self-reference (circular dependency)
            "meta_evolution_strategy",
            "meta_evolution_model",
            "meta_evolution_budget",

            # Performance and system settings (user-controlled)
            "max_concurrent_dbs",
            "phase1_timeout",
            "sql_timeout",
            "evolution_timeout",
            "debug_log_probability"
        ]

    def get_config_for_validation(self, iteration: int) -> Dict[str, Any]:
        """
        Get configuration for validation WITHOUT executing side effects.

        This method is used during validation to check strategy references without
        triggering weighted random selection or other execution-time behaviors.

        Unlike get_config(), this method:
        - Does NOT execute weighted random selection
        - Does NOT cache results
        - Does NOT print selection messages
        - Returns config with weighted_random_configs pool intact

        Args:
            iteration: Iteration number

        Returns:
            Configuration dict for validation purposes
        """
        if iteration == 0:
            return self.get_defaults()

        if iteration == 1:
            defaults = self.get_defaults()
            user_config = self.iteration_configs.get(1, {})
            resolved = defaults.copy()
            resolved.update(user_config)
            return resolved

        # N >= 2: Inherit from N-1, apply deltas and schedules
        # But DON'T execute weighted random - just return pool for validation
        prev_config = self.get_config_for_validation(iteration - 1)
        config = prev_config.copy()

        # Apply explicit delta if exists
        if iteration in self.iteration_configs:
            config.update(self.iteration_configs[iteration])

        # Apply meta_config_schedule[N] if exists
        meta_schedule = config.get("meta_config_schedule", {})
        if str(iteration) in meta_schedule:
            meta_delta = meta_schedule[str(iteration)]
            config.update(meta_delta)

        # Apply config_schedule[N] if exists
        schedule = config.get("config_schedule", {})
        if str(iteration) in schedule:
            schedule_entry = schedule[str(iteration)]
            config.update(schedule_entry)

        # NOTE: We do NOT execute weighted random here
        # The config still contains weighted_random_configs pool for validation
        return config

    def set_current_iteration(self, iteration: int) -> None:
        """
        Set the current iteration being executed.

        This is used to control caching behavior - we only cache configs
        for iterations that are being executed or have been completed.
        This prevents pollution of checkpoint.json with future iteration data.

        Args:
            iteration: Current iteration number
        """
        self.current_iteration = iteration

    def get_config(self, iteration: int) -> Dict[str, Any]:
        """
        Get resolved configuration for iteration.

        Uses caching to avoid recomputing, but ONLY caches iterations that are
        being executed or have been completed (iteration <= current_iteration).

        This ensures lazy evaluation - future iterations are never materialized
        into iteration_configs or resolved_configs during validation.

        Args:
            iteration: Iteration number

        Returns:
            Resolved configuration dict (copy, so caller can modify)
        """
        if iteration in self.resolved_configs:
            return self.resolved_configs[iteration].copy()

        # Resolve config
        resolved = self._resolve_config(iteration)

        # Only cache if this iteration is being executed or was completed
        # Never cache future iterations (prevents validation pollution)
        if iteration <= self.current_iteration:
            self.resolved_configs[iteration] = resolved

        return resolved.copy()

    def clear_from_iteration(self, iteration: int) -> None:
        """
        Clear config state for iterations >= iteration.

        Used when --from-iteration restarts from a prior iteration.
        Removes deltas, resolved configs, and history entries for iterations
        that are being re-executed.

        Special handling: Preserves meta_config_schedule at the restart point
        (iteration N), as it was set by iteration N-1's meta-evolution and
        represents prospective changes.

        Args:
            iteration: Starting iteration to clear from (inclusive)
        """
        # Clear iteration_configs with special handling for restart point
        for i in list(self.iteration_configs.keys()):
            if i > iteration:
                # Iterations after restart point: delete entirely
                del self.iteration_configs[i]
            elif i == iteration:
                # Restart point: preserve only meta_config_schedule
                # (set by prior iteration's meta-evolution)
                config = self.iteration_configs[i]
                if "meta_config_schedule" in config:
                    self.iteration_configs[i] = {"meta_config_schedule": config["meta_config_schedule"]}
                else:
                    del self.iteration_configs[i]

        # Clear resolved_configs
        to_remove = [i for i in self.resolved_configs if i >= iteration]
        for i in to_remove:
            del self.resolved_configs[i]

        # Clear config_change_history
        self.config_change_history = [
            entry for entry in self.config_change_history
            if entry["iteration"] < iteration
        ]

    def validate_consistency(self, iteration: int) -> Tuple[bool, List[str]]:
        """
        Validate consistency between iteration_configs, resolved_configs,
        and config_change_history.

        Four-level validation:
        1. Structure completeness (deltas have resolved configs)
        2. Delta chain integrity (recompute and verify)
        3. Change history consistency (deltas match history)
        4. Immutable parameter protection

        Args:
            iteration: Current iteration to validate up to

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # 1. Structure completeness
        for iter_num in self.iteration_configs:
            # Only validate iterations that should have run (future scheduled iterations are OK)
            if iter_num <= iteration and iter_num not in self.resolved_configs:
                errors.append(
                    f"Iteration {iter_num} has delta but no resolved config"
                )

        # 2. Delta chain integrity (recompute and verify)
        # Parameters that have been removed from the system but may exist in old checkpoints
        deprecated_params = {"evolution_analyzer"}

        for iter_num in range(iteration + 1):
            expected = self._resolve_config(iter_num)
            actual = self.resolved_configs.get(iter_num, {})

            # Filter out deprecated parameters from both configs before comparison
            expected_filtered = {k: v for k, v in expected.items() if k not in deprecated_params}
            actual_filtered = {k: v for k, v in actual.items() if k not in deprecated_params}

            if expected_filtered != actual_filtered:
                errors.append(
                    f"Iteration {iter_num} resolved config mismatch "
                    f"(recomputed != cached)"
                )

        # 3. Change history consistency
        # Note: history_iterations can be a superset of delta_iterations when
        # meta-evolution schedules future changes. We only require that every
        # iteration with a delta has a corresponding history entry.
        history_iterations = {
            entry["iteration"]
            for entry in self.config_change_history
        }
        delta_iterations = set(self.iteration_configs.keys()) - {0}
        if not delta_iterations.issubset(history_iterations):
            missing = delta_iterations - history_iterations
            errors.append(
                f"Change history missing entries for iterations with deltas: "
                f"{sorted(missing)}"
            )

        # 4. Immutable parameter protection
        immutable = ["dataset", "random_seed", "initial_agents", "agents_directory"]
        if 1 in self.resolved_configs:
            iter1_config = self.resolved_configs[1]
            for iter_num in range(2, iteration + 1):
                if iter_num in self.resolved_configs:
                    config = self.resolved_configs[iter_num]
                    for param in immutable:
                        if config.get(param) != iter1_config.get(param):
                            errors.append(
                                f"Iteration {iter_num}: immutable param "
                                f"'{param}' changed from "
                                f"'{iter1_config.get(param)}' to '{config.get(param)}'"
                            )

        return (len(errors) == 0, errors)

    def to_checkpoint(self) -> Dict[str, Any]:
        """
        Serialize to checkpoint format.

        Returns:
            Dict with iteration_configs, resolved_configs, config_change_history, current_iteration
        """
        return {
            "iteration_configs": self.iteration_configs,
            "resolved_configs": self.resolved_configs,
            "config_change_history": self.config_change_history,
            "current_iteration": self.current_iteration
        }

    @classmethod
    def from_checkpoint(cls, data: Dict[str, Any]) -> 'ConfigManager':
        """
        Deserialize from checkpoint.

        Args:
            data: Checkpoint data dict

        Returns:
            ConfigManager instance
        """
        manager = cls()

        # Convert string keys to integers (JSON serialization converts int keys to strings)
        manager.iteration_configs = {
            int(k): v for k, v in data["iteration_configs"].items()
        }
        manager.resolved_configs = {
            int(k): v for k, v in data["resolved_configs"].items()
        }
        manager.config_change_history = data["config_change_history"]

        # Load current_iteration (defaults to 0 for old checkpoints)
        manager.current_iteration = data.get("current_iteration", 0)

        return manager

    # Private methods

    def _resolve_config(self, iteration: int) -> Dict[str, Any]:
        """
        Resolve configuration for iteration N from iteration N-1 + deltas.

        Resolution algorithm:
        - Iteration 0: Return defaults
        - Iteration 1: defaults + user overrides
        - Iteration N >= 2:
          1. Start with N-1 resolved config (inheritance)
          2. Apply explicit delta (updates ALL parameters uniformly)
          3. Apply meta_config_schedule[N] entries (can overwrite delta params)
          4. Apply config_schedule[N] entries (final precedence)
          5. If use_weighted_random is true, select from weighted_random_configs

        Note: All parameters (including schedules and weighted random configs) are
        treated uniformly - no special casing. Schedules are just parameters that
        happen to contain entries for specific iterations.

        Args:
            iteration: Iteration number

        Returns:
            Resolved configuration dict
        """
        if iteration == 0:
            return self.get_defaults()

        if iteration == 1:
            defaults = self.get_defaults()
            user_config = self.iteration_configs.get(1, {})
            resolved = defaults.copy()
            resolved.update(user_config)
            return resolved

        # N >= 2: Inherit from N-1, apply deltas, check schedules
        prev_config = self.get_config(iteration - 1)
        config = prev_config.copy()

        # Step 1: Apply explicit delta (updates ALL parameters uniformly)
        if iteration in self.iteration_configs:
            config.update(self.iteration_configs[iteration])

        # Step 2: Apply meta_config_schedule[N] if exists (can overwrite delta params)
        meta_schedule = config.get("meta_config_schedule", {})
        if str(iteration) in meta_schedule:
            meta_delta = meta_schedule[str(iteration)]
            config.update(meta_delta)
            # Note: Recording in history happens in integrate_meta_config_schedule()

        # Step 3: Apply config_schedule[N] if exists (final precedence)
        schedule = config.get("config_schedule", {})
        if str(iteration) in schedule:
            schedule_entry = schedule[str(iteration)]
            config.update(schedule_entry)

        # Step 4: Handle weighted random if enabled (after all schedule processing)
        if config.get("use_weighted_random", False):
            pool = config.get("weighted_random_configs", [])

            # Check history for prior selection (idempotency)
            prior_selection = next(
                (h for h in self.config_change_history
                 if h["iteration"] == iteration and h["source"] == "weighted_random"),
                None
            )

            if prior_selection:
                # Apply historical selection (idempotency - same result on recomputation)
                config.update(prior_selection["delta"])
            elif pool:
                # Make new selection
                selected = self._select_from_weighted_pool(pool)
                config.update(selected)

                # Find the weight of the selected config via exact match
                selected_weight = None
                for config_dict, weight in pool:
                    if config_dict == selected:
                        selected_weight = weight
                        break

                # Log the weighted random selection
                print(f"🎲 Weighted random selection for iteration {iteration}:")
                print(f"   Selected: {selected}")
                print(f"   Probability: {selected_weight}%")
                print(f"   Pool size: {len(pool)} configs")

                # Record this selection in history
                # History provides idempotency - we don't store in iteration_configs
                # because that would cause the result to apply at Step 1 instead of Step 4
                self.config_change_history.append({
                    "iteration": iteration,
                    "source": "weighted_random",
                    "delta": selected,
                    "weight": selected_weight,
                    "pool_size": len(pool),
                    "rationale": f"Weighted random selection from pool of {len(pool)} configs (selected with {selected_weight}% probability)"
                })

        return config

    def _select_from_weighted_pool(self,
                                   pool: List[Tuple[Dict[str, Any], int]]) -> Dict[str, Any]:
        """
        Select configuration from weighted random pool.

        Args:
            pool: List of [config_dict, weight] tuples

        Returns:
            Selected configuration dict

        Raises:
            ValueError: If weights don't sum to 100 or pool is invalid
        """
        if not pool:
            raise ValueError("Weighted random pool is empty")

        # Validate pool format
        for entry in pool:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise ValueError(
                    f"Invalid weighted random entry: {entry}. "
                    f"Expected [config_dict, weight]"
                )

        # Validate weights sum to 100
        total_weight = sum(weight for _, weight in pool)
        if total_weight != 100:
            raise ValueError(
                f"Weighted random percentages must sum to 100%, got {total_weight}%"
            )

        # Select based on weights
        rand = random.random() * 100
        cumulative = 0

        for config_dict, weight in pool:
            cumulative += weight
            if rand < cumulative:
                return config_dict.copy()

        # Fallback (shouldn't reach here due to weights summing to 100)
        return pool[-1][0].copy()

    def _validate_parameters(self,
                            config: Dict[str, Any],
                            context: str,
                            source: ConfigSource) -> None:
        """
        Validate that all parameters in config are known.

        Args:
            config: Configuration dict to validate
            context: Description for error messages (e.g., "Initial configuration")
            source: Source of configuration (CLI, META_EVOLUTION, etc.)

        Raises:
            ValueError: If config contains unknown parameters

        Note:
            The 'comment' field is allowed in any config for documentation purposes.
            It is preserved in checkpoint and config history but has no effect on execution.
        """
        # Check for common mistakes first (provide helpful error messages)
        # Only block meta_config_schedule if it's NOT from meta-evolution
        if 'meta_config_schedule' in config and source != ConfigSource.META_EVOLUTION:
            raise ValueError(
                f"{context} uses 'meta_config_schedule' which is only for meta-evolution.\n"
                f"Did you mean to use 'config_schedule' instead?\n\n"
                f"Note:\n"
                f"  - 'config_schedule': Schedule parameter changes per iteration (user-controlled)\n"
                f"  - 'meta_config_schedule': Schedule parameter changes per iteration (meta-evolution-controlled)\n\n"
                f"Both can schedule ANY parameter changes, but 'meta_config_schedule' is managed by\n"
                f"meta-evolution and should not be set manually in config files.\n\n"
                f"To fix: Replace 'meta_config_schedule' with 'config_schedule' in your config file."
            )

        defaults = self.get_defaults()
        # Allow 'comment' field for documentation (preserved but has no effect)
        config_params = set(config.keys()) - {'comment'}
        unknown_params = config_params - set(defaults.keys())

        if unknown_params:
            raise ValueError(
                f"{context} contains unknown parameters: {sorted(unknown_params)}\n"
                f"Valid parameters are: {sorted(defaults.keys())}"
            )

    def _validate_immutable_params(self,
                                   iteration: int,
                                   delta: Dict[str, Any]) -> None:
        """
        Validate immutable parameters not changed after iteration 1.

        Immutable = user-settable at iter 1, cannot change after.
        (Excludes system-managed params which have separate handling)

        Args:
            iteration: Iteration number
            delta: Delta to validate

        Raises:
            ValueError: If trying to modify immutable parameter after iteration 1
        """
        immutable = ["dataset", "initial_agents", "agents_directory"]

        if iteration > 1:
            for param in immutable:
                if param in delta:
                    raise ValueError(
                        f"Cannot modify immutable parameter '{param}' "
                        f"after iteration 1 (attempted at iteration {iteration})"
                    )
