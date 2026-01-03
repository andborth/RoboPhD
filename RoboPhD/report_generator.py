"""
Report generation for RoboPhD research system.

Handles creation of final and interim reports, including:
- Experiment configuration summaries
- Agent performance metrics and ELO rankings
- Evolution strategy tracking
- Cost analysis and timing
- Cache performance statistics
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from .config import SUPPORTED_MODELS

if TYPE_CHECKING:
    from .researcher import ParallelAgentResearcher


class ReportGenerator:
    """Generates comprehensive reports for RoboPhD research runs."""

    def __init__(self, researcher: 'ParallelAgentResearcher'):
        """
        Initialize report generator.

        Args:
            researcher: ParallelAgentResearcher instance to generate reports for
        """
        self.researcher = researcher

    def _format_model_name(self, model_short_name: str) -> str:
        """
        Format model name to include both short name and full version.

        Args:
            model_short_name: Short model name (e.g., 'sonnet-4.5', 'haiku-4.5')

        Returns:
            Formatted string with both names (e.g., 'sonnet-4.5 (claude-sonnet-4-5-20250929)')
        """
        if model_short_name in SUPPORTED_MODELS:
            full_name = SUPPORTED_MODELS[model_short_name]['name']
            return f"{model_short_name} ({full_name})"
        return model_short_name

    def _categorize_parameters(self):
        """
        Categorize configuration parameters for display in reports.

        Returns:
            Dict mapping category names to lists of parameter names
        """
        return {
            "Dataset & Sampling": [
                "dataset",
                "databases_per_iteration",
                "questions_per_database",
                "agents_per_iteration"
            ],
            "Models": [
                "eval_model",
                "analysis_model",
                "evolution_model",
                "meta_evolution_model"
            ],
            "Evolution": [
                "evolution_strategy",
                "new_agent_test_rounds"
            ],
            "Meta-Evolution": [
                "meta_evolution_strategy",
                "meta_evolution_budget"
            ],
            "SQL Generation": [
                "verification_retries",
                "temperature_strategy"
            ],
            "Performance": [
                "max_concurrent_dbs"
            ],
            "Timeouts": [
                "phase1_timeout",
                "sql_timeout",
                "evolution_timeout"
            ],
            "Other": [
                "debug_log_probability"
            ],
            "Immutable": [
                "initial_agents",
                "agents_directory",
                "initial_strategies",
                "strategies_directory"
            ]
        }

    def generate_interim_report(self, start_time: float, iteration: int):
        """Generate interim report after each iteration."""
        try:
            self._generate_report(start_time, report_type='interim', iteration=iteration)

            # Verify the file was actually written
            iteration_dir = self.researcher.experiment_dir / f"iteration_{iteration:03d}"
            report_file = iteration_dir / 'interim_report.md'

            if report_file.exists() and report_file.stat().st_size > 0:
                print(f"  📊 Generated interim report: {report_file.resolve().relative_to(Path.cwd())}")
            else:
                print(f"  ⚠️ Warning: Interim report file not created or is empty: {report_file}")
        except Exception as e:
            print(f"  ❌ Error generating interim report for iteration {iteration}: {str(e)}")
            # Log the full traceback for debugging
            import traceback
            traceback.print_exc()

    def generate_final_report(self, start_time: float):
        """Generate final comprehensive report."""
        self._generate_report(start_time, report_type='final')

    def _generate_elo_leadership_section(self) -> str:
        """Generate ELO leadership progression section for reports."""
        lines = []
        lines.append("## ELO Leadership Progression\n")

        if not self.researcher.test_history or len(self.researcher.test_history) == 0:
            lines.append("No test history available yet.")
            return '\n'.join(lines)

        # Use the researcher's ELO progression calculation method
        # This ensures we use the same ELO calculation logic as the actual system
        elo_leaders = self.researcher._calculate_elo_progression()

        # Display leadership changes
        if elo_leaders:
            lines.append("Iteration-by-iteration ELO leaders:\n")
            lines.append("| Iteration | Leader | ELO | Evolution Strategy |")
            lines.append("|-----------|--------|-----|-------------------|")

            for entry in elo_leaders:
                # Get evolution strategy from agent pool
                leader = entry['leader']
                evolution_strategy = self.researcher._get_agent_evolution_strategy(leader)

                lines.append(f"| {entry['iteration']} | {leader} | {entry['elo']:.0f} | {evolution_strategy} |")

        lines.append("")
        return '\n'.join(lines)

    def _generate_agent_evaluation_report(self, agent_id: str, iteration: int, databases: list,
                                         agent_time: float, agent_cost: float) -> None:
        """
        Generate an evaluation report for a specific agent in a specific iteration.

        Args:
            agent_id: ID of the agent
            iteration: Iteration number
            databases: List of databases tested
            agent_time: Estimated time spent on this agent (in seconds)
            agent_cost: Estimated cost for this agent
        """
        from datetime import datetime
        import json

        # Agent directory path
        agent_dir = self.researcher.experiment_dir / f"iteration_{iteration:03d}" / f"agent_{agent_id}"
        if not agent_dir.exists():
            print(f"  ⚠️  Agent directory not found: {agent_dir}")
            return

        report_path = agent_dir / "agent_evaluation_report.md"

        # Collect database results
        database_results = []
        total_questions = 0
        total_correct = 0
        total_sql_errors = 0
        warnings = []

        for db_name in databases:
            db_results_file = agent_dir / db_name / "results" / "evaluation.json"
            if db_results_file.exists():
                try:
                    with open(db_results_file, 'r') as f:
                        db_data = json.load(f)

                    # Extract metrics
                    questions = db_data.get('total_questions', 0)
                    correct = db_data.get('correct', 0)
                    accuracy = (correct / questions * 100) if questions > 0 else 0
                    sql_errors = db_data.get('prediction_errors', 0)

                    database_results.append({
                        'database': db_name,
                        'accuracy': accuracy,
                        'correct': correct,
                        'total': questions,
                        'sql_errors': sql_errors
                    })

                    total_questions += questions
                    total_correct += correct
                    total_sql_errors += sql_errors

                    # Check for unusual errors and generate warnings
                    pred_timeouts = db_data.get('prediction_timeouts', 0)
                    gt_errors = db_data.get('ground_truth_errors', 0)
                    gt_timeouts = db_data.get('ground_truth_timeouts', 0)

                    if pred_timeouts > 0:
                        warnings.append(f"Prediction timeouts in {db_name}: {pred_timeouts} questions")
                    if gt_errors > 0:
                        warnings.append(f"Ground truth errors in {db_name}: {gt_errors} questions")
                    if gt_timeouts > 0:
                        warnings.append(f"Ground truth timeouts in {db_name}: {gt_timeouts} questions")

                except Exception as e:
                    print(f"  ⚠️  Error reading {db_results_file}: {e}")
                    continue
            else:
                print(f"  ⚠️  Results file not found: {db_results_file}")

        # Calculate overall accuracy
        overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0

        # Determine agent source
        agent_info = self.researcher.agent_pool.get(agent_id, {})
        agent_source = agent_info.get('source', 'unknown')
        created_iteration = agent_info.get('created_iteration', 'N/A')

        if agent_source == 'initial':
            source_description = "initial"
            agent_file = f"RoboPhD/agents/{agent_id}.md"
        elif agent_source == 'evolution':
            source_description = f"evolved in iteration {created_iteration}"
            agent_file = f"agents/{agent_id}.md"
        else:
            source_description = agent_source
            agent_file = "unknown"

        # Generate report content
        report_lines = ["# Agent Evaluation Report\n"]

        # Configuration section
        report_lines.append("## Configuration")
        report_lines.append(f"- **Agent**: {agent_id}")
        report_lines.append(f"- **Iteration**: {iteration}")
        report_lines.append(f"- **Dataset**: {self.researcher.dataset}")
        report_lines.append(f"- **Evaluation model**: {self._format_model_name(self.researcher.eval_model)}")
        report_lines.append(f"- **Analysis model**: {self._format_model_name(self.researcher.analysis_model)}")
        report_lines.append(f"- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"- **Databases tested**: {len(databases)}")
        report_lines.append(f"- **Questions per database**: {self.researcher.questions_per_database}")
        report_lines.append(f"- **Total time**: {agent_time/60:.1f} minutes")
        report_lines.append(f"- **Total cost**: ${agent_cost:.2f}")

        # Overall results
        report_lines.append("\n## Overall Results")
        report_lines.append(f"- **Total questions**: {total_questions}")
        report_lines.append(f"- **Correct answers**: {total_correct}")
        report_lines.append(f"- **Overall accuracy**: {overall_accuracy:.2f}%")

        # Warnings section (if any)
        if warnings:
            report_lines.append("\n⚠️  **WARNINGS**")
            for warning in warnings:
                report_lines.append(f"- {warning}")

        # Per-database results table
        report_lines.append("\n## Per-Database Results")
        report_lines.append("| Database | Accuracy | Correct/Total | SQL Errors |")
        report_lines.append("|----------|----------|---------------|------------|")

        for result in database_results:
            report_lines.append(
                f"| {result['database']} | {result['accuracy']:.2f}% | "
                f"{result['correct']}/{result['total']} | {result['sql_errors']} |"
            )

        # Agent details
        report_lines.append("\n## Agent Details")
        report_lines.append(f"- **Source**: {source_description}")
        report_lines.append(f"- **Agent file**: {agent_file}")

        # Write report
        try:
            with open(report_path, 'w') as f:
                f.write('\n'.join(report_lines))
            print(f"  📋 Generated evaluation report: {report_path.resolve().relative_to(Path.cwd())}")
        except Exception as e:
            print(f"  ❌ Error writing evaluation report: {e}")

    def _generate_report(self, start_time: float, report_type: str = 'final', iteration: int = None):
        """
        Generate comprehensive report (final or interim).

        Args:
            start_time: Experiment start time
            report_type: 'final' or 'interim'
            iteration: Current iteration (for interim reports)
        """
        if report_type == 'final':
            report_lines = ["# Parallel Agent Research - Final Report\n"]
        else:
            report_lines = [f"# Parallel Agent Research - Interim Report (Iteration {iteration})\n"]

        # Configuration
        report_lines.append("## Experiment Configuration")
        report_lines.append("")
        report_lines.append(f"**Run ID**: {self.researcher.experiment_dir.name}")
        report_lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if report_type == 'final':
            report_lines.append(f"**Iterations**: {self.researcher.num_iterations}")
        else:
            report_lines.append(f"**Iterations**: {iteration}/{self.researcher.num_iterations}")

        # Initial Configuration (Iteration 1) - condensed with configurability labels
        report_lines.append("")
        report_lines.append("### Initial Configuration (Iteration 1)")
        report_lines.append("")

        config_iter1 = self.researcher.config_manager.get_config(1)

        # Get forbidden parameters (not configurable by meta-evolution)
        forbidden = set(self.researcher.config_manager._get_meta_evolution_forbidden_params())

        # Meta-parameters to filter out (implementation details)
        meta_params = {"config_schedule", "meta_config_schedule", "weighted_random_configs", "meta_evolution_strategy"}

        # Group configurable vs non-configurable
        configurable_params = []
        non_configurable_params = []

        categories = self._categorize_parameters()
        for category_name, param_names in categories.items():
            for param_name in param_names:
                if param_name in meta_params:
                    continue  # Skip meta-parameters
                value = config_iter1.get(param_name)
                # Format model names specially
                if param_name.endswith('_model') and value:
                    formatted_value = self._format_model_name(value)
                else:
                    formatted_value = value

                if param_name in forbidden:
                    non_configurable_params.append((param_name, formatted_value))
                else:
                    configurable_params.append((param_name, formatted_value))

        # Display configurable parameters (meta-evolution can modify)
        report_lines.append("#### Configurable by Meta-Evolution")
        for param_name, value in sorted(configurable_params):
            report_lines.append(f"- {param_name}: {value}")
        report_lines.append("")

        # Display non-configurable parameters (user-controlled / system-managed)
        report_lines.append("#### System Settings (User-Controlled)")
        system_settings = [(p, v) for p, v in non_configurable_params
                          if p in {"max_concurrent_dbs", "phase1_timeout", "sql_timeout",
                                  "evolution_timeout", "debug_log_probability"}]
        for param_name, value in sorted(system_settings):
            report_lines.append(f"- {param_name}: {value}")
        report_lines.append("")

        # Display immutable parameters
        report_lines.append("#### Immutable (Set Once at Start)")
        immutable = [(p, v) for p, v in non_configurable_params
                    if p in {"dataset", "initial_agents", "agents_directory",
                            "initial_strategies", "strategies_directory"}]
        for param_name, value in sorted(immutable):
            report_lines.append(f"- {param_name}: {value}")
        report_lines.append("")

        # Display meta-evolution self-reference (shown for completeness)
        report_lines.append("#### Meta-Evolution Settings (Self-Reference)")
        meta_evo = [(p, v) for p, v in non_configurable_params
                   if p in {"meta_evolution_strategy", "meta_evolution_model", "meta_evolution_budget"}]
        for param_name, value in sorted(meta_evo):
            report_lines.append(f"- {param_name}: {value}")
        report_lines.append("")

        # Configuration Changes by Iteration
        report_lines.append("### Configuration Changes by Iteration")
        report_lines.append("")

        # Determine how many iterations to show (only completed iterations)
        if report_type == 'final':
            # Use test_history length to get last completed iteration
            max_iter = len(self.researcher.test_history)
        else:
            max_iter = iteration

        # Track if we've seen any changes
        has_changes = False

        for iter_num in range(2, max_iter + 1):
            prev_config = self.researcher.config_manager.get_config(iter_num - 1)
            curr_config = self.researcher.config_manager.get_config(iter_num)

            # Compute delta (excluding meta-parameters)
            changes = {}
            for key in curr_config:
                if key in meta_params:
                    continue
                if curr_config[key] != prev_config.get(key):
                    changes[key] = curr_config[key]

            if changes:
                has_changes = True
                # Format the changes
                change_strs = []
                for param, value in sorted(changes.items()):
                    # Format model names specially
                    if param.endswith('_model') and value:
                        value = self._format_model_name(value)
                    change_strs.append(f"{param}={value}")
                report_lines.append(f"- **Iteration {iter_num}**: {', '.join(change_strs)}")

        if not has_changes:
            report_lines.append("(No configuration changes across iterations)")
        report_lines.append("")

        # Timing and costs - calculate from iteration times + evolution times
        if self.researcher.iteration_times:
            # Calculate total runtime including evolution time
            total_test_time = sum(self.researcher.iteration_times)
            total_evolution_time = sum(
                t.get('total', 0) if t is not None else 0
                for t in self.researcher.evolution_times
            )
            total_runtime = total_test_time + total_evolution_time
            elapsed_str = str(timedelta(seconds=int(total_runtime)))
        else:
            # Fallback to start_time if no iteration times yet
            elapsed = (datetime.now().timestamp() - start_time)
            elapsed_str = str(timedelta(seconds=int(elapsed)))

        # Evolution retries
        if hasattr(self.researcher.evolver, 'evolution_retries') and self.researcher.evolver.evolution_retries:
            report_lines.append("")
            report_lines.append("### Evolution Retries")
            report_lines.append(f"Total retries: {len(self.researcher.evolver.evolution_retries)}")
            for retry in self.researcher.evolver.evolution_retries:
                reason = retry.get('reason', 'unknown')
                success = "✅ Success" if retry.get('success') else "❌ Failed"
                report_lines.append(f"- Iteration {retry['iteration']}: {reason} - {success}")

        # 5-hour limit incidents
        if self.researcher.five_hour_limit_incidents:
            report_lines.append("")
            report_lines.append("### 5-Hour Limit Incidents")
            report_lines.append(f"Total incidents: {len(self.researcher.five_hour_limit_incidents)}")
            report_lines.append("\n| Iteration | Recovery Action | Phase 1 Failures |")
            report_lines.append("|-----------|----------------|------------------|")
            for incident in self.researcher.five_hour_limit_incidents:
                recovery = incident.get('recovery_action', 'Unknown')
                phase1_info = f"{incident['phase1_failures_detected']} (iter {incident['iteration']-1})" if incident['phase1_failures_detected'] > 0 else "0"
                report_lines.append(f"| {incident['iteration']} | {recovery} | {phase1_info} |")

        # Winning Agents by Evolution Strategy
        report_lines.append("\n## Winning Agents by Evolution Strategy\n")
        try:
            # Group agents by the evolution strategy that created them
            from collections import defaultdict
            strategy_winners = defaultdict(list)
            agent_wins = defaultdict(int)

            # Count wins for each agent
            for test_results in self.researcher.test_history:
                # Find winner(s) of this iteration
                if test_results:
                    max_accuracy = max(test_results[k]['accuracy'] for k in test_results.keys())
                    winners = [k for k in test_results.keys() if test_results[k]['accuracy'] == max_accuracy]

                    for winner in winners:
                        agent_wins[winner] += 1

            # Group agents by strategy
            for agent_id, win_count in agent_wins.items():
                if win_count > 0:
                    agent_info = self.researcher.agent_pool.get(agent_id, {})
                    strategy = agent_info.get('evolution_strategy')
                    if strategy is None:
                        strategy = 'initial' if agent_info.get('source') == 'initial' else 'unknown'

                    if agent_id not in [a for a, _ in strategy_winners[strategy]]:
                        strategy_winners[strategy].append((agent_id, win_count))

            # Calculate statistics
            total_wins = sum(agent_wins.values())

            # Sort strategies by total wins
            strategy_totals = {}
            for strategy, agents in strategy_winners.items():
                total = sum(wins for _, wins in agents)
                strategy_totals[strategy] = total

            sorted_strategies = sorted(strategy_totals.items(), key=lambda x: x[1], reverse=True)

            # Display results
            for strategy, win_count in sorted_strategies:
                if win_count > 0:
                    percentage = (win_count / total_wins * 100) if total_wins > 0 else 0
                    report_lines.append(f"#### **{strategy.replace('_', ' ').title()}** ({win_count} wins - {percentage:.1f}% of all wins)")

                    # Sort agents by number of wins
                    agents = sorted(strategy_winners[strategy], key=lambda x: x[1], reverse=True)

                    for agent, wins in agents:
                        agent_info = self.researcher.agent_pool.get(agent, {})
                        created_iter = agent_info.get('created_iteration', 'unknown')
                        report_lines.append(f"- **{agent}** ({wins} {'win' if wins == 1 else 'wins'}, created iteration {created_iter})")
                    report_lines.append("")

            # Summary statistics
            report_lines.append("#### Summary Statistics")
            report_lines.append(f"- **Total iterations with winners**: {len(self.researcher.test_history)}")
            report_lines.append(f"- **Total wins counted**: {total_wins} (includes ties)")
            report_lines.append(f"- **Unique winning agents**: {len(agent_wins)}")
            report_lines.append(f"- **Evolution strategies that produced winners**: {len([s for s, t in strategy_totals.items() if t > 0])}")
            report_lines.append("")
        except Exception as e:
            report_lines.append("### Winning Agents by Evolution Strategy\n")
            report_lines.append(f"⚠️ Error generating strategy analysis: {str(e)}\n")

        # Add comprehensive ranking table
        if self.researcher.test_history and len(self.researcher.test_history) > 0:
            ranking_table = self.researcher.evolver._generate_ranking_table(
                self.researcher.test_history,
                self.researcher.performance_records,
                for_evolution=False
            )
            report_lines.append(ranking_table)
            report_lines.append("\n")

        # Sort agents by ELO
        sorted_agents = sorted(self.researcher.performance_records.keys(),
                             key=lambda a: self.researcher.performance_records[a]['elo'],
                             reverse=True)

        # Simple summary table
        report_lines.append("### Quick Summary\n")
        report_lines.append("| Agent | ELO | Mean Accuracy | Tests |")
        report_lines.append("|-------|-----|---------------|-------|")

        for agent_id in sorted_agents:
            perf = self.researcher.performance_records[agent_id]
            report_lines.append(f"| {agent_id} | {perf['elo']:.0f} | "
                              f"{perf['mean_accuracy']:.1f}% | {perf['test_count']} |")

        # Iteration progression table
        report_lines.append("\n### Iteration Progression\n")
        report_lines.append("| Iter | Winner(s) | Accuracy | Evo Time | Test Time | Total Time |")
        report_lines.append("|------|-----------|----------|----------|-----------|------------|")

        for i, test_results in enumerate(self.researcher.test_history):
            iteration_num = i + 1

            # Find winner(s)
            if test_results:
                max_accuracy = max(test_results[k]['accuracy'] for k in test_results.keys())
                winners = [k for k in test_results.keys() if test_results[k]['accuracy'] == max_accuracy]
                winner_str = ', '.join(winners) if len(winners) <= 2 else f"{winners[0]} +{len(winners)-1}"
                accuracy_str = f"{max_accuracy:.1f}%"
            else:
                winner_str = "N/A"
                accuracy_str = "N/A"

            # Get timing data
            test_time = self.researcher.iteration_times[i] if i < len(self.researcher.iteration_times) else 0
            test_time_str = f"{test_time/60:.1f}m"

            # Get evolution timing
            if i < len(self.researcher.evolution_times) and self.researcher.evolution_times[i] is not None:
                evo_timing = self.researcher.evolution_times[i]
                evo_time = evo_timing.get('total', 0)
                evo_time_str = f"{evo_time/60:.1f}m"
                total_time = evo_time + test_time
                total_time_str = f"{total_time/60:.1f}m"
            else:
                evo_time_str = "-"
                total_time_str = test_time_str

            report_lines.append(f"| {iteration_num} | {winner_str} | {accuracy_str} | "
                              f"{evo_time_str} | {test_time_str} | {total_time_str} |")

        # Best agent
        if sorted_agents:
            best_agent = sorted_agents[0]
            report_lines.append(f"\n## Best Agent: {best_agent}")
            report_lines.append(f"- ELO Score: {self.researcher.performance_records[best_agent]['elo']:.0f}")
            report_lines.append(f"- Mean Accuracy: {self.researcher.performance_records[best_agent]['mean_accuracy']:.1f}%")

        # ELO Leadership Progression
        report_lines.append("\n")
        elo_leadership_section = self._generate_elo_leadership_section()
        report_lines.append(elo_leadership_section)

        # Phase 1 Cache Performance section
        report_lines.append("\n## Phase 1 Cache Performance")
        cache_stats = self.researcher.cache_manager.get_cache_stats()
        if cache_stats['total'] > 0:
            report_lines.append(f"- **Total Phase 1 calls**: {cache_stats['total']}")
            report_lines.append(f"- **Cache hits**: {cache_stats['hits']} ({cache_stats['hit_rate']:.1f}%)")
            report_lines.append(f"- **Cache misses**: {cache_stats['misses']}")
            report_lines.append(f"- **Estimated savings**: ${cache_stats['estimated_savings']:.2f}")
        else:
            report_lines.append("No Phase 1 analyses performed in this run")

        # Phase 2 Cache Performance section
        report_lines.append("\n## Phase 2 Cache Performance")
        # Calculate stats from aggregate counters
        phase2_hits = self.researcher.phase2_cache_hits
        phase2_misses = self.researcher.phase2_cache_misses
        phase2_total = phase2_hits + phase2_misses
        if phase2_total > 0:
            phase2_hit_rate = (phase2_hits / phase2_total * 100)
            phase2_savings = phase2_hits * 0.01  # ~$0.01 per cached call
            report_lines.append(f"- **Total Phase 2 SQL generation calls**: {phase2_total}")
            report_lines.append(f"- **Cache hits**: {phase2_hits} ({phase2_hit_rate:.1f}%)")
            report_lines.append(f"- **Cache misses**: {phase2_misses}")
            report_lines.append(f"- **Estimated savings**: ${phase2_savings:.2f}")
        else:
            report_lines.append("No Phase 2 SQL generation performed in this run")

        # Deep Focus Timing Analysis section
        report_lines.append("\n## Deep Focus Evolution Timing Analysis")

        # Filter out None values (iterations without evolution)
        evolution_timings = [t for t in self.researcher.evolution_times if t is not None]

        if evolution_timings:
            # Calculate total and average times
            total_evo_time = sum(t.get('total', 0) for t in evolution_timings)
            num_evolutions = len(evolution_timings)
            avg_evo_time = total_evo_time / num_evolutions if num_evolutions > 0 else 0

            # Calculate total test time for comparison
            total_test_time = sum(self.researcher.iteration_times)

            # Calculate phase averages for fixed phases
            avg_error_analyzer = sum(t.get('error_analyzer', 0) for t in evolution_timings) / num_evolutions
            avg_planning = sum(t.get('planning', 0) for t in evolution_timings) / num_evolutions
            avg_implementation = sum(t.get('implementation', 0) for t in evolution_timings) / num_evolutions

            # Dynamically calculate test round averages (excluding zeros for optional phases)
            test_round_averages = {}
            for i in range(1, 100):  # Check up to 100 test rounds (safety limit)
                key = f'test_refine_{i}'
                values = [t.get(key, 0) for t in evolution_timings if t.get(key, 0) > 0]
                if not values:
                    break  # No more test rounds found
                test_round_averages[key] = sum(values) / len(values)

            # Reflection average (excluding zeros for backward compatibility with old data)
            reflection_values = [t.get('reflection', 0) for t in evolution_timings if t.get('reflection', 0) > 0]
            avg_reflection = sum(reflection_values) / len(reflection_values) if reflection_values else 0

            # Recalculate total evolution time including all dynamic test rounds and reflection
            avg_evo_time = (avg_error_analyzer + avg_planning + avg_implementation +
                           sum(test_round_averages.values()) + avg_reflection)

            # Calculate percentages of total evolution time
            pct_error_analyzer = (avg_error_analyzer / avg_evo_time * 100) if avg_evo_time > 0 else 0
            pct_planning = (avg_planning / avg_evo_time * 100) if avg_evo_time > 0 else 0
            pct_implementation = (avg_implementation / avg_evo_time * 100) if avg_evo_time > 0 else 0
            pct_reflection = (avg_reflection / avg_evo_time * 100) if avg_evo_time > 0 else 0

            # Calculate percentages for each test round
            test_round_percentages = {}
            for key, avg_time in test_round_averages.items():
                test_round_percentages[key] = (avg_time / avg_evo_time * 100) if avg_evo_time > 0 else 0

            # Overall summary
            report_lines.append(f"- **Iterations with evolution**: {num_evolutions}/{len(self.researcher.evolution_times)}")
            report_lines.append(f"- **Total evolution time**: {total_evo_time/60:.1f} minutes")
            report_lines.append(f"- **Average evolution time**: {avg_evo_time/60:.1f} minutes")

            # Phase breakdown table
            report_lines.append("\n### Time Breakdown by Evolution Phase\n")
            report_lines.append("| Phase | Avg Time | % of Evolution |")
            report_lines.append("|-------|----------|----------------|")
            report_lines.append(f"| Error Analyzer | {avg_error_analyzer/60:.1f}m | {pct_error_analyzer:.1f}% |")
            report_lines.append(f"| Planning (Round 1) | {avg_planning/60:.1f}m | {pct_planning:.1f}% |")
            report_lines.append(f"| Implementation (Round 2) | {avg_implementation/60:.1f}m | {pct_implementation:.1f}% |")

            # Dynamic test round rows
            for i, key in enumerate(sorted(test_round_averages.keys()), start=1):
                avg_time = test_round_averages[key]
                pct = test_round_percentages[key]
                round_num = i + 2  # Rounds start at 3 (after planning=1, implementation=2)
                report_lines.append(f"| Test & Refine {i} (Round {round_num})* | {avg_time/60:.1f}m | {pct:.1f}% |")

            # Reflection row (always shown now that we track it)
            if avg_reflection > 0:
                report_lines.append(f"| Reflection | {avg_reflection/60:.1f}m | {pct_reflection:.1f}% |")

            report_lines.append(f"| **Total** | **{avg_evo_time/60:.1f}m** | **100.0%** |")

            # Footnote for test rounds (only if any exist)
            if test_round_averages:
                report_lines.append("\n*Average excludes iterations where test round was not used")

            # Evolution vs Test vs Meta-Evolution comparison
            report_lines.append("\n### Evolution vs Test vs Meta-Evolution Time\n")
            report_lines.append("| Metric | Time | Percentage | Avg/Iter | Avg/Occurrence |")
            report_lines.append("|--------|------|------------|----------|----------------|")

            # Calculate total meta-evolution time
            total_meta_evo_time = sum(self.researcher.meta_evolution_times)

            # Calculate total time including all three components
            total_time = total_evo_time + total_test_time + total_meta_evo_time
            evo_pct = (total_evo_time / total_time * 100) if total_time > 0 else 0
            test_pct = (total_test_time / total_time * 100) if total_time > 0 else 0
            meta_evo_pct = (total_meta_evo_time / total_time * 100) if total_time > 0 else 0

            # Calculate averages per iteration (includes all iterations, even when component didn't run)
            num_iterations = len(self.researcher.iteration_times)
            avg_evo_per_iter = total_evo_time / num_iterations if num_iterations > 0 else 0
            avg_test_per_iter = total_test_time / num_iterations if num_iterations > 0 else 0
            avg_meta_evo_per_iter = total_meta_evo_time / num_iterations if num_iterations > 0 else 0
            avg_total_per_iter = total_time / num_iterations if num_iterations > 0 else 0

            # Calculate averages per occurrence (excludes iterations where component didn't run)
            evo_occurrences = len(evolution_timings)  # Already filtered to non-None
            avg_evo_per_occurrence = total_evo_time / evo_occurrences if evo_occurrences > 0 else 0

            # Test always runs (no zeros)
            avg_test_per_occurrence = avg_test_per_iter

            # Meta-evolution occurrences (exclude zeros)
            meta_evo_occurrences = sum(1 for t in self.researcher.meta_evolution_times if t > 0)
            avg_meta_evo_per_occurrence = total_meta_evo_time / meta_evo_occurrences if meta_evo_occurrences > 0 else 0

            report_lines.append(f"| Total Evolution Time | {total_evo_time/60:.1f}m | {evo_pct:.1f}% | {avg_evo_per_iter/60:.1f}m | {avg_evo_per_occurrence/60:.1f}m |")
            report_lines.append(f"| Total Test Time | {total_test_time/60:.1f}m | {test_pct:.1f}% | {avg_test_per_iter/60:.1f}m | {avg_test_per_occurrence/60:.1f}m |")
            report_lines.append(f"| Total Meta-Evolution Time | {total_meta_evo_time/60:.1f}m | {meta_evo_pct:.1f}% | {avg_meta_evo_per_iter/60:.1f}m | {avg_meta_evo_per_occurrence/60:.1f}m |")
            report_lines.append(f"| **Total** | **{total_time/60:.1f}m** | **100.0%** | **{avg_total_per_iter/60:.1f}m** | **-** |")

            # Ratio summary
            evo_test_ratio = total_evo_time / total_test_time if total_test_time > 0 else 0
            report_lines.append(f"\n- **Evolution/Test ratio**: {evo_test_ratio:.2f}x")
            report_lines.append(f"  - For every minute of testing, {evo_test_ratio:.2f} minutes spent on evolution")
        else:
            report_lines.append("No Deep Focus evolution performed in this run")

        # Claude CLI Usage & Cost Analysis section
        report_lines.append("\n## Claude CLI Usage & Cost Analysis")

        if self.researcher.iteration_claude_costs:
            # Calculate totals
            total_phase1_cost = sum(ic.get('phase1_cost', 0.0) for ic in self.researcher.iteration_claude_costs)
            total_phase1_calls = sum(ic.get('phase1_calls', 0) for ic in self.researcher.iteration_claude_costs)
            total_phase1_tokens_in = sum(ic.get('phase1_tokens_in', 0) for ic in self.researcher.iteration_claude_costs)
            total_phase1_tokens_out = sum(ic.get('phase1_tokens_out', 0) for ic in self.researcher.iteration_claude_costs)
            total_phase1_cache_created = sum(ic.get('phase1_cache_created', 0) for ic in self.researcher.iteration_claude_costs)
            total_phase1_cache_read = sum(ic.get('phase1_cache_read', 0) for ic in self.researcher.iteration_claude_costs)

            total_evolution_cost = sum(ic.get('evolution_cost', 0.0) for ic in self.researcher.iteration_claude_costs)
            total_evolution_calls = sum(ic.get('evolution_calls', 0) for ic in self.researcher.iteration_claude_costs)
            total_evolution_tokens_in = sum(ic.get('evolution_tokens_in', 0) for ic in self.researcher.iteration_claude_costs)
            total_evolution_tokens_out = sum(ic.get('evolution_tokens_out', 0) for ic in self.researcher.iteration_claude_costs)
            total_evolution_cache_created = sum(ic.get('evolution_cache_created', 0) for ic in self.researcher.iteration_claude_costs)
            total_evolution_cache_read = sum(ic.get('evolution_cache_read', 0) for ic in self.researcher.iteration_claude_costs)

            total_meta_evolution_cost = sum(ic.get('meta_evolution_cost', 0.0) for ic in self.researcher.iteration_claude_costs)
            total_meta_evolution_calls = sum(ic.get('meta_evolution_calls', 0) for ic in self.researcher.iteration_claude_costs)
            total_meta_evolution_tokens_in = sum(ic.get('meta_evolution_tokens_in', 0) for ic in self.researcher.iteration_claude_costs)
            total_meta_evolution_tokens_out = sum(ic.get('meta_evolution_tokens_out', 0) for ic in self.researcher.iteration_claude_costs)

            # Calculate API costs (Phase 2 SQL generation)
            total_api_cost = sum(self.researcher.iteration_costs) if self.researcher.iteration_costs else 0.0

            total_cli_cost = total_phase1_cost + total_evolution_cost + total_meta_evolution_cost
            total_cli_calls = total_phase1_calls + total_evolution_calls + total_meta_evolution_calls
            total_cli_tokens_in = total_phase1_tokens_in + total_evolution_tokens_in + total_meta_evolution_tokens_in
            total_cli_tokens_out = total_phase1_tokens_out + total_evolution_tokens_out + total_meta_evolution_tokens_out

            # Grand total (CLI + API)
            grand_total_cost = total_cli_cost + total_api_cost

            # Overall summary
            report_lines.append(f"- **Total Cost (CLI + API)**: ${grand_total_cost:.2f}")
            report_lines.append(f"  - **Claude CLI Cost**: ${total_cli_cost:.2f}")
            report_lines.append(f"  - **API Cost**: ${total_api_cost:.2f}")
            report_lines.append(f"- **Total CLI Calls**: {total_cli_calls}")
            report_lines.append(f"- **Total CLI Input Tokens**: {total_cli_tokens_in:,}")
            report_lines.append(f"- **Total CLI Output Tokens**: {total_cli_tokens_out:,}")
            report_lines.append(f"- **Total Cache Created**: {total_phase1_cache_created + total_evolution_cache_created:,}")
            report_lines.append(f"- **Total Cache Read**: {total_phase1_cache_read + total_evolution_cache_read:,}")

            # Cost breakdown
            report_lines.append("\n### Cost Breakdown\n")
            report_lines.append("| Operation | Cost | % of Total | Calls | Tokens In | Tokens Out |")
            report_lines.append("|-----------|------|------------|-------|-----------|------------|")

            phase1_pct = (total_phase1_cost / grand_total_cost * 100) if grand_total_cost > 0 else 0
            api_pct = (total_api_cost / grand_total_cost * 100) if grand_total_cost > 0 else 0
            evolution_pct = (total_evolution_cost / grand_total_cost * 100) if grand_total_cost > 0 else 0
            meta_evolution_pct = (total_meta_evolution_cost / grand_total_cost * 100) if grand_total_cost > 0 else 0

            report_lines.append(
                f"| Phase 1 (DB Analysis - CLI) | ${total_phase1_cost:.2f} | {phase1_pct:.1f}% | "
                f"{total_phase1_calls} | {total_phase1_tokens_in:,} | {total_phase1_tokens_out:,} |"
            )
            report_lines.append(
                f"| Phase 2 (SQL Generation - API) | ${total_api_cost:.2f} | {api_pct:.1f}% | "
                f"- | - | - |"
            )
            report_lines.append(
                f"| Evolution (CLI) | ${total_evolution_cost:.2f} | {evolution_pct:.1f}% | "
                f"{total_evolution_calls} | {total_evolution_tokens_in:,} | {total_evolution_tokens_out:,} |"
            )
            if total_meta_evolution_cost > 0:
                report_lines.append(
                    f"| Meta-Evolution (CLI) | ${total_meta_evolution_cost:.2f} | {meta_evolution_pct:.1f}% | "
                    f"{total_meta_evolution_calls} | {total_meta_evolution_tokens_in:,} | {total_meta_evolution_tokens_out:,} |"
                )
            report_lines.append(
                f"| **Total** | **${grand_total_cost:.2f}** | **100%** | "
                f"**{total_cli_calls}** | **{total_cli_tokens_in:,}** | **{total_cli_tokens_out:,}** |"
            )

            # Answer the key question - identify the highest cost driver
            cost_drivers = [
                ('Phase 1 database analysis', phase1_pct),
                ('Phase 2 SQL generation', api_pct),
                ('Evolution', evolution_pct),
                ('Meta-Evolution', meta_evolution_pct)
            ]
            max_driver = max(cost_drivers, key=lambda x: x[1])
            report_lines.append(f"\n**Answer: {max_driver[0]} drives {max_driver[1]:.1f}% of total costs**")

            # Per-iteration cost breakdown
            report_lines.append("\n### Per-Iteration Cost Breakdown\n")
            report_lines.append("| Iter | Total | P1 Cost | P1 Calls | P1 In | P1 Out | P2 Cost | Evo Cost | Evo Calls | Evo In | Evo Out | Meta Cost | Meta Calls | Strategy | Meta-strategy |")
            report_lines.append("|------|-------|---------|----------|-------|--------|---------|----------|-----------|--------|---------|-----------|------------|----------|---------------|")

            for idx, cost_dict in enumerate(self.researcher.iteration_claude_costs):
                iter_num = idx + 1

                # Get evolution strategy for this iteration if available
                strategy_display = "-"
                if cost_dict.get('evolution_cost', 0) > 0:
                    # Find the agent created in this iteration
                    evolved_agent_id = None
                    for agent_id, agent_info in self.researcher.agent_pool.items():
                        if agent_info.get('created_iteration') == iter_num and agent_info.get('source') == 'evolution':
                            evolved_agent_id = agent_id
                            break

                    if evolved_agent_id:
                        strategy = self.researcher.agent_pool[evolved_agent_id].get('evolution_strategy', 'unknown')
                        strategy_display = strategy

                # Get meta-evolution strategy for this iteration if available
                meta_strategy_display = "-"
                if cost_dict.get('meta_evolution_cost', 0) > 0:
                    # Get the config for this iteration to find meta_evolution_strategy
                    iter_config = self.researcher.config_manager.get_config(iter_num)
                    meta_strategy = iter_config.get('meta_evolution_strategy')
                    if meta_strategy and meta_strategy != 'none':
                        meta_strategy_display = meta_strategy

                # Calculate totals for this iteration
                phase1_cost = cost_dict.get('phase1_cost', 0.0)
                phase1_calls = cost_dict.get('phase1_calls', 0)
                phase1_tokens_in = cost_dict.get('phase1_tokens_in', 0)
                phase1_tokens_out = cost_dict.get('phase1_tokens_out', 0)

                # Get API cost for this iteration (Phase 2 SQL generation)
                api_cost = self.researcher.iteration_costs[idx] if idx < len(self.researcher.iteration_costs) else 0.0

                evo_cost = cost_dict.get('evolution_cost', 0.0)
                evo_calls = cost_dict.get('evolution_calls', 0)
                evo_tokens_in = cost_dict.get('evolution_tokens_in', 0)
                evo_tokens_out = cost_dict.get('evolution_tokens_out', 0)

                meta_cost = cost_dict.get('meta_evolution_cost', 0.0)
                meta_calls = cost_dict.get('meta_evolution_calls', 0)

                total_cost = phase1_cost + api_cost + evo_cost + meta_cost
                total_calls = phase1_calls + evo_calls + meta_calls  # API calls not tracked
                total_tokens_in = phase1_tokens_in + evo_tokens_in  # API tokens not tracked here
                total_tokens_out = phase1_tokens_out + evo_tokens_out

                # Single row with all information
                report_lines.append(
                    f"| **{iter_num}** | **${total_cost:.2f}** | "
                    f"${phase1_cost:.2f} | {phase1_calls} | {phase1_tokens_in:,} | {phase1_tokens_out:,} | "
                    f"${api_cost:.2f} | "
                    f"${evo_cost:.2f} | {evo_calls} | {evo_tokens_in:,} | {evo_tokens_out:,} | "
                    f"${meta_cost:.2f} | {meta_calls} | "
                    f"{strategy_display} | {meta_strategy_display} |"
                )

            # Detailed per-iteration costs (one row per iteration with subtotals + components)
            report_lines.append("\n### Detailed Per-Iteration Costs\n")
            report_lines.append("| Iter | Total | **DB Analysis** | P1 | Test DB | **SQL Gen** | P2 | Test SQL | **Evolution** | Plan | Impl | Test Evo | Refl | Meta | Strategy | Meta-strategy |")
            report_lines.append("|------|-------|-----------------|----|---------|-----------|----|----------|---------------|------|------|----------|------|------|----------|---------------|")

            # Accumulators for totals
            totals = {
                'total_cost': 0.0,
                'total_db_analysis': 0.0,
                'phase1_base': 0.0,
                'test_db': 0.0,
                'total_sql_gen': 0.0,
                'phase2_base': 0.0,
                'test_sql': 0.0,
                'total_evolution': 0.0,
                'planning': 0.0,
                'implementation': 0.0,
                'test_evo': 0.0,
                'reflection': 0.0,
                'meta_evolution': 0.0
            }

            for idx, cost_dict in enumerate(self.researcher.iteration_claude_costs):
                iter_num = idx + 1

                # Get evolution strategy for this iteration if available
                strategy_display = "-"
                if cost_dict.get('evolution_cost', 0) > 0:
                    # Find the agent created in this iteration
                    evolved_agent_id = None
                    for agent_id, agent_info in self.researcher.agent_pool.items():
                        if agent_info.get('created_iteration') == iter_num and agent_info.get('source') == 'evolution':
                            evolved_agent_id = agent_id
                            break

                    if evolved_agent_id:
                        strategy = self.researcher.agent_pool[evolved_agent_id].get('evolution_strategy', 'unknown')
                        strategy_display = strategy

                # Get meta-evolution strategy for this iteration if available
                meta_strategy_display = "-"
                if cost_dict.get('meta_evolution_cost', 0) > 0:
                    # Get the config for this iteration to find meta_evolution_strategy
                    iter_config = self.researcher.config_manager.get_config(iter_num)
                    meta_strategy = iter_config.get('meta_evolution_strategy')
                    if meta_strategy and meta_strategy != 'none':
                        meta_strategy_display = meta_strategy

                # Get base costs
                phase1_base = cost_dict.get('phase1_cost', 0.0)
                phase2_base = self.researcher.iteration_costs[idx] if idx < len(self.researcher.iteration_costs) else 0.0

                # Get evolution breakdown
                evolution_breakdown = cost_dict.get('evolution_breakdown')

                # Extract test round costs
                test_db = 0.0
                test_sql = 0.0
                test_evo = 0.0
                planning = 0.0
                implementation = 0.0
                reflection = 0.0

                if evolution_breakdown:
                    # Planning and Implementation
                    planning = (evolution_breakdown or {}).get('planning', {}).get('cost', 0.0)
                    implementation = (evolution_breakdown or {}).get('implementation', {}).get('cost', 0.0)

                    # Test & Refine 1 (Round 3)
                    test_refine_1 = (evolution_breakdown or {}).get('test_refine_1', {})
                    test_db += test_refine_1.get('phase1', {}).get('cost', 0.0)
                    test_sql += test_refine_1.get('phase2', {}).get('cost', 0.0)
                    test_evo += test_refine_1.get('evolution', {}).get('cost', 0.0)

                    # Test & Refine 2 (Round 4)
                    test_refine_2 = (evolution_breakdown or {}).get('test_refine_2', {})
                    test_db += test_refine_2.get('phase1', {}).get('cost', 0.0)
                    test_sql += test_refine_2.get('phase2', {}).get('cost', 0.0)
                    test_evo += test_refine_2.get('evolution', {}).get('cost', 0.0)

                    # Reflection
                    reflection = (evolution_breakdown or {}).get('reflection', {}).get('cost', 0.0)

                # Get meta-evolution cost
                meta_evolution_cost = cost_dict.get('meta_evolution_cost', 0.0)

                # Calculate subtotals
                total_db_analysis = phase1_base + test_db
                total_sql_gen = phase2_base + test_sql
                total_evolution = planning + implementation + test_evo + reflection
                total_cost = total_db_analysis + total_sql_gen + total_evolution + meta_evolution_cost

                # Accumulate totals
                totals['total_cost'] += total_cost
                totals['total_db_analysis'] += total_db_analysis
                totals['phase1_base'] += phase1_base
                totals['test_db'] += test_db
                totals['total_sql_gen'] += total_sql_gen
                totals['phase2_base'] += phase2_base
                totals['test_sql'] += test_sql
                totals['total_evolution'] += total_evolution
                totals['planning'] += planning
                totals['implementation'] += implementation
                totals['test_evo'] += test_evo
                totals['reflection'] += reflection
                totals['meta_evolution'] += meta_evolution_cost

                # Format values (use "-" for zero values to keep table clean)
                def fmt(val):
                    return f"${val:.2f}" if val > 0 else "-"

                # Single row with all data
                report_lines.append(
                    f"| **{iter_num}** | ${total_cost:.2f} | "
                    f"**{fmt(total_db_analysis)}** | {fmt(phase1_base)} | {fmt(test_db)} | "
                    f"**{fmt(total_sql_gen)}** | {fmt(phase2_base)} | {fmt(test_sql)} | "
                    f"**{fmt(total_evolution)}** | {fmt(planning)} | {fmt(implementation)} | {fmt(test_evo)} | {fmt(reflection)} | "
                    f"{fmt(meta_evolution_cost)} | "
                    f"{strategy_display} | {meta_strategy_display} |"
                )

            # Add separator and totals row
            num_iterations = len(self.researcher.iteration_claude_costs)
            report_lines.append("|------|-------|-----------------|----|---------|-----------|----|----------|---------------|------|------|----------|------|------|----------|---------------|")
            report_lines.append(
                f"| **TOTAL** | **${totals['total_cost']:.2f}** | "
                f"**${totals['total_db_analysis']:.2f}** | ${totals['phase1_base']:.2f} | ${totals['test_db']:.2f} | "
                f"**${totals['total_sql_gen']:.2f}** | ${totals['phase2_base']:.2f} | ${totals['test_sql']:.2f} | "
                f"**${totals['total_evolution']:.2f}** | ${totals['planning']:.2f} | ${totals['implementation']:.2f} | ${totals['test_evo']:.2f} | ${totals['reflection']:.2f} | "
                f"${totals['meta_evolution']:.2f} | "
                f"- | - |"
            )

            # Evolution cost breakdown (if evolution happened)
            if total_evolution_cost > 0 and evolution_timings:
                report_lines.append("\n### Evolution Cost Breakdown (Avg per evolution)\n")
                report_lines.append("| Phase | Avg Cost | % of Evolution | Avg Tokens In | Avg Tokens Out |")
                report_lines.append("|-------|----------|----------------|---------------|----------------|")

                # Calculate averages from evolution_breakdown
                num_evolutions_with_cost = sum(1 for ic in self.researcher.iteration_claude_costs if ic.get('evolution_breakdown'))

                if num_evolutions_with_cost > 0:
                    avg_planning_cost = sum(
                        (ic.get('evolution_breakdown') or {}).get('planning', {}).get('cost', 0.0)
                        for ic in self.researcher.iteration_claude_costs
                    ) / num_evolutions_with_cost

                    avg_implementation_cost = sum(
                        (ic.get('evolution_breakdown') or {}).get('implementation', {}).get('cost', 0.0)
                        for ic in self.researcher.iteration_claude_costs
                    ) / num_evolutions_with_cost

                    # Test rounds - dynamically detect all test_refine_N keys
                    test_round_keys = set()
                    for ic in self.researcher.iteration_claude_costs:
                        evo_breakdown = ic.get('evolution_breakdown') or {}
                        for key in evo_breakdown.keys():
                            if key.startswith('test_refine_'):
                                test_round_keys.add(key)

                    # Calculate averages for each test round (costs and nested breakdown)
                    test_round_stats = {}
                    for key in sorted(test_round_keys):
                        test_round_stats[key] = {
                            'cost': sum(
                                (ic.get('evolution_breakdown') or {}).get(key, {}).get('cost', 0.0)
                                for ic in self.researcher.iteration_claude_costs
                            ) / num_evolutions_with_cost,
                            'phase1_cost': sum(
                                (ic.get('evolution_breakdown') or {}).get(key, {}).get('phase1', {}).get('cost', 0.0)
                                for ic in self.researcher.iteration_claude_costs
                            ) / num_evolutions_with_cost,
                            'evolution_cost': sum(
                                (ic.get('evolution_breakdown') or {}).get(key, {}).get('evolution', {}).get('cost', 0.0)
                                for ic in self.researcher.iteration_claude_costs
                            ) / num_evolutions_with_cost
                        }

                    avg_total_evo_cost = total_evolution_cost / num_evolutions_with_cost

                    # Tokens
                    avg_planning_tokens_in = sum(
                        (ic.get('evolution_breakdown') or {}).get('planning', {}).get('tokens_in', 0)
                        for ic in self.researcher.iteration_claude_costs
                    ) / num_evolutions_with_cost

                    avg_implementation_tokens_in = sum(
                        (ic.get('evolution_breakdown') or {}).get('implementation', {}).get('tokens_in', 0)
                        for ic in self.researcher.iteration_claude_costs
                    ) / num_evolutions_with_cost

                    # Add token statistics to test_round_stats
                    for key in sorted(test_round_keys):
                        test_round_stats[key]['tokens_in'] = sum(
                            (ic.get('evolution_breakdown') or {}).get(key, {}).get('tokens_in', 0)
                            for ic in self.researcher.iteration_claude_costs
                        ) / num_evolutions_with_cost

                        test_round_stats[key]['phase1_tokens_in'] = sum(
                            (ic.get('evolution_breakdown') or {}).get(key, {}).get('phase1', {}).get('tokens_in', 0)
                            for ic in self.researcher.iteration_claude_costs
                        ) / num_evolutions_with_cost

                        test_round_stats[key]['evolution_tokens_in'] = sum(
                            (ic.get('evolution_breakdown') or {}).get(key, {}).get('evolution', {}).get('tokens_in', 0)
                            for ic in self.researcher.iteration_claude_costs
                        ) / num_evolutions_with_cost

                    avg_planning_tokens_out = sum(
                        (ic.get('evolution_breakdown') or {}).get('planning', {}).get('tokens_out', 0)
                        for ic in self.researcher.iteration_claude_costs
                    ) / num_evolutions_with_cost

                    avg_implementation_tokens_out = sum(
                        (ic.get('evolution_breakdown') or {}).get('implementation', {}).get('tokens_out', 0)
                        for ic in self.researcher.iteration_claude_costs
                    ) / num_evolutions_with_cost

                    # Add tokens_out statistics to test_round_stats
                    for key in sorted(test_round_keys):
                        test_round_stats[key]['tokens_out'] = sum(
                            (ic.get('evolution_breakdown') or {}).get(key, {}).get('tokens_out', 0)
                            for ic in self.researcher.iteration_claude_costs
                        ) / num_evolutions_with_cost

                        test_round_stats[key]['phase1_tokens_out'] = sum(
                            (ic.get('evolution_breakdown') or {}).get(key, {}).get('phase1', {}).get('tokens_out', 0)
                            for ic in self.researcher.iteration_claude_costs
                        ) / num_evolutions_with_cost

                        test_round_stats[key]['evolution_tokens_out'] = sum(
                            (ic.get('evolution_breakdown') or {}).get(key, {}).get('evolution', {}).get('tokens_out', 0)
                            for ic in self.researcher.iteration_claude_costs
                        ) / num_evolutions_with_cost

                    # Reflection costs and tokens
                    avg_reflection_cost = sum(
                        (ic.get('evolution_breakdown') or {}).get('reflection', {}).get('cost', 0.0)
                        for ic in self.researcher.iteration_claude_costs
                    ) / num_evolutions_with_cost

                    avg_reflection_tokens_in = sum(
                        (ic.get('evolution_breakdown') or {}).get('reflection', {}).get('tokens_in', 0)
                        for ic in self.researcher.iteration_claude_costs
                    ) / num_evolutions_with_cost

                    avg_reflection_tokens_out = sum(
                        (ic.get('evolution_breakdown') or {}).get('reflection', {}).get('tokens_out', 0)
                        for ic in self.researcher.iteration_claude_costs
                    ) / num_evolutions_with_cost

                    # Percentages
                    pct_planning = (avg_planning_cost / avg_total_evo_cost * 100) if avg_total_evo_cost > 0 else 0
                    pct_implementation = (avg_implementation_cost / avg_total_evo_cost * 100) if avg_total_evo_cost > 0 else 0
                    pct_reflection = (avg_reflection_cost / avg_total_evo_cost * 100) if avg_total_evo_cost > 0 else 0

                    # Calculate percentages for each test round
                    for key in test_round_stats.keys():
                        test_round_stats[key]['pct'] = (test_round_stats[key]['cost'] / avg_total_evo_cost * 100) if avg_total_evo_cost > 0 else 0
                        test_round_stats[key]['pct_phase1'] = (test_round_stats[key]['phase1_cost'] / test_round_stats[key]['cost'] * 100) if test_round_stats[key]['cost'] > 0 else 0
                        test_round_stats[key]['pct_evolution'] = (test_round_stats[key]['evolution_cost'] / test_round_stats[key]['cost'] * 100) if test_round_stats[key]['cost'] > 0 else 0

                    report_lines.append(
                        f"| Planning (Round 1) | ${avg_planning_cost:.2f} | {pct_planning:.1f}% | "
                        f"{avg_planning_tokens_in:.0f} | {avg_planning_tokens_out:.0f} |"
                    )
                    report_lines.append(
                        f"| Implementation (Round 2) | ${avg_implementation_cost:.2f} | {pct_implementation:.1f}% | "
                        f"{avg_implementation_tokens_in:.0f} | {avg_implementation_tokens_out:.0f} |"
                    )

                    # Dynamic test round rows
                    for i, key in enumerate(sorted(test_round_keys), start=1):
                        stats = test_round_stats[key]
                        round_num = i + 2  # Rounds start at 3 (after planning=1, implementation=2)

                        # Total row
                        report_lines.append(
                            f"| **Test & Refine {i} (Round {round_num})** | **${stats['cost']:.2f}** | **{stats['pct']:.1f}%** | "
                            f"**{stats['tokens_in']:.0f}** | **{stats['tokens_out']:.0f}** |"
                        )

                        # Nested breakdown (if cost > 0)
                        if stats['cost'] > 0:
                            report_lines.append(
                                f"|   ├─ DB Analysis | ${stats['phase1_cost']:.2f} | {stats['pct_phase1']:.1f}% | "
                                f"{stats['phase1_tokens_in']:.0f} | {stats['phase1_tokens_out']:.0f} |"
                            )
                            report_lines.append(
                                f"|   └─ Evolution | ${stats['evolution_cost']:.2f} | {stats['pct_evolution']:.1f}% | "
                                f"{stats['evolution_tokens_in']:.0f} | {stats['evolution_tokens_out']:.0f} |"
                            )

                    # Reflection row (always shown now that we track it)
                    if avg_reflection_cost > 0:
                        pct_reflection = (avg_reflection_cost / avg_total_evo_cost * 100) if avg_total_evo_cost > 0 else 0
                        report_lines.append(
                            f"| Reflection | ${avg_reflection_cost:.2f} | {pct_reflection:.1f}% | "
                            f"{avg_reflection_tokens_in:.0f} | {avg_reflection_tokens_out:.0f} |"
                        )

                    # Calculate total tokens (dynamically including all test rounds)
                    total_tokens_in = avg_planning_tokens_in + avg_implementation_tokens_in + avg_reflection_tokens_in
                    total_tokens_out = avg_planning_tokens_out + avg_implementation_tokens_out + avg_reflection_tokens_out
                    for stats in test_round_stats.values():
                        total_tokens_in += stats['tokens_in']
                        total_tokens_out += stats['tokens_out']

                    report_lines.append(
                        f"| **Total** | **${avg_total_evo_cost:.2f}** | **100%** | "
                        f"**{total_tokens_in:.0f}** | **{total_tokens_out:.0f}** |"
                    )

            # Cache performance
            total_cache_created = total_phase1_cache_created + total_evolution_cache_created
            total_cache_read = total_phase1_cache_read + total_evolution_cache_read
            cache_hit_rate = (total_cache_read / (total_cache_read + total_cache_created) * 100) if (total_cache_read + total_cache_created) > 0 else 0

            report_lines.append("\n### Cache Performance\n")
            report_lines.append(f"- **Cache Creation**: {total_cache_created:,} tokens")
            report_lines.append(f"- **Cache Hits**: {total_cache_read:,} tokens ({cache_hit_rate:.1f}% hit rate)")
        else:
            report_lines.append("No Claude CLI cost tracking data available")

        # Phase 1 Failures section
        report_lines.append("\n## Phase 1 Failures")
        if self.researcher.phase1_failures:
            report_lines.append(f"Total Phase 1 failures: {len(self.researcher.phase1_failures)}")
            report_lines.append("\n| Agent | Database | Iteration |")
            report_lines.append("|-------|----------|-----------|")
            for agent_id, db_name, iteration_num in sorted(self.researcher.phase1_failures):
                report_lines.append(f"| {agent_id} | {db_name} | {iteration_num} |")
        else:
            report_lines.append("No Phase 1 failures encountered ✅")

        # Zero Accuracy Cases section
        report_lines.append("\n## Zero Accuracy Cases")
        if self.researcher.zero_accuracy_cases:
            report_lines.append(f"Total zero accuracy cases: {len(self.researcher.zero_accuracy_cases)}")
            report_lines.append("\n| Agent | Database | Iteration | Questions Tested |")
            report_lines.append("|-------|----------|-----------|------------------|")
            for agent_id, db_name, iteration_num, total_q in sorted(self.researcher.zero_accuracy_cases):
                report_lines.append(f"| {agent_id} | {db_name} | {iteration_num} | {total_q} |")
        else:
            report_lines.append("No zero accuracy cases encountered ✅")

        # Exception Failures section
        report_lines.append("\n## Exception Failures")
        if self.researcher.exception_failures:
            report_lines.append(f"Total exception failures: {len(self.researcher.exception_failures)}")
            report_lines.append("")
            report_lines.append("These are errors that occurred during SQL generation or evaluation (Phase 2).")
            report_lines.append("Unlike Phase 1 failures, these may indicate issues with generated SQL or evaluation logic.")
            report_lines.append("")
            report_lines.append("| Agent | Database | Iteration | Error | Questions |")
            report_lines.append("|-------|----------|-----------|-------|-----------|")
            for agent_id, db_name, iteration_num, error_msg, total_q in sorted(self.researcher.exception_failures):
                # Truncate error message if too long
                error_display = error_msg[:60] + "..." if len(error_msg) > 60 else error_msg
                report_lines.append(f"| {agent_id} | {db_name} | {iteration_num} | {error_display} | {total_q} |")
        else:
            report_lines.append("No exception failures encountered ✅")

        # Evolution Health section
        report_lines.append("\n## Evolution Health")

        # Validation failures
        if hasattr(self.researcher.evolver, 'evolution_validation_failures') and self.researcher.evolver.evolution_validation_failures:
            validation_failures = self.researcher.evolver.evolution_validation_failures
            report_lines.append(f"\n### Artifact Validation Failures")
            report_lines.append(f"Total validation failures: {len(validation_failures)}")
            report_lines.append("\n| Iteration | Errors | Used Continue |")
            report_lines.append("|-----------|--------|---------------|")
            for failure in validation_failures:
                errors_str = ', '.join(failure['errors'][:2])  # Show first 2 errors
                if len(failure['errors']) > 2:
                    errors_str += f" (+{len(failure['errors'])-2} more)"
                report_lines.append(f"| {failure['iteration']} | {errors_str} | {failure['used_continue']} |")
        else:
            report_lines.append("\nNo artifact validation failures ✅")

        # Write report
        if report_type == 'final':
            report_file = self.researcher.experiment_dir / 'final_report.md'
        else:
            iteration_dir = self.researcher.experiment_dir / f"iteration_{iteration:03d}"
            iteration_dir.mkdir(exist_ok=True)
            report_file = iteration_dir / 'interim_report.md'

        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))

        if report_type == 'final':
            print(f"\n📊 Final report saved to: {report_file}")