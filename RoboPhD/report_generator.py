"""
Report generation for RoboPhD research system.

Handles creation of final and interim reports, including:
- Experiment configuration summaries
- Agent performance metrics and ELO rankings
- Evolution strategy tracking
- Cost analysis and timing
- Cache performance statistics
"""

import os
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
                "examples_per_iteration",
                "agents_per_iteration"
            ],
            "Models": [
                "evolution_model",
                "meta_evolution_model"
            ],
            "Evolution": [
                "evolution_strategy",
                "new_agent_test_rounds"
            ],
            "Meta-Evolution": [
                "meta_evolution_strategy",
                "dollar_budget",
                "evaluation_budget"
            ],
            "Performance": [
                "max_workers"
            ],
            "Timeouts": [
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
                print(f"  📊 Generated interim report: {os.path.relpath(report_file)}")
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

        # Meta-parameters to filter out (implementation details)
        meta_params = {"config_schedule", "meta_config_schedule", "weighted_random_configs", "meta_evolution_strategy"}

        # System Defaults (Iteration 0)
        report_lines.append("")
        report_lines.append("### System Defaults (Iteration 0)")
        report_lines.append("")
        report_lines.append("Default values when no configuration is specified:")
        report_lines.append("")

        defaults = self.researcher.config_manager.get_config(0)
        for param_name in sorted(defaults.keys()):
            if param_name in meta_params:
                continue
            value = defaults[param_name]
            if param_name.endswith('_model') and value:
                value = self._format_model_name(value)
            report_lines.append(f"- {param_name}: {value}")
        report_lines.append("")

        # User Configuration (Iteration 1) - show delta only
        report_lines.append("### User Configuration (Iteration 1)")
        report_lines.append("")

        user_delta = self.researcher.config_manager.iteration_configs.get(1, {})
        # Filter out meta params
        user_delta_filtered = {k: v for k, v in user_delta.items() if k not in meta_params}

        if user_delta_filtered:
            report_lines.append("The following parameters were explicitly set, overriding defaults:")
            report_lines.append("")
            for param_name in sorted(user_delta_filtered.keys()):
                value = user_delta_filtered[param_name]
                if param_name.endswith('_model') and value:
                    value = self._format_model_name(value)
                report_lines.append(f"- {param_name}: {value}")
        else:
            report_lines.append("(No changes from defaults)")
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
            report_lines.append("\n| Iteration | Recovery Action |")
            report_lines.append("|-----------|----------------|")
            for incident in self.researcher.five_hour_limit_incidents:
                recovery = incident.get('recovery_action', 'Unknown')
                report_lines.append(f"| {incident['iteration']} | {recovery} |")

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
                    max_score = max(test_results[k]['average_score'] for k in test_results.keys())
                    winners = [k for k in test_results.keys() if test_results[k]['average_score'] == max_score]

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
        report_lines.append("| Agent | ELO | Mean Score | Tests |")
        report_lines.append("|-------|-----|------------|-------|")

        for agent_id in sorted_agents:
            perf = self.researcher.performance_records[agent_id]
            report_lines.append(f"| {agent_id} | {perf['elo']:.0f} | "
                              f"{perf['mean_score']:.3f} | {perf['test_count']} |")

        # Iteration progression table
        report_lines.append("\n### Iteration Progression\n")
        report_lines.append("| Iter | Winner(s) | Score | Evo Time | Test Time | Total Time |")
        report_lines.append("|------|-----------|-------|----------|-----------|------------|")

        for i, test_results in enumerate(self.researcher.test_history):
            iteration_num = i + 1

            # Find winner(s)
            if test_results:
                max_score = max(test_results[k]['average_score'] for k in test_results.keys())
                winners = [k for k in test_results.keys() if test_results[k]['average_score'] == max_score]
                winner_str = ', '.join(winners) if len(winners) <= 2 else f"{winners[0]} +{len(winners)-1}"
                score_str = f"{max_score:.3f}"
            else:
                winner_str = "N/A"
                score_str = "N/A"

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

            report_lines.append(f"| {iteration_num} | {winner_str} | {score_str} | "
                              f"{evo_time_str} | {test_time_str} | {total_time_str} |")

        # Best agent
        if sorted_agents:
            best_agent = sorted_agents[0]
            report_lines.append(f"\n## Best Agent: {best_agent}")
            report_lines.append(f"- ELO Score: {self.researcher.performance_records[best_agent]['elo']:.0f}")
            report_lines.append(f"- Mean Score: {self.researcher.performance_records[best_agent]['mean_score']:.3f}")

        # ELO Leadership Progression
        report_lines.append("\n")
        elo_leadership_section = self._generate_elo_leadership_section()
        report_lines.append(elo_leadership_section)

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
            avg_first_draft = sum(t.get('first_draft', 0) for t in evolution_timings) / num_evolutions

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
            avg_evo_time = (avg_error_analyzer + avg_first_draft +
                           sum(test_round_averages.values()) + avg_reflection)

            # Calculate percentages of total evolution time
            pct_error_analyzer = (avg_error_analyzer / avg_evo_time * 100) if avg_evo_time > 0 else 0
            pct_first_draft = (avg_first_draft / avg_evo_time * 100) if avg_evo_time > 0 else 0
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
            report_lines.append(f"| First Draft (Round 1) | {avg_first_draft/60:.1f}m | {pct_first_draft:.1f}% |")

            # Dynamic test round rows
            for i, key in enumerate(sorted(test_round_averages.keys()), start=1):
                avg_time = test_round_averages[key]
                pct = test_round_percentages[key]
                round_num = i + 1  # Rounds start at 2 (after first_draft=1)
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
            total_eval_cost = sum(ic.get('eval_cost', 0.0) for ic in self.researcher.iteration_claude_costs)

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

            total_cli_cost = total_evolution_cost + total_meta_evolution_cost
            total_cli_calls = total_evolution_calls + total_meta_evolution_calls
            total_cli_tokens_in = total_evolution_tokens_in + total_meta_evolution_tokens_in
            total_cli_tokens_out = total_evolution_tokens_out + total_meta_evolution_tokens_out

            # Grand total (Eval + CLI)
            grand_total_cost = total_eval_cost + total_cli_cost

            # Overall summary
            report_lines.append(f"- **Total Cost**: ${grand_total_cost:.2f}")
            report_lines.append(f"  - **Evaluation Cost**: ${total_eval_cost:.2f}")
            report_lines.append(f"  - **Evolution CLI Cost**: ${total_cli_cost:.2f}")
            report_lines.append(f"- **Total CLI Calls**: {total_cli_calls}")
            report_lines.append(f"- **Total CLI Input Tokens**: {total_cli_tokens_in:,}")
            report_lines.append(f"- **Total CLI Output Tokens**: {total_cli_tokens_out:,}")
            report_lines.append(f"- **Total Cache Created**: {total_evolution_cache_created:,}")
            report_lines.append(f"- **Total Cache Read**: {total_evolution_cache_read:,}")

            # Cost breakdown
            report_lines.append("\n### Cost Breakdown\n")
            report_lines.append("| Operation | Cost | % of Total | Calls | Tokens In | Tokens Out |")
            report_lines.append("|-----------|------|------------|-------|-----------|------------|")

            eval_pct = (total_eval_cost / grand_total_cost * 100) if grand_total_cost > 0 else 0
            evolution_pct = (total_evolution_cost / grand_total_cost * 100) if grand_total_cost > 0 else 0
            meta_evolution_pct = (total_meta_evolution_cost / grand_total_cost * 100) if grand_total_cost > 0 else 0

            report_lines.append(
                f"| Evaluation | ${total_eval_cost:.2f} | {eval_pct:.1f}% | "
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

            # Identify highest cost driver
            cost_drivers = [
                ('Evaluation', eval_pct),
                ('Evolution', evolution_pct),
                ('Meta-Evolution', meta_evolution_pct)
            ]
            max_driver = max(cost_drivers, key=lambda x: x[1])
            report_lines.append(f"\n**Answer: {max_driver[0]} drives {max_driver[1]:.1f}% of total costs**")

            # Per-iteration cost breakdown
            report_lines.append("\n### Per-Iteration Cost Breakdown\n")
            report_lines.append("| Iter | Total | Eval Cost | Evo Cost | Evo Calls | Evo In | Evo Out | Meta Cost | Meta Calls | Strategy | Meta-strategy |")
            report_lines.append("|------|-------|-----------|----------|-----------|--------|---------|-----------|------------|----------|---------------|")

            for idx, cost_dict in enumerate(self.researcher.iteration_claude_costs):
                iter_num = idx + 1

                # Get evolution strategy for this iteration if available
                strategy_display = "-"
                if cost_dict.get('evolution_cost', 0) > 0:
                    evolved_agent_id = None
                    for agent_id, agent_info in self.researcher.agent_pool.items():
                        if agent_info.get('created_iteration') == iter_num and agent_info.get('source') == 'evolution':
                            evolved_agent_id = agent_id
                            break
                    if evolved_agent_id:
                        strategy = self.researcher.agent_pool[evolved_agent_id].get('evolution_strategy', 'unknown')
                        strategy_display = strategy

                meta_strategy_display = "-"
                if cost_dict.get('meta_evolution_cost', 0) > 0:
                    iter_config = self.researcher.config_manager.get_config(iter_num)
                    meta_strategy = iter_config.get('meta_evolution_strategy')
                    if meta_strategy and meta_strategy != 'none':
                        meta_strategy_display = meta_strategy

                eval_cost = cost_dict.get('eval_cost', 0.0)
                evo_cost = cost_dict.get('evolution_cost', 0.0)
                evo_calls = cost_dict.get('evolution_calls', 0)
                evo_tokens_in = cost_dict.get('evolution_tokens_in', 0)
                evo_tokens_out = cost_dict.get('evolution_tokens_out', 0)
                meta_cost = cost_dict.get('meta_evolution_cost', 0.0)
                meta_calls = cost_dict.get('meta_evolution_calls', 0)
                total_cost = eval_cost + evo_cost + meta_cost

                report_lines.append(
                    f"| **{iter_num}** | **${total_cost:.2f}** | "
                    f"${eval_cost:.2f} | "
                    f"${evo_cost:.2f} | {evo_calls} | {evo_tokens_in:,} | {evo_tokens_out:,} | "
                    f"${meta_cost:.2f} | {meta_calls} | "
                    f"{strategy_display} | {meta_strategy_display} |"
                )

            # Detailed per-iteration costs
            report_lines.append("\n### Detailed Per-Iteration Costs\n")
            report_lines.append(f"| Iter | Total | **Evaluation** | **Evolution** | 1st Draft | Test Eval | Test Evo | Refl | Meta | Strategy | Meta-strategy |")
            report_lines.append("|------|-------|----------------|---------------|-----------|-----------|----------|------|------|----------|---------------|")

            # Accumulators for totals
            totals = {
                'total_cost': 0.0,
                'total_eval': 0.0,
                'total_evolution': 0.0,
                'first_draft': 0.0,
                'test_eval': 0.0,
                'test_evo': 0.0,
                'reflection': 0.0,
                'meta_evolution': 0.0
            }

            for idx, cost_dict in enumerate(self.researcher.iteration_claude_costs):
                iter_num = idx + 1

                strategy_display = "-"
                if cost_dict.get('evolution_cost', 0) > 0:
                    evolved_agent_id = None
                    for agent_id, agent_info in self.researcher.agent_pool.items():
                        if agent_info.get('created_iteration') == iter_num and agent_info.get('source') == 'evolution':
                            evolved_agent_id = agent_id
                            break
                    if evolved_agent_id:
                        strategy = self.researcher.agent_pool[evolved_agent_id].get('evolution_strategy', 'unknown')
                        strategy_display = strategy

                meta_strategy_display = "-"
                if cost_dict.get('meta_evolution_cost', 0) > 0:
                    iter_config = self.researcher.config_manager.get_config(iter_num)
                    meta_strategy = iter_config.get('meta_evolution_strategy')
                    if meta_strategy and meta_strategy != 'none':
                        meta_strategy_display = meta_strategy

                # Get base costs
                eval_base = cost_dict.get('eval_cost', 0.0)

                # Get evolution breakdown
                evolution_breakdown = cost_dict.get('evolution_breakdown')

                # Extract test round costs
                test_eval = 0.0
                test_evo = 0.0
                first_draft = 0.0
                reflection = 0.0

                if evolution_breakdown:
                    first_draft = evolution_breakdown.get('first_draft', {}).get('cost', 0.0)

                    test_refine_1 = evolution_breakdown.get('test_refine_1', {})
                    test_eval += test_refine_1.get('eval', {}).get('cost', 0.0)
                    test_evo += test_refine_1.get('evolution', {}).get('cost', 0.0)

                    test_refine_2 = evolution_breakdown.get('test_refine_2', {})
                    test_eval += test_refine_2.get('eval', {}).get('cost', 0.0)
                    test_evo += test_refine_2.get('evolution', {}).get('cost', 0.0)

                    reflection = evolution_breakdown.get('reflection', {}).get('cost', 0.0)

                meta_evolution_cost = cost_dict.get('meta_evolution_cost', 0.0)

                total_eval_iter = eval_base + test_eval
                total_evolution_iter = first_draft + test_evo + reflection
                total_cost = total_eval_iter + total_evolution_iter + meta_evolution_cost

                totals['total_cost'] += total_cost
                totals['total_eval'] += total_eval_iter
                totals['total_evolution'] += total_evolution_iter
                totals['first_draft'] += first_draft
                totals['test_eval'] += test_eval
                totals['test_evo'] += test_evo
                totals['reflection'] += reflection
                totals['meta_evolution'] += meta_evolution_cost

                def fmt(val):
                    return f"${val:.2f}" if val > 0 else "-"

                report_lines.append(
                    f"| **{iter_num}** | ${total_cost:.2f} | "
                    f"**{fmt(total_eval_iter)}** | "
                    f"**{fmt(total_evolution_iter)}** | {fmt(first_draft)} | {fmt(test_eval)} | {fmt(test_evo)} | {fmt(reflection)} | "
                    f"{fmt(meta_evolution_cost)} | "
                    f"{strategy_display} | {meta_strategy_display} |"
                )

            # Totals row
            report_lines.append("|------|-------|----------------|---------------|-----------|-----------|----------|------|------|----------|---------------|")
            report_lines.append(
                f"| **TOTAL** | **${totals['total_cost']:.2f}** | "
                f"**${totals['total_eval']:.2f}** | "
                f"**${totals['total_evolution']:.2f}** | ${totals['first_draft']:.2f} | ${totals['test_eval']:.2f} | ${totals['test_evo']:.2f} | ${totals['reflection']:.2f} | "
                f"${totals['meta_evolution']:.2f} | "
                f"- | - |"
            )

            # Evolution cost breakdown (if evolution happened)
            if total_evolution_cost > 0 and evolution_timings:
                report_lines.append("\n### Evolution Cost Breakdown (Avg per evolution)\n")
                report_lines.append("| Phase | Avg Cost | % of Evolution | Avg Tokens In | Avg Tokens Out |")
                report_lines.append("|-------|----------|----------------|---------------|----------------|")

                num_evolutions_with_cost = sum(1 for ic in self.researcher.iteration_claude_costs if ic.get('evolution_breakdown'))

                if num_evolutions_with_cost > 0:
                    avg_first_draft_cost = sum(
                        (ic.get('evolution_breakdown') or {}).get('first_draft', {}).get('cost', 0.0)
                        for ic in self.researcher.iteration_claude_costs
                    ) / num_evolutions_with_cost

                    # Test rounds - dynamically detect all test_refine_N keys
                    test_round_keys = set()
                    for ic in self.researcher.iteration_claude_costs:
                        evo_breakdown = ic.get('evolution_breakdown') or {}
                        for key in evo_breakdown.keys():
                            if key.startswith('test_refine_'):
                                test_round_keys.add(key)

                    test_round_stats = {}
                    for key in sorted(test_round_keys):
                        test_round_stats[key] = {
                            'cost': sum(
                                (ic.get('evolution_breakdown') or {}).get(key, {}).get('cost', 0.0)
                                for ic in self.researcher.iteration_claude_costs
                            ) / num_evolutions_with_cost,
                            'eval_cost': sum(
                                (ic.get('evolution_breakdown') or {}).get(key, {}).get('eval', {}).get('cost', 0.0)
                                for ic in self.researcher.iteration_claude_costs
                            ) / num_evolutions_with_cost,
                            'evolution_cost': sum(
                                (ic.get('evolution_breakdown') or {}).get(key, {}).get('evolution', {}).get('cost', 0.0)
                                for ic in self.researcher.iteration_claude_costs
                            ) / num_evolutions_with_cost
                        }

                    avg_total_evo_cost = total_evolution_cost / num_evolutions_with_cost

                    avg_first_draft_tokens_in = sum(
                        (ic.get('evolution_breakdown') or {}).get('first_draft', {}).get('tokens_in', 0)
                        for ic in self.researcher.iteration_claude_costs
                    ) / num_evolutions_with_cost

                    for key in sorted(test_round_keys):
                        test_round_stats[key]['tokens_in'] = sum(
                            (ic.get('evolution_breakdown') or {}).get(key, {}).get('tokens_in', 0)
                            for ic in self.researcher.iteration_claude_costs
                        ) / num_evolutions_with_cost

                        test_round_stats[key]['eval_tokens_in'] = sum(
                            (ic.get('evolution_breakdown') or {}).get(key, {}).get('eval', {}).get('tokens_in', 0)
                            for ic in self.researcher.iteration_claude_costs
                        ) / num_evolutions_with_cost

                        test_round_stats[key]['evolution_tokens_in'] = sum(
                            (ic.get('evolution_breakdown') or {}).get(key, {}).get('evolution', {}).get('tokens_in', 0)
                            for ic in self.researcher.iteration_claude_costs
                        ) / num_evolutions_with_cost

                    avg_first_draft_tokens_out = sum(
                        (ic.get('evolution_breakdown') or {}).get('first_draft', {}).get('tokens_out', 0)
                        for ic in self.researcher.iteration_claude_costs
                    ) / num_evolutions_with_cost

                    for key in sorted(test_round_keys):
                        test_round_stats[key]['tokens_out'] = sum(
                            (ic.get('evolution_breakdown') or {}).get(key, {}).get('tokens_out', 0)
                            for ic in self.researcher.iteration_claude_costs
                        ) / num_evolutions_with_cost

                        test_round_stats[key]['eval_tokens_out'] = sum(
                            (ic.get('evolution_breakdown') or {}).get(key, {}).get('eval', {}).get('tokens_out', 0)
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
                    pct_first_draft = (avg_first_draft_cost / avg_total_evo_cost * 100) if avg_total_evo_cost > 0 else 0

                    for key in test_round_stats.keys():
                        test_round_stats[key]['pct'] = (test_round_stats[key]['cost'] / avg_total_evo_cost * 100) if avg_total_evo_cost > 0 else 0
                        test_round_stats[key]['pct_eval'] = (test_round_stats[key]['eval_cost'] / test_round_stats[key]['cost'] * 100) if test_round_stats[key]['cost'] > 0 else 0
                        test_round_stats[key]['pct_evolution'] = (test_round_stats[key]['evolution_cost'] / test_round_stats[key]['cost'] * 100) if test_round_stats[key]['cost'] > 0 else 0

                    report_lines.append(
                        f"| First Draft (Round 1) | ${avg_first_draft_cost:.2f} | {pct_first_draft:.1f}% | "
                        f"{avg_first_draft_tokens_in:.0f} | {avg_first_draft_tokens_out:.0f} |"
                    )

                    for i, key in enumerate(sorted(test_round_keys), start=1):
                        stats = test_round_stats[key]
                        round_num = i + 1

                        report_lines.append(
                            f"| **Test & Refine {i} (Round {round_num})** | **${stats['cost']:.2f}** | **{stats['pct']:.1f}%** | "
                            f"**{stats['tokens_in']:.0f}** | **{stats['tokens_out']:.0f}** |"
                        )

                        if stats['cost'] > 0:
                            report_lines.append(
                                f"|   ├─ Evaluation | ${stats['eval_cost']:.2f} | {stats['pct_eval']:.1f}% | "
                                f"{stats['eval_tokens_in']:.0f} | {stats['eval_tokens_out']:.0f} |"
                            )
                            report_lines.append(
                                f"|   └─ Evolution | ${stats['evolution_cost']:.2f} | {stats['pct_evolution']:.1f}% | "
                                f"{stats['evolution_tokens_in']:.0f} | {stats['evolution_tokens_out']:.0f} |"
                            )

                    if avg_reflection_cost > 0:
                        pct_reflection = (avg_reflection_cost / avg_total_evo_cost * 100) if avg_total_evo_cost > 0 else 0
                        report_lines.append(
                            f"| Reflection | ${avg_reflection_cost:.2f} | {pct_reflection:.1f}% | "
                            f"{avg_reflection_tokens_in:.0f} | {avg_reflection_tokens_out:.0f} |"
                        )

                    total_tokens_in = avg_first_draft_tokens_in + avg_reflection_tokens_in
                    total_tokens_out = avg_first_draft_tokens_out + avg_reflection_tokens_out
                    for stats in test_round_stats.values():
                        total_tokens_in += stats['tokens_in']
                        total_tokens_out += stats['tokens_out']

                    report_lines.append(
                        f"| **Total** | **${avg_total_evo_cost:.2f}** | **100%** | "
                        f"**{total_tokens_in:.0f}** | **{total_tokens_out:.0f}** |"
                    )

            # Cache performance
            cache_hit_rate = (total_evolution_cache_read / (total_evolution_cache_read + total_evolution_cache_created) * 100) if (total_evolution_cache_read + total_evolution_cache_created) > 0 else 0

            report_lines.append("\n### Cache Performance\n")
            report_lines.append(f"- **Cache Creation**: {total_evolution_cache_created:,} tokens")
            report_lines.append(f"- **Cache Hits**: {total_evolution_cache_read:,} tokens ({cache_hit_rate:.1f}% hit rate)")
        else:
            report_lines.append("No cost tracking data available")

        # Zero Accuracy Cases section
        report_lines.append("\n## Zero Accuracy Cases")
        if self.researcher.zero_accuracy_cases:
            # Check if hierarchical (has context names) or non-hierarchical (context is None)
            is_hierarchical = any(ctx is not None for _, ctx, _, _ in self.researcher.zero_accuracy_cases)
            report_lines.append(f"Total zero accuracy cases: {len(self.researcher.zero_accuracy_cases)}")
            if is_hierarchical:
                # Hierarchical domain: show per-context breakdown
                context_label = getattr(self.researcher.domain, 'context_label', 'Database')
                report_lines.append(f"\n| Agent | {context_label} | Iteration | Questions Tested |")
                report_lines.append("|-------|----------|-----------|------------------|")
                for agent_id, ctx_name, iteration_num, total_q in sorted(self.researcher.zero_accuracy_cases):
                    report_lines.append(f"| {agent_id} | {ctx_name} | {iteration_num} | {total_q} |")
            else:
                # Non-hierarchical domain: show per-iteration only
                report_lines.append("\n| Agent | Iteration | Problems Tested |")
                report_lines.append("|-------|-----------|-----------------|")
                for agent_id, _, iteration_num, total_q in sorted(self.researcher.zero_accuracy_cases):
                    report_lines.append(f"| {agent_id} | {iteration_num} | {total_q} |")
        else:
            report_lines.append("No zero accuracy cases encountered ✅")

        # Exception Failures section
        report_lines.append("\n## Exception Failures")
        if self.researcher.exception_failures:
            report_lines.append(f"Total exception failures: {len(self.researcher.exception_failures)}")
            report_lines.append("")
            # Use domain-specific solution name
            exception_solution = getattr(self.researcher, 'domain', None)
            exception_solution_name = exception_solution.solution_name if exception_solution else "SQL"
            report_lines.append(f"These are errors that occurred during {exception_solution_name} generation or evaluation.")
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