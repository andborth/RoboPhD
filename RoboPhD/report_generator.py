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


def format_non_binary_scores(non_binary: dict) -> list:
    """Render non-binary scores (not 0.0 or 1.0) as report lines, grouped by agent."""
    if not non_binary:
        return []
    lines = ["## Non-Binary Scores", ""]
    for agent in sorted(non_binary):
        entries = sorted(non_binary[agent], key=lambda e: e['question_id'])
        items = ', '.join(f"{e['question_id']} ({e['score']:.2f})" for e in entries)
        lines.append(f"**{agent}** ({len(entries)}): {items}")
    lines.append("")
    return lines


def format_agent_errors(index: dict) -> list:
    """Render agent errors (crashes, rate limits) as report lines, grouped by agent."""
    by_agent = index.get('by_agent', {})
    agents_with_errors = {
        agent: sorted(stats.get('error_ids', []))
        for agent, stats in by_agent.items()
        if stats.get('error_ids')
    }
    if not agents_with_errors:
        return []

    total = sum(len(ids) for ids in agents_with_errors.values())
    lines = ["## Agent Errors", "",
             f"{total} problem(s) had infrastructure errors (agent crash, rate limit):", ""]
    for agent in sorted(agents_with_errors):
        error_ids = agents_with_errors[agent]
        lines.append(f"- **{agent}** ({len(error_ids)}): {', '.join(error_ids)}")
    lines.append("")
    return lines


def is_continuous_scoring(scores_by_question: dict) -> bool:
    """Detect whether scores are continuous (not mostly binary).

    Returns True if scores should use continuous-score report format.
    Binary format is used when >= 80% of scores are exactly 0 or 1,
    AND all scores are in [0, 1].
    """
    all_scores = []
    for qid, agent_scores in scores_by_question.items():
        for agent, score in agent_scores.items():
            all_scores.append(score)

    if not all_scores:
        return False

    # Any score outside [0, 1] → continuous
    if any(s < 0.0 or s > 1.0 for s in all_scores):
        return True

    # Less than 80% binary → continuous
    binary_count = sum(1 for s in all_scores if s == 0.0 or s == 1.0)
    if binary_count / len(all_scores) < 0.8:
        return True

    return False


# ---------------------------------------------------------------------------
# Binary-score error analysis report
# ---------------------------------------------------------------------------

def format_binary_report_comparative(index: dict) -> list[str]:
    """Build binary-score error analysis lines for the iteration (comparative) report.

    Includes accuracy table, consensus stats, split decisions, and non-binary appendix.
    """
    from collections import defaultdict

    summary = index.get('summary', {})
    agents = summary.get('agents', [])
    accuracies = summary.get('agent_accuracies', {})
    total_q = summary.get('total_questions', 0)

    lines = []

    if agents:
        agent_strs = [f"{a} ({accuracies.get(a, 0)}%)" for a in agents]
        lines.append(f"**Agents**: {', '.join(agent_strs)}")
        lines.append("")

    # Consensus stats
    consensus = summary.get('consensus_stats', {})
    lines.extend([
        "## Summary",
        f"- Total questions: {total_q}",
        f"- Consensus correct: {consensus.get('all_correct', 0)} ({consensus.get('all_correct_pct', 0)}%)",
        f"- Consensus failures: {consensus.get('all_failed', 0)} ({consensus.get('all_failed_pct', 0)}%)",
        f"- Split decisions: {consensus.get('split_decisions', 0)} ({consensus.get('split_decisions_pct', 0)}%)",
        ""
    ])

    # Per-agent accuracy table
    by_agent = index.get('by_agent', {})
    if by_agent and agents:
        lines.extend([
            "## Agent Accuracy",
            "",
            "| Agent | Correct | Failed | Errors | Accuracy |",
            "|-------|---------|--------|--------|----------|"
        ])
        for agent in agents:
            stats = by_agent.get(agent, {})
            correct = stats.get('total_correct', 0)
            failed = stats.get('total_failed', stats.get('total_errors', 0))
            errors = stats.get('total_errors', 0)
            accuracy = stats.get('accuracy', 0.0)
            lines.append(f"| {agent} | {correct} | {failed} | {errors} | {accuracy:.1f}% |")
        lines.append("")

    # Consensus failures (all agents failed)
    cross_patterns = index.get('cross_agent_patterns', {})
    consensus_failures = cross_patterns.get('consensus_failures', cross_patterns.get('consensus_errors', []))
    if consensus_failures:
        fail_count = len(consensus_failures)
        if fail_count <= 10:
            id_str = ', '.join(consensus_failures)
        else:
            id_str = ', '.join(consensus_failures[:10]) + ', ...'
        lines.extend([
            "## Consensus Failures",
            "",
            f"All agents failed on {fail_count} questions: {id_str}",
            ""
        ])

    # Split decisions
    split_decisions = cross_patterns.get('split_decisions', {})
    if split_decisions:
        lines.extend([
            "## Split Decisions",
            "",
            f"Total split decisions: {len(split_decisions)}",
            ""
        ])

        pattern_groups = defaultdict(list)
        for question_id, split_info in split_decisions.items():
            correct_agents = tuple(sorted(split_info.get('correct', [])))
            wrong_agents = tuple(sorted(split_info.get('wrong', [])))
            pattern_groups[(correct_agents, wrong_agents)].append(question_id)

        sorted_patterns = sorted(
            pattern_groups.items(),
            key=lambda x: (-len(x[1]), x[0])
        )

        for (correct_agents, wrong_agents), question_ids in sorted_patterns:
            correct_str = ', '.join(correct_agents)
            wrong_str = ', '.join(wrong_agents)
            question_ids_str = ', '.join(f"**{qid}**" for qid in sorted(question_ids))
            lines.append(f"- ✓ {correct_str} | ✗ {wrong_str}: {question_ids_str}")

        lines.append("")

    # Non-binary scores (e.g., partial credit)
    lines.extend(format_non_binary_scores(index.get('non_binary_scores', {})))

    # Agent errors (crashes, rate limits)
    lines.extend(format_agent_errors(index))

    return lines


def format_binary_report_deep_focus(index: dict) -> list[str]:
    """Build binary-score error analysis lines for the deep focus (new vs baseline) report.

    Includes accuracy table, unique successes/errors, consensus errors, and mixed results.
    """
    from collections import defaultdict

    summary = index.get('summary', {})
    newest_agent = summary.get('new_agent', 'unknown')
    baseline_agents = summary.get('baseline_agents', [])
    by_agent = index.get('by_agent', {})
    accuracies = {name: stats.get('accuracy', 0) for name, stats in by_agent.items()}

    lines = []

    lines.append(f"**New Agent**: {newest_agent} ({accuracies.get(newest_agent, 0)}%)")
    if baseline_agents:
        baseline_strs = [f"{a} ({accuracies.get(a, 0)}%)" for a in baseline_agents]
        lines.append(f"**Baselines**: {', '.join(baseline_strs)}")
    lines.append("")

    # Cross-agent analysis
    cross_agent = index.get('cross_agent_analysis', {}).get('new_vs_baseline', {})
    unique_successes = cross_agent.get('unique_successes', [])
    unique_failures = cross_agent.get('unique_failures', cross_agent.get('unique_errors', []))
    consensus_failures = cross_agent.get('consensus_failures', cross_agent.get('consensus_errors', []))
    mixed = cross_agent.get('mixed_results', [])

    lines.extend([
        "## Summary",
        f"- Unique successes (new succeeded, all baselines failed): {len(unique_successes)}",
        f"- Unique failures (new failed, all baselines succeeded): {len(unique_failures)}",
        f"- Consensus failures (all failed): {len(consensus_failures)}",
        f"- Mixed results: {len(mixed)}",
        ""
    ])

    # Per-agent accuracy table
    if by_agent:
        all_agents = [newest_agent] + baseline_agents
        lines.extend([
            "## Agent Accuracy",
            "",
            "| Agent | Correct | Failed | Errors | Accuracy |",
            "|-------|---------|--------|--------|----------|"
        ])
        for agent in all_agents:
            stats = by_agent.get(agent, {})
            correct = stats.get('total_correct', 0)
            failed = stats.get('total_failed', stats.get('total_errors', 0))
            errors = stats.get('total_errors', 0)
            accuracy = stats.get('accuracy', 0.0)
            lines.append(f"| {agent} | {correct} | {failed} | {errors} | {accuracy:.1f}% |")
        lines.append("")

    # Unique Successes
    if unique_successes:
        count = len(unique_successes)
        id_str = ', '.join(unique_successes[:10]) + (', ...' if count > 10 else '')
        lines.extend([
            "## Unique Successes",
            "",
            f"New agent succeeded but all baselines failed ({count}): {id_str}",
            ""
        ])

    # Unique Failures
    if unique_failures:
        count = len(unique_failures)
        id_str = ', '.join(unique_failures[:10]) + (', ...' if count > 10 else '')
        lines.extend([
            "## Unique Failures",
            "",
            f"New agent failed but all baselines succeeded ({count}): {id_str}",
            ""
        ])

    # Consensus failures
    if consensus_failures:
        count = len(consensus_failures)
        id_str = ', '.join(consensus_failures[:10]) + (', ...' if count > 10 else '')
        lines.extend([
            "## Consensus Failures",
            "",
            f"New agent AND all baselines failed ({count}): {id_str}",
            ""
        ])

    # Mixed results
    if mixed:
        lines.extend([
            "## Mixed Results",
            "",
            f"Total mixed results: {len(mixed)}",
            ""
        ])

        pattern_groups = defaultdict(list)
        mixed_data = cross_agent.get('mixed_results', {})
        for question_id, split_info in mixed_data.items():
            new_agent_correct = split_info.get('new_agent_correct', None)
            correct_baselines = tuple(sorted(split_info.get('baseline_correct', [])))
            wrong_baselines = tuple(sorted(split_info.get('baseline_wrong', [])))
            pattern_groups[(new_agent_correct, correct_baselines, wrong_baselines)].append(question_id)

        sorted_patterns = sorted(
            pattern_groups.items(),
            key=lambda x: (-len(x[1]), x[0])
        )

        for (new_agent_correct, correct_baselines, wrong_baselines), question_ids in sorted_patterns:
            new_agent_status = "✅ NEW" if new_agent_correct else "❌ NEW"
            correct_str = ', '.join(correct_baselines)
            wrong_str = ', '.join(wrong_baselines)
            question_ids_str = ', '.join(f"**{qid}**" for qid in sorted(question_ids))
            lines.append(f"- {new_agent_status} | ✓ {correct_str} | ✗ {wrong_str}: {question_ids_str}")

        lines.append("")

    # Non-binary scores
    lines.extend(format_non_binary_scores(index.get('non_binary_scores', {})))

    # Agent errors
    lines.extend(format_agent_errors(index))

    return lines


# ---------------------------------------------------------------------------
# Continuous-score error analysis report
# ---------------------------------------------------------------------------

def _ranking_str(agent_scores: list[tuple[str, float]]) -> str:
    """Build ranking string like 'A > B = C' from sorted (agent, score) pairs.

    Input must be sorted descending by score.
    """
    if not agent_scores:
        return ""

    parts = [agent_scores[0][0]]
    for i in range(1, len(agent_scores)):
        prev_score = agent_scores[i - 1][1]
        curr_score = agent_scores[i][1]
        separator = " = " if curr_score == prev_score else " > "
        parts.append(separator + agent_scores[i][0])
    return "".join(parts)


def _compute_agent_totals(scores_by_question: dict, agents: list[str]) -> dict[str, list[float]]:
    """Collect per-agent score lists from scores_by_question."""
    agent_totals: dict[str, list[float]] = {a: [] for a in agents}
    for qid, agent_scores in scores_by_question.items():
        for agent in agents:
            if agent in agent_scores:
                agent_totals[agent].append(agent_scores[agent])
    return agent_totals


def _format_score_summary(agent_totals: dict[str, list[float]], agents: list[str],
                          new_agent: str | None = None) -> list[str]:
    """Build the score summary table. Marks new_agent if provided."""
    lines = [
        "## Score Summary",
        "",
        "| Agent | Mean Score | Problems |",
        "|-------|-----------|----------|",
    ]
    for agent in agents:
        scores = agent_totals[agent]
        mean = sum(scores) / len(scores) if scores else 0.0
        marker = " **(new)**" if agent == new_agent else ""
        lines.append(f"| {agent}{marker} | {mean:.3f} | {len(scores)} |")
    lines.append("")
    return lines


def _format_score_rows(scores_by_question: dict, agents: list[str],
                       agent_labels: list[str],
                       extra_col_headers: list[str],
                       extra_cols_fn) -> list[str]:
    """Build the score comparison table rows with pluggable extra columns.

    extra_cols_fn(scored_sorted, agent_scores) -> (sort_key, [display_str, ...])
    where scored_sorted is [(agent, score)] descending and the display list
    has one entry per extra_col_headers.
    """
    rows = []
    for qid, agent_scores in scores_by_question.items():
        scored = [(a, agent_scores.get(a, float('-inf'))) for a in agents]
        scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
        sort_key, displays = extra_cols_fn(scored_sorted, agent_scores)
        rows.append((sort_key, qid, agent_scores, displays))

    rows.sort(key=lambda r: r[0])

    # Table header
    lines = ["## Score Comparison", ""]
    header = "| Problem |"
    separator = "|---------|"
    for label in agent_labels:
        header += f" {label} |"
        separator += "--------|"
    for col_header in extra_col_headers:
        header += f" {col_header} |"
        separator += "---------|"
    lines.append(header)
    lines.append(separator)

    for _, qid, agent_scores, displays in rows:
        row = f"| {qid} |"
        for agent in agents:
            score = agent_scores.get(agent)
            row += f" {score:.3f} |" if score is not None else " — |"
        for display in displays:
            row += f" {display} |"
        lines.append(row)

    lines.append("")
    return lines


def format_continuous_score_table(scores_by_question: dict, agents: list[str]) -> list[str]:
    """Build score comparison table for continuous-score tasks (iteration report).

    Sorted by delta(best-worst) descending — most differentiated problems first.
    """
    if not scores_by_question or not agents:
        return []

    agent_totals = _compute_agent_totals(scores_by_question, agents)
    agent_means = {
        a: (sum(scores) / len(scores) if scores else 0.0)
        for a, scores in agent_totals.items()
    }
    sorted_agents = sorted(agents, key=lambda a: agent_means[a], reverse=True)

    lines = _format_score_summary(agent_totals, sorted_agents)

    # Solo wins and solo losses per agent (with problem IDs)
    win_ids: dict[str, list[str]] = {a: [] for a in sorted_agents}
    loss_ids: dict[str, list[str]] = {a: [] for a in sorted_agents}
    for qid, agent_scores in scores_by_question.items():
        scored = [(a, agent_scores.get(a, float('-inf'))) for a in sorted_agents]
        best_score = max(s for _, s in scored)
        worst_score = min(s for _, s in scored)
        winners = [a for a, s in scored if round(s, 6) == round(best_score, 6)]
        losers = [a for a, s in scored if round(s, 6) == round(worst_score, 6)]
        if len(winners) == 1:
            win_ids[winners[0]].append(qid)
        if len(losers) == 1 and worst_score < best_score:
            loss_ids[losers[0]].append(qid)

    for agent in sorted_agents:
        wins = win_ids[agent]
        wins_str = ', '.join(wins) if wins else ''
        lines.append(f"- **Solo wins**: {agent}: {len(wins)}" + (f" ({wins_str})" if wins else ""))
    for agent in sorted_agents:
        losses = loss_ids[agent]
        losses_str = ', '.join(losses) if losses else ''
        lines.append(f"- **Solo losses**: {agent}: {len(losses)}" + (f" ({losses_str})" if losses else ""))
    lines.append("")

    def delta_cols(scored_sorted, agent_scores):
        scores_list = [s for _, s in scored_sorted]
        best = scores_list[0]
        worst = scores_list[-1]
        delta_bw = best - worst
        delta_12 = (scores_list[0] - scores_list[1]) if len(scores_list) >= 2 else 0.0
        # Sort descending by delta(best-worst) => negate for ascending sort
        sort_key = -delta_bw
        return sort_key, [f"{delta_bw:.3f}", f"{delta_12:.3f}"]

    lines.extend(_format_score_rows(
        scores_by_question, sorted_agents, sorted_agents,
        ["Δ(best-worst)", "Δ(#1-#2)"], delta_cols,
    ))
    return lines


def format_continuous_score_table_deep_focus(
    scores_by_question: dict,
    new_agent: str,
    baseline_agents: list[str],
) -> list[str]:
    """Build score comparison table for continuous-score tasks (deep focus report).

    Shows new-vs-best-baseline delta and new agent's rank.
    Sorted by new-vs-best ascending (worst problems for new agent first).
    """
    if not scores_by_question:
        return []

    all_agents = [new_agent] + baseline_agents
    agent_totals = _compute_agent_totals(scores_by_question, all_agents)

    lines = _format_score_summary(agent_totals, all_agents, new_agent=new_agent)

    agent_labels = [f"{a} (new)" if a == new_agent else a for a in all_agents]

    # Track wins/losses for summary
    new_wins: list[str] = []
    new_losses: list[str] = []

    def new_vs_best_cols(scored_sorted, agent_scores):
        new_score = agent_scores.get(new_agent, float('-inf'))
        baseline_scores = [agent_scores.get(a, float('-inf')) for a in baseline_agents]
        best_baseline = max(baseline_scores) if baseline_scores else float('-inf')
        delta = new_score - best_baseline

        # Compute rank
        new_rank = 1
        for a, s in scored_sorted:
            if a == new_agent:
                break
            if s > new_score:
                new_rank += 1
        at_same_score = [a for a, s in scored_sorted if s == new_score]
        rank_str = f"#{new_rank} (tied)" if len(at_same_score) > 1 else f"#{new_rank}"

        delta_str = f"{delta:+.3f}"
        # Sort ascending by delta (worst problems first)
        sort_key = (delta, -new_score)
        return sort_key, [delta_str, rank_str]

    # Pre-compute wins/losses before building the table
    for qid, agent_scores in scores_by_question.items():
        new_score = agent_scores.get(new_agent, float('-inf'))
        baseline_scores = [agent_scores.get(a, float('-inf')) for a in baseline_agents]
        best_baseline = max(baseline_scores) if baseline_scores else float('-inf')
        worst_all = min([new_score] + baseline_scores)
        best_all = max([new_score] + baseline_scores)
        if new_score > best_baseline and new_score == best_all:
            scored = [(a, agent_scores.get(a, float('-inf'))) for a in all_agents]
            winners = [a for a, s in scored if s == best_all]
            if len(winners) == 1:
                new_wins.append(qid)
        if new_score == worst_all and new_score < best_all:
            scored = [(a, agent_scores.get(a, float('-inf'))) for a in all_agents]
            losers = [a for a, s in scored if s == worst_all]
            if len(losers) == 1:
                new_losses.append(qid)

    lines.extend(_format_score_rows(
        scores_by_question, all_agents, agent_labels,
        ["New vs Best", "New Rank"], new_vs_best_cols,
    ))

    # Summary lines
    if new_wins:
        ids_str = ', '.join(new_wins[:10]) + (', ...' if len(new_wins) > 10 else '')
        lines.append(f"- **New agent wins**: {len(new_wins)} problems where new beat all baselines ({ids_str})")
    else:
        lines.append(f"- **New agent wins**: 0 problems where new beat all baselines")
    if new_losses:
        ids_str = ', '.join(new_losses[:10]) + (', ...' if len(new_losses) > 10 else '')
        lines.append(f"- **New agent losses**: {len(new_losses)} problems where new was worst ({ids_str})")
    else:
        lines.append(f"- **New agent losses**: 0 problems where new was worst")
    lines.append("")

    return lines


def _test_refine_sort_key(key: str) -> int:
    """Extract numeric suffix from test_refine_N keys for sorting."""
    try:
        return int(key.split('_')[-1])
    except ValueError:
        return 0


def _collect_test_refine_keys(iteration_claude_costs: list) -> list[str]:
    """Collect test_refine_* keys that have nonzero cost, sorted numerically."""
    all_keys = set()
    for ic in iteration_claude_costs:
        evo_bd = ic.get('evolution_breakdown') or {}
        for key in evo_bd:
            if key.startswith('test_refine_'):
                all_keys.add(key)
    return [
        key for key in sorted(all_keys, key=_test_refine_sort_key)
        if any(
            (ic.get('evolution_breakdown') or {}).get(key, {}).get('cost', 0.0) > 0
            for ic in iteration_claude_costs
        )
    ]


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

        # Task Configuration (from task defaults + --task-config overrides)
        task_config = getattr(self.researcher, 'task_config', {})
        if task_config:
            report_lines.append("### Task Configuration")
            report_lines.append("")
            report_lines.append("Task-specific parameters (from task defaults + --task-config overrides):")
            report_lines.append("")
            for param_name in sorted(task_config.keys()):
                value = task_config[param_name]
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

            # Build clone lookup for winner exclusion
            wins_clone_by_iter = {}
            for clone_id, _matched_id, clone_iter in getattr(self.researcher, 'clone_detections', []):
                wins_clone_by_iter.setdefault(clone_iter, set()).add(clone_id)

            # Count wins for each agent
            for i, test_results in enumerate(self.researcher.test_history):
                iteration_num = i + 1
                # Find winner(s) of this iteration, excluding clones
                if test_results:
                    iter_clones = wins_clone_by_iter.get(iteration_num, set())
                    eligible = {k: v for k, v in test_results.items() if k not in iter_clones}
                    if not eligible:
                        eligible = test_results
                    max_score = max(eligible[k]['average_score'] for k in eligible.keys())
                    winners = [k for k in eligible.keys() if eligible[k]['average_score'] == max_score]

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
            clone_agent_ids = set(c for c, _m, _i in getattr(self.researcher, 'clone_detections', []))
            ranking_table = self.researcher.evolver._generate_ranking_table(
                self.researcher.test_history,
                self.researcher.performance_records,
                for_evolution=False,
                clone_agent_ids=clone_agent_ids
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

        # Build clone lookup by iteration
        clone_by_iteration = {}
        for clone_id, _matched_id, clone_iter in getattr(self.researcher, 'clone_detections', []):
            clone_by_iteration.setdefault(clone_iter, []).append(clone_id)

        for i, test_results in enumerate(self.researcher.test_history):
            iteration_num = i + 1

            # Find winner(s), excluding clones
            if test_results:
                iter_clones = set(clone_by_iteration.get(iteration_num, []))
                eligible = {k: v for k, v in test_results.items() if k not in iter_clones}
                if not eligible:
                    eligible = test_results  # Fallback (shouldn't happen)
                max_score = max(eligible[k]['average_score'] for k in eligible.keys())
                winners = [k for k in eligible.keys() if eligible[k]['average_score'] == max_score]
                winner_str = ', '.join(winners) if len(winners) <= 2 else f"{winners[0]} +{len(winners)-1}"
                if iter_clones:
                    winner_str += f" ({', '.join(f'{c}*' for c in iter_clones)} excluded)"
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

        if clone_by_iteration:
            report_lines.append("")
            report_lines.append("\\* *Exact clone: identical per-problem scores to an existing agent on debut. "
                              "ELO penalized by 200; excluded from winner selection.*")

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
            avg_evo_time = (avg_first_draft +
                           sum(test_round_averages.values()) + avg_reflection)

            # Calculate percentages of total evolution time
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
            report_lines.append(f"| First Draft (Round 1) | {avg_first_draft/60:.1f}m | {pct_first_draft:.1f}% |")

            # Dynamic test round rows
            for i, key in enumerate(sorted(test_round_averages.keys(), key=_test_refine_sort_key), start=1):
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
            # Calculate totals — group DF eval costs with evaluation (not evolution)
            # so that the summary ties out with the Detailed Per-Iteration table
            total_eval_cost = 0.0
            total_evolution_cost = 0.0
            for ic in self.researcher.iteration_claude_costs:
                eval_base = ic.get('eval_cost', 0.0)
                breakdown = ic.get('evolution_breakdown')
                test_eval = 0.0
                first_draft = 0.0
                test_evo = 0.0
                reflection = 0.0
                if breakdown:
                    first_draft = breakdown.get('first_draft', {}).get('cost', 0.0)
                    for key in breakdown:
                        if key.startswith('test_refine_'):
                            test_eval += breakdown[key].get('eval', {}).get('cost', 0.0)
                            test_evo += breakdown[key].get('evolution', {}).get('cost', 0.0)
                    reflection = breakdown.get('reflection', {}).get('cost', 0.0)
                    total_eval_cost += eval_base + test_eval
                    total_evolution_cost += first_draft + test_evo + reflection
                else:
                    total_eval_cost += eval_base
                    total_evolution_cost += ic.get('evolution_cost', 0.0)
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

            # Detailed per-iteration costs
            report_lines.append("\n### Detailed Per-Iteration Costs\n")

            # Detect DF rounds from data (sorted numerically, zero-cost keys pruned)
            sorted_test_round_keys = _collect_test_refine_keys(self.researcher.iteration_claude_costs)
            num_df_rounds = len(sorted_test_round_keys)

            # Check if we have fresh eval data
            has_fresh = hasattr(self.researcher, 'iteration_fresh_evals') and self.researcher.iteration_fresh_evals

            # Build dynamic header
            header_cols = ["Iter", "Total", "**Eval**", "**Evo**", "Iter Eval", "1st Draft"]
            for i in range(num_df_rounds):
                round_label = f"DF Eval {i+1}" if num_df_rounds > 1 else "DF Eval"
                draft_label = f"Draft {i+2}"
                header_cols.extend([round_label, draft_label])
            header_cols.extend(["Refl", "Meta"])
            if has_fresh:
                header_cols.append("Fresh")
            header_cols.extend(["Strategy", "Meta-strategy"])

            header = "| " + " | ".join(header_cols) + " |"
            separator = "|" + "|".join(["------"] * len(header_cols)) + "|"
            report_lines.append(header)
            report_lines.append(separator)

            # Accumulators for totals
            totals = {
                'total_cost': 0.0,
                'total_eval': 0.0,
                'total_evolution': 0.0,
                'first_draft': 0.0,
                'reflection': 0.0,
                'meta_evolution': 0.0,
            }
            # Per-round accumulators
            round_totals_eval = [0.0] * num_df_rounds
            round_totals_draft = [0.0] * num_df_rounds
            total_fresh = 0

            def fmt(val):
                return f"${val:.2f}" if val > 0 else "-"

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
                evolution_breakdown = cost_dict.get('evolution_breakdown')

                # Extract costs dynamically
                first_draft = 0.0
                reflection = 0.0
                round_evals = [0.0] * num_df_rounds
                round_drafts = [0.0] * num_df_rounds

                if evolution_breakdown:
                    first_draft = evolution_breakdown.get('first_draft', {}).get('cost', 0.0)
                    for i, key in enumerate(sorted_test_round_keys):
                        round_data = evolution_breakdown.get(key, {})
                        round_evals[i] = round_data.get('eval', {}).get('cost', 0.0)
                        round_drafts[i] = round_data.get('evolution', {}).get('cost', 0.0)
                    reflection = evolution_breakdown.get('reflection', {}).get('cost', 0.0)

                meta_evolution_cost = cost_dict.get('meta_evolution_cost', 0.0)

                total_eval_iter = eval_base + sum(round_evals)
                total_evolution_iter = first_draft + sum(round_drafts) + reflection
                total_cost = total_eval_iter + total_evolution_iter + meta_evolution_cost

                totals['total_cost'] += total_cost
                totals['total_eval'] += total_eval_iter
                totals['total_evolution'] += total_evolution_iter
                totals['first_draft'] += first_draft
                totals['reflection'] += reflection
                totals['meta_evolution'] += meta_evolution_cost
                for i in range(num_df_rounds):
                    round_totals_eval[i] += round_evals[i]
                    round_totals_draft[i] += round_drafts[i]

                # Fresh evals
                fresh_val = 0
                if has_fresh and idx < len(self.researcher.iteration_fresh_evals):
                    fresh_val = self.researcher.iteration_fresh_evals[idx]
                    total_fresh += fresh_val

                # Build row
                row_vals = [f"**{iter_num}**", f"${total_cost:.2f}",
                            f"**{fmt(total_eval_iter)}**", f"**{fmt(total_evolution_iter)}**",
                            fmt(eval_base), fmt(first_draft)]
                for i in range(num_df_rounds):
                    row_vals.extend([fmt(round_evals[i]), fmt(round_drafts[i])])
                row_vals.extend([fmt(reflection), fmt(meta_evolution_cost)])
                if has_fresh:
                    row_vals.append(str(fresh_val) if fresh_val > 0 else "-")
                row_vals.extend([strategy_display, meta_strategy_display])

                report_lines.append("| " + " | ".join(row_vals) + " |")

            # Totals row
            report_lines.append(separator)
            total_vals = ["**TOTAL**", f"**${totals['total_cost']:.2f}**",
                          f"**${totals['total_eval']:.2f}**", f"**${totals['total_evolution']:.2f}**",
                          f"${totals['total_eval'] - sum(round_totals_eval):.2f}",
                          f"${totals['first_draft']:.2f}"]
            for i in range(num_df_rounds):
                total_vals.extend([f"${round_totals_eval[i]:.2f}", f"${round_totals_draft[i]:.2f}"])
            total_vals.extend([f"${totals['reflection']:.2f}", f"${totals['meta_evolution']:.2f}"])
            if has_fresh:
                total_vals.append(f"**{total_fresh}**")
            total_vals.extend(["-", "-"])
            report_lines.append("| " + " | ".join(total_vals) + " |")

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

                    # Test rounds - sorted numerically, zero-cost keys pruned
                    test_round_keys = _collect_test_refine_keys(self.researcher.iteration_claude_costs)

                    test_round_stats = {}
                    for key in test_round_keys:
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

                    for key in test_round_keys:
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

                    for key in test_round_keys:
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

                    for i, key in enumerate(test_round_keys, start=1):
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
            report_lines.append(f"Total zero accuracy cases: {len(self.researcher.zero_accuracy_cases)}")
            report_lines.append("\n| Agent | Iteration | Problems Tested |")
            report_lines.append("|-------|-----------|-----------------|")
            for entry in sorted(self.researcher.zero_accuracy_cases):
                # Support both 3-tuple (agent, iter, total) and legacy 4-tuple (agent, None, iter, total)
                agent_id, iteration_num, total_q = entry[0], entry[-2], entry[-1]
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