"""
Comparison report generation for Deep Focus evolution.

Generates reports comparing a new agent's performance against
historical agents from prior iterations, including per-database
breakdowns and actionable insights.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple


class ComparisonReportGenerator:
    """Generates comparison reports for Deep Focus testing rounds."""

    def __init__(self, experiment_dir: Path):
        """
        Initialize comparison report generator.

        Args:
            experiment_dir: Root directory of the research experiment
        """
        self.experiment_dir = experiment_dir

    def generate_comparison_report(
        self,
        test_iteration: int,
        new_agent_workspace: Path,
        databases: List[str],
        output_path: Path
    ) -> str:
        """
        Generate comparison report showing new agent vs historical agents.

        Args:
            test_iteration: Iteration number to compare against
            new_agent_workspace: Path to new agent's test workspace
            databases: List of databases tested
            output_path: Where to write the comparison report

        Returns:
            Generated report as markdown string
        """
        # Load new agent results
        new_agent_results = self._load_agent_results(new_agent_workspace, databases)

        # Load historical agent results from target iteration
        iteration_dir = self.experiment_dir / f"iteration_{test_iteration:03d}"
        historical_agents = self._load_iteration_agents(iteration_dir, databases)

        # Generate report
        report = self._build_report(
            test_iteration=test_iteration,
            new_agent_results=new_agent_results,
            historical_agents=historical_agents,
            databases=databases
        )

        # Write report
        output_path.write_text(report)
        return report

    def _load_agent_results(
        self,
        workspace: Path,
        databases: List[str]
    ) -> Dict:
        """
        Load agent results from a workspace.

        Args:
            workspace: Path to agent workspace
            databases: List of databases to load

        Returns:
            Dict with agent performance metrics
        """
        results = {
            'databases': databases,
            'by_database': {},
            'by_database_correct': {},
            'by_database_total': {},
            'total_questions': 0,
            'correct': 0,
            'phase1_failures': 0,
            'phase2_failures': 0
        }

        for db_name in databases:
            eval_file = workspace / db_name / "results" / "evaluation.json"

            if eval_file.exists():
                try:
                    with open(eval_file, 'r') as f:
                        db_data = json.load(f)

                    questions = db_data.get('total_questions', 0)
                    correct = db_data.get('correct', 0)
                    accuracy = (correct / questions * 100) if questions > 0 else 0.0

                    results['by_database'][db_name] = accuracy
                    results['by_database_correct'][db_name] = correct
                    results['by_database_total'][db_name] = questions
                    results['total_questions'] += questions
                    results['correct'] += correct

                except Exception as e:
                    print(f"Warning: Could not load {eval_file}: {e}")
                    # Treat as Phase 2 failure
                    results['phase2_failures'] += 1
            else:
                # Check if Phase 1 failed (no results directory)
                if not (workspace / db_name / "results").exists():
                    results['phase1_failures'] += 1
                else:
                    results['phase2_failures'] += 1

        # Calculate overall accuracy
        if results['total_questions'] > 0:
            results['overall_accuracy'] = (results['correct'] / results['total_questions']) * 100
        else:
            results['overall_accuracy'] = 0.0

        return results

    def _load_iteration_agents(
        self,
        iteration_dir: Path,
        databases: List[str]
    ) -> Dict[str, Dict]:
        """
        Load all agent results from an iteration.

        Args:
            iteration_dir: Path to iteration directory
            databases: List of databases to load

        Returns:
            Dict mapping agent_name -> agent results
        """
        agents = {}

        # Find all agent directories
        for agent_dir in iteration_dir.iterdir():
            if agent_dir.is_dir() and agent_dir.name.startswith('agent_'):
                agent_name = agent_dir.name.replace('agent_', '')
                agent_results = self._load_agent_results(agent_dir, databases)
                agents[agent_name] = agent_results

        return agents

    def _build_report(
        self,
        test_iteration: int,
        new_agent_results: Dict,
        historical_agents: Dict[str, Dict],
        databases: List[str]
    ) -> str:
        """
        Build comparison report markdown.

        Args:
            test_iteration: Iteration number being compared against
            new_agent_results: New agent's performance metrics
            historical_agents: Dict of historical agent metrics
            databases: List of databases tested

        Returns:
            Markdown report string
        """
        lines = []

        # Header
        lines.append(f"# Iteration {test_iteration} Comparison Report\n")

        # Overall Rankings section
        lines.append("## Overall Rankings\n")

        # Build comparison data
        comparison_data = []
        new_agent_accuracy = new_agent_results['overall_accuracy']
        new_agent_count = new_agent_results['total_questions']

        for agent_name, agent_results in historical_agents.items():
            comparison_data.append({
                'agent': agent_name,
                'accuracy': agent_results['overall_accuracy'],
                'questions': agent_results['total_questions'],
                'delta': new_agent_accuracy - agent_results['overall_accuracy'],
                'by_database': agent_results['by_database']
            })

        # Sort by accuracy (descending)
        comparison_data.sort(key=lambda x: x['accuracy'], reverse=True)

        # Calculate new agent rank
        new_agent_rank = sum(1 for x in comparison_data if x['accuracy'] > new_agent_accuracy) + 1

        # Generate rankings table
        lines.append("| Rank | Agent | Accuracy | Questions | Delta vs New |")
        lines.append("|------|-------|----------|-----------|--------------|")

        rank = 1
        new_agent_inserted = False

        for i, data in enumerate(comparison_data):
            # Insert new agent at appropriate position
            if not new_agent_inserted and rank == new_agent_rank:
                lines.append(
                    f"| {rank} | **NEW AGENT** | **{new_agent_accuracy:.1f}%** | "
                    f"**{new_agent_count}** | **baseline** |"
                )
                rank += 1
                new_agent_inserted = True

            delta_str = f"+{data['delta']:.1f}%" if data['delta'] > 0 else f"{data['delta']:.1f}%"
            arrow = "⬆️" if data['delta'] > 0 else "⬇️" if data['delta'] < 0 else ""
            lines.append(
                f"| {rank} | {data['agent']} | {data['accuracy']:.1f}% | "
                f"{data['questions']} | {delta_str} {arrow} |"
            )
            rank += 1

        # If new agent is last, insert it now
        if not new_agent_inserted:
            lines.append(
                f"| {rank} | **NEW AGENT** | **{new_agent_accuracy:.1f}%** | "
                f"**{new_agent_count}** | **baseline** |"
            )

        # Performance by Database section
        lines.append("\n## Performance by Database\n")

        # Build header
        header = "| Agent | " + " | ".join(databases) + " | Overall |"
        lines.append(header)
        separator = "|" + "---|" * (len(databases) + 2)
        lines.append(separator)

        # Add top 3 agents
        for data in comparison_data[:3]:
            row = f"| {data['agent']} | "
            for db in databases:
                db_acc = data['by_database'].get(db, 0.0)
                row += f"{db_acc:.1f}% | "
            row += f"{data['accuracy']:.1f}% |"
            lines.append(row)

        # Add new agent row
        row = f"| **NEW AGENT** | "
        for db in databases:
            db_acc = new_agent_results['by_database'].get(db, 0.0)
            row += f"**{db_acc:.1f}%** | "
        row += f"**{new_agent_accuracy:.1f}%** |"
        lines.append(row)

        # Key Insights section
        lines.append("\n### Key Insights for Analysis")
        lines.append("- Compare NEW AGENT performance across databases to identify strengths/weaknesses")
        lines.append("- Focus on databases where NEW AGENT shows significant performance differences vs top agents")
        lines.append("- Review evaluation.json files for detailed error patterns in challenging databases")

        # Detailed Results Location section
        lines.append("\n### Detailed Results Location\n")
        lines.append("```")
        lines.append(f"./iteration_{test_iteration:03d}_test/")

        for db in databases:
            db_correct = new_agent_results['by_database_correct'].get(db, 0)
            db_total = new_agent_results['by_database_total'].get(db, 0)
            db_acc = (db_correct / db_total * 100) if db_total > 0 else 0.0
            lines.append(f"  {db}/")
            lines.append(f"    evaluations.json          ← {db_correct}/{db_total} correct ({db_acc:.1f}%)")
            lines.append(f"    output/system_prompt.txt  ← Agent's database analysis")

        lines.append("```\n")
        lines.append("Review evaluations.json for per-question results (evaluated by result sets, not SQL) and system_prompt.txt for agent's database analysis.")

        # Assessment section
        # delta = new_agent_accuracy - historical_agent_accuracy
        # So delta > 0 means new agent won, delta < 0 means new agent lost
        beat_count = sum(1 for x in comparison_data if x['delta'] > 0)
        beat_percentage = (beat_count / len(comparison_data) * 100) if comparison_data else 0

        lines.append("\n## Assessment")
        if new_agent_rank == 1:
            lines.append(f"✅ NEW AGENT LEADS iteration {test_iteration} (new champion!)")
        elif beat_percentage >= 50:
            lines.append(f"✅ NEW AGENT beats {beat_percentage:.0f}% of iteration {test_iteration} agents")
        else:
            lines.append(f"❌ NEW AGENT underperforms most iteration {test_iteration} agents")

        # Add specific strengths/weaknesses if available
        if comparison_data:
            champion = comparison_data[0]
            champion_by_db = champion['by_database']
            new_by_db = new_agent_results['by_database']

            # Find biggest gaps
            strengths = []
            weaknesses = []

            for db in databases:
                if db in champion_by_db and db in new_by_db:
                    delta = new_by_db[db] - champion_by_db[db]
                    if delta >= 5.0:  # Strength threshold
                        strengths.append((db, delta))
                    elif delta <= -5.0:  # Weakness threshold
                        weaknesses.append((db, delta))

            if weaknesses:
                weaknesses.sort(key=lambda x: x[1])  # Sort by delta (most negative first)
                weakness_str = ", ".join([f"{db} ({delta:.1f}% vs champion)" for db, delta in weaknesses[:3]])
                lines.append(f"⚠️  Key weaknesses: {weakness_str}")

            if strengths:
                strengths.sort(key=lambda x: x[1], reverse=True)  # Sort by delta (most positive first)
                strength_str = ", ".join([f"{db} (+{delta:.1f}% vs champion)" for db, delta in strengths[:3]])
                lines.append(f"💡 Key strengths: {strength_str}")

        return '\n'.join(lines)
