"""
Comparison report generation for Deep Focus evolution.

Generates reports comparing a new agent's performance against
historical agents from prior iterations, including per-context
breakdowns and actionable insights.

Works with any domain by using the domain's load_agent_results() method.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from RoboPhD.domains.base import DomainInterface


class ComparisonReportGenerator:
    """Generates comparison reports for Deep Focus testing rounds."""

    def __init__(self, experiment_dir: Path, domain: Optional['DomainInterface'] = None):
        """
        Initialize comparison report generator.

        Args:
            experiment_dir: Root directory of the research experiment
            domain: Optional domain interface for loading results.
                   If provided, uses domain.load_agent_results().
                   If None, falls back to direct file reading (Text2SQL format).
        """
        self.experiment_dir = experiment_dir
        self.domain = domain

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
            contexts=databases  # Use 'contexts' parameter name for domain-agnostic support
        )

        # Write report
        output_path.write_text(report)
        return report

    def _load_agent_results(
        self,
        workspace: Path,
        contexts: List[str]
    ) -> Dict:
        """
        Load agent results from a workspace.

        Args:
            workspace: Path to agent workspace
            contexts: List of contexts (databases for Text2SQL, problem IDs for CodeGen)

        Returns:
            Dict with agent performance metrics
        """
        # Use domain interface if available
        if self.domain is not None:
            domain_results = self.domain.load_agent_results(workspace, contexts)

            # Use consistent 'by_context' naming
            results = {
                'contexts': contexts,
                'by_context': {},
                'by_context_correct': {},
                'by_context_total': {},
                'total_questions': domain_results.get('total_questions', 0),
                'correct': domain_results.get('correct', 0),
                'overall_accuracy': domain_results.get('overall_accuracy', 0.0),
                'phase1_failures': 0,
                'phase2_failures': 0,
            }

            for ctx_name, ctx_data in domain_results.get('by_context', {}).items():
                results['by_context'][ctx_name] = ctx_data.get('accuracy', 0.0)
                results['by_context_correct'][ctx_name] = ctx_data.get('correct', 0)
                results['by_context_total'][ctx_name] = ctx_data.get('total', 0)

            return results

        # Fallback: direct file reading (Text2SQL format)
        results = {
            'contexts': contexts,
            'by_context': {},
            'by_context_correct': {},
            'by_context_total': {},
            'total_questions': 0,
            'correct': 0,
            'phase1_failures': 0,
            'phase2_failures': 0
        }

        for ctx_name in contexts:
            eval_file = workspace / ctx_name / "results" / "evaluation.json"

            if eval_file.exists():
                try:
                    with open(eval_file, 'r') as f:
                        ctx_data = json.load(f)

                    questions = ctx_data.get('total_questions', 0)
                    correct = ctx_data.get('correct', 0)
                    accuracy = (correct / questions * 100) if questions > 0 else 0.0

                    results['by_context'][ctx_name] = accuracy
                    results['by_context_correct'][ctx_name] = correct
                    results['by_context_total'][ctx_name] = questions
                    results['total_questions'] += questions
                    results['correct'] += correct

                except Exception as e:
                    print(f"Warning: Could not load {eval_file}: {e}")
                    # Treat as Phase 2 failure
                    results['phase2_failures'] += 1
            else:
                # Check if Phase 1 failed (no results directory)
                if not (workspace / ctx_name / "results").exists():
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
        contexts: List[str]
    ) -> str:
        """
        Build comparison report markdown.

        Args:
            test_iteration: Iteration number being compared against
            new_agent_results: New agent's performance metrics
            historical_agents: Dict of historical agent metrics
            contexts: List of contexts tested (databases for Text2SQL, problem IDs for CodeGen)

        Returns:
            Markdown report string
        """
        # Use domain-specific terminology if available
        context_label = "Database"
        context_label_plural = "databases"
        is_hierarchical = True
        if self.domain is not None:
            context_label = self.domain.context_label
            context_label_plural = self.domain.context_label_plural
            is_hierarchical = self.domain.is_hierarchical

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
                'by_context': agent_results['by_context']
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

        # Performance by Context section (only for hierarchical domains with multiple contexts)
        if is_hierarchical and len(contexts) > 1:
            lines.append(f"\n## Performance by {context_label}\n")

            # Build header
            header = "| Agent | " + " | ".join(contexts) + " | Overall |"
            lines.append(header)
            separator = "|" + "---|" * (len(contexts) + 2)
            lines.append(separator)

            # Add top 3 agents
            for data in comparison_data[:3]:
                row = f"| {data['agent']} | "
                for ctx in contexts:
                    ctx_acc = data['by_context'].get(ctx, 0.0)
                    row += f"{ctx_acc:.1f}% | "
                row += f"{data['accuracy']:.1f}% |"
                lines.append(row)

            # Add new agent row
            row = f"| **NEW AGENT** | "
            for ctx in contexts:
                ctx_acc = new_agent_results['by_context'].get(ctx, 0.0)
                row += f"**{ctx_acc:.1f}%** | "
            row += f"**{new_agent_accuracy:.1f}%** |"
            lines.append(row)

        # Key Insights section
        lines.append("\n### Key Insights for Analysis")
        lines.append(f"- Compare NEW AGENT performance across {context_label_plural} to identify strengths/weaknesses")
        lines.append(f"- Focus on {context_label_plural} where NEW AGENT shows significant performance differences vs top agents")
        lines.append(f"- Review evaluation.json files for detailed error patterns in challenging {context_label_plural}")

        # Detailed Results Location section (only for hierarchical domains)
        if is_hierarchical:
            lines.append("\n### Detailed Results Location\n")
            lines.append("```")
            lines.append(f"./iteration_{test_iteration:03d}_test/")

            for ctx in contexts:
                ctx_correct = new_agent_results['by_context_correct'].get(ctx, 0)
                ctx_total = new_agent_results['by_context_total'].get(ctx, 0)
                ctx_acc = (ctx_correct / ctx_total * 100) if ctx_total > 0 else 0.0
                lines.append(f"  {ctx}/")
                lines.append(f"    evaluations.json          ← {ctx_correct}/{ctx_total} correct ({ctx_acc:.1f}%)")
                lines.append(f"    output/system_prompt.txt  ← Agent's {context_label.lower()} analysis")

            lines.append("```\n")
            lines.append(f"Review evaluations.json for per-question results and system_prompt.txt for agent's {context_label.lower()} analysis.")
        else:
            # For flat domains (CodeGen), just point to the evaluation.json
            lines.append("\n### Detailed Results Location\n")
            lines.append("```")
            lines.append(f"./iteration_{test_iteration:03d}_test/")
            lines.append("  evaluation.json  ← Per-problem results")
            lines.append("```")

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

        # Add specific strengths/weaknesses if available (only for hierarchical domains)
        if is_hierarchical and comparison_data:
            champion = comparison_data[0]
            champion_by_ctx = champion['by_context']
            new_by_ctx = new_agent_results['by_context']

            # Find biggest gaps
            strengths = []
            weaknesses = []

            for ctx in contexts:
                if ctx in champion_by_ctx and ctx in new_by_ctx:
                    delta = new_by_ctx[ctx] - champion_by_ctx[ctx]
                    if delta >= 5.0:  # Strength threshold
                        strengths.append((ctx, delta))
                    elif delta <= -5.0:  # Weakness threshold
                        weaknesses.append((ctx, delta))

            if weaknesses:
                weaknesses.sort(key=lambda x: x[1])  # Sort by delta (most negative first)
                weakness_str = ", ".join([f"{ctx} ({delta:.1f}% vs champion)" for ctx, delta in weaknesses[:3]])
                lines.append(f"⚠️  Key weaknesses: {weakness_str}")

            if strengths:
                strengths.sort(key=lambda x: x[1], reverse=True)  # Sort by delta (most positive first)
                strength_str = ", ".join([f"{ctx} (+{delta:.1f}% vs champion)" for ctx, delta in strengths[:3]])
                lines.append(f"💡 Key strengths: {strength_str}")

        return '\n'.join(lines)
