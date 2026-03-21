#!/usr/bin/env python3
"""
Build fast error index comparing newest agent vs baseline agents.

This tool:
1. Auto-detects the newest agent (by created_iteration or alphabetically)
2. Compares it against all other tested agents (baseline)
3. Builds global cross-agent analysis (new vs baseline)

Designed for efficiency - focuses on newest agent only.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

try:
    from RoboPhD.report_generator import is_continuous_scoring
except ImportError:
    # Fallback for standalone script usage
    def is_continuous_scoring(scores_by_question):
        all_scores = []
        for qid, agent_scores in scores_by_question.items():
            for agent, score in agent_scores.items():
                all_scores.append(score)
        if not all_scores:
            return False
        if any(s < 0.0 or s > 1.0 for s in all_scores):
            return True
        binary_count = sum(1 for s in all_scores if s == 0.0 or s == 1.0)
        return binary_count / len(all_scores) < 0.8


def strip_agent_prefix(agent_name: str) -> str:
    """Strip 'agent_' prefix from agent name for cleaner display."""
    return agent_name.replace('agent_', '', 1) if agent_name.startswith('agent_') else agent_name


def _find_eval_files(iteration_dir: Path) -> List[Path]:
    """
    Auto-detect evaluation files in an iteration directory.

    Tries hierarchical structure first (Text2SQL), falls back to flat (CodeGen, AIME).
    Warns when hierarchical files are found since per-database granularity is not preserved.
    """
    # Try hierarchical structure first (Text2SQL)
    eval_files = list(iteration_dir.glob("agent_*/*/evaluations/evaluation.json"))
    if not eval_files:
        eval_files = list(iteration_dir.glob("agent_*/*/results/evaluation.json"))

    if eval_files:
        print(f"Warning: Found hierarchical evaluation files in {iteration_dir}. "
              f"Per-database granularity will not be preserved in the error index.",
              file=sys.stderr)
        return eval_files

    # Flat structure (CodeGen, AIME)
    return list(iteration_dir.glob("agent_*/evaluation.json"))


def detect_newest_agent(iteration_dirs: List[Path], agents_dir: Path, new_agent: Optional[str] = None) -> Tuple[str, List[str]]:
    """
    Detect newest agent and baseline agents.

    Args:
        iteration_dirs: List of iteration directories to search
        agents_dir: Path to agents directory
        new_agent: Optional agent name to use as newest (if provided)

    Returns:
        (newest_agent, baseline_agents)
    """
    # Find all agents that were tested
    tested_agents = set()

    for iteration_dir in iteration_dirs:
        eval_files = _find_eval_files(iteration_dir)

        for eval_file in eval_files:
            agent_name = None
            for part in eval_file.parts:
                if part.startswith('agent_'):
                    agent_name = part
                    break
            if agent_name:
                tested_agents.add(agent_name)

    if not tested_agents:
        print("Error: No agents found in evaluation files", file=sys.stderr)
        sys.exit(1)

    # If new_agent was explicitly provided, use it
    if new_agent:
        # Ensure new_agent starts with 'agent_' prefix
        if not new_agent.startswith('agent_'):
            new_agent = f'agent_{new_agent}'

        if new_agent in tested_agents:
            baseline_agents = [a for a in tested_agents if a != new_agent]
            print(f"🎯 Newest agent (specified): {new_agent}", file=sys.stderr)
            print(f"📊 Baseline agents: {', '.join(baseline_agents)}", file=sys.stderr)
            return new_agent, baseline_agents
        else:
            print(f"Warning: Specified new agent '{new_agent}' not found in tested agents", file=sys.stderr)
            print(f"Will auto-detect from: {', '.join(tested_agents)}", file=sys.stderr)

    # Try to find newest by created_iteration
    agents_with_metadata = []
    for agent_id in tested_agents:
        agent_metadata_file = agents_dir / agent_id / "metadata.json"
        created_iter = -1

        if agent_metadata_file.exists():
            try:
                with open(agent_metadata_file, 'r') as f:
                    metadata = json.load(f)
                    created_iter = metadata.get('created_iteration', -1)
            except Exception:
                pass

        # Fallback: Try to extract iteration number from directory name
        if created_iter == -1:
            import re
            match = re.search(r'iter(\d+)', agent_id)
            if match:
                created_iter = int(match.group(1))

        agents_with_metadata.append((agent_id, created_iter))

    # Sort by (iter, name) descending
    agents_with_metadata.sort(key=lambda x: (x[1], x[0]), reverse=True)

    newest_agent = agents_with_metadata[0][0]
    baseline_agents = [agent_id for agent_id, _ in agents_with_metadata[1:]]

    print(f"🎯 Newest agent: {newest_agent}", file=sys.stderr)
    print(f"📊 Baseline agents: {', '.join(baseline_agents)}", file=sys.stderr)

    return newest_agent, baseline_agents


def load_evaluation_results(iteration_dirs: List[Path]) -> Dict:
    """
    Load all evaluation results from iteration directories.

    Results are flattened: the intermediate grouping layer (database name)
    is discarded since all active domains are flat.

    Args:
        iteration_dirs: List of iteration directories to process

    Returns:
        {
            'by_question': {question_id: {agent_name: result}},
            'by_agent': {agent_name: {question_id: result}}
        }
    """
    by_question = defaultdict(dict)
    by_agent = defaultdict(dict)

    for iteration_dir in iteration_dirs:
        eval_files = _find_eval_files(iteration_dir)

        for eval_file in eval_files:
            try:
                with open(eval_file, 'r') as f:
                    eval_data = json.load(f)
            except Exception as e:
                print(f"Warning: Could not read {eval_file}: {e}", file=sys.stderr)
                continue

            # Extract agent name from path
            agent_name = None
            for part in eval_file.parts:
                if part.startswith('agent_'):
                    agent_name = part
                    break

            if not agent_name:
                continue

            results = eval_data.get('results', {})
            # Handle case where results is a list (e.g., timeout with no predictions)
            if not isinstance(results, dict):
                continue
            for question_id, result in results.items():
                by_question[question_id][agent_name] = result
                by_agent[agent_name][question_id] = result

    return {
        'by_question': dict(by_question),
        'by_agent': dict(by_agent)
    }


def build_cross_agent_analysis(newest_agent: str, baseline_agents: List[str],
                               results: Dict, question_ids: Optional[Set[str]] = None) -> Dict:
    """
    Build cross-agent analysis for newest vs baseline agents.

    Args:
        newest_agent: Name of newest agent
        baseline_agents: List of baseline agent names
        results: Results dict from load_evaluation_results
        question_ids: Optional set to filter questions

    Returns:
        {
            'new_vs_baseline': {
                'unique_errors': [qids],      # New failed, all baselines succeeded
                'unique_successes': [qids],   # New succeeded, all baselines failed
                'consensus_errors': [qids],   # All failed (including new)
                'mixed_results': {qid: {'baseline_correct': [...], 'baseline_wrong': [...]}}
            },
            'new_vs_[agent]': {  // Pairwise for each baseline agent
                'new_wrong_A_right': [qids],
                'new_right_A_wrong': [qids],
                'both_wrong': [qids]
            }
        }
    """
    by_question = results['by_question']

    # Filter questions if specified
    if question_ids:
        filtered_questions = {qid: agents for qid, agents in by_question.items() if qid in question_ids}
    else:
        filtered_questions = by_question

    # Global: newest vs all baseline
    unique_successes = []  # New succeeded, all baselines failed
    unique_errors = []     # New failed, all baselines succeeded
    consensus_errors = []  # All failed (including new)
    mixed_results = {}

    for question_id, agents_results in filtered_questions.items():
        if newest_agent not in agents_results:
            continue

        new_result = agents_results[newest_agent]
        new_correct = new_result.get('score', 0) >= 0.5

        # Check baseline agents
        baseline_results = []
        for agent in baseline_agents:
            if agent in agents_results:
                baseline_correct = agents_results[agent].get('score', 0) >= 0.5
                baseline_results.append((agent, baseline_correct))

        if not baseline_results:
            continue

        # Count how many baseline agents got it right
        baseline_correct_agents = [agent for agent, correct in baseline_results if correct]
        baseline_wrong_agents = [agent for agent, correct in baseline_results if not correct]

        if not new_correct:
            # Newest agent got it wrong
            if len(baseline_correct_agents) == len(baseline_results):
                # All baseline agents got it right - UNIQUE ERROR
                unique_errors.append(question_id)
            elif len(baseline_wrong_agents) == len(baseline_results):
                # All failed (including new) - CONSENSUS ERROR
                consensus_errors.append(question_id)
            else:
                # Mixed - some baseline succeeded, some failed
                mixed_results[question_id] = {
                    'new_agent_correct': False,
                    'baseline_correct': baseline_correct_agents,
                    'baseline_wrong': baseline_wrong_agents
                }
        else:
            # Newest agent got it right
            if len(baseline_wrong_agents) == len(baseline_results):
                # All baseline agents got it wrong - UNIQUE SUCCESS
                unique_successes.append(question_id)
            elif len(baseline_correct_agents) == len(baseline_results):
                # All succeeded (including new) - consensus success, don't report
                pass
            else:
                # Mixed - some baseline succeeded, some failed
                mixed_results[question_id] = {
                    'new_agent_correct': True,
                    'baseline_correct': baseline_correct_agents,
                    'baseline_wrong': baseline_wrong_agents
                }

    analysis = {
        'new_vs_baseline': {
            'unique_successes': unique_successes,
            'unique_errors': unique_errors,
            'consensus_errors': consensus_errors,
            'mixed_results': mixed_results
        }
    }

    # Pairwise: newest vs each baseline agent
    for baseline_agent in baseline_agents:
        new_wrong_A_right = []
        new_right_A_wrong = []
        both_wrong = []

        for question_id, agents_results in filtered_questions.items():
            if newest_agent not in agents_results or baseline_agent not in agents_results:
                continue

            new_result = agents_results[newest_agent]
            baseline_result = agents_results[baseline_agent]

            new_correct = new_result.get('score', 0) >= 0.5
            baseline_correct = baseline_result.get('score', 0) >= 0.5

            if not new_correct and baseline_correct:
                new_wrong_A_right.append(question_id)
            elif new_correct and not baseline_correct:
                new_right_A_wrong.append(question_id)
            elif not new_correct and not baseline_correct:
                both_wrong.append(question_id)

        analysis[f'new_vs_{baseline_agent}'] = {
            'new_wrong_A_right': new_wrong_A_right,
            'new_right_A_wrong': new_right_A_wrong,
            'both_wrong': both_wrong
        }

    return analysis


def build_error_index(iteration_dirs: List[Path], new_agent: Optional[str] = None) -> Dict:
    """
    Build complete error index.

    Args:
        iteration_dirs: List of iteration directories to process
        new_agent: Optional name of new agent being tested
    """
    # Determine agents directory (look in most recent iteration dir)
    agents_dir = None
    for iteration_dir in reversed(iteration_dirs):
        # Resolve the path to handle relative paths (e.g., ../../iteration_004)
        resolved_dir = iteration_dir.resolve()

        # Check if this is a test workspace (nested in evolution_output)
        # Test workspace: experiment_root/evolution_output/iteration_XXX/iteration_YYY_test/
        # Normal iteration: experiment_root/iteration_XXX/
        if 'evolution_output' in resolved_dir.parts:
            # Test workspace: go up 3 levels to experiment root
            potential_agents_dir = resolved_dir.parent.parent.parent / "agents"
        else:
            # Normal iteration: go up 1 level to experiment root
            potential_agents_dir = resolved_dir.parent / "agents"

        if potential_agents_dir.exists():
            agents_dir = potential_agents_dir
            break

    if not agents_dir:
        print("Error: Could not find agents directory", file=sys.stderr)
        sys.exit(1)

    # Detect newest agent
    newest_agent, baseline_agents = detect_newest_agent(iteration_dirs, agents_dir, new_agent)

    # Load all results
    results = load_evaluation_results(iteration_dirs)

    # Build scores_by_question: {qid: {agent_display: score}}
    scores_by_question = {}
    for qid, agents_results in results['by_question'].items():
        scores_by_question[qid] = {
            strip_agent_prefix(agent): r.get('score', 0)
            for agent, r in agents_results.items()
        }

    continuous = is_continuous_scoring(scores_by_question)

    if continuous:
        return _build_continuous_index(newest_agent, baseline_agents, results, scores_by_question)
    else:
        return _build_binary_index(newest_agent, baseline_agents, results, scores_by_question)


def _build_continuous_index(newest_agent: str, baseline_agents: List[str],
                            results: Dict, scores_by_question: Dict) -> Dict:
    """Build error index for continuous-score domains."""
    new_display = strip_agent_prefix(newest_agent)
    baseline_displays = [strip_agent_prefix(a) for a in baseline_agents]

    # by_agent: mean_score, wins_vs_all_others, losses_vs_all_others
    by_agent = {}
    all_agents = [newest_agent] + baseline_agents
    for agent in all_agents:
        if agent not in results['by_agent']:
            continue
        agent_display = strip_agent_prefix(agent)
        scores = [r.get('score', 0) for r in results['by_agent'][agent].values()]
        mean_score = sum(scores) / len(scores) if scores else 0.0

        wins = []
        losses = []
        for qid, agent_scores in scores_by_question.items():
            if agent_display not in agent_scores:
                continue
            my_score = agent_scores[agent_display]
            others = [s for a, s in agent_scores.items() if a != agent_display]
            if not others:
                continue
            if my_score > max(others):
                wins.append(qid)
            if my_score < min(others):
                losses.append(qid)

        by_agent[agent_display] = {
            'mean_score': round(mean_score, 4),
            'wins_vs_all_others': wins,
            'losses_vs_all_others': losses,
        }

    # cross_agent_analysis
    cross_agent = {}

    # new_vs_best_baseline: delta = new - best_baseline per question
    new_vs_best_baseline = {}
    new_wins_all = []
    new_losses_all = []
    for qid, agent_scores in scores_by_question.items():
        if new_display not in agent_scores:
            continue
        baseline_scores = {a: agent_scores[a] for a in baseline_displays if a in agent_scores}
        if not baseline_scores:
            continue
        new_score = agent_scores[new_display]
        best_baseline = max(baseline_scores.values())
        delta = new_score - best_baseline
        new_vs_best_baseline[qid] = round(delta, 4)

        # Check if new agent beats ALL baselines
        if all(new_score > bs for bs in baseline_scores.values()):
            new_wins_all.append(qid)
        # Check if new agent is worse than ALL baselines
        if all(new_score < bs for bs in baseline_scores.values()):
            new_losses_all.append(qid)

    # Sort by delta ascending (worst first for new agent)
    new_vs_best_baseline = dict(sorted(new_vs_best_baseline.items(), key=lambda x: x[1]))

    cross_agent['new_vs_best_baseline'] = new_vs_best_baseline
    cross_agent['new_wins_all'] = new_wins_all
    cross_agent['new_losses_all'] = new_losses_all

    # Pairwise comparisons
    for baseline_agent in baseline_agents:
        bl_display = strip_agent_prefix(baseline_agent)
        deltas = []
        new_wins = 0
        new_losses = 0
        for qid, agent_scores in scores_by_question.items():
            if new_display not in agent_scores or bl_display not in agent_scores:
                continue
            d = agent_scores[new_display] - agent_scores[bl_display]
            deltas.append(d)
            if agent_scores[new_display] > agent_scores[bl_display]:
                new_wins += 1
            elif agent_scores[new_display] < agent_scores[bl_display]:
                new_losses += 1
        mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
        cross_agent[f'new_vs_{bl_display}'] = {
            'mean_delta': round(mean_delta, 4),
            'new_wins': new_wins,
            'new_losses': new_losses,
        }

    # summary
    new_mean = by_agent.get(new_display, {}).get('mean_score', 0.0)
    baseline_means = [by_agent[a]['mean_score'] for a in baseline_displays if a in by_agent]
    best_baseline_mean = max(baseline_means) if baseline_means else 0.0
    total_questions = len(results['by_agent'].get(newest_agent, {}))

    summary = {
        'new_agent': new_display,
        'baseline_agents': baseline_displays,
        'total_questions': total_questions,
        'new_agent_mean': round(new_mean, 4),
        'best_baseline_mean': round(best_baseline_mean, 4),
        'delta': round(new_mean - best_baseline_mean, 4),
        'new_wins_all_count': len(new_wins_all),
        'new_losses_all_count': len(new_losses_all),
    }

    return {
        'summary': summary,
        'by_agent': by_agent,
        'cross_agent_analysis': cross_agent,
        'non_binary_scores': {},
        'scores_by_question': scores_by_question,
    }


def _build_binary_index(newest_agent: str, baseline_agents: List[str],
                         results: Dict, scores_by_question: Dict) -> Dict:
    """Build error index for binary-score domains."""
    # Build by_agent summary
    by_agent = {}
    for agent in [newest_agent] + baseline_agents:
        if agent not in results['by_agent']:
            continue

        total_correct = 0
        total_questions = 0
        error_ids = []

        for question_id, result in results['by_agent'][agent].items():
            total_questions += 1
            if result.get('score', 0) >= 0.5:
                total_correct += 1
            else:
                error_ids.append(question_id)

        accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0

        by_agent[agent] = {
            'total_correct': total_correct,
            'total_errors': total_questions - total_correct,
            'accuracy': round(accuracy, 1),
            'error_ids': error_ids
        }

    # Build global cross-agent analysis
    global_cross_agent = build_cross_agent_analysis(newest_agent, baseline_agents, results)

    # Build summary
    new_agent_stats = by_agent.get(newest_agent, {})

    # Count baseline comparison categories
    unique_errors = len(global_cross_agent['new_vs_baseline']['unique_errors'])
    unique_successes = len(global_cross_agent['new_vs_baseline']['unique_successes'])
    consensus_errors = len(global_cross_agent['new_vs_baseline']['consensus_errors'])
    mixed_results = len(global_cross_agent['new_vs_baseline']['mixed_results'])

    summary = {
        'new_agent': newest_agent,
        'baseline_agents': baseline_agents,
        'total_questions': new_agent_stats.get('total_correct', 0) + new_agent_stats.get('total_errors', 0),
        'new_agent_errors': new_agent_stats.get('total_errors', 0),
        'new_agent_accuracy': new_agent_stats.get('accuracy', 0),
        'baseline_comparison': {
            'unique_errors': unique_errors,           # New failed, all baselines succeeded
            'unique_successes': unique_successes,     # New succeeded, all baselines failed
            'consensus_errors': consensus_errors,     # All failed (including new)
            'mixed_results': mixed_results            # Some baselines succeeded, some failed
        }
    }

    # Transform agent names: strip 'agent_' prefix for cleaner display
    summary['new_agent'] = strip_agent_prefix(summary['new_agent'])
    summary['baseline_agents'] = [strip_agent_prefix(a) for a in summary['baseline_agents']]

    # Transform by_agent keys
    by_agent = {
        strip_agent_prefix(agent): stats
        for agent, stats in by_agent.items()
    }

    # Transform global_cross_agent (agent names in mixed_results)
    for comparison_name, comparison_data in global_cross_agent.items():
        if isinstance(comparison_data, dict) and 'mixed_results' in comparison_data:
            transformed_mixed = {}
            for qid, result_data in comparison_data['mixed_results'].items():
                if 'baseline_correct' in result_data:
                    result_data['baseline_correct'] = [strip_agent_prefix(a) for a in result_data['baseline_correct']]
                if 'baseline_wrong' in result_data:
                    result_data['baseline_wrong'] = [strip_agent_prefix(a) for a in result_data['baseline_wrong']]
                transformed_mixed[qid] = result_data
            comparison_data['mixed_results'] = transformed_mixed

    # Collect non-binary scores (not 0.0 or 1.0), grouped by agent
    non_binary_scores = {}
    for agent in [newest_agent] + baseline_agents:
        if agent not in results['by_agent']:
            continue
        agent_display = strip_agent_prefix(agent)
        for qid, r in results['by_agent'][agent].items():
            s = r.get('score', 0)
            if s not in (0, 0.0, 1, 1.0):
                non_binary_scores.setdefault(agent_display, []).append(
                    {'question_id': qid, 'score': s}
                )

    return {
        'summary': summary,
        'by_agent': by_agent,
        'cross_agent_analysis': global_cross_agent,
        'non_binary_scores': non_binary_scores,
        'scores_by_question': scores_by_question,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Build error index comparing newest agent vs baseline agents'
    )

    parser.add_argument(
        '--iteration-dir',
        type=str,
        help='Single iteration directory'
    )

    parser.add_argument(
        '--iteration-dirs',
        type=str,
        help='Comma-separated iteration directories'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='error_index.json',
        help='Output JSON file (default: error_index.json)'
    )

    parser.add_argument(
        '--new-agent',
        type=str,
        help='Name of new agent being tested (optional, auto-detected if not specified)'
    )

    args = parser.parse_args()

    # Parse iteration directories
    if not args.iteration_dir and not args.iteration_dirs:
        parser.error("Must specify either --iteration-dir or --iteration-dirs")
    elif args.iteration_dir and args.iteration_dirs:
        parser.error("Cannot specify both --iteration-dir and --iteration-dirs")
    elif args.iteration_dirs:
        iteration_dirs = [Path(d.strip()) for d in args.iteration_dirs.split(',')]
    else:
        iteration_dirs = [Path(args.iteration_dir)]

    # Validate directories exist
    for iteration_dir in iteration_dirs:
        if not iteration_dir.exists():
            print(f"Error: Iteration directory not found: {iteration_dir}", file=sys.stderr)
            sys.exit(1)

    print(f"Building error index from {len(iteration_dirs)} iteration(s)", file=sys.stderr)

    # Build index
    index = build_error_index(iteration_dirs, args.new_agent)

    # Write output
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(index, f, indent=2)

    # Print summary
    summary = index['summary']
    print(f"\n=== Index Summary ===", file=sys.stderr)
    if 'new_agent_mean' in summary:
        # Continuous scoring
        print(f"Newest agent: {summary['new_agent']} (mean={summary['new_agent_mean']:.4f})", file=sys.stderr)
        print(f"Baseline agents: {', '.join(summary['baseline_agents'])}", file=sys.stderr)
        print(f"Best baseline mean: {summary['best_baseline_mean']:.4f}", file=sys.stderr)
        print(f"Delta (new - best baseline): {summary['delta']:+.4f}", file=sys.stderr)
        print(f"Questions where new wins all: {summary['new_wins_all_count']}", file=sys.stderr)
        print(f"Questions where new loses all: {summary['new_losses_all_count']}", file=sys.stderr)
    else:
        # Binary scoring
        print(f"Newest agent: {summary['new_agent']} ({summary['new_agent_accuracy']}%)", file=sys.stderr)
        print(f"Baseline agents: {', '.join(summary['baseline_agents'])}", file=sys.stderr)
        print(f"\nTotal errors: {summary['new_agent_errors']}", file=sys.stderr)
        print(f"  Unique errors (new failed, all baselines succeeded): {summary['baseline_comparison']['unique_errors']}", file=sys.stderr)
        print(f"  Unique successes (new succeeded, all baselines failed): {summary['baseline_comparison']['unique_successes']}", file=sys.stderr)
        print(f"  Consensus errors (all failed): {summary['baseline_comparison']['consensus_errors']}", file=sys.stderr)
        print(f"  Mixed results: {summary['baseline_comparison']['mixed_results']}", file=sys.stderr)
    print(f"\nWrote index to {output_file}", file=sys.stderr)


if __name__ == '__main__':
    main()
