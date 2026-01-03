#!/usr/bin/env python3
"""
Build fast error index comparing newest agent vs baseline agents.

This tool:
1. Auto-detects the newest agent (by created_iteration or alphabetically)
2. Compares it against all other tested agents (baseline)
3. Builds per-database cross-agent analysis
4. Builds global cross-agent analysis

Designed for efficiency - focuses on newest agent only.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict


def strip_agent_prefix(agent_name: str) -> str:
    """Strip 'agent_' prefix from agent name for cleaner display."""
    return agent_name.replace('agent_', '', 1) if agent_name.startswith('agent_') else agent_name


def detect_newest_agent(iteration_dirs: List[Path], agents_dir: Path, new_agent: Optional[str] = None) -> Tuple[str, List[str]]:
    """
    Detect newest agent and baseline agents.

    All iterations now use unified structure with agent_*/ directories.

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
        # All iterations use nested structure: agent_*/db_name/results/evaluation.json
        eval_files = list(iteration_dir.glob("agent_*/*/evaluations/evaluation.json"))
        if not eval_files:
            eval_files = list(iteration_dir.glob("agent_*/*/results/evaluation.json"))

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
        # Examples: agent_iter5_..., iter5_..., agent_naive (no iter number)
        if created_iter == -1:
            import re
            match = re.search(r'iter(\d+)', agent_id)
            if match:
                created_iter = int(match.group(1))

        agents_with_metadata.append((agent_id, created_iter))

    # Sort by (iter, name) descending
    # Agents with iterX numbers (highest iter wins)
    # Then agents without iterX (alphabetically last wins)
    agents_with_metadata.sort(key=lambda x: (x[1], x[0]), reverse=True)

    newest_agent = agents_with_metadata[0][0]
    baseline_agents = [agent_id for agent_id, _ in agents_with_metadata[1:]]

    print(f"🎯 Newest agent: {newest_agent}", file=sys.stderr)
    print(f"📊 Baseline agents: {', '.join(baseline_agents)}", file=sys.stderr)

    return newest_agent, baseline_agents


def load_evaluation_results(iteration_dirs: List[Path]) -> Dict:
    """
    Load all evaluation results from iteration directories.

    All iterations now use unified structure with agent_*/ directories.

    Returns:
        {
            'by_question': {question_id: {agent_name: result}},
            'by_agent_db': {agent_name: {db_name: {question_id: result}}},
            'databases': set of database names
        }
    """
    by_question = defaultdict(dict)
    by_agent_db = defaultdict(lambda: defaultdict(dict))
    databases = set()

    for iteration_dir in iteration_dirs:
        # All iterations use nested structure: agent_*/db_name/results/evaluation.json
        eval_files = list(iteration_dir.glob("agent_*/*/evaluations/evaluation.json"))
        if not eval_files:
            eval_files = list(iteration_dir.glob("agent_*/*/results/evaluation.json"))

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

            db_name = eval_data.get('database', 'unknown')
            databases.add(db_name)

            results = eval_data.get('results', {})
            # Handle case where results is a list (e.g., timeout with no predictions)
            if not isinstance(results, dict):
                continue
            for question_id, result in results.items():
                # Store by question
                by_question[question_id][agent_name] = result

                # Store by agent + database
                by_agent_db[agent_name][db_name][question_id] = result

    return {
        'by_question': dict(by_question),
        'by_agent_db': dict(by_agent_db),
        'databases': databases
    }


def build_cross_agent_analysis(newest_agent: str, baseline_agents: List[str],
                               results: Dict, question_ids: Optional[Set[str]] = None) -> Dict:
    """
    Build cross-agent analysis for newest vs baseline agents.

    Args:
        newest_agent: Name of newest agent
        baseline_agents: List of baseline agent names
        results: Results dict from load_evaluation_results
        question_ids: Optional set to filter questions (for per-database analysis)

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
        new_correct = new_result.get('matches', False)

        # Check baseline agents
        baseline_results = []
        for agent in baseline_agents:
            if agent in agents_results:
                baseline_correct = agents_results[agent].get('matches', False)
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

            new_correct = new_result.get('matches', False)
            baseline_correct = baseline_result.get('matches', False)

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

    # Build by_agent summary
    by_agent = {}
    for agent in [newest_agent] + baseline_agents:
        if agent not in results['by_agent_db']:
            continue

        total_correct = 0
        total_questions = 0
        errors_by_database = defaultdict(list)

        for db_name, questions in results['by_agent_db'][agent].items():
            for question_id, result in questions.items():
                total_questions += 1
                if result.get('matches', False):
                    total_correct += 1
                else:
                    errors_by_database[db_name].append(question_id)

        accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0

        by_agent[agent] = {
            'total_correct': total_correct,
            'total_errors': total_questions - total_correct,
            'accuracy': round(accuracy, 1),
            'errors_by_database': dict(errors_by_database)
        }

    # Build by_database with cross-agent analysis
    by_database = {}
    for db_name in sorted(results['databases']):
        # Get questions in this database (from newest agent)
        if newest_agent not in results['by_agent_db']:
            continue

        db_questions = results['by_agent_db'][newest_agent].get(db_name, {})
        total_questions = len(db_questions)

        new_correct = sum(1 for r in db_questions.values() if r.get('matches', False))
        new_errors = total_questions - new_correct
        error_question_ids = [qid for qid, r in db_questions.items() if not r.get('matches', False)]

        # Build cross-agent analysis for this database
        db_question_ids = set(db_questions.keys())
        cross_agent_analysis = build_cross_agent_analysis(
            newest_agent, baseline_agents, results, db_question_ids
        )

        # Build agent_stats for this database (for accuracy table)
        agent_stats = {}
        all_agents = [newest_agent] + baseline_agents
        for agent in all_agents:
            if agent in results['by_agent_db'] and db_name in results['by_agent_db'][agent]:
                agent_db_questions = results['by_agent_db'][agent][db_name]
                correct = sum(1 for r in agent_db_questions.values() if r.get('matches', False))
                total = len(agent_db_questions)
                agent_stats[agent] = {
                    'correct': correct,
                    'errors': total - correct,
                    'accuracy': round(correct / total * 100, 1) if total > 0 else 0.0
                }

        by_database[db_name] = {
            'total_questions': total_questions,
            'new_agent_correct': new_correct,
            'new_agent_errors': new_errors,
            'error_question_ids': error_question_ids,
            'agent_stats': agent_stats,
            'cross_agent_analysis': cross_agent_analysis
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

    # Shared errors: newest wrong + all baseline wrong
    shared_errors = 0
    for question_id, agents_results in results['by_question'].items():
        if newest_agent not in agents_results:
            continue
        new_correct = agents_results[newest_agent].get('matches', False)
        if new_correct:
            continue

        # Check if all baseline also wrong
        all_baseline_wrong = True
        for baseline in baseline_agents:
            if baseline in agents_results:
                if agents_results[baseline].get('matches', False):
                    all_baseline_wrong = False
                    break

        if all_baseline_wrong and len([b for b in baseline_agents if b in agents_results]) > 0:
            shared_errors += 1

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
    # This makes output consistent with how agents are conceptually named
    # (e.g., "naive" instead of "agent_naive")

    # Transform summary
    summary['new_agent'] = strip_agent_prefix(summary['new_agent'])
    summary['baseline_agents'] = [strip_agent_prefix(a) for a in summary['baseline_agents']]

    # Transform by_agent keys
    by_agent = {
        strip_agent_prefix(agent): stats
        for agent, stats in by_agent.items()
    }

    # Transform by_database (agent names in cross_agent_analysis)
    for db_name, db_stats in by_database.items():
        # Transform agent_stats keys
        if 'agent_stats' in db_stats:
            db_stats['agent_stats'] = {
                strip_agent_prefix(agent): stats
                for agent, stats in db_stats['agent_stats'].items()
            }

        # Transform cross_agent_analysis agent references
        if 'cross_agent_analysis' in db_stats:
            for comparison_name, comparison_data in db_stats['cross_agent_analysis'].items():
                if isinstance(comparison_data, dict) and 'mixed_results' in comparison_data:
                    # Transform agent names in mixed_results
                    transformed_mixed = {}
                    for qid, result_data in comparison_data['mixed_results'].items():
                        if 'baseline_correct' in result_data:
                            result_data['baseline_correct'] = [strip_agent_prefix(a) for a in result_data['baseline_correct']]
                        if 'baseline_wrong' in result_data:
                            result_data['baseline_wrong'] = [strip_agent_prefix(a) for a in result_data['baseline_wrong']]
                        transformed_mixed[qid] = result_data
                    comparison_data['mixed_results'] = transformed_mixed

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

    return {
        'summary': summary,
        'by_database': by_database,
        'by_agent': by_agent,
        'cross_agent_analysis': global_cross_agent
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
    print(f"\n=== Index Summary ===", file=sys.stderr)
    print(f"Newest agent: {index['summary']['new_agent']} ({index['summary']['new_agent_accuracy']}%)", file=sys.stderr)
    print(f"Baseline agents: {', '.join(index['summary']['baseline_agents'])}", file=sys.stderr)
    print(f"\nTotal errors: {index['summary']['new_agent_errors']}", file=sys.stderr)
    print(f"  🔴 Unique errors (new failed, all baselines succeeded): {index['summary']['baseline_comparison']['unique_errors']}", file=sys.stderr)
    print(f"  ✅ Unique successes (new succeeded, all baselines failed): {index['summary']['baseline_comparison']['unique_successes']}", file=sys.stderr)
    print(f"  🔵 Consensus errors (all failed): {index['summary']['baseline_comparison']['consensus_errors']}", file=sys.stderr)
    print(f"  ⚪ Mixed results: {index['summary']['baseline_comparison']['mixed_results']}", file=sys.stderr)
    print(f"\n✅ Wrote index to {output_file}", file=sys.stderr)


if __name__ == '__main__':
    main()
