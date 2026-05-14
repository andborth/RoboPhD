#!/usr/bin/env python3
"""
Build fast comparative error index treating all agents symmetrically.

This tool:
1. Loads evaluation results from all agents in an iteration
2. Builds symmetric comparisons (no "newest" agent bias)
3. Identifies consensus patterns and unique agent characteristics
4. Outputs structured JSON for minimal-token navigation

Designed for Round 1 initial analysis where all agents are equal.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

from RoboPhD.report_generator import is_continuous_scoring


def strip_agent_prefix(agent_name: str) -> str:
    """Strip 'agent_' prefix from agent name for cleaner display."""
    return agent_name.replace('agent_', '', 1) if agent_name.startswith('agent_') else agent_name


def load_evaluation_results(iteration_dir: Path) -> Dict:
    """
    Load all evaluation results from iteration directory.

    Supports both:
    - Hierarchical domains (Text2SQL): agent_*/*/results/evaluation.json
    - Flat domains (CodeGen, AIME): agent_*/evaluation.json

    Results are flattened: the intermediate grouping layer (database name)
    is discarded since all active domains are flat.

    Returns:
        {
            'by_question': {question_id: {agent_name: result}},
            'by_agent': {agent_name: {question_id: result}},
            'agents': set of agent names
        }
    """
    by_question = defaultdict(dict)
    by_agent = defaultdict(dict)
    agents = set()
    # Per-agent iteration-aggregate summary fields. Populated from
    # evaluation.json's `summary` block. The aggregate_explanation
    # surfaces in the iteration report's per-agent score table when
    # any value is non-empty; the aggregate_score is the canonical
    # number (aggregator output) ELO compared.
    agent_summaries: Dict[str, Dict] = {}

    # Find all evaluation files - try hierarchical structure first (Text2SQL)
    eval_files = list(iteration_dir.glob("agent_*/*/evaluations/evaluation.json"))
    if not eval_files:
        eval_files = list(iteration_dir.glob("agent_*/*/results/evaluation.json"))

    if eval_files:
        print(f"Warning: Found hierarchical evaluation files. "
              f"Per-database granularity will not be preserved in the error index.",
              file=sys.stderr)

    # If no hierarchical files found, try flat structure (CodeGen, AIME)
    if not eval_files:
        eval_files = list(iteration_dir.glob("agent_*/evaluation.json"))

    for eval_file in eval_files:
        try:
            with open(eval_file, 'r') as f:
                eval_data = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load {eval_file}: {e}", file=sys.stderr)
            continue

        # Extract agent name from path
        agent_name = None
        for part in eval_file.parts:
            if part.startswith('agent_'):
                agent_name = part
                break

        if not agent_name:
            continue

        agents.add(agent_name)

        # Process evaluation results
        if not isinstance(eval_data, dict):
            continue

        # Capture per-agent summary (aggregate_explanation + aggregate
        # score) from evaluation.json. Persisted by ExternalEvaluatorDomain
        # so this subprocess can propagate it into error_index.json
        # without seeing in-memory researcher state.
        summary = eval_data.get('summary') or {}
        if summary:
            agent_summaries[agent_name] = {
                'aggregate_explanation': summary.get('aggregate_explanation', ''),
                'average_score': summary.get('average_score', 0.0),
                'score_sum': summary.get('score_sum', 0.0),
                'total_problems': summary.get('total_problems', 0),
            }

        results_dict = eval_data.get('results', {})
        if not results_dict:
            continue

        # Handle both dict (standard) and list (legacy) formats
        if isinstance(results_dict, list):
            # Legacy list format: convert to dict
            results_dict = {str(r.get('question_id', '')): r for r in results_dict if r.get('question_id')}

        for question_id, result in results_dict.items():
            question_id = str(question_id)
            if not question_id:
                continue

            # Derive correctness from score (>= 0.5 counts as correct)
            is_match = result.get('score', 0) >= 0.5
            status = 'MATCH' if is_match else 'FAILED'
            has_error = result.get('error', False)

            score = result.get('score', 0)
            processed_result = {
                'question_id': question_id,
                'status': status,
                'matches': is_match,
                'score': score,
                'error': has_error,
            }

            by_question[question_id][agent_name] = processed_result
            by_agent[agent_name][question_id] = processed_result

    return {
        'by_question': dict(by_question),
        'by_agent': dict(by_agent),
        'agents': agents,
        'agent_summaries': agent_summaries,
    }


def build_consensus_patterns(agents: Set[str], results: Dict) -> Dict:
    """
    Build consensus patterns: where all agents agree or disagree.

    Returns:
        {
            'consensus_correct': [question_ids],  # All agents got right
            'consensus_failures': [question_ids],    # All agents got wrong
            'split_decisions': {
                'question_id': {
                    'correct': [agent_names],
                    'wrong': [agent_names]
                }
            }
        }
    """
    consensus_correct = []
    consensus_failures = []
    split_decisions = {}

    for question_id, agents_results in results['by_question'].items():
        # Skip if not all agents tested this question
        if len(agents_results) != len(agents):
            continue

        correct_agents = []
        wrong_agents = []

        for agent, result in agents_results.items():
            if result.get('status') == 'MATCH':
                correct_agents.append(agent)
            else:
                wrong_agents.append(agent)

        if len(correct_agents) == len(agents):
            consensus_correct.append(question_id)
        elif len(wrong_agents) == len(agents):
            consensus_failures.append(question_id)
        else:
            split_decisions[question_id] = {
                'correct': sorted(correct_agents),
                'wrong': sorted(wrong_agents)
            }

    return {
        'consensus_correct': consensus_correct,
        'consensus_failures': consensus_failures,
        'split_decisions': split_decisions
    }


def build_agent_stats(agent: str, agents: Set[str], results: Dict) -> Dict:
    """
    Build statistics for a specific agent.

    Returns stats including unique successes/failures.
    """
    total_correct = 0
    total_questions = 0
    failed_ids = []
    error_ids = []  # Actual errors (agent crash, rate limit) — distinct from incorrect answers
    unique_successes = []
    unique_failures = []

    # Check if agent has any results (may have failed Phase 1)
    if agent not in results['by_agent']:
        return {
            'total_correct': 0,
            'total_questions': 0,
            'accuracy': 0.0,
            'failed_ids': [],
            'error_ids': [],
            'unique_successes': [],
            'unique_failures': []
        }

    # Collect all questions this agent was tested on
    for question_id, result in results['by_agent'][agent].items():
        total_questions += 1

        if result.get('status') == 'MATCH':
            total_correct += 1

            # Check if this is a unique success
            other_agents_results = results['by_question'][question_id]
            if len(other_agents_results) == len(agents):
                all_others_wrong = all(
                    other_agents_results[other].get('status') != 'MATCH'
                    for other in agents if other != agent
                )
                if all_others_wrong:
                    unique_successes.append(question_id)
        else:
            failed_ids.append(question_id)
            if result.get('error'):
                error_ids.append(question_id)

            # Check if this is a unique failure
            other_agents_results = results['by_question'][question_id]
            if len(other_agents_results) == len(agents):
                all_others_correct = all(
                    other_agents_results[other].get('status') == 'MATCH'
                    for other in agents if other != agent
                )
                if all_others_correct:
                    unique_failures.append(question_id)

    accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0

    return {
        'total_correct': total_correct,
        'total_failed': total_questions - total_correct,
        'total_errors': len(error_ids),
        'total_questions': total_questions,
        'accuracy': round(accuracy, 1),
        'unique_successes': unique_successes,
        'unique_failures': unique_failures,
        'failed_ids': failed_ids,
        'error_ids': error_ids,
    }


def _warn_sampling_inconsistency(total_unique: int, total_comparable: int):
    """Warn if agents were tested on different question sets."""
    if total_unique > total_comparable:
        difference = total_unique - total_comparable
        pct_not_comparable = round(difference / total_unique * 100, 1)

        print(f"\nWARNING: Question sampling inconsistency detected!", file=sys.stderr)
        print(f"   Total unique questions: {total_unique}", file=sys.stderr)
        print(f"   Questions tested by ALL agents: {total_comparable}", file=sys.stderr)
        print(f"   Questions tested by SOME agents: {difference} ({pct_not_comparable}%)", file=sys.stderr)
        print(f"   ", file=sys.stderr)
        print(f"   This indicates agents were not tested on identical question sets.", file=sys.stderr)
        print(f"   Comparison metrics may not be meaningful for {pct_not_comparable}% of questions.", file=sys.stderr)
        print(f"   Consider re-running with fixed question sampling (see RoboPhD/unit_tests/).", file=sys.stderr)
        print(f"", file=sys.stderr)


def build_error_index(iteration_dir: Path) -> Dict:
    """
    Build complete symmetric comparative error index.
    """
    print(f"Loading evaluation results from {iteration_dir}", file=sys.stderr)
    results = load_evaluation_results(iteration_dir)

    if not results['agents']:
        print("Error: No agents found", file=sys.stderr)
        sys.exit(1)

    agents = results['agents']
    print(f"Found {len(agents)} agents: {', '.join(sorted(agents))}", file=sys.stderr)

    # Build scores_by_question: {qid: {agent_display: score}}
    scores_by_question = {}
    for qid, agents_results in results['by_question'].items():
        scores_by_question[qid] = {
            strip_agent_prefix(agent): r.get('score', 0)
            for agent, r in agents_results.items()
        }

    # Total unique questions across all agents
    total_unique_questions = len(results['by_question'])

    # Detect continuous scoring mode
    continuous = is_continuous_scoring(scores_by_question)
    if continuous:
        print(f"Detected continuous scoring domain", file=sys.stderr)

    if continuous:
        return _build_continuous_index(agents, results, scores_by_question, total_unique_questions)
    else:
        return _build_binary_index(agents, results, scores_by_question, total_unique_questions)


def _build_continuous_index(agents: Set[str], results: Dict, scores_by_question: Dict, total_unique_questions: int) -> Dict:
    """Build error index for continuous-score domains (e.g., Can't Be Late)."""
    sorted_agents = sorted(agents)

    # Find questions tested by ALL agents (comparable)
    comparable_qids = [
        qid for qid, agent_results in results['by_question'].items()
        if len(agent_results) == len(agents)
    ]
    total_comparable = len(comparable_qids)

    _warn_sampling_inconsistency(total_unique_questions, total_comparable)

    # Per-agent stats
    by_agent = {}
    for agent in sorted_agents:
        display = strip_agent_prefix(agent)
        agent_questions = results['by_agent'].get(agent, {})
        scores = [r.get('score', 0) for r in agent_questions.values()]
        mean_score = round(sum(scores) / len(scores), 4) if scores else 0.0

        # Solo wins/losses: uniquely highest/lowest among comparable questions
        solo_wins = []
        solo_losses = []
        for qid in comparable_qids:
            agent_score = results['by_question'][qid][agent].get('score', 0)
            other_scores = [
                results['by_question'][qid][other].get('score', 0)
                for other in agents if other != agent
            ]
            if not other_scores:
                continue
            if agent_score > max(other_scores):
                solo_wins.append(qid)
            if agent_score < min(other_scores):
                solo_losses.append(qid)

        by_agent[display] = {
            'total_questions': len(agent_questions),
            'mean_score': mean_score,
            'solo_wins': solo_wins,
            'solo_losses': solo_losses,
        }

    # Cross-agent differentiation patterns (on comparable questions only)
    deltas = []
    for qid in comparable_qids:
        agent_scores = scores_by_question[qid]
        score_vals = list(agent_scores.values())
        delta = max(score_vals) - min(score_vals)
        deltas.append({'id': qid, 'delta': round(delta, 4)})

    deltas_sorted_desc = sorted(deltas, key=lambda x: x['delta'], reverse=True)
    most_differentiated = deltas_sorted_desc[:20]
    least_differentiated = sorted(deltas, key=lambda x: x['delta'])[:20]

    cross_agent_patterns = {
        'most_differentiated': most_differentiated,
        'least_differentiated': least_differentiated,
    }

    # Summary
    agent_mean_scores = {
        strip_agent_prefix(a): by_agent[strip_agent_prefix(a)]['mean_score']
        for a in sorted_agents
    }

    all_deltas = [d['delta'] for d in deltas] if deltas else [0.0]
    differentiation_stats = {
        'mean_delta': round(sum(all_deltas) / len(all_deltas), 4),
        'max_delta': round(max(all_deltas), 4),
        'min_delta': round(min(all_deltas), 4),
    }

    # Aggregator outputs from evaluation.json summaries — propagated
    # into the index so the report layer can render the dual-column
    # layout (Mean Raw Score + Mean Score + Notes) when any agent has
    # a non-empty explanation. Empty/missing summaries → empty
    # explanations + per-question mean fallback for aggregate scores.
    agent_summaries = results.get('agent_summaries', {})
    agent_explanations = {
        strip_agent_prefix(a): (agent_summaries.get(a, {}) or {}).get('aggregate_explanation', '')
        for a in sorted_agents
    }
    agent_aggregate_scores = {
        strip_agent_prefix(a): (agent_summaries.get(a, {}) or {}).get('average_score', agent_mean_scores.get(strip_agent_prefix(a), 0.0))
        for a in sorted_agents
    }

    summary = {
        'agents': [strip_agent_prefix(a) for a in sorted_agents],
        'total_questions': total_comparable,
        'total_unique_questions': total_unique_questions,
        'agent_mean_scores': agent_mean_scores,
        'agent_aggregate_scores': agent_aggregate_scores,
        'agent_explanations': agent_explanations,
        'differentiation_stats': differentiation_stats,
    }

    return {
        'summary': summary,
        'by_agent': by_agent,
        'cross_agent_patterns': cross_agent_patterns,
        'non_binary_scores': {},
        'scores_by_question': scores_by_question,
    }


def _build_binary_index(agents: Set[str], results: Dict, scores_by_question: Dict, total_unique_questions: int) -> Dict:
    """Build error index for binary-score domains (original behavior)."""

    # Build consensus patterns
    cross_agent_patterns = build_consensus_patterns(agents, results)

    # Build per-agent stats
    by_agent = {}
    for agent in sorted(agents):
        by_agent[agent] = build_agent_stats(agent, agents, results)

    # Build summary
    agent_accuracies = {
        agent: stats['accuracy']
        for agent, stats in by_agent.items()
    }

    # Count questions where ALL agents have results (for meaningful comparison)
    consensus_correct_count = len(cross_agent_patterns['consensus_correct'])
    consensus_failures_count = len(cross_agent_patterns['consensus_failures'])
    split_count = len(cross_agent_patterns['split_decisions'])
    total_comparable_questions = consensus_correct_count + consensus_failures_count + split_count

    _warn_sampling_inconsistency(total_unique_questions, total_comparable_questions)

    # Aggregator outputs from evaluation.json summaries — same role as
    # in the continuous branch (drives dual-column layout when any
    # explanation is non-empty). DS-1000 lands here under the new
    # design (per-example scores are pure 0/1, so is_continuous_scoring
    # returns False) and needs these fields populated.
    agent_summaries = results.get('agent_summaries', {})
    sorted_agent_list = sorted(agents)
    agent_explanations = {
        strip_agent_prefix(a): (agent_summaries.get(a, {}) or {}).get('aggregate_explanation', '')
        for a in sorted_agent_list
    }
    # Aggregate score (aggregator output) is what ELO compares. Fallback
    # to correct/total — the default aggregator's answer for binary tasks.
    # Avoids the `accuracy / 100` percentage round-trip, whose `100`
    # literal visually collides with SCORE_SCALE (also 100).
    def _raw_mean_fallback(agent_name: str) -> float:
        stats = by_agent.get(agent_name, {})
        total = stats.get('total_questions', 0)
        return (stats.get('total_correct', 0) / total) if total else 0.0
    agent_aggregate_scores = {
        strip_agent_prefix(a): (agent_summaries.get(a, {}) or {}).get(
            'average_score', _raw_mean_fallback(a)
        )
        for a in sorted_agent_list
    }

    summary = {
        'agents': sorted(agents),
        'total_questions': total_comparable_questions,  # Only comparable questions
        'total_unique_questions': total_unique_questions,  # All questions (for diagnostics)
        'agent_accuracies': agent_accuracies,
        'agent_aggregate_scores': agent_aggregate_scores,
        'agent_explanations': agent_explanations,
        'consensus_stats': {
            'all_correct': consensus_correct_count,
            'all_correct_pct': round(consensus_correct_count / total_comparable_questions * 100, 1) if total_comparable_questions > 0 else 0,
            'all_failed': consensus_failures_count,
            'all_failed_pct': round(consensus_failures_count / total_comparable_questions * 100, 1) if total_comparable_questions > 0 else 0,
            'split_decisions': split_count,
            'split_decisions_pct': round(split_count / total_comparable_questions * 100, 1) if total_comparable_questions > 0 else 0
        }
    }

    # Transform agent names: strip 'agent_' prefix for cleaner display
    summary['agents'] = [strip_agent_prefix(a) for a in summary['agents']]
    summary['agent_accuracies'] = {
        strip_agent_prefix(agent): acc
        for agent, acc in agent_accuracies.items()
    }

    by_agent = {
        strip_agent_prefix(agent): stats
        for agent, stats in by_agent.items()
    }

    # Transform cross_agent_patterns (agent names in split_decisions)
    for qid, split_info in cross_agent_patterns['split_decisions'].items():
        split_info['correct'] = [strip_agent_prefix(a) for a in split_info['correct']]
        split_info['wrong'] = [strip_agent_prefix(a) for a in split_info['wrong']]

    # Collect non-binary scores (not 0.0 or 1.0), grouped by agent
    non_binary_scores = {}
    for agent, questions in results['by_agent'].items():
        agent_display = strip_agent_prefix(agent)
        for qid, r in questions.items():
            s = r.get('score', 0)
            if s not in (0, 0.0, 1, 1.0):
                non_binary_scores.setdefault(agent_display, []).append(
                    {'question_id': qid, 'score': s}
                )

    return {
        'summary': summary,
        'by_agent': by_agent,
        'cross_agent_patterns': cross_agent_patterns,
        'non_binary_scores': non_binary_scores,
        'scores_by_question': scores_by_question,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Build symmetric comparative error index for all agents'
    )

    parser.add_argument(
        '--iteration-dir',
        type=str,
        required=True,
        help='Iteration directory to analyze'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='error_index.json',
        help='Output JSON file (default: error_index.json)'
    )

    args = parser.parse_args()

    iteration_dir = Path(args.iteration_dir)
    if not iteration_dir.exists():
        print(f"Error: Iteration directory not found: {iteration_dir}", file=sys.stderr)
        sys.exit(1)

    # Build index
    index = build_error_index(iteration_dir)

    # Write output
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(index, f, indent=2)

    # Print summary
    print(f"\n=== Index Summary ===", file=sys.stderr)
    agents = index['summary']['agents']

    if 'agent_mean_scores' in index['summary']:
        # Continuous scoring
        mean_scores = index['summary']['agent_mean_scores']
        print(f"Agents: {', '.join(f'{a} (mean={mean_scores[a]})' for a in agents)}", file=sys.stderr)

        diff = index['summary']['differentiation_stats']
        print(f"\nDifferentiation stats:", file=sys.stderr)
        print(f"  Mean delta: {diff['mean_delta']}", file=sys.stderr)
        print(f"  Max delta: {diff['max_delta']}", file=sys.stderr)
        print(f"  Min delta: {diff['min_delta']}", file=sys.stderr)
    else:
        # Binary scoring
        accuracies = index['summary']['agent_accuracies']
        print(f"Agents: {', '.join(f'{a} ({accuracies[a]}%)' for a in agents)}", file=sys.stderr)

        consensus = index['summary']['consensus_stats']
        print(f"\nConsensus patterns:", file=sys.stderr)
        print(f"  All correct: {consensus['all_correct']} ({consensus['all_correct_pct']}%)", file=sys.stderr)
        print(f"  All failed: {consensus['all_failed']} ({consensus['all_failed_pct']}%)", file=sys.stderr)
        print(f"  Split decisions: {consensus['split_decisions']} ({consensus['split_decisions_pct']}%)", file=sys.stderr)

    print(f"\nWrote index to {output_file}", file=sys.stderr)


if __name__ == '__main__':
    main()
