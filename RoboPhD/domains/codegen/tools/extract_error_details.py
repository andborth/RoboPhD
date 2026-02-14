#!/usr/bin/env python3
"""
Extract full error details for specific problem IDs (CodeGen domain).

This tool reads evaluation.json files from CodeGen runs and extracts
complete problem nodes with v1/v2 pass status, verdict accuracy, and
improvement/regression information.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set


def strip_agent_prefix(agent_name: str) -> str:
    """Strip 'agent_' prefix from agent name for cleaner display."""
    return agent_name.replace('agent_', '', 1) if agent_name.startswith('agent_') else agent_name


def find_problem_in_evaluations(problem_ids: Set[str], iteration_dir: Path) -> Dict:
    """
    Find and extract full details for specified problem IDs from CodeGen evaluations.

    Returns:
        Dict mapping problem_id -> agent_name -> full_result_node
    """
    results = {}

    # CodeGen uses flat evaluation.json at agent level or in output subdirectory
    eval_files = list(iteration_dir.glob("agent_*/evaluation.json"))
    if not eval_files:
        eval_files = list(iteration_dir.glob("agent_*/*/evaluation.json"))
    if not eval_files:
        # Also check for non-agent directories (like during test rounds)
        eval_files = list(iteration_dir.glob("*/evaluation.json"))

    print(f"Searching {len(eval_files)} evaluation files...", file=sys.stderr)

    for eval_file in eval_files:
        try:
            with open(eval_file, 'r') as f:
                eval_data = json.load(f)
        except Exception as e:
            print(f"Warning: Could not read {eval_file}: {e}", file=sys.stderr)
            continue

        # Extract agent name from path
        agent_name = 'unknown'
        for part in eval_file.parts:
            if part.startswith('agent_'):
                agent_name = part
                break

        # CodeGen evaluation structure: {summary: {...}, results: [...]}
        eval_results = eval_data.get('results', [])

        # Check each problem result
        for result in eval_results:
            problem_id = str(result.get('question_id', result.get('problem_id', '')))

            if problem_id in problem_ids:
                # Found a match!
                if problem_id not in results:
                    results[problem_id] = {}

                # Store full result node
                clean_agent_name = strip_agent_prefix(agent_name)
                full_result = {
                    'agent': clean_agent_name,
                    'problem_id': problem_id,
                    'v1_passed': result.get('v1_passed', False),
                    'v2_passed': result.get('v2_passed', False),
                    'correct': result.get('correct', False),
                    'verdict': result.get('verdict'),
                    'verdict_correct': result.get('verdict_correct', False),
                    'improved': result.get('improved', False),
                    'regressed': result.get('regressed', False),
                    'error': result.get('error'),
                }

                # Include any additional fields from the result
                for key in result:
                    if key not in full_result:
                        full_result[key] = result[key]

                results[problem_id][clean_agent_name] = full_result

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Extract full error details for specific problem IDs (CodeGen)'
    )

    parser.add_argument(
        '--problem-ids',
        type=str,
        required=True,
        help='Comma-separated list of problem IDs (e.g., "1234,5678,9012")'
    )

    parser.add_argument(
        '--iteration-dir',
        type=str,
        help='Path to single iteration directory'
    )

    parser.add_argument(
        '--iteration-dirs',
        type=str,
        help='Comma-separated iteration directories'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='extracted_errors.json',
        help='Output JSON file (default: extracted_errors.json)'
    )

    parser.add_argument(
        '--agent',
        type=str,
        help='Filter to specific agent only (optional)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.iteration_dir and not args.iteration_dirs:
        # Default to auto-detect
        test_dirs = list(Path('.').glob('iteration_*_test'))
        if test_dirs:
            # Deep Focus test round
            test_dir = sorted(test_dirs)[-1]
            prev_iter = int(test_dir.name.split('_')[1])
            iteration_dirs = [Path(f'../../iteration_{prev_iter:03d}'), test_dir]
            print(f"Auto-detected Deep Focus test round", file=sys.stderr)
        else:
            # Normal - use previous iteration
            current_iter = int(Path.cwd().name.split('_')[-1])
            prev_iter = current_iter - 1
            iteration_dirs = [Path(f'../../iteration_{prev_iter:03d}')]
            print(f"Auto-detected normal evolution", file=sys.stderr)
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

    # Parse problem IDs
    problem_ids = set(pid.strip() for pid in args.problem_ids.split(','))
    print(f"Extracting details for {len(problem_ids)} problem IDs", file=sys.stderr)
    print(f"Searching {len(iteration_dirs)} directory(ies)", file=sys.stderr)

    # Extract details from all directories and merge
    all_results = {}
    for iteration_dir in iteration_dirs:
        results = find_problem_in_evaluations(problem_ids, iteration_dir)
        # Merge results
        for pid, agents in results.items():
            if pid not in all_results:
                all_results[pid] = {}
            all_results[pid].update(agents)

    results = all_results

    # Filter by agent if specified
    if args.agent:
        print(f"Filtering to agent: {args.agent}", file=sys.stderr)
        filtered_results = {}
        for pid, agents in results.items():
            if args.agent in agents:
                filtered_results[pid] = agents[args.agent]
        results = filtered_results

        if not results:
            print(f"Warning: No matching problems found for agent '{args.agent}'", file=sys.stderr)
            sys.exit(0)
    elif not results:
        print("Warning: No matching problems found", file=sys.stderr)
        sys.exit(0)

    # Write output
    output_file = Path(args.output)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n=== Extraction Summary ===", file=sys.stderr)
    if args.agent:
        print(f"Agent: {args.agent}", file=sys.stderr)
    print(f"Problems found: {len(results)}/{len(problem_ids)}", file=sys.stderr)

    # Show cross-agent cases (only if not filtering by agent)
    if not args.agent:
        cross_agent_cases = [pid for pid, agents in results.items() if len(agents) > 1]
        if cross_agent_cases:
            print(f"\nCross-agent cases: {len(cross_agent_cases)}", file=sys.stderr)
            for pid in cross_agent_cases:
                agents = list(results[pid].keys())
                print(f"  Problem {pid}: {len(agents)} agents ({', '.join(agents)})", file=sys.stderr)

    # Show improvement/regression stats
    improved_count = 0
    regressed_count = 0
    verdict_correct_count = 0
    total_with_verdict = 0

    if args.agent:
        for pid, result in results.items():
            if result.get('improved'):
                improved_count += 1
            if result.get('regressed'):
                regressed_count += 1
            if result.get('verdict') is not None:
                total_with_verdict += 1
                if result.get('verdict_correct'):
                    verdict_correct_count += 1
    else:
        for pid, agents in results.items():
            for agent_name, result in agents.items():
                if result.get('improved'):
                    improved_count += 1
                if result.get('regressed'):
                    regressed_count += 1
                if result.get('verdict') is not None:
                    total_with_verdict += 1
                    if result.get('verdict_correct'):
                        verdict_correct_count += 1

    if improved_count or regressed_count:
        print(f"\nImprovement/Regression:", file=sys.stderr)
        print(f"  Improved: {improved_count}", file=sys.stderr)
        print(f"  Regressed: {regressed_count}", file=sys.stderr)

    if total_with_verdict > 0:
        verdict_accuracy = (verdict_correct_count / total_with_verdict) * 100
        print(f"\nVerdict Accuracy: {verdict_correct_count}/{total_with_verdict} ({verdict_accuracy:.1f}%)", file=sys.stderr)

    print(f"\n✅ Wrote extracted details to {output_file}", file=sys.stderr)

    # Show missing IDs if any
    found_ids = set(results.keys())
    missing_ids = problem_ids - found_ids
    if missing_ids:
        print(f"\n⚠️  Missing problem IDs: {', '.join(sorted(missing_ids))}", file=sys.stderr)


if __name__ == '__main__':
    main()
