#!/usr/bin/env python3
"""
Extract full error details for specific question IDs.

This tool reads evaluation.json files and extracts complete question nodes,
including verification_info when verification_attempts > 1.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set


def strip_agent_prefix(agent_name: str) -> str:
    """Strip 'agent_' prefix from agent name for cleaner display."""
    return agent_name.replace('agent_', '', 1) if agent_name.startswith('agent_') else agent_name


def find_question_in_evaluations(question_ids: Set[str], iteration_dir: Path) -> Dict:
    """
    Find and extract full details for specified question IDs.

    Returns:
        Dict mapping question_id -> agent_name -> full_result_node
    """
    results = {}

    # Find all evaluation files
    eval_files = list(iteration_dir.glob("agent_*/*/evaluations/evaluation.json"))
    if not eval_files:
        eval_files = list(iteration_dir.glob("agent_*/*/results/evaluation.json"))

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

        database = eval_data.get('database', 'unknown')
        eval_results = eval_data.get('results', {})

        # Check each question
        for question_id, result in eval_results.items():
            if question_id in question_ids:
                # Found a match!
                if question_id not in results:
                    results[question_id] = {}

                # Store full result node
                # Add context about which agent and database
                # Strip 'agent_' prefix for cleaner display
                clean_agent_name = strip_agent_prefix(agent_name)
                full_result = {
                    'agent': clean_agent_name,
                    'database': database,
                    **result  # Include all fields from evaluation.json
                }

                # Store under clean agent name to handle cross-agent cases
                results[question_id][clean_agent_name] = full_result

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Extract full error details for specific question IDs'
    )

    parser.add_argument(
        '--question-ids',
        type=str,
        required=True,
        help='Comma-separated list of question IDs (e.g., "123,145,167")'
    )

    parser.add_argument(
        '--iteration-dir',
        type=str,
        help='Path to single iteration directory (e.g., ../../iteration_047)'
    )

    parser.add_argument(
        '--iteration-dirs',
        type=str,
        help='Comma-separated iteration directories (e.g., "../../iteration_047,round_3_test_iteration_047")'
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
        # Check for test round directories in current directory
        test_dirs = list(Path('.').glob('round_*_test_iteration_*'))
        if test_dirs:
            # Deep Focus test round - use both previous iteration and test dir
            test_dir = sorted(test_dirs)[-1]
            prev_iter = int(test_dir.name.split('iteration_')[-1])
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

    # Parse question IDs
    question_ids = set(qid.strip() for qid in args.question_ids.split(','))
    print(f"Extracting details for {len(question_ids)} question IDs", file=sys.stderr)
    print(f"Searching {len(iteration_dirs)} directory(ies)", file=sys.stderr)

    # Extract details from all directories and merge
    all_results = {}
    for iteration_dir in iteration_dirs:
        results = find_question_in_evaluations(question_ids, iteration_dir)
        # Merge results
        for qid, agents in results.items():
            if qid not in all_results:
                all_results[qid] = {}
            all_results[qid].update(agents)

    results = all_results

    # Filter by agent if specified
    if args.agent:
        print(f"Filtering to agent: {args.agent}", file=sys.stderr)
        filtered_results = {}
        for qid, agents in results.items():
            if args.agent in agents:
                # Simplify structure when filtering to single agent
                filtered_results[qid] = agents[args.agent]
        results = filtered_results

        if not results:
            print(f"Warning: No matching questions found for agent '{args.agent}'", file=sys.stderr)
            sys.exit(0)
    elif not results:
        print("Warning: No matching questions found", file=sys.stderr)
        sys.exit(0)

    # Write output
    output_file = Path(args.output)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n=== Extraction Summary ===", file=sys.stderr)
    if args.agent:
        print(f"Agent: {args.agent}", file=sys.stderr)
    print(f"Questions found: {len(results)}/{len(question_ids)}", file=sys.stderr)

    # Show cross-agent cases (only if not filtering by agent)
    if not args.agent:
        cross_agent_cases = [qid for qid, agents in results.items() if len(agents) > 1]
        if cross_agent_cases:
            print(f"\nCross-agent opportunities found: {len(cross_agent_cases)}", file=sys.stderr)
            for qid in cross_agent_cases:
                agents = list(results[qid].keys())
                print(f"  Question {qid}: {len(agents)} agents ({', '.join(agents)})", file=sys.stderr)

    # Show verification cases
    verification_cases = []
    if args.agent:
        # Simple structure: {qid: result}
        for qid, result in results.items():
            verification_info = result.get('verification_info', {})
            attempts = verification_info.get('attempts', 1) if verification_info else 1
            if attempts > 1:
                verification_cases.append((qid, args.agent, attempts))
    else:
        # Multi-agent structure: {qid: {agent: result}}
        for qid, agents in results.items():
            for agent_name, result in agents.items():
                verification_info = result.get('verification_info', {})
                attempts = verification_info.get('attempts', 1) if verification_info else 1
                if attempts > 1:
                    verification_cases.append((qid, agent_name, attempts))

    if verification_cases:
        print(f"\nVerification retries found: {len(verification_cases)}", file=sys.stderr)
        for qid, agent, attempts in verification_cases:
            print(f"  Question {qid} ({agent}): {attempts} attempts", file=sys.stderr)

    print(f"\n✅ Wrote extracted details to {output_file}", file=sys.stderr)

    # Show missing IDs if any
    found_ids = set(results.keys())
    missing_ids = question_ids - found_ids
    if missing_ids:
        print(f"\n⚠️  Missing question IDs: {', '.join(sorted(missing_ids))}", file=sys.stderr)


if __name__ == '__main__':
    main()
