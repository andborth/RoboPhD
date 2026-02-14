#!/usr/bin/env python3
"""
Test script to verify CodeGenDomain abstraction works correctly.

Tests:
1. load_problems() - loads from cache directory
2. get_contexts() - returns problem IDs
3. sample_problems() - samples problems correctly
4. run_evaluation() - calls run_critic_evaluation.py (optional, requires cache)

Usage:
    python RoboPhD/tools/test_codegen_domain.py
    python RoboPhD/tools/test_codegen_domain.py --run-eval  # Also test evaluation
"""

import argparse
import random
import sys
from pathlib import Path


def test_load_problems(domain):
    """Test that load_problems() returns expected structure.

    Returns:
        (passed, problems) tuple - problems returned for use by subsequent tests
    """
    print("\n=== Testing load_problems() ===")

    problems = domain.load_problems()

    if not problems:
        print("WARNING: No problems loaded. Is cache_dir correct?")
        print(f"  cache_dir: {domain.cache_dir}")
        return False, {}

    print(f"Loaded {len(problems)} problems")

    # Check structure of first problem
    first_id = next(iter(problems.keys()))
    first_problem_list = problems[first_id]

    print(f"Sample problem ID: {first_id}")
    print(f"  Type: {type(first_problem_list)}")

    if not isinstance(first_problem_list, list):
        print(f"ERROR: Expected list, got {type(first_problem_list)}")
        return False, {}

    if not first_problem_list:
        print("ERROR: Problem list is empty")
        return False, {}

    first_problem = first_problem_list[0]
    print(f"  Problem dict keys: {list(first_problem.keys())}")

    # Check required fields
    required_fields = ['question_id', '_cache_dir']
    missing = [f for f in required_fields if f not in first_problem]
    if missing:
        print(f"ERROR: Missing required fields: {missing}")
        return False, {}

    print("OK: load_problems() returns correct structure")
    return True, problems


def test_get_contexts(domain, problems):
    """Test that get_contexts() returns problem IDs."""
    print("\n=== Testing get_contexts() ===")

    contexts = domain.get_contexts()

    if not contexts:
        print("WARNING: No contexts returned")
        return False

    print(f"Got {len(contexts)} contexts")
    print(f"Sample contexts: {contexts[:5]}")

    # Verify contexts match problem keys (problems passed from prior test)
    if set(contexts) != set(problems.keys()):
        print("ERROR: Contexts don't match problem keys")
        return False

    print("OK: get_contexts() returns problem IDs")
    return True


def test_sample_problems(domain):
    """Test that sample_problems() samples correctly."""
    print("\n=== Testing sample_problems() ===")

    rng = random.Random(42)
    config = {'contexts_per_iteration': 5}

    sampled = domain.sample_problems(config, rng)

    print(f"Sampled {len(sampled.contexts)} contexts")
    print(f"Contexts: {sampled.contexts}")

    # Check structure
    if len(sampled.contexts) != 5:
        all_contexts = domain.get_contexts()
        if len(all_contexts) >= 5:
            print(f"ERROR: Expected 5 contexts, got {len(sampled.contexts)}")
            return False

    # Check problems_by_context
    for ctx in sampled.contexts:
        if ctx not in sampled.problems_by_context:
            print(f"ERROR: Context {ctx} not in problems_by_context")
            return False
        problems = sampled.problems_by_context[ctx]
        if not problems:
            print(f"ERROR: No problems for context {ctx}")
            return False

    print("OK: sample_problems() works correctly")
    return True


def test_properties(domain):
    """Test domain properties."""
    print("\n=== Testing domain properties ===")

    props = [
        ('phase1_input_name', 'problem context'),
        ('solution_name', 'code'),
        ('evolution_strategies_dir', 'evolution_strategies_codegen'),
        ('context_label', 'Problem'),
        ('sampling_style', 'flat'),
    ]

    all_ok = True
    for prop_name, expected in props:
        actual = getattr(domain, prop_name)
        status = "OK" if actual == expected else "MISMATCH"
        print(f"  {prop_name}: {actual} ({status})")
        if actual != expected:
            print(f"    Expected: {expected}")
            all_ok = False

    if all_ok:
        print("OK: All properties correct")
    return all_ok


def test_run_evaluation(domain, agent_path: Path, num_problems: int = 3):
    """Test that run_evaluation() calls subprocess correctly."""
    print("\n=== Testing run_evaluation() ===")

    # Sample a few problems
    rng = random.Random(42)
    config = {'contexts_per_iteration': num_problems}
    sampled = domain.sample_problems(config, rng)

    print(f"Testing with {len(sampled.contexts)} problems: {sampled.contexts}")
    print(f"Agent path: {agent_path}")

    # Create temp output dir
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "eval_output"
        output_dir.mkdir()

        eval_config = {
            'coder_model': 'haiku-4.5',
            'critic_model': 'haiku-4.5',
            'timeout': 300,
            'max_concurrent': 2,
        }

        print("Running evaluation (this may take a while)...")
        result = domain.run_evaluation(
            sampled=sampled,
            agent_path=agent_path,
            output_dir=output_dir,
            config=eval_config
        )

        print(f"\nResults:")
        print(f"  Accuracy: {result.accuracy:.1f}%")
        print(f"  Correct: {result.correct}/{result.total}")

        if result.metadata.get('error'):
            print(f"  Error: {result.metadata['error']}")
            return False

        print(f"  v1_pass_rate: {result.metadata.get('v1_pass_rate', 'N/A')}")
        print(f"  improvement: {result.metadata.get('improvement', 'N/A')}")

        # Check that evaluation.json was created
        eval_file = output_dir / "evaluation.json"
        if eval_file.exists():
            print(f"  evaluation.json created: {eval_file}")
        else:
            print("  WARNING: evaluation.json not found")

    print("OK: run_evaluation() completed")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test CodeGenDomain abstraction")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="codegen_cache/haiku-4.5_v6",
        help="Path to codegen cache directory"
    )
    parser.add_argument(
        "--run-eval",
        action="store_true",
        help="Also test run_evaluation() (requires cache and agent)"
    )
    parser.add_argument(
        "--critic-agent",
        type=str,
        default="RoboPhD/codegen_agents/naive_critic",
        help="Path to critic agent directory"
    )
    parser.add_argument(
        "--num-problems",
        type=int,
        default=3,
        help="Number of problems for evaluation test"
    )
    parser.add_argument(
        "--evolution-set",
        action="store_true",
        help="Use evolution set only (contest_date < 2024-11-01)"
    )
    args = parser.parse_args()

    # Import domain
    from RoboPhD.domains.codegen.domain import CodeGenDomain

    # Create domain
    config = {
        'cache_dir': args.cache_dir,
        'evolution_set_only': args.evolution_set,
    }
    domain = CodeGenDomain(config)

    print("=" * 60)
    print("CodeGenDomain Abstraction Test")
    print("=" * 60)
    print(f"Cache directory: {domain.cache_dir}")
    print(f"Evolution set only: {domain.evolution_set_only}")

    # Run tests
    results = []

    load_passed, problems = test_load_problems(domain)
    results.append(("load_problems", load_passed))
    results.append(("get_contexts", test_get_contexts(domain, problems)))
    results.append(("sample_problems", test_sample_problems(domain)))
    results.append(("properties", test_properties(domain)))

    if args.run_eval:
        agent_path = Path(args.critic_agent)
        if not agent_path.exists():
            print(f"\nERROR: Critic agent not found: {agent_path}")
            results.append(("run_evaluation", False))
        else:
            results.append(("run_evaluation", test_run_evaluation(
                domain, agent_path, args.num_problems
            )))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
