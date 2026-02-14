#!/usr/bin/env python3
"""
Evaluate generated solutions against LiveCodeBench test cases.

Runs solutions against private test cases and reports Pass@1 metric.

Usage:
    python RoboPhD/tools/evaluate_livecodebench.py --cache-dir codegen_cache/haiku_v6
    python RoboPhD/tools/evaluate_livecodebench.py --cache-dir codegen_cache/haiku_v6 --limit 10
"""

import argparse
import base64
import importlib.util
import json
import logging
import multiprocessing
import multiprocessing.pool
import os
import pickle
import subprocess
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_dataset(version: str = "v6") -> Dict[str, Dict]:
    """
    Load LiveCodeBench dataset from HuggingFace.

    Returns dict mapping question_id -> problem data.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    # Map version to files
    version_files = {
        "v1": ["test.jsonl"],
        "v2": ["test.jsonl", "test2.jsonl"],
        "v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
        "v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
        "v5": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"],
        "v6": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
    }

    files_to_load = version_files.get(version, version_files["v6"])
    logger.info(f"Loading LiveCodeBench {version} from HuggingFace...")

    problems = {}
    for filename in files_to_load:
        local_path = hf_hub_download(
            repo_id="livecodebench/code_generation_lite",
            filename=filename,
            repo_type="dataset",
        )
        with open(local_path, "r") as f:
            for line in f:
                if line.strip():
                    problem = json.loads(line)
                    question_id = problem.get("question_id", "")
                    problems[question_id] = problem

    logger.info(f"Loaded {len(problems)} problems")
    return problems


def decompress_test_cases(compressed: str) -> List[Dict]:
    """Decompress private test cases from base64+zlib+pickle+json format.

    Note: pickle.loads is used here on data from the official LiveCodeBench
    HuggingFace repo (livecodebench/code_generation_lite), which is trusted.
    """
    try:
        decoded = base64.b64decode(compressed)
        decompressed = zlib.decompress(decoded)
        unpickled = pickle.loads(decompressed)
        return json.loads(unpickled)
    except Exception as e:
        logger.warning(f"Failed to decompress test cases: {e}")
        return []


def run_solution(solution_path: Path, test_input: str, timeout: int = 10) -> Tuple[bool, str, str]:
    """
    Run solution with test input.

    Returns (success, stdout, stderr).
    """
    try:
        # Suppress MallocStackLogging warnings on macOS
        env = os.environ.copy()
        env["MallocStackLogging"] = ""
        env["MallocNanoZone"] = "0"

        result = subprocess.run(
            ["python3", str(solution_path)],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        # Filter out MallocStackLogging warnings from stderr
        stderr = "\n".join(
            line for line in result.stderr.split("\n")
            if "MallocStackLogging" not in line
        )
        return result.returncode == 0, result.stdout, stderr
    except subprocess.TimeoutExpired:
        return False, "", "TIMEOUT"
    except Exception as e:
        return False, "", str(e)


def categorize_error(stderr: str) -> str:
    """Categorize runtime error from stderr for cleaner reporting."""
    if stderr == "TIMEOUT":
        return "TIMEOUT"
    error_types = [
        "SyntaxError", "NameError", "TypeError", "ValueError",
        "IndexError", "KeyError", "MemoryError", "RecursionError",
        "AttributeError", "ZeroDivisionError", "FileNotFoundError",
    ]
    for err in error_types:
        if err in stderr:
            return err
    return "OTHER_ERROR"


def compare_output(stdout: str, expected: str) -> bool:
    """
    Compare solution output to expected output.

    Uses semantic JSON comparison when possible to handle whitespace
    differences (e.g., [0,3,5] vs [0, 3, 5]). Falls back to string
    comparison for non-JSON output (Codeforces-style problems).
    """
    stdout = stdout.strip()
    expected = expected.strip()

    # Try semantic JSON comparison first
    try:
        return json.loads(stdout) == json.loads(expected)
    except json.JSONDecodeError:
        # Fall back to string comparison for non-JSON output
        return stdout == expected


def _run_leetcode_in_process(solution_path: str, method_name: str, args: List[Any]) -> Any:
    """
    Run LeetCode solution in a separate process to enable timeout.

    This function is called via multiprocessing to isolate the solution
    execution and allow for clean timeout handling.

    Supports both:
    - class Solution with method (standard LeetCode format)
    - Standalone function (alternative format)
    """
    # Import the solution module
    spec = importlib.util.spec_from_file_location("solution", solution_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Try class Solution first
    if hasattr(module, 'Solution'):
        solution = module.Solution()

        # Find the method - if method_name not provided, find the first non-dunder method
        if not method_name:
            methods = [m for m in dir(solution) if not m.startswith('_') and callable(getattr(solution, m))]
            if not methods:
                raise RuntimeError("No public methods found in Solution class")
            method_name = methods[0]  # Alphabetical order (deterministic)

        method = getattr(solution, method_name)
        return method(*args)

    # Fall back to standalone function
    # Find module-level functions (exclude imports and builtins)
    functions = [
        name for name in dir(module)
        if not name.startswith('_')
        and callable(getattr(module, name))
        and getattr(getattr(module, name), '__module__', None) == 'solution'
    ]

    if not functions:
        raise RuntimeError("No Solution class or standalone function found")

    # Use the first function found (or specified method_name)
    func_name = method_name if method_name and method_name in functions else functions[0]
    func = getattr(module, func_name)
    return func(*args)


def run_leetcode_solution(
    solution_path: Path,
    test_input: str,
    timeout: int = 10,
    pool: Optional[multiprocessing.pool.Pool] = None,
) -> Tuple[bool, Any, str]:
    """
    Run LeetCode solution by importing and calling the Solution class method.

    Args:
        solution_path: Path to solution.py
        test_input: Newline-separated JSON arguments
        timeout: Maximum execution time in seconds
        pool: Optional multiprocessing pool (reuse for efficiency)

    Returns:
        (success, result, error_message)
    """
    try:
        # Parse arguments - each line is a JSON value
        args = [json.loads(line) for line in test_input.strip().split('\n') if line.strip()]

        # Run in a separate process with timeout
        # Use provided pool or create one (caller should provide for efficiency)
        if pool is None:
            with multiprocessing.Pool(1) as temp_pool:
                async_result = temp_pool.apply_async(
                    _run_leetcode_in_process,
                    (str(solution_path), "", args)
                )
                result = async_result.get(timeout=timeout)
        else:
            async_result = pool.apply_async(
                _run_leetcode_in_process,
                (str(solution_path), "", args)
            )
            result = async_result.get(timeout=timeout)

        return True, result, ""

    except multiprocessing.TimeoutError:
        return False, None, "TIMEOUT"
    except json.JSONDecodeError as e:
        return False, None, f"JSON_PARSE_ERROR: {e}"
    except Exception as e:
        return False, None, str(e)


def evaluate_leetcode_problem(
    solution_path: Path,
    test_cases: List[Dict],
    timeout: int = 10,
) -> Tuple[bool, int, int, str]:
    """
    Evaluate LeetCode solution by calling the Solution class method directly.

    Returns (all_passed, tests_passed, total_tests, failure_reason).
    """
    if not test_cases:
        return False, 0, 0, "NO_TEST_CASES"

    passed = 0
    failure_reason = ""

    # Create pool once for all test cases (avoid per-test overhead)
    with multiprocessing.Pool(1) as pool:
        for i, tc in enumerate(test_cases):
            test_input = tc.get("input", "")
            expected_output = tc.get("output", tc.get("expected_output", ""))

            success, result, error = run_leetcode_solution(solution_path, test_input, timeout, pool=pool)

            if not success:
                if error == "TIMEOUT":
                    return False, passed, len(test_cases), f"TIMEOUT on test {i+1}"
                if not failure_reason:
                    failure_reason = categorize_error(error) if error else "RUNTIME_ERROR"
                continue

            # Compare result to expected (parse expected as JSON)
            try:
                expected = json.loads(expected_output)
                if result == expected:
                    passed += 1
                else:
                    if not failure_reason:
                        failure_reason = f"WRONG_ANSWER on test {i+1}"
            except json.JSONDecodeError:
                # If expected isn't valid JSON, compare as strings
                if str(result) == expected_output.strip():
                    passed += 1
                else:
                    if not failure_reason:
                        failure_reason = f"WRONG_ANSWER on test {i+1}"

    all_passed = passed == len(test_cases)
    return all_passed, passed, len(test_cases), failure_reason


def evaluate_problem(
    solution_path: Path,
    test_cases: List[Dict],
    timeout: int = 10,
) -> Tuple[bool, int, int, str]:
    """
    Evaluate solution against test cases.

    Returns (all_passed, tests_passed, total_tests, failure_reason).
    """
    if not test_cases:
        return False, 0, 0, "NO_TEST_CASES"

    passed = 0
    failure_reason = ""

    for i, tc in enumerate(test_cases):
        test_input = tc.get("input", "")
        expected_output = tc.get("output", tc.get("expected_output", ""))

        success, stdout, stderr = run_solution(solution_path, test_input, timeout)

        if not success:
            if stderr == "TIMEOUT":
                # Fail fast - don't try remaining tests after timeout
                return False, passed, len(test_cases), f"TIMEOUT on test {i+1}"
            if not failure_reason:
                failure_reason = categorize_error(stderr) if stderr else "RUNTIME_ERROR"
            continue

        # Compare output (semantic JSON comparison when possible)
        if compare_output(stdout, expected_output):
            passed += 1
        else:
            if not failure_reason:
                failure_reason = f"WRONG_ANSWER on test {i+1}"

    all_passed = passed == len(test_cases)
    return all_passed, passed, len(test_cases), failure_reason


def evaluate_single(
    problem_dir: Path,
    problem_data: Dict,
    timeout: int = 10,
    solution_path: Optional[Path] = None,
) -> Dict:
    """Evaluate a single problem. Returns result dict."""
    question_id = problem_dir.name
    solution_path = solution_path or (problem_dir / "solution.py")

    if not solution_path.exists():
        return {
            "question_id": question_id,
            "passed": False,
            "tests_passed": 0,
            "total_tests": 0,
            "reason": "NO_SOLUTION",
        }

    # Get private test cases
    private_tests_compressed = problem_data.get("private_test_cases", "")
    if private_tests_compressed:
        test_cases = decompress_test_cases(private_tests_compressed)
    else:
        # Fallback to public test cases
        public_tests = problem_data.get("public_test_cases", "")
        if isinstance(public_tests, str) and public_tests:
            try:
                test_cases = json.loads(public_tests)
            except json.JSONDecodeError:
                test_cases = []
        elif isinstance(public_tests, list):
            test_cases = public_tests
        else:
            test_cases = []

    # Choose evaluation method based on platform
    platform = problem_data.get("platform", "")
    if platform == "leetcode":
        all_passed, tests_passed, total_tests, reason = evaluate_leetcode_problem(
            solution_path, test_cases, timeout
        )
    else:
        # Codeforces and other platforms use stdin/stdout
        all_passed, tests_passed, total_tests, reason = evaluate_problem(
            solution_path, test_cases, timeout
        )

    return {
        "question_id": question_id,
        "passed": all_passed,
        "tests_passed": tests_passed,
        "total_tests": total_tests,
        "reason": reason if not all_passed else "PASS",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LiveCodeBench solutions"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="Directory containing solution directories (e.g., codegen_cache/haiku_v6)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of problems to evaluate (for testing)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=6,
        help="Timeout per test case in seconds (default: 6, matches official benchmark)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=8,
        help="Maximum concurrent evaluations (default: 8)",
    )
    parser.add_argument(
        "--max-concurrent-retries",
        type=int,
        default=4,
        help="Maximum concurrent retries (default: 4, lower to reduce memory pressure)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show per-problem results",
    )
    parser.add_argument(
        "--retry-only",
        action="store_true",
        help="Only retry failed problems from previous evaluation",
    )
    parser.add_argument(
        "--problems",
        type=str,
        help="Comma-separated list of problem IDs to evaluate (e.g., abc304_d,abc305_e)",
    )

    args = parser.parse_args()
    cache_dir = Path(args.cache_dir)

    if not cache_dir.exists():
        logger.error(f"Cache directory not found: {cache_dir}")
        sys.exit(1)

    # Load dataset
    problems = load_dataset()

    # Handle --retry-only or --problems mode (both merge with existing results)
    output_file = cache_dir.parent / f"{cache_dir.name}_evaluation.json"
    previous_results = {}
    failed_ids = set()

    if args.retry_only or args.problems:
        if not output_file.exists():
            if args.retry_only:
                logger.error(f"No previous evaluation found: {output_file}")
                sys.exit(1)
            # For --problems without existing file, just continue (no merge needed)
        else:
            with open(output_file, "r") as f:
                prev_data = json.load(f)

            # Index previous results by question_id
            for r in prev_data.get("results", []):
                previous_results[r["question_id"]] = r

            if args.retry_only:
                # Find failed problems to retry
                failed_ids = {
                    r["question_id"] for r in prev_data.get("results", [])
                    if not r["passed"]
                }
                logger.info(f"Found {len(failed_ids)} failed problems to retry")

    # Find solution directories
    solution_dirs = sorted([
        d for d in cache_dir.iterdir()
        if d.is_dir() and (d / "solution.py").exists()
    ])

    # Filter to failed problems if --retry-only
    if args.retry_only:
        solution_dirs = [d for d in solution_dirs if d.name in failed_ids]

    # Filter to specific problems if --problems
    if args.problems:
        problem_ids = set(p.strip() for p in args.problems.split(","))
        solution_dirs = [d for d in solution_dirs if d.name in problem_ids]

    if args.limit:
        solution_dirs = solution_dirs[:args.limit]

    logger.info(f"Evaluating {len(solution_dirs)} solutions...")

    # Evaluate solutions
    results = []
    total_tests = 0
    total_tests_passed = 0
    problems_passed = 0

    with ThreadPoolExecutor(max_workers=args.max_concurrent) as executor:
        future_to_dir = {
            executor.submit(
                evaluate_single,
                d,
                problems.get(d.name, {}),
                args.timeout,
            ): d
            for d in solution_dirs
        }

        for future in as_completed(future_to_dir):
            result = future.result()
            results.append(result)

            total_tests += result["total_tests"]
            total_tests_passed += result["tests_passed"]
            if result["passed"]:
                problems_passed += 1

            # Show per-problem results if --verbose or --problems (targeted evaluation)
            if args.verbose or args.problems:
                status = "PASS" if result["passed"] else "FAIL"
                logger.info(
                    f"  {result['question_id']}: {status} "
                    f"({result['tests_passed']}/{result['total_tests']} tests) "
                    f"{result['reason']}"
                )

            # Progress update every 50 problems
            if len(results) % 50 == 0:
                logger.info(f"Progress: {len(results)}/{len(solution_dirs)}")

    # Retry failed problems (catches transient system issues)
    # Retry all failures except WRONG_ANSWER (those won't improve with retry)
    failed_results = [
        r for r in results
        if not r["passed"] and not r["reason"].startswith("WRONG_ANSWER")
    ]

    if failed_results:
        logger.info(f"\nRetrying {len(failed_results)} failed problems ({args.max_concurrent_retries} concurrent)...")

        # Build retry tasks
        retry_tasks = []
        for failed in failed_results:
            qid = failed["question_id"]
            problem_dir = cache_dir / qid
            retry_tasks.append((qid, problem_dir, failed))

        # Run retries concurrently
        with ThreadPoolExecutor(max_workers=args.max_concurrent_retries) as executor:
            future_to_task = {
                executor.submit(
                    evaluate_single,
                    task[1],  # problem_dir
                    problems.get(task[0], {}),  # problem_data
                    args.timeout,
                ): task
                for task in retry_tasks
            }

            for future in as_completed(future_to_task):
                qid, problem_dir, failed = future_to_task[future]
                retry_result = future.result()

                # Update if improved
                if retry_result["passed"] or retry_result["tests_passed"] > failed["tests_passed"]:
                    # Update totals
                    total_tests -= failed["total_tests"]
                    total_tests_passed -= failed["tests_passed"]
                    total_tests += retry_result["total_tests"]
                    total_tests_passed += retry_result["tests_passed"]
                    if retry_result["passed"]:
                        problems_passed += 1

                    # Replace result
                    idx = next(j for j, r in enumerate(results) if r["question_id"] == qid)
                    results[idx] = retry_result

                    status = "PASS" if retry_result["passed"] else "IMPROVED"
                    logger.info(f"  {qid}: {status} ({retry_result['tests_passed']}/{retry_result['total_tests']} tests)")
                else:
                    logger.info(f"  {qid}: still failed ({retry_result['reason']})")

    # Second retry for NO_TEST_CASES only (sequential, to minimize memory pressure)
    # These are decompression failures that may succeed with zero concurrency
    no_test_cases = [r for r in results if r["reason"] == "NO_TEST_CASES"]
    if no_test_cases:
        logger.info(f"\nRetrying {len(no_test_cases)} NO_TEST_CASES problems sequentially...")
        for failed in no_test_cases:
            qid = failed["question_id"]
            problem_dir = cache_dir / qid

            retry_result = evaluate_single(
                problem_dir,
                problems.get(qid, {}),
                timeout=args.timeout,
            )

            # Update if improved
            if retry_result["reason"] != "NO_TEST_CASES":
                # Update totals
                total_tests -= failed["total_tests"]
                total_tests_passed -= failed["tests_passed"]
                total_tests += retry_result["total_tests"]
                total_tests_passed += retry_result["tests_passed"]
                if retry_result["passed"]:
                    problems_passed += 1

                # Replace result
                idx = next(j for j, r in enumerate(results) if r["question_id"] == qid)
                results[idx] = retry_result

                status = "PASS" if retry_result["passed"] else retry_result["reason"]
                logger.info(f"  {qid}: {status} ({retry_result['tests_passed']}/{retry_result['total_tests']} tests)")
            else:
                logger.info(f"  {qid}: still NO_TEST_CASES")

    # Sort results by question_id
    results.sort(key=lambda x: x["question_id"])

    # Merge with previous results if --retry-only or --problems
    if (args.retry_only or args.problems) and previous_results:
        # Update previous results with new retry results
        for r in results:
            previous_results[r["question_id"]] = r

        # Rebuild full results list
        results = sorted(previous_results.values(), key=lambda x: x["question_id"])

        # Recalculate totals from merged results
        total_tests = sum(r["total_tests"] for r in results)
        total_tests_passed = sum(r["tests_passed"] for r in results)
        problems_passed = sum(1 for r in results if r["passed"])

    # Summary
    pass_rate = 100 * problems_passed / len(results) if results else 0
    test_pass_rate = 100 * total_tests_passed / total_tests if total_tests else 0

    logger.info(f"\n{'='*50}")
    logger.info(f"RESULTS")
    logger.info(f"{'='*50}")
    logger.info(f"  Pass@1: {problems_passed}/{len(results)} ({pass_rate:.1f}%)")
    logger.info(f"  Test cases: {total_tests_passed}/{total_tests} ({test_pass_rate:.1f}%)")
    logger.info(f"{'='*50}")

    # Failure breakdown (group by error type, ignoring "on test X" suffix)
    failure_reasons = {}
    for r in results:
        if not r["passed"]:
            reason = r["reason"].split(" on test")[0].split(":")[0]
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

    if failure_reasons:
        logger.info("\nFailure breakdown:")
        for reason, count in sorted(failure_reasons.items(), key=lambda x: -x[1]):
            logger.info(f"  {reason}: {count}")

    # Save results to JSON (sibling to cache dir for easy identification)
    with open(output_file, "w") as f:
        json.dump({
            "summary": {
                "pass_at_1": problems_passed,
                "total_problems": len(results),
                "pass_rate": pass_rate,
                "total_tests": total_tests,
                "tests_passed": total_tests_passed,
                "test_pass_rate": test_pass_rate,
            },
            "failure_breakdown": failure_reasons,
            "results": results,
        }, f, indent=2)
    logger.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
