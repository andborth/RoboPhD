#!/usr/bin/env python3
"""
Pre-compute ground truth results for BIRD benchmark questions.

This script executes all ground truth SQL queries and caches the results
BEFORE running research experiments. This prevents:
- Concurrent database access during agent evaluation
- "Database is locked" errors
- File descriptor exhaustion from connection pooling
- Cache thrashing from thundering herd

By default, processes ALL questions in the dataset. Use --questions-per-database
only for testing with a limited sample.

NOTE: The 'test' dataset does not have ground truth SQL (blind evaluation),
so this script will skip pre-computation for test mode.

Usage:
    # Pre-compute ALL questions for train-filtered dataset (recommended)
    python RoboPhD/tools/precompute_ground_truth.py

    # Pre-compute for dev dataset
    python RoboPhD/tools/precompute_ground_truth.py --dataset dev

    # Pre-compute for specific databases only
    python RoboPhD/tools/precompute_ground_truth.py \
        --databases food_inspection movie_platform regional_sales

    # Test with limited questions (for debugging)
    python RoboPhD/tools/precompute_ground_truth.py \
        --dataset dev \
        --questions-per-database 5 \
        --max-concurrent 1

Author: RoboPhD System
"""

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from RoboPhD.core import DatabaseManager
from RoboPhD.utilities.cached_sql_executor import CachedSQLExecutor, SQLCache


def load_questions_and_databases(
    dataset: str
) -> Tuple[Dict[str, List[Dict]], Path, Path]:
    """
    Load questions grouped by database.

    Args:
        dataset: Dataset name ('train', 'train-filtered', 'train-no-evidence', 'dev', 'dev-no-evidence', 'test')

    Returns:
        Tuple of (questions_by_db, questions_file, db_root)
    """
    # Determine paths based on dataset
    if dataset == 'train':
        questions_file = Path("benchmark_resources/datasets/train/train/train.json")
        db_root = Path("benchmark_resources/datasets/train/train/train_databases")
    elif dataset == 'train-filtered':
        questions_file = Path("benchmark_resources/datasets/train-filtered/train_filtered.json")
        db_root = Path("benchmark_resources/datasets/train/train/train_databases")
    elif dataset == 'train-no-evidence':
        questions_file = Path("benchmark_resources/datasets/train-no-evidence/train_filtered_no_evidence.json")
        db_root = Path("benchmark_resources/datasets/train/train/train_databases")
    elif dataset == 'test':
        questions_file = Path("benchmark_resources/datasets/test/test/test.json")
        db_root = Path("benchmark_resources/datasets/test/test/test_databases")
    elif dataset == 'dev-no-evidence':
        questions_file = Path("benchmark_resources/datasets/dev-no-evidence/dev_no_evidence.json")
        db_root = Path("benchmark_resources/datasets/dev/dev_20240627/dev_databases")
    else:  # dev
        questions_file = Path("benchmark_resources/datasets/dev/dev_20240627/dev.json")
        db_root = Path("benchmark_resources/datasets/dev/dev_20240627/dev_databases")

    if not questions_file.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_file}")

    if not db_root.exists():
        raise FileNotFoundError(f"Database root not found: {db_root}")

    # Load questions
    with open(questions_file, 'r') as f:
        all_questions = json.load(f)

    # Group questions by database
    questions_by_db = {}
    for idx, q in enumerate(all_questions):
        # Add question_id if missing (using array index)
        if 'question_id' not in q:
            q['question_id'] = idx

        db_name = q['db_id']
        if db_name not in questions_by_db:
            questions_by_db[db_name] = []
        questions_by_db[db_name].append(q)

    return questions_by_db, questions_file, db_root


def precompute_database_ground_truth(
    db_name: str,
    questions: List[Dict],
    questions_per_db: Optional[int],
    db_root: Path,
    executor: CachedSQLExecutor,
    random_seed: Optional[int] = None,
    timeout_seconds: int = 300
) -> Tuple[int, int, int, int, List[str]]:
    """
    Pre-compute ground truth results for one database.

    Args:
        db_name: Database name
        questions: All questions for this database
        questions_per_db: Number of questions to sample (None = all questions)
        db_root: Root directory containing databases
        executor: SQL executor with caching
        random_seed: Random seed for reproducible sampling (only used if sampling)
        timeout_seconds: Timeout for SQL execution in seconds (default: 300)

    Returns:
        Tuple of (cached_count, error_count, skip_count, too_large_count, error_messages)
    """
    db_path = db_root / db_name / f"{db_name}.sqlite"

    if not db_path.exists():
        return 0, 0, len(questions), 0, [f"Database file not found: {db_path}"]

    # Use ALL questions by default, or sample if limit specified
    if questions_per_db is not None and questions_per_db < len(questions):
        # Sampling mode (for testing only)
        if random_seed is not None:
            random.seed(random_seed)
        questions_to_process = random.sample(questions, questions_per_db)
    else:
        # Default: Process ALL questions
        questions_to_process = questions

    cached_count = 0
    error_count = 0
    skip_count = 0
    too_large_count = 0
    error_messages = []

    for q in questions_to_process:
        gt_sql = q.get('SQL')
        if not gt_sql:
            skip_count += 1
            continue

        try:
            # Execute and cache (will use existing cache if present)
            results = executor.execute_sql(
                gt_sql,
                str(db_path),
                is_ground_truth=True,
                timeout_seconds=timeout_seconds
            )

            # Check if results were too large to cache
            if len(results) > SQLCache.MAX_RESULT_SIZE:
                too_large_count += 1
                error_msg = f"Question {q.get('question_id')}: Result too large ({len(results)} rows, max {SQLCache.MAX_RESULT_SIZE})"
                error_messages.append(error_msg)
            else:
                # Check byte size too
                import sys
                estimated_size = sys.getsizeof(str(results))
                if estimated_size > SQLCache.MAX_RESULT_BYTES:
                    too_large_count += 1
                    error_msg = f"Question {q.get('question_id')}: Result too large ({estimated_size/(1024*1024):.1f}MB, max {SQLCache.MAX_RESULT_BYTES/(1024*1024):.1f}MB)"
                    error_messages.append(error_msg)
                else:
                    cached_count += 1

        except Exception as e:
            error_count += 1
            error_msg = f"Question {q.get('question_id')}: {str(e)[:100]}"
            error_messages.append(error_msg)

    return cached_count, error_count, skip_count, too_large_count, error_messages


def precompute_ground_truth(
    dataset: str = 'train-filtered',
    databases: Optional[List[str]] = None,
    questions_per_database: Optional[int] = None,
    max_concurrent: int = 3,
    random_seed: Optional[int] = None,
    timeout_seconds: int = 300,
    verbose: bool = True
) -> Dict:
    """
    Pre-compute ground truth results for selected databases.

    Args:
        dataset: Dataset to use (train/train-filtered/dev/test)
        databases: List of databases (None = all available)
        questions_per_database: Questions per database (None = all questions)
        max_concurrent: Concurrent databases (lower = safer)
        random_seed: Random seed for reproducible sampling (only used if sampling)
        timeout_seconds: Timeout for SQL execution in seconds (default: 300)
        verbose: Print progress messages

    Returns:
        Dictionary with statistics
    """
    start_time = time.time()

    if verbose:
        print("=" * 70)
        print("Ground Truth Pre-Computation")
        print("=" * 70)
        print(f"Dataset: {dataset}")

    # Check if this is test dataset (no ground truth available)
    if dataset == 'test':
        if verbose:
            print()
            print("ℹ️  Test dataset does not contain ground truth SQL.")
            print("   Ground truth pre-computation is not needed for test evaluation.")
            print("   Test mode uses TestOutputGenerator (no accuracy calculation).")
            print()
            print("   Skipping pre-computation.")
            print()
        return {
            'success': True,
            'skipped': True,
            'reason': 'Test dataset has no ground truth',
            'databases_processed': 0,
            'total_cached': 0,
            'total_errors': 0,
            'total_skipped': 0
        }

    if verbose:
        if questions_per_database is None:
            print("Mode: ALL questions (recommended for production)")
        else:
            print(f"Mode: LIMITED sampling ({questions_per_database} questions per database)")
            print("      Note: Use ALL questions for production to avoid cache misses")
        print(f"Max concurrent databases: {max_concurrent}")
        if random_seed is not None:
            print(f"Random seed: {random_seed}")
        print()

    # Load questions and databases
    try:
        questions_by_db, questions_file, db_root = load_questions_and_databases(dataset)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return {'success': False, 'error': str(e)}

    # Get available databases
    excluded_dbs = DatabaseManager.get_blacklisted_databases(dataset)
    available_dbs = []

    for db_dir in db_root.iterdir():
        if db_dir.is_dir() and db_dir.name not in excluded_dbs:
            db_path = db_dir / f"{db_dir.name}.sqlite"
            if db_path.exists() and db_dir.name in questions_by_db:
                available_dbs.append(db_dir.name)

    available_dbs.sort()

    # Filter to requested databases
    if databases:
        requested = set(databases)
        available_set = set(available_dbs)
        missing = requested - available_set
        if missing:
            print(f"⚠️  Warning: Databases not found: {', '.join(sorted(missing))}")
        databases_to_process = [db for db in available_dbs if db in requested]
    else:
        databases_to_process = available_dbs

    if not databases_to_process:
        print("❌ No databases to process")
        return {'success': False, 'error': 'No databases to process'}

    if verbose:
        print(f"📊 Processing {len(databases_to_process)} databases:")
        for db_name in databases_to_process:
            num_questions = len(questions_by_db.get(db_name, []))
            if questions_per_database is None:
                sampled = num_questions
                suffix = "(ALL)"
            else:
                sampled = min(questions_per_database, num_questions)
                suffix = f"(sampled from {num_questions})"
            print(f"  - {db_name}: {sampled} questions {suffix}")
        print()

    # Create executor
    executor = CachedSQLExecutor()

    # Process databases with limited concurrency
    results = {}
    total_cached = 0
    total_errors = 0
    total_skipped = 0
    total_too_large = 0

    if verbose:
        print("🔄 Pre-computing ground truth results...")
        print()

    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        # Submit all tasks
        future_to_db = {}
        for db_name in databases_to_process:
            questions = questions_by_db.get(db_name, [])
            future = pool.submit(
                precompute_database_ground_truth,
                db_name,
                questions,
                questions_per_database,
                db_root,
                executor,
                random_seed,
                timeout_seconds
            )
            future_to_db[future] = db_name

        # Wait for completion and report progress
        for future in as_completed(future_to_db):
            db_name = future_to_db[future]

            try:
                cached, errors, skipped, too_large, error_msgs = future.result()

                results[db_name] = {
                    'cached': cached,
                    'errors': errors,
                    'skipped': skipped,
                    'too_large': too_large,
                    'error_messages': error_msgs
                }

                total_cached += cached
                total_errors += errors
                total_skipped += skipped
                total_too_large += too_large

                # Print status
                if errors == 0 and too_large == 0:
                    status = "✅"
                elif too_large > 0:
                    status = "⚠️"
                else:
                    status = "❌"

                if verbose:
                    status_parts = [f"{cached} cached"]
                    if too_large > 0:
                        status_parts.append(f"{too_large} too large")
                    if errors > 0:
                        status_parts.append(f"{errors} errors")
                    if skipped > 0:
                        status_parts.append(f"{skipped} skipped")

                    print(f"{status} {db_name}: {', '.join(status_parts)}")

                    if error_msgs and verbose:
                        for msg in error_msgs[:3]:  # Show first 3 errors/warnings
                            print(f"     {msg}")
                        if len(error_msgs) > 3:
                            print(f"     ... and {len(error_msgs) - 3} more messages")

            except Exception as e:
                results[db_name] = {
                    'cached': 0,
                    'errors': 0,
                    'skipped': 0,
                    'error_messages': [str(e)]
                }
                if verbose:
                    print(f"❌ {db_name}: Failed - {e}")

    elapsed = time.time() - start_time

    # Get cache statistics
    cache_stats = executor.get_cache_stats()
    gt_cache = cache_stats.get('ground_truth_cache', {})

    if verbose:
        print()
        print("=" * 70)
        print("Summary")
        print("=" * 70)
        print(f"Databases processed: {len(databases_to_process)}")
        print(f"Total cached: {total_cached}")
        print(f"Total too large: {total_too_large}")
        print(f"Total errors: {total_errors}")
        print(f"Total skipped: {total_skipped}")
        print(f"Cache entries: {gt_cache.get('total_entries', 0)}")
        print(f"Cache size: {gt_cache.get('cache_size_mb', 0):.2f} MB")
        print(f"Elapsed time: {elapsed:.1f}s")
        print()

        if total_too_large > 0:
            print(f"⚠️  {total_too_large} queries returned results too large to cache (>{SQLCache.MAX_RESULT_SIZE} rows).")
            print("   These queries will be executed during each evaluation run.")
            print()

        if total_errors > 0:
            print(f"⚠️  {total_errors} ground truth queries failed. These questions will be")
            print("   excluded from accuracy calculations during evaluation.")
            print()

    return {
        'success': True,
        'databases_processed': len(databases_to_process),
        'total_cached': total_cached,
        'total_too_large': total_too_large,
        'total_errors': total_errors,
        'total_skipped': total_skipped,
        'cache_stats': cache_stats,
        'elapsed_seconds': elapsed,
        'per_database': results
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Pre-compute ground truth results for BIRD benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pre-compute ALL questions for train-filtered dataset (DEFAULT)
  python RoboPhD/tools/precompute_ground_truth.py

  # Pre-compute for dev dataset
  python RoboPhD/tools/precompute_ground_truth.py --dataset dev

  # Pre-compute for specific databases only
  python RoboPhD/tools/precompute_ground_truth.py \\
      --databases food_inspection movie_platform regional_sales

  # Use lower concurrency to avoid file descriptor issues
  python RoboPhD/tools/precompute_ground_truth.py \\
      --max-concurrent 2

  # Test with limited questions (for debugging only)
  python RoboPhD/tools/precompute_ground_truth.py \\
      --dataset dev \\
      --questions-per-database 5 \\
      --max-concurrent 1

Note: Test dataset has no ground truth, so pre-computation is not needed.
      By default, ALL questions are processed. Use --questions-per-database
      only for testing purposes to avoid cache misses during research.
        """
    )

    parser.add_argument(
        '--dataset',
        choices=['train', 'train-filtered', 'train-no-evidence', 'dev', 'dev-no-evidence', 'test'],
        default='train-filtered',
        help='Dataset to use (default: train-filtered). Note: test has no ground truth.'
    )

    parser.add_argument(
        '--databases',
        nargs='+',
        help='Specific databases to process (default: all)'
    )

    parser.add_argument(
        '--questions-per-database',
        type=int,
        default=None,
        help='Questions per database (default: None = ALL questions). '
             'Only use this for testing with limited samples.'
    )

    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=3,
        help='Max concurrent databases (default: 3, lower = safer)'
    )

    parser.add_argument(
        '--random-seed',
        type=int,
        help='Random seed for reproducible question sampling (only used if --questions-per-database is set)'
    )

    parser.add_argument(
        '--timeout',
        type=int,
        default=300,
        help='SQL execution timeout in seconds (default: 300). Increase for complex queries that need more time.'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )

    args = parser.parse_args()

    try:
        result = precompute_ground_truth(
            dataset=args.dataset,
            databases=args.databases,
            questions_per_database=args.questions_per_database,
            max_concurrent=args.max_concurrent,
            random_seed=args.random_seed,
            timeout_seconds=args.timeout,
            verbose=not args.quiet
        )

        if result['success']:
            sys.exit(0)
        else:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()