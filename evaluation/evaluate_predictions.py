#!/usr/bin/env python3
"""
BIRD Text-to-SQL Evaluation Script
Evaluates predictions with dual caching and proper error tracking
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utilities.cached_sql_executor import CachedSQLExecutor, compare_execution_results


def load_predictions(predictions_file: str) -> Dict[str, str]:
    """Load predictions from JSON file"""
    with open(predictions_file, 'r') as f:
        data = json.load(f)
    
    # Handle both individual predictions files and merged predictions
    if 'predictions' in data:
        return data['predictions']
    else:
        return data


def load_dev_data(dev_data_file: str) -> Dict[str, Dict]:
    """Load dev data and index by question_id"""
    with open(dev_data_file, 'r') as f:
        dev_data = json.load(f)
    
    # Handle both dev (with question_id) and train (without) formats
    result = {}
    for i, item in enumerate(dev_data):
        if 'question_id' in item:
            key = str(item['question_id'])
        else:
            # For train data, use index as question_id
            key = str(i)
        result[key] = item
    
    return result


def extract_ground_truth_from_dev_data(dev_data: Dict[str, Dict]) -> Dict[str, str]:
    """Extract ground truth SQL from dev data"""
    ground_truth = {}
    
    for question_id_str, item in dev_data.items():
        # Get the SQL from dev data
        sql = item.get('SQL')
        if sql:
            ground_truth[question_id_str] = sql
    
    return ground_truth


def extract_sql_from_prediction(prediction: str) -> str:
    """Extract SQL from BIRD prediction format"""
    # Format: "SQL\t----- bird -----\tdb_name"
    parts = prediction.split('\t----- bird -----\t')
    return parts[0].strip()


def extract_db_from_prediction(prediction: str) -> str:
    """Extract database name from BIRD prediction format"""
    # Format: "SQL\t----- bird -----\tdb_name"
    parts = prediction.split('\t----- bird -----\t')
    if len(parts) >= 2:
        return parts[1].strip()
    else:
        raise ValueError(f"Invalid prediction format: {prediction}")


def evaluate_predictions(predictions_file: str, dev_data_file: str, 
                        db_root: str,
                        timeout_seconds: int = 30,
                        gt_timeout_seconds: int = 300,
                        verbose: bool = True) -> Dict[str, Any]:
    """Evaluate predictions with cached execution and proper error tracking"""
    
    # Set up database-scoped cache directory structure
    cache_root = "cache"
    os.makedirs(f"{cache_root}/ground_truth", exist_ok=True)
    os.makedirs(f"{cache_root}/predictions", exist_ok=True)
    
    # Initialize database-scoped cache executor
    # Cache files will be organized per database:
    #   cache/ground_truth/{database_name}.json
    #   cache/predictions/{database_name}.json
    if verbose:
        print(f"Using database-scoped cache structure:")
        print(f"  Ground truth: {cache_root}/ground_truth/{{database}}.json")
        print(f"  Predictions: {cache_root}/predictions/{{database}}.json")
    
    executor = CachedSQLExecutor(cache_dir=cache_root)
    
    # Load data
    if verbose:
        print("Loading data...")
    predictions = load_predictions(predictions_file)
    dev_data = load_dev_data(dev_data_file)
    ground_truth_queries = extract_ground_truth_from_dev_data(dev_data)
    
    if verbose:
        print(f"Evaluating {len(predictions)} predictions...")
    
    results = {}
    error_counts = {
        "gt_error": 0, "gt_timeout": 0,  # GT issues (highest priority)
        "match": 0, "mismatch": 0,       # Both succeeded
        "pred_error": 0, "pred_timeout": 0  # Pred issues only
    }
    
    for i, (question_id, prediction) in enumerate(predictions.items(), 1):
        if i % 100 == 0 and verbose:
            print(f"Processed {i}/{len(predictions)} questions...")
        
        try:
            predicted_sql = extract_sql_from_prediction(prediction)
            
            # Look up ground truth - handle missing questions
            if question_id not in ground_truth_queries:
                if verbose:
                    print(f"Warning: No ground truth found for question {question_id}")
                comparison_result = {
                    "matches": False,
                    "status": "missing_ground_truth",
                    "predicted_results": None,
                    "ground_truth_results": None,
                    "predicted_error": f"No ground truth found for question {question_id}",
                    "ground_truth_error": None
                }
            else:
                ground_truth_sql = ground_truth_queries[question_id]
                db_name = extract_db_from_prediction(prediction)
                db_path = os.path.join(db_root, db_name, f"{db_name}.sqlite")
                
                if not os.path.exists(db_path):
                    if verbose:
                        print(f"Warning: Database not found: {db_path}")
                    comparison_result = {
                        "matches": False,
                        "status": "db_not_found",
                        "predicted_results": None,
                        "ground_truth_results": None,
                        "predicted_error": f"Database not found: {db_path}",
                        "ground_truth_error": None
                    }
                else:
                    comparison_result = compare_execution_results(predicted_sql, ground_truth_sql, db_path, executor, timeout_seconds, gt_timeout_seconds)
            
            # Track status counts
            status = comparison_result["status"]
            if status in error_counts:
                error_counts[status] += 1
            else:
                error_counts["pred_error"] += 1  # Default for unknown errors
            
            # Extract question context from dev_data
            question_data = dev_data.get(question_id, {})

            results[question_id] = {
                # Context first (what was asked)
                "question": question_data.get("question", ""),
                "evidence": question_data.get("evidence", ""),
                "difficulty": question_data.get("difficulty", ""),

                # Execution results
                "matches": comparison_result["matches"],
                "status": comparison_result["status"],
                "predicted_error": comparison_result["predicted_error"],
                "ground_truth_error": comparison_result["ground_truth_error"],

                # SQL queries
                "predicted_sql": predicted_sql,
                "ground_truth_sql": ground_truth_sql,
                "database": db_name,

                # Query results (deduplicated, sorted, truncated)
                "predicted_results": comparison_result["predicted_results"],
                "ground_truth_results": comparison_result["ground_truth_results"],

                # Result metadata flags
                "predicted_results_truncated": comparison_result.get("predicted_results_truncated", False),
                "ground_truth_results_truncated": comparison_result.get("ground_truth_results_truncated", False),
                "predicted_results_deduplicated": comparison_result.get("predicted_results_deduplicated", False),
                "ground_truth_results_deduplicated": comparison_result.get("ground_truth_results_deduplicated", False)
            }
            
        except Exception as e:
            # Log errors quietly - they're part of normal evaluation
            import logging
            logging.debug(f"Error processing question {question_id}: {e}")
            error_counts["pred_error"] += 1

            # Extract question context from dev_data
            question_data = dev_data.get(question_id, {})

            results[question_id] = {
                # Context first (what was asked)
                "question": question_data.get("question", ""),
                "evidence": question_data.get("evidence", ""),
                "difficulty": question_data.get("difficulty", ""),

                # Execution results
                "matches": False,
                "status": "processing_error",
                "predicted_error": str(e),
                "ground_truth_error": None,

                # SQL queries
                "predicted_sql": None,
                "ground_truth_sql": None,
                "database": None,

                # Query results
                "predicted_results": None,
                "ground_truth_results": None,

                # Result metadata flags
                "predicted_results_truncated": False,
                "ground_truth_results_truncated": False,
                "predicted_results_deduplicated": False,
                "ground_truth_results_deduplicated": False
            }
    
    # Calculate final statistics
    total_questions = len(results)
    matches = sum(1 for r in results.values() if r["matches"])
    accuracy = matches / total_questions if total_questions > 0 else 0
    
    # Calculate data quality metrics
    data_quality_issues = error_counts["gt_error"] + error_counts["gt_timeout"]
    evaluable_questions = total_questions - data_quality_issues
    evaluable_accuracy = matches / evaluable_questions if evaluable_questions > 0 else 0
    
    # Report summary
    if verbose:
        print(f"\n=== EVALUATION SUMMARY ===")
        print(f"Total questions: {total_questions}")
        if data_quality_issues > 0:
            print(f"Data quality issues: {data_quality_issues} ({data_quality_issues/total_questions*100:.1f}%)")
            print(f"  - GT errors: {error_counts['gt_error']}")
            print(f"  - GT timeouts: {error_counts['gt_timeout']}")
            print(f"Evaluable questions: {evaluable_questions}")
            print(f"Correct predictions: {matches}/{evaluable_questions} ({evaluable_accuracy:.1%})")
        else:
            print(f"Correct answers: {matches}")
            print(f"Overall accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print(f"\n=== STATUS BREAKDOWN ===")
        print(f"Matches: {error_counts['match']}")
        print(f"Mismatches: {error_counts['mismatch']}")
        print(f"Prediction errors: {error_counts['pred_error']}")
        print(f"Prediction timeouts: {error_counts['pred_timeout']}")
        print(f"Ground truth errors: {error_counts['gt_error']}")
        print(f"Ground truth timeouts: {error_counts['gt_timeout']}")
        
        if data_quality_issues > 0:
            print(f"\nWARNING: {data_quality_issues} ground truth issues need resolution!")
        
        # Get cache statistics
        cache_stats = executor.get_cache_stats()
        print(f"\n=== CACHE STATISTICS ===")
        print(f"Ground truth cache hits: {cache_stats['execution_stats']['gt_cache_hits']}")
        print(f"Ground truth cache misses: {cache_stats['execution_stats']['gt_cache_misses']}")
        print(f"Ground truth hit rate: {cache_stats['ground_truth_cache']['hit_rate']:.2%}")
        print(f"Prediction cache hits: {cache_stats['execution_stats']['pred_cache_hits']}")
        print(f"Prediction cache misses: {cache_stats['execution_stats']['pred_cache_misses']}")
        print(f"Prediction hit rate: {cache_stats['prediction_cache']['hit_rate']:.2%}")
    else:
        # Get cache statistics even when not verbose, for the return value
        cache_stats = executor.get_cache_stats()
    
    return {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "predictions_file": predictions_file,
            "total_questions": total_questions,
            "accuracy": accuracy,
            "error_counts": error_counts,
            "cache_stats": cache_stats
        },
        "results": results
    }


def analyze_by_database(results: Dict[str, Dict]) -> Dict[str, Dict]:
    """Analyze results by database"""
    db_stats = {}
    
    for result in results.values():
        db_name = result.get("database")
        if db_name:
            if db_name not in db_stats:
                db_stats[db_name] = {
                    "total": 0, "correct": 0, 
                    "errors": {
                        "gt_error": 0, "gt_timeout": 0,
                        "mismatch": 0, "pred_error": 0, "pred_timeout": 0
                    }
                }
            
            db_stats[db_name]["total"] += 1
            if result["matches"]:
                db_stats[db_name]["correct"] += 1
            else:
                status = result["status"]
                if status in db_stats[db_name]["errors"]:
                    db_stats[db_name]["errors"][status] += 1
                else:
                    # Handle legacy or unexpected statuses
                    if status in ["processing_error", "db_not_found"]:
                        db_stats[db_name]["errors"]["pred_error"] += 1
                    elif status in ["both_error"]:  # Legacy status
                        db_stats[db_name]["errors"]["gt_error"] += 1
    
    # Calculate accuracy per database
    for db_name, stats in db_stats.items():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
    
    return db_stats


def main():
    parser = argparse.ArgumentParser(description='BIRD Text-to-SQL Evaluation')
    parser.add_argument('--predictions', required=True, help='Path to predictions JSON file')
    parser.add_argument('--dev_data', required=True, help='Path to dev.json')
    parser.add_argument('--db_root', required=True, help='Path to dev_databases/ directory')
    parser.add_argument('--output', help='Output file for detailed results')
    parser.add_argument('--clear-pred-cache', action='store_true', help='Clear all prediction caches before evaluation')
    parser.add_argument('--clear-gt-cache', action='store_true', help='Clear all ground truth caches before evaluation')
    parser.add_argument('--clear-db-cache', help='Clear cache for specific database (e.g., california_schools)')
    parser.add_argument('--timeout', type=int, default=30, help='SQL query timeout in seconds for predictions (default: 30)')
    parser.add_argument('--gt_timeout', type=int, default=300, help='SQL query timeout in seconds for ground truth (default: 300)')
    
    args = parser.parse_args()
    
    # Handle cache clearing options
    if args.clear_gt_cache:
        # Clear all ground truth caches
        gt_cache_dir = "cache/ground_truth"
        if os.path.exists(gt_cache_dir):
            for cache_file in os.listdir(gt_cache_dir):
                if cache_file.endswith('.json'):
                    os.remove(os.path.join(gt_cache_dir, cache_file))
            print(f"✅ Cleared all ground truth caches in {gt_cache_dir}")
        else:
            print(f"ℹ️  Ground truth cache directory not found: {gt_cache_dir}")
    
    if args.clear_pred_cache:
        # Clear all prediction caches
        pred_cache_dir = "cache/predictions"
        if os.path.exists(pred_cache_dir):
            for cache_file in os.listdir(pred_cache_dir):
                if cache_file.endswith('.json'):
                    os.remove(os.path.join(pred_cache_dir, cache_file))
            print(f"✅ Cleared all prediction caches in {pred_cache_dir}")
        else:
            print(f"ℹ️  Prediction cache directory not found: {pred_cache_dir}")
    
    if args.clear_db_cache:
        # Clear cache for specific database
        db_name = args.clear_db_cache
        cleared = False
        
        gt_file = f"cache/ground_truth/{db_name}.json"
        if os.path.exists(gt_file):
            os.remove(gt_file)
            print(f"✅ Cleared ground truth cache for {db_name}")
            cleared = True
        
        pred_file = f"cache/predictions/{db_name}.json"
        if os.path.exists(pred_file):
            os.remove(pred_file)
            print(f"✅ Cleared prediction cache for {db_name}")
            cleared = True
        
        if not cleared:
            print(f"ℹ️  No cache files found for database: {db_name}")
    
    # Run evaluation
    evaluation_results = evaluate_predictions(
        args.predictions, args.dev_data, args.db_root,
        args.timeout, args.gt_timeout
    )
    
    # Add database analysis
    evaluation_results["database_analysis"] = analyze_by_database(evaluation_results["results"])
    
    # Save detailed results
    if args.output:
        print(f"\nSaving detailed results to {args.output}")
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
    
    print(f"\nEvaluation complete!")


if __name__ == "__main__":
    main()