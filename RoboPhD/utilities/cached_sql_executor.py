#!/usr/bin/env python3
"""
Cached SQL Executor for BIRD Text-to-SQL Evaluation
Provides dual caching for ground truth and prediction queries
"""

import json
import sqlite3
import hashlib
import time
import os
import threading
import queue
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from contextlib import contextmanager


class SQLTimeoutError(Exception):
    """Exception raised when SQL query times out"""
    pass


class ReadOnlyConnectionPool:
    """Thread-safe connection pool for read-only SQLite access"""
    
    def __init__(self, max_connections_per_db: int = 5):
        self.max_connections_per_db = max_connections_per_db
        self.pools = {}  # db_path -> list of connections
        self.pool_locks = {}  # db_path -> lock
        self.connection_counts = {}  # db_path -> current count
        self.global_lock = threading.Lock()
    
    def _clear_db_locks_if_needed(self, db_path: str):
        """Automatically clear SQLite lock files since we're read-only"""
        try:
            lock_files = [
                f"{db_path}-wal",
                f"{db_path}-shm", 
                f"{db_path}-journal"
            ]
            
            for lock_file in lock_files:
                if os.path.exists(lock_file):
                    os.remove(lock_file)
                    # Log lock file cleanup at debug level
                    import logging
                    logging.debug(f"Auto-cleared read-only lock file: {lock_file}")
        except Exception as e:
            # Log warning at debug level - not critical
            import logging
            logging.debug(f"Could not clear lock files for {db_path}: {e}")
    
    def _create_optimized_connection(self, db_path: str) -> sqlite3.Connection:
        """Create connection with read-only optimizations"""
        # Auto-clear any lock files first (safe for read-only)
        self._clear_db_locks_if_needed(db_path)

        # Create connection with optimizations
        # Note: We do NOT use mode=ro as SQLite needs to create temporary tables
        # for some queries (e.g., ORDER BY, GROUP BY operations).
        # The PRAGMA query_only setting below prevents actual data modifications.
        conn = sqlite3.connect(db_path, timeout=10.0, check_same_thread=False)

        # Read-only PRAGMA optimizations
        conn.execute("PRAGMA query_only = 1")  # Prevent writes to the database
        conn.execute("PRAGMA read_uncommitted = 1")  # Faster reads
        conn.execute("PRAGMA synchronous = OFF")  # No sync for reads
        conn.execute("PRAGMA journal_mode = WAL")  # WAL mode for concurrent reads
        conn.execute("PRAGMA temp_store = MEMORY")  # Use memory for temp tables
        conn.execute("PRAGMA mmap_size = 268435456")  # 256MB mmap for faster reads
        conn.execute("PRAGMA cache_size = -64000")  # 64MB cache (negative = KB)
        conn.execute("PRAGMA page_size = 4096")  # Optimal page size for read performance

        return conn
    
    @contextmanager
    def get_connection(self, db_path: str):
        """Get a pooled connection with context manager"""
        # Initialize pool for this database if needed
        with self.global_lock:
            if db_path not in self.pools:
                self.pools[db_path] = []
                self.pool_locks[db_path] = threading.Lock()
                self.connection_counts[db_path] = 0
        
        pool_lock = self.pool_locks[db_path]
        conn = None
        
        try:
            # Try to get existing connection from pool
            with pool_lock:
                if self.pools[db_path]:
                    conn = self.pools[db_path].pop()
                elif self.connection_counts[db_path] < self.max_connections_per_db:
                    conn = self._create_optimized_connection(db_path)
                    self.connection_counts[db_path] += 1
            
            # If no connection available, create temporary one
            if conn is None:
                conn = self._create_optimized_connection(db_path)
                temp_connection = True
            else:
                temp_connection = False
            
            yield conn
            
        finally:
            # Return connection to pool or close if temporary
            if conn and not temp_connection:
                with pool_lock:
                    if len(self.pools[db_path]) < self.max_connections_per_db:
                        self.pools[db_path].append(conn)
                    else:
                        conn.close()
                        self.connection_counts[db_path] -= 1
            elif conn and temp_connection:
                conn.close()
    
    def close_database(self, db_path: str):
        """Close all connections for a specific database"""
        with self.global_lock:
            if db_path in self.pools:
                with self.pool_locks[db_path]:
                    for conn in self.pools[db_path]:
                        try:
                            conn.close()
                        except:
                            pass  # Ignore close errors
                    self.pools[db_path].clear()
                    self.connection_counts[db_path] = 0

    def close_all(self):
        """Close all pooled connections"""
        with self.global_lock:
            for db_path, pool in self.pools.items():
                with self.pool_locks[db_path]:
                    for conn in pool:
                        conn.close()
                    pool.clear()
                    self.connection_counts[db_path] = 0


# Global connection pool instance
_connection_pool = ReadOnlyConnectionPool()

# Cleanup handler for proper connection closing
import atexit
atexit.register(_connection_pool.close_all)


def close_database_connections(db_path: str):
    """
    Close all cached connections for a specific database.
    Call this after SQL generation to release database locks before evaluation.
    """
    _connection_pool.close_database(db_path)


def close_all_connections():
    """
    Close all pooled database connections.
    Call this between iterations to prevent file descriptor accumulation.
    """
    _connection_pool.close_all()


def execute_sql_with_timeout(db_path: str, sql: str, timeout_seconds: int) -> List[Tuple]:
    """
    Execute SQL with timeout using optimized read-only connection pool
    Much faster due to connection reuse and read-only optimizations
    """
    result_queue = queue.Queue()
    exception_queue = queue.Queue()
    
    def target():
        try:
            # Use pooled connection with read-only optimizations
            with _connection_pool.get_connection(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(sql)
                results = cursor.fetchall()
                result_queue.put(results)
        except Exception as e:
            exception_queue.put(e)
    
    # Start execution in a separate thread
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    
    # Wait for completion or timeout
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        # Thread is still running - timeout occurred
        # Note: We can't actually kill the thread, but we can abandon it
        raise SQLTimeoutError(f"SQL query timed out after {timeout_seconds} seconds")
    
    # Check for exceptions
    if not exception_queue.empty():
        raise exception_queue.get()
    
    # Return results
    if not result_queue.empty():
        return result_queue.get()
    else:
        raise RuntimeError("SQL execution completed but no result returned")


def check_database_lock(db_path: str) -> bool:
    """
    Check if database has active locks (with auto-clearing for read-only)
    Returns True if database appears to be locked after clearing attempts
    """
    try:
        # First try with pooled connection (will auto-clear locks)
        with _connection_pool.get_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            return False  # No lock
        
    except sqlite3.OperationalError as e:
        if "database is locked" in str(e).lower():
            # Try manual lock clearing as last resort
            try:
                lock_files = [f"{db_path}-wal", f"{db_path}-shm", f"{db_path}-journal"]
                for lock_file in lock_files:
                    if os.path.exists(lock_file):
                        os.remove(lock_file)
                        print(f"Cleared persistent lock file: {lock_file}")
                
                # Try again after clearing
                with _connection_pool.get_connection(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
                    return False
            except Exception:
                pass
            
            return True  # Still locked after clearing attempts
        raise  # Other error
    except Exception:
        return True  # Assume locked if any other issue


def normalize_sql(sql: str) -> str:
    """Normalize SQL for consistent cache keys"""
    return ' '.join(sql.strip().split())


def generate_cache_key(sql: str) -> str:
    """
    Generate cache key from SQL query.

    Args:
        sql: SQL query to cache

    Returns:
        Cache key based on normalized SQL only

    Note: Schema hash is not needed because caches are already database-scoped
    (each database has its own cache file).
    """
    sql_normalized = normalize_sql(sql)
    return hashlib.md5(sql_normalized.encode()).hexdigest()


class SQLCache:
    """Cache for SQL execution results with size limits"""

    MAX_RESULT_SIZE = 2500  # Max rows to cache (increased for slow databases like retails, sales_in_weather)
    MAX_RESULT_BYTES = 1024 * 1024  # 1MB limit

    def __init__(self, cache_file: str):
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self._write_lock = threading.Lock()  # Protect concurrent writes
    
    def _load_cache(self) -> Dict:
        """Load cache from file"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                print(f"Warning: Could not load cache from {self.cache_file}, starting fresh")
        return {}
    
    def _save_cache(self):
        """Save cache to file"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save cache to {self.cache_file}: {e}")
    
    def should_cache(self, results: List[Tuple]) -> bool:
        """Only cache if results are reasonably sized"""
        # Check size without deduplication (evolution agents need to see actual results)
        if len(results) > self.MAX_RESULT_SIZE:
            return False

        # Quick size check on actual results
        estimated_size = len(str(results).encode())
        return estimated_size <= self.MAX_RESULT_BYTES
    
    def get_cached_results(self, sql: str) -> Optional[List[Tuple]]:
        """Get cached results or None if not cached"""
        cache_key = generate_cache_key(sql)
        cached_entry = self.cache.get(cache_key)

        if cached_entry:
            # Convert back to tuples (JSON serializes tuples as lists)
            return [tuple(row) for row in cached_entry['results']]
        else:
            # DEBUG: Log cache misses to help diagnose lookup failures
            import logging
            logging.debug(f"Cache miss: key={cache_key}, sql={sql[:80]}...")

        return None
    
    def cache_results(self, sql: str, results: List[Tuple]):
        """Cache results only if they're small enough"""
        if self.should_cache(results):
            cache_key = generate_cache_key(sql)

            # Store actual results (no deduplication - evolution agents need real output)
            # Note: BIRD evaluation still uses set() comparison, but cache stores full results

            # Thread-safe cache update and file write
            with self._write_lock:
                # Store metadata with results
                self.cache[cache_key] = {
                    'results': results,  # JSON will convert tuples to lists
                    'timestamp': time.time(),
                    'row_count': len(results),
                    'sql_preview': sql[:400] + ('...' if len(sql) > 400 else '')
                }

                self._save_cache()
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'cache_file': self.cache_file,
            'total_entries': len(self.cache),
            'cache_size_mb': os.path.getsize(self.cache_file) / (1024*1024) if os.path.exists(self.cache_file) else 0
        }


class CachedSQLExecutor:
    """Execute SQL with database-scoped dual caching for ground truth and predictions"""
    
    def __init__(self, cache_dir: str = "cache"):
        """
        Initialize with database-scoped cache structure.
        Cache files are organized per database for isolation.
        """
        self.cache_dir = cache_dir
        self.db_caches = {}  # {(db_name, cache_type): SQLCache}
        
        # Ensure cache directories exist
        os.makedirs(f"{cache_dir}/ground_truth", exist_ok=True)
        os.makedirs(f"{cache_dir}/predictions", exist_ok=True)
        
        # Statistics per database
        self.stats = {}  # {db_name: {stat_type: count}}
    
    def _get_db_name(self, db_path: str) -> str:
        """Extract database name from path"""
        # Extract database name from path like .../dev_databases/california_schools/california_schools.sqlite
        path = Path(db_path)
        
        # Check if parent directory name matches the database pattern
        # (typical for BIRD where database folder contains database.sqlite)
        parent_name = path.parent.name
        if path.stem == parent_name:
            return parent_name
        
        # Otherwise use the filename without extension
        return path.stem
    
    def _get_cache(self, db_name: str, is_ground_truth: bool) -> SQLCache:
        """Get or create database-specific cache"""
        cache_type = 'ground_truth' if is_ground_truth else 'predictions'
        cache_key = (db_name, cache_type)

        if cache_key not in self.db_caches:
            cache_file = f"{self.cache_dir}/{cache_type}/{db_name}.json"
            self.db_caches[cache_key] = SQLCache(cache_file)

        return self.db_caches[cache_key]
    
    def _get_stats(self, db_name: str) -> Dict:
        """Get or create database-specific statistics"""
        if db_name not in self.stats:
            self.stats[db_name] = {
                'gt_cache_hits': 0,
                'gt_cache_misses': 0,
                'pred_cache_hits': 0,
                'pred_cache_misses': 0
            }
        return self.stats[db_name]
    
    def execute_sql(self, sql: str, db_path: str, is_ground_truth: bool = False, timeout_seconds: int = 30) -> List[Tuple]:
        """Execute SQL with database-scoped caching"""
        db_name = self._get_db_name(db_path)
        cache = self._get_cache(db_name, is_ground_truth)
        db_stats = self._get_stats(db_name)
        stat_prefix = 'gt' if is_ground_truth else 'pred'

        # Check cache first
        cached_results = cache.get_cached_results(sql)
        if cached_results is not None:
            db_stats[f'{stat_prefix}_cache_hits'] += 1
            return cached_results
        
        # Cache miss - execute
        db_stats[f'{stat_prefix}_cache_misses'] += 1
        
        # Auto-clear locks and use optimized connection pool
        results = execute_sql_with_timeout(db_path, sql, timeout_seconds)
        
        # Cache only if results are small enough
        cache.cache_results(sql, results)
        
        return results
    
    def get_cache_stats(self) -> Dict:
        """Get comprehensive cache statistics across all databases"""
        # Aggregate statistics across all databases
        total_stats = {
            'gt_cache_hits': 0,
            'gt_cache_misses': 0,
            'pred_cache_hits': 0,
            'pred_cache_misses': 0
        }
        
        per_db_stats = {}
        for db_name, db_stats in self.stats.items():
            per_db_stats[db_name] = db_stats.copy()
            for stat_key, value in db_stats.items():
                total_stats[stat_key] += value
        
        total_gt = total_stats['gt_cache_hits'] + total_stats['gt_cache_misses']
        total_pred = total_stats['pred_cache_hits'] + total_stats['pred_cache_misses']
        
        # Get cache file sizes
        gt_cache_size = 0
        pred_cache_size = 0
        gt_entries = 0
        pred_entries = 0
        
        for (db_name, cache_type), cache in self.db_caches.items():
            cache_stats = cache.get_cache_stats()
            if cache_type == 'ground_truth':
                gt_cache_size += cache_stats['cache_size_mb']
                gt_entries += cache_stats['total_entries']
            else:
                pred_cache_size += cache_stats['cache_size_mb']
                pred_entries += cache_stats['total_entries']
        
        return {
            'execution_stats': total_stats,
            'per_database_stats': per_db_stats,
            'ground_truth_cache': {
                'total_entries': gt_entries,
                'cache_size_mb': gt_cache_size,
                'hits': total_stats['gt_cache_hits'],
                'misses': total_stats['gt_cache_misses'],
                'hit_rate': total_stats['gt_cache_hits'] / total_gt * 100 if total_gt > 0 else 0
            },
            'prediction_cache': {
                'total_entries': pred_entries,
                'cache_size_mb': pred_cache_size,
                'hits': total_stats['pred_cache_hits'],
                'misses': total_stats['pred_cache_misses'],
                'hit_rate': total_stats['pred_cache_hits'] / total_pred * 100 if total_pred > 0 else 0
            }
        }


def _truncate_results(results: List[Tuple], max_rows: int = 50, max_size_mb: float = 0.5) -> Dict:
    """
    Truncate results to prevent massive file sizes.

    Applies set() deduplication before truncation to match BIRD evaluation methodology
    and reduce file sizes.

    Args:
        results: SQL query results
        max_rows: Maximum number of rows to keep (default: 50, reduced from 100)
        max_size_mb: Maximum size in MB for the results (default: 0.5, reduced from 1.0)

    Returns:
        Dict with:
        - 'data': Deduplicated and truncated results (sorted for consistency)
        - 'truncated': Whether truncation occurred
        - 'deduplicated': Whether any duplicates were removed
        - 'original_rows': Count before deduplication
        - 'deduplicated_rows': Count after deduplication
        - 'kept_rows': Count after truncation
    """
    if results is None:
        return {
            "data": None,
            "truncated": False,
            "deduplicated": False,
            "original_rows": 0,
            "deduplicated_rows": 0,
            "kept_rows": 0
        }

    # Deduplicate using set (matches BIRD evaluation logic), then sort for consistency
    import json
    original_count = len(results)
    deduplicated = sorted(set(results))  # set removes duplicates, sorted ensures consistency
    deduplicated_count = len(deduplicated)
    deduplicated_flag = deduplicated_count < original_count

    # Now apply truncation to deduplicated results
    truncated = False
    truncated_results = []
    total_size = 0

    for i, row in enumerate(deduplicated):
        # Check row count limit
        if i >= max_rows:
            truncated = True
            break
            
        # Check size limit
        row_str = json.dumps(row, default=str)
        row_size = len(row_str)
        
        # If a single row is too large, truncate its content
        if row_size > max_size_mb * 1024 * 1024:
            truncated = True
            # Truncate individual cell values that are too large
            truncated_row = []
            for cell in row:
                cell_str = str(cell)
                if len(cell_str) > 1000:  # Truncate cells larger than 1000 chars
                    truncated_row.append(cell_str[:1000] + f"... [truncated, {len(cell_str)} chars total]")
                else:
                    truncated_row.append(cell)
            truncated_results.append(tuple(truncated_row))
        else:
            # Check cumulative size
            if total_size + row_size > max_size_mb * 1024 * 1024:
                truncated = True
                break
            truncated_results.append(row)
            total_size += row_size
    
    return {
        "data": truncated_results,
        "truncated": truncated,
        "deduplicated": deduplicated_flag,
        "original_rows": original_count,
        "deduplicated_rows": deduplicated_count,
        "kept_rows": len(truncated_results)
    }


def compare_execution_results(predicted_sql: str, ground_truth_sql: str, 
                            db_path: str, executor: CachedSQLExecutor, 
                            pred_timeout_seconds: int = 30, gt_timeout_seconds: int = 300) -> Dict:
    """Compare results with proper error handling and debugging info"""
    
    # Extract database name for consistent logging
    from pathlib import Path
    from datetime import datetime
    db_name = Path(db_path).stem if db_path else "unknown"
    
    # Execute prediction
    try:
        predicted_res = executor.execute_sql(predicted_sql, db_path, is_ground_truth=False, timeout_seconds=pred_timeout_seconds)
        pred_error = None
        pred_timeout = False
    except SQLTimeoutError as e:
        # Timeouts are expected for complex queries
        import logging
        logging.debug(f"Prediction SQL timed out for {db_name}: {e}")
        predicted_res = None
        pred_error = str(e)
        pred_timeout = True
    except Exception as e:
        # SQL errors are expected - models don't always generate valid SQL
        # Log quietly to debug level instead of printing to stdout
        import logging
        logging.debug(f"SQL execution error for {db_name}: {e}")
        predicted_res = None
        pred_error = str(e)
        pred_timeout = False
    
    # Execute ground truth with retry for database locks
    ground_truth_res = None
    gt_error = None
    gt_timeout = False
    max_retries = 3

    for attempt in range(max_retries):
        try:
            ground_truth_res = executor.execute_sql(ground_truth_sql, db_path, is_ground_truth=True, timeout_seconds=gt_timeout_seconds)
            gt_error = None
            gt_timeout = False
            break  # Success
        except SQLTimeoutError as e:
            # Ground truth timeouts indicate potential issues with the dataset
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"    [{timestamp}] WARNING: Ground truth SQL timed out for {db_name} - may need investigation")
            import logging
            logging.debug(f"GT timeout details for {db_name}: {e}")
            ground_truth_res = None
            gt_error = str(e)
            gt_timeout = True
            break  # Don't retry timeouts
        except Exception as e:
            # Check if it's a database lock error and we have retries left
            if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                # Wait and retry with exponential backoff
                import time
                wait_time = 1.0 * (2 ** attempt)  # 1s, 2s, 4s
                time.sleep(wait_time)
                continue
            # Not a lock error or last attempt - log and break
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"    [{timestamp}] WARNING: Ground truth SQL error for {db_name} - dataset issue: {e}")
            import logging
            logging.debug(f"GT SQL for {db_name}: {ground_truth_sql}")
            ground_truth_res = None
            gt_error = str(e)
            gt_timeout = False
            break
    
    # Determine status with GT issues taking priority
    if gt_error and gt_timeout:
        # This shouldn't happen (timeout is subset of error), but handle gracefully
        status = "gt_timeout" if "timed out" in gt_error.lower() else "gt_error"
    elif gt_error:
        status = "gt_error"
    elif gt_timeout:
        status = "gt_timeout"
    elif pred_error and pred_timeout:
        # This shouldn't happen (timeout is subset of error), but handle gracefully
        status = "pred_timeout" if "timed out" in pred_error.lower() else "pred_error"
    elif pred_error:
        status = "pred_error"
    elif pred_timeout:
        status = "pred_timeout"
    else:
        # Both succeeded - compare results using set comparison (BIRD evaluation logic)
        if set(predicted_res) == set(ground_truth_res):
            status = "match"
        else:
            status = "mismatch"
    
    # Truncate results to prevent massive files
    pred_truncated = _truncate_results(predicted_res)
    gt_truncated = _truncate_results(ground_truth_res)
    
    return {
        "matches": status == "match",
        "status": status,
        "predicted_results": pred_truncated["data"],
        "ground_truth_results": gt_truncated["data"],
        "predicted_results_truncated": pred_truncated["truncated"],
        "ground_truth_results_truncated": gt_truncated["truncated"],
        "predicted_results_deduplicated": pred_truncated.get("deduplicated", False),
        "ground_truth_results_deduplicated": gt_truncated.get("deduplicated", False),
        "predicted_error": pred_error,
        "ground_truth_error": gt_error
    }