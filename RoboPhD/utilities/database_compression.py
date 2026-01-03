"""
Database compression utilities for RoboPhD.

Provides transparent on-demand decompression of large databases with automatic
cache management (at most 1 large database uncompressed at a time).

Large databases are auto-detected as any database with a .tar.gz file.
"""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Set

logger = logging.getLogger(__name__)


def get_large_databases(db_root: Path) -> Set[str]:
    """
    Auto-detect large databases (those with .tar.gz files).

    Args:
        db_root: Path to database root directory

    Returns:
        Set of database names that have .tar.gz files
    """
    large_dbs = set()

    if not db_root.exists():
        return large_dbs

    for item in db_root.iterdir():
        # Check for .tar.gz files (compressed databases)
        if item.is_file() and item.suffix == '.gz' and item.stem.endswith('.tar'):
            # Extract database name (e.g., "bike_share_1.tar.gz" -> "bike_share_1")
            db_name = item.stem[:-4]  # Remove ".tar" from "bike_share_1.tar"
            large_dbs.add(db_name)

    return large_dbs


def get_currently_uncompressed_large_db(db_root: Path, large_dbs: Set[str]) -> Optional[str]:
    """
    Find which large database (if any) is currently uncompressed.

    Stateless check - scans filesystem each time.

    Args:
        db_root: Path to database root directory
        large_dbs: Set of large database names

    Returns:
        Name of uncompressed large database, or None if none are uncompressed
    """
    for db_name in large_dbs:
        db_dir = db_root / db_name
        if db_dir.exists() and db_dir.is_dir():
            # Check if it actually contains a .sqlite file (confirms it's uncompressed)
            sqlite_file = db_dir / f"{db_name}.sqlite"
            if sqlite_file.exists():
                return db_name

    return None


def ensure_database_decompressed(db_root: Path, db_name: str) -> Path:
    """
    Ensure a database is decompressed and available.

    Implements automatic cache management:
    - If database directory exists: return immediately (fast path)
    - If .tar.gz exists and database needed:
      1. Delete any other uncompressed large database (cache eviction)
      2. Decompress the requested database
    - If neither exists: raise FileNotFoundError

    This function is NOT thread-safe but is only called during single-threaded
    database selection (before any parallel processing starts).

    Args:
        db_root: Path to database root directory
        db_name: Name of database to ensure is decompressed

    Returns:
        Path to database directory

    Raises:
        FileNotFoundError: If database doesn't exist (compressed or uncompressed)
        RuntimeError: If decompression fails
    """
    db_dir = db_root / db_name
    compressed_file = db_root / f"{db_name}.tar.gz"

    # Fast path: database already uncompressed
    if db_dir.exists() and db_dir.is_dir():
        sqlite_file = db_dir / f"{db_name}.sqlite"
        if sqlite_file.exists():
            return db_dir

    # Check if compressed version exists
    if not compressed_file.exists():
        # Database doesn't exist at all
        raise FileNotFoundError(
            f"Database '{db_name}' not found. "
            f"Checked for directory: {db_dir} and compressed file: {compressed_file}"
        )

    # Need to decompress - first, evict any other uncompressed large database
    large_dbs = get_large_databases(db_root)
    currently_uncompressed = get_currently_uncompressed_large_db(db_root, large_dbs)

    if currently_uncompressed and currently_uncompressed != db_name:
        logger.info(f"  Evicting previously uncompressed large database: {currently_uncompressed}")
        evict_dir = db_root / currently_uncompressed
        try:
            shutil.rmtree(evict_dir)
            logger.info(f"  ✓ Deleted {currently_uncompressed} directory")
        except Exception as e:
            logger.warning(f"  ⚠ Failed to delete {currently_uncompressed}: {e}")
            # Continue anyway - not fatal

    # Decompress the requested database
    logger.info(f"  Decompressing {db_name} from {compressed_file.name}...")

    try:
        # Use tar to decompress: tar -xzf database.tar.gz -C db_root/
        result = subprocess.run(
            ['tar', '-xzf', str(compressed_file), '-C', str(db_root)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout for large databases
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Decompression failed with exit code {result.returncode}. "
                f"stderr: {result.stderr}"
            )

        # Verify decompression succeeded
        if not db_dir.exists():
            raise RuntimeError(
                f"Decompression appeared to succeed but directory not found: {db_dir}"
            )

        sqlite_file = db_dir / f"{db_name}.sqlite"
        if not sqlite_file.exists():
            raise RuntimeError(
                f"Decompression appeared to succeed but .sqlite file not found: {sqlite_file}"
            )

        # Get size for logging
        size_mb = sqlite_file.stat().st_size / (1024 * 1024)
        logger.info(f"  ✓ Decompressed {db_name} ({size_mb:.1f} MB)")

        return db_dir

    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Decompression of {db_name} timed out after 300 seconds"
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to decompress {db_name}: {e}"
        )
