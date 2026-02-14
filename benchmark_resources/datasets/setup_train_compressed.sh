#!/bin/bash
set -e  # Exit on error

# Space-optimized train dataset setup for 26 GB environments
# This script uses selective extraction to avoid exceeding space limits

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$SCRIPT_DIR/train"
TARGET_DIR="$TRAIN_DIR/train/train_databases"

# Databases to compress (5 largest)
LARGE_DBS=(
    "bike_share_1"
    "donor"
    "codebase_comments"
    "movie_platform"
    "world_development_indicators"
)

echo "=========================================="
echo "Train Dataset Setup - Space Optimized"
echo "=========================================="
echo ""
echo "This script will:"
echo "  1. Download and extract train.zip (peak: 17.4 GB)"
echo "  2. Selectively extract and compress 5 large databases (peak: 15 GB)"
echo "  3. Extract remaining 64 small databases (peak: 22.3 GB)"
echo "  4. Final storage: 13.6 GB base, 19.9 GB peak during operation"
echo ""
echo "Peak space required during setup: 22.3 GB"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Check available space
AVAILABLE_GB=$(df -BG "$SCRIPT_DIR" | awk 'NR==2 {print $4}' | sed 's/G//')
echo ""
echo "Available space: ${AVAILABLE_GB} GB"
if [ "$AVAILABLE_GB" -lt 23 ]; then
    echo "ERROR: Insufficient space. Need at least 23 GB available."
    exit 1
fi

cd "$SCRIPT_DIR"

# Phase 1: Download and extract train.zip
echo ""
echo "=========================================="
echo "Phase 1: Download and extract train.zip"
echo "=========================================="
echo ""

if [ ! -f "train.zip" ]; then
    echo "Downloading train.zip (8.7 GB)..."
    wget https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip

    echo "Verifying MD5 checksum..."
    ACTUAL_MD5=$(md5 -q train.zip 2>/dev/null || md5sum train.zip | awk '{print $1}')
    EXPECTED_MD5="abf4e64a4d8fc246bb284c45e756abd3"
    if [ "$ACTUAL_MD5" != "$EXPECTED_MD5" ]; then
        echo "ERROR: MD5 mismatch. Expected $EXPECTED_MD5, got $ACTUAL_MD5"
        exit 1
    fi
    echo "✓ Checksum verified"
else
    echo "train.zip already exists, skipping download"
fi

if [ ! -f "$TRAIN_DIR/train_databases.zip" ]; then
    echo "Extracting train.zip..."
    unzip -q train.zip
    echo "✓ Extracted"

    echo "Deleting train.zip to free space..."
    rm train.zip
    echo "✓ Deleted (saved 8.7 GB)"
else
    echo "train/ already extracted, skipping"
    if [ -f "train.zip" ]; then
        echo "Deleting train.zip to free space..."
        rm train.zip
        echo "✓ Deleted (saved 8.7 GB)"
    fi
fi

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Phase 2: Selectively extract and compress large databases
echo ""
echo "=========================================="
echo "Phase 2: Extract and compress large DBs"
echo "=========================================="
echo ""

cd "$TRAIN_DIR"

for db in "${LARGE_DBS[@]}"; do
    if [ -f "train_databases/${db}.tar.gz" ]; then
        echo "✓ $db already compressed, skipping"
        continue
    fi

    if [ -d "train_databases/$db" ]; then
        echo "✓ $db already extracted, compressing..."
    else
        echo "Extracting $db from train_databases.zip..."
        unzip -q train_databases.zip "train_databases/$db/*" -d .
        echo "✓ Extracted $db"
    fi

    echo "Compressing $db..."
    cd train_databases
    tar -czf "${db}.tar.gz" "$db"
    echo "✓ Compressed to ${db}.tar.gz"

    echo "Deleting uncompressed $db..."
    rm -rf "$db"
    echo "✓ Deleted uncompressed version"
    cd ..

    echo ""
done

# Phase 3: Extract remaining small databases
echo ""
echo "=========================================="
echo "Phase 3: Extract small databases"
echo "=========================================="
echo ""

echo "Extracting all remaining databases from train_databases.zip..."
echo "(This will extract 64 small databases, ~6.66 GB total)"
echo ""

# Create list of databases to exclude (already compressed)
EXCLUDE_ARGS=""
for db in "${LARGE_DBS[@]}"; do
    EXCLUDE_ARGS="$EXCLUDE_ARGS -x train_databases/$db/*"
done

# Extract everything except the large databases
unzip -q train_databases.zip -d . $EXCLUDE_ARGS || true

echo "✓ Small databases extracted"
echo ""

# Phase 4: Cleanup
echo ""
echo "=========================================="
echo "Phase 4: Cleanup"
echo "=========================================="
echo ""

if [ -f "train_databases.zip" ]; then
    echo "Deleting train_databases.zip..."
    rm train_databases.zip
    echo "✓ Deleted (saved 8.7 GB)"
fi

# Verify setup
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""

TOTAL_SIZE=$(du -sh train_databases | cut -f1)
echo "Total database storage: $TOTAL_SIZE"
echo ""

echo "Compressed databases (5):"
for db in "${LARGE_DBS[@]}"; do
    if [ -f "train_databases/${db}.tar.gz" ]; then
        SIZE=$(du -sh "train_databases/${db}.tar.gz" | cut -f1)
        echo "  ✓ ${db}.tar.gz ($SIZE)"
    else
        echo "  ✗ ${db}.tar.gz MISSING"
    fi
done

echo ""
echo "Uncompressed databases: 64 (always ready)"
echo ""
echo "Storage profile:"
echo "  • Base storage: ~13.6 GB"
echo "  • Peak storage during operation: ~19.9 GB"
echo ""
echo "To decompress a database when needed:"
echo "  cd $TARGET_DIR"
echo "  tar -xzf database_name.tar.gz"
echo ""
