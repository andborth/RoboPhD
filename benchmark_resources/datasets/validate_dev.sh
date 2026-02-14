#!/bin/bash
# Validate dev dataset

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Expected MD5 checksum
DEV_JSON_MD5="af311ef1348945573b1e49e41309edb1"

echo "=========================================="
echo "Dev Dataset Validation"
echo "=========================================="
echo ""

# Function to compute MD5 (works on both macOS and Linux)
compute_md5() {
    if command -v md5 &> /dev/null; then
        md5 -q "$1" 2>/dev/null
    elif command -v md5sum &> /dev/null; then
        md5sum "$1" 2>/dev/null | awk '{print $1}'
    else
        echo "ERROR: Neither md5 nor md5sum command found"
        exit 1
    fi
}

# Validate dev.json
echo "Checking dev.json..."
DEV_JSON_PATH="dev/dev_20240627/dev.json"

if [ ! -f "$DEV_JSON_PATH" ]; then
    echo "  ✗ NOT FOUND: $DEV_JSON_PATH"
    echo ""
    echo "To download and extract dev dataset:"
    echo "  cd benchmark_resources/datasets/dev/"
    echo "  wget https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
    echo "  unzip dev.zip"
    echo "  cd dev_20240627 && unzip dev_databases.zip"
    exit 1
fi

ACTUAL_MD5=$(compute_md5 "$DEV_JSON_PATH")

if [ "$ACTUAL_MD5" = "$DEV_JSON_MD5" ]; then
    echo "  ✓ VALID: $DEV_JSON_PATH"
    echo "    MD5: $ACTUAL_MD5"

    # Check if databases are extracted
    echo ""
    echo "Checking dev databases..."
    DEV_DB_DIR="dev/dev_20240627/dev_databases"

    if [ ! -d "$DEV_DB_DIR" ]; then
        echo "  ⚠ WARNING: dev_databases/ directory not found"
        echo "    Extract databases: cd dev/dev_20240627 && unzip dev_databases.zip"
    else
        DB_COUNT=$(find "$DEV_DB_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
        if [ "$DB_COUNT" -eq 11 ]; then
            echo "  ✓ All 11 databases found in $DEV_DB_DIR"
        else
            echo "  ⚠ WARNING: Found $DB_COUNT databases (expected 11)"
        fi
    fi

    echo ""
    echo "=========================================="
    echo "✓ Dev dataset validated successfully"
    echo "=========================================="
    exit 0
else
    echo "  ✗ INVALID: $DEV_JSON_PATH"
    echo "    Expected MD5: $DEV_JSON_MD5"
    echo "    Actual MD5:   $ACTUAL_MD5"
    echo ""
    echo "The dev.json file may be corrupted. Try re-downloading:"
    echo "  cd benchmark_resources/datasets/dev/"
    echo "  rm dev.zip"
    echo "  wget https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
    echo "  unzip -o dev.zip"
    exit 1
fi
