#!/bin/bash
# Validate train and train-filtered datasets

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Expected MD5 checksums
TRAIN_JSON_MD5="37131456941dabe670c3f1c2fd07f008"
TRAIN_FILTERED_JSON_MD5="96afa2c5be5f5154ce7299365a59f557"

echo "=========================================="
echo "Train Dataset Validation"
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

# Validate train.json
echo "Checking train.json..."
TRAIN_JSON_PATH="train/train/train.json"

if [ ! -f "$TRAIN_JSON_PATH" ]; then
    echo "  ✗ NOT FOUND: $TRAIN_JSON_PATH"
    echo "    Run setup_train_compressed.sh to download and extract"
    TRAIN_VALID=false
else
    ACTUAL_MD5=$(compute_md5 "$TRAIN_JSON_PATH")
    if [ "$ACTUAL_MD5" = "$TRAIN_JSON_MD5" ]; then
        echo "  ✓ VALID: $TRAIN_JSON_PATH"
        echo "    MD5: $ACTUAL_MD5"
        TRAIN_VALID=true
    else
        echo "  ✗ INVALID: $TRAIN_JSON_PATH"
        echo "    Expected MD5: $TRAIN_JSON_MD5"
        echo "    Actual MD5:   $ACTUAL_MD5"
        TRAIN_VALID=false
    fi
fi

echo ""

# Validate train_filtered.json
echo "Checking train_filtered.json..."
TRAIN_FILTERED_PATH="train-filtered/train_filtered.json"

if [ ! -f "$TRAIN_FILTERED_PATH" ]; then
    echo "  ✗ NOT FOUND: $TRAIN_FILTERED_PATH"
    echo "    This file should be in git. Try: git checkout $TRAIN_FILTERED_PATH"
    FILTERED_VALID=false
else
    ACTUAL_MD5=$(compute_md5 "$TRAIN_FILTERED_PATH")
    if [ "$ACTUAL_MD5" = "$TRAIN_FILTERED_JSON_MD5" ]; then
        echo "  ✓ VALID: $TRAIN_FILTERED_PATH"
        echo "    MD5: $ACTUAL_MD5"
        FILTERED_VALID=true
    else
        echo "  ✗ INVALID: $TRAIN_FILTERED_PATH"
        echo "    Expected MD5: $TRAIN_FILTERED_JSON_MD5"
        echo "    Actual MD5:   $ACTUAL_MD5"
        FILTERED_VALID=false
    fi
fi

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="

if [ "$TRAIN_VALID" = true ] && [ "$FILTERED_VALID" = true ]; then
    echo "✓ All train datasets validated successfully"
    exit 0
elif [ "$TRAIN_VALID" = false ] && [ "$FILTERED_VALID" = false ]; then
    echo "✗ Both datasets failed validation"
    exit 1
else
    if [ "$TRAIN_VALID" = false ]; then
        echo "✗ train.json validation failed"
    fi
    if [ "$FILTERED_VALID" = false ]; then
        echo "✗ train_filtered.json validation failed"
    fi
    exit 1
fi
