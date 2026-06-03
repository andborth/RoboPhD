#!/bin/bash
# BIRD Benchmark Dataset Download Script
# Downloads and extracts the BIRD Text-to-SQL benchmark dataset

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS_DIR="$SCRIPT_DIR/datasets"

echo "==================================="
echo "BIRD Benchmark Dataset Downloader"
echo "==================================="
echo ""
echo "This script will download the BIRD benchmark dataset."
echo "Total size: ~50GB (compressed), ~80GB (extracted)"
echo ""
echo "Dataset source: https://bird-bench.github.io/"
echo ""

# Create datasets directory
mkdir -p "$DATASETS_DIR"
cd "$DATASETS_DIR"

# unzip is needed both to extract and to integrity-check downloads below.
if ! command -v unzip &> /dev/null; then
    echo "Error: 'unzip' is required but not found. Install it and re-run." >&2
    exit 1
fi

# Download $url -> $output, resuming partials and verifying the result is a
# complete, valid zip. The BIRD host (Aliyun OSS, Beijing) resets connections
# mid-transfer, which is fatal for a plain `curl -o` on the 40GB train set.
# Defenses, in layers:
#   - curl -C - / wget -c        : resume a partial file instead of restarting
#   - --retry-all-errors / --tries: auto-retry within one invocation, incl. on
#                                    connection resets (plain --retry skips those)
#   - outer attempt loop          : re-invoke if the tool still exits early
#   - progress guard              : if the file stops GROWING across attempts,
#                                    the partial is unrecoverable (e.g. a host
#                                    mishandling range requests) — discard it and
#                                    download fresh instead of resuming corruption
#                                    forever. Keyed on progress, not attempt
#                                    count, so a legitimately slow multi-reset
#                                    download (which grows each attempt) is never
#                                    thrown away.
#   - unzip -t gate               : only trust a file whose archive verifies,
#                                    so a truncated download can never be unzipped
# The download tool's exit code is intentionally ignored (|| true): a complete
# file makes `curl -C -` exit non-zero (416 range), and `set -e` would abort.
# Completeness is decided solely by `unzip -t`.
download_and_verify_zip() {
    local url="$1"
    local output="$2"
    local attempts="${3:-8}"
    local i prev_size=-1 cur_size stalls=0
    for ((i=1; i<=attempts; i++)); do
        if [ -f "$output" ] && unzip -tq "$output" >/dev/null 2>&1; then
            echo "$output verified (complete)."
            return 0
        fi
        # Progress guard: discard a partial that isn't growing (two stalled
        # attempts) and download fresh, rather than resuming a stuck file.
        if [ -f "$output" ]; then
            cur_size=$(wc -c < "$output" 2>/dev/null || echo 0)
            if [ "$cur_size" -le "$prev_size" ]; then stalls=$((stalls + 1)); else stalls=0; fi
            prev_size="$cur_size"
            if [ "$stalls" -ge 2 ]; then
                echo "Resume stalled at ${cur_size} bytes; discarding $output and restarting clean."
                rm -f "$output"
                prev_size=-1
                stalls=0
            fi
        fi
        echo "Downloading $output (attempt $i/$attempts)..."
        if command -v curl &> /dev/null; then
            curl -L -C - --retry 10 --retry-delay 5 --retry-all-errors \
                --progress-bar -o "$output" "$url" || true
        elif command -v wget &> /dev/null; then
            wget -c --tries=10 --retry-connrefused --waitretry=5 \
                --show-progress -O "$output" "$url" || true
        else
            echo "Error: curl or wget required" >&2
            exit 1
        fi
    done
    if [ -f "$output" ] && unzip -tq "$output" >/dev/null 2>&1; then
        echo "$output verified (complete)."
        return 0
    fi
    echo "ERROR: could not obtain a complete $output after $attempts attempts." >&2
    echo "       The partial file is kept — re-run to resume. If it keeps failing" >&2
    echo "       at the same size, delete $output to force a clean download." >&2
    return 1
}

echo ""
echo "Step 1: Downloading Dev Set (~330MB)"
echo "-----------------------------------"

# Dev set - from BIRD benchmark
if [ ! -d "dev/dev_20240627/dev_databases" ]; then
    download_and_verify_zip "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip" "dev.zip"
    echo "Extracting dev.zip..."
    unzip -q dev.zip
    # Create expected directory structure
    mkdir -p dev
    mv dev_20240627 dev/
    # Extract nested databases zip
    cd dev/dev_20240627
    unzip -q dev_databases.zip
    rm -rf __MACOSX 2>/dev/null || true
    cd "$DATASETS_DIR"
    echo "Dev set extracted successfully."
else
    echo "Dev set already exists, skipping."
fi

echo ""
echo "Step 2: Downloading Train Set (~40GB)"
echo "--------------------------------------"

if [ ! -d "train/train/train_databases" ]; then
    download_and_verify_zip "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip" "train.zip"

    echo "Extracting train.zip..."
    unzip -q train.zip
    # train.zip extracts to train/ folder, but we need train/train/
    # Move contents to create nested structure
    if [ -d "train" ] && [ ! -d "train/train" ]; then
        mkdir -p train_temp
        mv train/* train_temp/
        mkdir -p train/train
        mv train_temp/* train/train/
        rmdir train_temp
    fi
    rm -rf __MACOSX 2>/dev/null || true

    # Extract train_databases.zip inside train/train/
    if [ -f "train/train/train_databases.zip" ]; then
        echo "Extracting train_databases.zip (~9GB)..."
        cd train/train
        unzip -q train_databases.zip
        rm -rf __MACOSX 2>/dev/null || true
        cd "$DATASETS_DIR"
    fi
    echo "Train set extracted successfully."
else
    echo "Train set already exists, skipping."
fi

echo ""
echo "Step 3: Creating train-filtered subset"
echo "---------------------------------------"

# The train-filtered dataset is a curated subset with better quality
if [ -d "train/train" ] && [ ! -d "train-filtered" ]; then
    echo "Creating train-filtered directory..."
    mkdir -p train-filtered
    # train_filtered.json should be generated or copied if available
    if [ -f "train/train/train_filtered.json" ]; then
        cp train/train/train_filtered.json train-filtered/
        echo "train-filtered dataset ready."
    else
        echo "Note: train_filtered.json needs to be created from train.json"
        echo "This is a curated subset with 6,601 questions (vs 9,428 in full train)."
        echo "See documentation for the filtering criteria."
    fi
fi

echo ""
echo "==================================="
echo "Download Complete"
echo "==================================="
echo ""
echo "Directory structure:"
ls -la "$DATASETS_DIR"
echo ""
echo "Next steps:"
echo "1. Pre-compute ground truth: python RoboPhD/tools/precompute_ground_truth.py"
echo "2. Run a test: python RoboPhD/researcher.py --num-iterations 1 --config '{\"examples_per_iteration\": 1}'"
echo ""
