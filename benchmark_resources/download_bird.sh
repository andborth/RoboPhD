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

# Function to download with progress
download_file() {
    local url="$1"
    local output="$2"
    echo "Downloading: $output"
    if command -v curl &> /dev/null; then
        curl -L -o "$output" "$url" --progress-bar
    elif command -v wget &> /dev/null; then
        wget -O "$output" "$url" --show-progress
    else
        echo "Error: curl or wget required"
        exit 1
    fi
}

echo ""
echo "Step 1: Downloading Dev Set (~330MB)"
echo "-----------------------------------"

# Dev set - from BIRD benchmark
if [ ! -d "dev/dev_20240627/dev_databases" ]; then
    if [ ! -f "dev.zip" ]; then
        download_file "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip" "dev.zip"
    fi

    if [ -f "dev.zip" ]; then
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
    fi
else
    echo "Dev set already exists, skipping."
fi

echo ""
echo "Step 2: Downloading Train Set (~40GB)"
echo "--------------------------------------"

if [ ! -d "train/train/train_databases" ]; then
    if [ ! -f "train.zip" ]; then
        download_file "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip" "train.zip"
    fi

    if [ -f "train.zip" ]; then
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
    fi
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
echo "2. Run a test: python RoboPhD/researcher.py --num-iterations 1 --config '{\"databases_per_iteration\": 1}'"
echo ""
