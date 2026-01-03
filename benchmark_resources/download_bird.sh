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
echo "Step 1: Downloading Dev Set (~2GB)"
echo "-----------------------------------"

# Dev set - from BIRD benchmark
if [ ! -d "dev" ]; then
    if [ ! -f "dev.zip" ]; then
        echo "Please download the dev set manually from:"
        echo "  https://bird-bench.github.io/"
        echo ""
        echo "Then place dev.zip in: $DATASETS_DIR"
        echo "And run this script again."
        echo ""
        echo "Direct link (may require authentication):"
        echo "  https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
    fi

    if [ -f "dev.zip" ]; then
        echo "Extracting dev.zip..."
        unzip -q dev.zip
        echo "Dev set extracted successfully."
    fi
else
    echo "Dev set already exists, skipping."
fi

echo ""
echo "Step 2: Downloading Train Set (~40GB)"
echo "--------------------------------------"

if [ ! -d "train" ]; then
    if [ ! -f "train.zip" ]; then
        echo "Please download the train set manually from:"
        echo "  https://bird-bench.github.io/"
        echo ""
        echo "Then place train.zip in: $DATASETS_DIR"
        echo "And run this script again."
        echo ""
        echo "Direct link (may require authentication):"
        echo "  https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip"
    fi

    if [ -f "train.zip" ]; then
        echo "Extracting train.zip..."
        unzip -q train.zip
        echo "Train set extracted successfully."
    fi
else
    echo "Train set already exists, skipping."
fi

echo ""
echo "Step 3: Creating train-filtered subset"
echo "---------------------------------------"

# The train-filtered dataset is a curated subset with better quality
# It should be generated from the train set
if [ -d "train" ] && [ ! -f "train/train_filtered.json" ]; then
    echo "Note: train_filtered.json needs to be created from train.json"
    echo "This is a curated subset with 6,601 questions (vs 9,428 in full train)."
    echo "See documentation for the filtering criteria."
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
