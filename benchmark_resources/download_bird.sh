#!/bin/bash
# BIRD Benchmark Dataset Download Script
# Downloads and extracts the BIRD Text-to-SQL benchmark dataset

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Data root: defaults to the repo's benchmark_resources/datasets, but can be
# pointed at an external drive via BIRD_DATA_DIR (the dataset is ~130GB and may
# not fit on the internal disk). examples/text2sql reads the same variable.
DATASETS_DIR="${BIRD_DATA_DIR:-$SCRIPT_DIR/datasets}"

echo "==================================="
echo "BIRD Benchmark Dataset Downloader"
echo "==================================="
echo ""
echo "This script will download the BIRD benchmark dataset."
echo "Total size: ~50GB (compressed), ~80GB (extracted)"
echo ""
echo "Dataset source: https://bird-bench.github.io/"
echo "Data directory: $DATASETS_DIR"
echo ""

# Create datasets directory
mkdir -p "$DATASETS_DIR"
cd "$DATASETS_DIR"

# unzip is needed both to extract and to integrity-check downloads below.
if ! command -v unzip &> /dev/null; then
    echo "Error: 'unzip' is required but not found. Install it and re-run." >&2
    exit 1
fi

# Pick a downloader. aria2c is strongly preferred: the BIRD host (Aliyun OSS,
# Beijing) throttles PER CONNECTION — a single stream gets ~17 KB/s, so the
# ~8 GB train.zip would take *weeks*. aria2c's 16 parallel connections bypass
# that (~6 MB/s, ~25 min). curl/wget are single-stream fallbacks: they work
# (resume + retry), but are painfully slow on this host.
if command -v aria2c &> /dev/null; then _DL_TOOL=aria2c
elif command -v curl &> /dev/null; then _DL_TOOL=curl
elif command -v wget &> /dev/null; then _DL_TOOL=wget
else
    echo "Error: need a downloader — install aria2 (recommended), curl, or wget." >&2
    exit 1
fi
if [ "$_DL_TOOL" = aria2c ]; then
    echo "Downloader: aria2c (16 parallel connections)"
else
    echo "Downloader: $_DL_TOOL — WARNING: single-stream. This host throttles"
    echo "  per-connection (~17 KB/s), so the train set can take many hours/days."
    echo "  Install aria2 for ~100x faster downloads (e.g. 'brew install aria2')."
fi
echo ""

# Fetch $1 -> ./$2 (cwd is the data dir). Each tool resumes a partial and
# auto-retries on the connection resets this host does mid-transfer. The exit
# code is intentionally ignored (|| true): completeness is decided solely by
# `unzip -t` in download_and_verify_zip below — and a complete file makes
# `curl -C -` exit non-zero (416 range), which would otherwise trip `set -e`.
_fetch() {
    local url="$1" output="$2"
    case "$_DL_TOOL" in
        aria2c)
            aria2c -x16 -s16 -k1M -c --max-tries=20 --retry-wait=5 \
                --file-allocation=none --console-log-level=warn --summary-interval=30 \
                -o "$output" "$url" || true ;;
        curl)
            curl -L -C - --retry 10 --retry-delay 5 --retry-all-errors \
                --progress-bar -o "$output" "$url" || true ;;
        wget)
            wget -c --tries=10 --retry-connrefused --waitretry=5 \
                --show-progress -O "$output" "$url" || true ;;
    esac
}

# Download $url -> $output and verify it's a complete, valid zip. Layered:
#   - _fetch (above)     : resume + retry; aria2c parallel, or curl/wget single-stream
#   - outer attempt loop : re-invoke if the tool exits early
#   - progress guard     : if a partial stops GROWING across attempts (a host
#                          mishandling range requests, or a wedged aria2c
#                          segment), discard it — file AND any .aria2 control
#                          file — and download fresh instead of resuming
#                          corruption forever. Measured by on-disk blocks (du),
#                          which grow for both a sequential file and aria2c's
#                          sparse segmented writes.
#   - unzip -t gate      : only trust a file whose archive verifies, so a
#                          truncated download can never be unzipped.
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
        # Progress guard (see header). Discard a partial that isn't growing
        # across two attempts and download fresh. Uses on-disk blocks (du) so it
        # works for aria2c's sparse segmented file too, and clears the .aria2
        # control file so the clean restart doesn't resume stale segment state.
        if [ -f "$output" ]; then
            cur_size=$(du -k "$output" 2>/dev/null | cut -f1); cur_size=${cur_size:-0}
            if [ "$cur_size" -le "$prev_size" ]; then stalls=$((stalls + 1)); else stalls=0; fi
            prev_size="$cur_size"
            if [ "$stalls" -ge 2 ]; then
                echo "Resume stalled at ${cur_size} KB; discarding $output and restarting clean."
                rm -f "$output" "$output.aria2"
                prev_size=-1
                stalls=0
            fi
        fi
        echo "Downloading $output (attempt $i/$attempts)..."
        _fetch "$url" "$output"
    done
    if [ -f "$output" ] && unzip -tq "$output" >/dev/null 2>&1; then
        echo "$output verified (complete)."
        return 0
    fi
    echo "ERROR: could not obtain a complete $output after $attempts attempts." >&2
    echo "       The partial file is kept — re-run to resume. If it keeps failing," >&2
    echo "       delete $output (and $output.aria2 if present) to force a clean download." >&2
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
echo "Step 3: train-filtered subset (canonical, from HuggingFace)"
echo "-----------------------------------------------------------"

# The curated 6,601-question filtered subset (vs 9,428 in full train) is NOT in
# the BIRD train.zip. It's published by the official birdsql org on HuggingFace
# (datasets/birdsql/bird23-train-filtered) as JSONL — verified identical to the
# subset the text2sql evaluator's default --dataset expects. Pull it from there
# (fast US CDN, no throttle) and convert JSONL -> the JSON array json.load wants.
# Same db_root as full train (Step 2). Non-fatal: a failure here doesn't sink
# the train/dev sets already downloaded above.
TRAIN_FILTERED="train-filtered/train_filtered.json"
HF_TRAIN_FILTERED_URL="https://huggingface.co/datasets/birdsql/bird23-train-filtered/resolve/main/data/train-00000-of-00001.jsonl"
if [ ! -f "$TRAIN_FILTERED" ]; then
    mkdir -p train-filtered
    _tf_tmp="train-filtered/.train_filtered.jsonl"
    echo "Fetching from $HF_TRAIN_FILTERED_URL"
    if curl -fsSL --retry 5 -o "$_tf_tmp" "$HF_TRAIN_FILTERED_URL" 2>/dev/null \
       || wget -q -O "$_tf_tmp" "$HF_TRAIN_FILTERED_URL"; then
        if command -v python3 &> /dev/null; then
            # Convert to a .partial and mv into place only on success, so a
            # crash mid-write (open(...,'w') truncates first) can't leave a
            # corrupt train_filtered.json that the [ -f ] check treats as done.
            if python3 -c "import json,sys; rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]; json.dump(rows, open(sys.argv[2],'w')); print(f'train-filtered ready: {len(rows)} questions')" "$_tf_tmp" "$TRAIN_FILTERED.partial"; then
                mv -f "$TRAIN_FILTERED.partial" "$TRAIN_FILTERED"
            else
                echo "  WARNING: JSONL -> JSON conversion failed; train-filtered unavailable."
                rm -f "$TRAIN_FILTERED.partial"
            fi
        else
            echo "  WARNING: need python3 to convert HF JSONL -> JSON array; train-filtered unavailable."
        fi
    else
        echo "  WARNING: could not fetch train-filtered from HuggingFace (network?)."
        echo "           The default --dataset train-filtered will be unavailable;"
        echo "           use --dataset train (full 9,428) or --dataset dev instead."
    fi
    rm -f "$_tf_tmp"
else
    echo "train-filtered already exists, skipping."
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
