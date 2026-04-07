#!/usr/bin/env bash
# Download Can't Be Late trace data from UCB-ADRS repository.
#
# Usage:
#   bash scripts/download_cant_be_late_traces.sh
#
# Downloads and extracts the trace directory to:
#   RoboPhD/data/cant_be_late/traces/
#
# After extraction, the directory contains:
#   traces/ddl=search+task=48+overhead=0.02/real/<env>/traces/random_start/*.json
#   traces/ddl=search+task=48+overhead=0.20/real/<env>/traces/random_start/*.json
#   traces/ddl=search+task=48+overhead=0.40/real/<env>/traces/random_start/*.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET_DIR="$REPO_ROOT/RoboPhD/data/cant_be_late/traces"

ARCHIVE_URL="https://github.com/UCB-ADRS/ADRS/raw/main/openevolve/examples/ADRS/cant-be-late/simulator/real_traces.tar.gz"

if [ -d "$TARGET_DIR/ddl=search+task=48+overhead=0.20" ]; then
    echo "Traces already exist at $TARGET_DIR"
    echo "To re-download, remove the directory first: rm -rf $TARGET_DIR"
    exit 0
fi

mkdir -p "$TARGET_DIR"

echo "Downloading Can't Be Late traces..."
echo "  URL: $ARCHIVE_URL"
echo "  Target: $TARGET_DIR"

TMPFILE=$(mktemp /tmp/cant_be_late_traces.XXXXXX.tar.gz)
trap 'rm -f "$TMPFILE"' EXIT

curl -L --fail --progress-bar -o "$TMPFILE" "$ARCHIVE_URL"

echo "Extracting traces..."
# Archive has real/ prefix — strip it so ddl=... dirs are directly under TARGET_DIR
tar xzf "$TMPFILE" -C "$TARGET_DIR" --strip-components=1

echo "Done! Traces extracted to: $TARGET_DIR"
echo ""
echo "Verify with:"
echo "  ls $TARGET_DIR/"
echo "  python -c \"from examples.cant_be_late.evaluator import load_dataset; d = load_dataset(); print({k: len(v) for k, v in d.items()})\""
