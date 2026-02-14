#!/bin/bash
# Helper script to decompress a database on-demand

DB_NAME=$1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DB_DIR="$SCRIPT_DIR/train/train/train_databases"

if [ -z "$DB_NAME" ]; then
    echo "Usage: $0 <database_name>"
    echo ""
    echo "Example: $0 bike_share_1"
    echo ""
    echo "Compressed databases:"
    echo "  - bike_share_1"
    echo "  - donor"
    echo "  - codebase_comments"
    echo "  - movie_platform"
    echo "  - world_development_indicators"
    exit 1
fi

if [ ! -d "$DB_DIR" ]; then
    echo "Error: Database directory not found: $DB_DIR"
    echo "Have you run setup_train_compressed.sh?"
    exit 1
fi

cd "$DB_DIR"

# Check if already decompressed
if [ -d "$DB_NAME" ]; then
    echo "✓ Database $DB_NAME is already decompressed"
    exit 0
fi

# Check if compressed archive exists
if [ ! -f "${DB_NAME}.tar.gz" ]; then
    echo "Error: ${DB_NAME}.tar.gz not found"
    echo ""
    echo "Available compressed databases:"
    ls -1 *.tar.gz 2>/dev/null | sed 's/.tar.gz$//' | sed 's/^/  - /'
    exit 1
fi

# Decompress
echo "Decompressing $DB_NAME..."
tar -xzf "${DB_NAME}.tar.gz"

if [ -d "$DB_NAME" ]; then
    SIZE=$(du -sh "$DB_NAME" | cut -f1)
    echo "✓ Successfully decompressed $DB_NAME ($SIZE)"
else
    echo "Error: Decompression failed"
    exit 1
fi
