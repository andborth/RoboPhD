#!/bin/bash
# Helper script to re-compress a database after use

DB_NAME=$1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DB_DIR="$SCRIPT_DIR/train/train/train_databases"

if [ -z "$DB_NAME" ]; then
    echo "Usage: $0 <database_name>"
    echo ""
    echo "Example: $0 bike_share_1"
    echo ""
    echo "This will compress the database and delete the uncompressed version"
    echo "to reclaim disk space."
    exit 1
fi

if [ ! -d "$DB_DIR" ]; then
    echo "Error: Database directory not found: $DB_DIR"
    exit 1
fi

cd "$DB_DIR"

# Check if uncompressed version exists
if [ ! -d "$DB_NAME" ]; then
    echo "Database $DB_NAME is not currently decompressed (nothing to do)"
    exit 0
fi

# Check if compressed archive already exists
if [ -f "${DB_NAME}.tar.gz" ]; then
    echo "Warning: ${DB_NAME}.tar.gz already exists"
    read -p "Overwrite? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
    rm "${DB_NAME}.tar.gz"
fi

# Compress
echo "Compressing $DB_NAME..."
UNCOMPRESSED_SIZE=$(du -sh "$DB_NAME" | cut -f1)
tar -czf "${DB_NAME}.tar.gz" "$DB_NAME"

if [ -f "${DB_NAME}.tar.gz" ]; then
    COMPRESSED_SIZE=$(du -sh "${DB_NAME}.tar.gz" | cut -f1)
    echo "✓ Successfully compressed $DB_NAME ($UNCOMPRESSED_SIZE → $COMPRESSED_SIZE)"

    echo "Deleting uncompressed version..."
    rm -rf "$DB_NAME"
    echo "✓ Deleted uncompressed version"
    echo "✓ Space reclaimed: $UNCOMPRESSED_SIZE"
else
    echo "Error: Compression failed"
    exit 1
fi
