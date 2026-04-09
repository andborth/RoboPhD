#!/usr/bin/env python3
"""
Baseline schema extractor - extracts raw DDL using sqlite3.
"""

import sqlite3
import sys


def extract_schema(db_path: str, output_file: str):
    """Extract complete database schema as DDL."""

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get all schema DDL statements
        cursor.execute("""
            SELECT sql || ';'
            FROM sqlite_master
            WHERE sql IS NOT NULL
            ORDER BY tbl_name, type DESC, name
        """)

        schema_statements = cursor.fetchall()

        # Format output
        output_lines = []
        for (sql,) in schema_statements:
            output_lines.append(sql)

        schema_output = '\n'.join(output_lines)

        # Write to output file
        with open(output_file, 'w') as f:
            f.write(schema_output)

        # --- Demonstration: print() output is captured as tool_stdout in diagnostics ---
        print(f"Schema extraction complete - {len(schema_statements)} DDL statements, {len(schema_output)} chars")
        conn.close()
        return 0

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    exit(extract_schema("database.sqlite", "analysis.txt"))
