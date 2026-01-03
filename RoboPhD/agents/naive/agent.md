---
name: naive
description: Baseline agent that outputs raw DDL schema using tool-only execution
execution_mode: tool_only
tool_command: python tools/extract_schema.py
tool_output_file: tool_output/schema.txt
---

# Naive DDL Extractor (Tool-Only)

This agent uses deterministic tool-only execution to extract raw database schema.

## Process

1. **Run Schema Extraction Tool**
   ```bash
   python tools/extract_schema.py
   ```

2. **Read and Output Results**
   - Read the generated schema from `tool_output/schema.txt`
   - Write the complete output to `./output/agent_output.txt`

## Error Recovery

If the tool fails:

1. Check `database.sqlite` exists
2. Verify Python environment has sqlite3 library
3. Examine any error messages in `tool_output/`
4. Attempt to run the tool manually to see errors
5. Fall back to manual schema extraction using SQLite CLI:
   ```bash
   sqlite3 database.sqlite ".schema" > output/agent_output.txt
   ```
