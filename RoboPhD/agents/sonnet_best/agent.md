---
name: comprehensive-unified-cross-pollinator
description: Cross-pollinated tool combining best patterns from iter6, iter7, iter9, iter10, iter14, iter15 with enhanced evidence handling
execution_mode: tool_only
tool_command: python tools/comprehensive_unified_analyzer.py
tool_output_file: tool_output/schema_analysis.txt
---

# Comprehensive Unified Cross-Pollinator (Tool-Only)

This agent uses deterministic tool-only execution combining successful patterns from multiple top-performing agents:

- **iter6** (ELO 1531): Column ownership, cross-table hints, output limiting
- **iter7** (ELO 1514): City enumeration, spelling variation detection
- **iter9** (ELO 1472): Entity tables, currency detection, authoritative sources
- **iter10** (ELO 1523): Unified analysis with enhanced patterns
- **iter14** (ELO 1550): Semantic column classification, pre-aggregated detection
- **iter15** (ELO 1503): Date format detection, strftime compatibility, JOIN patterns

## Key Innovations in iter16

Based on error analysis from iteration 015:

1. **Evidence-First Emphasis**: Output prominently reminds to follow evidence literally
2. **Boolean/Flag Column Detection**: Identifies columns like `user_subscriber` that represent boolean states
3. **Enhanced Timestamp Analysis**: Helps decide between SUBSTR and datetime() for year calculations
4. **Column Selection Warnings**: Emphasizes returning ONLY requested columns

## Process

1. **Run Cross-Pollinated Analysis Tool**
   - Execute: `python tools/comprehensive_unified_analyzer.py`
   - Tool performs comprehensive database analysis combining all techniques

2. **Read and Output Results**
   - Read the generated analysis from `tool_output/schema_analysis.txt`
   - Write the complete output to `./output/agent_output.txt`

## Error Recovery

If the tool fails:

1. Check database.sqlite exists in current directory
2. Verify Python environment has sqlite3 library
3. Examine any error messages in tool_output/
4. Check for database access permissions
5. Fall back to manual analysis if needed:
   - List all tables with `SELECT name FROM sqlite_master WHERE type='table'`
   - Get schema for each table with `PRAGMA table_info(table_name)`
   - Sample values from key columns
   - Identify foreign key relationships

## Output Structure

The tool generates analysis in three sections:

1. **CRITICAL GUIDANCE**: Evidence-following reminders
2. **SECTION 1: DATABASE SCHEMA**:
   - Boolean/flag column warnings
   - Date/timestamp analysis with year extraction hints
   - Authoritative source tables
   - Duplicate counting risks
   - Pre-aggregated column warnings
   - Similar table warnings
   - Raw DDL with annotations

3. **SECTION 2: COLUMN DISAMBIGUATION**:
   - Columns in multiple tables with value comparisons
   - Values-differ warnings

4. **SECTION 3: QUERY SUPPORT**:
   - Value variation warnings
   - Cross-table lookup hints
   - Column samples with markers
   - Column ownership map
   - Evidence reconciliation hints
