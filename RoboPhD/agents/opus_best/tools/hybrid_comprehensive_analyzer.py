#!/usr/bin/env python3
"""
Hybrid Comprehensive Analyzer - Cross-pollination of iter16, iter9, and iter3.

Key Innovation: Combines adaptive scaling (iter16), advanced features (iter9),
and precision techniques (iter3) with NEW string matching and NULL/NaN detection.

Cross-pollinated techniques:
- iter16: Database size classification, adaptive truncation, context budget tracking
- iter9: Semantic patterns, cross-table validation, statistical hints, 10-section output
- iter3: ALL column coverage, enum detection, cardinality analysis, value ranges

NEW Enhancements:
- String matching guidance (LIKE vs = detection based on enum values)
- NULL/NaN pattern detection (separate tracking of NULL, 'nan', empty strings)
- Evidence complexity warnings (common pitfalls in query guidance)
"""

import sqlite3
import sys
import re
from typing import Dict, List, Tuple, Any, Optional, Set


# ============================================================================
# DATABASE SIZE CLASSIFICATION (from iter16)
# ============================================================================

def classify_database_size(cursor, tables: List[str]) -> Dict[str, Any]:
    """
    Classify database size to determine appropriate analysis depth.

    Returns dict with size-specific configuration for adaptive analysis.
    """
    total_tables = len(tables)
    total_columns = 0

    for table in tables:
        cursor.execute(f'PRAGMA table_info("{table}")')
        total_columns += len(cursor.fetchall())

    if total_tables <= 15 and total_columns <= 150:
        size = "small"
        strategy = "Full comprehensive analysis (iter9 style)"
        sample_limit = 10
        enum_limit = None  # Show all
        skip_advanced = False
    elif total_tables <= 40 and total_columns <= 300:
        size = "medium"
        strategy = "Smart truncation with comprehensive features"
        sample_limit = 5
        enum_limit = 15
        skip_advanced = False
    elif total_columns <= 400:
        size = "large"
        strategy = "Aggressive truncation to prevent context overflow"
        sample_limit = 3
        enum_limit = 5
        skip_advanced = True  # Skip semantic patterns, validation, stats
    else:  # Ultra-large (works_cycles: 455 columns)
        size = "ultra_large"
        strategy = "ULTRA-COMPACT: Minimal detail to avoid overflow"
        sample_limit = 1
        enum_limit = 0  # Skip enum detection entirely
        skip_advanced = True

    return {
        'size': size,
        'table_count': total_tables,
        'column_count': total_columns,
        'strategy': strategy,
        'sample_limit': sample_limit,
        'enum_limit': enum_limit,
        'skip_advanced': skip_advanced
    }


# ============================================================================
# CORE SCHEMA FUNCTIONS (from iter3 - proven foundation)
# ============================================================================

def extract_schema(cursor) -> List[Tuple[str, str]]:
    """Extract complete database schema as DDL."""
    cursor.execute("""
        SELECT name, sql || ';'
        FROM sqlite_master
        WHERE sql IS NOT NULL
        ORDER BY tbl_name, type DESC, name
    """)
    return cursor.fetchall()


def get_tables(cursor) -> List[str]:
    """Get list of all tables in database."""
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
    """)
    return [row[0] for row in cursor.fetchall()]


def get_table_info(cursor, table: str) -> List[Dict[str, Any]]:
    """Get column information for a table."""
    cursor.execute(f'PRAGMA table_info("{table}")')
    columns = []
    for row in cursor.fetchall():
        columns.append({
            'name': row[1],
            'type': row[2],
            'notnull': row[3] == 1,
            'pk': row[5] > 0
        })
    return columns


def get_row_count(cursor, table: str) -> int:
    """Get row count for a table."""
    try:
        cursor.execute(f"SELECT COUNT(*) FROM `{table}`")
        return cursor.fetchone()[0]
    except:
        return 0


def analyze_foreign_keys(cursor, table: str) -> List[Dict[str, Any]]:
    """Analyze foreign key relationships for a table."""
    cursor.execute(f'PRAGMA foreign_key_list("{table}")')
    fks = []
    for row in cursor.fetchall():
        fks.append({
            'from_column': row[3],
            'to_table': row[2],
            'to_column': row[4]
        })
    return fks


# ============================================================================
# DATA SAMPLING FUNCTIONS (adaptive based on size)
# ============================================================================

def sample_column_values(cursor, table: str, column: str, limit: int) -> List[Any]:
    """Sample distinct values from a column with adaptive limit."""
    try:
        cursor.execute(f"""
            SELECT DISTINCT `{column}`
            FROM `{table}`
            WHERE `{column}` IS NOT NULL
            LIMIT {limit}
        """)
        return [row[0] for row in cursor.fetchall()]
    except:
        return []


def get_null_percentage(cursor, table: str, column: str) -> float:
    """Calculate percentage of NULL values in a column."""
    try:
        cursor.execute(f"SELECT COUNT(*) FROM `{table}` WHERE `{column}` IS NULL")
        null_count = cursor.fetchone()[0]
        cursor.execute(f"SELECT COUNT(*) FROM `{table}`")
        total_count = cursor.fetchone()[0]
        return (null_count / total_count * 100) if total_count > 0 else 0.0
    except:
        return 0.0


# ============================================================================
# FORMAT DETECTION (from iter3 + iter9)
# ============================================================================

def detect_value_format(values: List[Any]) -> Dict[str, Any]:
    """
    Detect format patterns in column values.
    Combined from iter3 and iter9 patterns.
    """
    if not values:
        return {'format': 'empty', 'pattern': None}

    sample = str(values[0]) if values else ""

    # Currency detection
    if re.match(r'^\$[\d,]+\.?\d*$', sample):
        return {
            'format': 'currency',
            'pattern': 'Dollar amount with $ and commas',
            'parsing': "CAST(REPLACE(REPLACE(column, ',', ''), '$', '') AS REAL)",
            'example': sample
        }

    # Percentage detection
    if re.match(r'^\d+\.?\d*%$', sample):
        return {
            'format': 'percentage',
            'pattern': 'Percentage with % symbol',
            'parsing': "CAST(REPLACE(column, '%', '') AS REAL)",
            'example': sample
        }

    # Time duration detection (HH:MM:SS or H:MM:SS)
    if re.match(r'^\d{1,2}:\d{2}:\d{2}$', sample):
        return {
            'format': 'time_duration',
            'pattern': 'Time duration as HH:MM:SS',
            'parsing': "Use direct comparison - SQLite sorts time strings correctly",
            'example': sample
        }

    # Date detection (YYYY-MM-DD)
    if re.match(r'^\d{4}-\d{2}-\d{2}', sample):
        return {
            'format': 'date',
            'pattern': 'ISO date format (YYYY-MM-DD)',
            'parsing': "Direct comparison works, or use DATE() function",
            'example': sample
        }

    # Code pattern detection (e.g., 'E0', 'D1', 'B1')
    if re.match(r'^[A-Z]\d+$', sample) and len(sample) <= 4:
        return {
            'format': 'code',
            'pattern': f'Code pattern (letter + digits): {sample}',
            'example': sample
        }

    # Numeric detection
    if isinstance(values[0], (int, float)):
        return {
            'format': 'numeric',
            'pattern': f'Numeric ({type(values[0]).__name__})',
            'example': sample
        }

    return {
        'format': 'text',
        'pattern': 'Text/string value',
        'example': sample
    }


# ============================================================================
# PRECISION ANALYSIS FUNCTIONS (from iter3)
# ============================================================================

def detect_enum_values(cursor, table: str, column: str, threshold: int = 20,
                      row_count: Optional[int] = None, max_rows: int = 1000000) -> Optional[List[Any]]:
    """
    Detect if column is an enum (has limited distinct values) and return all values.
    Returns None if column has more than threshold distinct values.
    (from iter3, with iter9's performance optimization)
    """
    try:
        # Skip very large tables to avoid performance issues
        if row_count is None:
            cursor.execute(f"SELECT COUNT(*) FROM `{table}`")
            row_count = cursor.fetchone()[0]

        if row_count > max_rows:
            return None

        # Count distinct values
        cursor.execute(f"SELECT COUNT(DISTINCT `{column}`) FROM `{table}` WHERE `{column}` IS NOT NULL")
        distinct_count = cursor.fetchone()[0]

        if distinct_count == 0 or distinct_count > threshold:
            return None

        # Get all distinct values
        cursor.execute(f"""
            SELECT DISTINCT `{column}`
            FROM `{table}`
            WHERE `{column}` IS NOT NULL
            ORDER BY `{column}`
        """)
        return [row[0] for row in cursor.fetchall()]
    except:
        return None


def get_value_range(cursor, table: str, column: str, col_type: str) -> Optional[Dict[str, Any]]:
    """
    Get MIN/MAX for numeric columns (from iter3).
    """
    # Only analyze numeric types
    if col_type.upper() not in ('INTEGER', 'REAL', 'NUMERIC', 'FLOAT', 'DOUBLE'):
        return None

    try:
        cursor.execute(f"SELECT MIN(`{column}`), MAX(`{column}`) FROM `{table}` WHERE `{column}` IS NOT NULL")
        result = cursor.fetchone()
        if result and result[0] is not None:
            return {'min': result[0], 'max': result[1]}
        return None
    except:
        return None


def analyze_cardinality(cursor, table: str, fk: Dict[str, Any]) -> Optional[str]:
    """
    Analyze cardinality of foreign key relationship (from iter3).
    Returns hint like "1:N (avg 5.2 rows per parent)" or "1:1".
    """
    try:
        from_col = fk['from_column']
        to_table = fk['to_table']

        # Skip self-referencing foreign keys
        if table == to_table:
            return "self-reference"

        # Count how many child rows per parent
        cursor.execute(f"""
            SELECT AVG(cnt) as avg_cnt, MAX(cnt) as max_cnt
            FROM (
                SELECT `{from_col}`, COUNT(*) as cnt
                FROM `{table}`
                WHERE `{from_col}` IS NOT NULL
                GROUP BY `{from_col}`
            )
        """)
        result = cursor.fetchone()

        if result and result[0] is not None:
            avg_cnt = result[0]
            max_cnt = result[1]

            if avg_cnt < 1.1 and max_cnt == 1:
                return "1:1"
            elif avg_cnt < 2.0:
                return f"1:N (avg {avg_cnt:.1f} rows per parent)"
            else:
                return f"1:N (avg {avg_cnt:.1f} rows per parent, max {max_cnt})"

        return None
    except:
        return None


# ============================================================================
# NEW: NULL/NaN PATTERN DETECTION
# ============================================================================

def detect_null_patterns(cursor, table: str, column: str, row_count: int) -> Dict[str, Any]:
    """
    NEW: Detect NULL, 'nan', and empty string patterns.

    Returns counts and handling hints for each pattern type.
    """
    patterns = {
        'null_count': 0,
        'null_percentage': 0.0,
        'nan_string_count': 0,
        'empty_string_count': 0,
        'handling_hint': None
    }

    try:
        # NULL count
        cursor.execute(f"SELECT COUNT(*) FROM `{table}` WHERE `{column}` IS NULL")
        null_count = cursor.fetchone()[0]
        patterns['null_count'] = null_count
        patterns['null_percentage'] = (null_count / row_count * 100) if row_count > 0 else 0.0

        # String 'nan' count (case-insensitive)
        cursor.execute(f"""
            SELECT COUNT(*) FROM `{table}`
            WHERE LOWER(CAST(`{column}` AS TEXT)) IN ('nan', 'n/a', 'null', 'none')
        """)
        nan_string_count = cursor.fetchone()[0]
        patterns['nan_string_count'] = nan_string_count

        # Empty string count
        cursor.execute(f"SELECT COUNT(*) FROM `{table}` WHERE `{column}` = ''")
        empty_count = cursor.fetchone()[0]
        patterns['empty_string_count'] = empty_count

        # Generate handling hint
        if nan_string_count > 0:
            patterns['handling_hint'] = "Use REPLACE(column, 'nan', '0') before CAST() for numeric operations"
        elif null_count > row_count * 0.5:
            patterns['handling_hint'] = "High NULL percentage - consider LEFT JOIN instead of INNER JOIN"
        elif empty_count > 0:
            patterns['handling_hint'] = "Contains empty strings - use LENGTH(column) > 0 or column != ''"

    except:
        pass  # Return default patterns if query fails

    return patterns


# ============================================================================
# NEW: STRING MATCHING PATTERN DETECTION
# ============================================================================

def detect_string_matching_pattern(column_name: str, enum_values: Optional[List[Any]]) -> Optional[str]:
    """
    NEW: Determine if column should use LIKE or = for matching.

    Rules:
    - If enum values contain delimiters (comma, pipe, semicolon): Suggest LIKE
    - If column name is plural but values are composite: Warn about LIKE
    - Otherwise: Use exact = matching
    """
    if not enum_values:
        return None

    # Check for multi-value delimiters in enum values
    delimiters = [',', '|', ';', '&']
    has_delimiters = any(
        any(delimiter in str(val) for delimiter in delimiters)
        for val in enum_values[:10]  # Check first 10 values
    )

    if has_delimiters:
        return "Multi-value column detected - may need LIKE pattern matching instead of exact ="

    # Check if column name is plural (ends with 's')
    if column_name.endswith('s') and not column_name.endswith('ss'):
        return "Column name is plural - check if values are single or multi-category"

    return None


# ============================================================================
# ADVANCED ANALYSIS FUNCTIONS (from iter9 - NEW capabilities)
# ============================================================================

def detect_semantic_patterns(cursor, tables: List[str]) -> Dict[str, Any]:
    """
    Detect semantic patterns across database (from iter9).

    Patterns detected:
    - Temporal sequences: Multiple date/time columns suggesting a sequence
    - Hierarchical relationships: Columns suggesting category/subcategory structure
    - Composite identifiers: Multi-column patterns suggesting natural keys
    """
    patterns = {
        'temporal_sequences': [],
        'hierarchical_relationships': [],
        'composite_identifiers': []
    }

    for table in tables:
        try:
            columns = get_table_info(cursor, table)
            col_names = [col['name'].lower() for col in columns]

            # Detect temporal sequences (2+ date/time columns)
            date_time_cols = [col['name'] for col in columns
                            if any(keyword in col['name'].lower()
                                 for keyword in ['date', 'time', 'timestamp', 'created', 'updated', 'modified'])]
            if len(date_time_cols) >= 2:
                patterns['temporal_sequences'].append({
                    'table': table,
                    'columns': date_time_cols,
                    'hint': f'Temporal sequence detected - consider date ordering and filtering'
                })

            # Detect hierarchical relationships (category/subcategory patterns)
            hierarchy_keywords = [
                ('category', 'subcategory'),
                ('parent', 'child'),
                ('type', 'subtype'),
                ('class', 'subclass')
            ]
            for parent_kw, child_kw in hierarchy_keywords:
                if parent_kw in col_names and child_kw in col_names:
                    patterns['hierarchical_relationships'].append({
                        'table': table,
                        'parent_col': [c['name'] for c in columns if parent_kw in c['name'].lower()][0],
                        'child_col': [c['name'] for c in columns if child_kw in c['name'].lower()][0],
                        'hint': 'Hierarchical structure - may need recursive queries or multi-level filtering'
                    })

            # Detect composite identifiers (multiple columns with 'id', 'code', 'num' in name)
            id_cols = [col['name'] for col in columns
                      if any(keyword in col['name'].lower()
                           for keyword in ['_id', '_code', '_num', '_key'])
                      and not col['pk']]
            if len(id_cols) >= 2:
                patterns['composite_identifiers'].append({
                    'table': table,
                    'columns': id_cols,
                    'hint': 'Multiple identifier columns - may form composite key or require multi-column matching'
                })

        except Exception as e:
            # Skip tables that cause errors
            continue

    return patterns


def validate_cross_table_values(cursor, tables: List[str]) -> List[Dict[str, Any]]:
    """
    Check if foreign key values actually exist in parent tables (from iter9).

    Detects:
    - Orphaned foreign keys (FK values with no matching parent)
    - Broken relationships that might cause unexpected JOIN results
    """
    validation_results = []

    for table in tables:
        try:
            fks = analyze_foreign_keys(cursor, table)
            for fk in fks:
                from_col = fk['from_column']
                to_table = fk['to_table']
                to_col = fk['to_column']

                # Skip self-referencing FKs
                if table == to_table:
                    continue

                # Check for orphaned foreign keys
                cursor.execute(f"""
                    SELECT COUNT(*) FROM `{table}` t
                    LEFT JOIN `{to_table}` p ON t.`{from_col}` = p.`{to_col}`
                    WHERE t.`{from_col}` IS NOT NULL AND p.`{to_col}` IS NULL
                """)
                orphaned = cursor.fetchone()[0]

                # Get total non-null FK values
                cursor.execute(f"SELECT COUNT(*) FROM `{table}` WHERE `{from_col}` IS NOT NULL")
                total_fk_values = cursor.fetchone()[0]

                if orphaned > 0:
                    match_rate = ((total_fk_values - orphaned) / total_fk_values * 100) if total_fk_values > 0 else 0
                    validation_results.append({
                        'table': table,
                        'fk_column': from_col,
                        'parent_table': to_table,
                        'parent_column': to_col,
                        'orphaned_count': orphaned,
                        'total_count': total_fk_values,
                        'match_rate': f'{match_rate:.1f}%',
                        'warning': f'⚠️ {orphaned}/{total_fk_values} foreign key values have no match in parent table'
                    })

        except Exception as e:
            # Skip tables that cause errors
            continue

    return validation_results


def analyze_distributions(cursor, table: str, columns: List[Dict[str, Any]],
                         row_count: int) -> Dict[str, Dict[str, Any]]:
    """
    Provide statistical distribution hints for numeric columns (from iter9).

    Analyzes:
    - Cardinality (distinct values / total values)
    - Uniqueness patterns
    - Distribution characteristics
    """
    distributions = {}

    for col in columns:
        col_name = col['name']
        col_type = col['type']

        # Only analyze numeric columns
        if col_type.upper() not in ('INTEGER', 'REAL', 'NUMERIC', 'FLOAT', 'DOUBLE'):
            continue

        try:
            cursor.execute(f"""
                SELECT
                    MIN(`{col_name}`) as min_val,
                    MAX(`{col_name}`) as max_val,
                    AVG(`{col_name}`) as avg_val,
                    COUNT(DISTINCT `{col_name}`) as distinct_count,
                    COUNT(*) as non_null_count
                FROM `{table}`
                WHERE `{col_name}` IS NOT NULL
            """)
            stats = cursor.fetchone()

            if stats and stats[3] is not None:  # distinct_count
                min_val, max_val, avg_val, distinct, non_null = stats

                # Calculate cardinality ratio
                cardinality = distinct / non_null if non_null > 0 else 0

                # Classify distribution
                if distinct < 20:
                    dist_type = 'enum-like'
                elif cardinality > 0.95:
                    dist_type = 'unique-like'
                elif cardinality < 0.1:
                    dist_type = 'low-cardinality'
                else:
                    dist_type = 'normal'

                distributions[col_name] = {
                    'range': (min_val, max_val),
                    'average': round(avg_val, 2) if avg_val is not None else None,
                    'distinct_values': distinct,
                    'cardinality_ratio': round(cardinality, 3),
                    'distribution_type': dist_type
                }

        except Exception as e:
            # Skip columns that cause errors
            continue

    return distributions


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def analyze_database(db_path: str, output_file: str) -> int:
    """
    Generate comprehensive hybrid database analysis.

    Combines best features from iter16, iter9, and iter3:
    - Adaptive scaling based on database size
    - Complete schema and table overview
    - ALL column coverage with samples and formats
    - Enum detection, value ranges, cardinality
    - Semantic patterns, cross-table validation, statistical hints (when size allows)
    - NEW: String matching guidance and NULL/NaN pattern detection
    - Comprehensive query guidance with evidence warnings
    """

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        output = []

        # Get tables and classify database size
        tables = get_tables(cursor)
        size_config = classify_database_size(cursor, tables)
        table_row_counts = {table: get_row_count(cursor, table) for table in tables}

        # Header with size classification
        output.append("# HYBRID COMPREHENSIVE DATABASE ANALYSIS")
        output.append("")
        output.append(f"**Database Size**: {size_config['size'].upper()}")
        output.append(f"**Tables**: {size_config['table_count']}, **Columns**: {size_config['column_count']}")
        output.append(f"**Analysis Strategy**: {size_config['strategy']}")
        output.append("")
        output.append("Cross-pollinated analyzer combining:")
        output.append("- Adaptive scaling (iter16) - prevents context overflow")
        output.append("- Advanced features (iter9) - semantic patterns, validation, statistics")
        output.append("- Precision techniques (iter3) - ALL column coverage, enums, ranges")
        output.append("- NEW: String matching guidance and NULL/NaN pattern detection")
        output.append("")

        # Track data for summary sections
        all_enums = {}
        all_ranges = {}
        all_distributions = {}
        all_null_patterns = {}
        all_string_patterns = {}

        # ====================================================================
        # SECTION 1: COMPLETE SCHEMA (DDL)
        # ====================================================================
        output.append("## 1. COMPLETE SCHEMA (DDL)")
        output.append("")
        schema_items = extract_schema(cursor)
        for name, sql in schema_items:
            output.append(sql)
            output.append("")

        # ====================================================================
        # SECTION 2: TABLE OVERVIEW
        # ====================================================================
        output.append(f"## 2. TABLE OVERVIEW")
        output.append(f"")
        output.append(f"Total tables: {len(tables)}")
        output.append("")

        for table in tables:
            row_count = table_row_counts.get(table, 0)
            output.append(f"- **{table}**: {row_count:,} rows")
        output.append("")

        # ====================================================================
        # SECTION 3: DETAILED COLUMN ANALYSIS (ALL COLUMNS)
        # ====================================================================
        output.append("## 3. DETAILED COLUMN ANALYSIS")
        output.append("")
        output.append("**Coverage**: Analyzing ALL columns in each table (not limited to first 10)")
        output.append("")

        for table in tables:
            output.append(f"### Table: {table}")
            output.append("")

            row_count = table_row_counts.get(table, 0)

            # Column info
            columns = get_table_info(cursor, table)
            output.append("**Columns:**")
            for col in columns:
                flags = []
                if col['pk']:
                    flags.append('PRIMARY KEY')
                if col['notnull']:
                    flags.append('NOT NULL')
                flag_str = f" ({', '.join(flags)})" if flags else ""
                output.append(f"- `{col['name']}` {col['type']}{flag_str}")
            output.append("")

            # Foreign keys with cardinality
            fks = analyze_foreign_keys(cursor, table)
            if fks:
                output.append("**Foreign Keys:**")
                for fk in fks:
                    cardinality = analyze_cardinality(cursor, table, fk)
                    card_str = f" [{cardinality}]" if cardinality else ""
                    output.append(f"- `{fk['from_column']}` → `{fk['to_table']}.{fk['to_column']}`{card_str}")
                output.append("")

            # Sample data and analysis for each column
            output.append("**Column Details:**")
            for col in columns:
                col_name = col['name']
                col_type = col['type']

                # Sample values (adaptive limit)
                samples = sample_column_values(cursor, table, col_name, size_config['sample_limit'])

                # Format detection
                format_info = detect_value_format(samples)

                # NULL patterns (NEW)
                null_patterns = detect_null_patterns(cursor, table, col_name, row_count)

                # Build column detail output
                detail_parts = [f"`{col_name}`"]

                if samples:
                    detail_parts.append(f"Samples: {samples}")

                if format_info['format'] != 'text':
                    detail_parts.append(f"Format: {format_info['pattern']}")
                    if 'parsing' in format_info:
                        detail_parts.append(f"Parse: {format_info['parsing']}")

                if null_patterns['null_percentage'] > 0:
                    detail_parts.append(f"NULL: {null_patterns['null_percentage']:.1f}%")

                if null_patterns['nan_string_count'] > 0:
                    detail_parts.append(f"String 'nan': {null_patterns['nan_string_count']}")
                    all_null_patterns[f"{table}.{col_name}"] = null_patterns

                output.append(f"  - {' | '.join(detail_parts)}")

                # Enum detection (adaptive threshold)
                if size_config['enum_limit'] is None or size_config['enum_limit'] > 0:
                    enum_threshold = size_config['enum_limit'] if size_config['enum_limit'] else 20
                    enum_values = detect_enum_values(cursor, table, col_name, enum_threshold, row_count)

                    if enum_values:
                        # Truncate if needed
                        if size_config['enum_limit'] and len(enum_values) > size_config['enum_limit']:
                            enum_display = enum_values[:size_config['enum_limit']]
                            all_enums[f"{table}.{col_name}"] = (enum_display, len(enum_values))
                        else:
                            all_enums[f"{table}.{col_name}"] = (enum_values, len(enum_values))

                        # String matching pattern detection (NEW)
                        matching_pattern = detect_string_matching_pattern(col_name, enum_values)
                        if matching_pattern:
                            all_string_patterns[f"{table}.{col_name}"] = matching_pattern

                # Value range for numeric columns
                value_range = get_value_range(cursor, table, col_name, col_type)
                if value_range:
                    all_ranges[f"{table}.{col_name}"] = value_range

            output.append("")

            # Statistical distributions (if not skipping advanced)
            if not size_config['skip_advanced']:
                distributions = analyze_distributions(cursor, table, columns, row_count)
                if distributions:
                    all_distributions[table] = distributions

        # ====================================================================
        # SECTION 4: RELATIONSHIP MAP
        # ====================================================================
        output.append("## 4. RELATIONSHIP MAP")
        output.append("")
        output.append("Foreign key relationships with cardinality hints:")
        output.append("")

        has_relationships = False
        for table in tables:
            fks = analyze_foreign_keys(cursor, table)
            if fks:
                has_relationships = True
                for fk in fks:
                    cardinality = analyze_cardinality(cursor, table, fk)
                    card_str = f" [{cardinality}]" if cardinality else ""
                    output.append(f"- `{table}.{fk['from_column']}` → `{fk['to_table']}.{fk['to_column']}`{card_str}")

        if not has_relationships:
            output.append("No foreign key relationships defined.")

        output.append("")

        # ====================================================================
        # SECTION 5: ENUM VALUE REFERENCE
        # ====================================================================
        output.append("## 5. ENUM VALUE REFERENCE")
        output.append("")
        output.append("Columns with limited distinct values (use for exact matching):")
        output.append("")

        if all_enums:
            for col_path, (values, total_count) in sorted(all_enums.items()):
                truncated = " (truncated)" if len(values) < total_count else ""
                output.append(f"**{col_path}** ({total_count} distinct values{truncated}):")
                output.append(f"  Values: {values}")
                output.append("")
        else:
            output.append("No enum columns detected (or enum detection skipped for large database).")
            output.append("")

        # ====================================================================
        # SECTION 6: VALUE RANGE SUMMARY
        # ====================================================================
        output.append("## 6. VALUE RANGE SUMMARY")
        output.append("")
        output.append("Numeric column boundaries (MIN/MAX):")
        output.append("")

        if all_ranges:
            for col_path, range_info in sorted(all_ranges.items()):
                output.append(f"- **{col_path}**: MIN={range_info['min']}, MAX={range_info['max']}")
        else:
            output.append("No numeric ranges calculated.")

        output.append("")

        # ====================================================================
        # SECTION 7: FORMAT DETECTION SUMMARY (NEW - includes NULL/NaN)
        # ====================================================================
        output.append("## 7. FORMAT DETECTION SUMMARY")
        output.append("")

        if all_null_patterns:
            output.append("**NULL/NaN Patterns Detected:**")
            output.append("")
            for col_path, patterns in sorted(all_null_patterns.items()):
                output.append(f"**{col_path}**:")
                output.append(f"  - NULL count: {patterns['null_count']} ({patterns['null_percentage']:.1f}%)")
                if patterns['nan_string_count'] > 0:
                    output.append(f"  - String 'nan' count: {patterns['nan_string_count']}")
                if patterns['empty_string_count'] > 0:
                    output.append(f"  - Empty string count: {patterns['empty_string_count']}")
                if patterns['handling_hint']:
                    output.append(f"  - **Handling**: {patterns['handling_hint']}")
                output.append("")
        else:
            output.append("No significant NULL/NaN patterns detected.")
            output.append("")

        # ====================================================================
        # SECTION 8: STRING MATCHING GUIDANCE (NEW)
        # ====================================================================
        if all_string_patterns:
            output.append("## 8. STRING MATCHING GUIDANCE")
            output.append("")
            output.append("Columns requiring special matching consideration:")
            output.append("")
            for col_path, pattern in sorted(all_string_patterns.items()):
                output.append(f"**{col_path}**: {pattern}")
            output.append("")

        # ====================================================================
        # SECTION 9: SEMANTIC PATTERNS (if not skipping advanced)
        # ====================================================================
        if not size_config['skip_advanced']:
            output.append("## 9. SEMANTIC PATTERNS")
            output.append("")

            patterns = detect_semantic_patterns(cursor, tables)

            if patterns['temporal_sequences']:
                output.append("**Temporal Sequences:**")
                for p in patterns['temporal_sequences']:
                    output.append(f"- Table `{p['table']}`: {p['columns']}")
                    output.append(f"  Hint: {p['hint']}")
                output.append("")

            if patterns['hierarchical_relationships']:
                output.append("**Hierarchical Relationships:**")
                for p in patterns['hierarchical_relationships']:
                    output.append(f"- Table `{p['table']}`: `{p['parent_col']}` → `{p['child_col']}`")
                    output.append(f"  Hint: {p['hint']}")
                output.append("")

            if patterns['composite_identifiers']:
                output.append("**Composite Identifiers:**")
                for p in patterns['composite_identifiers']:
                    output.append(f"- Table `{p['table']}`: {p['columns']}")
                    output.append(f"  Hint: {p['hint']}")
                output.append("")

            if not any(patterns.values()):
                output.append("No semantic patterns detected.")
                output.append("")

        # ====================================================================
        # SECTION 10: CROSS-TABLE VALIDATION (if not skipping advanced)
        # ====================================================================
        if not size_config['skip_advanced']:
            output.append("## 10. CROSS-TABLE VALIDATION")
            output.append("")

            validation_results = validate_cross_table_values(cursor, tables)

            if validation_results:
                output.append("**Orphaned Foreign Key Warnings:**")
                output.append("")
                for v in validation_results:
                    output.append(f"**{v['table']}.{v['fk_column']}** → **{v['parent_table']}.{v['parent_column']}**")
                    output.append(f"  {v['warning']}")
                    output.append(f"  Match rate: {v['match_rate']}")
                    output.append(f"  **Recommendation**: Consider LEFT JOIN to preserve {v['orphaned_count']} unmatched rows")
                    output.append("")
            else:
                output.append("All foreign key values have matching parents - no orphaned relationships detected.")
                output.append("")

        # ====================================================================
        # SECTION 11: STATISTICAL DISTRIBUTION HINTS (if not skipping advanced)
        # ====================================================================
        if not size_config['skip_advanced'] and all_distributions:
            output.append("## 11. STATISTICAL DISTRIBUTION HINTS")
            output.append("")

            for table, distributions in all_distributions.items():
                if distributions:
                    output.append(f"**Table: {table}**")
                    for col_name, dist in distributions.items():
                        output.append(f"- `{col_name}`: {dist['distribution_type']}")
                        output.append(f"  Range: {dist['range'][0]} to {dist['range'][1]}")
                        if dist['average']:
                            output.append(f"  Average: {dist['average']}")
                        output.append(f"  Distinct values: {dist['distinct_values']} (cardinality: {dist['cardinality_ratio']})")
                    output.append("")

        # ====================================================================
        # SECTION 12: QUERY GUIDANCE
        # ====================================================================
        output.append("## 12. QUERY GUIDANCE")
        output.append("")

        output.append("### General Best Practices")
        output.append("")
        output.append("1. **Check Section 5 (ENUM VALUE REFERENCE) for exact string values** - SQLite is case-sensitive")
        output.append("2. **Check Section 7 (FORMAT DETECTION) for NULL/NaN patterns** - handle appropriately")
        output.append("3. **Check Section 4 (RELATIONSHIP MAP) for JOIN relationships** - note cardinality hints")

        if not size_config['skip_advanced']:
            output.append("4. **Check Section 10 (CROSS-TABLE VALIDATION) for orphaned FK warnings** - use LEFT JOIN if needed")

        output.append("")

        output.append("### Evidence Interpretation Warnings")
        output.append("")
        output.append("⚠️ **Common Evidence Pitfalls**:")
        output.append("")
        output.append("1. **Value Contradictions**: Evidence may say 'X = 1.0' when question mentions '-1'")
        output.append("   - Trust the evidence definition, not the question phrasing")
        output.append("")
        output.append("2. **Column Plurals**: If evidence says 'Genre = X' but table has 'Genres' (plural)")
        output.append("   - Check Section 5 for actual values - may be multi-category")
        output.append("   - Check Section 8 for string matching guidance")
        output.append("   - May need LIKE '%X%' instead of exact match")
        output.append("")
        output.append("3. **NaN Handling**: String 'nan' vs NULL vs numeric NaN")
        output.append("   - Check Section 7 for 'nan' string counts")
        output.append("   - Use REPLACE(column, 'nan', '0') before CAST() if present")
        output.append("   - Section 7 provides specific handling hints")
        output.append("")
        output.append("4. **Multi-Condition Formulas**: Break complex evidence into parts")
        output.append("   - Identify numerator conditions vs denominator base")
        output.append("   - Apply all conditions even if not obvious from question")
        output.append("")

        if all_string_patterns:
            output.append("### String Matching Recommendations")
            output.append("")
            output.append("Based on enum value analysis, these columns may require LIKE pattern matching:")
            for col_path in sorted(all_string_patterns.keys()):
                output.append(f"- {col_path}")
            output.append("")

        if all_null_patterns:
            output.append("### NULL/NaN Handling Recommendations")
            output.append("")
            output.append("Columns with special NULL/NaN patterns requiring handling:")
            for col_path, patterns in sorted(all_null_patterns.items()):
                if patterns['handling_hint']:
                    output.append(f"- {col_path}: {patterns['handling_hint']}")
            output.append("")

        output.append("### Database-Specific Patterns")
        output.append("")
        output.append(f"This is a **{size_config['size']}** database with {size_config['column_count']} columns.")

        if size_config['size'] == 'small':
            output.append("Full comprehensive analysis provided - all advanced features available.")
        elif size_config['size'] == 'medium':
            output.append("Smart truncation applied - essential features preserved with abbreviated detail.")
        elif size_config['size'] == 'large':
            output.append("Aggressive truncation applied - core schema and relationships preserved.")
        else:  # ultra_large
            output.append("Ultra-compact analysis - minimal detail to prevent context overflow.")

        output.append("")

        # Write output
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output))

        print(f"✅ Hybrid comprehensive analysis complete ({size_config['size']} database)")
        print(f"   Tables: {size_config['table_count']}, Columns: {size_config['column_count']}")
        print(f"   Strategy: {size_config['strategy']}")
        print(f"   Output: {output_file}")

        conn.close()
        return 0

    except Exception as e:
        print(f"❌ ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(analyze_database("database.sqlite", "tool_output/analysis.txt"))
