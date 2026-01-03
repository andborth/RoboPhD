#!/usr/bin/env python3
"""
Comprehensive Unified Cross-Pollinated Analyzer - Iteration 016

Combines the best tool-only patterns from top-performing agents:
- iter6 (ELO 1531): Column ownership, cross-table hints, output limiting
- iter7 (ELO 1514): City enumeration, spelling variation detection
- iter9 (ELO 1472): Entity tables, currency detection, authoritative sources
- iter10 (ELO 1523): Unified analysis with enhanced patterns
- iter14 (ELO 1550): Semantic column classification, pre-aggregated detection
- iter15 (ELO 1503): Date format detection, strftime compatibility, JOIN patterns

Key innovations in iter16:
1. Evidence-first emphasis in output (based on error analysis)
2. Enhanced column selection warnings
3. Timestamp pattern analysis (SUBSTR vs datetime)
4. Boolean/flag column detection
5. Combined semantic + structural analysis
"""

import sqlite3
import os
import re
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple, Set


# OUTPUT SIZE LIMITS (proven from iter6/iter15)
MAX_OUTPUT_CHARS = 72000
MAX_SAMPLE_COLUMNS = 85
MAX_CATEGORICAL_COLUMNS = 50
MAX_SAMPLES_PER_COLUMN = 5
MAX_CATEGORICAL_VALUES = 12
SKIP_SAMPLES_THRESHOLD = 40
MAX_VARIATION_COLUMNS = 30
MAX_VALUE_ENUMERATION = 100


def connect_db(db_path: str = "database.sqlite") -> sqlite3.Connection:
    """Connect to the SQLite database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def get_raw_ddl(cursor) -> List[Tuple[str, str]]:
    """Get raw DDL statements preserving exact table name quoting."""
    cursor.execute("""
        SELECT name, sql
        FROM sqlite_master
        WHERE type='table' AND sql IS NOT NULL AND name NOT LIKE 'sqlite_%'
        ORDER BY name
    """)
    return [(row[0], row[1]) for row in cursor.fetchall()]


def get_tables(cursor) -> List[str]:
    """Get all table names from the database."""
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
    """)
    return [row[0] for row in cursor.fetchall()]


def needs_quoting(table_name: str) -> bool:
    """Check if table name needs quoting."""
    if not table_name:
        return False
    if ' ' in table_name or '-' in table_name:
        return True
    if table_name[0].isdigit():
        return True
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table_name):
        return True
    return False


def get_quoted_name(table_name: str) -> str:
    """Return properly quoted table name if needed."""
    if needs_quoting(table_name):
        return f'`{table_name}`'
    return table_name


def is_history_table(table_name: str) -> bool:
    """Detect if table is a history/audit table."""
    name_lower = table_name.lower()
    history_indicators = ['history', 'historical', 'audit', 'log', 'archive', 'backup', '_old', '_bak']
    return any(ind in name_lower for ind in history_indicators)


def is_entity_table(table_name: str, columns: List[Dict]) -> bool:
    """Detect if table is an authoritative entity table (from iter9/iter10)."""
    name_lower = table_name.lower()
    entity_patterns = [
        'employee', 'customer', 'product', 'user', 'person', 'patient',
        'student', 'teacher', 'store', 'country', 'city', 'region',
        'member', 'account', 'supplier', 'vendor', 'actor', 'director',
        'athlete', 'competitor', 'player', 'team', 'company', 'organization',
        'gene', 'classification', 'legislator', 'dish', 'menu', 'paper',
        'station', 'trip', 'weather', 'institution', 'school', 'university',
        'movie', 'film', 'list', 'rating', 'review', 'publisher', 'author'
    ]

    for pattern in entity_patterns:
        if pattern in name_lower:
            return True

    # Check for name-like columns (suggests entity table)
    name_columns = sum(1 for c in columns if 'name' in c['name'].lower() or c['name'].lower() in ['first', 'last', 'title'])
    return name_columns >= 1


def detect_date_format_from_values(cursor, table: str, column: str) -> Dict:
    """
    Detect actual date format from sample values.
    From iter13/iter15 - Critical for databases with different date formats.
    """
    format_info = {
        'detected_format': None,
        'is_iso': False,
        'has_time': False,
        'sample': None,
        'strftime_compatible': False,
        'parse_hint': None,
        'year_extraction': None
    }

    try:
        quoted_table = get_quoted_name(table)
        cursor.execute(f"""
            SELECT DISTINCT `{column}` FROM {quoted_table}
            WHERE `{column}` IS NOT NULL
            LIMIT 5
        """)
        samples = [row[0] for row in cursor.fetchall()]
    except:
        return format_info

    if not samples:
        return format_info

    for sample in samples:
        if sample is None:
            continue
        s = str(sample)
        format_info['sample'] = s

        # Check for ISO format (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
        if re.match(r'^\d{4}-\d{2}-\d{2}', s):
            format_info['detected_format'] = 'ISO (YYYY-MM-DD)'
            format_info['is_iso'] = True
            format_info['strftime_compatible'] = True
            format_info['has_time'] = ' ' in s or 'T' in s
            format_info['parse_hint'] = "strftime('%Y', col) works"
            format_info['year_extraction'] = f"strftime('%Y', `{column}`)"
            break

        # Check for US format (M/D/YYYY or MM/DD/YYYY with optional time)
        if re.match(r'^\d{1,2}/\d{1,2}/\d{4}', s):
            format_info['detected_format'] = 'US (M/D/YYYY)'
            format_info['is_iso'] = False
            format_info['strftime_compatible'] = False
            format_info['has_time'] = ' ' in s
            if format_info['has_time']:
                format_info['parse_hint'] = "Use SUBSTR for date extraction"
                format_info['year_extraction'] = f"SUBSTR(`{column}`, -4, 4)"
            else:
                format_info['parse_hint'] = "Year: SUBSTR(col, -4) NOT strftime()"
                format_info['year_extraction'] = f"SUBSTR(`{column}`, -4, 4)"
            break

        # Check for timestamp format (YYYY-MM-DD HH:MM:SS or similar)
        if re.match(r'^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}', s):
            format_info['detected_format'] = 'ISO Timestamp'
            format_info['is_iso'] = True
            format_info['strftime_compatible'] = True
            format_info['has_time'] = True
            format_info['parse_hint'] = "strftime() and datetime() work"
            format_info['year_extraction'] = f"strftime('%Y', `{column}`)"
            break

        # Check for plain year
        if re.match(r'^\d{4}$', s):
            format_info['detected_format'] = 'Year Only'
            format_info['is_iso'] = True
            format_info['strftime_compatible'] = False
            format_info['has_time'] = False
            format_info['parse_hint'] = "Already a year - use directly"
            format_info['year_extraction'] = f"`{column}`"
            break

    return format_info


def detect_date_column(col_name: str, col_type: str) -> bool:
    """Check if column is likely a date column based on name and type."""
    name_lower = col_name.lower()
    type_upper = col_type.upper() if col_type else ''

    date_keywords = ['date', 'time', 'day', 'month', 'year', 'start', 'end', 'created', 'updated',
                     'birthday', 'birthdate', 'deathdate', 'timestamp', 'utc']

    if any(kw in name_lower for kw in date_keywords):
        return True

    if 'DATE' in type_upper or 'TIME' in type_upper:
        return True

    return False


def detect_timestamp_column(col_name: str) -> bool:
    """Detect if column is a timestamp (for SUBSTR vs datetime decisions)."""
    name_lower = col_name.lower()
    timestamp_keywords = ['timestamp', '_utc', 'created_at', 'updated_at', 'datetime']
    return any(kw in name_lower for kw in timestamp_keywords)


def find_date_join_patterns(table_info: Dict, date_formats: Dict) -> List[Dict]:
    """
    Find potential date JOINs between tables and generate recommendations.
    From iter13/iter15.
    """
    patterns = []

    date_tables = []
    for table, formats in date_formats.items():
        for col, fmt_info in formats.items():
            if fmt_info.get('detected_format'):
                date_tables.append({
                    'table': table,
                    'column': col,
                    'format': fmt_info['detected_format'],
                    'has_time': fmt_info.get('has_time', False),
                    'is_iso': fmt_info.get('is_iso', False),
                    'sample': fmt_info.get('sample'),
                    'year_extraction': fmt_info.get('year_extraction')
                })

    for i, dt1 in enumerate(date_tables):
        for dt2 in date_tables[i+1:]:
            if dt1['table'] == dt2['table']:
                continue

            if dt1['format'] != dt2['format'] or dt1['has_time'] != dt2['has_time']:
                suggestion = generate_date_join_suggestion(dt1, dt2)
                patterns.append({
                    'table1': dt1['table'],
                    'column1': dt1['column'],
                    'format1': dt1['format'],
                    'sample1': dt1['sample'],
                    'table2': dt2['table'],
                    'column2': dt2['column'],
                    'format2': dt2['format'],
                    'sample2': dt2['sample'],
                    'warning': 'FORMAT MISMATCH - dates may not compare correctly!',
                    'suggestion': suggestion
                })

    return patterns[:6]


def generate_date_join_suggestion(dt1: Dict, dt2: Dict) -> str:
    """Generate SQL JOIN suggestion for date columns with different formats."""
    t1, c1 = dt1['table'], dt1['column']
    t2, c2 = dt2['table'], dt2['column']

    qt1 = get_quoted_name(t1)
    qt2 = get_quoted_name(t2)

    if dt1.get('has_time') and not dt2.get('has_time') and not dt1.get('is_iso'):
        return f"JOIN {qt2} ON SUBSTR({qt1}.`{c1}`, 1, instr({qt1}.`{c1}`, ' ')-1) = {qt2}.`{c2}`"

    if dt2.get('has_time') and not dt1.get('has_time') and not dt2.get('is_iso'):
        return f"JOIN {qt2} ON {qt1}.`{c1}` = SUBSTR({qt2}.`{c2}`, 1, instr({qt2}.`{c2}`, ' ')-1)"

    if dt1.get('is_iso') and dt2.get('is_iso'):
        if dt1.get('has_time') and not dt2.get('has_time'):
            return f"JOIN {qt2} ON DATE({qt1}.`{c1}`) = {qt2}.`{c2}`"
        if dt2.get('has_time') and not dt1.get('has_time'):
            return f"JOIN {qt2} ON {qt1}.`{c1}` = DATE({qt2}.`{c2}`)"

    return f"JOIN {qt2} ON {qt1}.`{c1}` = {qt2}.`{c2}` -- CHECK formats match!"


def detect_currency_format(values: List[Any]) -> bool:
    """Detect if values appear to be currency strings (from iter9/iter10)."""
    if not values:
        return False

    str_values = [str(v) for v in values if v is not None]
    currency_count = 0
    for val in str_values[:5]:
        if val.startswith('$') or val.startswith('-$'):
            currency_count += 1
        elif re.match(r'^[\d,]+\.\d{2}$', val):
            currency_count += 1

    return currency_count >= len(str_values[:5]) * 0.5 if str_values else False


def is_pre_aggregated_column(col_name: str) -> Tuple[bool, Optional[str]]:
    """
    Detect columns that already contain aggregated values.
    These need SUM() not COUNT() in many cases.
    From iter14/iter15.
    """
    col_lower = col_name.lower()

    # Exact matches or suffix matches for count-type columns
    count_suffixes = ['_count', '_cnt', '_num', '_total', '_qty', '_quantity', '_followers']
    count_exact = ['count', 'cnt', 'total', 'quantity', 'qty', 'votes', 'units', 'followers']

    for suffix in count_suffixes:
        if col_lower.endswith(suffix):
            return True, f"Pre-aggregated count. Use SUM({col_name}) for totals, not COUNT(*)."

    for exact in count_exact:
        if col_lower == exact:
            return True, f"Pre-aggregated count. Use SUM({col_name}) for totals, not COUNT(*)."

    # Population columns
    population_patterns = ['population', 'male_pop', 'female_pop', 'pop_']
    for pattern in population_patterns:
        if pattern in col_lower:
            return True, f"Population count. Use SUM({col_name}) for totals."

    # Graduation cohort columns
    grad_patterns = ['grad_cohort', 'grad_100', 'grad_150', 'cohort']
    for pattern in grad_patterns:
        if pattern in col_lower:
            return True, f"Cohort count. May need SUM({col_name}) for totals OR COUNT(*) for records."

    return False, None


def is_boolean_flag_column(col_name: str, values: List[Any]) -> Tuple[bool, Optional[str]]:
    """
    NEW: Detect boolean/flag columns that indicate states.
    Critical for movie_platform errors where user_subscriber was misinterpreted.
    """
    col_lower = col_name.lower()

    # Common flag column patterns
    flag_patterns = ['_subscriber', 'is_', 'has_', '_flag', '_active', '_enabled', '_valid', '_paid']

    for pattern in flag_patterns:
        if pattern in col_lower:
            # Check if values are 0/1 or True/False
            if values:
                val_set = set(str(v) for v in values[:10] if v is not None)
                if val_set.issubset({'0', '1', 'True', 'False', 'true', 'false', 'Y', 'N', 'Yes', 'No'}):
                    return True, f"Boolean flag column. 1/True = yes, 0/False = no. Follow evidence literally!"

    # Check for 0/1 only values
    if values:
        val_set = set(str(v) for v in values[:10] if v is not None)
        if val_set == {'0', '1'} or val_set == {'0'} or val_set == {'1'}:
            return True, f"Binary flag (0/1). Use exactly as specified in evidence."

    return False, None


def get_table_info(cursor, table: str) -> Dict[str, Any]:
    """Get comprehensive information about a table."""
    info = {
        "columns": [],
        "primary_keys": [],
        "foreign_keys": [],
        "row_count": 0,
        "is_empty": False,
        "is_junction": False,
        "is_history": is_history_table(table),
        "is_entity": False,
        "needs_quoting": needs_quoting(table)
    }

    cursor.execute(f"PRAGMA table_info(`{table}`)")
    for row in cursor.fetchall():
        col_info = {
            "name": row[1],
            "type": row[2] or "TEXT",
            "nullable": row[3] == 0,
            "default": row[4],
            "primary_key": row[5] == 1
        }
        info["columns"].append(col_info)
        if row[5] == 1:
            info["primary_keys"].append(row[1])

    cursor.execute(f"PRAGMA foreign_key_list(`{table}`)")
    for row in cursor.fetchall():
        fk_info = {
            "column": row[3],
            "ref_table": row[2],
            "ref_column": row[4]
        }
        info["foreign_keys"].append(fk_info)

    try:
        quoted = get_quoted_name(table)
        cursor.execute(f"SELECT COUNT(*) FROM {quoted}")
        info["row_count"] = cursor.fetchone()[0]
    except:
        info["row_count"] = 0

    info["is_empty"] = info["row_count"] == 0

    fk_count = len(info["foreign_keys"])
    col_count = len(info["columns"])
    info["is_junction"] = fk_count >= 2 and fk_count >= col_count - 2

    info["is_entity"] = is_entity_table(table, info["columns"]) and not info["is_junction"]

    return info


def get_sample_values(cursor, table: str, column: str, limit: int = 5) -> List[Any]:
    """Get sample values from a column, preserving exact case."""
    try:
        quoted_table = get_quoted_name(table)
        cursor.execute(f"""
            SELECT DISTINCT `{column}` FROM {quoted_table}
            WHERE `{column}` IS NOT NULL
            LIMIT {limit}
        """)
        return [row[0] for row in cursor.fetchall()]
    except:
        return []


def get_distinct_count(cursor, table: str, column: str) -> int:
    """Get count of distinct values in a column."""
    try:
        quoted_table = get_quoted_name(table)
        cursor.execute(f"SELECT COUNT(DISTINCT `{column}`) FROM {quoted_table}")
        return cursor.fetchone()[0]
    except:
        return 0


def get_all_values(cursor, table: str, column: str, limit: int = 100) -> List[Any]:
    """Get all distinct values from a column up to limit."""
    try:
        quoted_table = get_quoted_name(table)
        cursor.execute(f"""
            SELECT DISTINCT `{column}` FROM {quoted_table}
            WHERE `{column}` IS NOT NULL
            ORDER BY `{column}`
            LIMIT {limit}
        """)
        return [row[0] for row in cursor.fetchall()]
    except:
        return []


def get_categorical_values(cursor, table: str, column: str, max_values: int = 12) -> Optional[List[Any]]:
    """Get all values for low-cardinality columns."""
    distinct_count = get_distinct_count(cursor, table, column)
    if distinct_count > max_values or distinct_count == 0:
        return None

    try:
        quoted_table = get_quoted_name(table)
        cursor.execute(f"""
            SELECT DISTINCT `{column}` FROM {quoted_table}
            WHERE `{column}` IS NOT NULL
            ORDER BY `{column}`
            LIMIT {max_values}
        """)
        return [row[0] for row in cursor.fetchall()]
    except:
        return None


def detect_value_variations(values: List[str]) -> List[Tuple[str, str, str]]:
    """
    Detect value variations (spelling, spacing, case differences).
    From iter7/iter10.
    """
    variations = []
    seen_normalized = {}

    str_values = [str(v) for v in values if v is not None]

    for val in str_values:
        norm_no_space = re.sub(r'\s+', '', val.lower())
        norm_single_space = re.sub(r'\s+', ' ', val.lower().strip())
        norm_no_punct = re.sub(r'[\s\-_]', '', val.lower())

        for norm_key, norm_type in [(norm_no_space, 'spacing'),
                                     (norm_single_space, 'spacing'),
                                     (norm_no_punct, 'punctuation')]:
            if norm_key in seen_normalized:
                original = seen_normalized[norm_key]
                if original != val and (original, val, norm_type) not in variations:
                    if '  ' in val or '  ' in original:
                        reason = "DOUBLE SPACES"
                    elif ' ' in val and ' ' not in original:
                        reason = "spacing differs"
                    elif '-' in val or '-' in original:
                        reason = "hyphen differs"
                    else:
                        reason = "case/format differs"
                    variations.append((original, val, reason))
            else:
                seen_normalized[norm_key] = val

    return variations[:6]


def find_values_with_spacing_issues(values: List[str]) -> List[str]:
    """Find values with spacing issues."""
    issues = []
    for val in values:
        if val is None:
            continue
        str_val = str(val)
        if '  ' in str_val:
            issues.append(str_val)
        elif str_val != str_val.strip():
            issues.append(str_val)
    return issues[:4]


def analyze_column(cursor, table: str, col_info: Dict, row_count: int) -> Dict[str, Any]:
    """Analyze a single column for patterns."""
    col_name = col_info["name"]
    col_type = col_info["type"].upper()

    analysis = {
        "type": col_type,
        "samples": [],
        "distinct_count": 0,
        "is_low_cardinality": False,
        "is_likely_id": False,
        "is_likely_name": False,
        "is_likely_city": False,
        "is_likely_date": False,
        "is_likely_category": False,
        "is_currency": False,
        "is_pre_aggregated": False,
        "pre_aggregated_note": None,
        "is_boolean_flag": False,
        "boolean_note": None,
        "is_timestamp": False,
        "date_format": None,
        "categorical_values": None
    }

    if row_count == 0:
        return analysis

    analysis["samples"] = get_sample_values(cursor, table, col_name, MAX_SAMPLES_PER_COLUMN)
    analysis["distinct_count"] = get_distinct_count(cursor, table, col_name)

    analysis["is_low_cardinality"] = analysis["distinct_count"] < 50
    analysis["is_likely_id"] = col_name.lower().endswith(('_id', 'id', '_code', 'code', '_key'))
    analysis["is_likely_name"] = col_name.lower().endswith(('_name', 'name', 'title', 'description')) or col_name.lower() in ['first', 'last', 'middle']
    analysis["is_likely_city"] = 'city' in col_name.lower()
    analysis["is_likely_date"] = detect_date_column(col_name, col_type)
    analysis["is_timestamp"] = detect_timestamp_column(col_name)
    analysis["is_likely_category"] = col_name.lower() in ['type', 'class', 'category', 'status', 'state', 'essential', 'localization', 'function', 'phenotype', 'gender', 'religion', 'events', 'race', 'cohort', 'role']

    # Pre-aggregated column detection
    is_pre_agg, note = is_pre_aggregated_column(col_name)
    analysis["is_pre_aggregated"] = is_pre_agg
    analysis["pre_aggregated_note"] = note

    # Boolean/flag column detection (NEW)
    is_bool, bool_note = is_boolean_flag_column(col_name, analysis["samples"])
    analysis["is_boolean_flag"] = is_bool
    analysis["boolean_note"] = bool_note

    if col_type in ['TEXT', 'VARCHAR', 'CHAR', '']:
        analysis["is_currency"] = detect_currency_format(analysis["samples"])

        if analysis["is_low_cardinality"] and analysis["distinct_count"] <= MAX_CATEGORICAL_VALUES:
            analysis["categorical_values"] = get_categorical_values(
                cursor, table, col_name, MAX_CATEGORICAL_VALUES
            )

    return analysis


def determine_ownership(table_info: Dict[str, Dict]) -> Dict[str, Dict]:
    """Determine column ownership for each table (from iter2/iter6)."""
    ownership = {}

    for table, info in table_info.items():
        fk_columns = {fk["column"] for fk in info["foreign_keys"]}

        owns = []
        references = []

        for col in info["columns"]:
            col_name = col["name"]
            if col_name in fk_columns:
                references.append(col_name)
            elif not (col["primary_key"] and col_name.lower() == 'id'):
                owns.append(col_name)

        ownership[table] = {
            "owns": owns,
            "references": references
        }

    return ownership


def detect_ambiguous_columns(table_info: Dict[str, Dict], cursor) -> List[Dict]:
    """Detect columns that exist in multiple tables with value comparison (from iter13)."""
    column_tables = defaultdict(list)

    for table, info in table_info.items():
        for col in info["columns"]:
            column_tables[col["name"]].append(table)

    ambiguous = []
    for col_name, tables in column_tables.items():
        if len(tables) > 1:
            table_values = {}
            values_differ = False
            for t in tables[:4]:
                try:
                    quoted_table = get_quoted_name(t)
                    cursor.execute(f"""
                        SELECT DISTINCT `{col_name}` FROM {quoted_table}
                        WHERE `{col_name}` IS NOT NULL
                        ORDER BY `{col_name}` LIMIT 5
                    """)
                    vals = [str(row[0]) for row in cursor.fetchall()]
                    table_values[t] = vals
                except:
                    table_values[t] = []

            all_vals = list(table_values.values())
            if len(all_vals) >= 2 and all_vals[0] and all_vals[1]:
                if set(all_vals[0]) != set(all_vals[1]):
                    values_differ = True

            ambiguous.append({
                "column": col_name,
                "tables": tables,
                "values_differ": values_differ,
                "sample_values": table_values
            })

    ambiguous.sort(key=lambda x: (not x.get('values_differ', False), -len(x['tables'])))
    return ambiguous


def find_similar_tables(tables: List[str], table_info: Dict[str, Dict]) -> List[Tuple[str, str, str]]:
    """Find tables with similar names that might cause confusion."""
    similar_pairs = []

    def normalize(name: str) -> str:
        return re.sub(r'[^a-z]', '', name.lower())

    confusing_prefixes = [
        ('sales', 'purchase'),
        ('order', 'purchaseorder'),
        ('customer', 'vendor'),
        ('current', 'historical'),
        ('current', 'history'),
        ('employee', 'sales'),
        ('product', 'productlist'),
        ('person', 'male'),
        ('person', 'female'),
        ('disabled', 'unemployed'),
        ('enrolled', 'enlist'),
        ('gene', 'classification'),
        ('menu', 'menuitem'),
        ('dish', 'menuitem'),
        ('trip', 'station'),
        ('trip', 'weather'),
        ('institution', 'grads'),
        ('details', 'grads'),
        ('lists', 'lists_users'),
        ('ratings', 'movies'),
        ('user', 'subscriber'),
    ]

    for i, t1 in enumerate(tables):
        t1_lower = t1.lower()
        t1_norm = normalize(t1)

        for t2 in tables[i+1:]:
            t2_lower = t2.lower()
            t2_norm = normalize(t2)

            reason = None

            if len(t1_norm) > 5 and len(t2_norm) > 5:
                for suffix_len in range(min(len(t1_norm), len(t2_norm)), 4, -1):
                    if t1_norm.endswith(t2_norm[-suffix_len:]) or t2_norm.endswith(t1_norm[-suffix_len:]):
                        common_suffix = t1_norm[-suffix_len:] if t1_norm.endswith(t2_norm[-suffix_len:]) else t2_norm[-suffix_len:]
                        if len(common_suffix) >= 5:
                            reason = f"shared suffix: {common_suffix}"
                            break

            for prefix1, prefix2 in confusing_prefixes:
                if (prefix1 in t1_lower and prefix2 in t2_lower) or \
                   (prefix2 in t1_lower and prefix1 in t2_lower):
                    reason = f"similar purpose ({prefix1} vs {prefix2})"
                    break

            if reason:
                similar_pairs.append((t1, t2, reason))

    return similar_pairs[:12]


def find_cross_table_relationships(table_info: Dict[str, Dict], ownership: Dict[str, Dict]) -> List[str]:
    """Identify common cross-table lookup patterns (from iter6)."""
    hints = []

    for table, info in table_info.items():
        for fk in info["foreign_keys"]:
            ref_table = fk["ref_table"]
            ref_owns = ownership.get(ref_table, {}).get("owns", [])
            if any('name' in c.lower() for c in ref_owns):
                hints.append(f"{table} -> {ref_table}: lookup names via {fk['column']}")

    return hints[:10]


def identify_authoritative_tables(table_info: Dict[str, Dict]) -> Dict[str, str]:
    """Identify authoritative source tables for entity counts (from iter9/iter10)."""
    authoritative = {}

    for table, info in table_info.items():
        if info["is_entity"] and not info["is_junction"] and not info["is_history"]:
            name_lower = table.lower()

            entity_mappings = [
                ('employee', 'employee'), ('customer', 'customer'), ('product', 'product'),
                ('patient', 'patient'), ('person', 'person'), ('student', 'student'),
                ('user', 'user'), ('member', 'member'), ('store', 'store'),
                ('supplier', 'supplier'), ('vendor', 'vendor'), ('competitor', 'competitor'),
                ('athlete', 'athlete'), ('player', 'player'), ('team', 'team'),
                ('gene', 'gene'), ('classification', 'classification'),
                ('legislator', 'legislator'), ('actor', 'actor'), ('director', 'director'),
                ('dish', 'dish'), ('menu', 'menu'), ('paper', 'paper'),
                ('station', 'station'), ('trip', 'trip'), ('weather', 'weather'),
                ('institution', 'institution'), ('school', 'school'), ('university', 'university'),
                ('movie', 'movie'), ('film', 'film'), ('list', 'list'), ('rating', 'rating')
            ]

            for pattern, entity_type in entity_mappings:
                if pattern in name_lower and 'history' not in name_lower:
                    if entity_type not in authoritative:
                        authoritative[entity_type] = table

    return authoritative


def detect_duplicate_counting_risks(table_info: Dict[str, Dict]) -> List[str]:
    """Detect table relationships that could cause duplicate counting (from iter10/iter13)."""
    risks = []

    for table, info in table_info.items():
        if info.get("is_entity") and not info.get("is_junction") and not info.get("is_empty"):
            pk = info["primary_keys"][0] if info["primary_keys"] else None
            if not pk:
                continue

            referencing_tables = []
            for other_table, other_info in table_info.items():
                if other_table == table:
                    continue
                for fk in other_info["foreign_keys"]:
                    if fk["ref_table"] == table:
                        referencing_tables.append((other_table, fk["column"]))

            if referencing_tables:
                ref_list = ", ".join([t for t, _ in referencing_tables[:4]])
                if len(referencing_tables) > 4:
                    ref_list += f" +{len(referencing_tables)-4} more"

                entity_count = info["row_count"]
                risks.append(
                    f"{table} ({entity_count:,} entities) -> referenced by [{ref_list}]. "
                    f"For 'how many {table}' questions, query {table} directly or use COUNT(DISTINCT)."
                )

    return risks[:8]


def collect_pre_aggregated_warnings(column_analysis: Dict[str, Dict[str, Dict]]) -> List[str]:
    """Collect warnings about pre-aggregated columns."""
    warnings = []

    for table, analysis in column_analysis.items():
        for col_name, col_info in analysis.items():
            if col_info.get("is_pre_aggregated"):
                note = col_info.get("pre_aggregated_note", "")
                warnings.append(f"{table}.{col_name}: {note}")

    return warnings[:12]


def collect_boolean_flag_warnings(column_analysis: Dict[str, Dict[str, Dict]]) -> List[str]:
    """NEW: Collect warnings about boolean/flag columns."""
    warnings = []

    for table, analysis in column_analysis.items():
        for col_name, col_info in analysis.items():
            if col_info.get("is_boolean_flag"):
                note = col_info.get("boolean_note", "")
                warnings.append(f"{table}.{col_name}: {note}")

    return warnings[:10]


def collect_value_warnings(cursor, table_info: Dict[str, Dict], column_analysis: Dict[str, Dict[str, Dict]]) -> List[str]:
    """Collect value variation warnings for high-value text columns (from iter7/iter10)."""
    warnings = []
    columns_checked = 0

    priority_columns = []
    regular_columns = []

    for table, analysis in column_analysis.items():
        for col_name, col_info in analysis.items():
            if col_info.get("type", "").upper() in ['TEXT', 'VARCHAR', 'CHAR', '']:
                distinct_count = col_info.get("distinct_count", 0)
                if 2 <= distinct_count <= 100:
                    item = (table, col_name, col_info)
                    if col_info.get("is_likely_name") or col_info.get("is_likely_city") or col_info.get("is_likely_category"):
                        priority_columns.append(item)
                    else:
                        regular_columns.append(item)

    for table, col_name, col_info in priority_columns + regular_columns:
        if columns_checked >= MAX_VARIATION_COLUMNS:
            break

        all_values = get_all_values(cursor, table, col_name, MAX_VALUE_ENUMERATION)
        str_values = [str(v) for v in all_values if v is not None]

        columns_checked += 1

        variations = detect_value_variations(str_values)
        for v1, v2, reason in variations:
            warnings.append(f"{table}.{col_name}: '{v1}' vs '{v2}' ({reason})")

        spacing_issues = find_values_with_spacing_issues(str_values)
        for val in spacing_issues[:2]:
            warnings.append(f"{table}.{col_name}: '{val}' has spacing issues")

    return warnings[:15]


def generate_output(
    raw_ddl: List[Tuple[str, str]],
    table_info: Dict[str, Dict],
    column_analysis: Dict[str, Dict[str, Dict]],
    ownership: Dict[str, Dict],
    ambiguous_columns: List[Dict],
    similar_tables: List[Tuple[str, str, str]],
    cross_table_hints: List[str],
    value_warnings: List[str],
    authoritative_tables: Dict[str, str],
    duplicate_risks: List[str],
    pre_aggregated_warnings: List[str],
    boolean_flag_warnings: List[str],
    date_formats: Dict[str, Dict],
    date_join_patterns: List[Dict],
    cursor,
    is_large_db: bool
) -> str:
    """Generate comprehensive analysis output."""

    lines = []
    total_columns_sampled = 0
    total_categorical_sampled = 0

    # ===== CRITICAL GUIDANCE FIRST =====
    lines.append("=" * 70)
    lines.append("CRITICAL: FOLLOW EVIDENCE LITERALLY")
    lines.append("=" * 70)
    lines.append("")
    lines.append("EVIDENCE IS YOUR PRIMARY GUIDE. Even if evidence seems wrong:")
    lines.append("- If evidence says 'user_subscriber = 0', use EXACTLY that condition")
    lines.append("- Do NOT 'fix' evidence by substituting different columns")
    lines.append("- Do NOT add extra JOINs or conditions beyond what evidence specifies")
    lines.append("- Return ONLY the columns asked for in the question - no extras!")
    lines.append("")

    # ===== SECTION 1: DATABASE SCHEMA =====
    lines.append("=" * 70)
    lines.append("SECTION 1: DATABASE SCHEMA")
    lines.append("=" * 70)
    lines.append("")

    # Database summary
    total_tables = len(table_info)
    total_rows = sum(info["row_count"] for info in table_info.values())
    empty_tables = [t for t, info in table_info.items() if info["is_empty"]]
    junction_tables = [t for t, info in table_info.items() if info["is_junction"]]
    history_tables = [t for t, info in table_info.items() if info["is_history"]]
    entity_tables = [t for t, info in table_info.items() if info["is_entity"]]
    quoted_tables = [t for t, info in table_info.items() if info["needs_quoting"]]

    lines.append(f"Tables: {total_tables} | Total Rows: {total_rows:,}")
    if empty_tables:
        lines.append(f"EMPTY TABLES: {', '.join(empty_tables[:10])}")
    if junction_tables:
        lines.append(f"Junction Tables: {', '.join(junction_tables[:10])}")
    if history_tables:
        lines.append(f"History Tables (use for 'oldest/all time' queries): {', '.join(history_tables)}")
    if quoted_tables:
        lines.append(f"TABLES REQUIRING QUOTES: {', '.join(quoted_tables)}")
    lines.append("")

    # BOOLEAN/FLAG COLUMN WARNINGS (NEW - from error analysis)
    if boolean_flag_warnings:
        lines.append("*** BOOLEAN/FLAG COLUMNS (FOLLOW EVIDENCE EXACTLY!) ***")
        lines.append("These columns represent states (subscriber/not, active/not, etc.)")
        lines.append("When evidence mentions these, use the EXACT condition given!")
        lines.append("")
        for warning in boolean_flag_warnings:
            lines.append(f"  * {warning}")
        lines.append("")

    # DATE FORMAT ANALYSIS
    non_iso_dates = []
    timestamp_cols = []
    for table, formats in date_formats.items():
        for col, fmt_info in formats.items():
            if fmt_info.get('detected_format'):
                if not fmt_info.get('is_iso'):
                    non_iso_dates.append((table, col, fmt_info))
                if 'timestamp' in col.lower() or 'utc' in col.lower():
                    timestamp_cols.append((table, col, fmt_info))

    if non_iso_dates or date_join_patterns or timestamp_cols:
        lines.append("*** DATE/TIMESTAMP ANALYSIS ***")
        lines.append("")

        if timestamp_cols:
            lines.append("TIMESTAMP COLUMNS (for year arithmetic, consider SUBSTR vs datetime):")
            for table, col, fmt_info in timestamp_cols[:4]:
                sample = fmt_info.get('sample', 'N/A')
                year_extract = fmt_info.get('year_extraction', 'unknown')
                lines.append(f"  {table}.{col}: Sample '{sample}'")
                lines.append(f"    Year extraction: {year_extract}")
            lines.append("")

        if non_iso_dates:
            lines.append("NON-ISO DATE FORMATS (strftime may NOT work):")
            for table, col, fmt_info in non_iso_dates[:6]:
                sample = fmt_info.get('sample', 'N/A')
                hint = fmt_info.get('parse_hint', '')
                lines.append(f"  {table}.{col}: {fmt_info['detected_format']}")
                lines.append(f"    Sample: '{sample}' | Hint: {hint}")
            lines.append("")

        if date_join_patterns:
            lines.append("DATE JOIN ISSUES (format mismatches):")
            for pattern in date_join_patterns[:4]:
                lines.append(f"  {pattern['table1']}.{pattern['column1']} <-> {pattern['table2']}.{pattern['column2']}")
                lines.append(f"    WARNING: {pattern['warning']}")
            lines.append("")

    # Authoritative Source Tables
    if authoritative_tables:
        lines.append("*** AUTHORITATIVE SOURCE TABLES ***")
        lines.append("For 'how many X' questions, query these tables directly:")
        lines.append("")
        for entity, table in sorted(authoritative_tables.items()):
            row_count = table_info.get(table, {}).get("row_count", 0)
            lines.append(f"  {entity.upper()}: {table} ({row_count:,} rows)")
        lines.append("")

    # Duplicate Counting Risks
    if duplicate_risks:
        lines.append("*** DUPLICATE COUNTING RISKS ***")
        lines.append("These entity tables are referenced by detail tables (1:many).")
        lines.append("When counting entities, use the entity table directly or COUNT(DISTINCT)!")
        lines.append("")
        for risk in duplicate_risks:
            lines.append(f"  * {risk}")
        lines.append("")

    # PRE-AGGREGATED COLUMNS
    if pre_aggregated_warnings:
        lines.append("*** PRE-AGGREGATED COLUMNS (COUNT vs SUM) ***")
        lines.append("These columns already contain counts/totals - check before using COUNT(*)!")
        lines.append("")
        for warning in pre_aggregated_warnings:
            lines.append(f"  * {warning}")
        lines.append("")

    # Similar Table Warnings
    if similar_tables:
        lines.append("*** SIMILAR TABLES - CHOOSE CAREFULLY ***")
        lines.append("")
        for t1, t2, reason in similar_tables[:6]:
            r1 = table_info.get(t1, {}).get("row_count", 0)
            r2 = table_info.get(t2, {}).get("row_count", 0)
            lines.append(f"  {t1} ({r1:,} rows) vs {t2} ({r2:,} rows)")
            lines.append(f"    -> {reason}")
        lines.append("")

    # Table quoting guidance
    if quoted_tables:
        lines.append("** TABLE NAME QUOTING **")
        for t in quoted_tables[:3]:
            lines.append(f'  CORRECT: SELECT * FROM `{t}`')
        lines.append("")

    # Raw DDL with row counts and markers
    for table_name, ddl in raw_ddl:
        info = table_info.get(table_name, {})
        row_count = info.get("row_count", 0)

        markers = []
        if info.get("is_empty"):
            markers.append("EMPTY")
        if info.get("is_junction"):
            markers.append("JUNCTION")
        if info.get("is_history"):
            markers.append("HISTORY")
        if info.get("is_entity"):
            markers.append("ENTITY")
        if info.get("needs_quoting"):
            markers.append("QUOTE")

        marker_str = f" [{', '.join(markers)}]" if markers else ""

        lines.append(f"-- {table_name}: {row_count:,} rows{marker_str}")
        lines.append(ddl + ";")
        lines.append("")

    # ===== SECTION 2: COLUMN DISAMBIGUATION =====
    lines.append("=" * 70)
    lines.append("SECTION 2: COLUMN DISAMBIGUATION (CRITICAL)")
    lines.append("=" * 70)
    lines.append("")

    if ambiguous_columns:
        lines.append("*** COLUMNS IN MULTIPLE TABLES - ALWAYS QUALIFY! ***")
        lines.append("")
        lines.append("WHEN USING THESE COLUMNS, SPECIFY THE TABLE:")
        lines.append("")

        for amb in ambiguous_columns[:15]:
            col = amb['column']
            tables = amb['tables']
            values_differ = amb.get('values_differ', False)
            sample_values = amb.get('sample_values', {})

            if values_differ:
                lines.append(f"  {col}: *** VALUES DIFFER BETWEEN TABLES ***")
            else:
                lines.append(f"  {col}:")

            for t in tables[:5]:
                row_count = table_info.get(t, {}).get("row_count", 0)
                vals = sample_values.get(t, [])
                if vals and values_differ:
                    val_preview = ", ".join(f"'{v}'" for v in vals[:3])
                    lines.append(f"    - {t}.{col} ({row_count:,} rows) e.g., {val_preview}")
                else:
                    lines.append(f"    - {t}.{col} ({row_count:,} rows)")
            if len(tables) > 5:
                lines.append(f"    ... and {len(tables) - 5} more tables")
            lines.append("")
    else:
        lines.append("  No ambiguous columns detected.")
        lines.append("")

    # ===== SECTION 3: QUERY SUPPORT =====
    lines.append("=" * 70)
    lines.append("SECTION 3: QUERY SUPPORT")
    lines.append("=" * 70)
    lines.append("")

    if is_large_db:
        lines.append("NOTE: Large database - showing essential information only")
        lines.append("")

    # Value Variation Warnings
    if value_warnings:
        lines.append("## VALUE VARIATION WARNINGS")
        lines.append("Check exact spelling - variations exist in the data!")
        lines.append("")
        for warning in value_warnings:
            lines.append(f"  * {warning}")
        lines.append("")

    # Cross-Table Lookup Hints
    if cross_table_hints:
        lines.append("## CROSS-TABLE LOOKUPS")
        lines.append("When filtering by names/values from another table, use JOIN:")
        lines.append("")
        for hint in cross_table_hints:
            lines.append(f"  {hint}")
        lines.append("")

    # Column Samples
    if not is_large_db:
        lines.append("## COLUMN SAMPLES")
        lines.append("")

        sorted_tables = sorted(table_info.keys(),
                               key=lambda t: table_info[t]["row_count"],
                               reverse=True)

        for table in sorted_tables:
            if total_columns_sampled >= MAX_SAMPLE_COLUMNS:
                lines.append(f"\n[Samples truncated - showing first {MAX_SAMPLE_COLUMNS} columns]")
                break

            info = table_info[table]
            analysis = column_analysis.get(table, {})

            if info["is_empty"]:
                continue

            quoted_name = get_quoted_name(table)
            header_parts = [f"{quoted_name}" if info["needs_quoting"] else table]
            header_parts.append(f"({info['row_count']:,} rows)")

            if info["is_junction"]:
                header_parts.append("[JUNCTION]")
            if info["is_history"]:
                header_parts.append("[HISTORY]")
            if info["is_entity"]:
                header_parts.append("[ENTITY]")

            lines.append(f"### {' '.join(header_parts)}")

            for col in info["columns"]:
                if total_columns_sampled >= MAX_SAMPLE_COLUMNS:
                    break

                col_name = col["name"]
                col_type = col["type"]
                col_analysis = analysis.get(col_name, {})

                line = f"  - {col_name} ({col_type})"

                if col["primary_key"]:
                    line += " [PK]"

                for fk in info["foreign_keys"]:
                    if fk["column"] == col_name:
                        ref_table = get_quoted_name(fk["ref_table"]) if needs_quoting(fk["ref_table"]) else fk["ref_table"]
                        line += f" -> {ref_table}"
                        break

                # Date format info
                if table in date_formats and col_name in date_formats[table]:
                    fmt_info = date_formats[table][col_name]
                    if fmt_info.get('detected_format'):
                        line += f" [DATE: {fmt_info['detected_format']}"
                        if fmt_info.get('has_time'):
                            line += " + TIME"
                        if not fmt_info.get('strftime_compatible'):
                            line += " - NO strftime!"
                        line += "]"

                if col_analysis.get("is_currency"):
                    line += " [CURRENCY - parse for numeric ops]"

                if col_analysis.get("is_pre_aggregated"):
                    line += " [PRE-AGG - may need SUM]"

                if col_analysis.get("is_boolean_flag"):
                    line += " [FLAG 0/1]"

                samples_shown = False
                if total_categorical_sampled < MAX_CATEGORICAL_COLUMNS:
                    categorical = col_analysis.get("categorical_values")
                    if categorical and len(categorical) <= MAX_CATEGORICAL_VALUES:
                        cat_str = ", ".join(repr(v) for v in categorical)
                        if len(cat_str) <= 120:
                            line += f" VALUES: [{cat_str}]"
                            total_categorical_sampled += 1
                            samples_shown = True

                if not samples_shown:
                    samples = col_analysis.get("samples", [])
                    if samples:
                        sample_str = ", ".join(repr(s) for s in samples[:4])
                        if len(sample_str) <= 90:
                            line += f" e.g., {sample_str}"

                lines.append(line)
                total_columns_sampled += 1

            lines.append("")

    # Column Ownership Map
    lines.append("## COLUMN OWNERSHIP (for correct JOINs)")
    lines.append("")
    lines.append("When you need data from a column, check which table OWNS it:")
    lines.append("")

    ownership_lines = []
    for table in sorted(ownership.keys()):
        own = ownership[table]
        info = table_info.get(table, {})

        if info.get("is_junction"):
            ownership_lines.append(f"  {table}: JUNCTION (FK references only)")
        elif own["owns"]:
            owned_str = ", ".join(own["owns"][:6])
            if len(own["owns"]) > 6:
                owned_str += f" +{len(own['owns'])-6} more"
            ownership_lines.append(f"  {table}: {owned_str}")

    if is_large_db and len(ownership_lines) > 20:
        lines.extend(ownership_lines[:20])
        lines.append(f"  ... and {len(ownership_lines) - 20} more tables")
    else:
        lines.extend(ownership_lines)

    lines.append("")

    # Evidence Reconciliation Hints
    lines.append("## EVIDENCE RECONCILIATION")
    lines.append("")

    name_columns = []
    date_columns = []
    id_columns = []
    category_columns = []
    flag_columns = []

    for table, analysis in column_analysis.items():
        for col_name, col_info in analysis.items():
            full_name = f"{table}.{col_name}"
            if col_info.get("is_likely_name"):
                name_columns.append(full_name)
            if col_info.get("is_likely_date"):
                date_columns.append(full_name)
            if col_info.get("is_likely_id"):
                id_columns.append(full_name)
            if col_info.get("is_likely_category"):
                category_columns.append(full_name)
            if col_info.get("is_boolean_flag"):
                flag_columns.append(full_name)

    if name_columns:
        lines.append(f"'name' could refer to: {', '.join(name_columns[:8])}")
    if date_columns:
        lines.append(f"'date' columns: {', '.join(date_columns[:8])}")
    if category_columns:
        lines.append(f"'category/type' columns: {', '.join(category_columns[:8])}")
    if flag_columns:
        lines.append(f"'flag/boolean' columns: {', '.join(flag_columns[:8])}")
    if id_columns:
        lines.append(f"'id/code' columns: {', '.join(id_columns[:8])}")
    lines.append("")

    return "\n".join(lines)


def main():
    """Main execution function."""

    db_path = "database.sqlite"
    output_dir = "tool_output"
    output_file = os.path.join(output_dir, "schema_analysis.txt")

    os.makedirs(output_dir, exist_ok=True)

    try:
        conn = connect_db(db_path)
        cursor = conn.cursor()

        raw_ddl = get_raw_ddl(cursor)
        num_tables = len(raw_ddl)
        print(f"Found {num_tables} tables")

        is_large_db = num_tables > SKIP_SAMPLES_THRESHOLD
        if is_large_db:
            print(f"Large database detected (>{SKIP_SAMPLES_THRESHOLD} tables) - using compact output")

        tables = get_tables(cursor)

        # Gather table information
        table_info = {}
        for table in tables:
            table_info[table] = get_table_info(cursor, table)

        # Analyze columns
        column_analysis = {}
        if not is_large_db:
            for table in tables:
                info = table_info[table]
                column_analysis[table] = {}
                for col in info["columns"]:
                    column_analysis[table][col["name"]] = analyze_column(
                        cursor, table, col, info["row_count"]
                    )

        # Detect date formats for each date-like column
        date_formats = {}
        for table in tables:
            info = table_info[table]
            table_dates = {}
            for col in info["columns"]:
                if detect_date_column(col["name"], col["type"]):
                    fmt_info = detect_date_format_from_values(cursor, table, col["name"])
                    if fmt_info.get('detected_format'):
                        table_dates[col["name"]] = fmt_info
            if table_dates:
                date_formats[table] = table_dates

        # Find date JOIN patterns
        date_join_patterns = find_date_join_patterns(table_info, date_formats)

        # Standard analysis
        ownership = determine_ownership(table_info)
        ambiguous_columns = detect_ambiguous_columns(table_info, cursor)
        similar_tables = find_similar_tables(tables, table_info)
        cross_table_hints = find_cross_table_relationships(table_info, ownership)

        value_warnings = []
        if not is_large_db:
            value_warnings = collect_value_warnings(cursor, table_info, column_analysis)

        authoritative_tables = identify_authoritative_tables(table_info)
        duplicate_risks = detect_duplicate_counting_risks(table_info)

        # Pre-aggregated and boolean column warnings
        pre_aggregated_warnings = collect_pre_aggregated_warnings(column_analysis)
        boolean_flag_warnings = collect_boolean_flag_warnings(column_analysis)

        # Generate output
        output = generate_output(
            raw_ddl, table_info, column_analysis, ownership,
            ambiguous_columns, similar_tables, cross_table_hints,
            value_warnings, authoritative_tables, duplicate_risks,
            pre_aggregated_warnings, boolean_flag_warnings,
            date_formats, date_join_patterns,
            cursor, is_large_db
        )

        if len(output) > MAX_OUTPUT_CHARS:
            print(f"Warning: Output too large ({len(output)} chars), truncating to {MAX_OUTPUT_CHARS}")
            output = output[:MAX_OUTPUT_CHARS]
            output += "\n\n[OUTPUT TRUNCATED - Large database]"

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output)

        # Summary
        total_rows = sum(info["row_count"] for info in table_info.values())
        junction_count = sum(1 for info in table_info.values() if info["is_junction"])
        history_count = sum(1 for info in table_info.values() if info["is_history"])
        entity_count = sum(1 for info in table_info.values() if info["is_entity"])
        date_col_count = sum(len(cols) for cols in date_formats.values())
        pre_agg_count = len(pre_aggregated_warnings)
        bool_flag_count = len(boolean_flag_warnings)

        print(f"Analysis complete!")
        print(f"  - Tables: {len(tables)}")
        print(f"  - Total rows: {total_rows:,}")
        print(f"  - Entity tables: {entity_count}")
        print(f"  - Junction tables: {junction_count}")
        print(f"  - History tables: {history_count}")
        print(f"  - Date columns analyzed: {date_col_count}")
        print(f"  - Date JOIN patterns: {len(date_join_patterns)}")
        print(f"  - Pre-aggregated columns: {pre_agg_count}")
        print(f"  - Boolean/flag columns: {bool_flag_count}")
        print(f"  - Similar table pairs: {len(similar_tables)}")
        print(f"  - Ambiguous columns: {len(ambiguous_columns)}")
        print(f"  - Value variation warnings: {len(value_warnings)}")
        print(f"  - Authoritative sources: {len(authoritative_tables)}")
        print(f"  - Duplicate counting risks: {len(duplicate_risks)}")
        print(f"  - Cross-table hints: {len(cross_table_hints)}")
        print(f"  - Output size: {len(output):,} chars")
        print(f"  - Output: {output_file}")

        conn.close()
        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        with open(output_file, 'w') as f:
            f.write(f"ANALYSIS ERROR: {e}\n")
            f.write("Please check that database.sqlite exists and is readable.\n")
        return 1


if __name__ == "__main__":
    exit(main())
