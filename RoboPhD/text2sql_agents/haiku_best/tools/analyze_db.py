#!/usr/bin/env python3
"""
Mega Cross-Pollinated Database Analyzer (Iteration 15)

Combines ALL best techniques from top-performing agents:

From iter10_ultimate_cross_pollinated_analyzer (ELO 1597):
  - Person/entity table detection for "Who" questions
  - Date format detection with filtering guidance
  - Enhanced COUNT(DISTINCT) guidance

From iter11_evolved_3095 (ELO 1569):
  - Person attribute keyword mappings (marital, gender, race)
  - Stronger NO EXTRA COLUMNS emphasis
  - Combined vs per-entity aggregation guidance

From iter6_unified_cross_pollinated_analyzer (ELO 1526):
  - Domain-specific table detection (SC, Post, WC suffixes)
  - Implicit FK detection via naming patterns
  - Table role classification (ENTITY, LOOKUP, JUNCTION, TRANSACTION)
  - Unicode character detection
  - Similar table name disambiguation

From iter9_comprehensive_cross_pollinated_analyzer (ELO 1524):
  - Name variant detection (typo/spelling warnings)
  - Aggregation semantics guidance
  - Time-series pattern detection
  - Academic database semantic mappings
  - State/country abbreviation detection

From iter3_enhanced_cross_pollinated_precision_analyzer (ELO 1508):
  - Enhanced selectivity scoring with cardinality classification
  - Multi-word case pitfall detection with common_mistakes
  - Adaptive output sizing for large databases

NEW for Iteration 15:
  - Rating/quality column detection and interpretation guidance
  - Multi-part question guidance for complex queries
  - Enhanced "most" vs "highest" disambiguation
  - Movie/media database specific patterns

Outputs comprehensive analysis to tool_output/schema_analysis.txt
"""

import sqlite3
import os
import sys
import re
from collections import defaultdict


# Domain-specific table suffixes and their meanings
DOMAIN_SUFFIXES = {
    'SC': 'Stanley Cup',
    'Post': 'Playoffs/Post-season',
    'WC': 'World Cup',
    'Regular': 'Regular season',
    'Allstar': 'All-Star',
    'AllStar': 'All-Star',
    'Finals': 'Finals/Championship',
    'Playoff': 'Playoff',
    'Season': 'Season-specific',
}


def get_schema_ddl(cursor):
    """Extract CREATE TABLE statements."""
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL ORDER BY name")
    return [row[0] for row in cursor.fetchall()]


def get_table_row_count(cursor, table):
    """Get row count for a table."""
    try:
        cursor.execute(f'SELECT COUNT(*) FROM "{table}"')
        return cursor.fetchone()[0]
    except:
        return 0


def get_foreign_keys(cursor, table):
    """Get foreign key relationships for a table."""
    try:
        cursor.execute(f'PRAGMA foreign_key_list("{table}")')
        fks = []
        for row in cursor.fetchall():
            fks.append({
                'from_column': row[3],
                'to_table': row[2],
                'to_column': row[4]
            })
        return fks
    except:
        return []


def detect_domain_specific_tables(tables):
    """Detect tables with domain-specific suffixes."""
    domain_tables = []

    for table in tables:
        for suffix, meaning in DOMAIN_SUFFIXES.items():
            if table.endswith(suffix):
                base_name = table[:-len(suffix)]
                domain_tables.append({
                    'table': table,
                    'suffix': suffix,
                    'meaning': meaning,
                    'base_table': base_name if base_name in tables else None,
                    'keywords': [meaning.lower(), suffix.lower()]
                })
                break
            if table.endswith('_' + suffix):
                base_name = table[:-(len(suffix) + 1)]
                domain_tables.append({
                    'table': table,
                    'suffix': suffix,
                    'meaning': meaning,
                    'base_table': base_name if base_name in tables else None,
                    'keywords': [meaning.lower(), suffix.lower()]
                })
                break

    return domain_tables


def detect_person_entity_tables(cursor, tables, columns_by_table):
    """
    Detect tables that represent people/entities with name columns.
    Helps with "Who" questions - should return names, not IDs.
    """
    person_tables = []

    name_patterns = ['first', 'last', 'firstname', 'lastname', 'first_name', 'last_name',
                     'name', 'fullname', 'full_name', 'patient', 'author', 'customer',
                     'employee', 'user', 'person', 'owner', 'actor', 'director']

    for table in tables:
        columns = columns_by_table.get(table, [])
        col_names_lower = [c[1].lower() for c in columns]

        has_first = any(p in col_names_lower for p in ['first', 'firstname', 'first_name'])
        has_last = any(p in col_names_lower for p in ['last', 'lastname', 'last_name'])
        has_name = 'name' in col_names_lower or 'fullname' in col_names_lower or 'full_name' in col_names_lower
        has_id = any('id' in c.lower() for c in col_names_lower)

        name_columns = []
        for col in columns:
            col_lower = col[1].lower()
            if any(p in col_lower for p in name_patterns):
                name_columns.append(col[1])

        if (has_first and has_last) or has_name or len(name_columns) >= 2:
            table_lower = table.lower()
            is_person_table = any(p in table_lower for p in ['patient', 'user', 'customer', 'employee',
                                                              'author', 'person', 'member', 'owner',
                                                              'staff', 'actor', 'director'])

            person_tables.append({
                'table': table,
                'name_columns': name_columns,
                'has_first_last': has_first and has_last,
                'has_id': has_id,
                'is_person_table': is_person_table
            })

    return person_tables


def detect_person_attributes(cursor, table, columns):
    """
    Detect person attribute columns that map to keywords.
    E.g., marital = 'S' means 'single', gender = 'M' means 'male'
    """
    attributes = []

    attr_keywords = {
        'marital': {
            'mappings': {'single': 'S', 'married': 'M', 'divorced': 'D', 'widowed': 'W'},
            'description': 'Marital status codes'
        },
        'gender': {
            'mappings': {'male': 'M', 'female': 'F', 'men': 'M', 'women': 'F', 'man': 'M', 'woman': 'F'},
            'description': 'Gender codes'
        },
        'sex': {
            'mappings': {'male': 'M', 'female': 'F'},
            'description': 'Sex codes'
        },
        'race': {
            'mappings': {},
            'description': 'Race/ethnicity codes'
        },
        'ethnicity': {
            'mappings': {},
            'description': 'Ethnicity codes'
        }
    }

    for col in columns:
        col_name = col[1]
        col_lower = col_name.lower()

        for attr_type, attr_info in attr_keywords.items():
            if attr_type in col_lower:
                try:
                    cursor.execute(f'SELECT DISTINCT "{col_name}" FROM "{table}" WHERE "{col_name}" IS NOT NULL LIMIT 30')
                    values = [str(row[0]) for row in cursor.fetchall() if row[0] is not None]

                    reverse_map = {}
                    for keyword, code in attr_info['mappings'].items():
                        if code in values:
                            reverse_map[code] = keyword

                    if attr_type in ('race', 'ethnicity') and values:
                        for v in values:
                            if len(v) <= 3:
                                reverse_map[v] = f"code for {v}"

                    attributes.append({
                        'table': table,
                        'column': col_name,
                        'type': attr_type,
                        'description': attr_info['description'],
                        'values': values[:15],
                        'keyword_mappings': attr_info['mappings'],
                        'reverse_map': reverse_map
                    })
                except:
                    pass

    return attributes


def detect_rating_quality_columns(cursor, table, columns):
    """
    NEW: Detect rating/quality/score columns with ordinal values.
    Critical for movielens and similar databases.
    """
    rating_columns = []

    rating_patterns = ['rating', 'quality', 'score', 'rank', 'star', 'grade', 'level']

    for col in columns:
        col_name = col[1]
        col_lower = col_name.lower()
        col_type = (col[2] or '').upper()

        # Check if column name suggests rating/quality
        is_rating_col = any(p in col_lower for p in rating_patterns)

        if not is_rating_col:
            continue

        try:
            cursor.execute(f'SELECT DISTINCT "{col_name}" FROM "{table}" WHERE "{col_name}" IS NOT NULL ORDER BY "{col_name}" LIMIT 20')
            values = [row[0] for row in cursor.fetchall()]

            if not values:
                continue

            # Check if values are ordinal (small integers or short strings)
            is_ordinal = False
            min_val = None
            max_val = None

            if all(isinstance(v, (int, float)) or (isinstance(v, str) and v.isdigit()) for v in values):
                numeric_values = [int(v) if isinstance(v, str) else v for v in values]
                min_val = min(numeric_values)
                max_val = max(numeric_values)
                # Ordinal if range is small (typically 0-5, 1-10, etc.)
                is_ordinal = (max_val - min_val) <= 10 and min_val >= 0

            if is_ordinal or len(values) <= 10:
                # Determine interpretation
                interpretation = []
                if 'quality' in col_lower:
                    if max_val is not None:
                        interpretation.append(f"{col_name}={max_val} = BEST quality")
                        interpretation.append(f"{col_name}={min_val} = WORST quality")
                elif 'rating' in col_lower:
                    if max_val is not None:
                        interpretation.append(f"{col_name}='{max_val}' = HIGHEST rating")
                        interpretation.append(f"{col_name}='{min_val}' = LOWEST rating")
                        interpretation.append("'most rated' = COUNT(*) of ratings (not MAX rating!)")
                        interpretation.append("'highest rated' = ORDER BY AVG(rating) DESC")

                rating_columns.append({
                    'table': table,
                    'column': col_name,
                    'values': values,
                    'is_ordinal': is_ordinal,
                    'min_val': min_val,
                    'max_val': max_val,
                    'interpretation': interpretation
                })
        except:
            pass

    return rating_columns


def analyze_column_selectivity_enhanced(cursor, table, columns, total_rows):
    """Enhanced column selectivity analysis with priority classification."""
    result = {}

    for col in columns:
        col_name = col[1]
        col_type = col[2] or ''
        is_pk = col[5] == 1

        col_info = {
            'type': col_type,
            'is_pk': is_pk,
            'cardinality': 'UNKNOWN',
            'distinct_count': 0,
            'selectivity_score': 0.0,
            'priority': 'NORMAL',
            'null_count': 0,
            'null_pct': 0.0
        }

        if total_rows > 0:
            try:
                cursor.execute(f'SELECT COUNT(DISTINCT "{col_name}") FROM "{table}"')
                distinct = cursor.fetchone()[0]
                col_info['distinct_count'] = distinct

                cursor.execute(f'SELECT COUNT(*) FROM "{table}" WHERE "{col_name}" IS NULL')
                null_count = cursor.fetchone()[0]
                col_info['null_count'] = null_count
                col_info['null_pct'] = round(null_count / total_rows * 100, 1)

                selectivity = distinct / total_rows if total_rows > 0 else 0
                col_info['selectivity_score'] = round(selectivity, 3)

                if distinct == 1:
                    col_info['cardinality'] = 'CONSTANT'
                    col_info['priority'] = 'LOW'
                elif distinct <= 10:
                    col_info['cardinality'] = 'VERY_LOW'
                    col_info['priority'] = 'MEDIUM'
                elif distinct <= 50:
                    col_info['cardinality'] = 'LOW'
                    col_info['priority'] = 'MEDIUM'
                elif selectivity < 0.1:
                    col_info['cardinality'] = 'MEDIUM'
                    col_info['priority'] = 'NORMAL'
                elif selectivity < 0.5:
                    col_info['cardinality'] = 'MEDIUM_HIGH'
                    col_info['priority'] = 'NORMAL'
                elif selectivity < 0.9:
                    col_info['cardinality'] = 'HIGH'
                    col_info['priority'] = 'NORMAL'
                else:
                    col_info['cardinality'] = 'VERY_HIGH'
                    col_info['priority'] = 'HIGH'

                if is_pk or 'id' in col_name.lower():
                    col_info['priority'] = 'HIGH'
                if 'name' in col_name.lower() or 'title' in col_name.lower():
                    col_info['priority'] = 'HIGH'
                if any(t in col_type.lower() for t in ['date', 'time', 'year']):
                    if col_info['priority'] != 'HIGH':
                        col_info['priority'] = 'MEDIUM'
            except:
                pass

        result[col_name] = col_info

    return result


def detect_implicit_fks(cursor, tables, columns_by_table):
    """Detect implicit foreign key relationships via naming conventions."""
    implicit_fks = []
    table_names_lower = {t.lower(): t for t in tables}

    for table in tables:
        for col in columns_by_table.get(table, []):
            col_name = col[1].lower()

            if col_name.endswith('_id'):
                base_name = col_name[:-3]
                if base_name in table_names_lower:
                    implicit_fks.append({
                        'from_table': table,
                        'from_column': col[1],
                        'to_table': table_names_lower[base_name],
                        'confidence': 'HIGH',
                        'reason': 'Column name ends with _id matching table name'
                    })
                elif base_name + 's' in table_names_lower:
                    implicit_fks.append({
                        'from_table': table,
                        'from_column': col[1],
                        'to_table': table_names_lower[base_name + 's'],
                        'confidence': 'MEDIUM',
                        'reason': 'Column name matches plural table name'
                    })

            for t in tables:
                t_lower = t.lower()
                if col_name == t_lower + 'id' or col_name == t_lower + '_id':
                    if t != table:
                        implicit_fks.append({
                            'from_table': table,
                            'from_column': col[1],
                            'to_table': t,
                            'confidence': 'HIGH',
                            'reason': 'Column name matches table name + id pattern'
                        })

    return implicit_fks


def classify_table_role(cursor, table, columns, fk_count, referenced_count, row_count):
    """Classify table role based on structure."""
    col_count = len(columns)
    pk_cols = [c for c in columns if c[5] == 1]

    if fk_count >= 2 and col_count <= fk_count + 2:
        return 'JUNCTION'
    if row_count <= 50 and referenced_count > 0 and col_count <= 3:
        return 'LOOKUP'
    if len(pk_cols) > 0 and referenced_count > 0:
        return 'ENTITY'
    if fk_count >= 2:
        return 'TRANSACTION'
    return 'STANDARD'


def detect_unicode_issues(cursor, table, columns):
    """Detect non-ASCII characters in string columns."""
    unicode_warnings = []

    for col in columns:
        col_name = col[1]
        col_type = (col[2] or '').upper()

        if 'TEXT' not in col_type and 'CHAR' not in col_type and 'VARCHAR' not in col_type:
            continue

        try:
            cursor.execute(f'SELECT DISTINCT "{col_name}" FROM "{table}" WHERE "{col_name}" IS NOT NULL LIMIT 200')
            values = [row[0] for row in cursor.fetchall() if row[0] is not None]

            for v in values:
                if isinstance(v, str):
                    non_ascii = [c for c in v if ord(c) > 127]
                    if non_ascii:
                        special_chars = {
                            '\u2019': "curly apostrophe (') - use regular apostrophe (')",
                            '\u2018': "curly single quote (') - use regular apostrophe",
                            '\u201c': 'curly double quote (") - use straight quote',
                            '\u201d': 'curly double quote (") - use straight quote',
                            '\u2013': 'en-dash (-) - use regular hyphen',
                            '\u2014': 'em-dash (--) - use regular hyphen',
                            '\u00a0': 'non-breaking space - use regular space',
                        }
                        for char in set(non_ascii):
                            desc = special_chars.get(char, f'Unicode U+{ord(char):04X}')
                            unicode_warnings.append({
                                'table': table,
                                'column': col_name,
                                'value': v[:50],
                                'character': char,
                                'description': desc
                            })
        except:
            pass

    return unicode_warnings


def analyze_case_patterns_enhanced(cursor, table, columns):
    """Enhanced case pattern detection with multi-word pitfall detection."""
    patterns = {
        'lowercase_only': [],
        'mixed_case': [],
        'case_warnings': []
    }

    for col in columns:
        col_name = col[1]
        col_type = (col[2] or '').upper()

        if 'TEXT' not in col_type and 'CHAR' not in col_type and 'VARCHAR' not in col_type:
            continue

        try:
            cursor.execute(f'SELECT DISTINCT "{col_name}" FROM "{table}" WHERE "{col_name}" IS NOT NULL LIMIT 500')
            values = [str(row[0]) for row in cursor.fetchall() if row[0] is not None]

            if not values:
                continue

            all_lowercase = all(v.islower() or not any(c.isalpha() for c in v) for v in values)
            has_upper = any(any(c.isupper() for c in v) for v in values)
            has_alpha = any(any(c.isalpha() for c in v) for v in values)

            if all_lowercase and has_alpha:
                patterns['lowercase_only'].append({
                    'column': col_name,
                    'samples': values[:10]
                })

                for v in values[:40]:
                    if ' ' in v and len(v.split()) >= 2:
                        title_case = v.title()
                        single_word = ''.join(v.split())
                        if title_case != v:
                            patterns['case_warnings'].append({
                                'column': col_name,
                                'database_value': v,
                                'common_mistakes': [
                                    {'wrong': title_case, 'reason': 'Wrong case - DB uses lowercase'},
                                    {'wrong': single_word, 'reason': 'Missing space'}
                                ]
                            })
                    elif len(v) > 3 and any(c.isalpha() for c in v):
                        title_case = v.title()
                        upper_case = v.upper()
                        if title_case != v:
                            patterns['case_warnings'].append({
                                'column': col_name,
                                'database_value': v,
                                'common_mistakes': [
                                    {'wrong': title_case, 'reason': 'Wrong case - DB uses lowercase'},
                                    {'wrong': upper_case, 'reason': 'Wrong case - DB uses lowercase'}
                                ]
                            })
            elif has_upper:
                patterns['mixed_case'].append({
                    'column': col_name,
                    'samples': values[:10]
                })
        except:
            pass

    return patterns


def detect_name_variants(cursor, table, columns):
    """Refined name variant detection (conservative to reduce false positives)."""
    name_warnings = []

    for col in columns:
        col_name = col[1].lower()
        col_type = (col[2] or '').upper()

        is_name_col = any(n in col_name for n in ['name', 'first', 'last', 'title', 'author'])
        is_text = 'TEXT' in col_type or 'CHAR' in col_type or 'VARCHAR' in col_type

        if not (is_name_col and is_text):
            continue

        try:
            cursor.execute(f'SELECT DISTINCT "{col[1]}" FROM "{table}" WHERE "{col[1]}" IS NOT NULL LIMIT 300')
            values = [str(row[0]) for row in cursor.fetchall() if row[0] is not None]

            for i, v1 in enumerate(values[:80]):
                for v2 in values[i+1:80]:
                    if len(v1) > 3 and len(v2) > 3:
                        if abs(len(v1) - len(v2)) <= 1:
                            diff_count = sum(1 for a, b in zip(v1.lower(), v2.lower()) if a != b)
                            len_diff = abs(len(v1) - len(v2))
                            if diff_count + len_diff == 1:
                                name_warnings.append({
                                    'table': table,
                                    'column': col[1],
                                    'value1': v1,
                                    'value2': v2,
                                    'warning': f"Similar names: '{v1}' and '{v2}' - verify spelling"
                                })
        except:
            pass

    return name_warnings[:15]


def detect_time_series_patterns(cursor, table, columns, row_count):
    """Detect time-series data patterns for AVG() guidance."""
    time_series_info = []

    date_cols = [c for c in columns if any(d in c[1].lower() for d in ['date', 'time', 'year', 'month', 'day'])]
    value_cols = [c for c in columns if any(v in c[1].lower() for v in ['price', 'amount', 'value', 'score', 'count', 'rate', 'cost'])]

    if date_cols and value_cols:
        for date_col in date_cols:
            try:
                cursor.execute(f'''
                    SELECT COUNT(*) as row_count, COUNT(DISTINCT "{date_col[1]}") as date_count
                    FROM "{table}"
                    WHERE "{date_col[1]}" IS NOT NULL
                ''')
                result = cursor.fetchone()
                if result and result[0] > result[1]:
                    time_series_info.append({
                        'table': table,
                        'date_column': date_col[1],
                        'value_columns': [c[1] for c in value_cols],
                        'has_multiple_per_date': True,
                        'hint': f"Multiple rows per date - 'the {value_cols[0][1]} in [year]' may need AVG()"
                    })
            except:
                pass

    return time_series_info


def detect_date_formats(cursor, table, columns):
    """Detect date column formats and provide filtering guidance."""
    date_formats = []

    date_keywords = ['date', 'time', 'day', 'month', 'year', 'created', 'updated', 'modified']

    for col in columns:
        col_name = col[1]
        col_type = (col[2] or '').upper()

        is_date_col = any(kw in col_name.lower() for kw in date_keywords)
        is_text_type = 'TEXT' in col_type or 'CHAR' in col_type or 'VARCHAR' in col_type or col_type == ''

        if not is_date_col:
            continue

        try:
            cursor.execute(f'SELECT DISTINCT "{col_name}" FROM "{table}" WHERE "{col_name}" IS NOT NULL LIMIT 10')
            values = [str(row[0]) for row in cursor.fetchall() if row[0] is not None]

            if not values:
                continue

            sample = values[0] if values else ''
            format_hint = None
            filter_example = None

            if '/' in sample:
                parts = sample.split('/')
                if len(parts) == 3:
                    if len(parts[2]) == 2:
                        format_hint = "M/D/YY format"
                        filter_example = "Year 2018: LIKE '%/%/18' | Month Jan 2018: LIKE '1/%/18'"
                    elif len(parts[2]) == 4:
                        format_hint = "M/D/YYYY format"
                        filter_example = "Year 2018: LIKE '%/%/2018'"
            elif '-' in sample and len(sample) >= 10 and sample[:4].isdigit():
                format_hint = "YYYY-MM-DD format (ISO)"
                filter_example = "Year 2018: STRFTIME('%Y', col) = '2018'"
            elif '-' in sample:
                format_hint = "Non-ISO date format - check sample values"

            if format_hint or is_text_type:
                date_formats.append({
                    'table': table,
                    'column': col_name,
                    'samples': values[:5],
                    'format_hint': format_hint,
                    'filter_example': filter_example,
                    'is_text': is_text_type
                })
        except:
            pass

    return date_formats


def detect_aggregation_patterns(cursor, table, columns, row_count):
    """Detect patterns that may require special aggregation handling."""
    agg_hints = []

    id_cols = [c for c in columns if 'id' in c[1].lower() and c[5] == 1]
    name_cols = [c for c in columns if any(n in c[1].lower() for n in ['name', 'title', 'address'])]

    if id_cols and name_cols:
        for id_col in id_cols:
            for name_col in name_cols:
                try:
                    cursor.execute(f'''
                        SELECT "{name_col[1]}", COUNT(*) as cnt
                        FROM "{table}"
                        WHERE "{name_col[1]}" IS NOT NULL
                        GROUP BY "{name_col[1]}"
                        HAVING COUNT(*) > 1
                        LIMIT 5
                    ''')
                    duplicates = cursor.fetchall()
                    if duplicates:
                        agg_hints.append({
                            'table': table,
                            'id_column': id_col[1],
                            'name_column': name_col[1],
                            'has_duplicate_names': True,
                            'examples': [d[0] for d in duplicates[:3]],
                            'hint': f"Multiple rows share same {name_col[1]} - for 'which {name_col[1]} has most X', GROUP BY {id_col[1]} then get {name_col[1]}"
                        })
                except:
                    pass

    return agg_hints


def detect_academic_database(tables, columns_by_table):
    """Detect if this is an academic paper database."""
    table_names_lower = [t.lower() for t in tables]

    is_academic = ('paper' in table_names_lower or 'papers' in table_names_lower) and \
                  ('author' in table_names_lower or 'authors' in table_names_lower or
                   'paperauthor' in table_names_lower)

    if not is_academic:
        return None

    paper_table = None
    author_table = None
    paper_author_table = None
    conference_table = None
    journal_table = None

    for t in tables:
        t_lower = t.lower()
        if t_lower == 'paper' or t_lower == 'papers':
            paper_table = t
        elif t_lower == 'author' or t_lower == 'authors':
            author_table = t
        elif t_lower == 'paperauthor' or t_lower == 'paper_author':
            paper_author_table = t
        elif t_lower == 'conference' or t_lower == 'conferences':
            conference_table = t
        elif t_lower == 'journal' or t_lower == 'journals':
            journal_table = t

    paper_cols = columns_by_table.get(paper_table, []) if paper_table else []
    paper_col_names = [c[1].lower() for c in paper_cols]

    has_conference_id = 'conferenceid' in paper_col_names
    has_journal_id = 'journalid' in paper_col_names

    paper_author_cols = columns_by_table.get(paper_author_table, []) if paper_author_table else []
    paper_author_has_name = any(c[1].lower() == 'name' for c in paper_author_cols)

    return {
        'is_academic': True,
        'paper_table': paper_table,
        'author_table': author_table,
        'paper_author_table': paper_author_table,
        'conference_table': conference_table,
        'journal_table': journal_table,
        'has_conference_id': has_conference_id,
        'has_journal_id': has_journal_id,
        'paper_author_has_name': paper_author_has_name
    }


def detect_movie_database(tables, columns_by_table):
    """NEW: Detect if this is a movie/media database (like movielens)."""
    table_names_lower = [t.lower() for t in tables]

    # Check for movie-related tables
    has_movies = any('movie' in t for t in table_names_lower)
    has_actors = any('actor' in t for t in table_names_lower)
    has_directors = any('director' in t for t in table_names_lower)
    has_ratings = any('rating' in t or 'u2base' in t for t in table_names_lower)

    if not (has_movies and (has_actors or has_directors or has_ratings)):
        return None

    movie_info = {
        'is_movie_db': True,
        'tables': {
            'movies': None,
            'actors': None,
            'directors': None,
            'ratings': None,
            'movie_actors': None,
            'movie_directors': None
        },
        'quality_columns': [],
        'rating_columns': []
    }

    for t in tables:
        t_lower = t.lower()
        if t_lower in ('movie', 'movies'):
            movie_info['tables']['movies'] = t
        elif t_lower in ('actor', 'actors'):
            movie_info['tables']['actors'] = t
        elif t_lower in ('director', 'directors'):
            movie_info['tables']['directors'] = t
        elif 'u2base' in t_lower or t_lower in ('rating', 'ratings'):
            movie_info['tables']['ratings'] = t
        elif 'movies2actor' in t_lower or 'movie_actor' in t_lower:
            movie_info['tables']['movie_actors'] = t
        elif 'movies2director' in t_lower or 'movie_director' in t_lower:
            movie_info['tables']['movie_directors'] = t

    # Check for quality columns
    for t in tables:
        cols = columns_by_table.get(t, [])
        for col in cols:
            col_lower = col[1].lower()
            if 'quality' in col_lower:
                movie_info['quality_columns'].append(f"{t}.{col[1]}")
            elif 'rating' in col_lower:
                movie_info['rating_columns'].append(f"{t}.{col[1]}")

    return movie_info


def get_value_samples(cursor, table, columns, max_samples=15):
    """Get sample values for each column with enhanced metadata."""
    samples = {}

    for col in columns:
        col_name = col[1]
        col_type = (col[2] or '').upper()

        try:
            cursor.execute(f'SELECT DISTINCT "{col_name}" FROM "{table}" WHERE "{col_name}" IS NOT NULL LIMIT {max_samples}')
            values = [row[0] for row in cursor.fetchall()]

            cursor.execute(f'SELECT COUNT(DISTINCT "{col_name}") FROM "{table}"')
            total_distinct = cursor.fetchone()[0]

            is_text = 'TEXT' in col_type or 'CHAR' in col_type or 'VARCHAR' in col_type

            is_location = any(kw in col_name.lower() for kw in ['state', 'country', 'city', 'region', 'province'])
            has_abbreviations = all(isinstance(v, str) and len(v) <= 3 and v.isupper() for v in values[:10] if v)

            samples[col_name] = {
                'values': values,
                'total_distinct': total_distinct,
                'is_complete': total_distinct <= max_samples,
                'is_text': is_text,
                'is_location': is_location,
                'has_abbreviations': has_abbreviations and is_location
            }
        except:
            samples[col_name] = {'values': [], 'total_distinct': 0, 'is_complete': True, 'is_text': False}

    return samples


def identify_isolated_tables(cursor, tables):
    """Find tables with no foreign key relationships."""
    isolated = []
    partially_isolated = []

    outgoing_fks = {}
    incoming_fks = defaultdict(list)

    for table in tables:
        fks = []
        try:
            cursor.execute(f'PRAGMA foreign_key_list("{table}")')
            for row in cursor.fetchall():
                fks.append(row[2])
                incoming_fks[row[2]].append(table)
        except:
            pass
        outgoing_fks[table] = fks

    for table in tables:
        has_outgoing = len(outgoing_fks.get(table, [])) > 0
        has_incoming = len(incoming_fks.get(table, [])) > 0

        if not has_outgoing and not has_incoming:
            isolated.append(table)
        elif not has_outgoing and has_incoming:
            partially_isolated.append({
                'table': table,
                'referenced_by': incoming_fks[table]
            })

    return isolated, partially_isolated


def find_similar_table_names(tables):
    """Find tables with similar names."""
    similar_groups = []

    def normalize(name):
        return re.sub(r'[_\-\s]', '', name.lower())

    prefix_groups = defaultdict(list)
    for table in tables:
        normalized = normalize(table)
        if len(normalized) >= 4:
            prefix = normalized[:4]
            prefix_groups[prefix].append(table)

    for prefix, group in prefix_groups.items():
        if len(group) > 1:
            similar_groups.append(group)

    return similar_groups


def get_columns_needing_quotes(cursor, tables):
    """Identify columns that need backtick quoting."""
    needs_quotes = {}
    reserved = ['group', 'order', 'table', 'select', 'where', 'from', 'index', 'key',
                'primary', 'join', 'left', 'right', 'inner', 'outer', 'on', 'as',
                'and', 'or', 'not', 'in', 'like', 'between', 'is', 'null', 'true',
                'false', 'case', 'when', 'then', 'else', 'end', 'limit', 'offset',
                'union', 'except', 'intersect', 'all', 'distinct', 'having', 'by',
                'asc', 'desc', 'create', 'drop', 'alter', 'insert', 'update', 'delete',
                '+/-', 'check', 'default', 'values']

    for table in tables:
        try:
            cursor.execute(f'PRAGMA table_info("{table}")')
            columns = cursor.fetchall()

            problematic = []
            for col in columns:
                col_name = col[1]
                if ' ' in col_name or '-' in col_name or '.' in col_name or '+' in col_name:
                    problematic.append(col_name)
                if col_name.lower() in reserved:
                    problematic.append(col_name)

            if problematic:
                needs_quotes[table] = list(set(problematic))
        except:
            pass

    return needs_quotes


def build_join_paths(cursor, tables):
    """Document foreign key relationships and join recommendations."""
    paths = {}

    for table in tables:
        fks = []
        try:
            cursor.execute(f'PRAGMA foreign_key_list("{table}")')
            for row in cursor.fetchall():
                from_table = f'"{table}"' if ' ' in table else table
                to_table = f'"{row[2]}"' if ' ' in row[2] else row[2]
                from_col = f'"{row[3]}"' if ' ' in row[3] else row[3]
                to_col = f'"{row[4]}"' if ' ' in row[4] else row[4]
                fks.append({
                    'from': f"{from_table}.{from_col}",
                    'to': f"{to_table}.{to_col}",
                    'join': f"JOIN {to_table} ON {from_table}.{from_col} = {to_table}.{to_col}"
                })
        except:
            pass

        if fks:
            paths[table] = fks

    return paths


def analyze_database(db_path, output_file, max_output_chars=150000):
    """Main analysis function - generates comprehensive documentation."""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]

    columns_by_table = {}
    total_columns = 0
    for table in tables:
        try:
            cursor.execute(f'PRAGMA table_info("{table}")')
            columns_by_table[table] = cursor.fetchall()
            total_columns += len(columns_by_table[table])
        except:
            columns_by_table[table] = []

    row_counts = {}
    for table in tables:
        row_counts[table] = get_table_row_count(cursor, table)

    is_large_db = len(tables) > 30 or total_columns > 200
    is_medium_db = len(tables) > 15 or total_columns > 100

    output = []

    # === SECTION 0: DOMAIN-SPECIFIC TABLES ===
    domain_tables = detect_domain_specific_tables(tables)

    if domain_tables:
        output.append("!" * 70)
        output.append("CRITICAL: DOMAIN-SPECIFIC TABLES DETECTED")
        output.append("!" * 70)
        output.append("")
        output.append("These tables contain SPECIALIZED data. Match question keywords to tables!")
        output.append("")
        output.append("DOMAIN TABLE MAPPING:")
        for dt in domain_tables:
            output.append(f"  {dt['table']}:")
            output.append(f"    Domain: {dt['meaning']}")
            output.append(f"    Keywords: {', '.join(dt['keywords'])}")
            if dt['base_table']:
                output.append(f"    Base table: {dt['base_table']} (general data)")
            output.append(f"    USE WHEN: Question mentions '{dt['meaning'].lower()}'")
            output.append("")
        output.append("RULE: If question asks about a specific domain (e.g., 'Stanley Cup'),")
        output.append("      use the domain-specific table, NOT the general table!")
        output.append("")

    # === SECTION 1: PERSON/ENTITY TABLES ===
    person_tables = detect_person_entity_tables(cursor, tables, columns_by_table)

    if person_tables:
        output.append("!" * 70)
        output.append("CRITICAL: PERSON/ENTITY TABLES - FOR 'WHO' QUESTIONS")
        output.append("!" * 70)
        output.append("")
        output.append("When question asks 'Who...', return NAMES from these tables, not IDs!")
        output.append("")
        for pt in person_tables:
            if pt['is_person_table'] or pt['has_first_last']:
                output.append(f"  {pt['table']}:")
                output.append(f"    Name columns: {', '.join(pt['name_columns'])}")
                if pt['has_first_last']:
                    output.append("    Has first/last name columns - use for 'Who' questions")
                output.append("")
        output.append("RULE: 'Who received X?' -> Return first, last (names), NOT patient_id!")
        output.append("")
        output.append("COUNT GUIDANCE FOR PERSON TABLES:")
        output.append("  - 'How many care plans/medications/procedures...' -> COUNT(*) (counting ITEMS)")
        output.append("  - 'How many UNIQUE/DIFFERENT patients...' -> COUNT(DISTINCT patient_id)")
        output.append("  - 'How many patients with condition X' (with JOIN) -> COUNT(DISTINCT patient_id)")
        output.append("  ONLY use COUNT(DISTINCT person_id) when:")
        output.append("    1. Question explicitly says 'unique', 'different', or 'distinct'")
        output.append("    2. JOIN creates one-to-many (multiple rows per person)")
        output.append("  DEFAULT: COUNT(*) unless above conditions apply!")
        output.append("")

    # === NEW SECTION: PERSON ATTRIBUTE KEYWORDS ===
    all_person_attributes = []
    for table in tables:
        columns = columns_by_table[table]
        attrs = detect_person_attributes(cursor, table, columns)
        all_person_attributes.extend(attrs)

    if all_person_attributes:
        output.append("!" * 70)
        output.append("CRITICAL: PERSON ATTRIBUTE KEYWORD MAPPINGS")
        output.append("!" * 70)
        output.append("")
        output.append("Questions may use WORDS that map to CODES in the database!")
        output.append("")
        for attr in all_person_attributes:
            output.append(f"  {attr['table']}.{attr['column']} ({attr['description']}):")
            output.append(f"    Values in DB: {attr['values']}")
            if attr['keyword_mappings']:
                output.append("    Keyword -> Code mappings:")
                for keyword, code in attr['keyword_mappings'].items():
                    if code in attr['values']:
                        output.append(f"      '{keyword}' -> '{code}'")
            output.append("")
        output.append("EXAMPLE: 'single patients' -> WHERE marital = 'S'")
        output.append("         'married patients' -> WHERE marital = 'M'")
        output.append("")

    # === NEW SECTION: RATING/QUALITY COLUMNS ===
    all_rating_columns = []
    for table in tables:
        columns = columns_by_table[table]
        rating_cols = detect_rating_quality_columns(cursor, table, columns)
        all_rating_columns.extend(rating_cols)

    if all_rating_columns:
        output.append("!" * 70)
        output.append("CRITICAL: RATING/QUALITY COLUMN INTERPRETATION")
        output.append("!" * 70)
        output.append("")
        output.append("These columns contain ORDINAL values. Understand the difference:")
        output.append("  - 'most rated' = COUNT(*) of ratings (how many times rated)")
        output.append("  - 'highest rated' = ORDER BY AVG(rating) DESC (best average rating)")
        output.append("  - 'best quality' = WHERE quality = MAX_VALUE")
        output.append("  - 'worst quality' = WHERE quality = MIN_VALUE")
        output.append("")
        for rc in all_rating_columns:
            output.append(f"  {rc['table']}.{rc['column']}:")
            output.append(f"    Values: {rc['values']}")
            if rc['min_val'] is not None:
                output.append(f"    Range: {rc['min_val']} (worst) to {rc['max_val']} (best)")
            for interp in rc['interpretation']:
                output.append(f"    {interp}")
            output.append("")

    # === SECTION: MOVIE DATABASE SEMANTICS ===
    movie_info = detect_movie_database(tables, columns_by_table)

    if movie_info:
        output.append("!" * 70)
        output.append("CRITICAL: MOVIE/MEDIA DATABASE PATTERNS")
        output.append("!" * 70)
        output.append("")
        output.append("This is a movie/media database. Key patterns:")
        output.append("")
        if movie_info['quality_columns']:
            output.append("QUALITY COLUMNS (ordinal ratings of people/things):")
            for qc in movie_info['quality_columns']:
                output.append(f"  - {qc}: Higher value = BETTER quality")
            output.append("  'best directors' -> WHERE d_quality = 5 (or MAX)")
            output.append("  'worst actors' -> WHERE a_quality = 0 (or MIN)")
            output.append("")
        if movie_info['rating_columns']:
            output.append("RATING COLUMNS (user ratings of movies):")
            for rc in movie_info['rating_columns']:
                output.append(f"  - {rc}")
            output.append("  'movie most rated by users' = COUNT(*) ratings (popularity)")
            output.append("  'movie highest rated' = AVG(rating) (quality)")
            output.append("")
        output.append("JUNCTION TABLE PATTERNS:")
        output.append("  - movies2actors: connects movies to actors")
        output.append("  - movies2directors: connects movies to directors")
        output.append("  - u2base: user ratings of movies")
        output.append("")

    # === SECTION 2: TABLE ROW COUNTS ===
    output.append("=" * 70)
    output.append("TABLE ROW COUNTS")
    output.append("=" * 70)
    output.append("")

    large_tables = []
    for table in sorted(tables):
        count = row_counts[table]
        if count > 100000:
            output.append(f"  {table}: {count:,} rows  *** LARGE TABLE ***")
            large_tables.append(table)
        else:
            output.append(f"  {table}: {count:,} rows")
    output.append("")

    if large_tables:
        output.append("WARNING: Large tables detected. Use filtering in queries.")
        output.append("")

    # === SECTION 3: AGGREGATION GUIDANCE ===
    all_agg_hints = []
    all_time_series = []

    for table in tables:
        columns = columns_by_table[table]
        row_count = row_counts[table]

        agg_hints = detect_aggregation_patterns(cursor, table, columns, row_count)
        all_agg_hints.extend(agg_hints)

        time_series = detect_time_series_patterns(cursor, table, columns, row_count)
        all_time_series.extend(time_series)

    if all_agg_hints or all_time_series:
        output.append("=" * 70)
        output.append("AGGREGATION GUIDANCE (addresses common errors)")
        output.append("=" * 70)
        output.append("")

        # Combined vs per-entity guidance
        output.append("COMBINED TOTALS vs PER-ENTITY RESULTS:")
        output.append("")
        output.append("  'Calculate X for A and B' = ONE combined total (SUM both together)")
        output.append("  'Calculate X for A and for B' = TWO values (one per entity)")
        output.append("  'Compare X between A and B' = TWO values (one per entity)")
        output.append("")

        if all_agg_hints:
            output.append("DUPLICATE NAME/ADDRESS WARNINGS:")
            output.append("")
            output.append("When question asks 'which X has the most Y', GROUP BY the ID column,")
            output.append("not the display column (name/address) which may have duplicates.")
            output.append("")
            for hint in all_agg_hints[:5]:
                output.append(f"  {hint['table']}:")
                output.append(f"    {hint['name_column']} has duplicates: {hint['examples']}")
                output.append(f"    {hint['hint']}")
                output.append("")

        if all_time_series:
            output.append("TIME-SERIES DATA PATTERNS:")
            output.append("")
            output.append("For 'the X in [year]' questions, consider AVG() across the year")
            output.append("rather than a single arbitrary value.")
            output.append("")
            for ts in all_time_series[:3]:
                output.append(f"  {ts['table']}:")
                output.append(f"    Date column: {ts['date_column']}")
                output.append(f"    Value columns: {', '.join(ts['value_columns'])}")
                output.append(f"    {ts['hint']}")
                output.append("")

    # === SECTION: MULTI-PART QUESTION GUIDANCE (NEW) ===
    output.append("=" * 70)
    output.append("MULTI-PART QUESTION GUIDANCE")
    output.append("=" * 70)
    output.append("")
    output.append("Questions may ask for MULTIPLE pieces of information. Return ALL requested!")
    output.append("")
    output.append("PATTERNS TO RECOGNIZE:")
    output.append("  'What is X AND what is Y?' -> SELECT X, Y")
    output.append("  'List X as well as Y' -> SELECT X, Y")
    output.append("  'Show X. Also show Y.' -> SELECT X, Y")
    output.append("  'What is X? What is Y? List Z too.' -> SELECT X, Y, Z")
    output.append("")
    output.append("COLUMN ORDER: Return columns in the SAME ORDER as mentioned in question!")
    output.append("")
    output.append("BUT REMEMBER: NO EXTRA COLUMNS beyond what is asked!")
    output.append("  'Which movie has most ratings?' -> SELECT movieid ONLY (not movieid, count)")
    output.append("  Aggregation goes in ORDER BY, not SELECT!")
    output.append("")

    # === SECTION 4: DATE FORMAT WARNINGS ===
    all_date_formats = []
    for table in tables:
        columns = columns_by_table[table]
        date_formats = detect_date_formats(cursor, table, columns)
        all_date_formats.extend(date_formats)

    if all_date_formats:
        output.append("!" * 70)
        output.append("CRITICAL: DATE COLUMN FORMAT WARNINGS")
        output.append("!" * 70)
        output.append("")
        output.append("These date columns use NON-ISO formats. Check sample values before filtering!")
        output.append("")
        for df in all_date_formats:
            output.append(f"  {df['table']}.{df['column']}:")
            output.append(f"    Sample values: {df['samples']}")
            if df['format_hint']:
                output.append(f"    Format: {df['format_hint']}")
            if df.get('filter_example'):
                output.append(f"    Filtering: {df['filter_example']}")
            output.append("")
        output.append("RULE: For M/D/YY format dates:")
        output.append("  - Year 2018: use LIKE '%/%/18' (NOT STRFTIME or SUBSTR)")
        output.append("  - January 2020: use LIKE '1/%/20' (month is NOT zero-padded)")
        output.append("  - Specific date: use LIKE '1/15/20' for Jan 15, 2020")
        output.append("")

    # === SECTION 5: NAME VARIANT WARNINGS ===
    all_name_warnings = []
    for table in tables:
        columns = columns_by_table[table]
        name_warnings = detect_name_variants(cursor, table, columns)
        all_name_warnings.extend(name_warnings)

    if all_name_warnings:
        output.append("=" * 70)
        output.append("NAME VARIANT WARNINGS (potential typos/spelling)")
        output.append("=" * 70)
        output.append("")
        output.append("These similar names exist in the database. If your query returns empty,")
        output.append("check if the question has a typo (e.g., 'Joe' vs 'Joye').")
        output.append("")
        for warning in all_name_warnings[:10]:
            output.append(f"  {warning['table']}.{warning['column']}:")
            output.append(f"    {warning['warning']}")
            output.append("")

    # === SECTION 6: ACADEMIC DATABASE SEMANTICS ===
    academic_info = detect_academic_database(tables, columns_by_table)

    if academic_info:
        output.append("=" * 70)
        output.append("ACADEMIC DATABASE SEMANTIC MAPPINGS")
        output.append("=" * 70)
        output.append("")
        output.append("This is an academic paper database. Use these semantic mappings:")
        output.append("")

        if academic_info['has_conference_id'] and academic_info['has_journal_id']:
            output.append("PUBLICATION TYPE CONDITIONS:")
            output.append(f"  - 'preprint' -> ConferenceId = 0 AND JournalId = 0")
            output.append(f"  - 'conference paper' -> ConferenceId != 0")
            output.append(f"  - 'journal paper' -> JournalId != 0")
            output.append("")

        if academic_info['paper_author_table'] and academic_info['paper_author_has_name']:
            output.append("AUTHOR LOOKUP GUIDANCE:")
            output.append(f"  The {academic_info['paper_author_table']} table has a 'Name' column!")
            output.append(f"  To find papers by author: {academic_info['paper_author_table']}.Name = 'Author Name'")
            output.append("")

    # === SECTION 7: RAW DDL SCHEMA ===
    output.append("=" * 70)
    output.append("DATABASE SCHEMA DOCUMENTATION")
    output.append("=" * 70)
    output.append("")

    ddl_statements = get_schema_ddl(cursor)
    for ddl in ddl_statements:
        output.append(ddl + ";")
        output.append("")

    # === SECTION 8: CRITICAL WARNINGS ===
    output.append("=" * 70)
    output.append("CRITICAL: COLUMN SELECTION & VALUE MATCHING DISCIPLINE")
    output.append("=" * 70)
    output.append("")
    output.append("This section addresses the top error patterns (90% of errors).")
    output.append("")

    # Part 1: Column Selection
    output.append("-" * 70)
    output.append("PART 1: COLUMN SELECTION DISCIPLINE (40% of errors)")
    output.append("-" * 70)
    output.append("")
    output.append("!" * 70)
    output.append("!!! NO EXTRA COLUMNS - EVER - THIS IS THE #1 ERROR !!!")
    output.append("!" * 70)
    output.append("")
    output.append("WRONG: 'Which X has the most Y?' -> SELECT X, COUNT(*)")
    output.append("RIGHT: 'Which X has the most Y?' -> SELECT X ... ORDER BY COUNT(*) DESC LIMIT 1")
    output.append("")
    output.append("The aggregation belongs in ORDER BY, NOT in SELECT!")
    output.append("")
    output.append("GENERAL GUIDANCE:")
    output.append("- Return ONLY columns EXPLICITLY requested in the question")
    output.append("- Typical queries select 1-3 columns total")
    output.append("- AVOID SELECT * unless specifically requested")
    output.append("- SELECT columns in the SAME ORDER as mentioned in the question")
    output.append("- If unsure, fewer columns is ALWAYS better")
    output.append("")

    output.append("PER-TABLE COLUMN GUIDANCE:")
    output.append("")

    tables_for_guidance = tables[:25] if is_large_db else (tables[:40] if is_medium_db else tables)

    for table in tables_for_guidance:
        columns = columns_by_table[table]
        row_count = row_counts[table]
        selectivity = analyze_column_selectivity_enhanced(cursor, table, columns, row_count)

        high_pri = [c for c, info in selectivity.items() if info['priority'] == 'HIGH']
        medium_pri = [c for c, info in selectivity.items() if info['priority'] == 'MEDIUM']
        low_pri = [c for c, info in selectivity.items() if info['priority'] == 'LOW']

        output.append(f"Table: {table} ({len(columns)} columns, {row_count:,} rows)")
        if high_pri:
            output.append(f"  High-priority: {', '.join(high_pri)}")
        if medium_pri:
            output.append(f"  Medium-priority: {', '.join(medium_pri)}")
        if low_pri:
            output.append(f"  Low-priority (rarely needed): {', '.join(low_pri)}")

        very_high = [c for c, info in selectivity.items() if info['cardinality'] == 'VERY_HIGH']
        low_card = [c for c, info in selectivity.items() if info['cardinality'] in ('LOW', 'VERY_LOW')]

        if very_high:
            output.append(f"  Unique identifiers: {', '.join(very_high)}")
        if low_card:
            output.append(f"  Limited value sets: {', '.join(low_card)}")

        high_null = [(c, info['null_pct']) for c, info in selectivity.items() if info['null_pct'] > 20]
        if high_null:
            null_info = ', '.join([f"{c} ({pct}%)" for c, pct in high_null[:3]])
            output.append(f"  High null %: {null_info}")

        output.append("")

    if len(tables) > len(tables_for_guidance):
        output.append(f"... {len(tables) - len(tables_for_guidance)} additional tables not shown")
        output.append("")

    # Part 2: Case Sensitivity
    output.append("-" * 70)
    output.append("PART 2: CASE SENSITIVITY & VALUE FORMAT WARNINGS (30% of errors)")
    output.append("-" * 70)
    output.append("")

    all_case_warnings = []
    all_lowercase_columns = []
    all_value_guides = []
    all_unicode_warnings = []

    for table in tables:
        columns = columns_by_table[table]

        case_patterns = analyze_case_patterns_enhanced(cursor, table, columns)
        value_samples = get_value_samples(cursor, table, columns)
        unicode_warnings = detect_unicode_issues(cursor, table, columns)

        for lc in case_patterns['lowercase_only']:
            all_lowercase_columns.append({
                'table': table,
                'column': lc['column'],
                'samples': lc['samples']
            })

        for warning in case_patterns['case_warnings'][:15]:
            all_case_warnings.append({
                'table': table,
                **warning
            })

        all_unicode_warnings.extend(unicode_warnings)

        for col_name, sample_info in value_samples.items():
            if sample_info['total_distinct'] > 0 and sample_info['total_distinct'] <= 50 and sample_info['is_text']:
                priority = 0 if sample_info.get('is_location') else (1 if sample_info.get('has_abbreviations') else 2)
                all_value_guides.append({
                    'table': table,
                    'column': col_name,
                    'cardinality': 'LOW' if sample_info['is_complete'] else 'MEDIUM',
                    'values': sample_info['values'],
                    'total': sample_info['total_distinct'],
                    'priority': priority,
                    'has_abbreviations': sample_info.get('has_abbreviations', False)
                })

    if all_unicode_warnings:
        output.append("UNICODE CHARACTER WARNINGS (special characters that need exact matching):")
        output.append("")
        seen = set()
        for warning in all_unicode_warnings[:15]:
            key = (warning['table'], warning['column'], warning['character'])
            if key not in seen:
                seen.add(key)
                output.append(f"  {warning['table']}.{warning['column']}:")
                output.append(f"    Value: '{warning['value']}'")
                output.append(f"    Special char: {warning['description']}")
                output.append("")

    if all_lowercase_columns:
        output.append("LOWERCASE-ONLY COLUMNS (use lowercase in WHERE clauses):")
        output.append("")
        max_lc_cols = 15 if is_large_db else (25 if is_medium_db else 35)
        for item in all_lowercase_columns[:max_lc_cols]:
            output.append(f"  {item['table']}.{item['column']}:")
            output.append(f"    ALL values are lowercase")
            output.append(f"    Sample values: {item['samples'][:5]}")
            output.append(f"    WARNING: Do NOT use title case or capitalized values")
            output.append("")

    if all_case_warnings:
        output.append("CASE SENSITIVITY PITFALLS (common mistakes to avoid):")
        output.append("")
        max_warnings = 20 if is_large_db else (35 if is_medium_db else 50)
        for warning in all_case_warnings[:max_warnings]:
            output.append(f"  {warning['table']}.{warning['column']}:")
            output.append(f"    Database has: '{warning['database_value']}'")
            for mistake in warning.get('common_mistakes', [])[:2]:
                output.append(f"    WRONG: '{mistake['wrong']}' - {mistake['reason']}")
            output.append("")

    if all_value_guides:
        all_value_guides.sort(key=lambda x: (x.get('priority', 2), 0 if x['cardinality'] == 'LOW' else 1))
        output.append("VALUE FORMAT GUIDE (exact values for matching):")
        output.append("")

        abbrev_cols = [g for g in all_value_guides if g.get('has_abbreviations')]
        if abbrev_cols:
            output.append("STATE/COUNTRY ABBREVIATION COLUMNS (use codes, not full names):")
            for guide in abbrev_cols[:5]:
                output.append(f"  {guide['table']}.{guide['column']}: {guide['values'][:10]}")
                output.append(f"    USE CODES like 'MA', 'CA' - NOT 'Massachusetts', 'California'")
            output.append("")

        max_guides = 25 if is_large_db else (40 if is_medium_db else 60)
        for guide in all_value_guides[:max_guides]:
            max_vals = 10 if is_large_db else 15
            vals_to_show = guide['values'][:max_vals]
            output.append(f"  {guide['table']}.{guide['column']} ({guide['cardinality']} cardinality, {guide['total']} values):")
            if guide['cardinality'] == 'LOW':
                output.append(f"    All possible values: {vals_to_show}")
            else:
                output.append(f"    Sample values: {vals_to_show}")
            output.append("")

    # Part 3: Isolated Tables
    output.append("-" * 70)
    output.append("PART 3: ISOLATED TABLES WARNING (20% of errors)")
    output.append("-" * 70)
    output.append("")

    isolated, partially_isolated = identify_isolated_tables(cursor, tables)

    if isolated:
        output.append("FULLY ISOLATED TABLES (NO foreign key relationships):")
        output.append("")
        for table in isolated:
            output.append(f"  Table: {table}")
            output.append("    Foreign Keys: NONE")
            output.append("    WARNING: DO NOT JOIN or UNION this table with other tables")
            output.append("    Usage: Can ONLY be used directly or in subqueries")
            output.append("")

    if partially_isolated:
        output.append("PARTIALLY ISOLATED TABLES (no outgoing FKs, but referenced by others):")
        output.append("")
        for item in partially_isolated:
            output.append(f"  Table: {item['table']}")
            output.append(f"    Referenced by: {', '.join(item['referenced_by'])}")
            output.append("    WARNING: Can be JOINed TO but cannot JOIN other tables")
            output.append("")

    if not isolated and not partially_isolated:
        output.append("No isolated tables detected - all tables have foreign key relationships.")
        output.append("")

    # === SECTION 9: SIMILAR TABLE DISAMBIGUATION ===
    similar_groups = find_similar_table_names(tables)
    if similar_groups:
        output.append("!" * 70)
        output.append("CRITICAL: SIMILAR TABLE NAMES - DO NOT UNION UNLESS ASKED!")
        output.append("!" * 70)
        output.append("")
        output.append("These tables have similar names. DO NOT UNION them unless explicitly asked!")
        output.append("Use ONLY ONE table - typically the smaller one or the one mentioned in evidence.")
        output.append("")

        for group in similar_groups:
            output.append(f"  Similar tables: {', '.join(group)}")
            for table in group:
                row_count = row_counts.get(table, 0)
                cols = columns_by_table.get(table, [])
                col_names = [c[1] for c in cols[:5]]
                output.append(f"    {table}: {row_count:,} rows, columns: {', '.join(col_names)}")
            smallest_table = min(group, key=lambda t: row_counts.get(t, 0))
            output.append(f"  DEFAULT: Use '{smallest_table}' (smallest) unless evidence specifies otherwise")
            output.append("  RULE: Do NOT UNION these tables together unless question asks for 'all' or 'combined' data!")
            output.append("")

    # === SECTION 10: COLUMN QUOTING REQUIREMENTS ===
    output.append("=" * 70)
    output.append("COLUMN QUOTING REQUIREMENTS")
    output.append("=" * 70)
    output.append("")

    needs_quotes = get_columns_needing_quotes(cursor, tables)

    if needs_quotes:
        for table, cols in needs_quotes.items():
            output.append(f"Table: {table}")
            for col in cols:
                output.append(f"  - `{col}` - requires backtick quoting")
            output.append("")
    else:
        output.append("No columns require backtick quoting in this database.")
        output.append("")

    # === SECTION 11: RELATIONSHIP & JOIN PATTERNS ===
    output.append("=" * 70)
    output.append("RELATIONSHIP & JOIN PATTERNS")
    output.append("=" * 70)
    output.append("")

    output.append("EXPLICIT FOREIGN KEY RELATIONSHIPS:")
    output.append("")

    join_paths = build_join_paths(cursor, tables)

    if join_paths:
        max_tables_for_joins = 25 if is_large_db else (40 if is_medium_db else len(join_paths))
        tables_shown = 0
        for table, fks in join_paths.items():
            if tables_shown >= max_tables_for_joins:
                break
            output.append(f"  {table}:")
            for fk in fks[:5]:
                output.append(f"    - {fk['from']} -> {fk['to']}")
                output.append(f"      {fk['join']}")
            if len(fks) > 5:
                output.append(f"    ... and {len(fks) - 5} more relationships")
            output.append("")
            tables_shown += 1
    else:
        output.append("  No explicit foreign key relationships defined.")
        output.append("")

    implicit_fks = detect_implicit_fks(cursor, tables, columns_by_table)
    if implicit_fks:
        output.append("IMPLICIT RELATIONSHIPS (detected via naming patterns):")
        output.append("")
        for fk in implicit_fks[:20]:
            output.append(f"  {fk['from_table']}.{fk['from_column']} -> {fk['to_table']}")
            output.append(f"    Confidence: {fk['confidence']} ({fk['reason']})")
        output.append("")

    output.append("JOIN RECOMMENDATIONS:")
    output.append("")
    output.append("  - Use INNER JOIN when both sides must have matching records")
    output.append("  - Use LEFT JOIN when right side is optional")
    output.append("  - Prefer JOINs over nested IN subqueries for aggregations")
    output.append("  - Follow the FK paths documented above")
    output.append("  - NEVER JOIN isolated tables")
    output.append("")

    # === SECTION 12: TABLE ROLE CLASSIFICATION ===
    output.append("=" * 70)
    output.append("TABLE ROLE CLASSIFICATION")
    output.append("=" * 70)
    output.append("")

    fk_counts = {}
    referenced_counts = defaultdict(int)
    for table in tables:
        fks = get_foreign_keys(cursor, table)
        fk_counts[table] = len(fks)
        for fk in fks:
            referenced_counts[fk['to_table']] += 1

    role_groups = defaultdict(list)
    for table in tables:
        row_count = row_counts[table]
        role = classify_table_role(
            cursor, table, columns_by_table.get(table, []),
            fk_counts.get(table, 0), referenced_counts.get(table, 0), row_count
        )
        role_groups[role].append((table, row_count))

    for role in ['ENTITY', 'LOOKUP', 'JUNCTION', 'TRANSACTION', 'STANDARD']:
        if role_groups[role]:
            output.append(f"{role} TABLES:")
            for table, row_count in role_groups[role]:
                output.append(f"  - {table} ({row_count:,} rows)")
            output.append("")

    # === SUMMARY ===
    output.append("=" * 70)
    output.append("ANALYSIS SUMMARY")
    output.append("=" * 70)
    output.append("")
    db_size_label = "LARGE" if is_large_db else ("MEDIUM" if is_medium_db else "SMALL")
    output.append(f"Database size: {db_size_label} ({len(tables)} tables, {total_columns} columns)")
    output.append(f"Domain-specific tables: {len(domain_tables)}")
    output.append(f"Person/Entity tables: {len([p for p in person_tables if p['is_person_table'] or p['has_first_last']])}")
    output.append(f"Person attribute columns: {len(all_person_attributes)}")
    output.append(f"Rating/quality columns: {len(all_rating_columns)}")
    output.append(f"Date format warnings: {len(all_date_formats)}")
    output.append(f"Lowercase-only columns detected: {len(all_lowercase_columns)}")
    output.append(f"Case sensitivity warnings: {len(all_case_warnings)}")
    output.append(f"Unicode character warnings: {len(all_unicode_warnings)}")
    output.append(f"Name variant warnings: {len(all_name_warnings)}")
    output.append(f"Aggregation hints: {len(all_agg_hints)}")
    output.append(f"Time-series patterns: {len(all_time_series)}")
    output.append(f"Similar table groups: {len(similar_groups)}")
    output.append(f"Fully isolated tables: {len(isolated)}")
    output.append(f"Partially isolated tables: {len(partially_isolated)}")
    output.append(f"Implicit FK relationships: {len(implicit_fks)}")
    if academic_info:
        output.append(f"Database type: ACADEMIC PAPER DATABASE")
    if movie_info:
        output.append(f"Database type: MOVIE/MEDIA DATABASE")
    output.append("")

    conn.close()

    # Write output
    output_text = '\n'.join(output)
    output_size_kb = len(output_text) / 1024

    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else 'tool_output', exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(output_text)

    print(f"Mega cross-pollinated analysis complete - wrote to {output_file}")
    print(f"  - {len(tables)} tables analyzed")
    print(f"  - Database size classification: {db_size_label}")
    print(f"  - {len(domain_tables)} domain-specific tables detected")
    print(f"  - {len([p for p in person_tables if p['is_person_table'] or p['has_first_last']])} person/entity tables")
    print(f"  - {len(all_person_attributes)} person attribute columns")
    print(f"  - {len(all_rating_columns)} rating/quality columns")
    print(f"  - {len(all_date_formats)} date format warnings")
    print(f"  - {len(all_lowercase_columns)} lowercase-only columns found")
    print(f"  - {len(all_case_warnings)} case warnings generated")
    print(f"  - {len(all_unicode_warnings)} unicode warnings")
    print(f"  - {len(all_name_warnings)} name variant warnings")
    print(f"  - {len(all_agg_hints)} aggregation hints")
    print(f"  - {len(all_time_series)} time-series patterns")
    print(f"  - {len(similar_groups)} similar table groups")
    print(f"  - {len(isolated)} fully isolated tables")
    print(f"  - {len(partially_isolated)} partially isolated tables")
    print(f"  - {len(implicit_fks)} implicit FK relationships")
    print(f"  - Output size: {output_size_kb:.1f} KB")

    if output_size_kb > 100:
        print(f"  WARNING: Output is {output_size_kb:.0f} KB - may approach context limits")

    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mega_cross_pollinated_analyzer.py [database.sqlite]")
        print("Default: database.sqlite")
        db_path = "database.sqlite"
    else:
        db_path = sys.argv[1]

    if not os.path.exists(db_path):
        print(f"Error: Database file not found: {db_path}")
        sys.exit(1)

    output_file = "tool_output/schema_analysis.txt"
    exit_code = analyze_database(db_path, output_file)
    sys.exit(exit_code)
