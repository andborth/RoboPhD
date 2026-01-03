---
name: mega-cross-pollinated-analyzer-v15
description: Cross-pollinated tool combining best patterns from iter3, iter6, iter9, iter10, iter11 plus new rating/quality detection
execution_mode: tool_only
tool_command: python tools/mega_cross_pollinated_analyzer.py
tool_output_file: tool_output/schema_analysis.txt
---

# Mega Cross-Pollinated Database Analyzer v15 (Tool-Only)

This agent uses deterministic tool-only execution combining the most successful patterns from top-performing agents in the experiment:

- **iter10_ultimate_cross_pollinated_analyzer** (ELO 1597): Person/entity detection, date format warnings, COUNT(DISTINCT) guidance
- **iter11_evolved_3095** (ELO 1569): Person attribute keyword mappings, stronger column discipline, combined vs per-entity guidance
- **iter6_unified_cross_pollinated_analyzer** (ELO 1526): Domain-specific table detection, implicit FK detection, table role classification, unicode warnings
- **iter9_comprehensive_cross_pollinated_analyzer** (ELO 1524): Aggregation guidance, time-series patterns, name variant detection, academic DB semantics
- **iter3_enhanced_cross_pollinated_precision_analyzer** (ELO 1508): Enhanced selectivity scoring, multi-word case pitfall detection, adaptive output sizing

**New for Iteration 15:**
- **RATING/QUALITY COLUMN DETECTION**: Identifies ordinal rating/quality columns and provides interpretation guidance ("most rated" vs "highest rated")
- **MOVIE/MEDIA DATABASE PATTERNS**: Specific handling for movielens-style databases with actor/director quality columns
- **MULTI-PART QUESTION GUIDANCE**: Explicit guidance for questions asking for multiple pieces of information
- **ENHANCED "MOST" vs "HIGHEST" DISAMBIGUATION**: Clearer guidance on COUNT vs AVG aggregations

## Process

1. **Run Mega Cross-Pollinated Analysis Tool**
   - Execute: `python tools/mega_cross_pollinated_analyzer.py`
   - Output: `tool_output/schema_analysis.txt`

2. **Read and Output Results**
   - Read the generated analysis from `tool_output/schema_analysis.txt`
   - Write the complete output to `./output/agent_output.txt`

## Output Structure

The tool generates comprehensive database documentation with these sections:

1. **DOMAIN-SPECIFIC TABLES** - Tables with specialized data (Stanley Cup, Playoffs, etc.)
2. **PERSON/ENTITY TABLES** - Tables with name columns for "Who" questions
3. **PERSON ATTRIBUTE KEYWORDS** - Mappings like 'single' -> marital='S'
4. **RATING/QUALITY COLUMNS** - Ordinal columns with interpretation guidance (NEW)
5. **MOVIE/MEDIA DATABASE PATTERNS** - Actor/director quality, rating semantics (NEW)
6. **TABLE ROW COUNTS** - Size information for all tables
7. **AGGREGATION GUIDANCE** - GROUP BY selection, time-series AVG hints, combined vs per-entity
8. **MULTI-PART QUESTION GUIDANCE** - How to handle complex multi-output questions (NEW)
9. **DATE FORMAT WARNINGS** - Non-ISO date formats with filtering guidance
10. **NAME VARIANT WARNINGS** - Similar names that might be typos
11. **ACADEMIC DATABASE SEMANTICS** - Special rules for paper/author databases
12. **DATABASE SCHEMA DOCUMENTATION** - Raw DDL statements
13. **CRITICAL COLUMN SELECTION & VALUE MATCHING** - The core error-prevention section:
    - Part 1: Column Selection Discipline (40% of errors)
    - Part 2: Case Sensitivity & Value Format Warnings (30% of errors)
    - Part 3: Isolated Tables Warning (20% of errors)
14. **SIMILAR TABLE NAMES** - Disambiguation for confusing table names
15. **COLUMN QUOTING REQUIREMENTS** - Reserved words and special characters
16. **RELATIONSHIP & JOIN PATTERNS** - FK paths and JOIN templates
17. **TABLE ROLE CLASSIFICATION** - ENTITY, LOOKUP, JUNCTION, TRANSACTION roles
18. **ANALYSIS SUMMARY** - Database statistics and warning counts

## Error Recovery

If the tool fails:

1. Check `database.sqlite` exists in the working directory
2. Verify Python environment has sqlite3 (standard library)
3. Examine any error messages in `tool_output/`
4. Attempt to run the tool manually to see errors:
   ```bash
   python tools/mega_cross_pollinated_analyzer.py database.sqlite
   ```
5. Fall back to manual schema analysis if needed:
   ```sql
   SELECT sql FROM sqlite_master WHERE type='table';
   ```

## Key Features

### From iter10 (Foundation)
- Person/entity table detection for "Who" questions
- Date format detection with filtering patterns
- Enhanced COUNT(DISTINCT) guidance

### From iter11 (Attribute Intelligence)
- Person attribute keyword mappings (marital, gender, race codes)
- Stronger "NO EXTRA COLUMNS" emphasis
- Combined vs per-entity aggregation guidance

### From iter6 (Domain Intelligence)
- Domain suffix detection (SC, Post, WC, Finals)
- Implicit FK detection via naming patterns
- Unicode character warnings (curly quotes, em-dashes)
- Table role classification

### From iter9 (Semantics)
- Aggregation pattern detection (duplicate names requiring GROUP BY id)
- Time-series data patterns (AVG for yearly values)
- Academic database semantic mappings
- State/country abbreviation detection

### From iter3 (Precision)
- Adaptive output sizing for large/medium/small databases
- Priority classification for columns (HIGH/MEDIUM/LOW)
- Multi-word case pitfall detection with `common_mistakes`

### New for Iteration 15
- **Rating/Quality Column Detection**: Identifies columns like `rating`, `quality`, `score` and provides interpretation
- **Movie/Media Database Patterns**: Specific guidance for movielens-style databases
- **Multi-Part Question Guidance**: Explicit handling of "what is X AND what is Y" questions
- **"Most" vs "Highest" Disambiguation**: COUNT(*) for "most rated", AVG for "highest rated"

## Expected Accuracy Improvements

Based on error analysis and cross-pollination strategy:
- +5-8% from rating/quality interpretation (addresses movielens 40% error rate)
- +2-3% from multi-part question handling
- +1-2% from enhanced "most" vs "highest" guidance
- +1% from stronger NO EXTRA COLUMNS emphasis

Target: 80-82% accuracy on new databases (up from 76-78% current best)
