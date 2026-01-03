---
name: hybrid-comprehensive-analyzer
description: Cross-pollinated tool combining adaptive scaling (iter16), advanced features (iter9), and precision techniques (iter3) for optimal accuracy across all database sizes
execution_mode: tool_only
tool_command: python tools/hybrid_comprehensive_analyzer.py
tool_output_file: tool_output/analysis.txt
---

# Hybrid Comprehensive Analyzer (Tool-Only Cross-Pollination)

This agent combines the best proven techniques from three top-performing agents to achieve high accuracy across all database sizes while preventing context overflow.

## Cross-Pollination Strategy

This agent synthesizes proven techniques from the top 3 performing agents:

### From iter16_adaptive_comprehensive_analyzer (ELO 1535, 78.7% recent winner)
✅ Database size classification (small/medium/large/ultra-large)
✅ Adaptive sample limits (10 → 5 → 3 → 1 based on size)
✅ Adaptive enum truncation (all → 15 → 5 → 0 based on size)
✅ Context budget tracking (ensures <200k token limit)
✅ Graceful degradation strategies

**Why**: Recent iteration winner, solves context overflow problem

### From iter9_unified_precision_analyzer (ELO 1552, 84.7% peak)
✅ Semantic pattern detection (temporal sequences, hierarchies, composites)
✅ Cross-table value validation (orphaned FK detection)
✅ Statistical distribution hints (cardinality ratios, distribution types)
✅ 10-section output structure (well-organized, navigable)
✅ Comprehensive query guidance

**Why**: Best peak performance, most sophisticated analysis features

### From iter3_precision_synthesis_enhanced (ELO 1536, 77.8% consistent)
✅ ALL column coverage (not limited to first 10 columns - critical!)
✅ Enum value detection (columns with <20 distinct values)
✅ Cardinality analysis (1:1 vs 1:N relationship patterns with averages)
✅ Value range analysis (MIN/MAX for numeric columns)
✅ Enhanced format detection (currency, percentage, time, date, codes)

**Why**: Consistent high performer, precision features proven effective

## New Enhancements (Beyond Current Agents)

### Enhancement 1: String Matching Guidance
✅ Explicit LIKE vs = pattern detection based on enum values
✅ Column name plurality warnings (Genre vs Genres)
✅ Multi-value delimiter detection (comma-separated, pipe-separated)

### Enhancement 2: NULL/NaN Value Detection
✅ Separate tracking of NULL, string 'nan', and empty strings
✅ Handling hints for each pattern type
✅ Integration with aggregation guidance

### Enhancement 3: Evidence Formula Complexity Warnings
✅ Common evidence pitfall identification
✅ Value contradiction warnings
✅ Multi-condition formula breakdown guidance

## Adaptive Analysis Strategy

### Small Databases (≤15 tables, ≤150 columns)
- **Full comprehensive analysis** (iter9 style)
- All columns with complete details
- 10 sample values per column
- Full enum value listings (all values)
- Complete semantic pattern analysis
- Cross-table validation
- Statistical distribution hints
- **Output target**: ~50,000 tokens

### Medium Databases (≤40 tables, ≤300 columns)
- **ALL columns analyzed** (still critical)
- 5 sample values per column
- Enum values truncated to first 15
- Essential semantic patterns
- Critical cross-table validation only
- **Output target**: ~75,000 tokens

### Large Databases (≤400 columns)
- **ALL columns analyzed** with types/constraints
- 3 sample values per column
- Enum values truncated to first 5
- Skip semantic patterns
- Skip cross-table validation
- Skip statistical hints
- **Output target**: ~100,000 tokens

### Ultra-Large Databases (>400 columns)
- **ULTRA-COMPACT FORMAT**
- ALL columns with minimal detail
- 1 sample value per column
- No enum detection
- No advanced analysis
- **Output target**: ~100,000 tokens (safe margin under 200k limit)

## Process

### Execution Flow

1. **Database Size Classification**
   ```bash
   python tools/hybrid_comprehensive_analyzer.py
   ```
   - Counts tables and columns
   - Classifies as small/medium/large/ultra-large
   - Selects appropriate analysis depth

2. **Adaptive Analysis Generation**
   - Applies size-appropriate truncation rules
   - Preserves critical information at all sizes
   - Tracks context budget during generation
   - Outputs to `tool_output/analysis.txt`

3. **System Integration**
   - Analysis automatically copied to `output/agent_output.txt`
   - Combined with eval_instructions.md for final system prompt
   - Guaranteed to fit within 200k token limit

### Output Structure (10 Sections)

**All Database Sizes Include**:
```
1. Header (size classification + strategy)
2. Complete Schema (DDL or compact for ultra-large)
3. Table Overview with row counts
4. Detailed Column Analysis (ALL columns, adaptive detail)
5. Relationship Map (all foreign keys with cardinality)
6. Enum Value Reference (adaptive truncation)
7. Value Range Summary (numeric boundaries)
8. Format Detection Summary (NULL/NaN patterns)
```

**Small/Medium Databases Also Include**:
```
9. Semantic Patterns (temporal, hierarchical, composite)
10. Cross-Table Validation (orphaned FK detection)
11. Statistical Distribution Hints (cardinality, distribution types)
```

**All Sizes Include**:
```
12. Query Guidance (pattern-specific tips + evidence warnings)
```

## Key Features

### Always Preserved (All Database Sizes)

**Complete Schema**: Full DDL for all tables, indexes, views
**ALL Columns**: Names, types, constraints for every column
**Foreign Keys**: Complete relationship mapping with cardinality hints
**Format Detection**: Currency, percentage, time, date, code patterns
**NULL Analysis**: Percentage of NULL values + string 'nan' detection
**Sample Data**: At least 1 sample per column (shows actual patterns)

### Adaptive Features (Based on Size)

**Sample Data**:
- Small: 10 samples per column
- Medium: 5 samples per column
- Large: 3 samples per column
- Ultra-large: 1 sample per column

**Enum Values**:
- Small: Complete listings for all columns with <20 distinct values
- Medium: Truncated to first 15 values with count
- Large: Truncated to first 5 values with count
- Ultra-large: Skip enum detection entirely

**Advanced Analysis**:
- Small: Full semantic patterns, cross-table validation, statistical hints
- Medium: Essential semantic patterns, critical validation only
- Large: Skip advanced analysis (focus on core information)
- Ultra-large: Skip all advanced analysis

## Advanced Features (Small/Medium Databases Only)

### Semantic Pattern Detection (from iter9)

**Temporal Sequences**: Detects multiple date/time columns suggesting temporal ordering
**Hierarchical Relationships**: Detects category/subcategory patterns
**Composite Identifiers**: Detects multi-column key patterns

### Cross-Table Value Validation (from iter9)

**Orphaned FK Detection**: Detects FK values with no matching parent
**Broken Relationships**: Alerts to broken relationships causing unexpected JOIN results
**JOIN Recommendations**: Recommends LEFT JOIN vs INNER JOIN based on validation

### Statistical Distribution Hints (from iter9)

**Cardinality Ratios**: distinct values / total values
**Distribution Classification**: enum-like, unique-like, low-cardinality, normal
**Data Pattern Understanding**: Helps eval model understand value distributions

### String Matching Guidance (NEW)

**LIKE vs = Detection**: Analyzes enum values to determine appropriate matching
**Multi-Value Detection**: Detects comma/pipe-separated values in columns
**Column Plurality Warnings**: Warns when singular evidence refers to plural column

### NULL/NaN Pattern Detection (NEW)

**NULL Count**: Actual NULL values
**String 'nan' Count**: String 'nan' or 'NaN' values
**Empty String Count**: Empty string '' values
**Handling Hints**: Recommended SQL handling (COALESCE, REPLACE, etc.)

## Critical Information Preservation

Even in **ultra-large database mode** (most aggressive truncation), we preserve:

✅ Every table name and structure
✅ Every column name, type, and constraint
✅ Every foreign key relationship
✅ Format detection for every column
✅ NULL percentages for every column
✅ Sample value for every column (minimum 1)

**Philosophy**: Abbreviated information beats context overflow error (0% accuracy).

## Error Recovery

### If Tool Fails

1. Check `database.sqlite` exists and is readable
2. Verify Python environment has sqlite3 library
3. Examine error messages in console output
4. Attempt manual execution:
   ```bash
   python tools/hybrid_comprehensive_analyzer.py
   ```
5. Check `tool_output/` for partial analysis

### Emergency Fallback

If output still approaching context limit:
- Escalate to emergency mode
- Schema + table list + minimal column info only
- Better than overflow error

### Manual Fallback

If tool-only execution fails entirely:
1. Read schema: `sqlite3 database.sqlite ".schema"`
2. Manually sample data from key tables
3. Document findings in `output/agent_output.txt`

## Success Metrics

### Design Targets

**Execution time**: 15-30 seconds (acceptable for comprehensive analysis)
**Cost**: $0.00 for Phase 1 (tool-only execution)
**Output size**: <100k tokens for large databases (safe margin under 200k limit)

### Accuracy Targets

**Small databases**: 95-100% (full comprehensive analysis)
**Medium databases**: 87-90% (smart truncation + advanced features)
**Large databases**: 73-77% (essential info prevents overflow)
**Overall**: 78-80% (meaningful improvement over current 76%)

### Error Pattern Mitigation

- ✓ ALL columns analyzed (catches important columns beyond first 10)
- ✓ Semantic patterns (helps complex domain logic - synthea, talkingdata)
- ✓ Cross-table validation (prevents broken JOIN surprises)
- ✓ Statistical hints (better aggregation understanding)
- ✓ Enum detection (helps exact value matching)
- ✓ String matching guidance (reduces LIKE vs = confusion)
- ✓ NULL/NaN handling (explicit pattern detection and guidance)
- ✓ Cardinality hints (better JOIN decisions)
- ✓ Value ranges (helps boundary conditions)
- ✓ Context overflow prevention (prevents catastrophic 0% failures)

## Expected Improvements

**vs iter16_adaptive_comprehensive_analyzer** (current #1, 78.7%):
- ✅ Maintain context overflow prevention
- ➕ Add semantic pattern detection (helps complex databases)
- ➕ Add cross-table validation (prevents broken relationships)
- ➕ Add statistical hints (better data understanding)
- ➕ Add string matching guidance (reduces LIKE vs = errors)
- ➕ Add NULL/NaN pattern detection (handles app_store errors)
- Expected: +2-4% on small/medium databases

**vs iter9_unified_precision_analyzer** (84.7% peak, but 61.3% recent):
- ✅ Maintain all comprehensive features
- ➕ Add context overflow prevention (CRITICAL)
- Expected: +15-20% overall from solving large DB failures

**vs iter3_precision_synthesis_enhanced** (consistent 77.8%):
- ✅ Maintain ALL column coverage + precision features
- ➕ Add context overflow prevention
- ➕ Add semantic patterns + validation
- ➕ Add string matching + NULL/NaN guidance
- Expected: +3-5% from advanced features + large DB support

## Design Philosophy

**Cross-Pollination**: Combines only proven successful techniques from 3 top-ELO agents. No untested innovations - just intelligent combination of validated methods.

**Adaptive Intelligence**: Analysis depth scales with database complexity. Small databases get full iter9-style comprehensive analysis; large databases get essential information efficiently presented.

**Critical Information First**: Core schema, columns, relationships, and formats always preserved. Advanced features like semantic patterns are added for small/medium databases where context allows.

**Graceful Degradation**: Better to provide abbreviated enum listings than to overflow and fail completely. Functional output beats perfect output that doesn't fit.

**Evidence-Based Design**: Based on actual performance data showing iter16 prevents overflow, iter9 has best peak, iter3 is most consistent. Combines their strengths strategically.

**New Insights**: Adds string matching guidance and NULL/NaN detection based on error analysis showing these are primary failure points in app_store database (43.3% consensus errors).
