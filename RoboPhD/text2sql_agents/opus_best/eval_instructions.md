# SQL Generation Instructions

You will receive a comprehensive database analysis followed by a question and optional evidence. The analysis may be abbreviated for large databases to fit context limits. Generate accurate SQL queries by following these rules carefully.

## Core Requirements

Generate clean, executable SQL:
- No markdown formatting or code blocks
- No comments or explanatory text
- Only the SQL statement
- Use SQLite syntax

## CRITICAL: Column Selection and Order Rules

**RULE 1: Return ONLY the columns explicitly requested**

- Read the question carefully to identify what is being asked for
- Do NOT include columns used only for filtering (WHERE clause)
- Do NOT include columns used only for sorting (ORDER BY clause)
- Do NOT include columns used only for grouping (GROUP BY clause)

Examples:
- ❌ WRONG: "Which movie has the best rating?" → `SELECT Title, Rating FROM movie ORDER BY Rating DESC LIMIT 1` (2 columns)
- ✅ CORRECT: "Which movie has the best rating?" → `SELECT Title FROM movie ORDER BY Rating DESC LIMIT 1` (1 column)

- ❌ WRONG: "List product name" → `SELECT ProductID, Name FROM product` (2 columns)
- ✅ CORRECT: "List product name" → `SELECT Name FROM product` (1 column)

**Compound questions (asking for multiple things):**
- Question: "Which professor taught the most courses and what is the position?"
- This asks for TWO things: the professor (p_id) AND the position
- ✅ CORRECT: `SELECT p_id, hasPosition FROM ...` (return BOTH)
- ❌ WRONG: `SELECT hasPosition FROM ...` (missing p_id!)

- Question: "Which course has more teachers, course no.16 or course no.18?"
- This asks for ONE thing: which course (just the course_id)
- ✅ CORRECT: `SELECT CASE WHEN ... THEN 16 WHEN ... THEN 18 END` (return just course_id)
- ❌ WRONG: `SELECT course_id, COUNT(*) FROM ... LIMIT 1` (includes count that wasn't requested!)

**Multi-part questions (count AND list):**
- Question: "How many checks were issued? Please list their check numbers."
- This asks for TWO things: count AND list
- ✅ CORRECT: Return just the list (one row per item) - count is implicit from row count
- ❌ WRONG: Return COUNT and GROUP_CONCAT in single row
- Example: `SELECT checkNumber FROM payments ...` (let row count = "how many")

**Column order in compound questions:**
When question asks for multiple things, return them in the ORDER mentioned:
- Question: "Describe the specific description and case locations"
- Order: description FIRST, then locations
- ✅ CORRECT: `SELECT secondary_description, latitude, longitude FROM ...`
- ❌ WRONG: `SELECT latitude, longitude, secondary_description FROM ...` (wrong order!)

**RULE 2: When evidence lists multiple columns, return ALL of them in EXACT order**

**ONLY when evidence explicitly says "X refers to A, B, C":**
- Return ALL the columns listed
- Return them in the EXACT order specified in evidence
- Follow evidence order, not question phrasing

**If evidence does NOT list columns, follow the question literally:**
- Question asks for "address" → Return only the address column
- Question asks for "name" → Return only the name column
- Do NOT infer additional columns unless evidence specifies them

Examples:
- Evidence: "full address refers to street_num, street_name, city"
  - ✅ CORRECT: `SELECT street_num, street_name, city FROM location`
  - ❌ WRONG: `SELECT street_name, street_num, city FROM location` (wrong order!)
  - ❌ WRONG: `SELECT street_name FROM location` (missing columns!)

- Evidence: "the division refers to Div, name"
  - ✅ CORRECT: `SELECT Div, name FROM divisions`
  - ❌ WRONG: `SELECT name, Div FROM divisions` (wrong order!)

- NO evidence about "address" components:
  - Question: "What is the address?"
  - ✅ CORRECT: `SELECT address FROM table`
  - ❌ WRONG: `SELECT address, address2, city, state FROM table` (extra columns!)

**RULE 3: Use LIMIT 1 for ranking/superlative questions**

**ALWAYS use LIMIT 1 for these keywords:**
- "best", "worst", "top", "first"
- "highest", "lowest", "most", "fewest", "least"
- "maximum", "minimum", "max", "min"
- "largest", "smallest", "biggest", "oldest", "youngest"

**Do NOT use LIMIT 1 for:**
- Simple filtering questions (even if singular noun)
- Comparison questions asking "which of A or B" without superlative

Examples:
- ✅ CORRECT: "Which movie has the best rating?" → `... ORDER BY rating DESC LIMIT 1` (superlative)
- ✅ CORRECT: "Customer with highest number of inhabitants" → `... ORDER BY inhabitants DESC LIMIT 1` (superlative)
- ✅ CORRECT: "Which professor taught the most courses?" → `... ORDER BY COUNT(*) DESC LIMIT 1` (superlative)
- ❌ WRONG: "What division was the match in?" → don't use LIMIT 1 (filtering, not superlative)
- ❌ WRONG: "Which course has more teachers, 16 or 18?" → don't use LIMIT 1 in subquery (comparison, not superlative)

**Tie Handling**: For "best/worst/highest/lowest" questions, prefer `ORDER BY...LIMIT 1` over subquery approach.

## CRITICAL: Evidence Field Interpretation

**The evidence field is CRITICAL - follow it exactly!**

### Evidence Formula Patterns

**Pattern 1: "X refers to COLUMN"** - Return the column value
- Evidence: "winner refers to FTR"
- Example: `SELECT FTR FROM matchs ...`

**Pattern 2: "X refers to COLUMN = 'VALUE'"** - This is a FILTER condition, not the output
- Evidence: "winner refers to FTR = 'A'"
- Interpretation: Filter where FTR = 'A', but return the actual winner (team name), not 'A'
- Example: `SELECT CASE WHEN FTR = 'H' THEN HomeTeam WHEN FTR = 'A' THEN AwayTeam END ...`
- ❌ WRONG: `SELECT FTR FROM matchs WHERE FTR = 'A'` (returns 'A', not team name)

**Pattern 3: "X refers to COLUMN1, COLUMN2, COLUMN3"** - Return ALL columns in that order
- Evidence: "full address refers to street_num, street_name, city"
- Example: `SELECT street_num, street_name, city FROM location`
- ❌ WRONG: Return only some columns or change order

**Pattern 4: Evidence with negation**
- Evidence: "minus exclusions for X = 'value'"
- Interpretation: Use X = 'value' AS STATED (don't negate it)
- Trust the evidence definition, not English phrasing

**Pattern 5: "calculation hint"** - Use suggested approach
- Evidence: "first hired = MIN(HireDate)"
- Example: `SELECT MIN(HireDate) FROM employee ...`

**Pattern 6: Multiple mappings in one evidence**
- Evidence: "winner refers to FTR; FTR = 'H' means home team; FTR = 'A' means away team"
- Parse each part: Use FTR column, understand 'H'/'A' values

**Pattern 7: Ambiguous location/attribute references**
- Question: "UK Sales Rep who have highest credit limit"
- Evidence: "UK is a country; Sales Rep is a job title"
- Careful: Does "UK" modify "Sales Rep" (employee location) or "credit limit" (customer location)?
- Check question context: "credit limit" belongs to customers, not employees
- ✅ CORRECT: Filter customers by country='UK', then find their sales reps
- ❌ WRONG: Filter employees by office.country='UK'

### Evidence Complexity Handling

**Simple evidence**:
- "Accountant is a job title" → `WHERE JobTitle = 'Accountant'`

**Complex evidence with multiple parts**:
- "premium customer refers to Customer_Grade = 1; total amount = SUM(Amount)"
- Apply both: Filter by Customer_Grade = 1, then SUM(Amount)

**Implicit column name guidance**:
- "the 1st row" → Use LIMIT 1 with appropriate ORDER BY
- "the last row" → Use ORDER BY DESC LIMIT 1

## CRITICAL: String Value Matching

**Always check Section 6 (ENUM VALUE REFERENCE) for exact string values!**

The database analysis includes **Section 6: ENUM VALUE REFERENCE** showing ALL possible values for columns with limited distinct values.

**Workflow:**
1. Read question and identify key values (e.g., "Napa Valley", "Premier League", "Weather")
2. Find the column in **Section 6: ENUM VALUE REFERENCE**
3. Look for matching value in the enum list
4. Copy the EXACT value (case-sensitive) from the list

**Case Sensitivity:**
- SQLite string comparison is case-sensitive
- If question says "Napa Valley" but enum shows "napa valley", use "napa valley"
- Never guess - always check Section 6

Examples:
- Question mentions "Napa Valley" → Check Section 6
- Section 6 shows: ['napa valley', 'sonoma', 'central coast'] (lowercase)
- ✅ CORRECT: `region = 'napa valley'`
- ❌ WRONG: `region = 'Napa Valley'` (won't match)

### NEW: LIKE vs = Matching Rules

**Check Section 8 (FORMAT DETECTION SUMMARY) for string matching guidance!**

**Rule 1: Default to = for exact matching**
- Most columns require exact matches
- Use = unless Section 8 explicitly says to use LIKE

**Rule 2: Use LIKE when Section 8 indicates multi-value columns**
- Section 8 may note: "Multi-value column (comma-separated)" or "Multi-value column (pipe-separated)"
- Evidence: "Genre = 'Weather'" but table has "Genres" (plural)
- If Section 6 shows values like: ['Action, Drama', 'Comedy, Romance']
- Then use LIKE: `WHERE Genres LIKE '%Weather%'`

**Rule 3: Column name plurality warning**
- Evidence says "Genre = 'X'" (singular) but table has "Genres" (plural)
- This suggests multi-value column - check Section 6 for actual values
- If values contain delimiters (comma, pipe, semicolon): Use LIKE
- If values are single words: Use =

Examples:
- Evidence: "weather app refers to Genre = 'Weather'"
- Table has column: "Genres" (plural!)
- Section 6 shows: ['Art & Design', 'Auto & Vehicles', 'Beauty', ...]
- Notice values contain '&' and spaces - likely multi-category
- ✅ CORRECT: `WHERE Genres LIKE '%Weather%'`
- ❌ WRONG: `WHERE Genres = 'Weather'` (exact match won't work)

## CRITICAL: NULL and NaN Value Handling

**Check Section 8 (FORMAT DETECTION SUMMARY) for NULL/NaN patterns!**

Section 8 now includes explicit NULL/NaN detection for each column showing:
- NULL percentage (actual NULL values)
- String 'nan' count (if any)
- Empty string count (if any)
- Recommended handling approach

### Pattern 1: String 'nan' values

**Only use REPLACE() when performing NUMERIC operations on columns with 'nan':**

**If Section 8 shows string 'nan' count > 0 AND you're doing numeric operations (CAST, SUM, AVG):**
- Use: `CAST(REPLACE(column, 'nan', '0') AS REAL)` for aggregations
- Use: `CAST(REPLACE(column, 'nan', '0') AS REAL)` for comparisons (<, >, =)

**Do NOT use REPLACE() when:**
- Just filtering or joining (WHERE column = 'value')
- Column is already numeric (INTEGER, REAL types don't have string 'nan')
- Not performing mathematical operations

**Examples:**
- ✅ CORRECT: `AVG(CAST(REPLACE(Sentiment_Polarity, 'nan', '0') AS REAL))` (aggregation with 'nan')
- ✅ CORRECT: `CAST(Sentiment_Polarity AS REAL) < -0.5` (no 'nan' in this column per Section 8)
- ❌ WRONG: `WHERE REPLACE(column, 'nan', '0') = 'value'` (not numeric operation!)

**Check Section 8 first:**
- If column NOT listed in Section 8 NULL/NaN Patterns: Don't add REPLACE()
- If column listed but 'nan' count = 0: Don't add REPLACE()
- If column listed with 'nan' count > 0 AND doing numeric ops: Add REPLACE()

### Pattern 2: High NULL percentage columns

**If Section 8 shows high NULL percentage (>50%):**
- Many rows will have NULL in this column
- Consider LEFT JOIN instead of INNER JOIN
- INNER JOIN will exclude all NULL rows
- COUNT(column) will exclude NULLs, COUNT(*) will include them

### Pattern 3: Aggregation with NULLs

**Remember:**
- COUNT(*) includes NULL rows
- COUNT(column) excludes NULL rows
- SUM/AVG automatically exclude NULL (treat as 0 for division purposes)

**Check Section 8 guidance:**
- May suggest: "Use COALESCE(column, 0) for calculations"
- May suggest: "Use LEFT JOIN to preserve NULL rows"

### Pattern 4: Do NOT over-filter for NULL/'nan'

**IMPORTANT: Only filter NULL/'nan' when question explicitly requires it**

- Question asks for "all reviews" → Include NULL and 'nan' reviews
- Question asks for "all apps" → Include apps even if some columns have NULL/'nan'
- Question asks for "apps with positive sentiment" → Filter Sentiment, but don't filter other columns

**Examples:**
- ✅ CORRECT: "List all reviews" → `SELECT Translated_Review FROM user_reviews` (includes NULL)
- ❌ WRONG: "List all reviews" → `SELECT Translated_Review FROM ... WHERE Translated_Review IS NOT NULL` (loses data!)

- ✅ CORRECT: "Apps with rating > 4" → `SELECT App FROM playstore WHERE Rating > 4` (simple filter)
- ❌ WRONG: "Apps with rating > 4" → `SELECT App FROM playstore WHERE Rating > 4 AND App != 'nan'` (unnecessary!)

## CRITICAL: Percentage Calculations

Percentage formulas in evidence require careful reading:

**Pattern 1: "percentage of X that are Y"**
- Numerator: COUNT(rows where Y is true AND X is true)
- Denominator: COUNT(rows where X is true)
- Example: "percentage of all tied games did the Sassuolo team play in"
  - Numerator: COUNT(games where tied AND Sassuolo played)
  - Denominator: COUNT(ALL tied games)
  - Formula: `DIVIDE(COUNT(FTR='D' AND team=Sassuolo), COUNT(FTR='D'))`

**Pattern 2: "percentage of Y that are X"** (reverse order!)
- Numerator: COUNT(rows where X is true AND Y is true)
- Denominator: COUNT(rows where Y is true)
- Example: "percentage of Sassuolo games that were tied"
  - Numerator: COUNT(games where Sassuolo played AND tied)
  - Denominator: COUNT(Sassuolo games)
  - Formula: `DIVIDE(COUNT(FTR='D' AND team=Sassuolo), COUNT(team=Sassuolo))`

**Key difference**: The word order matters! "percentage of A that are B" ≠ "percentage of B that are A"

**Always use CAST for percentage calculations:**
```sql
CAST(COUNT(CASE WHEN condition THEN 1 END) AS REAL) / COUNT(*) * 100
```

**Evidence may specify DIVIDE() or MULTIPLY():**
- Follow the evidence formula exactly
- DIVIDE(a, b) = a / b
- MULTIPLY(a, b) = a * b

## CRITICAL: Aggregation Detection

**Look for aggregation keywords:**
- "total", "sum", "average", "mean"
- "count", "how many", "number of"
- "most", "fewest", "least", "highest", "lowest", "best", "worst"
- "maximum", "minimum", "max", "min"

**Check the Evidence field:**
- Evidence often contains explicit aggregation hints
- Example: "fewest orders refer to MIN(Quantity)" → use MIN() or ORDER BY ... LIMIT 1
- Example: "total refers to SUM(amount)" → use SUM()
- Example: "percentage = DIVIDE(COUNT(...), COUNT(...))" → use COUNT()

**Distinguish between detail and summary:**
- "What is THE total gross..." → SUM(gross) for aggregated total
- "List the gross for each..." → Individual gross values, no SUM

**Multi-condition aggregations:**
- Question: "How many apps with rating 5 and installs >1M?"
- Use: `COUNT(CASE WHEN Rating=5 AND Installs>1000000 THEN 1 END)`
- Or: `COUNT(*) FROM ... WHERE Rating=5 AND Installs>1000000`

## CRITICAL: JOIN Selection

**Check Section 5 (RELATIONSHIP MAP) for cardinality hints**

Section 5 shows foreign key relationships with cardinality patterns:
- "1:1" → Each parent has exactly one child
- "1:N (avg 5.2 rows per parent)" → Each parent has multiple children
- "self-reference" → Table references itself

**Check Section 10/11 (CROSS-TABLE VALIDATION) if present**

For small/medium databases, Section 10 shows orphaned foreign key detection:
- "⚠️ 150/1000 foreign key values have no match in parent table"
- This means 15% of rows will be dropped by INNER JOIN
- Consider LEFT JOIN to preserve these rows

**JOIN Type Selection:**

**Use INNER JOIN when:**
- Question asks for items that MUST have matching records
- Example: "List products with categories" (products without category excluded)
- Low NULL percentage in FK column (<10%)
- No orphaned FK warning in Section 10

**Use LEFT JOIN when:**
- Question asks for all items, even without matches
- Example: "List all products and their categories if available"
- High NULL percentage in FK column (>30%)
- Section 10 shows orphaned FK warning
- Want to preserve rows even if no match exists

## CRITICAL: Multi-Table Queries

**When joining multiple tables:**

1. **Check Section 5 (RELATIONSHIP MAP) for all foreign keys**
   - Identifies which columns connect which tables
   - Shows relationship cardinality

2. **Build JOIN chain from question entities**
   - Identify main entity (usually in FROM clause)
   - Add JOINs for each related table needed
   - Use foreign keys from Section 5

3. **Watch for many-to-many relationships**
   - If two tables both have 1:N to a junction table
   - Need to join through the junction table
   - May need DISTINCT to avoid duplicate rows

4. **Check Section 10 (CROSS-TABLE VALIDATION) for warnings**
   - Orphaned FKs indicate broken relationships
   - May need LEFT JOIN instead of INNER JOIN

**Example:**
- Question: "List app names and their review sentiments"
- Section 5 shows: playstore.App → user_reviews.App (1:N relationship)
- Section 10 warns: Some apps have no reviews
- Use: `SELECT p.App, ur.Sentiment FROM playstore p LEFT JOIN user_reviews ur ON p.App = ur.App`

## CRITICAL: Common SQL Patterns

### Pattern 1: Top-N queries
```sql
SELECT column FROM table ORDER BY ranking_column DESC LIMIT N
```

### Pattern 2: Aggregation with grouping
```sql
SELECT category, COUNT(*) FROM table GROUP BY category
```

### Pattern 3: Filtering aggregates
```sql
SELECT category, COUNT(*) FROM table
WHERE condition
GROUP BY category
HAVING COUNT(*) > threshold
```

### Pattern 4: Subquery for filtering
```sql
SELECT * FROM table1 WHERE id IN (SELECT id FROM table2 WHERE condition)
```

### Pattern 5: Conditional aggregation
```sql
SELECT
  COUNT(CASE WHEN status = 'active' THEN 1 END) as active_count,
  COUNT(CASE WHEN status = 'inactive' THEN 1 END) as inactive_count
FROM table
```

### Pattern 6: Handling special values in aggregations
```sql
-- For string 'nan' values (check Section 8 first!)
SUM(CAST(REPLACE(column, 'nan', '0') AS REAL))

-- For NULL values
SUM(COALESCE(column, 0))
```

## Format-Specific Handling

**Check Section 7 (VALUE RANGE SUMMARY) and Section 8 (FORMAT DETECTION SUMMARY)**

### Currency Values
- Format: `$123,456.78`
- Parsing: `CAST(REPLACE(REPLACE(column, ',', ''), '$', '') AS REAL)`
- Sorting: Use parsed numeric value

### Percentages
- Format: `45.5%`
- Parsing: `CAST(REPLACE(column, '%', '') AS REAL)`
- For percentage calculations, may need to divide by 100

### Time Durations
- Format: `HH:MM:SS` or `H:MM:SS`
- Sorting: Direct comparison works (SQLite sorts time strings correctly)
- No special parsing needed for comparisons

### Dates
- Format: `YYYY-MM-DD`
- Comparison: Direct comparison works
- Date functions: Can use DATE() function if needed
- Substring for year: `substr(date_column, 1, 4)` or `substr(date_column, -4, 4)` (check evidence)

### Code Patterns
- Format: `E0`, `D1`, `B1` (letter + digits)
- Use exact string matching from Section 6 enum values

## Query Guidance from Database Analysis

**Always read Section 12 (QUERY GUIDANCE) for database-specific tips!**

Section 12 provides:
- Common query patterns for this specific database
- Evidence complexity warnings
- String matching recommendations
- NULL/NaN handling guidance
- Multi-table query tips

## CRITICAL: Evidence Formula Complexity Warnings

**Watch for these common evidence pitfalls:**

### Pitfall 1: Value Contradictions
- Evidence: "highest amount of -1 sentiment polarity score refers to MAX(Count(Sentiment_Polarity = 1.0))"
- Notice: Question says "-1" but evidence says "1.0"
- **Trust the evidence definition**, not question phrasing
- Use: `WHERE Sentiment_Polarity = 1.0` (as evidence states)

### Pitfall 2: Column Name Plurality
- Evidence: "Genre = 'Weather'" (singular)
- Table has: "Genres" column (plural)
- **Check Section 6**: Are values single words or multi-category?
- If multi-category: Use LIKE pattern matching
- If single words but column is plural: Still use exact =

### Pitfall 3: Multi-Condition Formulas
- Evidence: "percentage = DIVIDE(SUM(Genres = 'X' and Rating>4.5 and LastUpdated>'2018'), COUNT(Type='Free'))"
- Break into parts:
  1. Numerator conditions: Genres='X' AND Rating>4.5 AND LastUpdated>'2018'
  2. Denominator base: Type='Free'
  3. Apply formula carefully with proper CAST

### Pitfall 4: Implicit Filtering
- Evidence may specify filtering that's not obvious in question
- Example: Evidence adds "where Type='Free'" but question doesn't mention it
- **Apply all evidence conditions**, even if not in question

## Final Checklist

Before finalizing your SQL:

1. ✅ Checked Section 6 for exact enum values (case-sensitive)
2. ✅ Checked Section 8 for NULL/NaN patterns and string matching guidance
3. ✅ Checked Section 5 for foreign key relationships and cardinality
4. ✅ Checked Section 10 (if exists) for orphaned FK warnings
5. ✅ Used = for exact matching unless Section 8 says use LIKE
6. ✅ Handled string 'nan' with REPLACE() ONLY when doing numeric operations AND Section 8 shows 'nan'
7. ✅ Used appropriate JOIN type based on NULL% and orphaned FK warnings
8. ✅ Followed evidence formula exactly, especially for percentages
9. ✅ Returned ONLY requested columns (don't add extra columns unless evidence specifies)
10. ✅ Used LIMIT 1 for superlative questions only
11. ✅ Applied all evidence conditions, but don't over-filter for NULL/'nan'
12. ✅ Read Section 12 for database-specific query guidance

## Remember

- **The database analysis is your source of truth** - it shows actual values, not assumptions
- **Evidence overrides question phrasing** - if they conflict, trust evidence
- **Section 6 enum values are exact** - copy them character-for-character
- **Section 8 provides critical NULL/NaN and string matching guidance** - use it
- **Section 10 warns about data quality issues** - adjust JOINs accordingly
- **When in doubt, check the relevant section** - all information is organized by section number

Generate your SQL query now, following all rules above.
