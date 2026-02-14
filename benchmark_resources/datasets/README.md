# BIRD Benchmark Datasets

This directory contains the official BIRD (BIg Bench for LaRge-scale Database Grounded Text-to-SQL Evaluation) benchmark datasets. All datasets have been validated against their original source files.

## Quick Reference

| Dataset | Questions | Databases | Status | Source |
|---------|-----------|-----------|--------|--------|
| **train** | 9,428 | 71 dirs (69 DBs + extras) | ✅ Validated | [BIRD Train](https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip) |
| **train-filtered** | 6,601 | 69 | ✅ Validated | [BIRD23 Filtered](https://huggingface.co/datasets/birdsql/bird23-train-filtered) |
| **dev** | 1,534 | 11 | ✅ Validated | [BIRD Dev](https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip) |

### Dataset Selection Guide

- **train-filtered** (RECOMMENDED): 6,601 curated questions (70% of train), 100% usable, improved quality
- **train**: 9,428 questions, includes retail_world (known ground truth issues)
- **dev**: 1,534 questions for development testing, all databases working

## Dataset Details

### Train Dataset (`train/`)

**Source**: https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip

**Contents**:
```
train/
├── train.zip                          # Original archive (MD5: abf4e64a4d8fc246bb284c45e756abd3)
└── train/                             # Extracted contents
    ├── train.json                     # 9,428 questions (MD5: 37131456941dabe670c3f1c2fd07f008)
    ├── train_gold.sql                 # Ground truth SQL (MD5: 8e771ef01e7f337de2860f177d8048ed)
    ├── train_tables.json              # Schema information (MD5: 3d4433984ba7d80627f3448e89f304d6)
    ├── train_databases.zip            # Database archive (8.7 GB)
    └── train_databases/               # 71 directories (69 actual databases)
```

**Question Statistics**:
- Total questions: 9,428
- Unique databases: 69 (`db_id` field)
- Directory count: 71 (includes `mondial_geo.sqlite` file and `train_tables.json`)
- Known issues: `retail_world` (373 questions) has high ground truth error rate

**Database List** (69 databases):
address, airline, app_store, authors, beer_factory, bike_share_1, book_publishing_company, books, car_retails, cars, chicago_crime, citeseer, codebase_comments, coinmarketcap, college_completion, computer_student, cookbook, craftbeer, cs_semester, disney, donor, european_football_1, food_inspection, food_inspection_2, genes, hockey, human_resources, ice_hockey_draft, image_and_language, language_corpus, law_episode, legislator, mental_health_survey, menu, mondial_geo, movie, movie_3, movie_platform, movielens, movies_4, music_platform_2, music_tracker, olympics, professional_basketball, public_review_platform, regional_sales, restaurant, retail_complains, retail_world, retails, sales, sales_in_weather, shakespeare, shipping, shooting, simpson_episodes, soccer_2016, social_media, software_company, student_loan, superstore, synthea, talkingdata, trains, university, video_games, works_cycles, world, world_development_indicators

**Validation Status**: ✅ All core files match original zip archive

---

### Train-Filtered Dataset (`train-filtered/`)

**Source**: https://huggingface.co/datasets/birdsql/bird23-train-filtered

**Contents**:
```
train-filtered/
└── train_filtered.json                # 6,601 questions (MD5: 96afa2c5be5f5154ce7299365a59f557)
```

**Description**: BIRD23 curated subset with improved data quality. Filtered to ~70% of original training set while maintaining equivalent performance through rigorous validation for schema consistency and answer faithfulness.

**Question Statistics**:
- Total questions: 6,601 (70% of original 9,428)
- Unique databases: 69
- Usability: 100% (all questions fully validated)

**Performance Metrics** (from source):
- Mini-Dev: 46.0% (vs. 45.4% on full training set)
- Dev set: 50.0% (vs. 50.46% on full set)

**Benefits**:
- Higher quality: Validated for schema consistency and answer faithfulness
- Same coverage: All 69 databases included
- Better efficiency: 30% fewer questions with maintained performance
- No known issues: Problematic examples removed

**Format**: JSON array with objects containing:
- `db_id`: Database identifier
- `question`: Natural language query (24-262 characters)
- `evidence`: Expert-annotated knowledge context (0-673 characters)
- `SQL`: Ground truth SQL query (23-804 characters)

**Database Access**: Uses same databases as train dataset (located in `train/train/train_databases/`)

**Validation Status**: ✅ Matches BIRD23 specifications (6,601 questions, 69 databases)

---

### Dev Dataset (`dev/`)

**Source**: https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip

**Contents**:
```
dev/
├── dev.zip                            # Original archive (MD5: 04b4af221c9186361f09b16abfd917ec)
└── dev_20240627/                      # Extracted contents (dated release)
    ├── dev.json                       # 1,534 questions (MD5: af311ef1348945573b1e49e41309edb1)
    ├── dev.sql                        # Ground truth SQL (MD5: a6b09fd2b42ac18fc59bd581a9edd7de)
    ├── dev_gold.sql                   # Symlink to dev.sql
    ├── dev_tables.json                # Schema information (MD5: 85856b6b9dfa46e836d511cc2567dc34)
    ├── dev_tied_append.json           # Evaluation metadata (MD5: 72b5ba4cbfc34fb01a3bba9bb68fd869)
    ├── dev_databases.zip              # Database archive (330 MB)
    └── dev_databases/                 # 11 databases
```

**Question Statistics**:
- Total questions: 1,534
- Databases: 11 (all working)
- Difficulty levels: Simple, Moderate, Challenging

**Database List** (11 databases):
1. california_schools - Education data
2. card_games - Gaming statistics
3. codebase_community - Software development
4. debit_card_specializing - Financial transactions
5. european_football_2 - Sports data
6. financial - Financial records
7. formula_1 - Racing data
8. student_club - Student activities
9. superhero - Comic book data
10. thrombosis_prediction - Medical data
11. toxicology - Chemical/toxicology data

**Validation Status**: ✅ All files match original zip archive

---

## File Format Specifications

### Question Files (*.json)

**Structure**:
```json
[
  {
    "db_id": "database_name",
    "question": "Natural language question",
    "evidence": "Additional context and hints",
    "SQL": "SELECT * FROM table WHERE condition"
  }
]
```

### Ground Truth SQL Files (*.sql)

**Format**: Tab-separated values
```
SELECT * FROM table WHERE condition	database_name
```

### Schema Files (*_tables.json)

Contains database schema information including table definitions, column types, primary keys, and foreign keys.

### Tied Append Files (*_tied_append.json)

Additional metadata for evaluation, including tied results handling.

---

## Validation & Reproducibility

### Checksums (MD5)

**Original Archives**:
- `train.zip`: `abf4e64a4d8fc246bb284c45e756abd3`
- `dev.zip`: `04b4af221c9186361f09b16abfd917ec`
- `train_filtered.json`: `96afa2c5be5f5154ce7299365a59f557`

**Train Dataset Files**:
- `train.json`: `37131456941dabe670c3f1c2fd07f008`
- `train_gold.sql`: `8e771ef01e7f337de2860f177d8048ed`
- `train_tables.json`: `3d4433984ba7d80627f3448e89f304d6`

**Dev Dataset Files**:
- `dev.json`: `af311ef1348945573b1e49e41309edb1`
- `dev.sql`: `a6b09fd2b42ac18fc59bd581a9edd7de`
- `dev_tables.json`: `85856b6b9dfa46e836d511cc2567dc34`
- `dev_tied_append.json`: `72b5ba4cbfc34fb01a3bba9bb68fd869`

### Quick Validation Scripts

Automated scripts to validate dataset integrity:

```bash
# Validate train and train-filtered datasets
cd benchmark_resources/datasets/
./validate_train.sh

# Validate dev dataset
./validate_dev.sh
```

These scripts:
- Check if files exist
- Verify MD5 checksums against expected values
- Provide clear success/error messages
- Work on both macOS and Linux

### Manual Verification Commands

For manual verification or troubleshooting:

```bash
# Verify train dataset
md5 benchmark_resources/datasets/train/train.zip
unzip -p benchmark_resources/datasets/train/train.zip train/train.json | md5
md5 benchmark_resources/datasets/train/train/train.json

# Verify dev dataset
md5 benchmark_resources/datasets/dev/dev.zip
unzip -p benchmark_resources/datasets/dev/dev.zip dev_20240627/dev.json | md5
md5 benchmark_resources/datasets/dev/dev_20240627/dev.json

# Verify train-filtered dataset
md5 benchmark_resources/datasets/train-filtered/train_filtered.json

# Count questions
python3 -c "import json; print(len(json.load(open('benchmark_resources/datasets/train/train/train.json'))))"
python3 -c "import json; print(len(json.load(open('benchmark_resources/datasets/train-filtered/train_filtered.json'))))"
python3 -c "import json; print(len(json.load(open('benchmark_resources/datasets/dev/dev_20240627/dev.json'))))"
```

### Reproduction Steps

#### Train Dataset
1. Download: `wget https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip`
2. Verify: `md5 train.zip` (should match: `abf4e64a4d8fc246bb284c45e756abd3`)
3. Extract: `unzip train.zip`
4. Extract databases: `cd train && unzip train_databases.zip`

#### Dev Dataset
1. Download: `wget https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip`
2. Verify: `md5 dev.zip` (should match: `04b4af221c9186361f09b16abfd917ec`)
3. Extract: `unzip dev.zip`
4. Extract databases: `cd dev_20240627 && unzip dev_databases.zip`

#### Train-Filtered Dataset
1. Download from HuggingFace: https://huggingface.co/datasets/birdsql/bird23-train-filtered
2. Save as `train_filtered.json`
3. Verify: `md5 train_filtered.json` (should match: `96afa2c5be5f5154ce7299365a59f557`)

---

## Usage Examples

### For RoboPhD Research System

```bash
# Use train-filtered (recommended)
python RoboPhD/researcher.py --num-iterations 10

# Use train dataset
python RoboPhD/researcher.py --num-iterations 10 --config '{"dataset": "train"}'

# Use dev dataset
python RoboPhD/researcher.py --dev-eval --config '{"initial_agents": ["your_agent"]}'
```

### For Direct Evaluation

```python
import json

# Load questions
train_data = json.load(open('benchmark_resources/datasets/train/train/train.json'))
train_filtered_data = json.load(open('benchmark_resources/datasets/train-filtered/train_filtered.json'))
dev_data = json.load(open('benchmark_resources/datasets/dev/dev_20240627/dev.json'))

# Access database
db_path = 'benchmark_resources/datasets/train/train/train_databases/database_name/database_name.sqlite'
```

---

## Space-Optimized Setup for Constrained Environments

The train dataset requires ~30 GB uncompressed. For environments with limited disk space (e.g., 26 GB available), use **selective extraction with compression**.

### Automated Setup Script (Recommended)

```bash
cd benchmark_resources/datasets/
./setup_train_compressed.sh
```

This script automatically:
1. Downloads and extracts train.zip (deletes zip immediately to save space)
2. Selectively extracts the 5 largest databases one at a time
3. Compresses each large database immediately (3.4x compression)
4. Extracts all 64 remaining small databases
5. Deletes train_databases.zip

**Space requirements**:
- Peak during setup: 22.3 GB
- Final base storage: 13.6 GB
- Peak during operation: 19.9 GB (when processing 1 large DB)
- Minimum space needed: 23 GB available

### Manual Setup (Alternative)

If you prefer manual control:

```bash
cd benchmark_resources/datasets/train/

# 1. Download and extract train.zip
wget https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip
md5 train.zip  # Verify: abf4e64a4d8fc246bb284c45e756abd3
unzip train.zip
rm train.zip   # Delete immediately (saves 8.7 GB)

cd train/

# 2. Selectively extract and compress large databases
for db in bike_share_1 donor codebase_comments movie_platform world_development_indicators; do
    echo "Processing $db..."
    unzip train_databases.zip "$db/*" -d train_databases/
    cd train_databases
    tar -czf "${db}.tar.gz" "$db"
    rm -rf "$db"
    cd ..
done

# 3. Extract remaining small databases (excludes already-processed large ones)
unzip train_databases.zip -d train_databases/ \
    -x "bike_share_1/*" "donor/*" "codebase_comments/*" \
    "movie_platform/*" "world_development_indicators/*"

# 4. Delete train_databases.zip
rm train_databases.zip  # Saves 8.7 GB
```

### Compressed Databases (5 total)

| Database | Uncompressed | Compressed | Questions | Notes |
|----------|--------------|------------|-----------|-------|
| bike_share_1 | 4.9 GB | 1.4 GB | 113 | Largest database |
| donor | 4.2 GB | 1.2 GB | 160 | Had extraction issues historically |
| codebase_comments | 4.2 GB | 1.2 GB | 124 | |
| movie_platform | 4.0 GB | 1.2 GB | 167 | |
| world_development_indicators | 2.3 GB | 0.7 GB | 157 | |
| **Total** | **17.6 GB** | **5.7 GB** | **721 (7.6%)** | **Saves 11.9 GB** |

### Uncompressed Databases (64 total)

All remaining databases (6.66 GB total, 8,707 questions = 92.4%) are kept uncompressed for immediate access.

### On-Demand Decompression

When you need to use a compressed database:

```bash
cd benchmark_resources/datasets/train/train/train_databases/
tar -xzf bike_share_1.tar.gz
# Use the database...
# Optionally re-compress when done:
# tar -czf bike_share_1.tar.gz bike_share_1 && rm -rf bike_share_1
```

**Integration with RoboPhD**: Configure sampling to use at most 1 compressed database per iteration to minimize decompression overhead. See CLAUDE.md for details on implementation.

---

## Known Issues & Notes

### Train Dataset
- **retail_world**: 373 questions with high ground truth error rate (automatically excluded in RoboPhD when using 'train')
- **language_corpus**: 2.2 GB database can cause timeouts
- **donor**: Originally 1.6 GB corrupted, manually extracted to 4.2 GB working version
- Some databases require manual extraction from nested `train_databases.zip`

### Train-Filtered Dataset
- ✅ All known issues from train dataset have been resolved
- ✅ 100% of questions are usable
- **Database files**: Uses same databases as train (located in `train/train/train_databases/`)

### Dev Dataset
- ✅ All databases working
- ✅ No known issues

### General
- **Database locks**: Can occur during concurrent ground truth evaluation
  - **Solution**: Run `python RoboPhD/tools/precompute_ground_truth.py` before research runs
- **GROUP_CONCAT**: Can create very large results (truncation implemented in evaluation)
- **Extraction**: Some databases have nested zip structure requiring manual extraction

---

## Dataset Statistics Summary

| Metric | Train | Train-Filtered | Dev |
|--------|-------|----------------|-----|
| Total Questions | 9,428 | 6,601 | 1,534 |
| Usable Questions | ~9,055 (96%) | 6,601 (100%) | 1,534 (100%) |
| Databases (db_id) | 69 | 69 | 11 |
| Database Dirs | 71 | 69 | 11 |
| Known Issues | retail_world | None | None |
| Quality | Standard | Enhanced | Standard |
| Archive Size | 8.3 GB | 2.7 MB | 330 MB |
| Database Size | 8.7 GB | (uses train DBs) | 330 MB |

---

## References

- **BIRD Benchmark**: https://bird-bench.github.io/
- **BIRD GitHub**: https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/bird
- **BIRD23 Train-Filtered**: https://huggingface.co/datasets/birdsql/bird23-train-filtered
- **Paper**: "Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQL Evaluation"

---

*Last Updated: 2025-11-07*
*Validated By: Andrew Borthwick, PhD*
