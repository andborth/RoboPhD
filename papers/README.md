# BIRD Benchmark Top Methods - Research Papers

This directory contains research papers for top system methods achieving the highest execution accuracy scores on the [BIRD benchmark](https://bird-bench.github.io/).

## Top Systems by Execution Accuracy (Excluding Human Performance)

| Rank | Method | EX Score | Paper Available | Paper Location |
|------|--------|----------|----------------|----------------|
| 1 | LongData-SQL | 77.53% | ❌ | - |
| 2 | AskData + GPT-4o | 77.14% | ✅ | `askdata_gpt4o/Shkapenyuk_2025_AskData.pdf` |
| 3 | CHASE-SQL + Gemini | 76.02% | ✅ | `chase_sql_gemini/Pourreza_2024_CHASE_SQL.pdf` |
| 4 | TCDataAgent-SQL | 75.74% | ❌ | - |
| 5 | Contextual-SQL | 75.63% | ❌ | - |
| 6 | XiYan-SQL | 75.63% | ✅ | `xiyan_sql/Liu_2024_XiYan_SQL.pdf` |
| 7 | CYAN-SQL | 75.35% | ❌ | - |
| 8 | CSC-SQL + XiYanSQL-QwenCoder-32B-2412 | 73.67% | ✅ | `csc_sql/Sheng_2025_CSC_SQL.pdf` |
| 9 | ExSL + granite-34b-code | 73.17% | ❌ | - |
| 10 | Reasoning-SQL 14B | 72.78% | ✅ | `reasoning_sql_14b/Pourreza_2025_Reasoning_SQL.pdf` |
| 11 | OpenSearch-SQL, v2 + GPT-4o | 72.28% | ✅ | `opensearch_sql_gpt4o/Xie_2025_OpenSearch_SQL.pdf` |
| 12 | GenaSQL | 72.28% | ✅ | `genasql/Donder_2025_GenaSQL.pdf` |
| 13 | JT-SQL + XiYanSQL-QwenCoder-32B-2412 | 72.11% | ❌ | - |
| 14 | OmniSQL-32B | 72.05% | ✅ | `omnisql_32b/Li_2025_OmniSQL.pdf` |
| 15 | Distillery + GPT-4o | 71.83% | ✅ | `distillery_gpt4o/Maamari_2024_Distillery.pdf` |
| 16 | CSC-SQL + Qwen2.5-Coder-7B-Instruct | 71.72% | ✅ | `csc_sql/Sheng_2025_CSC_SQL.pdf` |
| 17 | CHESS_IR+CG+UT (Stanford) | 71.10% | ✅ | `chess_stanford/Talaei_2024_CHESS.pdf` |

## Paper Details

### Available Papers (11 papers)

1. **AskData + GPT-4o** (77.14% EX)
   - Paper: Shkapenyuk et al. '25
   - arXiv: https://arxiv.org/abs/2505.19988

2. **CHASE-SQL + Gemini** (76.02% EX)
   - Paper: Pourreza et al. '24
   - arXiv: https://arxiv.org/abs/2410.01943

3. **XiYan-SQL** (75.63% EX)
   - Paper: Liu et al. '24
   - arXiv: https://arxiv.org/abs/2507.04701

4. **CSC-SQL** (73.67% / 71.72% EX)
   - Paper: Sheng et al. '25
   - arXiv: https://arxiv.org/abs/2505.13271
   - Note: Same paper covers both XiYanSQL-QwenCoder-32B-2412 (73.67%) and Qwen2.5-Coder-7B-Instruct (71.72%) variants

5. **Reasoning-SQL 14B** (72.78% EX)
   - Paper: Pourreza et al. '25
   - arXiv: https://arxiv.org/abs/2503.23157

6. **OpenSearch-SQL, v2 + GPT-4o** (72.28% EX)
   - Paper: Xie et al. '25
   - arXiv: https://arxiv.org/abs/2502.14913

7. **OmniSQL-32B** (72.05% EX)
   - Paper: Li et al. '25
   - arXiv: https://arxiv.org/abs/2503.02240

8. **Distillery + GPT-4o** (71.83% EX)
   - Paper: Maamari et al. '24
   - arXiv: https://arxiv.org/abs/2408.07702

9. **GenaSQL** (72.28% EX)
   - Paper: Dönder et al. '25
   - Institution: Gena Co.
   - arXiv: https://arxiv.org/abs/2505.14174

10. **CHESS_IR+CG+UT** (71.10% EX)
    - Paper: Talaei et al. '24
    - Institution: Stanford
    - arXiv: https://arxiv.org/abs/2405.16755

### Methods Without Available Papers

- LongData-SQL, TCDataAgent-SQL, Contextual-SQL, CYAN-SQL, ExSL + granite-34b-code, JT-SQL + XiYanSQL-QwenCoder-32B-2412

Note: Some methods may have papers that are not yet publicly available or are proprietary industry solutions.

## Directory Structure

```
papers/
├── README.md
├── BIRD_Benchmark_2023.pdf
├── BIRD-Test_Submission_Guidelines.md
└── bird_methods/
    ├── askdata_gpt4o/
    │   └── Shkapenyuk_2025_AskData.pdf
    ├── chase_sql_gemini/
    │   └── Pourreza_2024_CHASE_SQL.pdf
    ├── chess_stanford/
    │   └── Talaei_2024_CHESS.pdf
    ├── csc_sql/
    │   └── Sheng_2025_CSC_SQL.pdf
    ├── distillery_gpt4o/
    │   └── Maamari_2024_Distillery.pdf
    ├── genasql/
    │   └── Donder_2025_GenaSQL.pdf
    ├── omnisql_32b/
    │   └── Li_2025_OmniSQL.pdf
    ├── opensearch_sql_gpt4o/
    │   └── Xie_2025_OpenSearch_SQL.pdf
    ├── reasoning_sql_14b/
    │   └── Pourreza_2025_Reasoning_SQL.pdf
    └── xiyan_sql/
        └── Liu_2024_XiYan_SQL.pdf
```

Data collected from: https://bird-bench.github.io/ (September 2025)