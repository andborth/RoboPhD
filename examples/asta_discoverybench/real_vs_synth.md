# real vs synth distributions in DiscoveryBench

This file documents empirical structural differences between the two halves of the DiscoveryBench dataset (`real/` and `synth/`). The differences are large enough that an agent's score on a synth-only evaluation should not be assumed to predict its score on the real splits, even when both are sampled out-of-training-set.

All measurements below are taken on the actual loaded splits via `evaluator.load_real(...)` and `evaluator.load_synth(...)`. Numbers are from a sweep on 2026-05-04.

## Dataset coverage

`real/` has 264 scoreable samples spread across 14 distinct dataset names:

| Split | Datasets | n samples |
|-------|---|---:|
| `real/validation` (paper's "training") | evolution_freshwater_fish, immigration_offshoring_effect_on_employment, nls_bmi, nls_bmi_raw | 25 |
| `real/test` (canonical leaderboard) | archaeology, introduction_pathways_non-native_plants, meta_regression, meta_regression_raw, nls_incarceration, nls_raw, nls_ses, requirements_engineering_for_ML_enabled_systems, worldbank_education_gdp, worldbank_education_gdp_indicators | 239 |

Several real datasets are paired raw/processed variants of the same underlying problems (e.g. `meta_regression` and `meta_regression_raw`, `worldbank_education_gdp` and `worldbank_education_gdp_indicators`).

**The real validation and test datasets are disjoint** — none of the 4 validation datasets appear in the 10 test datasets. An agent that trains on `real/validation` and is evaluated on `real/test` is being asked to generalize to entirely unseen datasets.

`synth/` has 703 scoreable samples spread across 35 themes:

| Split | Themes | n samples |
|-------|---|---:|
| `synth/train` | 27 themes (urban-gardening, space-tourism, musical-therapy, photography, virtual-reality, fine-arts, literary-classics, theater-productions, ancient-architecture, solar-power, …) | 550 |
| `synth/dev` | 8 themes (ancient-civilizations, extreme-sports, jazz-music, cosplay-culture, ancient-history, sustainable-fashion, puzzle-solving-games, virtual-concerts) | 153 |
| `synth/test` | (unscoreable; gold withheld upstream) | 200 |

**`synth/train` and `synth/dev` themes are also disjoint.**

**Real and synth share zero dataset/theme names.**

## Surface metric comparison

Measured on the queries themselves:

| Metric | synth/train | synth/dev | real/test |
|---|---:|---:|---:|
| n samples | 550 | 153 | 239 |
| Distinct themes/datasets | 27 | 8 | 10 |
| Median query length (chars) | 183 | 187 | 103 |
| Mean query length | 192 | 199 | 119 |
| Median commas per query | 2 | 2 | 0 |
| Mean commas per query | 2.0 | 2.0 | 0.6 |
| Median columns per dataset | 33 | 33 | 57 |
| "Is there a relationship" opener | 61% | 61% | 0% |
| "What is the relationship" opener | 29% | 28% | 2% |
| Combined templated-opener share | 89% | 89% | 16% |

`synth/train` and `synth/dev` are statistically indistinguishable on every measured surface metric except theme vocabulary. `real/test` differs from synth on every metric.

## Sample ID conventions

Different splits use different sample ID schemes:

| Split | Format | Example |
|-------|---|---|
| `real/validation`, `real/test` | `<dataset>\|<query_idx>\|<variant>` | `worldbank_education_gdp\|4\|0` |
| `synth/train`, `synth/dev`, `synth/test` | `<theme>_<a>_<b>__m<N>__q<N>` | `ancient-civilizations_0_1__m2__q65` |

Synth IDs encode an explicit Cartesian product: each base question (`q{N}`) gets multiple metadata variants (`m{N}`) that paraphrase or extend it with different variable sets. Real IDs only encode dataset name and a query index.

## Sample queries

### 10 real/test queries (one per dataset)

```
[archaeology|8|0]
  In which millenium did amber had the highest value and in what time interval did it peak?

[introduction_pathways_non-native_plants|4|0]
  How do introduction pathways interact with minimum residence time in affecting the success of
  non-native plant species in Catalonia?

[meta_regression|8|0]
  In which domain is there a more balanced gender representation of authors, particularly in
  replication studies?

[meta_regression_raw|8|0]
  In which domain is there a more balanced gender representation of authors, particularly in
  replication studies?

[nls_incarceration|8|0]
  Does a record of having criminal history points to lower wealth accumulation?

[nls_raw|8|0]
  How does the median wealth of white individuals compare to that of black and Hispanic
  individuals from 1985 onwards?

[nls_ses|8|0]
  Between which two races is the factor of BA degree completion -0.9568 when compared to the other?

[requirements_engineering_for_ML_enabled_systems|8|0]
  Which two documentation formats are the least used for requirements in ML-enabled system
  projects, with 10.13% (95% CI [9.926, 10.333]) and 4.366% (95% CI [4.231, 4.501]) of
  respondents indicating so, respectively, after bootstrapping for statistical significance?

[worldbank_education_gdp|4|0]
  How do labor productivity and education levels relate to economic output, particularly in
  terms of export growth?

[worldbank_education_gdp_indicators|4|0]
  How do labor productivity and education levels relate to economic output, particularly in
  terms of export growth?
```

### 10 synth/dev queries (mostly from `ancient-civilizations`, varying `m{N}__q{N}` variants)

```
[ancient-civilizations_0_0__m0__q36]
  What is the relationship between conflict count in ancient Mesopotamian cities over the last
  decade, the number of ziggurats, and the conflict ratio, and how might this influence future
  outcomes?

[ancient-civilizations_0_1__m0__q43]
  Is there a relationship between the annual festival count, economic status, availability of
  essential resources, average trade volume, military presence, and the city names in the
  dataset related to ancient civilizations?

[ancient-civilizations_0_1__m1__q54]
  What are the key factors that influenced the estimated population sizes in ancient
  civilizations, based on the provided data columns?

[ancient-civilizations_0_1__m2__q65]
  Is there a relationship between the distance from the capital and the composite value derived
  from various factors such as the number of temples, architectural complexity, cultural event
  frequency, economic opportunities rating, types of goods traded, proximity to water,
  religious significance, military presence, is_military_stronghold indicator, number of trade
  routes, cultural influence score, and central market district size in ancient civilizations?

[ancient-civilizations_0_2__m0__q40]
  Is there a relationship between the deity worship score and essential resources available in
  ancient civilizations?

[ancient-civilizations_0_2__m1__q46]
  What is the relationship between the number of trade schools, the presence of multilingual
  traders, and economic recessions in determining the predicted annual trade volume in ancient
  civilizations?

[ancient-civilizations_0_2__m2__q50]
  Does the cultural development index in ancient civilizations display a statistically
  significant correlation with each of the following variables: government structure
  complexity, trade intensity ratio, educational reach, societal values towards science,
  external trade level, social cohesion index, external conflict presence? If so, what type of
  relationship is observed?

[ancient-civilizations_0_2__m3__q51]
  What is the relationship between the type of government in ancient civilizations and the
  distance to the nearest water source from a city, taking into account the percentage of
  arable land available?

[ancient-civilizations_0_2__m4__q58]
  Is there a relationship between the number of temples, types of goods traded, and proximity
  to water in ancient civilizations?

[ancient-civilizations_0_2__m5__q62]
  Is there a relationship between the characteristics of cities in ancient civilizations such
  as military presence, trade routes, religious significance, military stronghold status,
  cultural influence score, central market district size, and their founding years?
```

## Observable patterns

Reading the samples and the metric table together:

- **Real queries are concise human research questions.** Average ~120 chars, varied verbs (relate, compare, influence, interact with, predict), references to specific real-world entities (Catalonia, Sub-Saharan Africa, NLSY79). Many don't fit a uniform syntactic template (e.g. `archaeology|8|0` is a "in which X did Y peak" form, `nls_ses|8|0` quotes a specific coefficient `-0.9568`).
- **Synth queries are templated combinations.** ~190 chars, 89% start with "Is there a relationship" or "What is the relationship", commonly followed by a comma-separated list of variables. The metadata-variant suffix (`__m{N}__q{N}`) signals that each base query has been mechanically rephrased into multiple forms that swap variable sets.
- **Real datasets are wider** (~57 columns/median vs ~33 for synth) and named after actual research data sources. Synth files are uniformly named `data.csv`.
- **Real includes very-specific statistical queries** that synth doesn't: `requirements_engineering|8|0` quotes percentages with bootstrapped confidence intervals; `nls_ses|8|0` requires reproducing a specific coefficient. These are the kind of queries the original DiscoveryBench paper authors curated from real research.

## Implications for evaluation

- An agent's score on `synth/train` and `synth/dev` measures performance within the same generated distribution — only the theme vocabulary differs.
- The leaderboard score is taken on `real/test`, which is structurally different from anything in `synth/`.
- Cross-distribution generalization (synth → real) is not implied by within-synth generalization (synth/train → synth/dev).

This file is descriptive only; it doesn't recommend any particular agent design choice.
