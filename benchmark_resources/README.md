# Benchmark Resources

This directory contains the BIRD benchmark dataset and related resources.

## Downloading the Dataset

Run the download script:

```bash
./download_bird.sh
```

Or download manually from [BIRD Benchmark](https://bird-bench.github.io/).

## Dataset Structure

After downloading:

```
benchmark_resources/
├── datasets/
│   ├── dev/
│   │   ├── dev.json                    # 1,534 questions
│   │   └── dev_databases/              # 11 databases
│   │       ├── california_schools/
│   │       ├── card_games/
│   │       └── ...
│   └── train/
│       ├── train.json                  # 9,428 questions
│       ├── train_filtered.json         # 6,601 curated questions
│       └── train_databases/            # 69 databases
│           ├── academic/
│           ├── activity_net/
│           └── ...
└── download_bird.sh
```

## Dataset Sizes

| Dataset | Questions | Databases | Disk Size |
|---------|-----------|-----------|-----------|
| dev | 1,534 | 11 | ~2GB |
| train | 9,428 | 62 | ~40GB |
| train-filtered | 6,601 | 69 | (subset of train) |

## Pre-computing Ground Truth

After downloading, pre-compute ground truth to prevent database lock errors:

```bash
# For train-filtered (default)
python RoboPhD/tools/precompute_ground_truth.py

# For dev
python RoboPhD/tools/precompute_ground_truth.py --dataset dev
```

## Dataset Details

### train-filtered
A curated subset of the training data with:
- 6,601 questions (70% of original 9,428)
- All 69 databases working correctly
- Problematic databases fixed via proper extraction
- 100% usable questions

### Known Issues
- `retail_world` database: High error rate in original train set (excluded)
- `language_corpus`: Large (2.2GB), may cause timeouts
- Some databases required manual extraction from nested archives

## Citation

If you use the BIRD benchmark, please cite:

```bibtex
@inproceedings{li2024bird,
  title={Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs},
  author={Li, Jinyang and others},
  booktitle={NeurIPS},
  year={2024}
}
```
