# Protein GO Example

Evolve agents that predict Gene Ontology Molecular Function (GO-MFO) terms for protein sequences. The agent receives a sequence and tool callables for BLAST against SwissProt, UniProt entry lookup, GO ancestor traversal, sequence features, plus standard `llm()` and `embed()` callables.

Reported metric: canonical CAFA Fmax via [CAFA-evaluator](https://github.com/BioComputingUP/CAFA-evaluator) (Piovesan et al., 2024), on two held-out sets: ProteInfer's clustered-split test set (moderate-homology regime) and Price-149 (homology-resistant regime, from the CLEAN paper).

## Setup

```bash
# 1. Install core dependencies (from repo root)
pip install -r requirements.txt

# 2. Install task-specific dependencies (biopython, cafaeval, tfrecord)
pip install -r examples/protein_go/requirements.txt

# 3. Install DIAMOND (the BLAST engine used by the blast() callable)
conda install -c bioconda diamond
# Or: see https://github.com/bbuchfink/diamond/releases for binary downloads

# 4. Set API keys
export ANTHROPIC_API_KEY_FOR_ROBOPHD="your_key"   # for evolution (Claude Code)
export OPENAI_API_KEY="your_key"                   # for gpt-4.1-mini and embeddings

# 5. Download data and build splits (~20-30 min, one-time)
bash examples/protein_go/setup.sh
```

The setup script downloads:
- SwissProt 2022_01 (~90 MB compressed) for the BLAST database and annotations
- GO ontology (go-basic.obo, ~30 MB)
- ec2go mapping (EC-to-GO term correspondence, from geneontology.org)
- ProteInfer clustered-split TFRecords (~60 MB) for train/val/test accession lists
- Price-149 CSV (~30 KB) from the CLEAN repository

It then builds the DIAMOND index and four JSONL splits (train / validation / test / price149). Total disk: ~3 GB.

## Quick Start

Run from the RoboPhD repo root:

```bash
# Smoke test (2 iterations)
python examples/protein_go/main.py --num-iterations 2

# Full run (budget-limited, typically ~21 iterations)
python examples/protein_go/main.py

# With test-set evaluation after optimization
python examples/protein_go/main.py --eval-test-set
```

## Resume / Extend

```bash
# Resume from checkpoint
python examples/protein_go/main.py --resume ../robophd_runs/robophd/protein_go_20260420_120000

# Add 5 more iterations
python examples/protein_go/main.py --resume <dir> --extend 5

# Test-eval only (no further optimization)
python examples/protein_go/main.py --eval-only --resume <dir>
```

## Configuration

```bash
# Different engine
python examples/protein_go/main.py --engine gepa
python examples/protein_go/main.py --engine autoresearch

# Different solver model
python examples/protein_go/main.py --model gpt-4.1

# Custom engine config
python examples/protein_go/main.py --engine-config '{"include_evolution_rankings": false}'

# Adjust concurrency
python examples/protein_go/main.py --max-workers 4

# Skip the canonical CAFA Fmax computation (per-protein only is faster)
python examples/protein_go/main.py --eval-test-set --skip-cafa-fmax
```

## About the Benchmark

GO-MFO prediction is the canonical task of the [CAFA challenges](https://biofunctionprediction.org/cafa/), which have run since 2010. Given a protein sequence, the agent assigns Gene Ontology Molecular Function terms (e.g. "protein kinase activity", "ATP binding") with confidence scores. Predictions are scored by Fmax: the maximum F1 over confidence thresholds, with both predictions and ground truth propagated to MFO ancestors in the GO DAG.

### Two difficulty regimes

**Primary: ProteInfer clustered split** — train/validation/test are drawn from ProteInfer's (Sanderson et al., 2023) UniRef50-based clustered split of SwissProt. No sequence in validation or test has >50% identity to any sequence in training. This is substantially harder than a random split because BLAST-based homology transfer cannot simply look up near-identical homologs; agents have to reason about moderate-identity evidence. ProteInfer itself reports Fmax ≈ 0.68 on this split (CNN-based, no LLM); BLAST-only baselines hit roughly 0.55-0.60. This is the same split used by ProtNote, ProtEx, ProtGO, and subsequent GO-prediction papers.

**Secondary: Price-149** — 149 enzymes assembled by Price et al. and popularized by CLEAN (Yu et al., 2023) as a homology-resistant held-out set. These proteins were selected specifically because homology-based annotation methods fail on them. Labels are EC numbers; we map them to GO-MFO terms via the ec2go file published by the GO consortium. This is the harder test — CLEAN reports F1 ≈ 0.50 for contrastive-learning methods on Price-149, and BLASTp alone drops substantially below that.

Reporting both numbers tells a graduated story: how the agent performs on moderate-homology cases (ProteInfer test) and how it performs when homology fails (Price-149).

### Note on data vintages

The BLAST database and SwissProt entries are pinned to SwissProt release 2022_01. ProteInfer's clustered split was built against a similar-era SwissProt, and Price-149 was assembled in 2023. There is a small risk that Price-149 proteins have homologs added to SwissProt after 2022_01 that would make BLAST artificially more useful than it was to the CLEAN authors. This is minor but worth noting when comparing absolute Price-149 numbers to CLEAN's published figures.

## Architecture Notes

- **Single-file agent** — `seeds/baseline/agent.py` defines `predict(sequence, blast, uniprot, go_ancestors, sequence_features, llm, embed) -> dict`. Returns `{"GO:XXXXXXX": confidence_in_[0,1], ...}`.
- **DIAMOND caching** — Per-process in-memory cache of DIAMOND results keyed by sequence hash. First call per sequence runs DIAMOND live (~0.5s); subsequent calls within the same run are dict lookups.
- **Cost budget** — $0.10 per protein. Only `llm()` and `embed()` count; BLAST and UniProt lookups are free. Over-budget correct predictions are penalized to 0.9 (same as ARC-AGI / DocFinQA).
- **Two scoring paths** — Per-protein Fmax during evolution (smooth signal for Elo); batch CAFA Fmax for the headline test number. Correlated but not identical; both reported.

## Published Baselines (for context)

These are numbers from published methods on the same / similar benchmarks, useful as anchor points for interpreting your evolved agent's performance. None of these methods use LLM reasoning.

| Method | Clustered-split Fmax (MFO) | Price-149 F1 |
|---|---|---|
| BLAST alone | ~0.55-0.60 | ~0.15-0.30 |
| ProteInfer (CNN) | 0.68 | not reported |
| CLEAN (EC task, comparable regime) | — | 0.50 |
| PhiGnet (GNN + structure) | ~0.80 | — |

If an evolved agent reaches 0.70+ on the clustered split and 0.40+ on Price-149, it's competitive with specialized deep-learning methods despite being a general-purpose optimization agent with no bioinformatics training.
