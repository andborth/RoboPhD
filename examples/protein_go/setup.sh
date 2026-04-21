#!/usr/bin/env bash
#
# Setup data for the protein GO-term prediction example (Molecular Function sub-ontology).
#
# Downloads (~2.5 GB total):
#   - SwissProt 2022_01 release tarball (~1.4 GB; UniProtKB curated subset, FASTA + flat file)
#   - Gene Ontology go-basic.obo (~30 MB)
#   - EC-to-GO mapping (ec2go, from geneontology.org; ~350 KB)
#   - ProteInfer clustered split TFRecords (~900 MB; UniRef50-based train/dev/test)
#   - Price-149 challenging held-out set (CLEAN paper, EC labels; ~60 KB)
#
# Builds:
#   - Parsed SwissProt entries pickle (accession -> name, organism, GO terms, ...)
#   - swissprot_train.fasta: SwissProt filtered to ProteInfer-train accessions
#   - DIAMOND index over swissprot_train.fasta (~3-5 min one-time)
#     (train-only DB matches the published ProteInfer / ProtNote / ProtEx / ProtGO
#      protocol and prevents test proteins from self-hitting the BLAST database)
#   - Validation / test JSONL splits aligned with ProteInfer dev / test
#   - Price-149 JSONL with EC labels mapped to GO-MFO terms via ec2go
#   - ESM-2 150M embedding cache over swissprot_train.fasta (~470MB .npy)
#     — backs the agent's esm_nearest() tool; precomputed once
#
# Total runtime: ~60-90 min on laptop CPU (dominated by step 9's ESM sweep);
# much faster on CUDA/MPS.
#
# Requirements:
#   - DIAMOND binary on PATH (conda install -c bioconda diamond)
#   - Python with biopython, tfrecord, fair-esm
#     (pip install -r examples/protein_go/requirements.txt)
#   - curl, gunzip
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$HERE/data"
mkdir -p "$DATA"

echo "============================================================"
echo "Protein GO (MFO) setup — ProteInfer clustered split"
echo "Data directory: $DATA"
echo "============================================================"

# ---------------------------------------------------------------------------
# 1. Check prerequisites
# ---------------------------------------------------------------------------

if ! command -v diamond >/dev/null 2>&1; then
    echo "ERROR: DIAMOND binary not found on PATH."
    echo "Install with: conda install -c bioconda diamond"
    exit 1
fi
echo "[1/9] DIAMOND found: $(diamond --version 2>&1 | head -1)"

if ! python3 -c "import Bio" 2>/dev/null; then
    echo "ERROR: biopython not installed."
    echo "Install with: pip install biopython"
    exit 1
fi
echo "[1/9] biopython found"

if ! python3 -c "import tfrecord" 2>/dev/null; then
    echo "ERROR: tfrecord package not installed."
    echo "Install with: pip install tfrecord"
    echo "(Lightweight TFRecord reader — does not require TensorFlow.)"
    exit 1
fi
echo "[1/9] tfrecord found"

if ! python3 -c "import esm" 2>/dev/null; then
    echo "ERROR: fair-esm not installed."
    echo "Install with: pip install -r examples/protein_go/requirements.txt"
    echo "(ESM-2 150M is needed by step 9's embedding-cache build and by the"
    echo " agent's esm_embed() / esm_nearest() tools at runtime.)"
    exit 1
fi
echo "[1/9] fair-esm found"


# ---------------------------------------------------------------------------
# 2. Download SwissProt
# ---------------------------------------------------------------------------

SWISSPROT_RELEASE="2022_01"
SWISSPROT_FASTA="$DATA/uniprot_sprot.fasta"
SWISSPROT_DAT="$DATA/uniprot_sprot.dat"

if [[ ! -f "$SWISSPROT_FASTA" ]]; then
    echo "[2/9] Downloading SwissProt $SWISSPROT_RELEASE release tarball (~1.4 GB)..."
    curl -fL --retry 3 \
        "https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-${SWISSPROT_RELEASE}/knowledgebase/uniprot_sprot-only${SWISSPROT_RELEASE}.tar.gz" \
        -o "$DATA/sprot.tar.gz"
    tar -xzf "$DATA/sprot.tar.gz" -C "$DATA" uniprot_sprot.fasta.gz uniprot_sprot.dat.gz
    rm -f "$DATA/sprot.tar.gz"
    gunzip "$DATA/uniprot_sprot.fasta.gz"
    gunzip "$DATA/uniprot_sprot.dat.gz"
else
    echo "[2/9] SwissProt already present, skipping"
fi


# ---------------------------------------------------------------------------
# 3. Parse SwissProt flat file to pickle dict
# ---------------------------------------------------------------------------

PARSED_PKL="$DATA/swissprot_entries.pkl"
if [[ ! -f "$PARSED_PKL" ]]; then
    echo "[3/9] Parsing SwissProt flat file (~3-5 min)..."
    python3 "$HERE/scripts/parse_swissprot.py" \
        --input "$SWISSPROT_DAT" \
        --output "$PARSED_PKL"
else
    echo "[3/9] Parsed SwissProt already present, skipping"
fi


# ---------------------------------------------------------------------------
# 4. Download GO ontology and ec2go mapping
# ---------------------------------------------------------------------------

GO_OBO="$DATA/go-basic.obo"
if [[ ! -f "$GO_OBO" ]]; then
    echo "[4/9] Downloading GO ontology..."
    curl -fL --retry 3 "http://purl.obolibrary.org/obo/go/go-basic.obo" -o "$GO_OBO"
else
    echo "[4/9] GO ontology already present, skipping"
fi

EC2GO="$DATA/ec2go.txt"
if [[ ! -f "$EC2GO" ]]; then
    echo "[4/9] Downloading ec2go mapping (EC number -> GO term)..."
    curl -fL --retry 3 "http://current.geneontology.org/ontology/external2go/ec2go" -o "$EC2GO"
else
    echo "[4/9] ec2go mapping already present, skipping"
fi


# ---------------------------------------------------------------------------
# 5. Download ProteInfer clustered-split TFRecords
# ---------------------------------------------------------------------------
#
# ProteInfer (Sanderson et al. 2023) constructs a UniRef50-based clustered split
# where no sequence in test/dev has >50% identity to any sequence in train.
# This is the canonical "hard homology" split used by published GO-prediction
# papers (ProteInfer, ProtNote, ProtEx, ProtGO). The TFRecord files are hosted
# on a public GCS bucket; we use curl to avoid requiring gcloud auth.

PROTEINFER_DIR="$DATA/proteinfer"
PROTEINFER_BASE="https://storage.googleapis.com/brain-genomics-public/research/proteins/proteinfer/datasets/swissprot/clustered"

if [[ ! -f "$PROTEINFER_DIR/test.tfrecord" ]]; then
    echo "[5/9] Downloading ProteInfer clustered-split TFRecords (~900 MB)..."
    mkdir -p "$PROTEINFER_DIR"
    for split in train dev test; do
        if [[ ! -f "$PROTEINFER_DIR/${split}.tfrecord" ]]; then
            curl -fL --retry 3 "${PROTEINFER_BASE}/${split}.tfrecord" \
                -o "$PROTEINFER_DIR/${split}.tfrecord"
        fi
    done
else
    echo "[5/9] ProteInfer split already downloaded, skipping"
fi


# ---------------------------------------------------------------------------
# 6. Download Price-149 (CLEAN paper's homology-resistant held-out set)
# ---------------------------------------------------------------------------
#
# Price-149 is 149 enzymes experimentally characterized by Price et al. and used
# as a "hard case" benchmark by CLEAN (Yu et al. 2023). Labels are EC numbers;
# we map them to GO-MFO via the ec2go file in step 4.

PRICE149_CSV="$DATA/price149_raw.csv"
if [[ ! -f "$PRICE149_CSV" ]]; then
    echo "[6/9] Downloading Price-149 from CLEAN repository..."
    # Note: the file has a .csv extension but is actually tab-separated.
    # build_splits.py reads it with delimiter="\t".
    curl -fL --retry 3 \
        "https://raw.githubusercontent.com/tttianhao/CLEAN/main/app/data/datasets/price.csv" \
        -o "$PRICE149_CSV"
else
    echo "[6/9] Price-149 already downloaded, skipping"
fi


# ---------------------------------------------------------------------------
# 7. Filter FASTA to ProteInfer train and build DIAMOND index
# ---------------------------------------------------------------------------
#
# The DIAMOND DB is built from ProteInfer's train accessions only, NOT the full
# SwissProt. This matches the published protocol and prevents test/dev proteins
# from self-hitting the BLAST DB during evaluation — the key defect that would
# otherwise trivialize the task via 100%-identity self-lookup.

SWISSPROT_TRAIN_FASTA="$DATA/swissprot_train.fasta"
DIAMOND_DB="$DATA/swissprot_train.dmnd"

if [[ ! -f "$SWISSPROT_TRAIN_FASTA" ]]; then
    echo "[7/9] Filtering SwissProt FASTA to ProteInfer train subset (~1 min)..."
    python3 "$HERE/scripts/filter_fasta_by_accessions.py" \
        --input "$SWISSPROT_FASTA" \
        --tfrecord "$PROTEINFER_DIR/train.tfrecord" \
        --output "$SWISSPROT_TRAIN_FASTA"
else
    echo "[7/9] Filtered train FASTA already present, skipping"
fi

if [[ ! -f "$DIAMOND_DB" ]]; then
    echo "[7/9] Building DIAMOND index on train subset (~1-2 min)..."
    diamond makedb --in "$SWISSPROT_TRAIN_FASTA" --db "$DIAMOND_DB" --quiet
else
    echo "[7/9] DIAMOND index already built, skipping"
fi


# ---------------------------------------------------------------------------
# 8. Build validation / test / price149 JSONL splits
# ---------------------------------------------------------------------------
#
# Note: no train.jsonl is written. ProteInfer train is the BLAST reference
# corpus (populated into the DIAMOND DB in step 7), not an evaluation set.

VAL_JSONL="$DATA/validation.jsonl"
TEST_JSONL="$DATA/test.jsonl"
PRICE149_JSONL="$DATA/price149.jsonl"

if [[ ! -f "$VAL_JSONL" || ! -f "$TEST_JSONL" || ! -f "$PRICE149_JSONL" ]]; then
    echo "[8/9] Building JSONL splits..."
    python3 "$HERE/scripts/build_splits.py" \
        --parsed "$PARSED_PKL" \
        --fasta "$SWISSPROT_FASTA" \
        --proteinfer-dir "$PROTEINFER_DIR" \
        --ec2go "$EC2GO" \
        --price149-csv "$PRICE149_CSV" \
        --val-out "$VAL_JSONL" \
        --test-out "$TEST_JSONL" \
        --price149-out "$PRICE149_JSONL" \
        --val-size 2000 \
        --test-size 1000 \
        --seed 0
else
    echo "[8/9] Splits already built, skipping"
fi


# ---------------------------------------------------------------------------
# 9. Precompute ESM-2 150M embeddings over the ProteInfer-train subset
# ---------------------------------------------------------------------------
#
# Populates the cache queried at runtime by the agent's esm_nearest() tool
# (BLAST-analogue in embedding space). First run downloads ~600MB of ESM-2
# weights into the torch.hub cache; the embedding sweep over ~183K train
# proteins takes ~45-90 min on laptop CPU, minutes on CUDA/MPS. Output:
# ~470MB .npy matrix + ~5MB .json accession index. Total extra ~1.1GB
# resident memory per evaluator process at runtime (weights + matrix).

ESM_EMBEDDINGS_NPY="$DATA/esm_train_embeddings.npy"
ESM_ACCESSIONS_JSON="$DATA/esm_train_accessions.json"

if [[ ! -f "$ESM_EMBEDDINGS_NPY" || ! -f "$ESM_ACCESSIONS_JSON" ]]; then
    echo "[9/9] Computing ESM-2 150M embeddings for ProteInfer-train (~45-90 min CPU, minutes on GPU)..."
    python3 "$HERE/scripts/compute_esm_embeddings.py" \
        --input-fasta "$SWISSPROT_TRAIN_FASTA" \
        --output-embeddings "$ESM_EMBEDDINGS_NPY" \
        --output-accessions "$ESM_ACCESSIONS_JSON"
else
    echo "[9/9] ESM embeddings already present, skipping"
fi


# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

echo
echo "============================================================"
echo "Setup complete!"
echo
echo "Data files:"
ls -lh "$DATA" | tail -n +2 | awk '{printf "  %-40s %s\n", $9, $5}' | grep -v '^  *$'
echo
echo "Quick test:"
echo "  python examples/protein_go/main.py --num-iterations 2"
echo "============================================================"
