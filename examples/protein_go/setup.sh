#!/usr/bin/env bash
#
# Setup data for the protein GO-term prediction example (Molecular Function sub-ontology).
#
# Downloads (~800 MB total):
#   - SwissProt 2022_01 release (UniProtKB curated subset, FASTA + flat file)
#   - Gene Ontology (go-basic.obo)
#   - EC-to-GO mapping (ec2go, from geneontology.org)
#   - ProteInfer clustered split (UniRef50-based, train/dev/test TFRecords)
#   - Price-149 challenging held-out set (CLEAN paper, EC labels)
#
# Builds:
#   - DIAMOND index over SwissProt (~5 min one-time)
#   - Parsed SwissProt entries pickle (accession -> name, organism, GO terms, ...)
#   - Train / validation / test JSONL splits aligned with ProteInfer's clustered split
#   - Price-149 JSONL with EC labels mapped to GO-MFO terms via ec2go
#
# Total runtime: ~20-30 min on a fast connection.
#
# Requirements:
#   - DIAMOND binary on PATH (conda install -c bioconda diamond)
#   - Python with biopython, tfrecord (pip install biopython tfrecord)
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
echo "[1/8] DIAMOND found: $(diamond --version 2>&1 | head -1)"

if ! python3 -c "import Bio" 2>/dev/null; then
    echo "ERROR: biopython not installed."
    echo "Install with: pip install biopython"
    exit 1
fi
echo "[1/8] biopython found"

if ! python3 -c "import tfrecord" 2>/dev/null; then
    echo "ERROR: tfrecord package not installed."
    echo "Install with: pip install tfrecord"
    echo "(Lightweight TFRecord reader — does not require TensorFlow.)"
    exit 1
fi
echo "[1/8] tfrecord found"


# ---------------------------------------------------------------------------
# 2. Download SwissProt
# ---------------------------------------------------------------------------

SWISSPROT_RELEASE="2022_01"
SWISSPROT_FASTA="$DATA/uniprot_sprot.fasta"
SWISSPROT_DAT="$DATA/uniprot_sprot.dat"

if [[ ! -f "$SWISSPROT_FASTA" ]]; then
    echo "[2/8] Downloading SwissProt $SWISSPROT_RELEASE (~90 MB)..."
    curl -fL --retry 3 \
        "https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-${SWISSPROT_RELEASE}/knowledgebase/uniprot_sprot-only${SWISSPROT_RELEASE}.tar.gz" \
        -o "$DATA/sprot.tar.gz"
    tar -xzf "$DATA/sprot.tar.gz" -C "$DATA" uniprot_sprot.fasta.gz uniprot_sprot.dat.gz
    rm -f "$DATA/sprot.tar.gz"
    gunzip "$DATA/uniprot_sprot.fasta.gz"
    gunzip "$DATA/uniprot_sprot.dat.gz"
else
    echo "[2/8] SwissProt already present, skipping"
fi


# ---------------------------------------------------------------------------
# 3. Build DIAMOND index
# ---------------------------------------------------------------------------

DIAMOND_DB="$DATA/swissprot.dmnd"
if [[ ! -f "$DIAMOND_DB" ]]; then
    echo "[3/8] Building DIAMOND index (~3-5 min)..."
    diamond makedb --in "$SWISSPROT_FASTA" --db "$DIAMOND_DB" --quiet
else
    echo "[3/8] DIAMOND index already built, skipping"
fi


# ---------------------------------------------------------------------------
# 4. Parse SwissProt flat file to pickle dict
# ---------------------------------------------------------------------------

PARSED_PKL="$DATA/swissprot_entries.pkl"
if [[ ! -f "$PARSED_PKL" ]]; then
    echo "[4/8] Parsing SwissProt flat file (~3-5 min)..."
    python3 "$HERE/scripts/parse_swissprot.py" \
        --input "$SWISSPROT_DAT" \
        --output "$PARSED_PKL"
else
    echo "[4/8] Parsed SwissProt already present, skipping"
fi


# ---------------------------------------------------------------------------
# 5. Download GO ontology and ec2go mapping
# ---------------------------------------------------------------------------

GO_OBO="$DATA/go-basic.obo"
if [[ ! -f "$GO_OBO" ]]; then
    echo "[5/8] Downloading GO ontology..."
    curl -fL --retry 3 "http://purl.obolibrary.org/obo/go/go-basic.obo" -o "$GO_OBO"
else
    echo "[5/8] GO ontology already present, skipping"
fi

EC2GO="$DATA/ec2go.txt"
if [[ ! -f "$EC2GO" ]]; then
    echo "[5/8] Downloading ec2go mapping (EC number -> GO term)..."
    curl -fL --retry 3 "http://current.geneontology.org/ontology/external2go/ec2go" -o "$EC2GO"
else
    echo "[5/8] ec2go mapping already present, skipping"
fi


# ---------------------------------------------------------------------------
# 6. Download ProteInfer clustered-split TFRecords
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
    echo "[6/8] Downloading ProteInfer clustered-split TFRecords (~60 MB)..."
    mkdir -p "$PROTEINFER_DIR"
    for split in train dev test; do
        if [[ ! -f "$PROTEINFER_DIR/${split}.tfrecord" ]]; then
            curl -fL --retry 3 "${PROTEINFER_BASE}/${split}.tfrecord" \
                -o "$PROTEINFER_DIR/${split}.tfrecord"
        fi
    done
else
    echo "[6/8] ProteInfer split already downloaded, skipping"
fi


# ---------------------------------------------------------------------------
# 7. Download Price-149 (CLEAN paper's homology-resistant held-out set)
# ---------------------------------------------------------------------------
#
# Price-149 is 149 enzymes experimentally characterized by Price et al. and used
# as a "hard case" benchmark by CLEAN (Yu et al. 2023). Labels are EC numbers;
# we map them to GO-MFO via the ec2go file in step 5.

PRICE149_CSV="$DATA/price149_raw.csv"
if [[ ! -f "$PRICE149_CSV" ]]; then
    echo "[7/8] Downloading Price-149 from CLEAN repository..."
    curl -fL --retry 3 \
        "https://raw.githubusercontent.com/tttianhao/CLEAN/main/app/data/price.csv" \
        -o "$PRICE149_CSV"
else
    echo "[7/8] Price-149 already downloaded, skipping"
fi


# ---------------------------------------------------------------------------
# 8. Build train / validation / test / price149 JSONL splits
# ---------------------------------------------------------------------------

TRAIN_JSONL="$DATA/train.jsonl"
VAL_JSONL="$DATA/validation.jsonl"
TEST_JSONL="$DATA/test.jsonl"
PRICE149_JSONL="$DATA/price149.jsonl"

if [[ ! -f "$TRAIN_JSONL" || ! -f "$VAL_JSONL" || ! -f "$TEST_JSONL" || ! -f "$PRICE149_JSONL" ]]; then
    echo "[8/8] Building JSONL splits..."
    python3 "$HERE/scripts/build_splits.py" \
        --parsed "$PARSED_PKL" \
        --fasta "$SWISSPROT_FASTA" \
        --proteinfer-dir "$PROTEINFER_DIR" \
        --ec2go "$EC2GO" \
        --price149-csv "$PRICE149_CSV" \
        --train-out "$TRAIN_JSONL" \
        --val-out "$VAL_JSONL" \
        --test-out "$TEST_JSONL" \
        --price149-out "$PRICE149_JSONL" \
        --train-size 3000 \
        --val-size 500 \
        --test-size 1200 \
        --seed 0
else
    echo "[8/8] Splits already built, skipping"
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
