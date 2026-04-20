#!/usr/bin/env python3
"""Build validation/test/price149 JSONL splits for the protein-GO benchmark.

Split source: ProteInfer's UniRef50-clustered split (Sanderson et al. 2023).
No sequence in dev/test has >50% identity to any train sequence, which makes
BLAST-based homology transfer genuinely difficult and tests generalization to
remote-homology cases. This is the canonical hard-split used by ProteInfer,
ProtNote, ProtEx, and ProtGO.

ProteInfer train is the BLAST reference *corpus* (built into the DIAMOND DB by
filter_fasta_by_accessions.py), not an evaluation set — no train.jsonl is
written. Evolution queries come from dev (val.jsonl); final eval comes from
test (test.jsonl); Price-149 is a separate homology-resistant secondary eval.

Pipeline:
  1. Read ProteInfer's dev/test TFRecords (lightweight via tfrecord package,
     no TensorFlow required) to get the canonical (accession, split) mapping.
  2. Intersect each split with the parsed SwissProt pickle (for annotations,
     descriptions, metadata) and the FASTA (for sequences).
  3. Filter to entries with experimentally-supported MFO labels and valid
     sequences (length 50-2000, standard amino acids only).
  4. Subsample each split to the requested sizes.
  5. For Price-149: parse the raw table, map EC numbers to GO-MFO terms via
     ec2go, filter to sequences where mapping succeeds and produces at least
     one MFO term.

Output JSONL format (all splits, including Price-149):
  {"accession": str, "sequence": str, "go_terms_mfo": list[str]}

For Price-149, "accession" is the RefSeq WP_* identifier from the CLEAN table;
"go_terms_mfo" is derived from EC labels via ec2go mapping.
"""

import argparse
import gzip
import json
import logging
import pickle
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from proteinfer_io import read_proteinfer_accessions

logger = logging.getLogger(__name__)

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LEN = 50
MAX_LEN = 2000


# ---------------------------------------------------------------------------
# SwissProt sequence loading (shared by all splits)
# ---------------------------------------------------------------------------

def load_swissprot_sequences(fasta_path: Path) -> Dict[str, str]:
    """Read SwissProt FASTA and return {accession: sequence}."""
    sequences: Dict[str, str] = {}
    accession = None
    seq_parts: List[str] = []

    if str(fasta_path).endswith(".gz"):
        opener = lambda p: gzip.open(p, "rt")
    else:
        opener = lambda p: open(p)

    with opener(fasta_path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if accession and seq_parts:
                    sequences[accession] = "".join(seq_parts)
                parts = line[1:].split("|")
                accession = parts[1] if len(parts) >= 2 else None
                seq_parts = []
            elif accession is not None:
                seq_parts.append(line)
        if accession and seq_parts:
            sequences[accession] = "".join(seq_parts)

    logger.info(f"Loaded {len(sequences)} sequences from {fasta_path}")
    return sequences


def is_valid_sequence(seq: str) -> bool:
    if not (MIN_LEN <= len(seq) <= MAX_LEN):
        return False
    return all(aa in VALID_AA for aa in seq)


# ---------------------------------------------------------------------------
# ProteInfer TFRecord reading
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Eligibility filter for SwissProt-based splits
# ---------------------------------------------------------------------------

def build_eligible_from_proteinfer(
    proteinfer_accessions: Set[str],
    parsed_entries: Dict[str, Dict],
    sequences: Dict[str, str],
) -> List[Tuple[str, str, List[str]]]:
    """Return (accession, sequence, go_terms_mfo) tuples for eligible proteins
    that appear in the given ProteInfer split.
    """
    eligible: List[Tuple[str, str, List[str]]] = []
    missing_parsed = 0
    missing_seq = 0
    no_mfo = 0
    bad_seq = 0

    for accession in proteinfer_accessions:
        entry = parsed_entries.get(accession)
        if entry is None:
            missing_parsed += 1
            continue

        go_terms = entry.get("go_terms_mfo", [])
        if not go_terms:
            no_mfo += 1
            continue

        seq = sequences.get(accession)
        if seq is None:
            missing_seq += 1
            continue
        if not is_valid_sequence(seq):
            bad_seq += 1
            continue

        eligible.append((accession, seq, sorted(go_terms)))

    logger.info(
        f"  Eligible: {len(eligible)} | "
        f"missing-in-parsed: {missing_parsed} | "
        f"missing-in-fasta: {missing_seq} | "
        f"no experimental MFO: {no_mfo} | "
        f"bad sequence: {bad_seq}"
    )
    return eligible


# ---------------------------------------------------------------------------
# Price-149 loading with EC -> GO-MFO mapping
# ---------------------------------------------------------------------------

_EC2GO_LINE_RE = re.compile(
    r"^EC:(?P<ec>[\d\.\-nN]+)\s+.*?>\s*GO:(?P<go_name>[^;]+?)\s*;\s*(?P<go_id>GO:\d{7})\s*$"
)


def load_ec2go(ec2go_path: Path) -> Dict[str, List[str]]:
    """Parse the ec2go file and return {ec_number: [go_ids]}.

    ec2go format (from geneontology.org/external2go/ec2go):
      EC:1.1.1.1 > GO:alcohol dehydrogenase (NAD+) activity ; GO:0004022
    """
    mapping: Dict[str, List[str]] = {}
    with open(ec2go_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("!"):
                continue
            m = _EC2GO_LINE_RE.match(line)
            if not m:
                continue
            ec = m.group("ec")
            go_id = m.group("go_id")
            mapping.setdefault(ec, []).append(go_id)

    logger.info(f"Loaded ec2go: {len(mapping)} EC numbers -> GO terms")
    return mapping


def ec_to_mfo_terms(ec: str, ec2go: Dict[str, List[str]]) -> List[str]:
    """Map an EC number to GO-MFO terms.

    Tries the full EC, then truncates trailing levels ("2.7.11.1" -> "2.7.11.-"
    -> "2.7.-.-" -> "2.-.-.-") to find any matching entry in ec2go. This
    handles the common case of Price-149 labels being more specific than
    ec2go's coverage.
    """
    parts = ec.split(".")
    # Try from most-specific to least-specific
    for i in range(len(parts), 0, -1):
        # Rebuild key with trailing "-" for truncated levels
        key_parts = parts[:i] + ["-"] * (len(parts) - i)
        key = ".".join(key_parts)
        if key in ec2go:
            return list(ec2go[key])
        # Also try without dashes (some entries use "2.7.11" directly)
        key_no_dash = ".".join(parts[:i])
        if key_no_dash in ec2go:
            return list(ec2go[key_no_dash])
    return []


def load_price149(
    csv_path: Path,
    ec2go: Dict[str, List[str]],
) -> List[Tuple[str, str, List[str]]]:
    """Parse Price-149 table and return (accession, sequence, go_terms_mfo).

    CLEAN's price.csv is **tab-separated** despite the .csv extension:
      Entry\tEC number\tSequence
      WP_063460136\t5.3.1.7\tMAIPPYPDFR...

    The "Entry" column holds RefSeq protein IDs (WP_*) rather than UniProt
    accessions; we pass them through unchanged as the stable identifier.

    We filter to rows where:
      - Sequence is valid (standard AAs, length in range)
      - At least one EC number maps to a GO-MFO term via ec2go
    """
    import csv as csv_module

    # Accumulate across rows: CLEAN's CSV sometimes has one row per (protein, EC)
    # pair, so a dedup-by-accession approach would silently drop additional EC
    # labels for multi-EC enzymes. Merge instead, then emit one example per
    # accession with all mappable GO terms unioned.
    accumulated: Dict[str, Dict[str, Any]] = {}
    bad_seq = 0
    skipped_rows = 0

    with open(csv_path) as f:
        # Delimiter is pinned to tab; the column-name fallbacks below only help
        # on future CLEAN revisions that keep tab separation but rename a column.
        # If CLEAN switches to comma delimiters, update this delimiter and the
        # setup.sh note together.
        reader = csv_module.DictReader(f, delimiter="\t")
        for row in reader:
            # CLEAN uses various column names across versions; try several
            accession = (row.get("Entry") or row.get("accession")
                         or row.get("uniprot_id") or "").strip()
            ec_field = (row.get("EC number") or row.get("EC")
                        or row.get("ec_number") or "").strip()
            sequence = (row.get("Sequence") or row.get("sequence")
                        or "").strip().upper()

            if not accession or not sequence or not ec_field:
                skipped_rows += 1
                continue

            if not is_valid_sequence(sequence):
                bad_seq += 1
                continue

            entry = accumulated.get(accession)
            if entry is None:
                entry = {"sequence": sequence, "go_terms": set()}
                accumulated[accession] = entry
            elif entry["sequence"] != sequence:
                # Same accession with different sequence: keep the first and skip.
                # Shouldn't happen in practice but guards against malformed CSVs.
                skipped_rows += 1
                continue

            # EC field may be a single EC or a ';'-separated list
            ecs = [e.strip() for e in re.split(r"[;,]", ec_field) if e.strip()]
            for ec in ecs:
                entry["go_terms"].update(ec_to_mfo_terms(ec, ec2go))

    examples: List[Tuple[str, str, List[str]]] = []
    no_mapping = 0
    for accession, entry in accumulated.items():
        if not entry["go_terms"]:
            no_mapping += 1
            continue
        examples.append((accession, entry["sequence"], sorted(entry["go_terms"])))

    logger.info(
        f"Price-149: {len(examples)} usable | "
        f"no EC->GO mapping: {no_mapping} | bad sequence: {bad_seq} | "
        f"skipped rows: {skipped_rows}"
    )
    return examples


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_jsonl(path: Path, examples: List[Tuple[str, str, List[str]]]) -> None:
    with open(path, "w") as f:
        for accession, seq, go_terms in examples:
            row = {
                "accession": accession,
                "sequence": seq,
                "go_terms_mfo": sorted(go_terms),
            }
            f.write(json.dumps(row) + "\n")
    logger.info(f"Wrote {len(examples)} examples to {path}")


def label_stats(examples: List[Tuple[str, str, List[str]]]) -> str:
    if not examples:
        return "empty"
    labels_per = [len(e[2]) for e in examples]
    return (f"mean labels/protein={sum(labels_per)/len(labels_per):.1f}, "
            f"max={max(labels_per)}, min={min(labels_per)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed", required=True, type=Path,
                        help="Pickle from parse_swissprot.py")
    parser.add_argument("--fasta", required=True, type=Path,
                        help="SwissProt FASTA")
    parser.add_argument("--proteinfer-dir", required=True, type=Path,
                        help="Directory with train.tfrecord, dev.tfrecord, test.tfrecord")
    parser.add_argument("--ec2go", required=True, type=Path,
                        help="Path to ec2go mapping file")
    parser.add_argument("--price149-csv", required=True, type=Path,
                        help="Path to Price-149 raw CSV")
    parser.add_argument("--val-out", required=True, type=Path)
    parser.add_argument("--test-out", required=True, type=Path)
    parser.add_argument("--price149-out", required=True, type=Path)
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    logger.info(f"Loading parsed SwissProt entries from {args.parsed}...")
    with open(args.parsed, "rb") as f:
        parsed_entries = pickle.load(f)
    logger.info(f"  {len(parsed_entries)} parsed entries")

    sequences = load_swissprot_sequences(args.fasta)

    # --- ProteInfer-aligned splits ---
    # Per-split RNG so reordering split_configs never silently changes which
    # proteins land in each split. The offsets ("dev"/"test") are fixed names,
    # not positional indices, so the mapping is stable across reorderings or
    # if a split is temporarily skipped.
    #
    # Note: "train" is not written. ProteInfer train is the BLAST reference
    # corpus (populated into the DIAMOND DB by filter_fasta_by_accessions.py
    # via train.tfrecord), not an evaluation set.
    split_configs = [
        ("dev",  args.val_size,  args.val_out),
        ("test", args.test_size, args.test_out),
    ]

    for split_name, target_size, out_path in split_configs:
        tfrecord_path = args.proteinfer_dir / f"{split_name}.tfrecord"
        if not tfrecord_path.exists():
            sys.exit(f"ProteInfer {split_name} TFRecord not found: {tfrecord_path}")
        logger.info(f"--- ProteInfer {split_name} split ---")
        accessions = read_proteinfer_accessions(tfrecord_path)
        eligible = build_eligible_from_proteinfer(accessions, parsed_entries, sequences)

        if len(eligible) < target_size:
            logger.warning(
                f"Only {len(eligible)} eligible in ProteInfer {split_name}, "
                f"writing all (requested {target_size})"
            )
            selected = eligible
        else:
            # Name-derived per-split seed: stable across code reorderings,
            # varies across splits so they aren't shuffled identically.
            split_rng = random.Random(f"{args.seed}:{split_name}")
            split_rng.shuffle(eligible)
            selected = eligible[:target_size]

        write_jsonl(out_path, selected)
        logger.info(f"  {split_name}: {label_stats(selected)}")

    # --- Price-149 ---
    logger.info("--- Price-149 ---")
    ec2go = load_ec2go(args.ec2go)
    price149 = load_price149(args.price149_csv, ec2go)
    write_jsonl(args.price149_out, price149)
    logger.info(f"  price149: {label_stats(price149)}")


if __name__ == "__main__":
    main()
