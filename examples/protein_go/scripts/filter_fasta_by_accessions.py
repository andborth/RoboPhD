#!/usr/bin/env python3
"""Filter a SwissProt FASTA to only records whose accession appears in a
ProteInfer TFRecord.

Used by setup.sh to build swissprot_train.fasta (the ProteInfer-train subset
of SwissProt), which is then indexed by `diamond makedb` as the BLAST reference
for the protein_go example. Restricting the DB to train-only matches the
published ProteInfer / ProtNote / ProtEx / ProtGO protocol and prevents test
proteins from self-hitting the DB.

Streaming: the input FASTA is read once top-to-bottom and kept-records are
written as they are encountered, so peak memory is bounded by one record.

CLI:
    python3 filter_fasta_by_accessions.py \\
        --input uniprot_sprot.fasta \\
        --tfrecord proteinfer/train.tfrecord \\
        --output swissprot_train.fasta
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from proteinfer_io import read_proteinfer_accessions

logger = logging.getLogger(__name__)


def _accession_from_header(header: str) -> str | None:
    """Extract the UniProt accession from a FASTA header like
    '>sp|P12345|NAME_SPECIES ...'. Returns None if no pipe-delimited accession.

    Only the pipe form is supported here — SwissProt FASTA always uses it. Note
    this is asymmetric with proteinfer_io.read_proteinfer_accessions, which
    also accepts plain-accession IDs because the TFRecord "id" field varies
    across ProteInfer releases.
    """
    # header still has the leading '>'; strip it
    body = header[1:] if header.startswith(">") else header
    parts = body.split("|")
    return parts[1] if len(parts) >= 2 else None


def filter_fasta(input_path: Path, accessions: set[str], output_path: Path) -> tuple[int, int]:
    """Write a new FASTA containing only records whose accession is in `accessions`.

    Returns (records_kept, records_seen).
    """
    kept = 0
    seen = 0
    keep_current = False
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            if line.startswith(">"):
                seen += 1
                acc = _accession_from_header(line.rstrip())
                keep_current = acc is not None and acc in accessions
                if keep_current:
                    kept += 1
                    fout.write(line)
            elif keep_current:
                fout.write(line)
    return kept, seen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="Input FASTA (e.g. uniprot_sprot.fasta)")
    parser.add_argument("--tfrecord", required=True, type=Path,
                        help="ProteInfer TFRecord providing the accession whitelist "
                             "(usually train.tfrecord)")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output FASTA containing only whitelisted records")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    if not args.input.exists():
        sys.exit(f"Input FASTA not found: {args.input}")
    if not args.tfrecord.exists():
        sys.exit(f"TFRecord not found: {args.tfrecord}")

    accessions = read_proteinfer_accessions(args.tfrecord)
    logger.info(f"Filtering {args.input.name} -> {args.output.name}"
                f" using {len(accessions)} accessions from {args.tfrecord.name}")

    kept, seen = filter_fasta(args.input, accessions, args.output)
    logger.info(f"Kept {kept} of {seen} FASTA records"
                f" ({kept / seen:.1%} of input; "
                f"{kept / len(accessions):.1%} of whitelist accessions found)")


if __name__ == "__main__":
    main()
