"""Shared helpers for reading ProteInfer's TFRecord distribution."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Set

logger = logging.getLogger(__name__)


def read_proteinfer_accessions(tfrecord_path: Path) -> Set[str]:
    """Read a ProteInfer TFRecord and return the set of UniProt accessions.

    ProteInfer's TFRecord schema contains (at minimum):
      - 'id':       bytes feature, e.g. b'sp|P0AEX9|MALE_ECOLI' or b'P0AEX9'
      - 'sequence': bytes feature (the amino-acid sequence)
      - 'labels':   bytes feature, comma-separated GO/EC labels

    We only need the accession here; annotations come from our parsed SwissProt
    pickle (which uses SwissProt 2022_01 labels, richer than ProteInfer's
    original 2019 labels).
    """
    try:
        from tfrecord.reader import tfrecord_loader
    except ImportError:
        sys.exit("tfrecord package not installed. Run: pip install tfrecord")

    accessions: Set[str] = set()
    description = {"id": "byte"}

    for record in tfrecord_loader(str(tfrecord_path), None, description):
        raw_id = record["id"]
        if isinstance(raw_id, (bytes, bytearray)):
            id_str = raw_id.decode("utf-8", errors="ignore")
        else:
            # tfrecord may return ndarray of bytes
            try:
                id_str = bytes(raw_id).decode("utf-8", errors="ignore")
            except Exception:
                continue

        # Extract accession from various possible formats
        if "|" in id_str:
            # e.g. "sp|P0AEX9|MALE_ECOLI"
            parts = id_str.split("|")
            if len(parts) >= 2:
                accessions.add(parts[1])
        else:
            # Plain accession
            accessions.add(id_str.strip())

    logger.info(f"Read {len(accessions)} accessions from {tfrecord_path.name}")
    return accessions
