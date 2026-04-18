#!/usr/bin/env python3
"""Parse a SwissProt flat file (.dat) into a pickle dict for fast lookup.

For each accession, we keep only what the agent / evaluator needs:
  - name (UniProt entry name, e.g. "KKCC1_HUMAN")
  - organism (full species name, e.g. "Homo sapiens (Human)")
  - description (function comment, first ~500 chars)
  - length (sequence length in residues)
  - go_terms_mfo (list of MFO GO IDs with experimental evidence codes)
  - all_go_terms_mfo (list of MFO GO IDs with any evidence code, including IEA)
  - deposition_date (YYYY-MM-DD; the date the entry was created in SwissProt)

We split GO annotations by namespace (MFO only) and by evidence code:
  - "experimental" = EXP, IDA, IPI, IMP, IGI, IEP, HTP, HDA, HMP, HGI, HEP, TAS, IC
  - "all" = experimental + electronic (IEA) + computational + author-stated

For ground truth we use experimental only; for BLAST hit annotations we use all
(otherwise BLAST hits would carry no labels for poorly-studied homologs).
"""

import argparse
import logging
import pickle
import sys
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)

EXPERIMENTAL_EVIDENCE = {
    "EXP", "IDA", "IPI", "IMP", "IGI", "IEP",
    "HTP", "HDA", "HMP", "HGI", "HEP",
    "TAS", "IC",
}


def parse_record(record) -> Dict:
    """Extract the fields we care about from a Bio.SwissProt record."""
    name = record.entry_name
    organism = record.organism

    # Description: prefer FUNCTION comment; fall back to short description fields.
    function_text = ""
    for cc in record.comments:
        if cc.startswith("FUNCTION:"):
            function_text = cc[len("FUNCTION:"):].strip()
            break
    if not function_text and hasattr(record, "description"):
        function_text = str(record.description)
    if len(function_text) > 500:
        function_text = function_text[:500].rstrip() + "..."

    length = record.sequence_length

    # Cross-references: GO annotations come as ("GO", "GO:0004674", "F:protein...", "ECO:0000269|PubMed:...")
    # The 3rd element starts with "F:" for MFO, "P:" for BPO, "C:" for CCO.
    # The 4th element contains evidence; we look for the standard 3-letter code.
    go_terms_mfo_exp: List[str] = []
    go_terms_mfo_all: List[str] = []
    seen_exp: set = set()
    seen_all: set = set()
    for xref in record.cross_references:
        if not xref or xref[0] != "GO":
            continue
        if len(xref) < 3:
            continue
        go_id = xref[1]
        aspect = xref[2][:2]  # "F:", "P:", "C:"
        if aspect != "F:":
            continue
        # Evidence code is in the 4th field if present, e.g. "IEA:UniProtKB-KW"
        evidence_code = ""
        if len(xref) >= 4 and xref[3]:
            evidence_code = xref[3].split(":", 1)[0].strip()
        if go_id not in seen_all:
            go_terms_mfo_all.append(go_id)
            seen_all.add(go_id)
        if evidence_code in EXPERIMENTAL_EVIDENCE and go_id not in seen_exp:
            go_terms_mfo_exp.append(go_id)
            seen_exp.add(go_id)

    # Deposition date: record.created is a tuple ("DD-MMM-YYYY", release_number)
    deposition_date = ""
    try:
        date_str = record.created[0]
        deposition_date = datetime.strptime(date_str, "%d-%b-%Y").strftime("%Y-%m-%d")
    except (AttributeError, ValueError, TypeError, IndexError):
        pass

    return {
        "name": name,
        "organism": organism,
        "description": function_text,
        "length": length,
        "go_terms_mfo": go_terms_mfo_exp,
        "all_go_terms_mfo": go_terms_mfo_all,
        "deposition_date": deposition_date,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to uniprot_sprot.dat (uncompressed)")
    parser.add_argument("--output", required=True, help="Path to write pickle dict")
    parser.add_argument("--log-every", type=int, default=50000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    try:
        from Bio import SwissProt
    except ImportError:
        sys.exit("biopython is required: pip install biopython")

    entries: Dict[str, Dict] = {}
    n = 0
    with open(args.input) as fh:
        for record in SwissProt.parse(fh):
            n += 1
            parsed = parse_record(record)
            for accession in record.accessions:
                entries[accession] = parsed
            if n % args.log_every == 0:
                logger.info(f"  Parsed {n} records, {len(entries)} accessions")

    logger.info(f"Total: {n} records, {len(entries)} accessions")
    logger.info(f"Writing to {args.output}...")
    with open(args.output, "wb") as f:
        pickle.dump(entries, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Done.")


if __name__ == "__main__":
    main()
