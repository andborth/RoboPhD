"""Tool callables for the protein GO-term prediction agent (Molecular Function sub-ontology).

Each tool is constructed by a factory that can close over per-evaluation state
(e.g., a cost tracker) and any shared caches. The factories are called once per
evaluation by the evaluator.

Design notes:
  - blast(): DIAMOND against SwissProt. First call per sequence runs DIAMOND live;
    results are cached in-process under the raw sequence hash at top_k=100 with no
    identity floor. Subsequent calls filter that cached superset. Thread-safe via
    a lock on the cache.
  - uniprot(): dict lookup over the parsed SwissProt flat file. Loaded lazily once
    per process (~1 min, ~1 GB RAM).
  - go_ancestors(): static lookup over the parsed GO DAG (MFO namespace only).
  - sequence_features(): pure computation, no cache.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data locations (populated by setup.sh)
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"
SWISSPROT_DB = DATA_DIR / "swissprot.dmnd"            # DIAMOND index
SWISSPROT_PARSED = DATA_DIR / "swissprot_entries.pkl"  # accession -> entry dict
GO_OBO = DATA_DIR / "go-basic.obo"                     # GO ontology


# ---------------------------------------------------------------------------
# Shared module-level caches (populated lazily; thread-safe)
# ---------------------------------------------------------------------------

_diamond_cache: Dict[str, List[Dict[str, Any]]] = {}
_diamond_cache_lock = threading.Lock()

_uniprot_dict: Optional[Dict[str, Dict[str, Any]]] = None
_uniprot_dict_lock = threading.Lock()

_go_dag: Optional[Dict[str, List[str]]] = None  # go_id -> list of ancestor go_ids (MFO only)
_go_dag_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Sequence features (no cache, pure computation)
# ---------------------------------------------------------------------------

_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

# Kyte-Doolittle hydrophobicity scale
_HYDROPHOBICITY = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

# Approximate monoisotopic mass (Da) of each amino acid residue
_AA_MASS = {
    "A": 71.04, "C": 103.01, "D": 115.03, "E": 129.04, "F": 147.07,
    "G": 57.02, "H": 137.06, "I": 113.08, "K": 128.09, "L": 113.08,
    "M": 131.04, "N": 114.04, "P": 97.05, "Q": 128.06, "R": 156.10,
    "S": 87.03, "T": 101.05, "V": 99.07, "W": 186.08, "Y": 163.06,
}


def _sequence_features(sequence: str) -> Dict[str, Any]:
    """Compute physicochemical features from sequence. Called by the callable factory."""
    seq = sequence.upper()
    n = len(seq)
    if n == 0:
        return {"length": 0, "composition": {}, "molecular_weight": 0.0, "mean_hydrophobicity": 0.0}

    composition = {aa: seq.count(aa) / n for aa in _AA_ALPHABET}
    mw = sum(_AA_MASS.get(aa, 0) for aa in seq) + 18.02  # +H2O for N- and C-termini
    mean_h = sum(_HYDROPHOBICITY.get(aa, 0) for aa in seq) / n

    # Rough predicted-disorder: fraction of residues in a sliding window with
    # mean hydrophobicity < 0 (disorder-promoting). Not a real disorder predictor,
    # just a crude signal.
    window = 21
    disorder_frac = 0.0
    if n >= window:
        disordered = 0
        for i in range(n - window + 1):
            mh = sum(_HYDROPHOBICITY.get(seq[j], 0) for j in range(i, i + window)) / window
            if mh < 0:
                disordered += 1
        disorder_frac = disordered / (n - window + 1)

    return {
        "length": n,
        "composition": composition,
        "molecular_weight": round(mw, 2),
        "mean_hydrophobicity": round(mean_h, 3),
        "approx_disorder_fraction": round(disorder_frac, 3),
    }


# ---------------------------------------------------------------------------
# SwissProt / UniProt loading (lazy, process-wide)
# ---------------------------------------------------------------------------

def _load_uniprot_dict() -> Dict[str, Dict[str, Any]]:
    """Load parsed SwissProt entries. Lazy, once per process."""
    global _uniprot_dict
    with _uniprot_dict_lock:
        if _uniprot_dict is not None:
            return _uniprot_dict
        if not SWISSPROT_PARSED.exists():
            raise FileNotFoundError(
                f"Parsed SwissProt not found at {SWISSPROT_PARSED}. "
                f"Run setup.sh first."
            )
        import pickle
        logger.info(f"Loading parsed SwissProt from {SWISSPROT_PARSED}...")
        with open(SWISSPROT_PARSED, "rb") as f:
            _uniprot_dict = pickle.load(f)
        logger.info(f"Loaded {len(_uniprot_dict)} SwissProt entries")
        return _uniprot_dict


# ---------------------------------------------------------------------------
# GO DAG loading
# ---------------------------------------------------------------------------

def _load_go_dag() -> Dict[str, List[str]]:
    """Parse the GO OBO file, restrict to MFO, compute transitive ancestors."""
    global _go_dag
    with _go_dag_lock:
        if _go_dag is not None:
            return _go_dag
        if not GO_OBO.exists():
            raise FileNotFoundError(
                f"GO OBO file not found at {GO_OBO}. Run setup.sh first."
            )
        logger.info(f"Loading GO DAG from {GO_OBO}...")
        _go_dag = _parse_mfo_ancestors(GO_OBO)
        logger.info(f"Loaded {len(_go_dag)} MFO GO terms with ancestors")
        return _go_dag


def _parse_mfo_ancestors(obo_path: Path) -> Dict[str, List[str]]:
    """Parse GO OBO file and return {go_id: [ancestor_ids]} for MFO namespace only."""
    # Parse all terms: id, namespace, is_a parents
    terms: Dict[str, Dict[str, Any]] = {}
    current: Optional[Dict[str, Any]] = None
    with open(obo_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "[Term]":
                current = {"parents": []}
            elif line == "" or line.startswith("["):
                if current and "id" in current:
                    terms[current["id"]] = current
                current = None
            elif current is not None:
                if line.startswith("id: "):
                    current["id"] = line[4:].strip()
                elif line.startswith("namespace: "):
                    current["namespace"] = line[11:].strip()
                elif line.startswith("is_a: "):
                    # "is_a: GO:0003674 ! molecular_function"
                    parent = line[6:].split(" ! ", 1)[0].strip()
                    current["parents"].append(parent)
                elif line.startswith("is_obsolete: true"):
                    current["obsolete"] = True
    if current and "id" in current:
        terms[current["id"]] = current

    # Restrict to MFO, non-obsolete, and compute transitive ancestors (excluding self)
    mfo = {k: v for k, v in terms.items()
           if v.get("namespace") == "molecular_function"
           and not v.get("obsolete", False)}

    def ancestors_of(go_id: str, memo: Dict[str, set]) -> set:
        if go_id in memo:
            return memo[go_id]
        direct = mfo.get(go_id, {}).get("parents", [])
        result = set()
        for p in direct:
            if p in mfo:  # only MFO parents
                result.add(p)
                result |= ancestors_of(p, memo)
        memo[go_id] = result
        return result

    memo: Dict[str, set] = {}
    return {go_id: sorted(ancestors_of(go_id, memo)) for go_id in mfo}


# ---------------------------------------------------------------------------
# DIAMOND BLAST runner
# ---------------------------------------------------------------------------

def _run_diamond(sequence: str, top_k: int = 100) -> List[Dict[str, Any]]:
    """Run DIAMOND against SwissProt, return top_k hits enriched with GO-MFO labels."""
    if not SWISSPROT_DB.exists():
        raise FileNotFoundError(
            f"DIAMOND index not found at {SWISSPROT_DB}. Run setup.sh first."
        )

    # Keep the temp file open across the subprocess call so context-manager exit
    # auto-deletes it even on exceptions. DIAMOND can still read the path while
    # Python holds the handle on macOS/Linux (the project's supported platforms).
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta") as qf:
        qf.write(f">query\n{sequence}\n")
        qf.flush()  # NamedTemporaryFile is buffered; DIAMOND needs data on disk.
        cmd = [
            "diamond", "blastp",
            "--db", str(SWISSPROT_DB),
            "--query", qf.name,
            "--max-target-seqs", str(top_k),
            "--outfmt", "6", "sseqid", "pident", "evalue", "qcovhsp", "bitscore",
            "--quiet",
            "--sensitive",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            # Malformed sequences, transient DIAMOND hiccups, etc. Degrade gracefully
            # (matches the contract of uniprot() / go_ancestors(), which also never raise).
            err_lines = (e.stderr or "").strip().splitlines()
            last_err = err_lines[-1] if err_lines else ""
            logger.warning(
                "DIAMOND failed on query of length %d (exit=%d): %s",
                len(sequence), e.returncode, last_err,
            )
            return []

    uniprot = _load_uniprot_dict()
    hits: List[Dict[str, Any]] = []
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split("\t")
        # sseqid may be "sp|P12345|NAME_SPECIES" — extract accession
        sseqid = parts[0]
        accession = sseqid.split("|")[1] if "|" in sseqid else sseqid
        entry = uniprot.get(accession, {})
        hits.append({
            "accession": accession,
            "identity": float(parts[1]),
            "e_value": float(parts[2]),
            "query_coverage": float(parts[3]),
            "bit_score": float(parts[4]),
            "go_terms": entry.get("go_terms_mfo", []),
            "description": entry.get("description", ""),
        })
    return hits


def _get_cached_diamond(sequence: str) -> List[Dict[str, Any]]:
    """Cache DIAMOND results keyed by sequence hash."""
    key = hashlib.sha1(sequence.encode()).hexdigest()[:16]
    with _diamond_cache_lock:
        if key in _diamond_cache:
            return _diamond_cache[key]
    # Run outside the lock so concurrent new sequences don't block each other
    hits = _run_diamond(sequence, top_k=100)
    with _diamond_cache_lock:
        _diamond_cache[key] = hits
    return hits


# ---------------------------------------------------------------------------
# Factory functions — called once per evaluation by the evaluator
# ---------------------------------------------------------------------------

def make_blast() -> Callable[..., List[Dict[str, Any]]]:
    """Create a blast() callable. Results cached module-wide across evaluations."""

    def blast(sequence: str, top_k: int = 50, min_identity: float = 0.0,
              min_coverage: float = 0.0) -> List[Dict[str, Any]]:
        raw = _get_cached_diamond(sequence)
        filtered = [h for h in raw
                    if h["identity"] >= min_identity
                    and h["query_coverage"] >= min_coverage]
        return filtered[:top_k]

    return blast


def make_uniprot() -> Callable[[str], Dict[str, Any]]:
    """Create a uniprot() callable."""

    def uniprot(accession: str) -> Dict[str, Any]:
        entries = _load_uniprot_dict()
        return entries.get(accession, {})

    return uniprot


def make_go_ancestors() -> Callable[[str], List[str]]:
    """Create a go_ancestors() callable."""

    def go_ancestors(go_id: str) -> List[str]:
        dag = _load_go_dag()
        return list(dag.get(go_id, []))

    return go_ancestors


def make_sequence_features() -> Callable[[str], Dict[str, Any]]:
    """Create a sequence_features() callable. No shared state; pure computation."""
    return _sequence_features


# ---------------------------------------------------------------------------
# GO hierarchy helper used by scoring (not exposed to agent)
# ---------------------------------------------------------------------------

def propagate_to_mfo_ancestors(go_terms: set) -> set:
    """Augment a set of GO terms with all MFO ancestors."""
    dag = _load_go_dag()
    result = set(go_terms)
    for term in list(go_terms):
        result.update(dag.get(term, []))
    return result
