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

from RoboPhD.scoring import fmax_with_ancestor_closure

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data locations (populated by setup.sh)
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"
SWISSPROT_DB = DATA_DIR / "swissprot_train.dmnd"      # DIAMOND index (ProteInfer-train subset)
SWISSPROT_PARSED = DATA_DIR / "swissprot_entries.pkl"  # accession -> entry dict
SWISSPROT_FASTA = DATA_DIR / "uniprot_sprot.fasta"     # full SwissProt FASTA (used for sequence lookup)
GO_OBO = DATA_DIR / "go-basic.obo"                     # GO ontology
ESM_EMBEDDINGS_NPY = DATA_DIR / "esm_train_embeddings.npy"    # ProteInfer-train ESM-2 150M cache
ESM_ACCESSIONS_JSON = DATA_DIR / "esm_train_accessions.json"  # row-aligned accession index
ESM_MODEL_NAME = "esm2_t30_150M_UR50D"  # 150M params, 640-dim, 30 layers


# ---------------------------------------------------------------------------
# Shared module-level caches (populated lazily; thread-safe)
# ---------------------------------------------------------------------------

_diamond_cache: Dict[str, List[Dict[str, Any]]] = {}
_diamond_cache_lock = threading.Lock()

_uniprot_dict: Optional[Dict[str, Dict[str, Any]]] = None
_uniprot_dict_lock = threading.Lock()

_sequence_dict: Optional[Dict[str, str]] = None  # accession -> amino-acid sequence (full SwissProt)
_sequence_dict_lock = threading.Lock()

_go_dag: Optional[Dict[str, List[str]]] = None  # go_id -> list of ancestor go_ids (MFO only)
_go_dag_lock = threading.Lock()

# ESM-2 caches — precomputed embeddings over ProteInfer-train and the lazily-
# loaded model itself.
_esm_embeddings: Optional[Any] = None   # numpy.ndarray of shape (N, 640), float32
_esm_accessions: Optional[List[str]] = None
_esm_cache_lock = threading.Lock()

_esm_model: Optional[Any] = None
_esm_batch_converter: Optional[Any] = None
_esm_repr_layer: Optional[int] = None
_esm_device: Optional[str] = None
_esm_model_lock = threading.Lock()

# Serializes actual ESM forward passes across worker threads. PyTorch model
# objects are not thread-safe under concurrent inference; sharing one model
# across parallel evaluator workers requires serialization (especially on
# CUDA/MPS where 8× concurrent calls risks OOM).
_esm_inference_lock = threading.Lock()

# Per-query embedding cache keyed by sequence hash. Agents that call both
# esm_embed(seq) and esm_nearest(seq) on the same sequence get one forward
# pass instead of two. Thread-safe via the inference lock on population.
_esm_embed_cache: Dict[str, Any] = {}  # sha1 -> numpy.ndarray (640,) float32


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


def _load_sequences() -> Dict[str, str]:
    """Load {accession: amino_acid_sequence} from the full SwissProt FASTA.

    Lazy, once per process. Populates `_sequence_dict` for use by `make_uniprot()`
    and `_run_diamond()` so hit results and entry lookups can include the raw
    sequence of each homolog.

    Memory: ~90MB resident after loading all ~567K SwissProt sequences. Held
    process-wide; if you spawn many evaluator instances in the same Python
    process, they share this cache (not per-instance).
    """
    global _sequence_dict
    with _sequence_dict_lock:
        if _sequence_dict is not None:
            return _sequence_dict
        if not SWISSPROT_FASTA.exists():
            raise FileNotFoundError(
                f"SwissProt FASTA not found at {SWISSPROT_FASTA}. "
                f"Run setup.sh first."
            )
        logger.info(f"Loading SwissProt sequences from {SWISSPROT_FASTA}...")
        sequences: Dict[str, str] = {}
        accession: Optional[str] = None
        seq_parts: List[str] = []
        with open(SWISSPROT_FASTA) as f:
            for line in f:
                line = line.rstrip()
                if line.startswith(">"):
                    if accession and seq_parts:
                        sequences[accession] = "".join(seq_parts)
                    # Header: ">sp|P12345|NAME_SPECIES Description ..."
                    parts = line[1:].split("|")
                    accession = parts[1] if len(parts) >= 2 else None
                    seq_parts = []
                elif accession is not None:
                    seq_parts.append(line)
            if accession and seq_parts:
                sequences[accession] = "".join(seq_parts)
        _sequence_dict = sequences
        logger.info(f"Loaded {len(_sequence_dict)} SwissProt sequences")
        return _sequence_dict


# ---------------------------------------------------------------------------
# GO DAG loading
# ---------------------------------------------------------------------------

def load_go_dag() -> Dict[str, List[str]]:
    """Parse the GO OBO file, restrict to MFO, compute transitive ancestors.

    Public by design: callers outside tools.py (evaluator.py, make_score) depend
    on it, so renames must be coordinated. Returns the same cached singleton
    across calls; safe to call repeatedly.
    """
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
            # Bound DIAMOND to 2 threads per subprocess. Evaluator runs many
            # queries in parallel (default max_workers=8), and DIAMOND's default
            # is to grab every CPU thread, which causes heavy oversubscription
            # and thermal throttling. A single query against a 183K DB finishes
            # in well under a second even single-threaded, so 2 is a comfortable
            # middle ground.
            "--threads", "2",
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
    sequences = _load_sequences()
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
            "sequence": sequences.get(accession, ""),
            "organism": entry.get("organism", ""),
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
    """Create a uniprot() callable. Each returned entry is enriched with the
    protein's raw amino-acid sequence (from the SwissProt FASTA) under the
    `sequence` key, on top of the metadata fields in the parsed pickle."""

    def uniprot(accession: str) -> Dict[str, Any]:
        entries = _load_uniprot_dict()
        sequences = _load_sequences()
        entry = entries.get(accession)
        if entry is None:
            return {}
        # Return a shallow copy with the sequence overlaid so we don't mutate
        # the cached pickle-sourced dict.
        enriched = dict(entry)
        enriched["sequence"] = sequences.get(accession, "")
        return enriched

    return uniprot


def make_go_ancestors() -> Callable[[str], List[str]]:
    """Create a go_ancestors() callable."""

    def go_ancestors(go_id: str) -> List[str]:
        dag = load_go_dag()
        return list(dag.get(go_id, []))

    return go_ancestors


def make_sequence_features() -> Callable[[str], Dict[str, Any]]:
    """Create a sequence_features() callable. No shared state; pure computation."""
    return _sequence_features


def make_score() -> Callable[[Dict[str, float], List[str]], Dict[str, Any]]:
    """Create a score(predictions, hypothesized_gt) callable.

    Thin adapter over `RoboPhD.scoring.fmax_with_ancestor_closure`: captures
    the loaded MFO ancestor DAG at factory time and delegates scoring to the
    canonical framework helper. Contains zero F1-sweep logic — the
    computation is shared with the evaluator's per-example scoring path.
    """
    # DAG dict is a module-wide cached singleton via load_go_dag().
    dag = load_go_dag()

    def score(predictions: Dict[str, float], hypothesized_gt: List[str]) -> Dict[str, Any]:
        return fmax_with_ancestor_closure(predictions, hypothesized_gt, dag)

    return score


# ---------------------------------------------------------------------------
# ESM-2 embeddings — precomputed train cache + lazy model
# ---------------------------------------------------------------------------

def _load_esm_cache():
    """Load the precomputed (embeddings_matrix, accessions_list) cache.

    Lazy, once per process. Both outputs are written by
    scripts/compute_esm_embeddings.py during setup.sh step 9.

    Memory: ~470MB resident (183K × 640 float32). Held process-wide.
    """
    global _esm_embeddings, _esm_accessions
    with _esm_cache_lock:
        if _esm_embeddings is not None and _esm_accessions is not None:
            return _esm_embeddings, _esm_accessions
        if not ESM_EMBEDDINGS_NPY.exists() or not ESM_ACCESSIONS_JSON.exists():
            raise FileNotFoundError(
                f"ESM-2 cache not found at {ESM_EMBEDDINGS_NPY} / {ESM_ACCESSIONS_JSON}. "
                f"Run setup.sh (step 9) to precompute embeddings over the ProteInfer-train "
                f"subset of SwissProt."
            )
        import json
        import numpy as np
        logger.info(f"Loading ESM embeddings from {ESM_EMBEDDINGS_NPY}...")
        _esm_embeddings = np.load(ESM_EMBEDDINGS_NPY)
        with open(ESM_ACCESSIONS_JSON) as f:
            _esm_accessions = json.load(f)
        logger.info(f"Loaded ESM cache: {_esm_embeddings.shape[0]} proteins × "
                    f"{_esm_embeddings.shape[1]}-dim")
        return _esm_embeddings, _esm_accessions


def _load_esm_model():
    """Load the ESM-2 150M model into memory.

    Lazy, once per process. First call downloads ~600MB of weights into
    torch.hub cache; subsequent calls reuse the cached weights. Held
    process-wide (~600MB model + ~140MB cached activations).
    """
    global _esm_model, _esm_batch_converter, _esm_repr_layer, _esm_device
    with _esm_model_lock:
        if _esm_model is not None:
            return _esm_model, _esm_batch_converter, _esm_repr_layer, _esm_device
        try:
            import esm
            import torch
        except ImportError as e:
            raise ImportError(
                "fair-esm / torch not installed. Run:\n"
                "    pip install -r examples/protein_go/requirements.txt"
            ) from e

        # Device: cuda > mps > cpu
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        logger.info(f"Loading ESM model {ESM_MODEL_NAME!r} onto {device}...")
        model_fn = getattr(esm.pretrained, ESM_MODEL_NAME)
        model, alphabet = model_fn()
        model.eval()
        model = model.to(device)

        _esm_model = model
        _esm_batch_converter = alphabet.get_batch_converter()
        _esm_repr_layer = model.num_layers  # 30 for the 150M variant
        _esm_device = device
        logger.info(f"ESM model ready (repr_layer={_esm_repr_layer}, "
                    f"embed_dim={model.embed_dim})")
        return _esm_model, _esm_batch_converter, _esm_repr_layer, _esm_device


def _embed_query(sequence: str, max_length: int = 1022):
    """Run ESM-2 forward pass on a single query, return an (embed_dim,)
    L2-normalized float32 numpy vector.

    Results are cached by sha1(sequence[:max_length]) for the lifetime of the
    process, so agents calling both esm_embed() and esm_nearest() on the same
    sequence pay a single forward pass. Forward passes are serialized across
    threads via `_esm_inference_lock` because PyTorch models are not
    thread-safe under concurrent inference (notably on CUDA/MPS).
    """
    import numpy as np
    import torch

    seq = sequence[:max_length] if max_length and len(sequence) > max_length else sequence
    cache_key = hashlib.sha1(seq.encode()).hexdigest()[:16]

    # Fast path: cache hit (no lock needed for reads of immutable arrays)
    cached = _esm_embed_cache.get(cache_key)
    if cached is not None:
        return cached

    model, batch_converter, repr_layer, device = _load_esm_model()

    # Serialize the forward pass: PyTorch modules are not thread-safe.
    # Also prevents duplicate work if multiple threads hit the same cache
    # miss simultaneously (double-checked locking pattern).
    with _esm_inference_lock:
        cached = _esm_embed_cache.get(cache_key)
        if cached is not None:
            return cached
        _, _, tokens = batch_converter([("query", seq)])
        tokens = tokens.to(device)
        with torch.no_grad():
            out = model(tokens, repr_layers=[repr_layer])
        reps = out["representations"][repr_layer]  # (1, seq_len, dim)
        # Residue indices: [1, len(seq)+1) — skip BOS at 0 and EOS at len(seq)+1.
        vec = reps[0, 1 : len(seq) + 1].mean(dim=0)
        vec = vec / (vec.norm() + 1e-12)
        result = vec.cpu().numpy().astype(np.float32)
        _esm_embed_cache[cache_key] = result
        return result


def make_esm_embed() -> Callable[[str], List[float]]:
    """Create an esm_embed(sequence) -> list[float] callable.

    Embeds the query sequence with ESM-2 150M, mean-pools over residues,
    L2-normalizes. Returns a 640-dim list of floats suitable for cosine
    similarity (a dot product on L2-normalized vectors).
    """

    def esm_embed(sequence: str) -> List[float]:
        vec = _embed_query(sequence)
        return vec.tolist()

    return esm_embed


def make_esm_nearest() -> Callable[..., List[Dict[str, Any]]]:
    """Create an esm_nearest(sequence, top_k=50, min_similarity=0.0) callable.

    Embeds the query and returns the top-K nearest ProteInfer-train proteins
    by cosine similarity. Return shape parallels blast() hits — same fields
    (accession, go_terms, description, sequence, organism) with
    cosine_similarity in place of identity — so agents can compose both
    retrievers uniformly. `min_similarity` is the embedding-space analogue
    of blast()'s min_identity: hits below the threshold are dropped before
    the top-K truncation.
    """

    def esm_nearest(
        sequence: str,
        top_k: int = 50,
        min_similarity: float = 0.0,
    ) -> List[Dict[str, Any]]:
        import numpy as np

        embeddings, accessions = _load_esm_cache()
        uniprot_dict = _load_uniprot_dict()
        sequences = _load_sequences()

        query_vec = _embed_query(sequence)
        # embeddings is already L2-normalized row-wise at setup time,
        # and query_vec is L2-normalized by _embed_query — so the
        # dot product equals cosine similarity.
        sims = embeddings @ query_vec  # shape (N,)

        # Filter by threshold first, then top-K. Partition on the
        # surviving subset for efficiency.
        if min_similarity > 0.0:
            mask = sims >= min_similarity
            survivor_idx = np.nonzero(mask)[0]
            if survivor_idx.size == 0:
                return []
            survivor_sims = sims[survivor_idx]
            k = min(top_k, survivor_idx.size)
            part = np.argpartition(-survivor_sims, k - 1)[:k]
            order = part[np.argsort(-survivor_sims[part])]
            idx = survivor_idx[order]
        else:
            k = min(top_k, sims.shape[0])
            idx = np.argpartition(-sims, k - 1)[:k]
            idx = idx[np.argsort(-sims[idx])]

        hits: List[Dict[str, Any]] = []
        for i in idx:
            acc = accessions[int(i)]
            entry = uniprot_dict.get(acc, {})
            hits.append({
                "accession": acc,
                "cosine_similarity": float(sims[int(i)]),
                "go_terms": entry.get("go_terms_mfo", []),
                "description": entry.get("description", ""),
                "sequence": sequences.get(acc, ""),
                "organism": entry.get("organism", ""),
            })
        return hits

    return esm_nearest


# ---------------------------------------------------------------------------
# GO hierarchy helper used by scoring (not exposed to agent)
# ---------------------------------------------------------------------------

def propagate_to_mfo_ancestors(go_terms: set) -> set:
    """Augment a set of GO terms with all MFO ancestors."""
    dag = load_go_dag()
    result = set(go_terms)
    for term in list(go_terms):
        result.update(dag.get(term, []))
    return result
