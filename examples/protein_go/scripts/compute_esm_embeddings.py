#!/usr/bin/env python3
"""Precompute ESM-2 150M embeddings for every sequence in a FASTA.

Used by setup.sh step 9 to populate the protein_go example's embedding
cache over the ProteInfer-train subset of SwissProt (swissprot_train.fasta,
~183K proteins). The cache is queried at runtime by the agent's
``esm_nearest`` tool as the BLAST-analogue in embedding space.

Outputs a (N, 640) float32 numpy matrix plus a JSON index of N accessions
(row-aligned). Mean-pools over the residue-level ESM-2 representation at
layer 30 (the final hidden state) and L2-normalizes each row so cosine
similarity is a plain dot product at query time.

First-time setup: downloads ~600MB of ESM-2 150M weights into the user's
torch.hub cache. Runtime: ~45-90min on laptop CPU, minutes on CUDA/MPS.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def stream_fasta(fasta_path: Path) -> Iterator[Tuple[str, str]]:
    """Yield (accession, sequence) from a SwissProt FASTA.

    Accession is the pipe-delimited UniProt accession from the header, e.g.
    '>sp|P12345|NAME_SPECIES ...' yields 'P12345'. Malformed headers are
    skipped.
    """
    accession: str | None = None
    seq_parts: List[str] = []
    with open(fasta_path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if accession and seq_parts:
                    yield accession, "".join(seq_parts)
                parts = line[1:].split("|")
                accession = parts[1] if len(parts) >= 2 else None
                seq_parts = []
            elif accession is not None:
                seq_parts.append(line)
        if accession and seq_parts:
            yield accession, "".join(seq_parts)


def pick_device(preference: str) -> str:
    import torch
    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def embed_batch(
    batch: List[Tuple[str, str]],
    model,
    batch_converter,
    repr_layer: int,
    device: str,
) -> np.ndarray:
    """Embed one batch of (accession, sequence) pairs and return an (N, dim)
    float32 numpy array of mean-pooled, L2-normalized embeddings."""
    import torch

    _, _, tokens = batch_converter(batch)
    tokens = tokens.to(device)
    with torch.no_grad():
        out = model(tokens, repr_layers=[repr_layer])
    reps = out["representations"][repr_layer]  # (N, seq_len, dim)

    results = np.empty((len(batch), reps.shape[-1]), dtype=np.float32)
    for i, (_, seq) in enumerate(batch):
        # Tokens include BOS at index 0 and EOS at index len(seq)+1.
        # Mean-pool over the actual residue positions only.
        protein_repr = reps[i, 1 : len(seq) + 1].mean(dim=0)
        protein_repr = protein_repr / (protein_repr.norm() + 1e-12)
        results[i] = protein_repr.cpu().numpy().astype(np.float32)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-fasta", required=True, type=Path,
                        help="FASTA to embed (e.g. data/swissprot_train.fasta)")
    parser.add_argument("--output-embeddings", required=True, type=Path,
                        help="Output .npy file, shape (N, 640) float32")
    parser.add_argument("--output-accessions", required=True, type=Path,
                        help="Output .json file, row-aligned list of accessions")
    parser.add_argument("--model-name", default="esm2_t30_150M_UR50D",
                        help="fair-esm pretrained model name. Default: 150M params, 640-dim.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for embedding. Lower if you hit OOM.")
    parser.add_argument("--max-length", type=int, default=1022,
                        help="Truncate sequences to this length (excluding BOS/EOS). "
                             "ESM-2 was trained on contexts up to 1024 tokens total.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                        help="Torch device. 'auto' picks cuda > mps > cpu.")
    parser.add_argument("--log-every", type=int, default=500,
                        help="Log progress every N sequences.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    if not args.input_fasta.exists():
        sys.exit(f"Input FASTA not found: {args.input_fasta}")

    try:
        import esm  # fair-esm package
        import torch
    except ImportError:
        sys.exit(
            "fair-esm / torch not installed. Run:\n"
            "    pip install -r examples/protein_go/requirements.txt"
        )

    device = pick_device(args.device)
    logger.info(f"Loading ESM model {args.model_name!r} onto {device}...")
    t0 = time.time()
    model_fn = getattr(esm.pretrained, args.model_name, None)
    if model_fn is None:
        sys.exit(f"Unknown ESM model: {args.model_name!r}")
    model, alphabet = model_fn()
    model.eval()
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()
    repr_layer = model.num_layers  # final layer (30 for 150M)
    logger.info(f"Model loaded in {time.time() - t0:.1f}s; "
                f"repr_layer={repr_layer}, embed_dim={model.embed_dim}")

    # Pre-create the output dir
    args.output_embeddings.parent.mkdir(parents=True, exist_ok=True)

    # Stream FASTA, collect accessions, embed in batches.
    accessions: List[str] = []
    all_embeddings: List[np.ndarray] = []
    batch: List[Tuple[str, str]] = []

    def flush_batch() -> None:
        if not batch:
            return
        vecs = embed_batch(batch, model, batch_converter, repr_layer, device)
        all_embeddings.append(vecs)
        for acc, _ in batch:
            accessions.append(acc)

    n_seen = 0
    t_start = time.time()
    for acc, seq in stream_fasta(args.input_fasta):
        if args.max_length and len(seq) > args.max_length:
            seq = seq[: args.max_length]
        batch.append((acc, seq))
        n_seen += 1
        if len(batch) >= args.batch_size:
            flush_batch()
            batch = []
        if n_seen % args.log_every == 0:
            rate = n_seen / max(time.time() - t_start, 1e-6)
            logger.info(f"Embedded {n_seen} sequences ({rate:.1f}/s)")
    flush_batch()

    if not all_embeddings:
        sys.exit(f"No sequences read from {args.input_fasta}")

    matrix = np.concatenate(all_embeddings, axis=0)
    assert matrix.shape[0] == len(accessions), (
        f"Shape mismatch: {matrix.shape[0]} rows vs {len(accessions)} accessions"
    )

    logger.info(f"Writing embeddings to {args.output_embeddings} "
                f"({matrix.shape[0]} x {matrix.shape[1]} {matrix.dtype})...")
    np.save(args.output_embeddings, matrix)

    logger.info(f"Writing accessions to {args.output_accessions}...")
    with open(args.output_accessions, "w") as f:
        json.dump(accessions, f)

    logger.info(f"Done in {time.time() - t_start:.1f}s total "
                f"(avg {n_seen / max(time.time() - t_start, 1e-6):.1f}/s)")


if __name__ == "__main__":
    main()
