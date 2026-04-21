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

Scope: by default, reads the entire FASTA into memory up front to enable
length-bucketed batching. That costs O(N) Python-string memory — ~200MB
for 183K SwissProt proteins, fine for this use case. If you ever point
this at a much larger FASTA (UniRef90 has ~100M entries), you'll need to
refactor to stream batches or shard the input. `--no-length-bucketed`
avoids the sort but still materializes everything because the batching
loop reads a pre-built list; a true streaming path would require
restructuring the main loop.

First-time setup: downloads ~600MB of ESM-2 150M weights into the user's
torch.hub cache. Runtime with length-bucketed batching:

  - CPU:       ~4-6 hours (FP32, no GPU hardware)
  - MPS:       ~3-4 hours (FP32 default — FP16 is slower on MPS, see --fp16)
  - CUDA:      ~5-15 minutes (FP16 default)

MPS numbers calibrated on a 48 GB M-series Mac with the default batch
size (48) under plugged-in power; may be slower on battery or with Low
Power Mode enabled.
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
    float32 numpy array of mean-pooled, L2-normalized embeddings.

    If the model weights are FP16, the forward pass runs in FP16; we upcast
    the mean-pooled vector to FP32 before normalization so the denominator
    can't lose precision on small norms. Output is always FP32.
    """
    import torch

    _, _, tokens = batch_converter(batch)
    tokens = tokens.to(device)
    with torch.no_grad():
        out = model(tokens, repr_layers=[repr_layer])
    reps = out["representations"][repr_layer]  # (N, seq_len, dim), dtype matches model

    results = np.empty((len(batch), reps.shape[-1]), dtype=np.float32)
    for i, (_, seq) in enumerate(batch):
        # Tokens include BOS at index 0 and EOS at index len(seq)+1.
        # Mean-pool over the actual residue positions only. Upcast to FP32
        # before the L2 normalize for numerical safety if reps is FP16.
        protein_repr = reps[i, 1 : len(seq) + 1].mean(dim=0).float()
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
    parser.add_argument("--batch-size", type=int, default=48,
                        help="Batch size for embedding. Default picked from an MPS sweep on a "
                             "48 GB M-series Mac (no OOM, best tok/s): FP32 b=48 beat b=32 "
                             "(+2%%) and b=64 (+6%%); MPS appears to saturate somewhere between "
                             "48 and 64. Works well with length-bucketed order (every batch "
                             "pads to a near-uniform length). Lower if you hit OOM.")
    parser.add_argument("--length-bucketed", action=argparse.BooleanOptionalAction, default=True,
                        help="Sort all sequences by length descending before batching, so each "
                             "batch pads to a near-uniform length. Dramatic speedup over unsorted "
                             "order (3-5x on MPS/CUDA) because random-order batching forces every "
                             "batch to pad to the longest sequence in it. Disable with "
                             "--no-length-bucketed to preserve FASTA order (slower; mostly useful "
                             "for debugging).")
    parser.add_argument("--max-length", type=int, default=1022,
                        help="Truncate sequences to this length (excluding BOS/EOS). "
                             "ESM-2 was trained on contexts up to 1024 tokens total.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                        help="Torch device. 'auto' picks cuda > mps > cpu.")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None,
                        help="Use FP16 forward pass. Default: auto — FP16 on CUDA (~2x "
                             "throughput), FP32 on MPS and CPU. An MPS sweep (48GB M-series "
                             "Mac, ESM-2 150M) measured FP16 consistently ~16%% SLOWER than "
                             "FP32 across batch sizes 32/48/64 due to PyTorch-MPS-transformer "
                             "op-boundary up/downcast overhead (known PyTorch MPS quirk). On "
                             "CPU, no FP16 hardware exists so FP32 always wins. Use "
                             "--fp16 / --no-fp16 to override the auto choice explicitly.")
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

    # FP16 resolution:
    #   --fp16 (explicit): force FP16 on GPU/MPS; no-op on CPU
    #   --no-fp16 (explicit): force FP32 everywhere
    #   (auto, default): FP16 on CUDA (2x speedup), FP32 on MPS (known
    #     PyTorch-MPS-transformer quirk makes FP16 ~16% slower than FP32 at
    #     every batch size we measured — see --fp16 help), FP32 on CPU.
    if args.fp16 is None:
        use_fp16 = device == "cuda"
        reason = "auto"
    else:
        use_fp16 = args.fp16 and device != "cpu"
        reason = "explicit --fp16" if args.fp16 else "explicit --no-fp16"

    if use_fp16:
        model = model.half()
        logger.info(f"Model converted to FP16 ({device}, {reason})")
    else:
        logger.info(f"Model kept in FP32 ({device}, {reason})")

    batch_converter = alphabet.get_batch_converter()
    repr_layer = model.num_layers  # final layer (30 for 150M)
    logger.info(f"Model loaded in {time.time() - t0:.1f}s; "
                f"repr_layer={repr_layer}, embed_dim={model.embed_dim}")

    # Pre-create the output dir
    args.output_embeddings.parent.mkdir(parents=True, exist_ok=True)

    # Read all records up front so we can optionally sort by length for
    # padding-efficient batching. Memory cost: ~200MB to hold all sequences
    # as Python strings (183K sequences averaging ~400 chars), negligible
    # vs the ~600MB ESM model weights and ~1GB per-batch activation peak.
    logger.info(f"Reading sequences from {args.input_fasta}...")
    all_records: List[Tuple[str, str]] = []
    for acc, seq in stream_fasta(args.input_fasta):
        if args.max_length and len(seq) > args.max_length:
            seq = seq[: args.max_length]
        all_records.append((acc, seq))
    total = len(all_records)
    if total == 0:
        sys.exit(f"No sequences read from {args.input_fasta}")
    logger.info(f"Read {total} sequences "
                f"(min_len={min(len(s) for _, s in all_records)}, "
                f"max_len={max(len(s) for _, s in all_records)})")

    # Length-bucketed batching: sort by length descending so every batch
    # pads to a near-uniform length. With random FASTA order, one 1022-
    # residue outlier in a batch forces all 31 others to be processed as
    # though they were 1022 residues too; the waste compounds. Output
    # accession/embedding order then reflects sort order — callers index
    # by (accession -> row) via the accessions JSON, so order is not
    # semantically meaningful.
    if args.length_bucketed:
        logger.info("Sorting sequences by length for padding-efficient batching...")
        all_records.sort(key=lambda x: len(x[1]), reverse=True)
        logger.info("Note: longest-first means the first batches are slowest; "
                    "throughput will accelerate as batches shorten.")

    accessions: List[str] = []
    all_embeddings: List[np.ndarray] = []

    # Pre-compute total token work for an honest ETA. Length-bucketed
    # order means seq/sec rate is NOT constant: it starts low (longest
    # sequences first) and climbs as batches shorten. Tokens-of-work
    # per second, however, stays roughly constant — so a token-based ETA
    # gives an accurate projection from iteration 1. +2 per sequence for
    # BOS/EOS tokens, matching what batch_converter adds.
    total_tokens = sum(len(seq) + 2 for _, seq in all_records)
    logger.info(f"Total tokens to process: {total_tokens / 1e6:.1f}M "
                f"(avg {total_tokens / total:.0f} per sequence)")

    # Log at explicit milestones so the predicate is correct regardless of
    # whether batch_size happens to divide log_every evenly.
    next_log_at = args.log_every

    n_seen = 0
    tokens_done = 0
    t_start = time.time()
    for start in range(0, total, args.batch_size):
        batch = all_records[start : start + args.batch_size]
        vecs = embed_batch(batch, model, batch_converter, repr_layer, device)
        all_embeddings.append(vecs)
        for acc, _ in batch:
            accessions.append(acc)
        n_seen += len(batch)
        tokens_done += sum(len(seq) + 2 for _, seq in batch)
        if n_seen >= next_log_at or n_seen == total:
            elapsed = max(time.time() - t_start, 1e-6)
            seq_rate = n_seen / elapsed
            tok_rate = tokens_done / elapsed
            remaining_tokens = total_tokens - tokens_done
            eta_sec = remaining_tokens / max(tok_rate, 1e-6)
            logger.info(f"Embedded {n_seen}/{total} "
                        f"({seq_rate:.1f} seq/s, {tok_rate:.0f} tok/s, "
                        f"current_batch_len={len(batch[0][1])}, "
                        f"ETA {eta_sec / 60:.1f} min)")
            # Advance past whatever milestone we just crossed.
            next_log_at = ((n_seen // args.log_every) + 1) * args.log_every

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
