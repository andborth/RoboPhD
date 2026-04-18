import re


def predict(sequence, blast, uniprot, go_ancestors, sequence_features, llm, embed):
    """Predict GO Molecular Function terms for a protein sequence.

    Returns: dict mapping GO term IDs to confidence scores in [0, 1].
    """
    predictions = {}

    # Start with BLAST: transfer GO-MFO terms from top homologs, weighted by identity
    hits = blast(sequence, top_k=10)
    print(f"Sequence length: {len(sequence)}, BLAST hits: {len(hits)}")
    if hits:
        print(f"Top hit: {hits[0]['accession']} at {hits[0]['identity']:.0f}% identity")

    for hit in hits[:5]:
        confidence = hit["identity"] / 100.0
        for go_id in hit.get("go_terms", []):
            predictions[go_id] = max(predictions.get(go_id, 0), confidence)

    # --- Demonstration: print() output is captured as agent_stdout in diagnostics ---
    print(f"Predictions from BLAST transfer: {len(predictions)} terms")

    # Fallback: if BLAST gave us nothing useful, ask the LLM
    if not predictions or (hits and hits[0]["identity"] < 30):
        print("Weak BLAST evidence, consulting LLM")
        prompt = (
            f"Predict GO Molecular Function terms for this protein.\n"
            f"Sequence (length {len(sequence)}): {sequence[:200]}...\n"
            f"Top BLAST hit: {hits[0]['description'] if hits else 'none'}\n"
            f"Return GO IDs in format GO:XXXXXXX, one per line."
        )
        response = llm(prompt)
        for match in re.finditer(r"GO:\d{7}", response):
            go_id = match.group(0)
            if go_id not in predictions:
                predictions[go_id] = 0.3  # low confidence for LLM-only

    print(f"Final predictions: {len(predictions)} terms")
    return predictions
