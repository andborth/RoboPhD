Each example is a protein: an amino acid sequence (string over 20 letters) and a set of ground-truth GO Molecular Function terms with experimental evidence codes.

GO Molecular Function (GO-MFO) terms describe what a protein does biochemically — e.g. "protein serine/threonine kinase activity" (GO:0004674), "ATP binding" (GO:0005524). Terms are organized in a directed acyclic graph (DAG) where child terms are more specific than parents. A single protein can have multiple terms.

Available tools:

  blast(sequence, top_k=50, min_identity=0.0, min_coverage=0.0) -> list[dict]
    Returns top BLAST hits against SwissProt. Each hit is a dict with:
    accession, identity (0-100), e_value, query_coverage (0-100), bit_score,
    go_terms (list of GO:XXXXXXX strings, MFO-only, experimentally supported),
    description (short text from SwissProt).
    Served from a precomputed cache; free.

  uniprot(accession) -> dict
    Returns full SwissProt entry for an accession. Contains: name, organism,
    go_terms_mfo (full list with evidence codes), description (function comment),
    length. Served from a precomputed cache; free.

  go_ancestors(go_id) -> list[str]
    Returns all MFO ancestors of a GO term in the ontology, excluding the term
    itself. Static lookup; free.

  sequence_features(sequence) -> dict
    Returns composition percentages, length, molecular weight, hydrophobicity
    profile. Pure computation; free.

  llm(prompt, temperature=0.0) -> str
    Call a language model. Expensive (~$0.003-0.01 per call).

  embed(text) -> list[float]
    Embed text for similarity comparisons. Cheap (~$0.0001 per call).

Scoring: The agent returns a dict mapping GO term IDs (e.g. "GO:0004674") to confidence scores in [0, 1]. Per-protein score is the maximum F1 over confidence thresholds 0.01 to 0.99, where at each threshold the predicted set is augmented with all GO-MFO ancestors of confident terms and compared against the ground-truth set (which is similarly propagated to ancestors). Empty predictions score 0.0.

A per-protein cost budget of $0.10 is enforced. Correct predictions within budget are scored normally. Predictions that exceed the budget are penalized by a 0.9 multiplier. Only llm() and embed() calls count against the budget; blast(), uniprot(), go_ancestors(), and sequence_features() are free.

Headline benchmark score: After evolution, the best agent is evaluated on the held-out test set using CAFA-evaluator (Piovesan et al., 2024), the official scoring tool for the CAFA challenges. This produces a Fmax number directly comparable to published protein function prediction methods. The per-protein score above is what drives evolution; CAFA Fmax is the final reported metric.

Diagnostics: Any print() output from the agent is captured and included in evaluation diagnostics as agent_stdout. Use print() to log anything you think would be helpful for you to see when improving the agent in later rounds.
