Each example is a protein: an amino acid sequence (string over 20 letters) and a set of ground-truth GO Molecular Function terms with experimental evidence codes.

GO Molecular Function (GO-MFO) terms describe what a protein does biochemically — e.g. "protein serine/threonine kinase activity" (GO:0004674), "ATP binding" (GO:0005524). Terms are organized in a directed acyclic graph (DAG) where child terms are more specific than parents. A single protein can have multiple terms.

Available tools:

  blast(sequence, top_k=50, min_identity=0.0, min_coverage=0.0) -> list[dict]
    Returns top BLAST hits (DIAMOND against the ProteInfer-train subset of
    SwissProt 2022_01). Served from a precomputed cache; free.
    Each hit:
      accession:       str, UniProt accession (e.g. "P12345")
      identity:        float in [0, 100], % sequence identity
      e_value:         float, BLAST expect value
      query_coverage:  float in [0, 100], % of query covered by alignment
      bit_score:       float, BLAST bit score
      go_terms:        list[str], the hit's GO-MFO terms, MFO-only and
                       experimentally supported (sourced from the hit's
                       uniprot() entry's go_terms_mfo field)
      description:     str, short function comment from SwissProt
      sequence:        str, the hit's amino-acid sequence (same alphabet
                       as the query; useful for few-shot prompting)
    Context: BLAST identity below ~25% is the "twilight zone" where
    homology-based label transfer becomes unreliable.
    Example return:
      [{"accession": "P12345", "identity": 82.1, "e_value": 1.2e-40,
        "query_coverage": 95.0, "bit_score": 312.5,
        "go_terms": ["GO:0004672", "GO:0005524"],
        "description": "Serine/threonine-protein kinase catalyzing...",
        "sequence": "MKVLWAALLV..."},
       ...]

  uniprot(accession) -> dict
    Returns the full parsed SwissProt entry for an accession. Served from
    a precomputed cache; free. Returns an empty dict if the accession is
    not in SwissProt 2022_01.
    Fields:
      name:               str, UniProt entry name (e.g. "KKCC1_HUMAN")
      organism:           str, full species name
      description:        str, FUNCTION comment, truncated to ~500 chars
      length:             int, sequence length in residues
      go_terms_mfo:       list[str], GO-MFO terms with experimental
                          evidence codes only (EXP, IDA, IPI, IMP, IGI,
                          IEP, HTP, HDA, HMP, HGI, HEP, TAS, IC).
                          Evidence codes themselves are not exposed —
                          only the filtered list of IDs.
      all_go_terms_mfo:   list[str], GO-MFO terms with any evidence code
                          (includes IEA/computational annotations)
      deposition_date:    str, YYYY-MM-DD, when the entry was created
      sequence:           str, the amino-acid sequence
    Example return:
      {"name": "MALE_ECOLI", "organism": "Escherichia coli (strain K12)",
       "description": "Involved in the high-affinity maltose membrane...",
       "length": 396, "go_terms_mfo": ["GO:0030247"],
       "all_go_terms_mfo": ["GO:0030247", "GO:0005515"],
       "deposition_date": "1987-08-13", "sequence": "MKIKTGARILALSAL..."}

  go_ancestors(go_id) -> list[str]
    Returns all MFO ancestors of a GO term in the ontology, excluding
    the term itself. Static lookup; free.
    Example: go_ancestors("GO:0004674") -> ["GO:0004672", "GO:0016301",
    "GO:0016740", "GO:0003824", "GO:0003674"]

  sequence_features(sequence) -> dict
    Pure computation; free.
    Fields:
      length:                     int, residue count
      composition:                dict[str, float], per-amino-acid
                                  fraction across the 20 standard AAs
      molecular_weight:           float, Da
      mean_hydrophobicity:        float, Kyte-Doolittle average
      approx_disorder_fraction:   float in [0, 1], sliding-window (n=21)
                                  fraction of residues in segments with
                                  mean hydrophobicity < 0. Crude
                                  heuristic, not a trained disorder
                                  predictor.
    Example return:
      {"length": 396, "composition": {"A": 0.093, "C": 0.002, ...},
       "molecular_weight": 43388.01, "mean_hydrophobicity": -0.024,
       "approx_disorder_fraction": 0.512}

  llm(prompt, temperature=0.0) -> str
    Call a language model. Counts against the cost budget.
    ~$0.003–0.01 per call depending on prompt/response length.

  embed(text) -> list[float]
    Embed text for semantic (English) similarity comparisons via
    text-embedding-3-small. Returns a list[float] of length 1536.
    Counts against the cost budget (~$0.0001 per call).
    This is a *text* embedder — it is not trained on protein sequences
    and will not produce meaningful biochemical similarity over raw
    amino-acid strings.

  score(predictions, hypothesized_gt) -> dict
    Compute the same max-F1-over-thresholds score the evaluator uses,
    but against a user-supplied list of GO terms rather than the real
    ground truth (which the agent never sees). MFO ancestor closure is
    applied to both sides, matching evaluator behavior. Pure
    computation; free.
    Fields of the return dict:
      fmax:                   float in [0, 1], max F1 across thresholds
      best_tau:               float in (0, 1), threshold at which fmax
                              was achieved
      precision_at_best:      float, precision at best_tau
      recall_at_best:         float, recall at best_tau
      TP, FP, FN:             int, confusion counts at best_tau
                              (on the closure sets)
      pred_set_at_best:       list[str], sorted closed prediction set
                              at best_tau
      gt_set:                 list[str], sorted closed ground-truth set
    Example return:
      {"fmax": 1.0, "best_tau": 0.31, "precision_at_best": 1.0,
       "recall_at_best": 1.0, "TP": 5, "FP": 0, "FN": 0,
       "pred_set_at_best": ["GO:0003674", "GO:0003824", "GO:0004672",
                            "GO:0016301", "GO:0016740"],
       "gt_set": ["GO:0003674", "GO:0003824", "GO:0004672",
                  "GO:0016301", "GO:0016740"]}

Scoring: The agent returns a dict mapping GO term IDs (e.g. "GO:0004674") to confidence scores in [0, 1]. Per-protein score is the maximum F1 over confidence thresholds 0.01 to 0.99 (stepped by 0.01), where at each threshold the predicted set is augmented with all GO-MFO ancestors of confident terms and compared against the ground-truth set (which is similarly propagated to ancestors). Empty predictions score 0.0.

Worked example of the scoring computation:
  predictions = {"GO:0004672": 0.70, "GO:0005524": 0.30}
  ground_truth = ["GO:0004672"]

  GO:0004672 MFO ancestors: {GO:0016301, GO:0016740, GO:0003824, GO:0003674}
  GO:0005524 MFO ancestors: {GO:0000166, GO:0036094, GO:0005488, GO:0003674}

  At tau = 0.25 (both terms pass the threshold):
    pred_set (after closure) = {GO:0004672, GO:0005524,
                                GO:0016301, GO:0016740, GO:0003824,
                                GO:0000166, GO:0036094, GO:0005488,
                                GO:0003674}                            # 9 terms
    gt_set   (after closure) = {GO:0004672, GO:0016301, GO:0016740,
                                GO:0003824, GO:0003674}                # 5 terms
    TP = 5, FP = 4, FN = 0
    precision = 5/9 ≈ 0.556,  recall = 5/5 = 1.0,  F1 ≈ 0.714

  At tau = 0.50 (only GO:0004672 passes):
    pred_set (after closure) = {GO:0004672, GO:0016301, GO:0016740,
                                GO:0003824, GO:0003674}                # 5 terms
    gt_set   = same (5 terms)
    TP = 5, FP = 0, FN = 0
    precision = 1.0,  recall = 1.0,  F1 = 1.0

  fmax = 1.0, first achieved at tau = 0.31 (the first threshold at which
  the wrong term GO:0005524 falls below the >=tau cutoff, since its score
  is exactly 0.30). The sweep is `for i in range(1, 100): tau = i / 100`.

A per-protein cost budget of $0.10 is enforced. Correct predictions within budget are scored normally. Predictions that exceed the budget are penalized by a 0.9 multiplier. Only llm() and embed() calls count against the budget; blast(), uniprot(), go_ancestors(), sequence_features(), and score() are free.

Headline benchmark score: After evolution, the best agent is evaluated on the held-out test set using CAFA-evaluator (Piovesan et al., 2024), the official scoring tool for the CAFA challenges. This produces a Fmax number directly comparable to published protein function prediction methods. The per-protein score above is what drives evolution; CAFA Fmax is the final reported metric.

Diagnostics: Any print() output from the agent is captured and included in evaluation diagnostics as agent_stdout. Use print() to log anything you think would be helpful for you to see when improving the agent in later rounds.
