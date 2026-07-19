src = open("agent.py").read()
end = src.index('"""', 3) + 3
newdoc = '''"""iter20_cite_proof: PaperFindingBench solver, iteration 20.

Base = iter18_triage_first with the SEMANTIC PIPELINE UNCHANGED (it won the
batch score in iterations 18 and 19 and is the cheapest of the line at
~$0.067/semantic query, ~$0.052 projected at the 73%-semantic test mix).
Iteration 19 tested the last open semantic question — reverting iter18's
retrieval stack to iter13's — and LOST by 3.3 points; the grade-3
attribution shows got_it in a three-way tie (54.1/53.8/53.8%), so the
retrieval-stack question is resolved as noise and this iteration touches
nothing on the semantic side.

WHAT CHANGED: three deterministic fixes on the exact-match paths, all free
(tool calls only, zero new LLM spend):

1. BODY-MENTION CITATION VERIFICATION (fixes metadata_42-type, 0.053 with a
   ~0.5 counterfactual). The refs check via get_paper_batch(references) is a
   broken instrument: on metadata_42 it returned reference lists for 67/72
   candidates but matched the target in only ~1 — S2 reference lists arrive
   truncated or id-less, so the check false-negatives at scale and
   "reference verification: 72 -> 6" discarded a candidate set that covered
   a 70-paper gold. New acceptance channel: scoped
   snippet_search(query=<short name>, paper_ids=<chunk of 25>, limit=100),
   accept a candidate iff a returned passage literally contains the cited
   work's short name (normalized, word-bounded). A paper that names
   "RoBERTa" in its body after being retrieved by a RoBERTa keyword search
   all but certainly cites it.

2. CONJUNCTION AUGMENTATION UNDER THE CITER CAP (metadata_26-type, 0.000
   for every agent). get_citations is recency-ordered and capped at 1000;
   on "papers citing the T5 paper and the spider paper" both lists cap, and
   the gold (all corpus_id 272M-276M, ~Oct 2024-Feb 2025) sits in an OLDER
   recency window that has scrolled out of the cap at eval time — the pure
   intersection can never see it. When a multi-target citing query hits the
   cap and the intersection is small (<40), add a mention-conjunction
   channel (keyword searches on the joined short names, a global snippet
   search "both A and B"), verify candidates by requiring body passages
   mentioning EVERY target (per-target scoped snippet verification,
   intersected), and admit verified extras up to 40 total. Bounded
   downside: these queries score 0.000 today.

3. _batch_bisect: EVERY get_paper_batch site now bisects on chunk failure
   (the _fetch_references pattern, factored out). Observed on metadata_42:
   the metadata backfill failed BOTH attempts on "Paper ... is newer than
   the date cutoff" — one poison id deleted 50 ids' metadata, and docs with
   citationCount=None/authors=None were then silently dropped by the cheap
   filters. Applies to the metadata backfill, _fill_abstracts, and the
   citation-expansion metadata fetch.

Inherited unchanged from iter18: 14-query diverse-category planner, POOL_CAP
320 lexical prescreen, compact stage-1 triage, citation expansion, stage-2
judge sim, grade-2 rescue, GPT_5_4 head verify, band ordering, longer
evidence cuts (abstract 2000 / snippets 900) with containment dedup, tail
sweep to 250, the specific path (retrieve wide, submit tight), the chunked
venue filter, cites_paper_titles conjunction parsing with citer-set
intersection, per-stage llm-usage telemetry.

Scoring context (re-confirmed on iteration-19 verdicts): the judge grades
exactly the first K submitted positions (K observed 12-228); recall counts
only grade-3s in that prefix; grade 3 needs EVERY weighted criterion judged
Perfectly Relevant from the submitted evidence alone. On exact-match paths
order never matters and F1 = harmonic(hits/#submitted, hits/#gold) — with a
large gold set, discarding plausible candidates is the costliest error.
"""'''
open("agent.py", "w").write(newdoc + src[end:])
print("docstring replaced")
