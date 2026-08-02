import asyncio, sys, json
sys.path.insert(0, ".")
import agent as A

QS = [
 ("Papers citing the DistilBERT paper after 2022 with more than 50 citations","metadata_f1"),
 ('NeurIPS papers 2022-2023 that cite the "RoBERTa" paper that are cited by at least 30 other paper written by more than 3 authors',"metadata_f1"),
 ("paper citing the T5 paper and the spider paper","metadata_f1"),
 ("What are the common model architectures for retrieval-augmented language models?","semantic_f1"),
 ("I am looking for research papers on the construction of multimodal foundation models that support both visual and audio inputs. These models should be pre-trained on large-scale datasets, including visual, audio, and audio-visual data. Please exclude survey papers.","semantic_f1"),
]

async def main():
    for q, st in QS:
        p = await A.analyze(q, st)
        print("Q:", q[:70])
        print("  crit:", [(c["name"], round(c["weight"],2)) for c in p["criteria"]])
        print("  ymin/ymax/years:", p["year_min"], p["year_max"], p["years"],
              "| min_cit:", p["min_citations"], "| authors lo/hi:", p["min_authors"], p["max_authors"],
              "| venues:", p["venues"], "| cites_ref:", p["cites_reference"])
        print("  hard:", A._hard_filters(p))
asyncio.run(main())
