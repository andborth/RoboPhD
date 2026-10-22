import asyncio, json, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))              # model_registry stub
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # agent.py
from inspect_ai.model import ModelName
from inspect_ai.solver import TaskState
from astabench.evals.paper_finder.paper_finder_utils import get_inserted_before_per_dataset_type
from astabench.tools import make_asta_mcp_tools
import agent

CASES = {
  "semantic_7": ("semantic_f1", "What is the performance of large language models in text summarization under reference-based and reference-free human evaluations?", ["254685611"]),
  "semantic_70": ("semantic_f1", "Have any research efforts been made to gather dialogue data via crowdworkers to enhance conversational information retrieval systems?", None),
  "semantic_125": ("semantic_f1", "What are the key references for super-resolution techniques used in measuring diameters in images?", ["271909739","264452520","19089789"]),
  "semantic_43": ("semantic_f1", "Could you direct me to research that evaluates few-shot slot tagging model performance by averaging micro-F1 scores across different test episodes?", None),
  "specific_39": ("specific_f1", "the SPIKE paper", ["219303477","8021519","8778480","942454","13903068"]),
  "semantic_57": ("semantic_f1", "Could you recommend a study that explores how language models are not robust to the surface form editing when testing commonsense knowledge?", ["267770271","218486778","276408140","218486812","277452550","232266176"]),
  "metadata_33": ("metadata_f1", "A SPLASH 2019 and beyond paper that cites any NeurIPS", ["227128926"]),
  "semantic_140": ("semantic_f1", "papers studying LLMs' abilities of generating new research ideas in NLP", ["272463952","277244603"]),
  "metadata_42": ("metadata_f1", 'NeurIPS papers 2022-2023 that cite the "RoBERTa" paper that are cited by at least 30 other paper written by more than 3 authors', ['263829712', '263152603', '260611350', '259262373', '259263947', '259137816', '259108779', '259096157', '259075732', '258959274', '258947152', '258887717', '258841328', '258832444', '258762543', '258888141', '258479968', '258049306', '257687654', '257232924', '257219775', '257079124', '256846990', '256697185', '256627727', '253306024', '253156381', '253115122', '254853701', '254685493', '254043765', '254017395', '252968266', '252693337', '252568062', '252545187', '252544768', '252089100', '252907757', '251135101', '250643915', '250491011', '250244080', '250243785', '249953673', '249889853', '249848279', '249712016', '249674874', '249642722', '249674655', '249538435', '249538460', '249538647', '249395549', '249462026', '249191271', '249151871', '249152172', '249097507', '249097871', '249062606', '248986465', '248965309', '248965287', '248239771', '248227479', '248266822', '246680398', '252873712']),
  "metadata_4": ("metadata_f1", "a Nature portfolio papers by David Harel", ["226312159","14549281","34517208"]),
  "metadata_25": ("metadata_f1", "Papers citing the DistilBERT paper after 2022 with more than 50 citations", None),
  "metadata_26": ("metadata_f1", "paper citing the T5 paper and the spider paper", ["272701638","276775091","276774742","276574850","276574911","276409203","276541757","275570708","275079271","274581681"]),
  "metadata_31": ("metadata_f1", "Journal articles by David Harel with at least 10 citations, citing papers by Gera Weiss, but not self-citations of Gera Weiss", ["2829044","249889772","246679917","208139460","53425415","3672059","13336039","3987221","5362134","16023098","4010707","17960881","1638978","4031399","34893","6904233"]),
  "semantic_220": ("semantic_f1", "What papers discuss the use of transformer architecture in 3d video generation", None),
  "specific_x": ("specific_f1", "the BART paper", ["204960716"]),
  "specific_20": ("specific_f1", "the cnn paper", ["195908774","206594692"]),
  "specific_9": ("specific_f1", "the MS^2 DeYong2021 paper", ["233231380"]),
  "specific_10": ("specific_f1", "the Multi-news fabri2019multinews paper", ["174799390"]),
  "semantic_138": ("semantic_f1", "Has any work been done using large language models (LLMs) to generate blog posts from scientific papers? Please summarize the results.", None),
  "semantic_145": ("semantic_f1", "earliest paper comparing two dialogue systems with win rates judged by humans", ["202538657","216553145"]),
  "semantic_77": ("semantic_f1", "Could you recommend research that evaluates the performance decline in various language models, like BLOOM, under 4-bit integer columnar weight-only quantization?", None),
  "metadata_14": ("metadata_f1", 'NAACL 2010 or 2012 papers co-authored by one of the authors of the "BERT" paper and at least one additional author', ["2468787","1999849","6937668","17192521","1173402","11265565"]),
  "semantic_193": ("semantic_f1", "Adaptive query expansion with LLMs, focusing on papers published on or after 2023", None),
  "semantic_33": ("semantic_f1", "Could you suggest a study that explores employing a beta distribution to sample span sizes within unsupervised learning frameworks for text representation?", ["219530980"]),
}

async def run(case):
    st, q, gold = CASES[case]
    tools = make_asta_mcp_tools(insertion_date=get_inserted_before_per_dataset_type("paper_finder"))
    state = TaskState(model=ModelName("openai/gpt-5.4-mini"), sample_id=case, epoch=0, input=q, messages=[],
                      metadata={"raw_query": q, "score_type": st})
    state.tools = tools
    t0 = time.time()
    state = await agent.make_solver()(state, None)
    out = json.loads(state.output.completion)["output"]["results"]
    ids = [r["paper_id"] for r in out]
    print(f"### {case}: {len(ids)} results in {time.time()-t0:.0f}s; first ids {ids[:5]}")
    if gold:
        hits = len(set(ids) & set(gold)); p = hits/max(1,len(ids)); r = hits/len(gold)
        print(f"### {case}: hits={hits} P={p:.2f} R={r:.2f} F1={(2*p*r/(p+r) if p+r else 0):.3f}")
        for g in gold:
            if g in ids: print(f"### gold {g} at position {ids.index(g)+1}")
    print("### evidence sample:", out[0]["markdown_evidence"][:300] if out else None)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), f"out_{case}_{os.environ.get('TAG','')}.json"),"w"), indent=1)

if __name__ == "__main__":
    asyncio.run(run(sys.argv[1]))
