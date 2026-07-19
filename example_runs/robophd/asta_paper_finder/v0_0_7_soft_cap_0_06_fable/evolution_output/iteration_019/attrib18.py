import json,re,glob,os,collections
GRADE={"Not Relevant":0,"Somewhat Relevant":1,"Highly Relevant":2,"Perfectly Relevant":3}
prob=collections.defaultdict(lambda: collections.defaultdict(dict)); order=collections.defaultdict(dict)
for ag in sorted(glob.glob("iteration_018/agent_*")):
    A=os.path.basename(ag)[6:]
    for pd_ in sorted(glob.glob(ag+"/problems/semantic_*")):
        P=os.path.basename(pd_)
        order[P][A]=[r["paper_id"] for r in json.load(open(pd_+"/submission.json"))["output"]["results"]]
        for m in re.finditer(r"^(\d+)\. (\S+) — (.+)$",open(pd_+"/judge_verdicts.md").read(),re.M):
            g=m.group(3).replace(" (known-good)","").replace(" (known-bad)","").strip()
            if g in GRADE: prob[P][A][m.group(2)]=(int(m.group(1)),GRADE[g])
agents=sorted({A for P in prob for A in prob[P]})
for A in agents:
    tot=collections.Counter()
    print(f"\n===== {A} =====")
    for P in sorted(prob):
        known3={pid for B in prob[P] for pid,(po,g) in prob[P][B].items() if g==3}
        depth=max((po for po,g in prob[P][A].values()),default=0)
        idx={pid:i+1 for i,pid in enumerate(order[P][A])}
        c=collections.Counter()
        for pid in known3:
            if pid not in idx: c["not_retrieved"]+=1
            elif idx[pid]>depth: c["stranded_below_K"]+=1
            elif prob[P][A].get(pid,(0,0))[1]==3: c["got_it"]+=1
            else: c["evidence_lost"]+=1
        tot.update(c)
        print(f"{P:14s} known3={len(known3):3d}  got={c['got_it']:3d} ev_lost={c['evidence_lost']:3d} stranded={c['stranded_below_K']:3d} not_retr={c['not_retrieved']:3d}")
    n=sum(tot.values()) or 1
    for k,v in tot.most_common(): print(f"  {k:18s} {v:4d}  {100*v/n:5.1f}%")
