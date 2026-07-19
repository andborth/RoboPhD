import json,re,glob,os,collections
GRADE={"Not Relevant":0,"Somewhat Relevant":1,"Highly Relevant":2,"Perfectly Relevant":3}
prob=collections.defaultdict(lambda: collections.defaultdict(dict))  # P -> agent -> pid -> (pos,grade)
order=collections.defaultdict(dict)  # P -> agent -> [pids in submitted order]
K={}
for ag in sorted(glob.glob("iteration_015/agent_*")):
    A=os.path.basename(ag)[6:]
    for pd_ in sorted(glob.glob(ag+"/problems/semantic_*")):
        P=os.path.basename(pd_)
        order[P][A]=[r["paper_id"] for r in json.load(open(pd_+"/submission.json"))["output"]["results"]]
        n=0
        for m in re.finditer(r"^(\d+)\. (\S+) — (.+)$",open(pd_+"/judge_verdicts.md").read(),re.M):
            g=m.group(3).replace(" (known-good)","").replace(" (known-bad)","").strip()
            if g in GRADE:
                prob[P][A][m.group(2)]=(int(m.group(1)),GRADE[g]); n=int(m.group(1))
        K[P]=max(K.get(P,0),n)

print(f"{'problem':14s} {'K':>4s} {'agent':22s} {'P@K':>5s} {'known3':>7s} {'stranded':>9s} {'ceil_recall':>12s} {'act_recall':>11s}")
for P in sorted(prob):
    known3={pid for A in prob[P] for pid,(po,g) in prob[P][A].items() if g==3}
    for A in sorted(prob[P]):
        pk=sum(1 for pid,(po,g) in prob[P][A].items() if g==3)
        sub=order[P][A]; idx={pid:i for i,pid in enumerate(sub)}
        # known-3 papers this agent submitted but placed beyond its judged depth
        depth=max((po for po,g in prob[P][A].values()),default=0)
        stranded=sum(1 for pid in known3 if pid in idx and idx[pid]>=depth)
        ceil=min(len(known3),K[P])/K[P]
        print(f"{P:14s} {K[P]:4d} {A:22s} {pk:5d} {len(known3):7d} {stranded:9d} {ceil:12.3f} {pk/K[P]:11.3f}")
