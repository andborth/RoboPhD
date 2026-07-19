import json,re,glob,os,collections
GRADE={"Not Relevant":0,"Somewhat Relevant":1,"Highly Relevant":2,"Perfectly Relevant":3}
STOP=set("the a an of in on for to and or with by from as at is are be been that this these those it its must paper papers focus specifically explicitly which such into how can may their they using use used address discuss describe present including include not".split())
def norm(s): return re.sub(r"[^a-z0-9 ]+"," ",(s or "").lower()).strip()
def words(t): return {w for w in norm(t).split() if len(w)>2 and w not in STOP}
tab=collections.defaultdict(dict); crit={}
for ag in sorted(glob.glob("iteration_015/agent_*")):
    A=os.path.basename(ag)[6:]
    for pd_ in sorted(glob.glob(ag+"/problems/semantic_*")):
        P=os.path.basename(pd_)
        crit[P]=json.load(open(pd_+"/gold_criteria.md"))["relevance_criteria"]
        ev={r["paper_id"]:r.get("markdown_evidence","") for r in json.load(open(pd_+"/submission.json"))["output"]["results"]}
        for m in re.finditer(r"^(\d+)\. (\S+) — (.+)$",open(pd_+"/judge_verdicts.md").read(),re.M):
            g=m.group(3).replace(" (known-good)","").replace(" (known-bad)","").strip()
            if g in GRADE: tab[(P,m.group(2))][A]=(GRADE[g],ev.get(m.group(2),""))

def feats(P,e):
    ps=[p.strip() for p in e.split(" ... ") if p.strip()]
    cw=set()
    for c in crit[P]: cw|=words(c["name"]+" "+c["description"])
    # per-passage relevance
    rels=[len(words(p)&cw)/max(1,len(cw)) for p in ps]
    trunc=sum(1 for p in ps if p and not p.rstrip().endswith(('.','!','?',')')))
    return dict(n=len(ps),avglen=sum(len(p) for p in ps)/max(1,len(ps)),
                maxlen=max((len(p) for p in ps),default=0),
                minrel=min(rels) if rels else 0, avgrel=sum(rels)/max(1,len(rels)),
                maxrel=max(rels) if rels else 0, trunc_frac=trunc/max(1,len(ps)),
                total=len(e))

flip=[(k,v) for k,v in tab.items() if len(v)>=2 and max(x[0] for x in v.values())==3 and min(x[0] for x in v.values())<3]
agg=collections.defaultdict(list)
for k,v in flip:
    hi=max(v.values(),key=lambda x:x[0]); lo=min(v.values(),key=lambda x:x[0])
    fh,fl=feats(k[0],hi[1]),feats(k[0],lo[1])
    for f in fh: agg[f].append(fh[f]-fl[f])
print(f"PAIRED (n={len(flip)} grade3-vs-lower flips of the SAME paper+query):")
print(f"{'feature':12s} {'mean diff':>12s} {'% grade3 higher':>16s}")
for f,v in agg.items():
    pos=100*sum(1 for x in v if x>0)/max(1,sum(1 for x in v if x!=0))
    print(f"  {f:10s} {sum(v)/len(v):+12.3f} {pos:15.0f}%")

# also 0/1 vs 2/3 split
flip2=[(k,v) for k,v in tab.items() if len(v)>=2 and max(x[0] for x in v.values())>=2 and min(x[0] for x in v.values())<=1]
agg2=collections.defaultdict(list)
for k,v in flip2:
    hi=max(v.values(),key=lambda x:x[0]); lo=min(v.values(),key=lambda x:x[0])
    fh,fl=feats(k[0],hi[1]),feats(k[0],lo[1])
    for f in fh: agg2[f].append(fh[f]-fl[f])
print(f"\nPAIRED (n={len(flip2)} high-vs-low(0/1) flips):")
for f,v in agg2.items():
    pos=100*sum(1 for x in v if x>0)/max(1,sum(1 for x in v if x!=0))
    print(f"  {f:10s} {sum(v)/len(v):+12.3f} {pos:15.0f}%")
