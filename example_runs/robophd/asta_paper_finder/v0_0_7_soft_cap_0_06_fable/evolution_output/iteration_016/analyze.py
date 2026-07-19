import json, re, glob, os, collections

STOP=set("the a an of in on for to and or with by from as at is are be been that this these those it its must paper papers focus specifically explicitly which such into how can may their they using use used address discuss describe present including include not".split())
def norm(s): return re.sub(r"[^a-z0-9 ]+"," ",(s or "").lower()).strip()
def words(t): return {w for w in norm(t).split() if len(w)>2 and w not in STOP}

GRADE={"Not Relevant":0,"Somewhat Relevant":1,"Highly Relevant":2,"Perfectly Relevant":3}
rows=[]
for ag in sorted(glob.glob("iteration_015/agent_*")):
    for pd_ in sorted(glob.glob(ag+"/problems/semantic_*")):
        gc=json.load(open(pd_+"/gold_criteria.md"))
        crits=gc["relevance_criteria"]
        sub=json.load(open(pd_+"/submission.json"))
        res=sub["output"]["results"]
        ev={r["paper_id"]:r.get("markdown_evidence","") for r in res}
        # distinctiveness-weighted vocab
        cw=[words(c["name"]+" "+c["description"]) for c in crits]
        df=collections.Counter()
        for s in cw:
            for w in s: df[w]+=1
        verd=open(pd_+"/judge_verdicts.md").read()
        for m in re.finditer(r"^(\d+)\. (\S+) — (Not Relevant|Somewhat Relevant|Highly Relevant|Perfectly Relevant)",verd,re.M):
            pos,pid,g=int(m.group(1)),m.group(2),m.group(3)
            e=ev.get(pid,"")
            if not e or "omitted" in e[:60].lower(): continue
            ew=words(e)
            cov=0
            for j,c in enumerate(crits):
                # distinctive words for j: df==1
                dist={w for w in cw[j] if df[w]==1} or cw[j]
                if dist & ew: cov+=1
            rows.append((os.path.basename(ag),os.path.basename(pd_),pos,GRADE[g],cov,len(crits),len(e),e.count(" ... ")+1))

print("n rows",len(rows))
# grade vs coverage fraction
byc=collections.defaultdict(list)
for r in rows: byc[round(r[4]/r[5],2)].append(r[3])
print("\ncoverage_frac -> mean grade, n, %grade3")
for k in sorted(byc):
    v=byc[k]; print(f"  {k}: mean={sum(v)/len(v):.2f} n={len(v)} pct3={100*sum(1 for x in v if x==3)/len(v):.1f}%")
# grade vs npassages
byp=collections.defaultdict(list)
for r in rows: byp[min(r[7],8)].append(r[3])
print("\nnpassages -> mean grade, n, %grade3")
for k in sorted(byp):
    v=byp[k]; print(f"  {k}: mean={sum(v)/len(v):.2f} n={len(v)} pct3={100*sum(1 for x in v if x==3)/len(v):.1f}%")
# grade vs evidence length bucket
byl=collections.defaultdict(list)
for r in rows: byl[min(r[6]//500,10)].append(r[3])
print("\nevlen/500 -> mean grade, n, %grade3")
for k in sorted(byl):
    v=byl[k]; print(f"  {k}: mean={sum(v)/len(v):.2f} n={len(v)} pct3={100*sum(1 for x in v if x==3)/len(v):.1f}%")
# grade vs submitted position
bypos=collections.defaultdict(list)
for r in rows: bypos[min(r[2]//25,9)].append(r[3])
print("\npos/25 -> mean grade, n, %grade3")
for k in sorted(bypos):
    v=bypos[k]; print(f"  {k}: mean={sum(v)/len(v):.2f} n={len(v)} pct3={100*sum(1 for x in v if x==3)/len(v):.1f}%")
