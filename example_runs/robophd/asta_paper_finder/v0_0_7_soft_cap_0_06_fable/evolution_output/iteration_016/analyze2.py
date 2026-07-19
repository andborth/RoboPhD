import json,re,glob,os,collections
GRADE={"Not Relevant":0,"Somewhat Relevant":1,"Highly Relevant":2,"Perfectly Relevant":3}
data={}
for ag in sorted(glob.glob("iteration_015/agent_*")):
    for pd_ in sorted(glob.glob(ag+"/problems/semantic_*")):
        verd=open(pd_+"/judge_verdicts.md").read()
        gs=[]
        for m in re.finditer(r"^(\d+)\. (\S+) — (.+)$",verd,re.M):
            g=m.group(3).replace(" (known-good)","").replace(" (known-bad)","").strip()
            if g in GRADE: gs.append((int(m.group(1)),GRADE[g]))
        data[(os.path.basename(ag),os.path.basename(pd_))]=gs

print("Within-query: grade-3 rate by DECILE of the judged prefix")
dec=collections.defaultdict(list)
for k,gs in data.items():
    n=len(gs)
    if n<20: continue
    for pos,g in gs:
        d=min(9,int(10*(pos-1)/n))
        dec[d].append(1 if g==3 else 0)
for d in sorted(dec):
    v=dec[d]; print(f"  decile {d}: pct3={100*sum(v)/len(v):5.1f}%  n={len(v)}")

print("\nWithin-query: mean grade by decile")
dec2=collections.defaultdict(list)
for k,gs in data.items():
    n=len(gs)
    if n<20: continue
    for pos,g in gs: dec2[min(9,int(10*(pos-1)/n))].append(g)
for d in sorted(dec2):
    v=dec2[d]; print(f"  decile {d}: mean={sum(v)/len(v):.2f} n={len(v)}")

print("\nPer-query grade-3 count in first half vs second half of judged prefix")
for k in sorted(data):
    gs=data[k]; n=len(gs)
    if n<20: continue
    h=n//2
    a=sum(1 for p,g in gs if p<=h and g==3); b=sum(1 for p,g in gs if p>h and g==3)
    print(f"  {k[0][6:26]:22s} {k[1]:14s} K={n:3d}  first-half 3s={a:3d}  second-half 3s={b:3d}")
