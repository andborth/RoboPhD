import json,re,glob,os,collections
GRADE={"Not Relevant":0,"Somewhat Relevant":1,"Highly Relevant":2,"Perfectly Relevant":3}
tab=collections.defaultdict(dict)
for ag in sorted(glob.glob("iteration_015/agent_*")):
    A=os.path.basename(ag)[6:]
    for pd_ in sorted(glob.glob(ag+"/problems/semantic_*")):
        P=os.path.basename(pd_)
        sub=json.load(open(pd_+"/submission.json"))
        ev={r["paper_id"]:r.get("markdown_evidence","") for r in sub["output"]["results"]}
        for m in re.finditer(r"^(\d+)\. (\S+) — (.+)$",open(pd_+"/judge_verdicts.md").read(),re.M):
            g=m.group(3).replace(" (known-good)","").replace(" (known-bad)","").strip()
            if g in GRADE: tab[(P,m.group(2))][A]=(GRADE[g],ev.get(m.group(2),""))
flip=[(k,v) for k,v in tab.items() if len(v)>=2 and max(x[0] for x in v.values())==3 and min(x[0] for x in v.values())<3]
shown=0
for k,v in flip:
    hi=max(v.values(),key=lambda x:x[0]); lo=min(v.values(),key=lambda x:x[0])
    hp=[p.strip() for p in hi[1].split(" ... ")]; lp=[p.strip() for p in lo[1].split(" ... ")]
    only_hi=[p for p in hp if p not in lp]
    if shown<4 and only_hi:
        print(f"\n##### {k[0]} paper {k[1]}  grade3-evidence vs grade{lo[0]}-evidence")
        print(f"  grade3 has {len(hp)} passages, lower has {len(lp)}")
        print(f"  PASSAGES ONLY IN GRADE-3 VERSION ({len(only_hi)}):")
        for p in only_hi[:3]: print(f"    - {p[:260]}")
        onlylo=[p for p in lp if p not in hp]
        print(f"  PASSAGES ONLY IN LOWER VERSION ({len(onlylo)}):")
        for p in onlylo[:2]: print(f"    - {p[:200]}")
        shown+=1
# aggregate: does grade3 version have MORE distinct passages beyond title/tldr/abstract?
more=0;less=0;eq=0
for k,v in flip:
    hi=max(v.values(),key=lambda x:x[0]); lo=min(v.values(),key=lambda x:x[0])
    a=len([p for p in hi[1].split(" ... ") if p.strip() not in lo[1]])
    b=len([p for p in lo[1].split(" ... ") if p.strip() not in hi[1]])
    if a>b: more+=1
    elif b>a: less+=1
    else: eq+=1
print(f"\n\nAcross {len(flip)} flips: grade3 version had MORE unique passages in {more}, FEWER in {less}, equal {eq}")
