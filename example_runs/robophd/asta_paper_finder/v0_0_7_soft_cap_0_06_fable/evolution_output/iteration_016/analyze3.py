import json,re,glob,os,collections
GRADE={"Not Relevant":0,"Somewhat Relevant":1,"Highly Relevant":2,"Perfectly Relevant":3}
# per (problem,pid) -> {agent: (grade, evidence)}
tab=collections.defaultdict(dict)
for ag in sorted(glob.glob("iteration_015/agent_*")):
    A=os.path.basename(ag)[6:]
    for pd_ in sorted(glob.glob(ag+"/problems/semantic_*")):
        P=os.path.basename(pd_)
        sub=json.load(open(pd_+"/submission.json"))
        ev={r["paper_id"]:r.get("markdown_evidence","") for r in sub["output"]["results"]}
        for m in re.finditer(r"^(\d+)\. (\S+) — (.+)$",open(pd_+"/judge_verdicts.md").read(),re.M):
            g=m.group(3).replace(" (known-good)","").replace(" (known-bad)","").strip()
            if g in GRADE: tab[(P,m.group(2))][A]=(GRADE[g],ev.get(m.group(2),""),int(m.group(1)))

disagree=[(k,v) for k,v in tab.items() if len(v)>=2 and len({x[0] for x in v.values()})>1]
print(f"papers judged by >=2 agents: {sum(1 for v in tab.values() if len(v)>=2)}")
print(f"  of which grades DISAGREE: {len(disagree)}")
# how often does one agent get 3 and another <3
flip=[(k,v) for k,v in disagree if max(x[0] for x in v.values())==3 and min(x[0] for x in v.values())<3]
print(f"  3-vs-not-3 flips: {len(flip)}")
# for flips: compare evidence length / passage count
dl=[];dp=[]
for k,v in flip:
    hi=[x for x in v.values() if x[0]==3]; lo=[x for x in v.values() if x[0]<3]
    dl.append(sum(len(x[1]) for x in hi)/len(hi) - sum(len(x[1]) for x in lo)/len(lo))
    dp.append(sum(x[1].count(" ... ")+1 for x in hi)/len(hi) - sum(x[1].count(" ... ")+1 for x in lo)/len(lo))
print(f"  mean evidence-length diff (grade3 minus lower): {sum(dl)/len(dl):+.0f} chars")
print(f"  mean passage-count diff: {sum(dp)/len(dp):+.2f}")
print(f"  fraction of flips where grade3 evidence was LONGER: {100*sum(1 for x in dl if x>0)/len(dl):.0f}%")

# consistency: same paper same evidence, different grade? (judge noise)
same=0;samediff=0
for k,v in tab.items():
    if len(v)<2: continue
    evs={x[1] for x in v.values()}
    if len(evs)==1:
        same+=1
        if len({x[0] for x in v.values()})>1: samediff+=1
print(f"\nIDENTICAL evidence across agents: {same} papers; grades still differ on {samediff} ({100*samediff/max(1,same):.0f}%) -> judge noise floor")
