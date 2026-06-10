"""Local check of the iter14 file-mock path against the real 919 problem.

Verifies:
1. extract_context pulls the runnable context (which reads animalData.csv).
2. FILE_READ_RE triggers on it (argument is a variable, not a literal).
3. With a verbatim CSV mock prepended, the harness-style exec surfaces the
   exact NameError the grader raised for iter12's submission, while the
   reference-style candidate runs clean.

sklearn is not installed locally, so a minimal stub stands in for
LogisticRegression — the NameError-vs-clean distinction doesn't depend on it.
"""
import os
import re
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- import the pure-python pieces from agent.py without inspect_ai ---
import ast as _ast
src = open(os.path.join(os.path.dirname(__file__), "agent.py")).read()
tree = _ast.parse(src)
wanted = {"extract_context", "find_target_vars", "clean_code"}
ns = {"re": re}
for node in tree.body:
    if isinstance(node, _ast.FunctionDef) and node.name in wanted:
        exec(compile(_ast.Module([node], []), "agent.py", "exec"), ns)
    if isinstance(node, _ast.Assign):
        t = node.targets[0]
        if isinstance(t, _ast.Name) and t.id == "FILE_READ_RE":
            exec(compile(_ast.Module([node], []), "agent.py", "exec"), ns)
extract_context = ns["extract_context"]
find_target_vars = ns["find_target_vars"]
FILE_READ_RE = ns["FILE_READ_RE"]

problem = open(
    "../../iteration_013/agent_iter12_thirdvote_adjudicate/problems/919/problem.md"
).read()

ctx = extract_context(problem)
print("targets:", find_target_vars(problem))
assert "read_csv" in ctx, "context should contain the read_csv line"
assert FILE_READ_RE.search(ctx), "FILE_READ_RE must trigger on 919's context"
print("FILE_READ_RE triggers: OK")

# --- stub sklearn so the 919 context import works locally ---
sk = types.ModuleType("sklearn")
lm = types.ModuleType("sklearn.linear_model")


class LogisticRegression:
    def fit(self, X, y):
        self._n = len(X)
        return self

    def predict(self, X):
        return [0] * len(X)


lm.LogisticRegression = LogisticRegression
sk.linear_model = lm
sys.modules["sklearn"] = sk
sys.modules["sklearn.linear_model"] = lm

# --- verbatim file mock, as the FILE_MOCK_PROMPT should produce ---
mock = '''# VERBATIM
with open("animalData.csv", "w") as _fh:
    _fh.write("""Name,teethLength,weight,length,hieght,speed,Calorie Intake,Bite Force,Prey Speed,PreySize,EyeSight,Smell,Class
T-Rex,12,15432,40,20,33,40000,12800,20,19841,0,0,Primary Hunter
Crocodile,4,2400,23,1.6,8,2500,3700,30,881,0,0,Primary Hunter
Lion,2.7,416,9.8,3.9,50,7236,650,35,1300,0,0,Primary Hunter
Bear,3.6,600,7,3.35,40,20000,975,0,0,0,0,Primary Scavenger
Tiger,3,260,12,3,40,7236,1050,37,160,0,0,Primary Hunter
Hyena,0.27,160,5,2,37,5000,1100,20,40,0,0,Primary Scavenger
Jaguar,2,220,5.5,2.5,40,5000,1350,15,300,0,0,Primary Hunter
Cheetah,1.5,154,4.9,2.9,70,2200,475,56,185,0,0,Primary Hunter
KomodoDragon,0.4,150,8.5,1,13,1994,240,24,110,0,0,Primary Scavenger
""")
'''

iter12_candidate = """X = dataframe.iloc[:, :-1].astype(float)
y = dataframe.iloc[:, -1].astype(int)

logReg.fit(X, y)
predict = logReg.predict(X)
"""

reference_style = """X = dataframe.iloc[:, :-1].astype(float)
y = dataframe.iloc[:, -1].astype(int)

logReg = LogisticRegression()
logReg.fit(X, y)
predict = logReg.predict(X)
"""

os.chdir(os.path.dirname(os.path.abspath(__file__)))
# Local pandas is stricter than the eval sandbox's pinned version (categorical
# replace raises here, only warns there); neutralize that version quirk so the
# NameError-vs-clean mechanism itself can be validated.
ctx = ctx.replace(", dtype='category'", "")
for label, cand in [("iter12 (bad)", iter12_candidate),
                    ("reference-style (good)", reference_style)]:
    g = {}
    try:
        exec(compile(mock + "\n" + ctx + "\n" + cand, "<cand>", "exec"), g)
        err = None
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    print(f"{label}: err={err}, predict_set={'predict' in g}")

assert os.path.exists("animalData.csv")
os.remove("animalData.csv")
print("DONE")
