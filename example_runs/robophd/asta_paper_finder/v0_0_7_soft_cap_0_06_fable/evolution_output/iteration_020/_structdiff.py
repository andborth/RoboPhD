import ast
def funcs(path):
    tree = ast.parse(open(path).read())
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[n.name] = ast.dump(n)
    return out
a = funcs("../../agents/iter18_triage_first/agent.py")
b = funcs("agent.py")
print("added:", sorted(set(b) - set(a)))
print("removed:", sorted(set(a) - set(b)))
changed = sorted(k for k in set(a) & set(b) if a[k] != b[k])
print("changed:", changed)
