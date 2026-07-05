# Exercise the new pure-python helpers from agent.py without inspect_ai installed.
import ast, io, re, sys, types, difflib, tokenize, builtins
src = open("agent.py").read()
tree = ast.parse(src)
keep = []
for node in tree.body:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        names = getattr(node, "module", "") or ""
        if "inspect_ai" in names or "model_registry" in names:
            continue
    if isinstance(node, ast.AsyncFunctionDef) or (
        isinstance(node, ast.FunctionDef) and node.name == "make_solver"
    ):
        continue
    keep.append(node)
mod = ast.Module(body=keep, type_ignores=[])
ns = {}
exec(compile(mod, "agent_subset", "exec"), ns)

cmp_tokens, containment = ns["cmp_tokens"], ns["containment"]
first_diff, looks_hardcoded = ns["first_diff"], ns["looks_hardcoded"]
choose_by_score, validate_expected = ns["choose_by_score"], ns["validate_expected"]
parse_expected, problem_line_set = ns["parse_expected"], ns["problem_line_set"]

# 445 case
exp = cmp_tokens("array([7, 6, 3, 1, 3, 6, 3, 1])")
assert containment(exp, cmp_tokens("[8 6 3 1 3 6 3 1]")) < 0.97
assert containment(exp, cmp_tokens("[7 6 3 1 3 6 3 1]")) == 1.0
print("445:", ns["first_diff"](exp, cmp_tokens("[8 6 3 1 3 6 3 1]")))

# NaN / inf / weird tokens
assert cmp_tokens("NaN 1.5e-05 inf True African_Swallow int64") == ["nan", "1.5e-05", "inf", "true", "african_swallow"]

# choose_by_score
assert choose_by_score({"A": 1.0, "B": 0.7}) == "A"
assert choose_by_score({"A": 1.0, "B": 0.95}) is None
assert choose_by_score({"A": 0.96}) is None
assert choose_by_score({"A": 0.99}) == "A"

# hardcode guard
assert looks_hardcoded("result = np.array([7, 6, 3, 1, 3, 6, 3, 1])", exp) is True
assert looks_hardcoded("result = len(a) - rankdata(a).astype(int)", exp) is False
assert looks_hardcoded("df = df.pivot(index='id', columns='date')", exp) is False

# parse_expected
r = parse_expected("EXPECTED_BEGIN\n   a  b\n0  1  2\nEXPECTED_END")
assert r == "   a  b\n0  1  2", repr(r)
assert parse_expected("NONE") is None
assert parse_expected("None.") is None
r2 = parse_expected("EXPECTED_BEGIN\n1 2 3")  # missing END marker
assert r2 == "1 2 3", repr(r2)

# validate_expected: verbatim lines pass, hallucinated fail
prob = "blah\nI want:\n   a  b\n0  1  2\ndone"
pl = problem_line_set(prob)
assert validate_expected("   a  b\n0  1  2", pl) is True
assert validate_expected("x y z q\n9 9 9 9", pl) is False

# comma/# normalization in line matching
pl2 = problem_line_set("# 0,1,2,3,4")
assert validate_expected("0 1 2 3 4", pl2) is True

# skeleton-echo guard scenario: expected fully inside skeleton tokens
skel = "a = np.array([0, 3, 0, 5, 7, 0])"
assert containment(cmp_tokens("[3 5 7]"), cmp_tokens(skel)) >= 0.85
print("ALL HELPER TESTS PASSED")
