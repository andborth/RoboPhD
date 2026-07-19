"""Wide-pool variant of smoke_test: enough uniques to trigger the lexical
prescreen (>360) and the full-depth broad tail sweep (>=400)."""
src = open("smoke_test.py").read()
src = src.replace("N_SEARCH_DOCS = 60", "N_SEARCH_DOCS = 100")
src = src.replace("(hash(keyword) % 7) * 50", "(hash(keyword) % 10) * 100")
exec(compile(src, "smoke_wide_exec.py", "exec"))
