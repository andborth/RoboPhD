import ast
src = open('agent.py').read()
head = src.split('@solver')[0]
keep = []
for ln in head.splitlines():
    if ln.startswith(('from inspect_ai', 'from model_registry', 'import ')):
        continue
    if ln.startswith(('MODEL', 'GEN_CONFIG', 'STRONG_MODEL', 'STRONG_CONFIG')):
        continue
    keep.append(ln)
keep = 'import ast, re, textwrap\n' + '\n'.join(keep)
ns = {}
exec(compile(ast.parse(keep), 'a', 'exec'), ns)

code269 = ("stacked = df.stack()\n"
           "stacked.index = [f'{c}_{r+1}' for r,c in stacked.index]\n"
           "df = pd.DataFrame(stacked).T")
print("269 idiom?", ns['_has_idiom_constraint']("most idiomatic way in Pandas?"),
      "loop?", ns['_has_loop'](code269))

skel420 = "import numpy as np\nx = 0.25\nx_min = 0\nx_max = 1"
print("420 func:", ns['_invent_function'](
    "define function named `smoothclamp` as solution", skel420))

skel723 = ("from scipy import sparse\nexample_sA = sparse.csr_matrix(1)\n"
           "def f(sA = example_sA, sB = example_sB):")
print("723 func (expect None):", ns['_invent_function'](
    "return the solution in this function", skel723))

out706 = ('AttributeError: MessageFactory no GetPrototype\n'
          'Traceback (most recent call last):\n'
          '  File "/ds1000/test_706.py", line 93\n'
          '  File "<string>", line 34, in <module>\n'
          'ValueError: save_format deprecated')
print("706 env?", ns['_env_noise'](out706),
      "actionable?", ns['_actionable_error'](out706))

pure = 'AttributeError: MessageFactory no GetPrototype\nNo module named foo'
print("pure env actionable?", ns['_actionable_error'](pure),
      "env?", ns['_env_noise'](pure))

print("vectorized loop?",
      ns['_has_loop']("df.index = df.index.map('{0[1]}_{0[0]}'.format)"))
print("plain for loop?", ns['_has_loop']("for i in range(3):\n    pass"))
