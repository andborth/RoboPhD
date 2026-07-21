<code>
from collections import defaultdict
from functools import reduce

def tree():
    return defaultdict(tree)

result = tree()
for row in df.itertuples(index=False, name=None):
    *keys, value = row
    reduce(lambda d, k: d[k], keys[:-1], result)[keys[-1]] = value

def freeze(d):
    return {k: freeze(v) if isinstance(v, defaultdict) else v for k, v in d.items()}

result = freeze(result)
</code>