<code>
from functools import reduce
from collections import defaultdict

def nested_defaultdict():
    return defaultdict(nested_defaultdict)

def to_dict(d):
    return d if not isinstance(d, defaultdict) else dict(map(lambda kv: (kv[0], to_dict(kv[1])), d.items()))

root = reduce(
    lambda acc, row: (lambda cur: (cur.__setitem__(row[-2], row[-1]), acc)[1])(
        reduce(lambda c, key: c[key], row[:-2], acc)
    ),
    df.itertuples(index=False, name=None),
    nested_defaultdict()
)

result = to_dict(root)
</code>