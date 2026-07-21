<code>
from functools import reduce

result = {}
for row in df.to_numpy():
    *keys, val = row
    reduce(dict.setdefault, keys[:-1], result)[keys[-1]] = val
</code>