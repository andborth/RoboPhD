<code>
from functools import reduce

result = {}
for row in df.itertuples(index=False, name=None):
    *keys, value = row
    reduce(dict.setdefault, keys[:-1], result)[keys[-1]] = value
</code>