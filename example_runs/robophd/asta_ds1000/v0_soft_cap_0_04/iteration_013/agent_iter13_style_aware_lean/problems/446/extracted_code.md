<code>
n = len(a)
# argsort by (-value, -index): sort descending by value, then descending by index
idx = np.lexsort((np.arange(n)[::-1], [-x for x in a]))
result = np.empty(n, dtype=int)
result[idx] = np.arange(n)
</code>