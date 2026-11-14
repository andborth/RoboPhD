<code>
result = np.empty(len(a), dtype=int)
result[np.argsort(a, kind='mergesort')] = np.arange(len(a) - 1, -1, -1)
</code>