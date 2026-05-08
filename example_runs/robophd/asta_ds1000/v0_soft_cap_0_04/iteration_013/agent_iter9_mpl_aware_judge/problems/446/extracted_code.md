<code>
result = rankdata(-np.arange(len(a)), method='ordinal')[np.argsort(np.argsort(a, kind='stable'), kind='stable')[::-1][::-1]]
result = np.empty(len(a), dtype=int)
result[np.argsort(a, kind='stable')] = np.arange(len(a) - 1, -1, -1)
</code>