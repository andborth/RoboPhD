<code>
n = min(a.shape[:2])
diagonal = np.diag_indices(n)
result = np.array([a[diagonal], np.fliplr(a)[:n, :n][diagonal]])
</code>