<code>
result = np.array([a[np.diag_indices(min(a.shape))], np.fliplr(a)[np.diag_indices(min(a.shape))]])
</code>