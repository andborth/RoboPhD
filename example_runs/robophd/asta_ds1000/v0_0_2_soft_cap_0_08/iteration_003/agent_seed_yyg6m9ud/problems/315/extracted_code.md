<code>
mask = ~np.any(np.isnan(a), axis=0)
a = a[:, mask]
</code>