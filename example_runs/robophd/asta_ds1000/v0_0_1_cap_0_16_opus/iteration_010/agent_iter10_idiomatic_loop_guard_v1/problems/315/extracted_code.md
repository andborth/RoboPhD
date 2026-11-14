<code>
a = a[:, ~np.any(np.isnan(a), axis=0)]
</code>