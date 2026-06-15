<code>
result = ((a.astype(object)[:, None] >> np.arange(m - 1, -1, -1)) & 1).astype(int)
</code>