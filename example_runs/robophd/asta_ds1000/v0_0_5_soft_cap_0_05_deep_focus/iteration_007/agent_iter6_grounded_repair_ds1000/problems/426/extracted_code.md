<code>
result = ((np.asarray(a).reshape(-1, 1) >> np.arange(m - 1, -1, -1)) & 1).astype(np.uint8)
</code>