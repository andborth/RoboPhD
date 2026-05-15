<code>
result = ((a[:, None] >> np.arange(m - 1, -1, -1)) & 1).astype(np.uint8)
</code>