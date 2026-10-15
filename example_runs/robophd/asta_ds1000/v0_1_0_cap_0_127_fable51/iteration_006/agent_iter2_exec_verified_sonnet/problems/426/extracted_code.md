<code>
result = (a[:, None] >> np.arange(m-1, -1, -1)) & 1
</code>