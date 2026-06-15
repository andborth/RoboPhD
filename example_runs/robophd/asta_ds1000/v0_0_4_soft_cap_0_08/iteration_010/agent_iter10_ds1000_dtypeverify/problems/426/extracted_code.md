<code>
result = np.array([[(int(x) >> i) & 1 for i in range(m - 1, -1, -1)] for x in a], dtype=int)
</code>