<code>
bits = (a[:, None] >> np.arange(m-1, -1, -1)) & 1
result = np.bitwise_xor.reduce(bits).reshape(1, m)
</code>