<code>
bits = ((a[:, None] & (1 << np.arange(m-1, -1, -1))) > 0).astype(np.uint8)
result = np.bitwise_xor.reduce(bits)
</code>