<code>
binary_matrix = ((a[:, np.newaxis] >> np.arange(m-1, -1, -1)) & 1).astype(np.uint8)
result = np.bitwise_xor.reduce(binary_matrix)
</code>