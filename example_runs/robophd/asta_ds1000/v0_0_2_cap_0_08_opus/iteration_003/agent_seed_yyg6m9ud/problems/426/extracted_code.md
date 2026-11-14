<code>
# Use an unsigned view with enough bits to cover m, then unpack and keep the last m bits.
# This works for positive/negative integers via two's-complement bit patterns.
dtype = np.min_scalar_type(1 << max(m - 1, 0))
if np.issubdtype(dtype, np.signedinteger):
    dtype = np.dtype(f'uint{dtype.itemsize * 8}')
else:
    dtype = np.dtype(f'uint{dtype.itemsize * 8}')

result = np.unpackbits(a.astype(dtype, copy=False).view(np.uint8)).reshape(len(a), -1)[:, -m:]
</code>