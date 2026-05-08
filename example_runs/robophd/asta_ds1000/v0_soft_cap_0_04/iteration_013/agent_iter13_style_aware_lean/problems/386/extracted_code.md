<code>
from numpy.lib.stride_tricks import as_strided

patch_size = 2
rows, cols = a.shape
out_rows = rows - patch_size + 1
out_cols = cols - patch_size + 1

result = as_strided(
    a,
    shape=(out_rows, out_cols, patch_size, patch_size),
    strides=(a.strides[0], a.strides[1], a.strides[0], a.strides[1])
).reshape(-1, patch_size, patch_size)
</code>