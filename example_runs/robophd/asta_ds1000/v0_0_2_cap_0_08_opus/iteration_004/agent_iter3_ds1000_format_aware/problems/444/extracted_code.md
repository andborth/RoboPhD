<code>
right_idx = np.searchsorted(B, A, side='right')
safe_idx = np.maximum(right_idx - 1, 0)
mask = (right_idx > 0) & (right_idx < len(B)) & (A != B[safe_idx])
C = A[mask]
</code>