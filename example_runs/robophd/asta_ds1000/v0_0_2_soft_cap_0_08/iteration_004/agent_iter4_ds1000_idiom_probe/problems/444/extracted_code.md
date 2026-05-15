<code>
mask = (A > B[0]) & (A < B[-1]) & ~np.isin(A, B)
C = A[mask]
</code>