<code>
mask = ((A > B[:-1, None]) & (A < B[1:, None])).any(axis=0)
C = A[mask]
</code>