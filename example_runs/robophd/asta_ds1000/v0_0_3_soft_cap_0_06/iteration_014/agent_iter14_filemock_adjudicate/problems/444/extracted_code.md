<code>
C = A[np.any((A[:, None] > B[:-1]) & (A[:, None] < B[1:]), axis=1)]
</code>