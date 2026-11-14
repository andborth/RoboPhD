<code>
C = A[((A[:, None] > B[:-1]) & (A[:, None] < B[1:])).any(axis=1)]
</code>