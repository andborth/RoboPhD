<code>
C = A[(A > B[:-1, None]).any(0) & (A < B[1:, None]).any(0)]
</code>