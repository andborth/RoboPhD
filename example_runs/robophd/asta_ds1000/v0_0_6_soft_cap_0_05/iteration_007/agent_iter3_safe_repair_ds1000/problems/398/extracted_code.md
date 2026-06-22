<code>
B = pd.Series(np.empty_like(A, dtype=float), index=A.index)
B.iloc[0] = a * A.iloc[0]
for t in range(1, len(A)):
    B.iloc[t] = a * A.iloc[t] + b * B.iloc[t - 1]
</code>