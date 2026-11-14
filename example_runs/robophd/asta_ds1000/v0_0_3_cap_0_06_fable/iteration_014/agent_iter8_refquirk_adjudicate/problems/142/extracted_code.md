<code>
result = df.melt(value_vars=list(df.columns))
n = df.columns.nlevels
result = result.iloc[:, list(range(n - 1, -1, -1)) + [n]]
result.columns = [f"variable_{i}" for i in range(n)] + ["value"]
</code>