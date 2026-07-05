<code>
df = pd.DataFrame(df.to_numpy().reshape(1, -1), columns=df.columns.repeat(len(df)).astype(str) + "_" + pd.Index(range(1, len(df) + 1)).repeat(len(df.columns)).astype(str))
</code>