<code>
df = pd.DataFrame([df.to_numpy().ravel()], columns=df.columns.repeat(len(df)).astype(str) + '_' + pd.Index(range(1, len(df) + 1)).repeat(len(df.columns)).astype(str))
</code>