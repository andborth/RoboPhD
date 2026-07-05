<code>
df = pd.DataFrame([df.to_numpy().ravel()], columns=df.columns.astype(str).repeat(len(df)).str.cat(pd.Index(range(1, len(df) + 1)).astype(str).repeat(len(df.columns)), sep="_"))
</code>