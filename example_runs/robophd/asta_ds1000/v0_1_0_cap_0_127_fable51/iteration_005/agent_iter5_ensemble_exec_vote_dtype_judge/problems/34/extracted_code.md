<code>
result = df.groupby('group').agg(lambda x: x.head(1) if x.dtype == 'object' else x.sum())
</code>