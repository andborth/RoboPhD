<code>
result = df.groupby('group').agg(lambda x: x.sum() if x.name.startswith('val') else x.head(1))
</code>