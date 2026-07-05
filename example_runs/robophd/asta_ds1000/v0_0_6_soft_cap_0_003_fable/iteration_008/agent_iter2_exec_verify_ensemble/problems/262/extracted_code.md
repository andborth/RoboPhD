<code>
result = df[filt.reindex(df.index.get_level_values('a')).to_numpy()]
</code>