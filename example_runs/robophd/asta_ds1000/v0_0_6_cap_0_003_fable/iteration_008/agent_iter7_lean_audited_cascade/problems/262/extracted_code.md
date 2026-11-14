<code>
result = df.loc[df.index.get_level_values('a').map(filt.fillna(False).astype(bool))]
</code>