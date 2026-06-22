<code>
result = df[df['key2'].eq('two')].groupby('key1', as_index=False).size().rename(columns={'size': 'count'})
</code>