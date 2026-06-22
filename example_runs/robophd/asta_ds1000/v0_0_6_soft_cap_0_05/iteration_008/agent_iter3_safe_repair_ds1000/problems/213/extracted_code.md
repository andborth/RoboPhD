<code>
result = df[df['key2'].eq('two')].groupby('key1').size().reset_index(name='count')
</code>