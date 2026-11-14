<code>
value_cols = [c for c in df.columns if c.startswith('val')]
result = df.groupby('group').agg({'group_color': 'first', **{c: 'sum' for c in value_cols}})
</code>