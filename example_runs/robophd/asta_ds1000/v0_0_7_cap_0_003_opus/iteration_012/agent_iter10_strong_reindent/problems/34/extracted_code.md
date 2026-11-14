<code>
result = df.groupby('group', as_index=True).agg(
    {'group_color': 'first', **{c: 'sum' for c in df.filter(like='val').columns}}
)
</code>