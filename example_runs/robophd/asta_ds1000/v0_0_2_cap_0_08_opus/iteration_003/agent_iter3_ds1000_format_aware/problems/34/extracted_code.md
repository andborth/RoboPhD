<code>
result = df.groupby('group').agg(
    {**{'group_color': 'first'}, **{col: 'sum' for col in df.columns if col.startswith('val')}}
)
</code>