<code>
val_cols = df.filter(like='val').columns.tolist()
agg_dict = {"group_color": "first"}
agg_dict.update(dict(zip(val_cols, ['sum']*len(val_cols))))
result = df.groupby('group').agg(agg_dict)
</code>