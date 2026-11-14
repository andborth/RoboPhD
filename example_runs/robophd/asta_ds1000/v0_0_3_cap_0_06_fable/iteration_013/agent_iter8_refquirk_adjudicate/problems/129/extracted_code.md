<code>
idx = df.groupby('UserId').apply(lambda x: x.sample(frac=0.2, random_state=0)).index.get_level_values(-1)
df.loc[idx, 'Quantity'] = 0
df = df.astype(float)
</code>