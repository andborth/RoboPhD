<code>
altered_indexes = df.groupby('UserId', group_keys=False).apply(lambda x: x.sample(frac=0.2, random_state=0)).index
df.loc[altered_indexes, 'Quantity'] = 0
df = df.astype(float)
</code>