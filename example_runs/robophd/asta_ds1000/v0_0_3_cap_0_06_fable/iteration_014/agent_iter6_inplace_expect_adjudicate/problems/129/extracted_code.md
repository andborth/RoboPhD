<code>
altered_indexes = df.groupby('UserId', group_keys=False).apply(
    lambda x: x.sample(n=max(int(len(x) * 0.2), 1), random_state=0)
).index
df.loc[altered_indexes, 'Quantity'] = 0
</code>