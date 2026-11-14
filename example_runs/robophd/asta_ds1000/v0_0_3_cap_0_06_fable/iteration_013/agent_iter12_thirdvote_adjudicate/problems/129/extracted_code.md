<code>
idx = df.groupby('UserId', group_keys=False).apply(lambda x: x.sample(n=int(len(x) * 0.2), random_state=0)).index
df = df.astype(float)
df.loc[idx, 'Quantity'] = 0.0
</code>