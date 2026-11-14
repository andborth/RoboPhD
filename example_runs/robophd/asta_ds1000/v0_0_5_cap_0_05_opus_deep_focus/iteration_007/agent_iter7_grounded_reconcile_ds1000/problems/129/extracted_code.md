<code>
df = df.astype(float)
idx = df.groupby('UserId').apply(lambda x: x.sample(n=int(len(x)*0.2), random_state=0)).index.get_level_values(1)
df.loc[idx, 'Quantity'] = 0.0
</code>