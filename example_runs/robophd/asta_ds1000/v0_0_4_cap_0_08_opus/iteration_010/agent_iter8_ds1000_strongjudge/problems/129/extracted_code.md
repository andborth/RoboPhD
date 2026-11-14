<code>
df = df.astype(float)
altered = df.groupby('UserId').apply(lambda x: x.sample(n=max(1, round(len(x)*0.2)), random_state=0)).index.get_level_values(1)
df.loc[altered, 'Quantity'] = 0
</code>