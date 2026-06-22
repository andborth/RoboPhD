<code>
df = df.astype(float)
df.loc[df.groupby('UserId').sample(frac=0.2, random_state=0).index, 'Quantity'] = 0.0
</code>