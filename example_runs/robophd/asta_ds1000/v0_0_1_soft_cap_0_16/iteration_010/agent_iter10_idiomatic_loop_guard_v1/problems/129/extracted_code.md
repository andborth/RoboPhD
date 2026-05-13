<code>
df = df.astype(float)

altered_idx = (
    df.groupby('UserId')
    .apply(lambda x: x.sample(n=max(1, int(len(x) * 0.2)), random_state=0))
    .index.get_level_values(1)
)

df.loc[altered_idx, 'Quantity'] = 0
</code>