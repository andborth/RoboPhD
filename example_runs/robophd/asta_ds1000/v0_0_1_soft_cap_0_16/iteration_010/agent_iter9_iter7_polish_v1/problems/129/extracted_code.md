<code>
df = df.astype(float)

altered_indexes = []

for user_id, group in df.groupby('UserId'):
    n = max(1, round(len(group) * 0.2))
    sampled = group.sample(n=n, random_state=0)
    altered_indexes.extend(sampled.index.tolist())

df.loc[altered_indexes, 'Quantity'] = 0
</code>