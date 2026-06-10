<code>
frequent = df.mode(axis=1)[0]
freq_count = df.eq(frequent, axis=0).sum(axis=1).astype(float)
df['frequent'] = frequent.astype(int)
df['freq_count'] = freq_count
</code>