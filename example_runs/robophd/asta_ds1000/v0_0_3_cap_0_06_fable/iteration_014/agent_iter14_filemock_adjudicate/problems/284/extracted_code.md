<code>
s = df.sum(axis=1)
n = df.shape[1]
df['frequent'] = (s > n / 2.0).astype(int)
df['freq_count'] = s.where(df['frequent'].eq(1), n - s).astype(float)
</code>