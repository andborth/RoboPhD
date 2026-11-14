<code>
n = df.shape[1]
s = df.sum(axis=1)
df['frequent'] = (s > n / 2.0).astype(df.dtypes.iloc[0])
df['freq_count'] = s.where(s > n / 2.0, n - s).astype(float)
</code>