<code>
col = df.columns[2]
dummies = df[col].explode().pipe(pd.get_dummies).groupby(level=0).sum()
dummies = dummies.reindex(columns=pd.unique(df[col].explode()), fill_value=0)
df_out = pd.concat([df.drop(columns=[col]), dummies], axis=1)
</code>