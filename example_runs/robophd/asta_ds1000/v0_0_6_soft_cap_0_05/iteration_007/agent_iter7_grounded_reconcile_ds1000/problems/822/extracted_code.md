<code>
df_out = df.drop(columns='Col3').join(
    df['Col3'].explode().str.get_dummies().groupby(level=0).sum()
).fillna(0).astype({c: int for c in df['Col3'].explode().dropna().unique()})
</code>