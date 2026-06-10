<code>
df_out = df.join(df['Col3'].str.join('|').str.get_dummies())
</code>