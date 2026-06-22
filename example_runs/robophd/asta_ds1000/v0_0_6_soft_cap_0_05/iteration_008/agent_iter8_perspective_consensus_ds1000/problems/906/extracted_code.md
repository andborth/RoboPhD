<code>
slopes = df1.drop(columns='Time').apply(lambda s: LinearRegression().fit(df1.loc[s.notna(), ['Time']], s[s.notna()]).coef_[0]).to_numpy()
</code>