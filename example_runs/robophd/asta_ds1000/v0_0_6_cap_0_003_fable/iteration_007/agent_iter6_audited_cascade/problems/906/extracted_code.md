<code>
cols = df1.columns.drop('Time')
slopes = df1[cols].apply(lambda s: LinearRegression().fit(df1.loc[s.notna(), ['Time']], s.dropna()).coef_[0]).to_numpy()
</code>