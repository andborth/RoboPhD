<code>
slopes = np.array([
    LinearRegression().fit(df1.loc[df1[c].notna(), ['Time']], df1.loc[df1[c].notna(), c]).coef_[0]
    for c in df1.columns if c != 'Time'
])
</code>