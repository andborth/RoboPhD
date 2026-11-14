<code>
slopes = []
for col in df1.columns[1:]:
    df2 = df1[~np.isnan(df1[col])]
    X = df2[['Time']].values
    Y = df2[col].values
    slopes.append(LinearRegression().fit(X, Y).coef_[0])
slopes = np.array(slopes)
</code>