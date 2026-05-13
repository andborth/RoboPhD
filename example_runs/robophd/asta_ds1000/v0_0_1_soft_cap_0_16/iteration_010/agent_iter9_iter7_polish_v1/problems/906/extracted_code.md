<code>
slopes = []
for col in df1.columns[1:]:
    d = df1[~np.isnan(df1[col])]
    X = d[['Time']].values
    Y = d[col].values
    slopes.append(LinearRegression().fit(X, Y).coef_[0])
slopes = np.array(slopes)
</code>