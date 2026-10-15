<code>
def fit_col(col):
    df2 = df1[~np.isnan(df1[col])]
    df3 = df2[['Time', col]]
    npMatrix = np.matrix(df3)
    X, Y = npMatrix[:, 0], npMatrix[:, 1]
    slope = LinearRegression().fit(X, Y)
    return slope.coef_[0]

cols = df1.columns.drop('Time')
slopes = np.array(list(map(fit_col, cols))).reshape(-1, 1)
</code>