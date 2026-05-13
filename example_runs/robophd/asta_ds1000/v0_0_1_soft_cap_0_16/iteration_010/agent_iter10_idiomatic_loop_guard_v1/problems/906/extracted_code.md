<code>
slopes = []

for col in df1.columns:
    if col == 'Time':
        continue
    df2 = df1[['Time', col]].dropna()
    X = df2[['Time']].values  # 2D array, avoids np.matrix
    Y = df2[col].values        # 1D array
    reg = LinearRegression().fit(X, Y)
    slopes.append(reg.coef_[0])

slopes = np.array(slopes)
</code>