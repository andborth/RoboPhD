<code>
X = dataframe.drop("Class", axis=1).astype(float)
y = dataframe["Class"].astype(int)
logReg.fit(X, y)
predict = logReg.predict(X)
</code>