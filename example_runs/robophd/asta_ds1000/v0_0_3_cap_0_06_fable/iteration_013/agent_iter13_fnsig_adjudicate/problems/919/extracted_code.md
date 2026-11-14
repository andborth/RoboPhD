<code>
X = dataframe.iloc[:, :-1].astype(float)
y = dataframe.iloc[:, -1].astype(int)

try:
    logReg
except NameError:
    logReg = LogisticRegression()

logReg.fit(X, y)
predict = logReg.predict(X)
</code>