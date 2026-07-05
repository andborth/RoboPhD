<code>
# Create X and y correctly
# X should be all columns except the last one ('Class')
# y should be the last column ('Class')
X = dataframe.iloc[:, :-1].astype(float)
y = dataframe.iloc[:, -1].astype(int)

logReg = LogisticRegression()
logReg.fit(X, y)
predict = logReg.predict(X)
</code>