<code>
from sklearn.svm import SVR
svr = SVR(kernel='rbf')
svr.fit(X, y)
predict = svr.predict(X)
</code>