<code>
from sklearn.svm import SVR

model = SVR(kernel='rbf')
model.fit(X, y)
predict = model.predict(X)
</code>