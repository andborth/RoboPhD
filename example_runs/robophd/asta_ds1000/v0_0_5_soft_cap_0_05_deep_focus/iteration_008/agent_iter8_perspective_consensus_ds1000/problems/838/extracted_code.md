<code>
gridsearch.fit(trainX, trainY, early_stopping_rounds=42, eval_metric="mae", eval_set=[(testX, testY)])
b = gridsearch.score(testX, testY)
c = gridsearch.predict(testX)
</code>