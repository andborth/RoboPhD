<code>
gridsearch.fit(trainX, trainY, early_stopping_rounds=42, eval_metric="mae", eval_set=[(testX, testY)], verbose=False)
b = gridsearch.best_score_
c = gridsearch.predict(testX)
</code>