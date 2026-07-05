<code>
gridsearch.fit(
    np.array(trainX),
    np.array(trainY),
    early_stopping_rounds=42,
    eval_metric="mae",
    eval_set=[(np.array(testX), np.array(testY))]
)

b = gridsearch.best_score_
c = gridsearch.predict(np.array(testX))
</code>