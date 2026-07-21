<code>
# Use a fixed external validation set for XGBoost early stopping during each GridSearchCV fit
gridsearch.fit(
    trainX,
    trainY,
    early_stopping_rounds=42,
    eval_metric="mae",
    eval_set=[(testX, testY)],
    verbose=False,
)

b = gridsearch.best_score_
c = gridsearch.predict(testX)
</code>