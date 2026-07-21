<code>
cv = TimeSeriesSplit(n_splits=3)

class XGBEarlyStopGridSearchCV(GridSearchCV):
    def fit(self, X, y=None, **fit_params):
        splits = list(self.cv.split(X, y))
        if not splits:
            return super().fit(X, y, **fit_params)
        train_idx, val_idx = splits[-1]
        fit_params = dict(fit_params)
        fit_params["eval_set"] = [(np.asarray(X)[val_idx], np.asarray(y)[val_idx])]
        return super().fit(X, y, **fit_params)

gridsearch.cv = cv
gridsearch.fit(
    trainX,
    trainY,
    early_stopping_rounds=42,
    eval_metric="mae",
    verbose=False,
)

b = gridsearch.best_score_
c = gridsearch.predict(testX)
</code>