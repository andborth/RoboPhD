<code>
from sklearn.calibration import CalibratedClassifierCV

proba = CalibratedClassifierCV(svmmodel).fit(X, y).predict_proba(x_test)
</code>