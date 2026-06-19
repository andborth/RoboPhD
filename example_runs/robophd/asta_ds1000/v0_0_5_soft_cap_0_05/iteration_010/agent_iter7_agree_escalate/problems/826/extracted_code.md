<code>
from sklearn.calibration import CalibratedClassifierCV

calibrated_svm = CalibratedClassifierCV(svmmodel, cv=5)
calibrated_svm.fit(X, y)
proba = calibrated_svm.predict_proba(x_test)
</code>