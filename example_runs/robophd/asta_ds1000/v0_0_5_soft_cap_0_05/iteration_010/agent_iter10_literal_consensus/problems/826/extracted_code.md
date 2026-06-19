<code>
from sklearn.calibration import CalibratedClassifierCV

calibrated = CalibratedClassifierCV(svmmodel)
calibrated.fit(X, y)
proba = calibrated.predict_proba(x_test)
</code>