<code>
from sklearn.calibration import CalibratedClassifierCV

calibrated_model = CalibratedClassifierCV(svmmodel, cv=5)
calibrated_model.fit(X, y)
proba = calibrated_model.predict_proba(x_test)
</code>