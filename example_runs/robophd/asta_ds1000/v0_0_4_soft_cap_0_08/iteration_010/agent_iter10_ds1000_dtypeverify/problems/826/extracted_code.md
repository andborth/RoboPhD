<code>
from sklearn.calibration import CalibratedClassifierCV

model = CalibratedClassifierCV(svmmodel)
model.fit(X, y)
proba = model.predict_proba(x_test)
</code>