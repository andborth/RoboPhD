<code>
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder()
new_X = encoder.fit_transform(X)
clf.fit(new_X, ['4', '5'])
</code>