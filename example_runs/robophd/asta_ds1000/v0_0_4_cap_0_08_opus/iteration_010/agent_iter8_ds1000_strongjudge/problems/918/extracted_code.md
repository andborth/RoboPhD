<code>
from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder()
new_X = enc.fit_transform(X)
clf.fit(new_X, ['4', '5'])
</code>