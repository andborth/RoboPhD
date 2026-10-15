<code>
from sklearn.feature_extraction import DictVectorizer
new_X = DictVectorizer(sparse=False).fit_transform(pd.DataFrame(X).to_dict('records'))
clf.fit(new_X, ['4', '5'])
</code>