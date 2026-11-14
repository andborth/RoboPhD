<code>
from sklearn.feature_selection import SelectFromModel

model = SelectFromModel(clf, prefit=True)
column_names = X.columns[model.get_support()].tolist()
</code>