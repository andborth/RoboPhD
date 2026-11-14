<code>
model = SelectFromModel(clf, prefit=True)
selected_cols = X.columns[model.get_support()]
column_names = selected_cols
</code>