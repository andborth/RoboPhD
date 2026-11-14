<code>
new_X = pd.get_dummies(pd.DataFrame(X)).values
clf.fit(new_X, ['4', '5'])
</code>