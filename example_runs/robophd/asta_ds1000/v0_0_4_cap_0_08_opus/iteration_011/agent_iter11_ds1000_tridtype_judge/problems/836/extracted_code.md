<code>
steps = list(clf.steps)
idx = [name for name, _ in steps].index('svdm')
steps.insert(idx, ('t1919810', PCA()))
clf = Pipeline(steps)
</code>