<code>
i = [name for name, _ in clf.steps].index('svdm')
clf = Pipeline(clf.steps[:i] + [('t1919810', PCA())] + clf.steps[i:])
</code>