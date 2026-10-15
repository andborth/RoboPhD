<code>
steps = list(clf.steps)
names = list(zip(*steps))[0]
idx = list(names).index('svdm')
steps.insert(idx, ('t1919810', PCA()))
clf.steps = steps
</code>