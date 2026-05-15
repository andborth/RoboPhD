<code>
steps = clf.steps
idx = [name for name, _ in steps].index('svdm')
steps.insert(idx, ('t1919810', PCA()))
clf.steps = steps
</code>