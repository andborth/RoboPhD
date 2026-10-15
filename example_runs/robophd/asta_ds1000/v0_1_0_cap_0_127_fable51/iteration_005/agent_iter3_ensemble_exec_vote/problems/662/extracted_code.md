<code>
g = sns.FacetGrid(df, row="b")
g.map(sns.pointplot, "a", "c")
for ax in g.axes.flatten():
    for ind, label in enumerate(ax.get_xticklabels()):
        if ind % 2 != 0:
            label.set_visible(False)
</code>