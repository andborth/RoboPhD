<code>
g = sns.FacetGrid(df, row="b")
g.map(sns.pointplot, "a", "c")

for ax in g.axes.flat:
    # Set ticks at every 1 unit
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    # Get all tick labels and hide every other one
    labels = ax.get_xticklabels()
    for i, label in enumerate(labels):
        if i % 2 != 0:
            label.set_visible(False)
    ax.set_xticklabels(ax.get_xticklabels())

plt.show()
</code>