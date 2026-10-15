<code>
g = sns.FacetGrid(df, row="b")
g.map(sns.pointplot, "a", "c")

for ax in g.axes.flat:
    n = len(ax.get_xticklabels())
    ax.set_xticks(range(n))
    ax.set_xticklabels([lbl.get_text() if i % 2 == 0 else "" for i, lbl in enumerate(ax.get_xticklabels())])
</code>