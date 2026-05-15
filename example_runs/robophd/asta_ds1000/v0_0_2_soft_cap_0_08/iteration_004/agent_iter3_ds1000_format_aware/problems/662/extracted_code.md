<code>
g = sns.FacetGrid(df, row="b")
g.map(sns.pointplot, "a", "c")

for ax in g.axes.flat:
    ax.set_xticks(range(10))
    ax.set_xticklabels([str(i) if i % 2 == 1 else "" for i in range(1, 11)])
</code>