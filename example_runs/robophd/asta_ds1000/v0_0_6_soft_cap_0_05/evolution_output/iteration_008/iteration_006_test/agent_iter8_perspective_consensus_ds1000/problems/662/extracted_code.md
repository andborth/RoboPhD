<code>
g = sns.FacetGrid(df, row="b")
g.map(sns.pointplot, "a", "c", order=np.arange(1, 31))
for ax in g.axes.flat:
    ax.set_xticks(np.arange(1, 31, 1))
    ax.set_xticklabels([str(i) if i % 2 == 0 else "" for i in range(1, 31)])
</code>