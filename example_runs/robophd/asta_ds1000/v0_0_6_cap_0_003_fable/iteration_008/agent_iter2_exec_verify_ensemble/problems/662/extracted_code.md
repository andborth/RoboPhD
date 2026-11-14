<code>
g = sns.FacetGrid(df, row="b")
g.map_dataframe(sns.pointplot, x="a", y="c", errorbar=None)

for ax in g.axes.flat:
    ax.set_xticks(np.arange(1, 31, 1))
    ax.set_xticklabels([str(v) if v % 2 == 0 else "" for v in range(1, 31)])
    ax.tick_params(axis="x", labelbottom=True)

plt.show()
</code>