<code>
g = sns.FacetGrid(df, row="b")
g.map_dataframe(sns.pointplot, x="a", y="c", errorbar=None, join=True)

for ax in g.axes.flat:
    vals = np.sort(df.loc[df["b"] == ax.get_title().split(" = ")[1], "a"].unique())
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels([str(v) if v % 2 == 1 else "" for v in vals])

plt.show()
</code>