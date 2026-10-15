<code>
def pointplot_with_sparse_xticklabels(data, color=None, **kws):
    ax = plt.gca()
    ticks = np.arange(data["a"].min(), data["a"].max() + 1, 1)
    sns.pointplot(data=data, x="a", y="c", order=ticks, color=color, ax=ax)
    ax.set_xticks(np.arange(ticks.size))
    ax.set_xticklabels(np.where(np.arange(ticks.size) % 2 == 0, ticks, ""))

g = sns.FacetGrid(df, row="b", sharex=False)
g.map_dataframe(pointplot_with_sparse_xticklabels)
</code>