<code>
g = sns.FacetGrid(df, row="b", sharex=True)
g.map_dataframe(sns.pointplot, x="a", y="c", order=np.arange(1, 31), errorbar=None)
for ax in g.axes.flat:
    ax.set_xticks(np.arange(0, 30, 1))
    ax.set_xticklabels([str(i) if i % 2 == 1 else "" for i in range(1, 31)])
plt.show()
</code>