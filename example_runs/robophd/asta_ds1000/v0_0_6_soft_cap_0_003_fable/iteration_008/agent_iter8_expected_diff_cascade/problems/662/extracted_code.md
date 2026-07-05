<code>
g = sns.FacetGrid(df, row="b", height=4, aspect=2)
g.map(sns.pointplot, "a", "c")

for ax in g.axes.flat:
    ax.set_xticks(range(1, 31, 1))
    ax.set_xticklabels([str(i) if i % 2 == 1 else "" for i in range(1, 31, 1)])

plt.tight_layout()
plt.show()
</code>