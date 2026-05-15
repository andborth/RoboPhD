<code>
g = sns.FacetGrid(df, row="b")
g.map(sns.pointplot, "a", "c")

for ax in g.axes.flat:
    # Get current x tick positions (0-indexed positions for each unique value of 'a')
    # pointplot uses categorical positions 0, 1, 2, ...
    # The x-axis in pointplot is categorical, positions 0..9 for each group
    xticks = ax.get_xticks()
    xticklabels = ax.get_xticklabels()

    # Set ticks at every position (interval of 1)
    # but only show labels at every 2nd position
    labels = [label.get_text() for label in xticklabels]

    # Show label only for every other tick (interval of 2)
    new_labels = [label if i % 2 == 0 else '' for i, label in enumerate(labels)]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(new_labels)
</code>