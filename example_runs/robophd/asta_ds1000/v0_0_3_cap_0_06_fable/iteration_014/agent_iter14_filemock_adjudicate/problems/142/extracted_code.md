<code>
tmp = df.copy()
tmp.columns = tmp.columns.reorder_levels(list(range(tmp.columns.nlevels - 1, -1, -1)))
tmp.columns.names = [None] * tmp.columns.nlevels
result = pd.melt(tmp)
</code>