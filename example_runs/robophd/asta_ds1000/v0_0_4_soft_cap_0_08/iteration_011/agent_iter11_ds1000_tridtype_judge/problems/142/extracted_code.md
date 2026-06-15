<code>
result = df.copy()
result.columns = result.columns.reorder_levels(list(range(result.columns.nlevels))[::-1])
result = result.melt()
</code>