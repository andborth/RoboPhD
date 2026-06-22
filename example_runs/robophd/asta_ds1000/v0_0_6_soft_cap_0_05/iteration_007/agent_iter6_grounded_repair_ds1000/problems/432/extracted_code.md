<code>
prob = np.percentile(np.ma.masked_where(DataArray < 0, DataArray).compressed(), percentile)
</code>