<code>
masked_data = np.ma.masked_where(DataArray < 0, DataArray)
prob = np.percentile(masked_data.compressed(), percentile)
</code>