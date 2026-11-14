<code>
counts = a["A1"].value_counts().sort_index()
weights = (counts / counts.sum()).reindex(np.arange(a["A1"].max() + 1), fill_value=0).to_numpy()
</code>